import csv
import gc
import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import get_calib_dataloader

init(autoreset=True)
logging.basicConfig(
    format=(f"{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] "
            f"{Fore.RESET}%(message)s"),
    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def _compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _angular(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cos = nn.functional.cosine_similarity(a, b, dim=-1)
    return torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7)) / torch.pi


def _lm_loss(logits, input_ids, attention_mask, answer_starts=None):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if answer_starts is not None:
        for row, s in enumerate(answer_starts):
            labels[row, :s] = -100
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), ignore_index=-100)


def _find_answer_starts(input_ids, tokenizer, answer_marker: str):
    marker_ids = tokenizer(answer_marker, add_special_tokens=False)["input_ids"]
    starts = []
    for row in input_ids:
        ids = row.tolist()
        pos = 0
        for j in range(len(ids) - len(marker_ids), -1, -1):
            if ids[j:j + len(marker_ids)] == marker_ids:
                pos = j + len(marker_ids)
                break
        starts.append(pos)
    return starts


def profile_distances(
    model_path: str,
    dataset: str,
    dataset_column: str = "text",
    batch_size: int = 1,
    max_length: int = 256,
    layers_to_skip: int = 8,
    dataset_size: Optional[int] = 128,
    dataset_subset: str = "train",
    use_4bit: bool = True,
    token: Optional[str] = None,
    compute_taylor: bool = True,
    taylor_veto_quantile: Optional[float] = 0.75,
    answer_only_loss: bool = False,
    answer_marker: str = "Answer:",
    csv_save_path: str = "./layer_distances.csv",
    distances_save_path: str = "./distances.pth",
    gc_every: int = 20,
) -> Dict[str, list]:
    """Rank removable blocks by activation and gradient signals.

    Block (start, end) denotes removal of 0-based layers start..end-1, matching
    truncate_model. Signals are measured between the hidden state entering that
    span, hidden_states[start], and the one leaving it, hidden_states[end].
    """
    compute_dtype = _compute_dtype()
    logging.info(f"compute dtype: {compute_dtype}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype) if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": 0} if torch.cuda.is_available() else "cpu",
        quantization_config=quant, output_hidden_states=True,
        token=token, trust_remote_code=True)
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    tok = AutoTokenizer.from_pretrained(model_path, token=token,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n_layers = model.config.num_hidden_layers
    n_cand = n_layers - layers_to_skip
    dev = next(model.parameters()).device

    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size,
        batch_size, tok)

    acc_act = [[] for _ in range(n_cand)]
    acc_grad = [[] for _ in range(n_cand)]
    acc_tay = [[] for _ in range(n_cand)]

    logging.info(
        f"{n_cand} candidates | signals: dist_act (all-token), dist_grad"
        + (", taylor" if compute_taylor else "")
        + (f" | answer_only_loss ('{answer_marker}')" if answer_only_loss
           else ""))

    for b_idx, batch in enumerate(tqdm(
            dataloader, desc=f"{Fore.GREEN}Profiling blocks{Fore.RESET}",
            dynamic_ncols=True, colour="green")):
        enc = tok(list(batch), return_tensors="pt", padding="longest",
                  max_length=max_length, truncation=True)
        enc = {k: v.to(dev) for k, v in enc.items()}

        answer_starts = None
        if answer_only_loss:
            answer_starts = _find_answer_starts(
                enc["input_ids"], tok, answer_marker)

        with torch.enable_grad():
            out = model(**enc, use_cache=False)
            hs = out.hidden_states
            for h in hs:
                if h.requires_grad:
                    h.retain_grad()
            loss = _lm_loss(out.logits, enc["input_ids"],
                            enc["attention_mask"], answer_starts)
            loss.backward()

        am = enc["attention_mask"].bool().view(-1)
        D = hs[0].shape[-1]

        for i in range(n_cand):
            enter, leave = i + 1, i + 1 + layers_to_skip

            h_in = hs[enter].detach().reshape(-1, D)[am].float()
            h_out = hs[leave].detach().reshape(-1, D)[am].float()
            acc_act[i].append(float(_angular(h_in, h_out).mean()))

            g_in_t = hs[enter].grad
            g_out_t = hs[leave].grad
            if g_in_t is not None and g_out_t is not None:
                g_in = g_in_t.detach().reshape(-1, D)[am].float()
                g_out = g_out_t.detach().reshape(-1, D)[am].float()
                acc_grad[i].append(float(_angular(g_in, g_out).mean()))
                if compute_taylor:
                    t = (g_out * (h_in - h_out)).sum(-1).abs()
                    acc_tay[i].append(float(t.mean()))

        model.zero_grad(set_to_none=True)
        del out, hs, loss
        if (b_idx + 1) % gc_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    dist_act = [float(np.mean(v)) for v in acc_act]
    dist_grad = [float(np.mean(v)) if v else float("nan") for v in acc_grad]
    taylor = [float(np.mean(v)) if v else float("nan") for v in acc_tay]

    if np.isnan(dist_grad).all():
        raise RuntimeError("dist_grad is all-NaN: backward produced no "
                           "activation gradients -- check "
                           "enable_input_require_grads / quantized graph.")

    r_act = np.argsort(np.argsort(dist_act)).astype(float)
    r_grad = np.argsort(np.argsort(dist_grad)).astype(float)
    score = (r_act + r_grad) / 2.0

    vetoed = set()
    if compute_taylor and taylor_veto_quantile is not None \
            and not np.isnan(taylor).all():
        thr = float(np.nanquantile(taylor, taylor_veto_quantile))
        vetoed = {i for i, t in enumerate(taylor)
                  if not np.isnan(t) and t > thr}
        for i in vetoed:
            score[i] = float("inf")
        logging.info(f"taylor veto: q={taylor_veto_quantile}, thr={thr:.4f}, "
                     f"vetoed {len(vetoed)} candidates")

    order = sorted(range(n_cand), key=lambda i: (score[i], dist_act[i]))
    best = order[0]
    selected_block = (best + 1, best + 1 + layers_to_skip)

    tied = [i for i in range(n_cand) if score[i] == score[best]]
    if len(tied) > 1:
        logging.info(
            "combined score ties at %.1f between %s; broken by dist_act"
            % (score[best], [(i + 1, i + 1 + layers_to_skip) for i in tied]))

    with open(csv_save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "block_start", "block_end", "dist_act", "dist_grad", "taylor",
            "rank_act", "rank_grad", "score", "vetoed", "selected"])
        w.writeheader()
        for i in range(n_cand):
            w.writerow({
                "block_start": i + 1, "block_end": i + 1 + layers_to_skip,
                "dist_act": dist_act[i], "dist_grad": dist_grad[i],
                "taylor": taylor[i],
                "rank_act": int(r_act[i]), "rank_grad": int(r_grad[i]),
                "score": score[i], "vetoed": i in vetoed,
                "selected": i == best})

    forced = [1.0] * n_cand
    forced[best] = 0.0
    torch.save(forced, distances_save_path)

    logging.info(f"CSV -> {csv_save_path} | distances -> {distances_save_path}")
    logging.info("top-3 by combined score:")
    for rank, i in enumerate(order[:3], 1):
        logging.info(
            f"  #{rank}: block ({i + 1},{i + 1 + layers_to_skip})  "
            f"act={dist_act[i]:.4f} (r{int(r_act[i])})  "
            f"grad={dist_grad[i]:.4f} (r{int(r_grad[i])})  "
            f"score={score[i]:.1f}")
    logging.info(f"{Fore.GREEN}SELECTED: {selected_block}{Fore.RESET} "
                 f"(combined rank score; pre-heal signal -- for the gold "
                 f"standard run held-out arbitration on the top-3)")

    return {"dist_act": dist_act, "dist_grad": dist_grad, "taylor": taylor,
            "score": score.tolist(), "selected_block": selected_block}
