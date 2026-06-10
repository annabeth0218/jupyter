from __future__ import annotations
import time, datetime, platform
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STAGE_DIR = "opath"
CACHE_PT = f"{os.path.expandvars(STAGE_DIR)}/cache_0503.pt"
PROMPT_FILE = Path(os.environ.get("PROMPT_FILE", "../prompts/base.txt"))
PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

LLM_ID = "Qwen/Qwen2.5-7B-Instruct"
V_TOKENS = 32              
LR = 1e-4                  
BSZ = 8
EPOCHS = 8
MAX_TXT_TOK = 96
FP16 = False
WEIGHT_DECAY = 0      
DROPOUT      = 0       
WARMUP_STEPS = 20 

# Split / checkpoint locations
VAL_FRAC = 0.10
SPLIT_PATH = Path("../checkpoints/val_idx.json")

def make_run_id() -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    n = int(time.time()) // 60   # minutes since epoch
    result = []
    while n:
        result.append(chars[n % 36])
        n //= 36
    return "".join(reversed(result)) 
    
RUN_ID = make_run_id()
CKPT_PATH = Path(f"../checkpoints/proj_{RUN_ID}.pt")
LOSS_PNG  = Path(f"../checkpoints/loss_{RUN_ID}.png")
MODEL_CARD_PATH = Path(f"../checkpoints/proj_card_{RUN_ID}.json")

# Fallback diagnosis text used when the manifest has no `disease` field.
MISSING_DIAGNOSIS = "Pathological diagnosis: cannot tell from this slide"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def write_model_card(path: Path, run_id: str, val_idx: list) -> None:
    card = {
        "run_id":        run_id,
        "created_at":    datetime.datetime.now().isoformat(),
        "host":          platform.node(),
        # data
        "cache":         CACHE_PT,
        "n_train":       N - len(val_idx),
        "n_val":         len(val_idx),
        # architecture
        "encoder":       "CONCH ViT-B/16",
        "llm":           LLM_ID,
        "v_tokens":      V_TOKENS,
        # training
        "lr":            LR,
        "batch_size":    BSZ,
        "epochs":        EPOCHS,
        "weight_decay":  WEIGHT_DECAY,
        "dropout":       DROPOUT,
        "warmup_steps":  WARMUP_STEPS,
        "max_txt_tok":   MAX_TXT_TOK,
        "fp16":          FP16,
        "val_frac":      VAL_FRAC,
        # paths
        "ckpt_path":     str(CKPT_PATH),
        "loss_png":      str(LOSS_PNG),
        "prompt_file":   str(PROMPT_FILE),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"Model card written to {path}")

# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache(path: str = CACHE_PT):
    """Load the CONCH embedding cache and pull out only the fields we use."""
    ckpt = torch.load(path, map_location="cpu")
    img_embs = ckpt["embeddings"].float()
    img_embs = torch.nn.functional.normalize(img_embs, dim=1)
    meta = ckpt.get("meta", {}) or {}
    n = img_embs.shape[0]

    captions = list(meta.get("captions", [""] * n))
    diseases = list(meta.get("disease", [""] * n))
    image_paths = list(meta.get("image_paths", [""] * n))

    # Make sure all parallel lists are the right length.
    for lst, name in (
        (captions, "captions"),
        (diseases, "disease"),
        (image_paths, "image_paths"),
    ):
        if len(lst) < n:
            lst.extend([""] * (n - len(lst)))
        elif len(lst) > n:
            del lst[n:]
        if name == "captions":
            captions[:] = lst
        elif name == "disease":
            diseases[:] = lst
        else:
            image_paths[:] = lst

    return img_embs, captions, diseases, image_paths


# Initial load (so this script behaves like the original when run top-to-bottom)
img_embs, captions, diseases, image_paths = load_cache(CACHE_PT)
N, D = img_embs.shape
print(f"Cache: {N} samples, CONCH dim={D}")


# ---------------------------------------------------------------------------
# LLM + projector
# ---------------------------------------------------------------------------

torch.cuda.empty_cache()
dtype = torch.float16 if FP16 else torch.bfloat16

tok = AutoTokenizer.from_pretrained(LLM_ID, use_fast=True,trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    dtype=dtype,
    #device_map="auto",
    device_map={"": 0},
    trust_remote_code=True
)
H = model.config.hidden_size
print("Hidden size:", H)


# Freeze the LLM; only the projector trains.
for p in model.parameters():
    p.requires_grad = False


class Projector(nn.Module):
    """Maps a CONCH image embedding [B, D] to V_TOKENS pseudo-tokens [B, V, H]."""

    def __init__(self, d_in: int, h_out: int, v_tokens: int = 2):
        super().__init__()
        self.v_tokens = v_tokens
        self.mlp = nn.Sequential(
            nn.Linear(d_in, h_out),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(h_out, h_out * v_tokens),
        )
        self.pos = nn.Parameter(torch.zeros(v_tokens, h_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)
        y = y.view(x.size(0), self.v_tokens, -1)
        return y + self.pos.unsqueeze(0)


device = next(model.parameters()).device
proj = Projector(D, H, V_TOKENS).to(device).to(dtype)
optim = torch.optim.AdamW(proj.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

from transformers import get_cosine_schedule_with_warmup
total_steps = (N * (1 - VAL_FRAC) // BSZ + 1) * EPOCHS
sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def diagnosis_line(disease: str) -> str:
    """Build the 'Pathological diagnosis: ...' line, with a graceful fallback."""
    disease = (disease or "").strip()
    if not disease:
        return MISSING_DIAGNOSIS
    return f"Pathological diagnosis: {disease}"


def build_target(caption: str, disease: str) -> str:
    """The supervision target: caption + diagnosis line."""
    caption = (caption or "").strip()
    return f"Microscopic findings: {caption}\n{diagnosis_line(disease)}"


class PairSet(Dataset):
    def __init__(
        self,
        embs: torch.Tensor,
        captions: List[str],
        diseases: List[str],
        tokenizer,
        max_len: int = MAX_TXT_TOK,
        prompt: str = PROMPT,
    ):
        assert embs.size(0) == len(captions) == len(diseases), (
            "embs / captions / diseases length mismatch"
        )
        self.embs = embs
        self.cap = captions
        self.dis = diseases
        self.tok = tokenizer
        self.max_len = max_len

        self.prompt_tpl = prompt
        enc = self.tok(self.prompt_tpl, add_special_tokens=False, return_tensors="pt")
        self.prompt_ids = enc.input_ids[0]
        self.prompt_len = self.prompt_ids.numel()

    def __len__(self) -> int:
        return self.embs.size(0)

    def __getitem__(self, i: int):
        target = build_target(self.cap[i], self.dis[i])
        text = self.prompt_tpl + target

        prompt_enc = self.tok(
            self.prompt_tpl,
            add_special_tokens=False,
            return_tensors="pt",
        )
        target_enc = self.tok(
            target,
            max_length=self.max_len,   # MAX_TXT_TOK now only limits target
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
    
        input_ids = torch.cat([prompt_enc.input_ids[0], target_enc.input_ids[0]])
        attn_mask = torch.ones(input_ids.size(0), dtype=torch.long)
    
        prompt_len = prompt_enc.input_ids[0].size(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
    
        eos = self.tok.eos_token_id
        if input_ids[-1].item() != eos:
            eos_tensor = torch.tensor([eos], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, eos_tensor])
            attn_mask = torch.cat([attn_mask, torch.ones(1, dtype=attn_mask.dtype)])
            labels = torch.cat([labels, eos_tensor]) 

        return {
            "emb": self.embs[i],
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "labels": labels,
            "prompt_len": prompt_len,
        }


img_embs_dev = img_embs.to(device=device, dtype=dtype)
ds = PairSet(img_embs_dev, captions, diseases, tok, MAX_TXT_TOK)


# ---------------------------------------------------------------------------
# Train / val split (random 80/20, persistable)
# ---------------------------------------------------------------------------

def make_split(
    n: int,
    val_frac: float = VAL_FRAC,
    seed: Optional[int] = None,
) -> List[int]:
    """Random 80/20 split. Returns sorted val_idx; train_idx is the complement."""
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = rng.choice(n, size=n_val, replace=False)
    return sorted(int(i) for i in val_idx)


def save_split(val_idx: List[int], path: Path = SPLIT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"val_idx": val_idx}), encoding="utf-8")


def load_split(path: Path = SPLIT_PATH) -> List[int]:
    return json.loads(path.read_text(encoding="utf-8"))["val_idx"]


def get_split(reuse: bool, n: int, path: Path = SPLIT_PATH) -> List[int]:
    """Load saved val_idx if `reuse` is True and the file exists; else resplit."""
    if reuse and path.exists():
        val_idx = load_split(path)
        # Drop indices that fell out of range if the dataset shrunk.
        val_idx = [i for i in val_idx if 0 <= i < n]
        print(f"Reusing saved val split from {path} ({len(val_idx)} indices).")
        return val_idx
    val_idx = make_split(n)
    save_split(val_idx, path)
    print(f"Created new val split, saved to {path} ({len(val_idx)} indices).")
    return val_idx


# Default behavior matches the file's old top-level execution: re-split each run.
# Flip REUSE_SPLIT to True (or use --reuse-split on the CLI) to reuse.
REUSE_SPLIT = False
val_idx = get_split(REUSE_SPLIT, N)
train_idx = [i for i in range(N) if i not in set(val_idx)]
print(f"train: {len(train_idx)} | val: {len(val_idx)}")

train = Subset(ds, train_idx)
val = Subset(ds, val_idx)


# ---------------------------------------------------------------------------
# Collate / training step
# ---------------------------------------------------------------------------

def collate(batch):
    emb = torch.stack([b["emb"] for b in batch])
    ids = [b["input_ids"] for b in batch]
    labs = [b["labels"] for b in batch]
    lens = [len(x) for x in ids]
    maxL = max(lens)
    pad_id = tok.pad_token_id or tok.eos_token_id

    ids_pad  = torch.full((len(batch), maxL), pad_id, dtype=torch.long)
    labs_pad = torch.full((len(batch), maxL), -100,   dtype=torch.long)  # pad labels with -100
    attn     = torch.zeros((len(batch), maxL), dtype=torch.long)

    for i, (x, l) in enumerate(zip(ids, labs)):
        ids_pad[i,  :len(x)] = x
        labs_pad[i, :len(l)] = l
        attn[i,     :len(x)] = 1

    return {
        "emb":  emb,
        "ids":  ids_pad.to(device),
        "labs": labs_pad.to(device),
        "attn": attn.to(device),
    }


loader = DataLoader(train, batch_size=BSZ, shuffle=True, collate_fn=collate, drop_last=False)
vloader = DataLoader(val, batch_size=BSZ, shuffle=False, collate_fn=collate)

loss_fct = nn.CrossEntropyLoss(ignore_index=-100)


def step_batch(batch, train_mode: bool = True) -> float:
    
    proj.train(train_mode)
    model.eval()
    optim.zero_grad(set_to_none=True)

    B = batch["emb"].size(0)
    v_emb = proj(batch["emb"])                              # [B, V, H]
    txt_emb = model.get_input_embeddings()(batch["ids"])    # [B, T, H]
    inputs_embeds = torch.cat([v_emb, txt_emb], dim=1)      # [B, V+T, H]

    vis_attn = torch.ones(B, V_TOKENS, device=device, dtype=batch["attn"].dtype)
    attn = torch.cat([vis_attn, batch["attn"]], dim=1)

    labels = torch.cat(
        [torch.full((B, V_TOKENS), -100, device=device, dtype=torch.long),
         batch["labs"]],
        dim=1,
    )

    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
    loss = out.loss

    if not torch.isfinite(loss):
        print("non-finite loss detected")
        print("  v_emb finite:",   torch.isfinite(v_emb).all().item(),
              "min/max:", v_emb.min().item(), v_emb.max().item())
        print("  txt_emb finite:", torch.isfinite(txt_emb).all().item())
        print("  logits stats:",   out.logits.min().item(), out.logits.max().item(),
              "any nan:", torch.isnan(out.logits).any().item())
        raise RuntimeError("stop")

    if train_mode:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        optim.step()
        sched.step()
        
    return float(loss.detach().cpu())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Per-batch loss histories (kept module-level so loss_graph() can plot them).
tl: List[float] = []
vl: List[float] = []


def train_loop(epochs: int = EPOCHS, ckpt_path: Path = CKPT_PATH) -> float:
    """Run the training loop with simple early stopping on mean val loss."""
    global tl, vl
    tl, vl = [], []
    best = math.inf
    stop = 0
    epoch_log = [] 

    for ep in range(epochs):
        # ---- TRAIN ----
        last_tloss = None
        epoch_tlosses = []
        for batch in tqdm(loader, desc=f"train ep{ep+1}/{epochs}"):
            last_tloss = step_batch(batch, train_mode=True)
            tl.append(last_tloss)

        # ---- VAL ----
        epoch_vlosses: List[float] = []
        with torch.no_grad():
            for batch in vloader:
                vloss = step_batch(batch, train_mode=False)
                epoch_vlosses.append(vloss)
                vl.append(vloss)

        if not epoch_vlosses:
            print(f"epoch {ep+1}: WARNING empty validation set, skipping early stop")
            continue

        mean_vl = sum(epoch_vlosses) / len(epoch_vlosses)
        print(f"epoch {ep+1}: train {last_tloss:.3f} | val {mean_vl:.3f}")

        if best - mean_vl > 0.01:
            best = mean_vl
            stop = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "proj": proj.state_dict(),
                    "D": D,
                    "H": H,
                    "V": V_TOKENS,
                    "llm": LLM_ID,
                },
                ckpt_path,
            )
        else:
            stop += 1
            if stop > 1:
                break

    return best


# ---------------------------------------------------------------------------
# Loss graph
# ---------------------------------------------------------------------------

def loss_graph(out_path: Path = LOSS_PNG) -> Path:
    """Plot the per-batch train and val loss curves and save to disk."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(tl, label="train (per batch)", alpha=0.7)
    plt.plot(vl, label="val (per batch)", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over training iterations")
    plt.ylim(0, 5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Demo generation on a single sample
# ---------------------------------------------------------------------------

def demo(
    i: Optional[int] = None,
    *,
    pool: Optional[List[int]] = None,
    prompt: str = PROMPT,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a caption from one cached embedding for a sanity check.

    If `i` is None, a random index is picked from `pool` (default: val_idx).
    Returns the generated text and prints both the reference and the output.
    """
    if i is None:
        candidates = pool if pool is not None else val_idx
        if not candidates:
            candidates = list(range(N))
        i = random.choice(candidates)

    ref = build_target(captions[i], diseases[i])
    print(f"--- Reference (idx={i}) ---")
    print(ref)

    proj.eval()
    demo_emb = img_embs_dev[i].unsqueeze(0)
    with torch.no_grad():
        v = proj(demo_emb)
        ids = tok(prompt, return_tensors="pt").to(device)
        t_emb = model.get_input_embeddings()(ids.input_ids)
        inp = torch.cat([v, t_emb], dim=1)
        attn = torch.cat(
            [
                torch.ones(1, V_TOKENS, device=device, dtype=torch.long),
                ids.attention_mask,
            ],
            dim=1,
        )
        gen = model.generate(
            inputs_embeds=inp,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(gen[0], skip_special_tokens=True)
    output = decoded.split(prompt, 1)[-1].strip()
    print("\n=== DEMO OUTPUT ===")
    print(output)
    return output


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train projector head for CONCH→Qwen2.")
    p.add_argument(
        "--reuse-split",
        action="store_true",
        help="Reuse the val_idx saved at SPLIT_PATH instead of re-splitting.",
    )
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument(
        "--demo",
        type=int,
        default=None,
        help="Run demo() on this index after training. -1 picks randomly.",
    )
    p.add_argument("--no-graph", action="store_true", help="Skip the loss graph.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Re-resolve the split if --reuse-split flips the default.
    if args.reuse_split and not REUSE_SPLIT:
        val_idx = get_split(True, N)
        train_idx = [i for i in range(N) if i not in set(val_idx)]
        train = Subset(ds, train_idx)
        val = Subset(ds, val_idx)
        loader = DataLoader(train, batch_size=BSZ, shuffle=True, collate_fn=collate, drop_last=False)
        vloader = DataLoader(val, batch_size=BSZ, shuffle=False, collate_fn=collate)
        print(f"(reuse) train: {len(train_idx)} | val: {len(val_idx)}")

    write_model_card(MODEL_CARD_PATH, RUN_ID, val_idx)
    best = train_loop(epochs=args.epochs)
    print(f"Best val loss: {best:.4f}")
    card = json.loads(MODEL_CARD_PATH.read_text())
    card["best_val_loss"] = round(best, 6)
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))

    if not args.no_graph:
        out = loss_graph()
        print(f"Loss curve saved to {out}")

    if args.demo is not None:
        demo(None if args.demo < 0 else args.demo)
