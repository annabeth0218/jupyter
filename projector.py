# Qwen + 2-layer projector fine-tune (projector-only) for a quick baseline demo
# Assumes: conch_cache.pt produced earlier
# torch.cuda.empty_cache()

import os, math, json, random, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# --- paths ---
STAGE_DIR = "who4e/for_demo"  # assuming cd ~/Anna_CONCH/CONCH
CACHE_PT  = f"{os.path.expandvars(STAGE_DIR)}/conch_cache.pt"

# --- config (safe defaults for shared 8×A100) ---
LLM_ID        = "Qwen/Qwen2-7B-Instruct"   # you can switch to Qwen2-7B if you prefer non-instruct
V_TOKENS      = 4                          # how many “visual tokens” to prepend
LR            = 8e-4                       # projector-only lr
BSZ           = 6                          # batch size (adjust to VRAM)
EPOCHS        = 6                          # tiny fit for a demo
MAX_TXT_TOK   = 120                        # cap length
FP16          = False

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- load cache ---
ckpt = torch.load(CACHE_PT, map_location="cpu")
img_embs = ckpt["embeddings"].float()    # [N, D]
N, D = img_embs.shape

print(f"Cache: {N} samples, CONCH dim={D}") # 221, 512
meta     = ckpt["meta"]
diseases = meta.get("disease", [""]*N)
subclses = meta.get("subcls", [""]*N)
clses    = meta.get("cls", [""]*N)

# --- combine title and cap ---
def low_st(txt):
    return txt[:1].lower() + txt[1:] if txt else ""
    
def fmt_cap(title, cap, disease):
    return ((f"{title} is showed. " if (title and disease.lower() not in title.lower()) else "")
    + (f"{cap}" if cap else ""))

captions = []
for i in range(N):
    title   = meta.get("title",   [""]*N)[i]
    cap     = meta.get("captions",[""]*N)[i]
    captions.append(fmt_cap(title, cap, diseases[i]))

assert N == len(captions), "meta mismatch"

# --- load Qwen ---
torch.cuda.empty_cache()
dtype = torch.float16 if FP16 else torch.bfloat16
tok = AutoTokenizer.from_pretrained(LLM_ID, use_fast=True)
# if tok.pad_token_id is None:
#     tok.pad_token_id = tok.eos_token_id # prevent invalid indices in labels
model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype=dtype,
    device_map="auto"
)
H = model.config.hidden_size
print("Qwen hidden size:", H) # 3584

# --- freeze LLM; train only projector ---
for p in model.parameters(): p.requires_grad = False

# projector: D -> (V_TOKENS * H)
class Projector(nn.Module):
    def __init__(self, d_in, h_out, v_tokens=2):
        super().__init__()
        self.v_tokens = v_tokens
        self.mlp = nn.Sequential(
            nn.Linear(d_in, h_out),
            nn.GELU(),
            nn.Linear(h_out, h_out * v_tokens)
        )
        # tiny learned pos offsets for visual tokens (helps a touch)
        self.pos = nn.Parameter(torch.zeros(v_tokens, h_out))
    def forward(self, x):  # x: [B, D]
        y = self.mlp(x)                          # [B, v*h]
        y = y.view(x.size(0), self.v_tokens, -1) # [B, V, H]
        return y + self.pos.unsqueeze(0)         # [B, V, H]

proj = Projector(D, H, V_TOKENS).to(next(model.parameters()).device).to(dtype)
optim = torch.optim.AdamW(proj.parameters(), lr=LR, weight_decay=0.01)

# --- dataset ---
class PairSet(Dataset):
    def __init__(self, embs, disease, subcls, cls_, cap, tokenizer, max_len=MAX_TXT_TOK):
        self.embs = embs
        self.dis = disease
        self.sub = subcls
        self.cls = cls_
        self.cap = cap
        self.tok  = tokenizer
        self.max_len = max_len
# -
        N = embs.size(0)
        assert len(captions) == N, "embs and captions length mismatch"
# -
        self.prompt_tpl = "A pathology image of the eye is presented. As a senior ophthalmic pathologist, provide a concise diagnostic description that identifies the disease, specifies its subclass and broader classification, and essential histologic findings.\n"
        _enc = self.tok(self.prompt_tpl, add_special_tokens=False, return_tensors="pt")
        self.prompt_ids = _enc.input_ids[0]
        self.prompt_len = self.prompt_ids.numel()
# -
    def __len__(self): 
        return self.embs.size(0)
# -
    def __getitem__(self, i):
        diag = f"Pathologic diagnosis: {self.dis[i]}" + \
               (f", classified as {low_st(self.sub[i])} within the broader category of {low_st(self.cls[i])}" 
                if (self.sub[i] or self.cls[i]) else "")
        micro = f".\nMicroscopic findings: {self.cap[i]}"
        target = f"{diag}\n{micro}"
        text = self.prompt_tpl + target
# -
        enc = self.tok(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
# -
        labels = enc.input_ids[0].clone()
        cut = min(self.prompt_len, labels.numel())
        labels[:cut] = -100  # ignore_index for CrossEntropyLoss, mask prompt for loss
# -
        return {
            "emb": self.embs[i],
            "input_ids": enc.input_ids[0],
            "attn_mask": enc.attention_mask[0],
            "labels": labels,
            "prompt_len": cut
        }

# move embeddings to the LLM device for speed
device = next(model.parameters()).device
img_embs_dev = img_embs.to(device=device, dtype=dtype)
ds   = PairSet(img_embs_dev, diseases, subclses, clses, captions, tok, MAX_TXT_TOK)

# --- split test/train ---
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
# -
df = pd.DataFrame({
    "idx": np.arange(N),
    "cls": clses,
    "subcls": subclses,
    "disease": diseases,
    "number": meta.get("number", [""]*N)
})
# -
fixed_train_idx = [15, 21, 53, 65, 71, 75, 76, 80, 92, 107, 161, 169, 185]
df_rest = df[~df["idx"].isin(fixed_train_idx)].copy()
is_val = np.zeros(len(df), dtype=bool)
rng = np.random.default_rng()
for d, sub in df_rest.groupby("disease"):
    nums = sub["number"].astype(str).fillna("").unique().tolist()
    if len(nums) >= 3:
        k = max(1, int(round(0.1 * len(nums))))
        pick = set(rng.choice(nums, size=k, replace=False))
        chosen_idx = sub.index[sub["number"].astype(str).isin(pick)]
        is_val[chosen_idx] = True

is_val[df.index[df["idx"].isin(fixed_train_idx)]] = False
val_idx = df.loc[is_val, "idx"].tolist()
# val_idx = [2, 38, 39, 44, 46, 77, 83, 89, 111, 114, 123, 130, 133, 147, 148, 172, 173, 176, 198, 205, 215]
# train_idx = [i for i in range(N) if i not in val_idx]
train_idx = df.loc[~is_val, "idx"].tolist()
'''
# --- k-fold ---
keep_df = df[~df["idx"].isin(exclude)].reset_index(drop=True)
keep_df["cls_subcls_disease"] = (
    keep_df["cls"].astype(str) + " | " +
    keep_df["subcls"].astype(str) + " | " +
    keep_df["disease"].astype(str)
)
X = keep_df["idx"].values
groups = keep_df["cls_subcls_disease"].values
gkf = GroupKFold(n_splits=10)
folds = []
for tr_i, va_i in gkf.split(X, groups=groups):
    train_idx = keep_df.loc[tr_i, "idx"].tolist() + exclude
    val_idx   = keep_df.loc[va_i, "idx"].tolist()
    folds.append((train_idx, val_idx))
k = 0
train_idx, val_idx = folds[k][0], folds[k][1]
print(len(train_idx), len(val_idx))
'''
print("train:", len(train_idx), "val:", len(val_idx))
train = torch.utils.data.Subset(ds, train_idx)
val   = torch.utils.data.Subset(ds, val_idx)

def collate(batch):
    emb = torch.stack([b["emb"] for b in batch])                         # [B, D]
    ids = [b["input_ids"] for b in batch]
    lens = [len(x) for x in ids]
    maxL = max(lens)
    pad_id = tok.pad_token_id or tok.eos_token_id
    ids_pad = torch.full((len(batch), maxL), pad_id, dtype=torch.long)
    attn    = torch.zeros((len(batch), maxL), dtype=torch.long)
    prompt_lens = []
    for i,(x,pl) in enumerate(zip(ids, [b["prompt_len"] for b in batch])):
        ids_pad[i,:len(x)] = x
        attn[i,:len(x)]    = 1
        prompt_lens.append(pl)
    return {"emb": emb, "ids": ids_pad.to(device), "attn": attn.to(device), "plens": torch.tensor(prompt_lens, device=device)}

loader = DataLoader(train, batch_size=BSZ, shuffle=True, collate_fn=collate, drop_last=False)
vloader = DataLoader(val,   batch_size=BSZ, shuffle=False, collate_fn=collate)

# --- training: build inputs_embeds = [visual_embeds, text_embeds], loss on text only (ignore visual & prompt) ---
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

def step_batch(batch, train_mode=True):
    proj.train(train_mode); model.eval()
    optim.zero_grad(set_to_none=True)
# -
    B = batch["emb"].size(0)
    v_emb = proj(batch["emb"])                                     # [B, V, H]
    txt_emb = model.get_input_embeddings()(batch["ids"])           # [B, T, H]
    inputs_embeds = torch.cat([v_emb, txt_emb], dim=1)             # [B, V+T, H]
# -
    # attention mask extends for V tokens
    attn = torch.cat([torch.ones(B, V_TOKENS, device=device, dtype=torch.long), batch["attn"]], dim=1)
# -
    # labels: predict next token for text positions; mask visual tokens and the prompt
    labels = batch["ids"].clone()
    # mask prompt region
    for i,plen in enumerate(batch["plens"]):
        if plen < labels.size(1):
            labels[i,:plen] = -100
    # shift labels to align with model outputs when using inputs_embeds
    # We'll feed labels padded with -100 for the V visual tokens in front
    labels = torch.cat([torch.full((B, V_TOKENS), -100, device=device, dtype=torch.long), labels], dim=1)
# -
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
    loss = out.loss
    if train_mode:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        optim.step()
    return float(loss.detach().cpu())

best = math.inf
import matplotlib.pyplot as plt
tl, vl = [], []

for ep in range(EPOCHS):
    for batch in tqdm(loader, desc=f"train ep{ep+1}/{EPOCHS}"):
        tloss = step_batch(batch, train_mode=True)
        tl.append(tloss)
# -
    # quick val
    with torch.no_grad():
        for batch in vloader:
            vloss = step_batch(batch, train_mode=False)
            vl.append(vloss)
# -
    print(f"epoch {ep+1}: train {min(tl):.3f} | val {min(vl):.3f}")
    if min(vl) < best:
        best = min(vl)
        torch.save({"proj": proj.state_dict(), "D": D, "H": H, "V": V_TOKENS, "llm": LLM_ID}, f"{STAGE_DIR}/projector_best.pt")
    else:
        if ep > 0:
            break

# --- loss func graph ---
plt.plot(tl, label="train (per batch)", alpha=0.7)
plt.plot(vl, label="val (per batch)", alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over training iterations")
plt.ylim(0, 5)
plt.savefig(f"{os.path.expandvars(STAGE_DIR)}/loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# --- demo generation on a random sample ---
proj.eval()
i = random.choice(val_idx) # 2
prompt   = f"A pathology image of the eye is presented. As a senior ophthalmic pathologist, provide a concise diagnostic description that identifies the disease, specifies its subclass and broader classification, and essential histologic findings.\n"
print(f"Pathologic diagnosis: {diseases[i]}" + (f", classified as {low_st(subclses[i])} within the broader category of {low_st(clses[i])}" if (subclses[i] or clses[i]) else "") + f".\nMicroscopic findings: {captions[i]}")

demo_emb = img_embs_dev[i].unsqueeze(0)           # [1, D]
with torch.no_grad():
    v = proj(demo_emb)                             # [1, V, H]
    ids = tok(prompt, return_tensors="pt").to(device)
    t_emb = model.get_input_embeddings()(ids.input_ids)   # [1, T, H]
    inp   = torch.cat([v, t_emb], dim=1)
    attn  = torch.cat([torch.ones(1, V_TOKENS, device=device, dtype=torch.long), ids.attention_mask], dim=1)
    # --
    gen = model.generate(
        inputs_embeds=inp,
        attention_mask=attn,
        max_new_tokens=96,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tok.eos_token_id
    )
    
# decode: skip the visual tokens (not textual), decode only the text portion
# generate returns full sequence for text tokens we provided; just decode tail
decoded = tok.decode(gen[0], skip_special_tokens=True)
print("\n=== DEMO OUTPUT ===")
print(decoded.split(prompt, 1)[-1].strip())
