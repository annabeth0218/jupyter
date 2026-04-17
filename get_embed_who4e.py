# last updated: 09/19/25 23:08

import os, json, torch, random
from PIL import Image
from tqdm import tqdm

# --- Quick manifest + image sanity check ---
import pathlib, io, sys

STAGE_DIR = "who4e/for_demo"  # assuming cd ~/Anna_CONCH/CONCH
MANIFEST  = f"{STAGE_DIR}/manifest.jsonl"
CACHE_OUT = f"{STAGE_DIR}/conch_cache.pt"

stage = pathlib.Path(os.path.expandvars(STAGE_DIR)).resolve()
man   = pathlib.Path(os.path.expandvars(MANIFEST)).resolve()
# assert stage.exists(), f"Stage dir not found: {stage}" # for check
# assert man.exists(),   f"Manifest not found: {man}" # for check

rows, bad = [], []
with man.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        rec = json.loads(line)
        # required fields
        for k in ["image","caption","title","number","cls","disease","subcls"]:
            if k not in rec: bad.append((i, f"missing {k}")); break
        else:
            img_path = stage / rec["image"]
            if not img_path.exists(): bad.append((i, f"missing image file: {img_path}"))
            else:
                # try opening to catch corrupt files
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    bad.append((i, f"image not readable: {img_path} ({e})"))
            rows.append(rec)

# --- loading CONCH encoder- --
device = "cuda" if torch.cuda.is_available() else "cpu"

from conch.open_clip_custom import create_model_from_pretrained # CONCH API
model, preprocess = create_model_from_pretrained(
    'conch_ViT-B-16', "hf_hub:MahmoodLab/conch", 
    hf_auth_token="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")

model = model.to(device).eval()

# --- Embed images and save cache ---
valid = []
for rec in rows:
    p = stage / rec["image"]
    if p.exists():
        valid.append((p, 
                      rec["caption"], 
                      rec.get("number",""),
                      rec.get("title",""), 
                      rec.get("disease",""), 
                      rec.get("subcls",""),
                      rec.get("cls","")
                     ))

print(f"Embedding {len(valid)} images (skipping {len(rows)-len(valid)} missing/corrupt).")

all_embs = []
meta = {"captions": [], "image_paths": [], "number": [], "title": [], "disease": [], "subcls": [], "cls": []}

with torch.inference_mode():
    for (img_path, cap, number, title, disease, subcls, cls) in tqdm(valid, total=len(valid), desc="CONCH embedding"):
        img = Image.open(img_path).convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(device)
        emb = model.encode_image(img_t, proj_contrast=False, normalize=False)  # [1, D]
        all_embs.append(emb.squeeze(0).cpu())
        meta["captions"].append(cap)
        meta["image_paths"].append(str(img_path))
        meta["number"].append(number)
        meta["title"].append(title)
        meta["disease"].append(disease)
        meta["subcls"].append(subcls)
        meta["cls"].append(cls)

assert len(all_embs) == len(valid) == len(meta["captions"]) == len(meta["image_paths"])
embeddings = torch.stack(all_embs) if all_embs else torch.empty((0,0))
torch.save({"embeddings": embeddings, "meta": meta, "encoder": "CONCH ViT-B/16"}, CACHE_OUT)
print(f"Saved cache to: {CACHE_OUT} | embeddings shape: {tuple(embeddings.shape)}")
# Saved cache to: who4e/upload/conch_cache.pt | embeddings shape: (221, 512)
