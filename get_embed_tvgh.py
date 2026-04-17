# last updated: 09/19/25 23:08

import os, json, torch, random
from PIL import Image
from tqdm import tqdm
import pathlib, io, sys, json

# --- Quick manifest + image sanity check ---
STAGE_DIR = "hamamatsu"  # assuming cd ~/Anna_CONCH/CONCH
MANIFEST  = f"{STAGE_DIR}/trial_0416_fixed.jsonl"
CACHE_OUT = f"{STAGE_DIR}/conch_cache.pt"
stage = pathlib.Path(os.path.expandvars(STAGE_DIR)).resolve()
man   = pathlib.Path(os.path.expandvars(MANIFEST)).resolve()
# assert stage.exists(), f"Stage dir not found: {stage}" # for check
# assert man.exists(),   f"Manifest not found: {man}" # for check

# clear '\n' –––––––––––––––––––
input_file = f"{STAGE_DIR}/trial_0416.jsonl"
output_file = f"{STAGE_DIR}/trial_0416_fixed.jsonl"
fixed_lines = []

with open(input_file, "r", encoding="utf-8") as f:
    # Read the whole file as a single string
    content = f.read()
    # Split by the closing brace followed by a newline 
    # to identify the actual end of a JSON object
    records = content.split('}\n{')
    for i, rec in enumerate(records):
        # Re-add the braces removed by split()
        if not rec.startswith('{'):
            rec = '{' + rec
        if not rec.endswith('}'):
            rec = rec + '}'
        # Replace literal newlines with spaces
        # but keep the one at the end of the JSON object
        clean_rec = rec.replace('\n', ' ')
        fixed_lines.append(clean_rec)

# Write out the new JSONL file
with open(output_file, "w", encoding="utf-8") as f:
    for line in fixed_lines:
        f.write(line + "\n")

rows, bad = [], []
with man.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        rec = json.loads(line)
        # required fields
        for k in ["image","id","cls","gross","microscopic"]:
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
                      rec["id"], 
                      rec.get("cls",""),
                      rec.get("gross",""), 
                      rec.get("microscopic",""), 
                     ))

print(f"Embedding {len(valid)} images (skipping {len(rows)-len(valid)} missing/corrupt).")

all_embs = []
meta = {"id": [], "image_paths": [], "cls": [], "gross": [], "microscopic": []}

with torch.inference_mode():
    for (img_path, id, cls, gross, microscopic) in tqdm(valid, total=len(valid), desc="CONCH embedding"):
        img = Image.open(img_path).convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(device)
        emb = model.encode_image(img_t, proj_contrast=False, normalize=False)  # [1, D]
        all_embs.append(emb.squeeze(0).cpu())
        meta["id"].append(id)
        meta["image_paths"].append(str(img_path))
        meta["cls"].append(cls)
        meta["gross"].append(gross)
        meta["microscopic"].append(microscopic)

assert len(all_embs) == len(valid) == len(meta["id"]) == len(meta["image_paths"])
embeddings = torch.stack(all_embs) if all_embs else torch.empty((0,0))
torch.save({"embeddings": embeddings, "meta": meta, "encoder": "CONCH ViT-B/16"}, CACHE_OUT)
torch.save(torch.load("conch_cache.pt")[0:], "conch_cache.pt")
print(f"Saved cache to: {CACHE_OUT} | embeddings shape: {tuple(embeddings.shape)}")

