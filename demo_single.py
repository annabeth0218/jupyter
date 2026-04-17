# cd Anna_CONCH/CONCH/

# ------------------------------single demo------------------------------
import os, torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- paths ---
STAGE_DIR = "./CONCH/hamamatsu"
CACHE_PT  = f"{os.path.expandvars(STAGE_DIR)}/conch_cache.pt"
CKPT_PT   = f"./CONCH/who4e/for_demo/projector_best_0928.pt"

# --- load cache ---
ckpt = torch.load(CACHE_PT, map_location="cpu")
img_embs = ckpt["embeddings"].float()
meta     = ckpt["meta"]
N, D = img_embs.shape
print("Cache loaded:", N, "samples, emb dim", D)

def low_st(txt):
    return txt[:1].lower() + txt[1:] if txt else ""

def fmt_cap(title, disease, subcls, cls, cap):
    return (
    f"Pathologic diagnosis: {disease}"
    + (f", classified as {low_st(subcls)} within the broader category of {low_st(cls)}" if subcls or cls else "")
    + (f".\nMicroscopic findings: ")
    + (f"{title} is showed. " if (title and disease.lower() not in title.lower()) else "")
    + (f"{cap}" if cap else "")
)

gt = [] # ground truth
for i in range(N): 
    title   = meta.get("title",   [""]*N)[i]
    disease = meta.get("disease", [""]*N)[i]
    subcls  = meta.get("subcls",  [""]*N)[i]
    cls     = meta.get("cls",     [""]*N)[i]
    cap     = meta.get("captions",[""]*N)[i]
    gt.append(fmt_cap(title, disease, subcls, cls, cap))

# --- load Qwen in 4-bit bf16 ---
LLM_ID = "Qwen/Qwen2-7B-Instruct"
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(LLM_ID, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
    
model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    quantization_config=bnb_cfg,
    device_map="auto"
)
device = next(model.parameters()).device
H = model.config.hidden_size
print("Qwen hidden size:", H)

# --- rebuild projector & load weights ---
class Projector(torch.nn.Module):
    def __init__(self, d_in, h_out, v_tokens=4):
        super().__init__()
        self.v_tokens = v_tokens
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in, h_out),
            torch.nn.GELU(),
            torch.nn.Linear(h_out, h_out * v_tokens)
        )
        self.pos = torch.nn.Parameter(torch.zeros(v_tokens, h_out))
    def forward(self, x):
        y = self.mlp(x)                           # [B, V*H]
        y = y.view(x.size(0), self.v_tokens, -1)  # [B, V, H]
        return y + self.pos.unsqueeze(0)

ckpt_proj = torch.load(CKPT_PT, map_location="cpu")
V_TOKENS  = ckpt_proj["V"]
proj = Projector(ckpt_proj["D"], ckpt_proj["H"], V_TOKENS).to(device)
proj.load_state_dict(ckpt_proj["proj"])
proj.eval()
print("Projector restored from", CKPT_PT)

# --- demo on a random validation sample (if you saved val_idx separately you can use it; else random all) ---
i = random.randrange(N) # i = 2 for 9800
demo_emb = img_embs[i].unsqueeze(0).to(device, dtype=torch.float32)
prompt = "A pathology image of the eye is presented. As a senior ophthalmic pathologist, provide a concise diagnostic description that identifies the disease, specifies its subclass and broader classification, and essential histologic findings.\n"

with torch.no_grad():
    v = proj(demo_emb)  # [1, V, H]
    llm_embed = model.get_input_embeddings()
    llm_dtype = llm_embed.weight.dtype
    v = v.to(llm_dtype)
    ids = tok(prompt, return_tensors="pt").to(device)
    t_emb = model.get_input_embeddings()(ids.input_ids)
    inp   = torch.cat([v, t_emb], dim=1)
    attn  = torch.cat([torch.ones(1, V_TOKENS, device=device, dtype=torch.long), ids.attention_mask], dim=1)
    # -
    gen = model.generate(
        inputs_embeds=inp,
        attention_mask=attn,
        max_new_tokens=96,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tok.eos_token_id
    )

decoded = tok.decode(gen[0], skip_special_tokens=True)
output_text = decoded.split(prompt, 1)[-1].strip()

print("=== demo index:", i, "/ number:", meta["number"][i], " ===")
print("\n--- Ground truth ---\n", gt[i])
print("\n--- Model output ---\n", output_text)


# --- write ref .json --- (draft)
import json
import os
def write_json(data, filename):
    """Utility function to write a dictionary/list to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Successfully created {filename}")
    
gts = {}
val_idx = [2, 38, 39, 44, 46, 77, 83, 89, 111, 114, 123, 130, 133, 147, 148, 172, 173, 176, 198, 205, 215]
for idx in val_idx:
    gts[idx] = [ {"caption": gt[idx]} ]
write_json(gts, f"{os.path.expandvars(STAGE_DIR)}/gts.json")

# --- write model .json---
RES = f"{os.path.expandvars(STAGE_DIR)}/res_1016.json"
with open(RES, 'r') as f:
    res = json.load(f)
idx = 2
entry = {
    "number": idx,
    "model": CKPT_PT,
    "caption": output_text
}
res.append(entry)
with open(RES, 'w') as f:
    json.dump(res, f, indent=4)
