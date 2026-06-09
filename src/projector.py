from __future__ import annotations
from typing import Any, Dict, List, Sequence
import torch, os
from pathlib import Path

PROMPT_FILE = Path(os.environ.get("PROMPT_FILE", "../prompts/base.txt"))
PROMPT = PROMPT_FILE.read_text(encoding="utf-8")
DEFAULT_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")


class Projector(torch.nn.Module):
    def __init__(self, d_in: int, h_out: int, v_tokens: int = 4, dropout: float = 0.1):
        super().__init__()
        self.v_tokens = v_tokens
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in, h_out),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h_out, h_out * v_tokens),
        )
        self.pos = torch.nn.Parameter(torch.zeros(v_tokens, h_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)
        y = y.view(x.size(0), self.v_tokens, -1)
        return y + self.pos.unsqueeze(0)


def low_st(text: str) -> str:
    return text[:1].lower() + text[1:] if text else ""


def format_reference_caption(meta: Dict[str, Sequence[Any]], index: int) -> str:
    title = _meta_value(meta, "title", index)
    disease = _meta_value(meta, "disease", index)
    subcls = _meta_value(meta, "subcls", index)
    cls = _meta_value(meta, "cls", index)
    caption = _meta_value(meta, "captions", index) or _meta_value(meta, "caption", index)
    diagnosis = f"Pathologic diagnosis: {disease}" if disease else "Pathologic diagnosis:"
    if subcls or cls:
        diagnosis += f", classified as {low_st(subcls)} within the broader category of {low_st(cls)}"
    finding_prefix = f"{title} is shown. " if title and disease.lower() not in title.lower() else ""
    return f"{diagnosis}.\nMicroscopic findings: {finding_prefix}{caption}".strip()


def generate_from_embedding(
    *,
    image_embedding: torch.Tensor,
    projector: Projector,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
) -> str:
    device = next(model.parameters()).device
    projector.eval()
    model.eval()

    with torch.no_grad():
        emb = image_embedding.unsqueeze(0).to(device=device, dtype=torch.float32)
        visual_tokens = projector(emb)
        llm_embed = model.get_input_embeddings()
        visual_tokens = visual_tokens.to(llm_embed.weight.dtype)
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        text_tokens = llm_embed(ids.input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(1, projector.v_tokens, device=device, dtype=torch.long),
                ids.attention_mask,
            ],
            dim=1,
        )
        generated = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    return decoded.split(prompt, 1)[-1].strip()


def load_projector(checkpoint_path: str, device: torch.device | str) -> Projector:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    projector = Projector(checkpoint["D"], checkpoint["H"], checkpoint["V"]).to(device)
    projector.load_state_dict(checkpoint["proj"])
    projector.eval()
    return projector


def _meta_value(meta: Dict[str, Sequence[Any]], key: str, index: int) -> str:
    values = meta.get(key, [])
    if index >= len(values):
        return ""
    value = values[index]
    return "" if value is None else str(value)

