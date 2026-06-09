"""
scoring_from_predictions.py

Compute BLEU-1, BLEU-4, ROUGE-L, CIDEr, and METEOR from a predictions.json
produced by eval.py.  The file format is:

    [
      {
        "idx":       0,
        "id":        "...",
        "image":     "...",
        "model":     "...",
        "caption":   "<model output>",
        "reference": "<ground truth>"   ← optional; samples without it are skipped
      },
      ...
    ]

Usage:
    python ../src/scoring.py \
        --input  ../outputs/ho82b/S321.json \
        --output ../outputs/ho82b/S321.json
"""

import argparse
import json
import os

import nltk
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NLTK data for METEOR
# ---------------------------------------------------------------------------

print("Downloading NLTK data for METEOR...")
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)   # NLTK 3.8+
    except Exception:
        nltk.download("punkt", quiet=True)
    print("✅ NLTK data downloaded successfully.\n")
except Exception as e:
    print(f"⚠️  Warning: Could not download NLTK data: {e}")
    print("METEOR scores may not work correctly.\n")

# ---------------------------------------------------------------------------
# Scorers (shared across all samples)
# ---------------------------------------------------------------------------

_bleu_scorer  = Bleu(4)
_rouge_scorer = Rouge()
_cider_scorer = Cider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(value) -> str:
    """Ensure a value is a plain non-empty string."""
    if isinstance(value, dict):
        value = value.get("text", value.get("caption", ""))
    return str(value).strip() if value else ""


def _compute_meteor(hypothesis: str, references: list[str]) -> float:
    try:
        from nltk.tokenize import word_tokenize
        from nltk.translate.meteor_score import meteor_score

        ref_tok = [word_tokenize(r) for r in references]
        hyp_tok = word_tokenize(hypothesis)
        return float(meteor_score(ref_tok, hyp_tok))
    except Exception as e:
        print(f"  METEOR error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Main scoring logic
# ---------------------------------------------------------------------------

def score_predictions(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Filter out samples that have no reference caption
    valid = [p for p in predictions if _safe_str(p.get("reference", ""))]
    skipped = len(predictions) - len(valid)
    if skipped:
        print(f"⚠️  Skipped {skipped} sample(s) with no 'reference' field.")
    print(f"Scoring {len(valid)} samples...\n")

    results = []
    all_gts: dict[str, list[str]] = {}
    all_res: dict[str, list[str]] = {}

    for entry in tqdm(valid, desc="Scoring"):
        image_id   = str(entry.get("idx", entry.get("id", id(entry))))
        hypothesis = _safe_str(entry.get("caption", ""))
        reference  = _safe_str(entry.get("reference", ""))

        gts_sample = {image_id: [reference]}
        res_sample = {image_id: [hypothesis]}

        # Accumulate for batch CIDEr
        all_gts[image_id] = [reference]
        all_res[image_id] = [hypothesis]

        scores: dict[str, float | None] = {}

        # ---- BLEU ----
        try:
            bleu_vals, _ = _bleu_scorer.compute_score(gts_sample, res_sample)
            scores["BLEU1"] = float(bleu_vals[0])
            scores["BLEU4"] = float(bleu_vals[3])
        except Exception as e:
            print(f"  BLEU error (idx={image_id}): {e}")
            scores["BLEU1"] = 0.0
            scores["BLEU4"] = 0.0

        # ---- ROUGE-L ----
        try:
            rouge_val, _ = _rouge_scorer.compute_score(gts_sample, res_sample)
            scores["ROUGE"] = float(rouge_val) if not isinstance(rouge_val, dict) \
                              else float(rouge_val.get("ROUGE-L", 0.0))
        except Exception as e:
            print(f"  ROUGE error (idx={image_id}): {e}")
            scores["ROUGE"] = 0.0

        # ---- METEOR ----
        scores["METEOR"] = _compute_meteor(hypothesis, [reference])

        # CIDEr placeholder — filled in after the loop
        scores["CIDEr-R"] = None

        results.append({
            "idx":    entry.get("idx", ""),
            "id":     entry.get("id", ""),
            "image":  entry.get("image", ""),
            "model":  entry.get("model", ""),
            "res":    hypothesis,
            "gts":    reference,
            **scores,
        })

    # ---- CIDEr (batch, needs corpus-level TF-IDF) ----
    print("\nComputing CIDEr in batch...")
    try:
        cider_mean, cider_per_image = _cider_scorer.compute_score(all_gts, all_res)

        if isinstance(cider_per_image, dict):
            for r in results:
                iid = str(r["idx"]) if r["idx"] != "" else str(r["id"])
                r["CIDEr-R"] = float(cider_per_image.get(iid, 0.0))
        elif hasattr(cider_per_image, "__len__"):
            for i, r in enumerate(results):
                r["CIDEr-R"] = float(cider_per_image[i]) if i < len(cider_per_image) else 0.0
        else:
            cval = float(cider_mean) if cider_mean else 0.0
            for r in results:
                r["CIDEr-R"] = cval

        print("CIDEr computed successfully.")
    except Exception as e:
        import traceback
        print(f"CIDEr batch error: {e}")
        traceback.print_exc()
        for r in results:
            r["CIDEr-R"] = 0.0

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nSaved scores for {len(results)} samples → {output_path}")

    # ---- Summary ----
    print("\n=== Score Summary ===")
    for metric in ["BLEU1", "BLEU4", "ROUGE", "CIDEr-R", "METEOR"]:
        vals = [r[metric] for r in results if r.get(metric) is not None]
        if vals:
            print(f"{metric:10s}: Mean={sum(vals)/len(vals):.4f}  "
                  f"Min={min(vals):.4f}  Max={max(vals):.4f}")
        else:
            print(f"{metric:10s}: No valid scores")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score predictions.json with BLEU/ROUGE/CIDEr/METEOR.")
    p.add_argument("--input",  "-i", required=True, help="Path to predictions.json from eval.py.")
    p.add_argument("--output", "-o", required=True, help="Path for the output scores JSON file.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    score_predictions(args.input, args.output)