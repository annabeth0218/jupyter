# last modified 1112 20:00

import os, json
import nltk
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm

# change here!
val_idx = [2, 38, 39, 44, 46, 77, 83, 89, 111, 114, 123, 
           130, 133, 147, 148, 172, 173, 176, 198, 205, 215]
STAGE_DIR = "./CONCH/who4e/for_demo"
RES = f"{os.path.expandvars(STAGE_DIR)}/res_modelv1016-3_1112.json"
GTS = f"{os.path.expandvars(STAGE_DIR)}/gts.json"
OUTPUT = f"{os.path.expandvars(STAGE_DIR)}/score_modelv1016-3_1112.json"

# METEOR requires refs, download from NLTK package
print("Downloading NLTK data for METEOR...")
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)  # for New version (NLTK 3.8+)
    except:
        nltk.download('punkt', quiet=True)  # Fallback for older versions
    print("✅ NLTK data downloaded successfully.\n")
except Exception as e:
    print(f"⚠️  Warning: Could not download NLTK data: {e}")
    print("METEOR scores may not work correctly.\n")

# The pycocoevalcap scorers expect two dictionaries:
# gts (Ground Truths): {image_id: [{'caption': str}, ...]}
# res (Results): {image_id: [{'caption': str}]}

# Load data
with open(RES, 'r') as f:
    res_list = json.load(f)  # List format: [{"number": ..., "model": ..., "caption": ...}, ...]
with open(GTS, 'r') as f:
    gts_dict_raw = json.load(f)  # Dict format: {idx (str or int): [{"caption": ...}]}

# Normalize gts_dict keys to handle both string and int keys
gts_dict = {}
for k, v in gts_dict_raw.items():
    # Convert key to int if possible, otherwise keep as string
    try:
        key = int(k)
    except (ValueError, TypeError):
        key = k
    gts_dict[key] = v

# Initialize scorers
scorers = {
    'BLEU1': Bleu(4),  # BLEU scorer supports multiple n-grams
    'BLEU4': Bleu(4),
    'ROUGE': Rouge(),
    'CIDEr-R': Cider()
    # METEOR will be computed using NLTK
}

# Process each sample
results = []

# Collect all data for batch CIDEr computation (CIDEr needs corpus-level statistics)
all_gts_for_cider = {}
all_res_for_cider = {}

print(f"Computing scores for {len(res_list)} samples...")
for idx, res_entry in enumerate(tqdm(res_list, desc="Scoring")):
    
    sample_idx = val_idx[idx] if idx < len(val_idx) else None
    
    # Get ground truth for this sample
    if sample_idx is None or sample_idx not in gts_dict:
        print(f"Warning: No ground truth found for index {idx}, skipping...")
        continue
    
    # Prepare data for pycocoevalcap (single sample format)
    image_id = str(sample_idx)
    
    # Extract caption from res_entry - ensure it's a string
    res_caption = res_entry.get('caption', '')
    if isinstance(res_caption, dict):
        # If caption is a dict, try to extract text from common keys
        res_caption = res_caption.get('text', res_caption.get('caption', ''))
    # Ensure it's a string
    res_caption = str(res_caption) if res_caption else ''
    
    # Extract caption from gts - ensure it's a list of dicts with 'caption' key
    gts_list = gts_dict[sample_idx]
    if not isinstance(gts_list, list):
        gts_list = [gts_list]
    
    gts_captions = []
    for gt_item in gts_list:
        if isinstance(gt_item, dict):
            # Extract caption from dict
            gt_caption = gt_item.get('caption', gt_item.get('text', ''))
            # If caption itself is a dict, extract further
            if isinstance(gt_caption, dict):
                gt_caption = gt_caption.get('text', gt_caption.get('caption', ''))
        else:
            # If not a dict, convert to string
            gt_caption = str(gt_item)
        # Ensure it's a string
        gt_caption = str(gt_caption) if gt_caption else ''
        # Append as dict with 'caption' key
        gts_captions.append({'caption': gt_caption})
    
    # Ensure we have at least one caption
    if not gts_captions:
        print(f"Warning: No valid captions found for idx {sample_idx}, skipping...")
        continue
    
    # Prepare final format for pycocoevalcap
    # pycocoevalcap expects: {image_id: [str, str, ...]} not {image_id: [{'caption': str}]}
    gts_strings = [item['caption'] for item in gts_captions]  # Extract strings from dicts
    gts_sample = {image_id: gts_strings}  # {image_id: [str, ...]}
    res_sample = {image_id: [res_caption]}  # {image_id: [str]}
    
    # Collect data for batch CIDEr computation
    all_gts_for_cider[image_id] = gts_strings
    all_res_for_cider[image_id] = [res_caption]
    
    # Debug: verify format (only for first sample)
    if idx == 0:
        print(f"\nDebug - Sample {sample_idx}:")
        print(f"  res_sample format: {type(res_sample[image_id])}, item type: {type(res_sample[image_id][0])}")
        print(f"  gts_sample format: {type(gts_sample[image_id])}, item type: {type(gts_sample[image_id][0])}")
    
    # Compute scores for this sample (except METEOR, which will be computed in batch)
    scores = {}
    
    try:
        # BLEU scores (returns list [Bleu_1, Bleu_2, Bleu_3, Bleu_4] or dict)
        bleu_scores, _ = scorers['BLEU1'].compute_score(gts_sample, res_sample)
        if isinstance(bleu_scores, (list, tuple)):
            scores['BLEU1'] = float(bleu_scores[0])  # Bleu_1
            scores['BLEU4'] = float(bleu_scores[3])  # Bleu_4
        elif isinstance(bleu_scores, dict):
            scores['BLEU1'] = float(bleu_scores.get('Bleu_1', 0.0))
            scores['BLEU4'] = float(bleu_scores.get('Bleu_4', 0.0))
        else:
            scores['BLEU1'] = float(bleu_scores)
            scores['BLEU4'] = float(bleu_scores)
    except Exception as e:
        import traceback
        print(f"Error computing BLEU for idx {sample_idx}: {e}")
        if idx == 0:  # Print full traceback for first error
            traceback.print_exc()
        scores['BLEU1'] = 0.0
        scores['BLEU4'] = 0.0
    
    try:
        # ROUGE score
        rouge_scores, _ = scorers['ROUGE'].compute_score(gts_sample, res_sample)
        if isinstance(rouge_scores, dict):
            scores['ROUGE'] = float(rouge_scores.get('ROUGE-L', 0.0))
        else:
            scores['ROUGE'] = float(rouge_scores)
    except Exception as e:
        import traceback
        print(f"Error computing ROUGE for idx {sample_idx}: {e}")
        if idx == 0:  # Print full traceback for first error
            traceback.print_exc()
        scores['ROUGE'] = 0.0
    
    # CIDEr-R will be computed in batch after the loop (needs corpus-level statistics)
    scores['CIDEr-R'] = None  # Placeholder
    
    # Compute METEOR using NLTK
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        
        # METEOR needs tokenized input:
        # - reference: list of lists of tokens (each reference is a list of words)
        # - hypothesis: list of tokens (single list of words)
        reference_tokenized = [word_tokenize(ref) for ref in gts_strings]  # List of tokenized ref
        hypothesis_tokenized = word_tokenize(res_caption)  # Tokenized hypothesis
        
        # Calculate METEOR score
        meteor_score_val = meteor_score(reference_tokenized, hypothesis_tokenized)
        scores['METEOR'] = float(meteor_score_val)
    except Exception as e:
        import traceback
        print(f"Error computing METEOR for idx {sample_idx}: {e}")
        if idx == 0:  # Print full traceback for first error
            traceback.print_exc()
        scores['METEOR'] = 0.0
    
    # Store result
    result_entry = {
        "idx": sample_idx,
        "number": res_entry.get("number", sample_idx),
        "res": res_caption,  # Use processed caption string
        "gts": gts_captions[0]['caption'] if gts_captions else '',  # Get first ground truth caption
        "score": scores
    }
    results.append(result_entry)

# Batch compute CIDEr scores for all samples (CIDEr needs corpus-level TF-IDF statistics)
print("\nComputing CIDEr scores in batch (requires corpus-level statistics)...")
try:
    cider_scores_all, cider_scores_dict = scorers['CIDEr-R'].compute_score(all_gts_for_cider, all_res_for_cider)
    
    # cider_scores_dict is a dict: {image_id: score}
    if isinstance(cider_scores_dict, dict):
        # Assign CIDEr scores back to results
        for result in results:
            image_id = str(result['idx'])
            if image_id in cider_scores_dict:
                result['score']['CIDEr-R'] = float(cider_scores_dict[image_id])
            else:
                result['score']['CIDEr-R'] = 0.0
    elif isinstance(cider_scores_all, (list, tuple)):
        # If it's a list, assign in order
        for i, result in enumerate(results):
            if i < len(cider_scores_all):
                result['score']['CIDEr-R'] = float(cider_scores_all[i])
            else:
                result['score']['CIDEr-R'] = 0.0
    else:
        # If it's a single value, assign to all
        cider_val = float(cider_scores_all) if cider_scores_all else 0.0
        for result in results:
            result['score']['CIDEr-R'] = cider_val
    
    print("CIDEr scores computed successfully.")
except Exception as e:
    import traceback
    print(f"Error computing CIDEr in batch: {e}")
    if len(results) > 0:
        traceback.print_exc()
    # Set CIDEr to 0.0 for all samples if batch computation fails
    for result in results:
        result['score']['CIDEr-R'] = 0.0

# Save results
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nSaved scores for {len(results)} samples to {OUTPUT}")

# Print summary statistics
if results:
    print("\n=== Score Summary ===")
    for metric in ['BLEU1', 'BLEU4', 'ROUGE', 'CIDEr-R', 'METEOR']:
        values = [r['score'][metric] for r in results if r['score'].get(metric) is not None]
        if values:
            print(f"{metric:10s}: Mean={sum(values)/len(values):.4f}, Min={min(values):.4f}, Max={max(values):.4f}")
        else:
            print(f"{metric:10s}: No valid scores")
