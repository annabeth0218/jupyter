# notes
# fitz crash when TOC are included e.g. "........"
# manually removed TOC from .pdf in advance

import fitz  # PyMuPDF
import pymupdf4llm
import os
import json
import re

# === CONFIG ===
pdf_path = "/Users/annabethlu/Projects/OPathChat/textbook/WHO4e/Eye_WHO_Classification_of_Tumours_8.pdf"  # replace me
out_dir = "/Users/annabethlu/Projects/OPathChat/textbook/WHO4e/text"  # replace me
os.makedirs(out_dir, exist_ok=True)

# === KNOWN SECTION BREAKS (manual map) === # require manually modifying!!
known_sections = {
    "Conjunctival stromal tumour": "Fibroblastic and myofibroblastic tumours",
    "Haemangiomas of the conjunctiva, uveal tract and retina": "Vascular Tumours",
    "Leiomyoma of the ciliary body": "Smooth-muscle tumours",
    "Rhabdomyosarcoma of the conjunctiva and carcuncle": "Skeletal muscle tumours",
    "Neurofibroma and ganglioneuroma": "Peripheral nerve sheath tumours",
    "Osteoma": "Tumours of uncertain derivation"
}

# === TEXT EXTRACTION ===
doc = fitz.open(pdf_path)
lines = []
for page in doc:
    page_text = page.get_text("text")
    lines.extend(page_text.splitlines())

# with open("debug_output.md", "w", encoding="utf-8") as f: # for debug
#      for line in lines:
#          f.write(line.strip() + "\n")

# === PARSING LOGIC ===
section = None
disease = None
disease_data = {}
references = []
current_subtitle = None
buffer = []

def flush_disease():
    if disease:
        out = {
            "section": section,
            "disease": disease,
            "content": disease_data.copy(),
            "references": sorted(set(references))
        }
        slug = re.sub(r"\W+", "_", disease.lower()).strip("_")
        out_path = os.path.join(out_dir, f"{slug}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"✔︎ Saved {out_path}")
    else:
        print("⚠︎ Skipped flush — no disease name found.")

def clean_text(text, subtitle = None):
    # (1) Fix line breaks inside references
    text = re.sub(r"\{([^{}]*)\}", lambda m: "{" + m.group(1).replace("\n", " ") + "}", text)

    # (2) Fix line breaks before 'Not recommended: '
    text = re.sub(r"\n(?=Not recommended:)", ". ", text)

    # (3) Fix line breaks around known headers
    special_headers = ["Genetic profile", "Genetic susceptibility", "Differential diagnosis"]
    for header in special_headers:
        pattern = rf"\n({header})\n"
        text = re.sub(pattern, rf" \1: ", text)

    # (4) ICD-O coding: use semicolon for line breaks
    if subtitle and subtitle.lower().startswith("icd-o coding"):
        text = text.replace("\n", "; ")

    else:
        # (5) all other \n become space
        text = re.sub(r"\s*\n\s*", " ", text)
    
    # (6) Remove image captions in 'Prognosis and prediction'
    if subtitle and subtitle.lower() == "prognosis and prediction":
        text = re.split(r"#\d+", text)[0].strip()

    return text

previous_line = None

for idx, line in enumerate(lines):
    line = line.strip()

    # skip empty or captions
    if not line or re.match(r"^\d{4}\b", line):
        continue

    # subtitles
    if line.endswith(":-"):
        # store last subtitle
        if current_subtitle and buffer:
            joined = "\n".join(buffer).strip()
            disease_data[current_subtitle] = clean_text(joined, subtitle=current_subtitle)
            buffer = []

        current_subtitle = line[:-2].strip()

        # identify disease
        if current_subtitle.lower() == "definition":
            if disease:
                flush_disease()

            disease = previous_line.strip() if previous_line else None
            section = known_sections.get(disease, None)
            disease_data = {}
            references = []
        
        continue

    # get descriptions under subtitle
    if current_subtitle:
        buffer.append(line)
        refs = re.findall(r"\{([^}]+)\}", line) # refs = re.findall(r"\{(\d+)\}", line)
        for ref in refs:
            references.extend(re.split(r"\s*;\s*", ref.strip()))
        references.extend(refs)

    previous_line = line

# final flush
if current_subtitle and buffer:
    joined = "\n".join(buffer).strip()
    disease_data[current_subtitle] = clean_text(joined, subtitle=current_subtitle)

flush_disease()


### fill in spreadsheet

import os
import json
import re
import pandas as pd

folder_path = "/Users/annabethlu/Projects/OPathChat/textbook/WHO4e/text"
rows = []

for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".json"):
            try:
                full_path = os.path.join(root, filename)
                folder_number = os.path.basename(os.path.dirname(full_path))  # e.g. "1", "2", ..., "8"
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    disease_name = os.path.splitext(filename)[0]
                    coding = data.get("content", {}).get("ICD-O coding")

                    if not coding or "Code according to" in str(coding):
                        rows.append({
                            "folder": folder_number,
                            "disease (WHO)": disease_name.lower().replace("_", " "),
                            "ICD-O": None,
                            "disease (ICD-O)": None
                        })
                    else:
                        entries = [e.strip() for e in coding.split(";") if e.strip()]
                        for entry in entries:
                            match = re.match(r"(.+?)\s+(\d{4}/\d)", entry)
                            if match:
                                desc, code = match.groups()
                                rows.append({
                                    "folder": folder_number,
                                    "disease (WHO)": disease_name.lower().replace("_", " "),
                                    "ICD-O": code,
                                    "disease (ICD-O)": desc.strip().lower().replace("_", " ") if desc else None
                                })
                            else:
                                rows.append({
                                    "folder": folder_number,
                                    "disease (WHO)": disease_name.lower().replace("_", " "),
                                    "ICD-O": None,
                                    "disease (ICD-O)": None
                                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")

df = pd.DataFrame(rows)
df.to_csv("/Users/annabethlu/Projects/OPathChat/textbook/WHO4e/disease_list.csv", index=False)
