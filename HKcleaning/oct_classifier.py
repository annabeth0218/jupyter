import easyocr
import re

#region local run
# Path to your image
image_path = "/Volumes/SU720/oph-brain/oph_image/H401230_278P/OCT/389665-9_20170411_OCT_00000439.bmp"  

# Initialize OCR
reader = easyocr.Reader(['en'])
ocr_results = reader.readtext(image_path, detail=0)
full_text = " ".join(ocr_results)

# Keywords
keywords = [
    "ONH and RNFL", "Ganglion Cell", "HD Cross", "Macula", "Retina Map",
    "Cross Line", "HD 5 Line Raster", "GPA", "Angio", "Full Retina Thickness Map",
    "RNFL Analysis", "Radial Lines", "3D Wide(H)", "3D Wide ", 
    "Significance Map", "Fundus Image"
]
keyword_hits = [kw for kw in keywords if re.search(re.escape(kw), full_text, re.IGNORECASE)]

# Extract gender by matching "male" or "female"
gender_match = re.search(r"\b\w*ale\w*\b", full_text, re.IGNORECASE)
gender = gender_match.group(0) if gender_match else None

# Extract DOB by finding substring containing "/19"
dob_match = re.search(r"\b\S*/19\S*\b", full_text)
dob = dob_match.group(0) if dob_match else None

print("Text:", full_text[500:])
print("Keyword hits:", keyword_hits)
print("Gender:", gender)
print("DOB:", dob)

#endregion

# tmux new -s gad
# conda activate gad-env
# tmux attach -t gad
# if forgot: tmux ls, tmux attach -t <session_name>
import easyocr
import re
import os
import csv
import pandas as pd
from datetime import datetime
from collections import defaultdict
import shutil
from pathlib import Path

#region oct
img_dir = "oph_image/H409_1258P/OCT"
output_csv = "NTG_oct.csv"

reader = easyocr.Reader(['en'], gpu=True)

keywords = [
    "ONH and RNFL", "Ganglion Cell", "HD Cross", "Retina Map",
    "Cross Line", "HD 5 Line Raster", "GPA", "Angio", "Full Retina Thickness Map",
    "RNFL Analysis", "Radial Lines", "3D Wide(H)", "3D Wide ", "Macula Thickness",
    "Significance Map", "Fundus Image"
]

flag = 0
sep = "__"
base_path = Path(img_dir)

with open(output_csv, mode="a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "num_keyword_hits", "keyword_hits", "gender", "dob"])

    for fname in os.listdir(img_dir):
        if flag == 5:
            break
        flag += 1

        if fname.lower().endswith(".bmp"):
            path = base_path / fname
            ocr_results = reader.readtext(str(path), detail=0)
            full_text = " ".join(ocr_results)

            # Keyword hits
            hits = [kw for kw in keywords if re.search(re.escape(kw), full_text, re.IGNORECASE)]

            # Gender
            match = re.search(r"\b\w*ale\w*\b", full_text, re.IGNORECASE)
            gender = match.group(0) if match else None

            # DOB
            dob_match = re.search(r"\b\S*/19\S*\b", full_text)
            dob = dob_match.group(0) if dob_match else None

            writer.writerow([fname, len(hits), "; ".join(hits), gender, dob])

            # Decide folder name
            if not hits:
                target_dir = base_path / "no_keyword"
            elif len(hits) >= 3:
                target_dir = base_path / "multi_keyword"
            else:
                sorted_hits = sorted(hits, key=str.lower)
                folder_name = sep.join(k.replace(" ", "_") for k in sorted_hits)
                target_dir = base_path / folder_name

            target_dir.mkdir(exist_ok=True)
            shutil.move(str(path), target_dir / fname)

#endregion

