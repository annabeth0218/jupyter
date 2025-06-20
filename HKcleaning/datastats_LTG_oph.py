import os
import csv
import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

mom_dir = '/Volumes/SU720/oph-brain/oph_image/H401230_278P'  # change

#region files under each folder count
folder_names = []
file_counts = []

for folder in os.listdir(mom_dir):
    folder_path = os.path.join(mom_dir, folder)
    if os.path.isdir(folder_path):
        count = sum(os.path.isfile(os.path.join(folder_path, f)) for f in os.listdir(folder_path))
        folder_names.append(folder)
        file_counts.append(count)

plt.figure(figsize=(10, 6))
bars = plt.bar(folder_names, file_counts)
plt.xlabel('Folder Name')
plt.ylabel('# of Files')
plt.title('File Counts per Folder')
plt.xticks(rotation=45, ha='right')

for bar, count in zip(bars, file_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
#endregion

#region patient ID vs. available data

out_csv = '/Users/annabethlu/Library/Mobile Documents/com~apple~CloudDocs/lab/25 HK CityU/LTG_oph_summary.csv'

def correct_minguo(fname, tag_path):
    parts = os.path.splitext(fname)[0].split('_')
    if len(parts) < 2:
        return fname  # skip if format is broken

    date_part = parts[1]

    # Handle Minguo format like '1020124-1'
    m = re.match(r"^(\d{3})(\d{4})-1$", date_part)
    if m:
        yyy = int(m.group(1))
        mmdd = m.group(2)
        yyyy = yyy + 1911
        corrected_date = f"{yyyy}{mmdd}"
        parts[1] = corrected_date
        new_fname = '_'.join(parts) + '.bmp'
        os.rename(os.path.join(tag_path, fname), os.path.join(tag_path, new_fname))
        print(f"Corrected Minguo date in: {fname} → {new_fname}")
        return new_fname

    # Handle malformed format like '202402-12' → '20240212'
    m = re.match(r"^(\d{6})-(\d{2})$", date_part)
    if m:
        corrected_date = m.group(1) + m.group(2)
        parts[1] = corrected_date
        new_fname = '_'.join(parts) + '.bmp'
        os.rename(os.path.join(tag_path, fname), os.path.join(tag_path, new_fname))
        print(f"Corrected malformed date in: {fname} → {new_fname}")
        return new_fname

    # Warn if year appears < 2000 based on first 4 digits
    if len(date_part) >= 4 and date_part[:4].isdigit():
        year_guess = int(date_part[:4])
        if year_guess < 2000:
            print(f"⚠️  Alert: {fname} has suspicious early year → {year_guess}")

    return fname


rows = []

for tag_folder in os.listdir(mom_dir):

    tag_path = os.path.join(mom_dir, tag_folder)
    if not os.path.isdir(tag_path):
        continue
    for fname in os.listdir(tag_path):
        if not fname.endswith('.bmp') or fname.startswith('._'):
            continue
        try:
            # print(f"{fname} → {parts}")
            fname = correct_minguo(fname, tag_path)
            parts = os.path.splitext(fname)[0].split('_')
            patient_id = parts[0]
            date_str = parts[1]
            date_fmt = datetime.strptime(date_str, "%Y%m%d")
            date_val = f"{date_fmt.year}/{date_fmt.month}/{date_fmt.day}"
            ori_tag = parts[2] if len(parts) > 2 and parts[2] else None
            rows.append([
                patient_id,
                fname,
                date_val,
                ori_tag,
                tag_folder
            ])
        except Exception as e:
            print(f"Skipping file {fname}: {e}")

# Write to CSV
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['patient_id', 'file_name', 'date', 'ori_tag', 'tag'])
    writer.writerows(rows)

#endregion

#region bypatient.csv

ct_csv = '/Users/annabethlu/Library/Mobile Documents/com~apple~CloudDocs/lab/25 HK CityU/LTG_ct_series_summary.csv'
oph_csv = '/Users/annabethlu/Library/Mobile Documents/com~apple~CloudDocs/lab/25 HK CityU/LTG_oph_summary.csv'
out_csv = '/Users/annabethlu/Library/Mobile Documents/com~apple~CloudDocs/lab/25 HK CityU/LTG_bypatient.csv'

ct_df = pd.read_csv(ct_csv)
oph_df = pd.read_csv(oph_csv) # under /mnt/data for ChatGPT

# Helper function to normalize date format from YYYY/M/D to YYYYMMDD
def normalize_ymd(date_str):
    try:
        parts = date_str.strip().split('/')
        if len(parts) != 3:
            return None
        y, m, d = parts
        return f"{int(y):04d}{int(m):02d}{int(d):02d}"
    except:
        return None

# Normalize CT dates
ct_info = {}
for _, row in ct_df.iterrows():
    pid = row['patient_id']
    ct_info[pid] = {
        'sex': row.get('sex', None),
        'age': row.get('age', None),
        'ct': 1,
        'num_studies': row.get('num_studies', None),
        'ct_dates': []
    }
    if pd.notna(row.get('date')):
        norm_date = normalize_ymd(row['date'])
        if norm_date:
            ct_info[pid]['ct_dates'].append(norm_date)

# Group oph data
patient_oph = defaultdict(lambda: {
    'tags': defaultdict(int),
    'tag_dates': defaultdict(set)
})
all_tags = set()

for _, row in oph_df.iterrows():
    pid = row['patient_id']
    tag = row['tag']
    norm_date = normalize_ymd(row['date'])
    if not norm_date:
        continue
    all_tags.add(tag)
    patient_oph[pid]['tags'][tag] += 1
    patient_oph[pid]['tag_dates'][tag].add(norm_date)

# Build the combined data
all_tags = sorted(all_tags)
header = ['patient_id', 'sex', 'age', 'ct', 'num_studies', 'ct_dates', 'oph_total']
for tag in all_tags:
    header += [tag, f'{tag}_dates']

data = []
all_patient_ids = sorted(set(ct_info) | set(patient_oph))

for pid in all_patient_ids:
    c = ct_info.get(pid, {})
    o = patient_oph.get(pid, {})

    sex = c.get('sex')
    age = c.get('age')
    ct = c.get('ct', 0)
    num_studies = c.get('num_studies')
    ct_dates = '|'.join(sorted(c.get('ct_dates', []))) if ct else None

    tag_counts = o.get('tags', {})
    tag_dates = o.get('tag_dates', {})
    oph_total = sum(tag_counts.values()) if tag_counts else 0

    row = [pid, sex, age, ct, num_studies, ct_dates, oph_total]
    for tag in all_tags:
        row.append(tag_counts.get(tag, 0) or 0)
        dates = tag_dates.get(tag)
        row.append('|'.join(sorted(dates)) if dates else None)
    data.append(row)

# Create dataframe and display
final_df = pd.DataFrame(data, columns=header)
# ChatGPT run import ace_tools as tools; tools.display_dataframe_to_user(name="LTG_bypatient.csv", dataframe=final_df)

#endregion

#region fetch and rename spurious files manually
import os
import re

log_file = '/path/to/your/log.txt'  # change

# === PARSE LOG ===
pattern = re.compile(r'(?:Alert: |Skipping file )([\w\-]+_\w+_OCT_\d+\.bmp)')
suspect_files = set()

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            suspect_files.add(match.group(1))

# === SEARCH AND RENAME ===
for filename in suspect_files:
    for root, dirs, files in os.walk(mom_dir):
        if filename in files:
            full_path = os.path.join(root, filename)
            print(f"Found: {full_path}")
            new_date = input(f"Type new date: ")
            parts = filename.split('_')
            if len(parts) >= 3:
                parts[1] = new_date
                new_filename = '_'.join(parts)
                new_path = os.path.join(root, new_filename)
                os.rename(full_path, new_path)
                print(f"Renamed to: {new_filename}\n")
            break
    else:
        print(f"Not found: {filename}\n")

#endregion
