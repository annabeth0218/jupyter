import os
import pandas as pd
from datetime import datetime
import pydicom

root_dir = "/Volumes/SU720/oph-brain/CT_brain_image/H401230_CT/H401230_9_10_(CT_Scan_brain)"
rows = []
flag = 0

for patient_id in os.listdir(root_dir):

    # if flag >= 5:
    #     break
    patient_path = os.path.join(root_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    studies = [s for s in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, s))]
    num_studies = len(studies)

    for study_name in studies:
        study_path = os.path.join(patient_path, study_name)
        series_list = [se for se in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, se))]

        for series_name in series_list:
            flag += 1
            # if flag >= 5:
            #     break
            series_path = os.path.join(study_path, series_name)
            dcm_files = [f for f in os.listdir(series_path) if f.lower().endswith('.dcm')]
            dcm_count = len(dcm_files)

            main = 1 if dcm_count > 3 else 0
            count_col = dcm_count if main else None

            try:
                dcm_path = os.path.join(series_path, dcm_files[0])
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                date_str = ds.get("AcquisitionDate", None)
                if date_str:
                    date_col = datetime.strptime(date_str, "%Y%m%d").strftime("%Y/%m/%d")
            except Exception:
                print(f"[{series_name}] Error reading AcquisitionDate")
                date_col = None

            sex_val = None
            age_val = None

            try:
                if 'PatientSex' in ds:
                    sex_val = 0 if ds.PatientSex == 'M' else 1 if ds.PatientSex == 'F' else None

                if 'PatientAge' in ds:
                    age_val = ds.PatientAge  # e.g., '080Y'
            except Exception as e:
                print(f"[{series_name}] Error reading sex/age")


            rows.append([
                patient_id, num_studies, study_name, series_name, 
                main, count_col, date_col, sex_val, age_val
                ])

df = pd.DataFrame(rows, columns=["patient_id", "num_studies", "study_name", "series_name", 
                                 "main", "dcm_count", "date", "sex", "age"])

df.to_csv("ct_series_summary.csv", index=False)
print("Saving to:", os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/annabethlu/Library/Mobile Documents/com~apple~CloudDocs/lab/25 HK CityU/cleaning/NTG_ct_series_summary.csv")
patient_study_counts = df[['patient_id', 'num_studies']].drop_duplicates()
study_distribution = patient_study_counts['num_studies'].value_counts().sort_index()

# Plot
plt.figure(figsize=(8, 5))
study_distribution.plot(kind='bar')
for i, val in enumerate(study_distribution.values):
    plt.text(i, val + 0.5, str(val), ha='center', va='bottom')
plt.xlabel("Number of Studies per Patient")
plt.ylabel("Number of Patients")
plt.title("Patient Distribution by Study Count")
plt.tight_layout()
plt.savefig("study_count_distribution.png")
plt.show()
