# import pandas as pd

# # 1. Load CSV
# csv_path = "~/CSC494_privileged_information/prompting/augmented_data_8B_mimic.csv"
# df = pd.read_csv(csv_path)

# # 2. Count NaNs per column, sorted descending, and compute ratio
# nan_counts = df.isna().sum()
# nan_ratios = df.isna().mean() * 100
# nan_summary = pd.DataFrame({
#     "MissingCount": nan_counts,
#     "MissingRatio(%)": nan_ratios
# })
# nan_summary = nan_summary[nan_summary["MissingCount"] > 0] \
#     .sort_values("MissingCount", ascending=False)

# print("Number and ratio of missing values per column (sorted descending):")
# for col, row in nan_summary.iterrows():
#     print(f"{col}: {int(row['MissingCount'])} missing ({row['MissingRatio(%)']:.2f}%)")

# # 3. Check 'label' column (binary target)
# if 'label' in df.columns:
#     print("\n🧪 'label' (Mortality) class distribution:")
#     print(df['label'].value_counts())
# else:
#     print("\n⚠️ The 'label' column does not exist.")


# # # Raw data as a multiline string (copy-paste from your input)
# # raw_data = """
# # sid,0,0.0
# # Hemoglobin,326,1.2
# # WBC,618,2.2
# # Platelets,626,2.2
# # Sodium,3626,12.9
# # Potassium,2444,8.7
# # Calcium,2992,10.6
# # Phosphate,2978,10.6
# # Magnesium,1424,5.1
# # INR,4173,14.8
# # ALP,15991,56.9
# # Bilirubin,15957,56.8
# # ALT,15857,56.4
# # Lactate,11155,39.7
# # PaCO2,11348,40.4
# # PaO2,13924,49.5
# # ph,11002,39.1
# # Bicarbonate,517,1.8
# # Creatinine,342,1.2
# # BloodUreaNitrogen,194,0.7
# # Troponin,22374,79.6
# # CreatinineKinase,20593,73.3
# # Antipsychotics,0,0.0
# # MRI,0,0.0
# # Xray,0,0.0
# # Anticoagulants,0,0.0
# # Propofol,0,0.0
# # Antiarrhythmics,0,0.0
# # PReplacement,0,0.0
# # Paralysis,0,0.0
# # Antibiotics,0,0.0
# # MgReplacement,0,0.0
# # Sedation,0,0.0
# # Antihypertensives,0,0.0
# # UltraSound,0,0.0
# # Antiepileptics,0,0.0
# # Ventilation,0,0.0
# # TPN,0,0.0
# # Transfusions,0,0.0
# # Dialysis,0,0.0
# # PronePosition,0,0.0
# # EnteralNutrition,0,0.0
# # ICPMonitor,0,0.0
# # CaReplacement,0,0.0
# # KReplacement,0,0.0
# # Analgesia,0,0.0
# # CTScan,0,0.0
# # Diuretics,0,0.0
# # PPI,0,0.0
# # UrineOutput,1149,4.1
# # MeanBloodPressure,101,0.4
# # Temperature,2290,8.1
# # HeartRate,14,0.0
# # SystolicBloodPressure,99,0.4
# # MinuteVentilation,15943,56.7
# # GCS,37,0.1
# # FiO2,13835,49.2
# # TidalVolume,16017,57.0
# # ICDSC,22896,81.4
# # DiastolicBloodPressure,101,0.4
# # AirwayPressure,16037,57.0
# # PEEP,15948,56.7
# # RespiratoryRate,50,0.2
# # SAS,24757,88.1
# # Vasopressors_221289,26879,95.6
# # Vasopressors_221653,27890,99.2
# # Vasopressors_221662,27502,97.8
# # Vasopressors_221749,21943,78.1
# # Vasopressors_221906,23788,84.6
# # Vasopressors_221986,27792,98.9
# # Vasopressors_222315,27308,97.1
# # Vasopressors_229617,28101,100.0
# # """

# # # Split and process the lines
# # lines = raw_data.strip().split('\n')
# # nan_20 = set()
# # nan_50 = set()

# # # Iterate through the lines and extract variable names by NaN ratio thresholds
# # for line in lines:
# #     parts = line.split(',')
# #     if len(parts) != 3:
# #         continue
# #     var, count, ratio = parts[0], int(parts[1]), float(parts[2])
# #     if ratio >= 20:
# #         nan_20.add(var)
# #     if ratio >= 50:
# #         nan_50.add(var)

# # # Output the sets
# # print("Variables with NaN ratio ≥ 20%:\n", nan_20)
# # print("Variables with NaN ratio ≥ 50%:\n", nan_50)
# # print("Variables with ≥ 50% NaN ratio ≥ 20%:\n", nan_20-nan_50)


import os
import pandas as pd
from sklearn.model_selection import train_test_split





import pandas as pd

# Define the path to the CSV file
file_path = 'prompting/augmented_data_8B_mimic.csv'
# Load the dataset into a DataFrame
df = pd.read_csv(file_path)

train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
numeric_cols = train_df.select_dtypes(include='number').columns
    
medians = train_df[numeric_cols].median()
    
train_df[numeric_cols] = train_df[numeric_cols].fillna(medians)
val_df[numeric_cols] = val_df[numeric_cols].fillna(medians)
test_df[numeric_cols] = test_df[numeric_cols].fillna(medians)

# Count the number of missing (NaN) values in each column
missing_counts = val_df.isna().sum()

# Sort the counts in descending order
missing_counts_sorted = missing_counts.sort_values(ascending=False)

# Print the sorted missing-value counts
print(missing_counts_sorted)