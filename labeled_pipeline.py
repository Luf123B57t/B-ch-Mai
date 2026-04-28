import pandas as pd
from features_mapping import ClinicalDataExtractor
from Rules import InfectionChecker, process_time_without_year
from tqdm import tqdm
import json

# NROWS = 15000000
parquet_dir = r'/home/user01/yte_BachMai/parquet'

print("Read .parquet files...")
cols_chart = ['subject_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']
chartevent = pd.read_parquet(f'{parquet_dir}/chartevents.parquet', columns=cols_chart)

cols_lab = ['subject_id', 'itemid', 'charttime', 'value', 'valuenum', 'comments']
labevent = pd.read_parquet(f'{parquet_dir}/labevents.parquet', columns=cols_lab)

discharge = pd.read_parquet(f'{parquet_dir}/discharge.parquet')
microbiologyevent = pd.read_parquet(f'{parquet_dir}/microbiologyevents.parquet')
procedureevent = pd.read_parquet(f'{parquet_dir}/procedureevents.parquet')
radiologyevent = pd.read_parquet(f'{parquet_dir}/radiology.parquet')
icu_stay = pd.read_parquet(f'{parquet_dir}/icustays.parquet')

mimic_data = {
        "chartevents": chartevent,
        "labevents": labevent,
        "discharge": discharge,
        "microbiology": microbiologyevent,
        "procedureevents": procedureevent,
        "radiology": radiologyevent
}

file_paths = {
    "VAP": "dataset_vap.jsonl",
    "CLABSI": "dataset_clabsi.jsonl",
    "CAUTI": "dataset_cauti.jsonl"
}

print("Khởi tạo Extractor...")
extractor = ClinicalDataExtractor(data_tables=mimic_data)
checker = InfectionChecker(extractor=extractor, verbose= False)

print("Processing...")
results = {
    "VAP": {"positive": 0, "negative": 0},
    "CLABSI": {"positive": 0, "negative": 0},
    "CAUTI": {"positive": 0, "negative": 0}
}

detailed_results = []
print("\nBắt đầu quét kiểm tra nhiễm trùng...")
with open(file_paths["VAP"], "w", encoding="utf-8") as f_vap, \
     open(file_paths["CLABSI"], "w", encoding="utf-8") as f_clabsi, \
     open(file_paths["CAUTI"], "w", encoding="utf-8") as f_cauti:

    for row in tqdm(icu_stay.itertuples(), total=len(icu_stay), desc="Đang xử lý bệnh nhân"):
        subject_id = row.subject_id
        intime = process_time_without_year(row.intime)
        outtime = process_time_without_year(row.outtime)
        stay_id = row.stay_id
        
        # print("="*50)
        # print("Processing patient {}....".format(subject_id))
        
        # --- 3. Đóng gói và ghi vào 3 file khác nhau ---
        
        # Record cho file VAP
        vap_record = checker.get_vap_features(subject_id=subject_id, stay_id=stay_id, in_time=intime, out_time=outtime)
        f_vap.write(json.dumps(vap_record, ensure_ascii=False) + "\n")
        
        # Record cho file CLABSI
        clabsi_record = checker.get_clabsi_features(subject_id=subject_id, stay_id=stay_id, in_time=intime, out_time=outtime)
        f_clabsi.write(json.dumps(clabsi_record, ensure_ascii=False) + "\n")
        
        # Record cho file CAUTI
        cauti_record = checker.get_cauti_features(subject_id=subject_id, stay_id=stay_id, in_time=intime, out_time=outtime)
        f_cauti.write(json.dumps(cauti_record, ensure_ascii=False) + "\n")

        vap_check = checker.check_vap_subject(subject_id=subject_id, stay_id=stay_id, in_time = intime, out_time = outtime)
        if vap_check['final_vap']:
            results["VAP"]["positive"] += 1
        else:
            results["VAP"]["negative"] += 1

        clabsi_check = checker.check_clabsi_subject(subject_id=subject_id, stay_id=stay_id, in_time = intime, out_time = outtime)
        if clabsi_check['final_clabsi']:
            results["CLABSI"]["positive"] += 1
        else:
            results["CLABSI"]["negative"] += 1

        cauti_check = checker.check_cauti_subject(subject_id=subject_id, stay_id=stay_id, in_time = intime, out_time = outtime)
        if cauti_check['final_cauti']:
            results["CAUTI"]["positive"] += 1
        else:
            results["CAUTI"]["negative"] += 1
        
        detailed_results.append({
            "subject_id": subject_id,
            "stay_id": stay_id,
            "final_vap": vap_check['final_vap'],
            "final_clabsi": clabsi_check['final_clabsi'],
            "final_cauti": cauti_check['final_cauti']
        })        
        checker.clear_cache()


print("\n--- KẾT QUẢ THỐNG KÊ ---")
for infection_type, stats in results.items():
    print(f"{infection_type}: {stats['positive']} Ca Mắc | {stats['negative']} Ca Không")

print("\nĐang lưu kết quả chi tiết ra file CSV...")
# Chuyển list các dictionary thành DataFrame
df_results = pd.DataFrame(detailed_results)

# Xuất ra file CSV (bạn có thể đổi tên file và đường dẫn tùy ý)
csv_filename = "infection_check_detailed_results.csv"
df_results.to_csv(csv_filename, index=False, encoding='utf-8')

print(f"Đã lưu thành công dữ liệu chi tiết vào file: {csv_filename}")
    