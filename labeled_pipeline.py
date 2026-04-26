import pandas as pd
from features_mapping import ClinicalDataExtractor
from Rules import InfectionChecker, process_time_without_year
from tqdm.notebook import tqdm

# NROWS = 15000000
parquet_dir = r'/home/user04/Yte_BachMai/dataset/parquet'

print("Read .parquet files...")
cols_chart = ['subject_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']
chartevent = pd.read_parquet(f'{parquet_dir}/chartevents_full.parquet', columns=cols_chart)

cols_lab = ['subject_id', 'itemid', 'charttime', 'value', 'valuenum', 'comments']
labevent = pd.read_parquet(f'{parquet_dir}/labevents_full.parquet', columns=cols_lab)

discharge = pd.read_parquet(f'{parquet_dir}/discharge.parquet')
microbiologyevent = pd.read_parquet(f'{parquet_dir}/microbiologyevents.parquet')
procedureevent = pd.read_parquet(f'{parquet_dir}/procedureevents.parquet')
radiologyevent = pd.read_parquet(f'{parquet_dir}/radiology.parquet')
icu_stay_raw = pd.read_parquet(f'{parquet_dir}/icustays.parquet')

valid_subject_ids = chartevent['subject_id'].unique()
icu_stay = icu_stay_raw[icu_stay_raw['subject_id'].isin(valid_subject_ids)].copy()
print(f"\nĐã lọc icu_stay: Còn lại {len(icu_stay)}/{len(icu_stay_raw)} ca có đủ dữ liệu để đánh giá.")

mimic_data = {
        "chartevents": chartevent,
        "labevents": labevent,
        "discharge": discharge,
        "microbiology": microbiologyevent,
        "procedureevents": procedureevent,
        "radiology": radiologyevent
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

for row in tqdm(icu_stay.itertuples(), total=len(icu_stay), desc="Đang xử lý bệnh nhân"):
    subject_id = row.subject_id
    intime = process_time_without_year(row.intime)
    outtime = process_time_without_year(row.outtime)
    stay_id = row.stay_id
    
    # print("="*50)
    # print("Processing patient {}....".format(subject_id))

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

    print()

    detailed_results.append({
        "subject_id": subject_id,
        "stay_id": stay_id,
        "rule1_vap": vap_check['rule1'],
        "rule2_vap": vap_check['rule2'],
        "rule3_vap": vap_check['rule3'],
        "final_vap": vap_check['final_vap'],
        "rule1_clabsi": clabsi_check['rule1'],
        "rule2_clabsi": clabsi_check['rule2'],
        "final_clabsi": clabsi_check['final_clabsi'],
        "rule1_cauti": cauti_check['rule1'],
        "rule2_cauti": cauti_check['rule2'],
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
    