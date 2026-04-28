import pandas as pd
import numpy as np
import os
import glob
import json
import gc
import torch
from tqdm import tqdm
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from features_mapping import ClinicalDataExtractor
from Rules import InfectionChecker

# ==========================================
# 1. CẤU HÌNH & VECTORIZED FUNCTIONS
# ==========================================
PARQUET_DIR = r'/home/user01/yte_BachMai/parquet'
BATCH_DIR = "/home/user01/yte_BachMai/infection_batches"
MODEL_DIR = r"/home/user01/yte_BachMai/clinicalbert_best"
MAX_WORKERS = 22  
GPU_BATCH_SIZE = 128 

os.makedirs(BATCH_DIR, exist_ok=True)

def vectorized_process_time(series):
    """Xử lý thời gian đưa về năm 2000 bằng vectorization (tốc độ cao)"""
    dt = pd.to_datetime(series, errors='coerce')
    # Trừ đi số năm chênh lệch để đưa về năm 2000 (Anchor Year)
    years_to_sub = dt.dt.year - 2000
    return dt - pd.to_timedelta(years_to_sub * 365.25, unit='D')

def safe_jsonl_write(file_obj, record):
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

# ==========================================
# 2. LOAD & PRE-PROCESS DATA
# ==========================================
print("🚀 Loading and vectorizing data...")
cols_chart = ["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"]
cols_lab = ["subject_id", "itemid", "charttime", "value", "valuenum", "comments"]
cols_icu = ["subject_id", "stay_id", "intime", "outtime"]

# Load Data
chartevent = pd.read_parquet(f"{PARQUET_DIR}/chartevents.parquet", columns=cols_chart)
labevent = pd.read_parquet(f"{PARQUET_DIR}/labevents.parquet", columns=cols_lab)
microbiologyevent = pd.read_parquet(f"{PARQUET_DIR}/microbiologyevents.parquet")
procedureevent = pd.read_parquet(f"{PARQUET_DIR}/procedureevents.parquet")
radiologyevent = pd.read_parquet(f"{PARQUET_DIR}/radiology.parquet")
icu_stay = pd.read_parquet(f"{PARQUET_DIR}/icustays.parquet", columns=cols_icu)

# Vectorized Time Conversion
for df, col in [(chartevent, 'charttime'), (labevent, 'charttime'), 
                (microbiologyevent, 'charttime'), (radiologyevent, 'charttime'),
                (icu_stay, 'intime'), (icu_stay, 'outtime')]:
    df[col] = vectorized_process_time(df[col])

procedureevent['starttime'] = vectorized_process_time(procedureevent['starttime'])
procedureevent['endtime'] = vectorized_process_time(procedureevent['endtime'])

# ==========================================
# 3. PRE-INFERENCE (DỰ ĐOÁN X-QUANG TRƯỚC)
# ==========================================
print("🧠 Running Batch Inference on GPU...")
unique_texts = radiologyevent['text'].dropna().unique().tolist()
prediction_map = {} # mapping: text -> is_positive (bool)

if unique_texts:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(unique_texts), GPU_BATCH_SIZE), desc="Inference"):
            batch = unique_texts[i : i + GPU_BATCH_SIZE]
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            for text, prob in zip(batch, probs):
                prediction_map[text] = bool(prob >= 0.35)

    # Giải phóng GPU ngay lập tức
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 4. HASH-MAP GROUPING (O(1) ACCESS)
# ==========================================
print("📦 Grouping tables by subject_id (Optimizing RAM usage)...")
mimic_data_grouped = {}
tables_to_group = {
    "chartevents": chartevent, "labevents": labevent,
    "microbiology": microbiologyevent, "procedureevents": procedureevent,
    "radiology": radiologyevent
}

for name, df in tables_to_group.items():
    if "subject_id" in df.columns:
        mimic_data_grouped[name] = dict(tuple(df.groupby("subject_id")))

# Giải phóng dataframe gốc để tiết kiệm RAM cho các worker
del chartevent, labevent, microbiologyevent, procedureevent, radiologyevent
gc.collect()

# ==========================================
# 5. MULTIPROCESSING WORKER
# ==========================================
def process_single_patient(patient_data):
    """Hàm chạy trong từng process con"""
    subject_id, stay_id, intime, outtime = patient_data
    
    # Khởi tạo extractor nội bộ cho từng worker (dùng chung mimic_data_grouped từ memory fork)
    # Lưu ý: prediction_map được truyền vào Rules để check imaging O(1)
    extractor = ClinicalDataExtractor(data_tables=mimic_data_grouped)
    checker = InfectionChecker(extractor=extractor, verbose=False)
    # Inject prediction_map vào checker để không phải chạy model nữa
    checker.prediction_map = prediction_map 

    try:
        # Lấy Features
        vap_feat = checker.get_vap_features(subject_id, stay_id, intime, outtime)
        clabsi_feat = checker.get_clabsi_features(subject_id, stay_id, intime, outtime)
        cauti_feat = checker.get_cauti_features(subject_id, stay_id, intime, outtime)
        
        # Check Rules
        vap_res = checker.check_vap_subject(subject_id, stay_id, intime, outtime)
        clabsi_res = checker.check_clabsi_subject(subject_id, stay_id, intime, outtime)
        cauti_res = checker.check_cauti_subject(subject_id, stay_id, intime, outtime)

        return {
            "success": True, "subject_id": subject_id, "stay_id": stay_id,
            "vap_feat": vap_feat, "clabsi_feat": clabsi_feat, "cauti_feat": cauti_feat,
            "results": [vap_res["final_vap"], clabsi_res["final_clabsi"], cauti_res["final_cauti"]]
        }
    except Exception as e:
        return {"success": False, "subject_id": subject_id, "error": str(e)}

# ==========================================
# 6. MAIN EXECUTION (BOSS PROCESS)
# ==========================================
if __name__ == "__main__":
    final_csv_path = "infection_check_detailed_results.csv"
    detailed_results = []

    with open("dataset_vap.jsonl", "w") as f_vap, \
         open("dataset_clabsi.jsonl", "w") as f_clabsi, \
         open("dataset_cauti.jsonl", "w") as f_cauti:

        tasks = [
            (row.subject_id, row.stay_id, row.intime, row.outtime)
            for row in icu_stay.itertuples(index=False)
        ]

        print(f"🔥 Starting Multiprocessing with {MAX_WORKERS} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_single_patient, t) for t in tasks]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                res = future.result()
                if res["success"]:
                    # Ghi JSONL tuần tự
                    safe_jsonl_write(f_vap, res["vap_feat"])
                    safe_jsonl_write(f_clabsi, res["clabsi_feat"])
                    safe_jsonl_write(f_cauti, res["cauti_feat"])
                    
                    # Lưu CSV kết quả
                    detailed_results.append({
                        "subject_id": res["subject_id"], "stay_id": res["stay_id"],
                        "final_vap": res["results"][0], "final_clabsi": res["results"][1],
                        "final_cauti": res["results"][2]
                    })
                else:
                    print(f"❌ Error patient {res['subject_id']}: {res['error']}")

    # Xuất file CSV cuối cùng
    pd.DataFrame(detailed_results).to_csv(final_csv_path, index=False)
    print(f"\n✅ All done! Results saved to {final_csv_path}")