import pandas as pd
import gc
from tqdm.notebook import tqdm

# from features_mapping import ClinicalDataExtractor
# from Rules import InfectionChecker, process_time_without_year

parquet_dir = "/kaggle/input/datasets/anhducleee/mimic-dataset/parquet"

print("Read .parquet files...")

cols_chart = ["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"]
cols_lab = ["subject_id", "itemid", "charttime", "value", "valuenum", "comments"]
cols_icu = ["subject_id", "stay_id", "intime", "outtime"]

chartevent = pd.read_parquet(f"{parquet_dir}/chartevents.parquet", columns=cols_chart)
labevent = pd.read_parquet(f"{parquet_dir}/labevents.parquet", columns=cols_lab)

discharge = pd.read_parquet(f"{parquet_dir}/discharge.parquet")
microbiologyevent = pd.read_parquet(f"{parquet_dir}/microbiologyevents.parquet")
procedureevent = pd.read_parquet(f"{parquet_dir}/procedureevents.parquet")
radiologyevent = pd.read_parquet(f"{parquet_dir}/radiology.parquet")
icu_stay_raw = pd.read_parquet(f"{parquet_dir}/icustays.parquet", columns=cols_icu)

print("Convert time columns...")

chartevent["charttime"] = process_time_without_year(chartevent["charttime"])
labevent["charttime"] = process_time_without_year(labevent["charttime"])
microbiologyevent["charttime"] = process_time_without_year(microbiologyevent["charttime"])
procedureevent["starttime"] = process_time_without_year(procedureevent["starttime"])
procedureevent["endtime"] = process_time_without_year(procedureevent["endtime"])
radiologyevent["charttime"] = process_time_without_year(radiologyevent["charttime"])
discharge["charttime"] = process_time_without_year(discharge["charttime"])
icu_stay_raw["intime"] = process_time_without_year(icu_stay_raw["intime"])
icu_stay_raw["outtime"] = process_time_without_year(icu_stay_raw["outtime"])

print("Filter only needed rows...")

needed_chart_itemids = [
    223761, 223762, 223991, 223992, 224370,
    220210, 220277, 226732, 223835,
    223986, 223987, 223988, 223989,
    220179,
]

needed_lab_itemids = [
    51516, 51987, 51487,
]

needed_proc_itemids = [
    225792, 229351,
]

central_line_locations = [
    "Left IJ", "Right IJ", "Left Subclavian", "Right Subclavian",
    "Left Femoral", "Left Femoral.", "Right Femoral", "Right Femoral.",
]

chartevent = chartevent[chartevent["itemid"].isin(needed_chart_itemids)].copy()
labevent = labevent[labevent["itemid"].isin(needed_lab_itemids)].copy()

procedureevent = procedureevent[
    procedureevent["itemid"].isin(needed_proc_itemids)
    | procedureevent["location"].isin(central_line_locations)
].copy()

micro_keep = [
    "BLOOD CULTURE",
    "URINE",
    "BRONCHIAL WASHINGS",
    "BRONCHOALVEOLAR LAVAGE",
    "PLEURAL FLUID",
    "SPUTUM",
]

if "spec_type_desc" in microbiologyevent.columns:
    microbiologyevent = microbiologyevent[
        microbiologyevent["spec_type_desc"].isin(micro_keep)
    ].copy()

valid_subject_ids = chartevent["subject_id"].unique()
icu_stay = icu_stay_raw[icu_stay_raw["subject_id"].isin(valid_subject_ids)].copy()

del icu_stay_raw
gc.collect()

print(f"\nĐã lọc icu_stay: Còn lại {len(icu_stay)} ca có đủ dữ liệu để đánh giá.")

print("Sort tables...")

chartevent.sort_values(["subject_id", "stay_id", "charttime"], inplace=True)
labevent.sort_values(["subject_id", "charttime"], inplace=True)
microbiologyevent.sort_values(["subject_id", "charttime"], inplace=True)
procedureevent.sort_values(["subject_id", "starttime"], inplace=True)
radiologyevent.sort_values(["subject_id", "charttime"], inplace=True)
discharge.sort_values(["subject_id", "charttime"], inplace=True)
icu_stay.sort_values(["subject_id", "stay_id"], inplace=True)

for df in [chartevent, labevent, microbiologyevent, procedureevent, radiologyevent, discharge, icu_stay]:
    df.reset_index(drop=True, inplace=True)

gc.collect()
icu_stay = icu_stay.head(100).copy()
print(f"TEST MODE: chỉ chạy {len(icu_stay)} dòng")

mimic_data = {
    "chartevents": chartevent,
    "labevents": labevent,
    "discharge": discharge,
    "microbiology": microbiologyevent,
    "procedureevents": procedureevent,
    "radiology": radiologyevent,
}

print("Khởi tạo Extractor...")
extractor = ClinicalDataExtractor(data_tables=mimic_data)
checker = InfectionChecker(extractor=extractor, verbose=False)

import os
import glob
import json
import gc

print("Processing...")

BATCH_SIZE = 30000
BATCH_DIR = "/kaggle/working/infection_batches"
os.makedirs(BATCH_DIR, exist_ok=True)

final_csv_path = "/kaggle/working/infection_check_detailed_results.csv"

final_json_paths = {
    "VAP": "/kaggle/working/dataset_vap.jsonl",
    "CLABSI": "/kaggle/working/dataset_clabsi.jsonl",
    "CAUTI": "/kaggle/working/dataset_cauti.jsonl",
}

def batch_done_path(start, end):
    return f"{BATCH_DIR}/DONE_{start}_{end}.txt"

def batch_csv_path(start, end):
    return f"{BATCH_DIR}/infection_check_detailed_results_{start}_{end}.csv"

def batch_json_path(kind, start, end):
    return f"{BATCH_DIR}/dataset_{kind.lower()}_{start}_{end}.jsonl"

def safe_jsonl_write(file_obj, record):
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\nBắt đầu quét kiểm tra nhiễm trùng theo batch...")

for batch_start in range(0, len(icu_stay), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(icu_stay))
    done_file = batch_done_path(batch_start, batch_end)

    if os.path.exists(done_file):
        print(f"Skip batch {batch_start}-{batch_end}: đã hoàn thành.")
        continue

    print(f"\n=== Processing batch {batch_start}-{batch_end} ===")

    icu_batch = icu_stay.iloc[batch_start:batch_end]

    csv_file = batch_csv_path(batch_start, batch_end)
    vap_file = batch_json_path("VAP", batch_start, batch_end)
    clabsi_file = batch_json_path("CLABSI", batch_start, batch_end)
    cauti_file = batch_json_path("CAUTI", batch_start, batch_end)

    tmp_csv = csv_file + ".tmp"
    tmp_vap = vap_file + ".tmp"
    tmp_clabsi = clabsi_file + ".tmp"
    tmp_cauti = cauti_file + ".tmp"

    results = {
        "VAP": {"positive": 0, "negative": 0},
        "CLABSI": {"positive": 0, "negative": 0},
        "CAUTI": {"positive": 0, "negative": 0},
    }

    detailed_results = []

    with open(tmp_vap, "w", encoding="utf-8") as f_vap, \
         open(tmp_clabsi, "w", encoding="utf-8") as f_clabsi, \
         open(tmp_cauti, "w", encoding="utf-8") as f_cauti:

        for row in tqdm(
            icu_batch.itertuples(index=False),
            total=len(icu_batch),
            desc=f"Batch {batch_start}-{batch_end}"
        ):
            subject_id = row.subject_id
            stay_id = row.stay_id
            intime = row.intime
            outtime = row.outtime

            wrote_vap = False
            wrote_clabsi = False
            wrote_cauti = False

            vap_check = {"final_vap": False}
            clabsi_check = {"final_clabsi": False}
            cauti_check = {"final_cauti": False}

            try:
                vap_record = checker.get_vap_features(
                    subject_id=subject_id,
                    stay_id=stay_id,
                    in_time=intime,
                    out_time=outtime
                )
                safe_jsonl_write(f_vap, vap_record)
                wrote_vap = True

                clabsi_record = checker.get_clabsi_features(
                    subject_id=subject_id,
                    stay_id=stay_id,
                    in_time=intime,
                    out_time=outtime
                )
                safe_jsonl_write(f_clabsi, clabsi_record)
                wrote_clabsi = True

                cauti_record = checker.get_cauti_features(
                    subject_id=subject_id,
                    stay_id=stay_id,
                    in_time=intime,
                    out_time=outtime
                )
                safe_jsonl_write(f_cauti, cauti_record)
                wrote_cauti = True

                vap_check = checker.check_vap_subject(
                    subject_id=subject_id,
                    stay_id=stay_id,
                    in_time=intime,
                    out_time=outtime
                )

                clabsi_check = checker.check_clabsi_subject(
                    subject_id=subject_id,
                    stay_id=stay_id,
                    in_time=intime,
                    out_time=outtime
                )

                cauti_check = checker.check_cauti_subject(
                    subject_id=subject_id,
                    stay_id=stay_id,
                    in_time=intime,
                    out_time=outtime
                )

            except Exception as e:
                print(f"Lỗi subject_id={subject_id}, stay_id={stay_id}: {e}")

                if not wrote_vap:
                    safe_jsonl_write(f_vap, [])
                if not wrote_clabsi:
                    safe_jsonl_write(f_clabsi, [])
                if not wrote_cauti:
                    safe_jsonl_write(f_cauti, [])

            results["VAP"]["positive"] += int(vap_check["final_vap"])
            results["VAP"]["negative"] += int(not vap_check["final_vap"])

            results["CLABSI"]["positive"] += int(clabsi_check["final_clabsi"])
            results["CLABSI"]["negative"] += int(not clabsi_check["final_clabsi"])

            results["CAUTI"]["positive"] += int(cauti_check["final_cauti"])
            results["CAUTI"]["negative"] += int(not cauti_check["final_cauti"])

            detailed_results.append({
                "subject_id": subject_id,
                "stay_id": stay_id,
                "final_vap": vap_check["final_vap"],
                "final_clabsi": clabsi_check["final_clabsi"],
                "final_cauti": cauti_check["final_cauti"],
            })

            checker.clear_cache()

    df_batch = pd.DataFrame(
        detailed_results,
        columns=["subject_id", "stay_id", "final_vap", "final_clabsi", "final_cauti"]
    )
    df_batch.to_csv(tmp_csv, index=False, encoding="utf-8")

    os.replace(tmp_csv, csv_file)
    os.replace(tmp_vap, vap_file)
    os.replace(tmp_clabsi, clabsi_file)
    os.replace(tmp_cauti, cauti_file)

    with open(done_file, "w", encoding="utf-8") as f:
        f.write("done\n")

    print(f"Saved batch CSV: {csv_file}")
    print(f"Saved VAP JSONL: {vap_file}")
    print(f"Saved CLABSI JSONL: {clabsi_file}")
    print(f"Saved CAUTI JSONL: {cauti_file}")

    for infection_type, stats in results.items():
        print(f"{infection_type}: {stats['positive']} Ca Mắc | {stats['negative']} Ca Không")

    del detailed_results, df_batch, icu_batch
    gc.collect()

print("\nGộp batch thành output cuối...")

csv_files = sorted(
    glob.glob(f"{BATCH_DIR}/infection_check_detailed_results_*.csv"),
    key=lambda x: int(os.path.basename(x).replace(".csv", "").split("_")[-2])
)

if not csv_files:
    raise RuntimeError("Không tìm thấy batch CSV nào để gộp.")

df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df_all.to_csv(final_csv_path, index=False, encoding="utf-8")
print(f"Saved final CSV: {final_csv_path}")

for kind, final_json_path in final_json_paths.items():
    json_files = sorted(
        glob.glob(f"{BATCH_DIR}/dataset_{kind.lower()}_*.jsonl"),
        key=lambda x: int(os.path.basename(x).replace(".jsonl", "").split("_")[-2])
    )

    if not json_files:
        raise RuntimeError(f"Không tìm thấy batch JSONL cho {kind}.")

    with open(final_json_path, "w", encoding="utf-8") as fout:
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

    print(f"Saved final JSONL {kind}: {final_json_path}")

print("\n--- KẾT QUẢ THỐNG KÊ FINAL ---")
for col, name in [
    ("final_vap", "VAP"),
    ("final_clabsi", "CLABSI"),
    ("final_cauti", "CAUTI"),
]:
    positive = int(df_all[col].astype(bool).sum())
    negative = int((~df_all[col].astype(bool)).sum())
    print(f"{name}: {positive} Ca Mắc | {negative} Ca Không")