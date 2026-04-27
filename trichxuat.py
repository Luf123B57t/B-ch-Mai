import json
import pandas as pd
from features_mapping import ClinicalDataExtractor
from Rules import InfectionChecker, process_time_without_year


# =====================================================
# 1. Read CSV files
# =====================================================
print("Read .csv files...")

chartevent = pd.read_csv("chartevents.csv", nrows=5000)
discharge = pd.read_csv("discharge.csv", nrows=5000)
labevent = pd.read_csv("labevents.csv", nrows=5000)
microbiologyevent = pd.read_csv("microbiologyevents.csv", nrows=5000)
procedureevent = pd.read_csv("procedureevents.csv", nrows=5000)
radiologyevent = pd.read_csv("radiology.csv", nrows=5000)
icu_stay = pd.read_csv("icustays.csv", nrows=1000)


mimic_data = {
    "chartevents": chartevent,
    "labevents": labevent,
    "discharge": discharge,
    "microbiology": microbiologyevent,
    "procedureevents": procedureevent,
    "radiology": radiologyevent,
}

extractor = ClinicalDataExtractor(data_tables=mimic_data)
checker = InfectionChecker(extractor=extractor, verbose=False)


# =====================================================
# 2. CLABSI criteria
# =====================================================
clabsi_criteria = [
    "Nhiệt độ",
    "Ớn lạnh",
    "Huyết áp tâm thu",
    "Cấy máu",
]

criteria_order = clabsi_criteria + [
    "Thời gian đặt catheter TMTT",
]


# =====================================================
# 3. Helper functions
# =====================================================
def get_row_time(row):
    time_col = row.get("mapped_time_column", None)

    if pd.notna(time_col) and time_col in row.index:
        return row[time_col]

    for col in ["charttime", "chartdate", "starttime"]:
        if col in row.index and pd.notna(row[col]):
            return row[col]

    return pd.NaT


def get_row_value(row):
    value_col = row.get("mapped_value_column", None)

    if pd.notna(value_col) and value_col in row.index:
        return row[value_col]

    for col in ["valuenum", "value", "org_name", "text"]:
        if col in row.index and pd.notna(row[col]):
            return row[col]

    return None


def safe_json_value(v):
    if pd.isna(v):
        return None

    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M:%S")

    if hasattr(v, "item"):
        return v.item()

    return v


# =====================================================
# 4. Extract CLABSI detail for all ICU stays
# =====================================================
flat_rows = []

for _, stay_row in icu_stay.iterrows():
    subject_id = stay_row["subject_id"]
    stay_id = stay_row["stay_id"]

    in_time = process_time_without_year(stay_row["intime"])
    out_time = process_time_without_year(stay_row["outtime"])

    if pd.isna(in_time):
        continue

    # =====================================================
    # 4.1 Normal CLABSI criteria
    # =====================================================
    for criteria in clabsi_criteria:
        try:
            df = extractor.get_variable_data(
                variable_name=criteria,
                subject_id=subject_id,
                stay_id=stay_id,
                in_time=in_time,
                time_process_func=process_time_without_year,
            )
        except Exception as e:
            print(f"Lỗi khi lấy {criteria} cho subject_id={subject_id}, stay_id={stay_id}: {e}")
            continue

        if df.empty:
            continue

        for _, row in df.iterrows():
            charttime = get_row_time(row)
            value = get_row_value(row)

            charttime = process_time_without_year(charttime)

            if pd.isna(charttime):
                continue

            if charttime < in_time:
                continue

            if pd.notna(out_time) and charttime > out_time:
                continue

            day_date = charttime.normalize()

            flat_rows.append({
                "subject_id": subject_id,
                "stay_id": stay_id,
                "day_date": day_date,
                "criteria": criteria,
                "charttime": charttime,
                "value": value,
            })

    # =====================================================
    # 4.2 Catheter duration criterion
    # =====================================================
    try:
        catheter = extractor.get_variable_data(
            variable_name="Thời gian đặt catheter TMTT",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year,
        )
    except Exception as e:
        print(
            f"Lỗi khi lấy Thời gian đặt catheter TMTT "
            f"cho subject_id={subject_id}, stay_id={stay_id}: {e}"
        )
        catheter = pd.DataFrame()

    if not catheter.empty:
        for _, row in catheter.iterrows():
            starttime = row["starttime"] if "starttime" in row.index else pd.NaT
            endtime = row["endtime"] if "endtime" in row.index else pd.NaT

            starttime = process_time_without_year(starttime)
            endtime = process_time_without_year(endtime)

            if pd.isna(starttime):
                continue

            if pd.isna(endtime):
                endtime = out_time

            if pd.isna(endtime):
                continue

            cath_start = max(starttime, in_time)

            if pd.notna(out_time):
                cath_end = min(endtime, out_time)
            else:
                cath_end = endtime

            if cath_end < cath_start:
                continue

            current_day = cath_start.normalize()
            last_day = cath_end.normalize()

            while current_day <= last_day:
                day_start = current_day
                day_end = current_day + pd.Timedelta(days=1)

                display_starttime = max(cath_start, day_start)

                if display_starttime < day_end and cath_end >= day_start:
                    flat_rows.append({
                        "subject_id": subject_id,
                        "stay_id": stay_id,
                        "day_date": current_day,
                        "criteria": "Thời gian đặt catheter TMTT",
                        "charttime": display_starttime,
                        "value": f"starttime: {display_starttime} - endtime: {cath_end}",
                    })

                current_day = current_day + pd.Timedelta(days=1)


# =====================================================
# 5. Create dataframe
# =====================================================
df_clabsi_detail = pd.DataFrame(
    flat_rows,
    columns=["subject_id", "stay_id", "day_date", "criteria", "charttime", "value"],
)


# =====================================================
# 6. Convert dataframe to nested JSON structure
# =====================================================
def build_infection_json(df_detail, criteria_order):
    all_subjects = []

    if df_detail.empty:
        return all_subjects

    df_detail = df_detail.sort_values(
        ["subject_id", "stay_id", "day_date", "criteria", "charttime"]
    )

    for subject_id, df_subject in df_detail.groupby("subject_id"):
        subject_obj = {
            "subject_id": int(subject_id),
            "icu_stays": []
        }

        for stay_id, df_stay in df_subject.groupby("stay_id"):
            stay_obj = {
                "stay_id": int(stay_id),
                "days": []
            }

            for day_date in sorted(df_stay["day_date"].unique()):
                df_day = df_stay[df_stay["day_date"] == day_date]

                day_obj = {
                    "day_date": pd.to_datetime(day_date).strftime("%Y-%m-%d"),
                    "criteria": {}
                }

                for criteria in criteria_order:
                    df_c = df_day[df_day["criteria"] == criteria].sort_values("charttime")

                    day_obj["criteria"][criteria] = [
                        {
                            "charttime": safe_json_value(r["charttime"]),
                            "value": safe_json_value(r["value"]),
                        }
                        for _, r in df_c.iterrows()
                    ]

                stay_obj["days"].append(day_obj)

            subject_obj["icu_stays"].append(stay_obj)

        all_subjects.append(subject_obj)

    return all_subjects


clabsi_json = build_infection_json(
    df_detail=df_clabsi_detail,
    criteria_order=criteria_order,
)


# =====================================================
# 7. Save to one JSON file
# =====================================================
output_path = "clabsi_json.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(clabsi_json, f, ensure_ascii=False, indent=2)

print(f"Đã lưu CLABSI JSON vào: {output_path}")
print(f"Số subject có dữ liệu: {len(clabsi_json)}")