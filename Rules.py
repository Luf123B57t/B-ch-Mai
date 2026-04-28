import pandas as pd
import numpy as np
import torch
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from features_mapping import ClinicalDataExtractor

def process_time_without_year(date_series):
    """
    Converts an object/string series or a single scalar to datetime and neutralizes the year.
    This allows you to calculate time differences (months, days, hours, mins) 
    while completely ignoring the original year.
    
    Parameters:
    date_series (pd.Series, list, or single value): The input date(s) as strings/objects.
    
    Returns:
    pd.Series or pd.Timestamp: Datetime object(s) with the year standardized to 2000.
    """
    # 1. Convert the object to a proper datetime type
    dt_series = pd.to_datetime(date_series, errors='coerce')
    
    # 2. Xử lý trường hợp đầu vào là một chuỗi/cột (pd.Series)
    if isinstance(dt_series, pd.Series):
        return dt_series.apply(lambda x: x.replace(year=2000) if pd.notnull(x) else x)
    
    # 3. Xử lý trường hợp đầu vào là một giá trị đơn (pd.Timestamp)
    else:
        if pd.notnull(dt_series):
            return dt_series.replace(year=2000)
        return dt_series  # Trả về NaT nếu giá trị truyền vào là null/rỗng
    
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

class InfectionChecker:
    """Class quản lý việc kiểm tra các loại nhiễm trùng (HAIs) của bệnh nhân."""
    
    def __init__(self,extractor: ClinicalDataExtractor, verbose: bool = True):
        self.extractor = extractor
        self.verbose = verbose
        MODEL_DIR = r"/home/user01/yte_BachMai/clinicalbert_best"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(self.device)

        self._data_cache = {}


    def clear_cache(self):
        self._data_cache = {}

    # def get_data_with_48h_rule(self, variable_name, subject_id, stay_id=None, in_time=None, out_time=None, check_48h=False):
    #     """
    #     Hàm Wrapper để lấy dữ liệu. Tự động lưu cache để dùng lại nếu gọi trùng tham số.
    #     """
    #     if subject_id is None or in_time is None:
    #         raise ValueError(f"Lỗi [{variable_name}]: Bắt buộc phải truyền subject_id và in_time.")
        
    #     if stay_id is None and out_time is None:
    #         raise ValueError(f"Lỗi [{variable_name}]: Nếu không có stay_id thì bắt buộc phải truyền out_time.")
        
    #     # Tạo một tuple khóa (key) đại diện cho lời gọi này
    #     cache_key = (variable_name, subject_id, stay_id, in_time, out_time, check_48h)
        
    #     if cache_key not in self._data_cache:
    #         # Tạo dictionary các tham số hợp lệ để truyền vào extractor
    #         kwargs = {
    #             'variable_name': variable_name,
    #             'subject_id': subject_id,
    #             'time_process_func': process_time_without_year
    #         }
    #         if stay_id is not None: kwargs['stay_id'] = stay_id
    #         if in_time is not None: kwargs['in_time'] = in_time
    #         if out_time is not None: kwargs['out_time'] = out_time
    #         if check_48h is not None: kwargs['check_48h'] = check_48h

    #         self._data_cache[cache_key] = self.extractor.get_variable_data(**kwargs)
            
    #     # Luôn trả về copy() để các hàm xử lý xóa/sửa cột không làm hỏng cache
    #     return self._data_cache[cache_key].copy()
    def _fetch_and_cache_raw_data(self, variable_name, subject_id, stay_id=None, in_time=None, out_time=None):
        """
        HÀM 1: Chỉ lấy dữ liệu gốc và đưa vào Cache. Không xử lý luật y khoa.
        """
        if subject_id is None or in_time is None:
            raise ValueError(f"Lỗi [{variable_name}]: Bắt buộc phải truyền subject_id và in_time.")
        
        if stay_id is None and out_time is None:
            raise ValueError(f"Lỗi [{variable_name}]: Bắt buộc truyền stay_id hoặc out_time.")
        
        cache_key = (variable_name, subject_id, stay_id, in_time, out_time)
        
        if cache_key not in self._data_cache:
            kwargs = {
                'variable_name': variable_name,
                'subject_id': subject_id
            }
            if stay_id is not None: kwargs['stay_id'] = stay_id
            if in_time is not None: kwargs['in_time'] = in_time
            if out_time is not None: kwargs['out_time'] = out_time
            # if check_48h is not None: kwargs['check_48h'] = check_48h

           
            self._data_cache[cache_key] = self.extractor.get_variable_data(**kwargs)
            
        return self._data_cache[cache_key].copy()


    def get_data_with_48h_rule(self, variable_name, subject_id, stay_id=None, in_time=None, out_time=None):
        cache_key = (variable_name, subject_id, stay_id, in_time, out_time)

        if cache_key not in self._data_cache:
            self._fetch_and_cache_raw_data(
                variable_name=variable_name,
                subject_id=subject_id,
                stay_id=stay_id,
                in_time=in_time,
                out_time=out_time
            )

        df = self._data_cache[cache_key]

        if df is None or df.empty:
            return df

        time_col = None

        if "mapped_time_column" in df.columns and df["mapped_time_column"].notna().any():
            time_col = df["mapped_time_column"].dropna().iloc[0]

        if time_col is None or time_col not in df.columns:
            for c in ["charttime", "starttime", "chartdate"]:
                if c in df.columns:
                    time_col = c
                    break

        if time_col is None or time_col not in df.columns:
            raise KeyError(f"Dữ liệu của {variable_name} không có cột thời gian để tính 48h.")

        return df[
            ((df[time_col] - in_time).dt.total_seconds() / 3600.0) >= 48.0
        ].copy()

    def _check_spo2_worsening(self, subject_id, stay_id, in_time:pd.Timestamp):
        """
        Tiêu chí SpO2:
        - SpO2 < 94
        HOẶC
        - giảm >= 4 điểm so với lần đo gần nhất trước đó

        Trả về:
        - spo2_positive: bool
        - first_spo2_time: thời điểm đầu tiên thỏa tiêu chí
        - spo2_df: bảng debug
        """

        spo2 = self.get_data_with_48h_rule(
            variable_name="SpO2",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )

        if spo2.empty:
            return False, None, pd.DataFrame()

        spo2 = spo2[spo2["valuenum"].notna()].copy()
        spo2 = spo2.sort_values("charttime").reset_index(drop=True)

        if spo2.empty:
            return False, None, pd.DataFrame()

        spo2["prev_spo2"] = spo2["valuenum"].shift(1)
        spo2["spo2_drop"] = spo2["prev_spo2"] - spo2["valuenum"]

        # CHỐT THEO BẠN
        spo2["spo2_lt_94"] = spo2["valuenum"] < 94
        spo2["spo2_drop_ge_4"] = spo2["spo2_drop"] >= 4
        spo2["spo2_worsened"] = spo2["spo2_lt_94"] | spo2["spo2_drop_ge_4"]

        positive_rows = spo2[spo2["spo2_worsened"] == True]

        if positive_rows.empty:
            return False, None, spo2

        first_spo2_time = positive_rows.iloc[0]["charttime"]
        return True, first_spo2_time, spo2

    def _get_vap_symptoms(self, subject_id: int, stay_id: int, in_time:pd.Timestamp):
        """
        Trả về:
        - symptom_flags: dict các cờ triệu chứng
        - symptom_count: tổng số triệu chứng dương tính
        - first_time: thời điểm sớm nhất xuất hiện 1 tiêu chí
        - spo2_debug: bảng debug SpO2
        """
        FAR_FUTURE = pd.Timestamp("2001-01-01")
        first_time = FAR_FUTURE

        sot = False
        ho_flag = False
        dom_mu_flag = False
        tho_nhanh_flag = False
        lung_sound_flag = False
        spo2_flag = False

        # -------- Sốt --------
        sot = False
        nhiet_do = self.get_data_with_48h_rule("Nhiệt độ", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        if not nhiet_do.empty:
            mask = ((nhiet_do['itemid'] == 223761) & (nhiet_do['valuenum'] > 100.4)) | \
                   ((nhiet_do['itemid'] == 223762) & (nhiet_do['valuenum'] > 38))
            df_sot = nhiet_do[mask]
            if not df_sot.empty:
                sot = True
                first_time = min(first_time, df_sot['charttime'].min())

        # -------- Ho --------
        ho = self.get_data_with_48h_rule("Ho", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        for sample in ho.itertuples(index=True, name="Pandas"):
            val = str(sample.value).strip().lower()
            if sample.itemid == 223991 and val in ["weak", "strong"]:
                ho_flag = True
                first_time = min(first_time, sample.charttime)

        # -------- Đờm mủ --------
        dom_mu = self.get_data_with_48h_rule("Đờm mủ", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        for sample in dom_mu.itertuples(index=True, name="Pandas"):
            val = str(sample.value).strip().lower()
            if val != "clear":
                dom_mu_flag = True
                first_time = min(first_time, sample.charttime)

        # -------- Thở nhanh --------
        nhip_tho = self.get_data_with_48h_rule("Nhịp thở", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        for sample in nhip_tho.itertuples(index=True, name="Pandas"):
            if pd.notna(sample.valuenum) and sample.valuenum > 20:
                tho_nhanh_flag = True
                first_time = min(first_time, sample.charttime)

        # -------- SpO2 --------
        spo2_positive, first_spo2_time, spo2_debug = self._check_spo2_worsening(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )
        if spo2_positive:
            spo2_flag = True
            if first_spo2_time is not None:
                first_time = min(first_time, first_spo2_time)

        # -------- Âm phổi --------
        lung_keywords = {
            "bronchial", "crackles", "diminished",
            "exp wheeze", "ins/exp wheeze", "insp wheeze",
            "rhonchi", "tubular"
        }

        rale = self.get_data_with_48h_rule("Rale", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        for sample in rale.itertuples(index=True, name="Pandas"):
            val = str(sample.value).strip().lower()
            if val in lung_keywords:
                lung_sound_flag = True
                first_time = min(first_time, sample.charttime)

        # bronchial = self.get_data_with_48h_rule("Tiếng thở phế quản", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        # for sample in bronchial.itertuples(index=True, name="Pandas"):
        #     val = str(sample.value).strip().lower()
        #     if val == "bronchial":
        #         lung_sound_flag = True
        #         first_time = min(first_time, sample.charttime)

        symptom_flags = {
            "sot": sot,
            "ho": ho_flag,
            "dom_mu": dom_mu_flag,
            "tho_nhanh": tho_nhanh_flag,
            "spo2": spo2_flag,
            "lung_sound": lung_sound_flag
        }

        symptom_count = sum(symptom_flags.values())

        if first_time == FAR_FUTURE:
            first_time = None

        return symptom_flags, symptom_count, first_time, spo2_debug

    def _get_vap_imaging_positive(self, subject_id: int,
    stay_id: int,
    in_time: pd.Timestamp,
    out_time: pd.Timestamp
):
        """
        Kiểm tra imaging gợi ý viêm phổi từ:
        - X-quang ngực
        - CT scan lồng ngực
        """

        micro_positive = self._get_vap_micro_positive(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )
        if micro_positive:
            return True

        imaging_positive = False

        xray = self.get_data_with_48h_rule("X-quang ngực", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        # ct = self.get_data_with_48h_rule("CT scan lồng ngực", subject_id=subject_id, stay_id=stay_id, in_time=in_time)

        THRESHOLD = 0.35
        MAX_LENGTH = 256

        text_list = []

        for df_img in [xray]:
            if df_img.empty:
                continue
            if "text" not in df_img.columns:
                continue

            for sample in df_img.itertuples(index=True, name="Pandas"):
                text = str(sample.text).strip().lower()
                if text and text != "nan":
                    text_list.append(text)

        if len(text_list) == 0:
            return False

        self.model.eval()

        def predict_batch(texts, batch_size=8):
            probs_all = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

                probs_all.extend(probs)

            return np.array(probs_all)

        probs = predict_batch(text_list)
        preds = (probs >= THRESHOLD).astype(int)

        if len(preds) > 0 and preds.max() == 1:
            imaging_positive = True

        return imaging_positive


    def _get_vap_micro_positive(self, subject_id: int, stay_id: int, in_time):
        """
        Rule microbiology:
        Cấy dịch đường hô hấp HOẶC cấy máu

        Sửa để dùng theo từng stay_id, tránh leak dữ liệu giữa các stay.
        """
        micro_positive = False

        cay_dich = self.get_data_with_48h_rule("Cấy dịch đường hô hấp", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        for sample in cay_dich.itertuples(index=True, name="Pandas"):
            if pd.notna(sample.org_name):
                micro_positive = True
                break

        if not micro_positive:
            cay_mau = self.get_data_with_48h_rule("Cấy máu", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
            for sample in cay_mau.itertuples(index=True, name="Pandas"):
                if pd.notna(sample.org_name):
                    micro_positive = True
                    break

        return micro_positive

    def _check_ventilation_before_first_time(self, subject_id: int, stay_id: int, in_time:pd.Timestamp, first_time:pd.Timestamp):
        """
        Có thở máy/đặt NKQ tại chỗ >= 2 ngày trước lần đầu xuất hiện 1 tiêu chí
        """
        if first_time is None:
            return False

        tho_may = self._fetch_and_cache_raw_data("Thời gian thở máy", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        for sample in tho_may.itertuples(index=True, name="Pandas"):
            if pd.notna(sample.starttime):
                if first_time - sample.starttime >= pd.Timedelta(days=2):
                    return True

        return False

    # =========================================================
    # VAP RULE 1
    # >= 2 triệu chứng
    # + imaging
    # + microbiology
    # + vent >= 2 ngày trước first_time
    # =========================================================
    def check_vap_rule1(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        symptom_flags, symptom_count, first_time, spo2_debug = self._get_vap_symptoms(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )

        imaging_positive = self._get_vap_imaging_positive(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            out_time=out_time
        )

        micro_positive = self._get_vap_micro_positive(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )

        ventilation_ok = self._check_ventilation_before_first_time(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            first_time=first_time
        )

        vap_rule1 = (
            symptom_count >= 2 and
            imaging_positive and
            micro_positive and
            ventilation_ok
        )

        return vap_rule1
    # =========================================================
    # VAP RULE 2
    # >= 3 triệu chứng
    # + vent >= 2 ngày trước first_time
    # =========================================================
    def check_vap_rule2(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        if in_time is None:
            return False

        symptom_flags, symptom_count, first_time, spo2_debug = self._get_vap_symptoms(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )

        ventilation_ok = self._check_ventilation_before_first_time(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            first_time=first_time
        )

        vap_rule2 = (
            symptom_count >= 3 and
            ventilation_ok
        )

        # if self.verbose:
        #     print("\n===== VAP RULE 2 =====")
        #     print("symptom_flags:", symptom_flags)
        #     print("symptom_count:", symptom_count)
        #     print("first_time:", first_time)
        #     print("ventilation_ok:", ventilation_ok)
        #     print("\nSpO2 debug:")
        #     print(spo2_debug)
        #     print("=> VAP RULE 2:", vap_rule2)

        return vap_rule2

    # =========================================================
    # VAP RULE 3
    # >= 2 triệu chứng
    # + imaging
    # + vent >= 2 ngày trước first_time
    # =========================================================
    def check_vap_rule3(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        symptom_flags, symptom_count, first_time, spo2_debug = self._get_vap_symptoms(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )

        imaging_positive = self._get_vap_imaging_positive(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            out_time=out_time
        )

        ventilation_ok = self._check_ventilation_before_first_time(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            first_time=first_time
        )

        vap_rule3 = (
            symptom_count >= 2 and
            imaging_positive and
            ventilation_ok
        )

        return vap_rule3

    # =========================================================
    # HÀM TỔNG HỢP
    # =========================================================
    def check_vap_subject(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        symptom_flags, symptom_count, first_time, spo2_debug = self._get_vap_symptoms(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )
    
        micro_positive = self._get_vap_micro_positive(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )
    
        imaging_positive = self._get_vap_imaging_positive(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            out_time=out_time
        )
    
        ventilation_ok = self._check_ventilation_before_first_time(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            first_time=first_time
        )

        r1 = symptom_count >= 2 and imaging_positive and micro_positive and ventilation_ok
        r2 = symptom_count >= 3 and ventilation_ok
        r3 = symptom_count >= 2 and imaging_positive and ventilation_ok
    
        return {
            "rule1": r1,
            "rule2": r2,
            "rule3": r3,
            "final_vap": r1 or r2 or r3
        }
            

    def check_clabsi_subject(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        FAR_FUTURE = pd.Timestamp('2001-01-01')
        first_time1 = FAR_FUTURE
        first_time2 = FAR_FUTURE

        skin_contaminants = [
            'COAGULASE NEGATIVE STAPHYLOCOCCUS',
            'STAPHYLOCOCCUS EPIDERMIDIS',
            'CORYNEBACTERIUM SPECIES',
            'PROPIONIBACTERIUM ACNES',
            'BACILLUS SPECIES',
            'MICROCOCCUS SPECIES',
            # 'VIRIDANS GROUP STREPTOCOCCI'
        ]

        # =========== RULE 1 ===========

        # Cấy máu dương tính với tác nhân gây bệnh
        cay_mau_duong_tinh = False
        cay_mau = self.get_data_with_48h_rule("Cấy máu", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        for sample in cay_mau.itertuples(index=True, name='Pandas'):
            if sample.spec_type_desc == "BLOOD CULTURE" and pd.notna(sample.org_name) and sample.org_name.upper() not in skin_contaminants:
                cay_mau_duong_tinh = True
                first_time1 = min(first_time1, sample.charttime)
        
        # Thời gian đặt catheter TMTT
        thoi_gian_dat_catheter = False
        catheter = self._fetch_and_cache_raw_data("Thời gian đặt catheter TMTT", subject_id=subject_id, out_time=out_time, in_time=in_time)
        for sample in catheter.itertuples(index=True, name='Pandas'):
            if first_time1 != FAR_FUTURE and first_time1 - sample.starttime >= pd.Timedelta(days=2):
                thoi_gian_dat_catheter = True
                break
        
        r1 = cay_mau_duong_tinh and thoi_gian_dat_catheter

        # =========== RULE 2 ===========

        # Sot
        sot = False
        nhiet_do = self.get_data_with_48h_rule("Nhiệt độ", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        if not nhiet_do.empty:
            mask = ((nhiet_do['itemid'] == 223761) & (nhiet_do['valuenum'] > 100.4)) | \
                   ((nhiet_do['itemid'] == 223762) & (nhiet_do['valuenum'] > 38))
            df_sot = nhiet_do[mask]
            if not df_sot.empty:
                sot = True
                first_time2 = min(first_time2, df_sot['charttime'].min())
                
        # On lanh
        onlanh = False
        on_lanh = self.get_data_with_48h_rule("Ớn lạnh", subject_id=subject_id, out_time=out_time, in_time=in_time)

        if not on_lanh.empty:

            onlanh = True

            min_charttime = on_lanh['charttime'].min()

            first_time2 = min(first_time2, min_charttime)
        
        # Huyết áp thấp
        huyet_ap_thap = False
        huyet_ap = self.get_data_with_48h_rule("Huyết áp tâm thu", subject_id=subject_id, stay_id = stay_id, in_time=in_time)
        if not huyet_ap.empty and 'valuenum' in huyet_ap.columns:
            df_huyet_ap_thap = huyet_ap[huyet_ap['valuenum'] <= 90]

            if not df_huyet_ap_thap.empty:
                huyet_ap_thap = True
                min_bp_time = df_huyet_ap_thap['charttime'].min()
                first_time2 = min(first_time2, min_bp_time)
        
        trieu_chung = sot or onlanh or huyet_ap_thap

        # Hai mau cay duong tinh
        cay_mau_duong_tinh_hop_le = False
        required_cols = ['spec_type_desc', 'org_name', 'charttime']
        if not cay_mau.empty and all(col in cay_mau.columns for col in required_cols):
            df_skin = cay_mau[
                (cay_mau['spec_type_desc'] == "BLOOD CULTURE") & 
                (cay_mau['org_name'].str.upper().isin(skin_contaminants))
            ].copy()

            df_skin = df_skin.sort_values(by=['org_name', 'charttime'])

            records = list(df_skin.itertuples(index=True, name='Pandas'))

            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    sample_i = records[i]
                    sample_j = records[j]
                    
                    # Nếu cùng loại vi khuẩn
                    if sample_i.org_name == sample_j.org_name:
                        time_diff = sample_j.charttime - sample_i.charttime
                        
                        if pd.Timedelta(0) <= time_diff <= pd.Timedelta(hours=48):
                            cay_mau_duong_tinh_hop_le = True
                            first_time2 = min(first_time2, sample_i.charttime)
                            break
                if cay_mau_duong_tinh_hop_le:
                    break

        # Thời gian đặt catheter TMTT
        thoi_gian_dat_catheter = False
        for sample in catheter.itertuples(index=True, name='Pandas'):
            if first_time2 != FAR_FUTURE and first_time2 - sample.starttime >= pd.Timedelta(days=2):
                thoi_gian_dat_catheter = True
                break
        
        r2 = trieu_chung and thoi_gian_dat_catheter and cay_mau_duong_tinh_hop_le

        final_clabsi = r1 or r2

        if self.verbose:
            print("\n===== FINAL CLABSI =====")
            print("rule1:", r1)
            print("rule2:", r2)
            print("=> FINAL CLABSI:", final_clabsi)

        return {
            "rule1": r1,
            "rule2": r2,
            "final_clabsi": final_clabsi
        }

    
        
    def check_cauti_subject(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        FAR_FUTURE = pd.Timestamp('2001-01-01')
        first_time1 = FAR_FUTURE
        first_time2 = FAR_FUTURE

        # =========== RULE 1 ===========
        # Sốt
        sot = False
        nhiet_do = self.get_data_with_48h_rule("Nhiệt độ", subject_id=subject_id, stay_id=stay_id, in_time=in_time)
        if not nhiet_do.empty:
            mask = ((nhiet_do['itemid'] == 223761) & (nhiet_do['valuenum'] > 100.4)) | \
                   ((nhiet_do['itemid'] == 223762) & (nhiet_do['valuenum'] > 38))
            df_sot = nhiet_do[mask]
            if not df_sot.empty:
                sot = True
                first_time1 = min(first_time1, df_sot['charttime'].min()) 

        # Tiểu gấp
        tieugap = False
        tieu_gap = self.get_data_with_48h_rule("Tiểu gấp", subject_id=subject_id, out_time=out_time, in_time=in_time)
        if not tieu_gap.empty:
            tieugap = True
            first_time1 = min(first_time1, tieu_gap['charttime'].min())

        # Tiểu nhiều
        tieunhieu = False
        tieu_nhieu = self.get_data_with_48h_rule("Tiểu nhiều lần", subject_id=subject_id, out_time=out_time, in_time=in_time)
        if not tieu_nhieu.empty:
            tieunhieu = True
            first_time1 = min(first_time1, tieu_nhieu['charttime'].min())
                
        # Tiểu buốt
        tieubuot = False
        tieu_buot = self.get_data_with_48h_rule("Tiểu buốt", subject_id=subject_id, in_time=in_time, out_time=out_time)
        if not tieu_buot.empty:
            tieubuot = True
            first_time1 = min(first_time1, tieu_buot['charttime'].min())
        
        # Đau hông sườn
        dauhongsuon = False
        dau_hong_suon = self.get_data_with_48h_rule("Đau hông sườn", subject_id=subject_id, in_time=in_time, out_time=out_time)
        if not dau_hong_suon.empty:
            dauhongsuon = True
            first_time1 = min(first_time1, dau_hong_suon['charttime'].min())

        # Đau xương mu
        dauxuongmu = False
        dau_xuong_mu = self.get_data_with_48h_rule("Đau/ấn đau vùng trên xương mu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        if not dau_xuong_mu.empty:
            dauxuongmu = True
            first_time1 = min(first_time1, dau_xuong_mu['charttime'].min())

        # Dùng hàm any() thay vì phép cộng (+) cho biến boolean
        trieuchung1 = any([sot, tieugap, tieubuot, tieunhieu, dauhongsuon, dauxuongmu])
        trieuchung1_mintime = first_time1

        # Cay nuoc tieu
        cay_nuoc_tieu_duong_tinh = False
        cay_nuoc_tieu = self.get_data_with_48h_rule("Cấy nước tiểu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        
        if not cay_nuoc_tieu.empty and 'charttime' in cay_nuoc_tieu.columns and 'quantity' in cay_nuoc_tieu.columns:
            # Điều kiện 1: quantity > 100,000 và 10^4-10^5
            mask_quantity = cay_nuoc_tieu['quantity'].astype(str).str.strip().str.contains("100,000", case=False, na=False)
            
            # Khởi tạo df_cay_valid mặc định
            df_cay_valid = pd.DataFrame()
            
            if 'comments' in cay_nuoc_tieu.columns:
                mask_comments = cay_nuoc_tieu['comments'].isna()
                df_cay_valid = cay_nuoc_tieu[mask_quantity & mask_comments]
            else:
                df_cay_valid = cay_nuoc_tieu[mask_quantity]
            
            if not df_cay_valid.empty:
                cay_nuoc_tieu_duong_tinh = True
                first_time1 = min(first_time1, df_cay_valid['charttime'].min())
            
        thoi_gian_dat_ong_thong_tieu = False
        ong_thong_tieu = self._fetch_and_cache_raw_data("Thời gian đặt ống thông tiểu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        
        if first_time1 != FAR_FUTURE and not ong_thong_tieu.empty and 'starttime' in ong_thong_tieu.columns:
            # Tạo mặt nạ kiểm tra: Thời gian đầu tiên xuất hiện triệu chứng - thời gian bắt đầu đặt >= 2 ngày
            mask_ong_thong = (first_time1 - ong_thong_tieu['starttime']) >= pd.Timedelta(days=2)
            # Nếu có BẤT KỲ dòng nào thỏa mãn (any) -> True
            if mask_ong_thong.any():
                thoi_gian_dat_ong_thong_tieu = True
        
        r1 = trieuchung1 and cay_nuoc_tieu_duong_tinh and thoi_gian_dat_ong_thong_tieu

        # =========== RULE 2 ===========

        # ---------------------------------------------------------
        # 1. TRIỆU CHỨNG LÂM SÀNG (trieuchung1)
        # ---------------------------------------------------------
        if (sot + tieugap + tieubuot + tieunhieu + dauhongsuon + dauxuongmu) >= 2:
            trieuchung1 = True
        first_time2 = trieuchung1_mintime

        # ---------------------------------------------------------
        # 2. XÉT NGHIỆM / CẬN LÂM SÀNG (trieuchung2)
        # ---------------------------------------------------------
        
        # Cấy nước tiểu (Tối ưu bằng .duplicated thay vì vòng lặp set)
        cay_nuoc_tieu_duong_tinh = False
        cay_nuoc_tieu = self.get_data_with_48h_rule("Cấy nước tiểu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        if not cay_nuoc_tieu.empty and 'charttime' in cay_nuoc_tieu.columns and 'org_name' in cay_nuoc_tieu.columns:
            # Lọc bỏ các dòng NaN
            df_cay_valid = cay_nuoc_tieu[cay_nuoc_tieu['org_name'].notna()].sort_values('charttime')
            # Lọc ra các dòng bị lặp lại tên vi khuẩn (xuất hiện >= 2 lần)
            dups = df_cay_valid[df_cay_valid.duplicated(subset=['org_name'], keep='first')]
            if not dups.empty:
                cay_nuoc_tieu_duong_tinh = True
                first_time2 = min(first_time2, dups['charttime'].min())

        # Bạch cầu niệu
        bachcaunieu = False
        bach_cau_nieu = self.get_data_with_48h_rule("Bạch cầu niệu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        if not bach_cau_nieu.empty and 'valuenum' in bach_cau_nieu.columns:
            df_bcn = bach_cau_nieu[bach_cau_nieu['valuenum'] > 5]
            if not df_bcn.empty:
                bachcaunieu = True
                first_time2 = min(first_time2, df_bcn['charttime'].min())
        
        # Mủ niệu
        munieu = False
        mu_nieu = self.get_data_with_48h_rule("Mủ niệu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        if not mu_nieu.empty and 'valuenum' in mu_nieu.columns:
            df_mn = mu_nieu[mu_nieu['valuenum'] > 5]
            if not df_mn.empty:
                munieu = True
                first_time2 = min(first_time2, df_mn['charttime'].min())

        # Nitrat niệu (Tối ưu bằng xử lý chuỗi .str)
        nitratnieu = False
        nitrat_nieu = self.get_data_with_48h_rule("Nitrat niệu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        if not nitrat_nieu.empty and 'value' in nitrat_nieu.columns:
            positive_values = ['POS', 'POSITIVE', 'P']
            # Chuyển thành text -> Cắt khoảng trắng -> In hoa -> Kiểm tra xem có nằm trong list dương tính không
            mask_nitrat = nitrat_nieu['value'].astype(str).str.strip().str.upper().isin(positive_values)
            df_nitrat = nitrat_nieu[mask_nitrat]
            if not df_nitrat.empty:
                nitratnieu = True
                first_time2 = min(first_time2, df_nitrat['charttime'].min())

        trieuchung2 = any([bachcaunieu, munieu, cay_nuoc_tieu_duong_tinh, nitratnieu])
        
        # ---------------------------------------------------------
        # 3. ĐIỀU KIỆN ỐNG THÔNG TIỂU
        # ---------------------------------------------------------
        thoi_gian_dat_ong_thong_tieu = False
        ong_thong_tieu = self._fetch_and_cache_raw_data("Thời gian đặt ống thông tiểu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        
        if first_time2 != FAR_FUTURE and not ong_thong_tieu.empty and 'starttime' in ong_thong_tieu.columns:
            # Tạo mặt nạ kiểm tra: Thời gian đầu tiên xuất hiện triệu chứng - thời gian bắt đầu đặt >= 2 ngày
            mask_ong_thong = (first_time2 - ong_thong_tieu['starttime']) >= pd.Timedelta(days=2)
            # Nếu có BẤT KỲ dòng nào thỏa mãn (any) -> True
            if mask_ong_thong.any():
                thoi_gian_dat_ong_thong_tieu = True
        
        # =========================================================
        # TỔNG HỢP FINAL
        # =========================================================
        r2 = trieuchung1 and trieuchung2 and thoi_gian_dat_ong_thong_tieu
        final_cauti = r1 or r2

        if self.verbose:
            print("\n===== FINAL CAUTI =====")
            print("rule1:", r1)
            print("rule2:", r2)
            print("=> FINAL CAUTI:", final_cauti)

        return {
            "rule1": r1,
            "rule2": r2,
            "final_cauti": final_cauti
        }

    def get_vap_features(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        vap_criteria = [
            "Nhiệt độ",
            "Ho",
            "Đờm mủ",
            "Nhịp thở",
            "SpO2",
            "Cần hỗ trợ oxy",
            "FiO2",
            "Rale",
            "Tiếng thở phế quản",
            "Cấy dịch đường hô hấp",
            "Cấy máu",
        ]

        vap_criteria_order = vap_criteria + [
            "Thở máy",
            "Imaging VAP (X-quang/CT)",
        ]
        vap_flat_rows = []
        for criteria in vap_criteria:
            try:
                df = self._fetch_and_cache_raw_data(criteria, subject_id=subject_id, stay_id=stay_id, in_time=in_time, out_time=out_time)
            except Exception as e:
                print(f"Lỗi VAP {criteria} | subject_id={subject_id}, stay_id={stay_id}: {e}")
                continue

            if df.empty:
                continue

            for _, row in df.iterrows():
                charttime = process_time_without_year(get_row_time(row))
                value = get_row_value(row)

                if pd.isna(charttime):
                    continue

                if charttime < in_time:
                    continue

                if pd.notna(out_time) and charttime > out_time:
                    continue

                vap_flat_rows.append({
                    "subject_id": subject_id,
                    "stay_id": stay_id,
                    "day_date": charttime.normalize(),
                    "criteria": criteria,
                    "charttime": charttime,
                    "value": value,
                })

        # =====================================================
        # 4.2 VAP mechanical ventilation
        # =====================================================
        try:
            tho_may = self._fetch_and_cache_raw_data(
                variable_name="Thời gian thở máy",
                subject_id=subject_id, stay_id=stay_id, in_time=in_time, out_time=out_time
            )
        except Exception as e:
            print(f"Lỗi Thời gian thở máy | subject_id={subject_id}, stay_id={stay_id}: {e}")
            tho_may = pd.DataFrame()

        if not tho_may.empty:
            for _, row in tho_may.iterrows():
                starttime = process_time_without_year(row["starttime"]) if "starttime" in row.index else pd.NaT
                endtime = process_time_without_year(row["endtime"]) if "endtime" in row.index else pd.NaT

                if pd.isna(starttime):
                    continue

                if pd.isna(endtime):
                    endtime = out_time

                if pd.isna(endtime):
                    continue

                vent_start = max(starttime, in_time)
                vent_end = min(endtime, out_time) if pd.notna(out_time) else endtime

                if vent_end < vent_start:
                    continue

                current_day = vent_start.normalize()
                last_day = vent_end.normalize()

                while current_day <= last_day:
                    day_start = current_day
                    day_end = current_day + pd.Timedelta(days=1)

                    display_starttime = max(vent_start, day_start)

                    if display_starttime < day_end and vent_end >= day_start:
                        vap_flat_rows.append({
                            "subject_id": subject_id,
                            "stay_id": stay_id,
                            "day_date": current_day,
                            "criteria": "Thở máy",
                            "charttime": display_starttime,
                            "value": f"starttime: {display_starttime} - endtime: {vent_end}",
                        })

                    current_day += pd.Timedelta(days=1)

        # =====================================================
        # 4.3 VAP imaging
        # =====================================================
        try:
            imaging_positive = self._get_vap_imaging_positive(
                subject_id=subject_id,
                stay_id=stay_id,
                in_time=in_time,
                out_time=out_time,
            )
        except TypeError:
            imaging_positive = self._fetch_and_cache_raw_data(
                subject_id=subject_id,
                stay_id=stay_id,
                in_time=in_time,
                out_time=out_time
            )
        except Exception as e:
            print(f"Lỗi Imaging VAP | subject_id={subject_id}, stay_id={stay_id}: {e}")
            imaging_positive = None

        vap_flat_rows.append({
            "subject_id": subject_id,
            "stay_id": stay_id,
            "day_date": in_time.normalize(),
            "criteria": "Imaging VAP (X-quang/CT)",
            "charttime": in_time,
            "value": imaging_positive,
        })

        df_vap_detail = pd.DataFrame(
            vap_flat_rows,
            columns=["subject_id", "stay_id", "day_date", "criteria", "charttime", "value"],
        )
        vap_json = build_infection_json(
            df_detail=df_vap_detail,
            criteria_order=vap_criteria_order,
        )
        return vap_json


    def get_clabsi_features(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        clabsi_criteria_stayid = [
            "Nhiệt độ", #stayid
            "Ớn lạnh", #stayid
            "Cấy máu" #stayid
        ]
        clabsi_criteria_non_stayid = ["Huyết áp tâm thu"]
        criteria_order = clabsi_criteria_stayid + clabsi_criteria_non_stayid + [
            "Thời gian đặt catheter TMTT",
        ]

        flat_rows = []
        for criteria in clabsi_criteria_stayid:
            try:
                df = self._fetch_and_cache_raw_data(criteria, subject_id=subject_id, stay_id=stay_id, in_time=in_time)
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

        for criteria in clabsi_criteria_non_stayid:
            try:
                df = self._fetch_and_cache_raw_data(criteria, subject_id=subject_id, out_time=out_time, in_time=in_time)
            except Exception as e:
                print(f"Lỗi khi lấy {criteria} cho subject_id={subject_id}")
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
            catheter = self._fetch_and_cache_raw_data("Thời gian đặt catheter TMTT", subject_id=subject_id, out_time=out_time, in_time=in_time)
        except Exception as e:
            print(
                f"Lỗi khi lấy Thời gian đặt catheter TMTT "
                f"cho subject_id={subject_id}"
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

        df_clabsi_detail = pd.DataFrame(
            flat_rows,
            columns=["subject_id", "stay_id", "day_date", "criteria", "charttime", "value"],
        )

        clabsi_json = build_infection_json(
            df_detail=df_clabsi_detail,
            criteria_order=criteria_order,
        )
        return clabsi_json


    def get_cauti_features(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        cauti_criteria_non_stay_id = [
            "Tiểu gấp",
            "Tiểu nhiều lần",
            "Tiểu buốt",
            "Đau hông sườn",
            "Đau/ấn đau vùng trên xương mu",
            "Cấy nước tiểu",
            "Bạch cầu niệu",
            "Mủ niệu",
            "Nitrat niệu",
        ]
        cauti_criteria_stay_id = ["Nhiệt độ"]
        criteria_order = cauti_criteria_non_stay_id + cauti_criteria_stay_id + [
            "Thời gian đặt ống thông tiểu"
        ]

        flat_rows = []
        for criteria in cauti_criteria_non_stay_id:
            try:
                df = self._fetch_and_cache_raw_data(criteria, subject_id=subject_id, in_time=in_time, out_time=out_time)
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
        
        for criteria in cauti_criteria_stay_id:
            try:
                df = self._fetch_and_cache_raw_data(criteria, subject_id=subject_id, stay_id=stay_id, in_time=in_time)
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
            catheter = self._fetch_and_cache_raw_data("Thời gian đặt ống thông tiểu", subject_id=subject_id, in_time=in_time, out_time=out_time)
        except Exception as e:
            print(
                f"Lỗi khi lấy Thời gian đặt ống thông tiểu "
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
                            "criteria": "Thời gian đặt ống thông tiểu",
                            "charttime": display_starttime,
                            "value": f"starttime: {display_starttime} - endtime: {cath_end}",
                        })

                    current_day = current_day + pd.Timedelta(days=1)

        df_cauti_detail = pd.DataFrame(
            flat_rows,
            columns=["subject_id", "stay_id", "day_date", "criteria", "charttime", "value"],
        )

        cauti_json = build_infection_json(
            df_detail=df_cauti_detail,
            criteria_order=criteria_order,
        )
        return cauti_json