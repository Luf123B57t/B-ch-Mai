import pandas as pd
import numpy as np
from features_mapping import ClinicalDataExtractor
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

class InfectionChecker:
    """Class quản lý việc kiểm tra các loại nhiễm trùng (HAIs) của bệnh nhân."""
    
    def __init__(self,extractor: ClinicalDataExtractor, verbose: bool = True):
        self.extractor = extractor
        self.verbose = verbose
        MODEL_DIR = "./Bert"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(self.device)

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
        spo2 = self.extractor.get_variable_data(
            variable_name="SpO2",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year
        ).copy()

        if spo2.empty:
            return False, None, pd.DataFrame()

        spo2["charttime"] = process_time_without_year(spo2["charttime"])
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
        nhiet_do = self.extractor.get_variable_data(
            variable_name="Nhiệt độ",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year
        )
        for temp in nhiet_do.itertuples(index=True, name="Pandas"):
            if (temp.itemid == 223761 and temp.valuenum > 100.4) or (temp.itemid == 223762 and temp.valuenum > 38):
                sot = True
                first_time = min(first_time, temp.charttime)

        # -------- Ho --------
        ho = self.extractor.get_variable_data(
            variable_name="Ho",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year
        )
        for sample in ho.itertuples(index=True, name="Pandas"):
            val = str(sample.value).strip().lower()
            if sample.itemid == 223991 and val in ["weak", "strong"]:
                ho_flag = True
                first_time = min(first_time, sample.charttime)

        # -------- Đờm mủ --------
        dom_mu = self.extractor.get_variable_data(
            variable_name="Đờm mủ",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year
        )
        for sample in dom_mu.itertuples(index=True, name="Pandas"):
            val = str(sample.value).strip().lower()
            if val != "clear":
                dom_mu_flag = True
                first_time = min(first_time, sample.charttime)

        # -------- Thở nhanh --------
        nhip_tho = self.extractor.get_variable_data(
            variable_name="Nhịp thở",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year
        )
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

        rale = self.extractor.get_variable_data(
            variable_name="Rale",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year
        )
        for sample in rale.itertuples(index=True, name="Pandas"):
            val = str(sample.value).strip().lower()
            if val in lung_keywords:
                lung_sound_flag = True
                first_time = min(first_time, sample.charttime)

        bronchial = self.extractor.get_variable_data(
            variable_name="Tiếng thở phế quản",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year
        )
        for sample in bronchial.itertuples(index=True, name="Pandas"):
            val = str(sample.value).strip().lower()
            if val == "bronchial":
                lung_sound_flag = True
                first_time = min(first_time, sample.charttime)

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

    def _get_vap_imaging_positive(self, subject_id: int, in_time:pd.Timestamp, out_time: pd.Timestamp):
        """
        Kiểm tra imaging gợi ý viêm phổi từ:
        - X-quang ngực
        - CT scan lồng ngực

        Với schema radiology mới:
        ['note_id', 'subject_id', 'hadm_id', 'note_type', 'note_seq',
        'charttime', 'storetime', 'text']
        """
        imaging_positive = False

        xray = self.extractor.get_variable_data(
            variable_name="X-quang ngực",
            subject_id=subject_id,
            in_time=in_time,
            time_process_func=process_time_without_year,
            out_time=out_time
        )

        ct = self.extractor.get_variable_data(
            variable_name="CT scan lồng ngực",
            subject_id=subject_id,
            in_time=in_time,
            time_process_func=process_time_without_year,
            out_time=out_time
        )

        THRESHOLD = 0.35 
        MAX_LENGTH = 256

        text_list = []
        
        for df_img in [xray, ct]:
            if df_img.empty:
                continue

            if "text" not in df_img.columns:
                continue

            for sample in df_img.itertuples(index=True, name="Pandas"):
                text = str(sample.text).lower()
                text_list.append(text)

        
        self.model.eval()

        def predict_batch(texts, batch_size=4):
            probs_all = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
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

        if len(preds) > 0 and preds.max() == 1: # Có ít nhất 1 dự đoán là 1
            imaging_positive = True

        return imaging_positive


    def _get_vap_micro_positive(self, subject_id: int, in_time:pd.Timestamp):
        """
        Rule microbiology:
        Cấy dịch đường hô hấp HOẶC cấy máu
        """
        micro_positive = False

        cay_dich = self.extractor.get_variable_data(
            variable_name="Cấy dịch đường hô hấp",
            subject_id=subject_id,
            in_time=in_time,
            time_process_func=process_time_without_year
        )
        for sample in cay_dich.itertuples(index=True, name="Pandas"):
            if pd.notna(sample.org_name):
                micro_positive = True
                break

        if not micro_positive:
            cay_mau = self.extractor.get_variable_data(
                variable_name="Cấy máu",
                subject_id=subject_id,
                in_time=in_time,
                time_process_func=process_time_without_year
            )
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

        tho_may = self.extractor.get_variable_data(
            variable_name="Thời gian thở máy",
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time,
            time_process_func=process_time_without_year,
            check_48h=False
        )

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

        imaging_positive = self._get_vap_imaging_positive(subject_id, in_time=in_time, out_time=out_time)
        micro_positive = self._get_vap_micro_positive(subject_id, in_time)

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

        # if self.verbose:
        #     print("\n===== VAP RULE 1 =====")
        #     print("symptom_flags:", symptom_flags)
        #     print("symptom_count:", symptom_count)
        #     print("first_time:", first_time)
        #     print("imaging_positive:", imaging_positive)
        #     print("micro_positive:", micro_positive)
        #     print("ventilation_ok:", ventilation_ok)
        #     print("\nSpO2 debug:")
        #     print(spo2_debug)
        #     print("=> VAP RULE 1:", vap_rule1)

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

        imaging_positive = self._get_vap_imaging_positive(subject_id, in_time=in_time, out_time=out_time)

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

        # if self.verbose:
        #     print("\n===== VAP RULE 3 =====")
        #     print("symptom_flags:", symptom_flags)
        #     print("symptom_count:", symptom_count)
        #     print("first_time:", first_time)
        #     print("imaging_positive:", imaging_positive)
        #     print("ventilation_ok:", ventilation_ok)
        #     print("\nSpO2 debug:")
        #     print(spo2_debug)
        #     print("=> VAP RULE 3:", vap_rule3)

        return vap_rule3

    # =========================================================
    # HÀM TỔNG HỢP
    # =========================================================
    def check_vap_subject(self, subject_id: int, stay_id: int, in_time: pd.Timestamp, out_time: pd.Timestamp):
        """
        Trả về dict tổng hợp cả 3 rule
        """
        r1 = self.check_vap_rule1(subject_id, stay_id, in_time = in_time, out_time = out_time)
        r2 = self.check_vap_rule2(subject_id, stay_id, in_time = in_time, out_time = out_time)
        r3 = self.check_vap_rule3(subject_id, stay_id, in_time = in_time, out_time = out_time)

        final_vap = r1 or r2 or r3

        if self.verbose:
            print("\n===== FINAL VAP =====")
            print("rule1:", r1)
            print("rule2:", r2)
            print("rule3:", r3)
            print("=> FINAL VAP:", final_vap)

        return {
            "rule1": r1,
            "rule2": r2,
            "rule3": r3,
            "final_vap": final_vap
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
            'VIRIDANS GROUP STREPTOCOCCI'
        ]

        # =========== RULE 1 ===========

        # Cấy máu dương tính với tác nhân gây bệnh
        cay_mau_duong_tinh = False
        cay_mau = self.extractor.get_variable_data(
            variable_name="Cấy máu", 
            subject_id=subject_id,
            stay_id = stay_id,
            in_time= in_time,
            time_process_func= process_time_without_year
        )
        for sample in cay_mau.itertuples(index=True, name='Pandas'):

            if sample.spec_type_desc == "BLOOD CULTURE" and pd.notna(sample.org_name) and sample.org_name.upper() not in skin_contaminants:
                cay_mau_duong_tinh = True
                first_time1 = min(first_time1, sample.charttime)
        
        # Thời gian đặt catheter TMTT
        thoi_gian_dat_catheter = False
        catheter = self.extractor.get_variable_data(
            variable_name="Thời gian đặt catheter TMTT",
            subject_id = subject_id,
            time_process_func=process_time_without_year,
            in_time=in_time, 
            out_time=out_time,
            check_48h = False
        )
        for sample in catheter.itertuples(index=True, name='Pandas'):
            if first_time1 != FAR_FUTURE and first_time1 - sample.starttime >= pd.Timedelta(days=2):
                thoi_gian_dat_catheter = True
                break
        
        r1 = cay_mau_duong_tinh and thoi_gian_dat_catheter

        # =========== RULE 2 ===========

        # Sot
        sot = False
        nhiet_do = self.extractor.get_variable_data(
            variable_name="Nhiệt độ", 
            subject_id=subject_id,
            stay_id = stay_id,
            in_time= in_time,
            time_process_func= process_time_without_year
        )
        for temp in nhiet_do.itertuples(index=True, name='Pandas'):
            if (temp.itemid == 223761 and temp.valuenum > 100.4) or (temp.itemid == 223762 and temp.valuenum > 38):
                first_time2 = min(first_time2, temp.charttime)
                sot = True
                
        # On lanh
        onlanh = False
        on_lanh = self.extractor.get_variable_data(
            variable_name="Ớn lạnh", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time=out_time
        )
        for sample in on_lanh.itertuples(index=True, name='Pandas'):
            first_time2 = min(first_time2, sample.charttime)
            onlanh = True
        
        # Huyết áp thấp
        huyet_ap_thap = False
        huyet_ap = self.extractor.get_variable_data(
            variable_name="Huyết áp", 
            subject_id=subject_id,
            stay_id = stay_id,
            in_time= in_time,
            time_process_func= process_time_without_year
        )
        for blood_pres in huyet_ap.itertuples(index=True, name = 'Pandas'):
            if blood_pres.valuenum <= 90:
                huyet_ap_thap = True
                first_time2 = min(first_time2, blood_pres.charttime)
        
        trieu_chung = sot or onlanh or huyet_ap_thap

        
        # df_skin = cay_mau[
        #     (cay_mau['spec_type_desc'] == "BLOOD CULTURE") & 
        #     (cay_mau['org_name'].str.upper().isin(skin_contaminants))
        # ].copy()

        # df_skin = df_skin.sort_values(by=['org_name', 'charttime'])

        # cay_mau_duong_tinh_hop_le = False
        # records = list(df_skin.itertuples(index=True, name='Pandas'))

        # for i in range(len(records)):
        #     for j in range(i + 1, len(records)):
        #         sample_i = records[i]
        #         sample_j = records[j]
                
        #         # Nếu cùng loại vi khuẩn
        #         if sample_i.org_name == sample_j.org_name:
        #             time_diff = sample_j.charttime - sample_i.charttime
                    
        #             if pd.Timedelta(0) < time_diff <= pd.Timedelta(hours=48):
        #                 cay_mau_duong_tinh_hop_le = True
        #                 first_time2 = min(first_time2, sample_i.charttime)
        #                 break
        #     if cay_mau_duong_tinh_hop_le:
        #         break
        cay_mau_duong_tinh_hop_le = False
        
        # Kiểm tra an toàn: Đảm bảo df không rỗng VÀ chứa đầy đủ các cột cần thiết
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
                        
                        if pd.Timedelta(0) < time_diff <= pd.Timedelta(hours=48):
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

        # =========== RULE 2 ===========

        # Sot
        sot = False
        nhiet_do = self.extractor.get_variable_data(
            variable_name="Nhiệt độ", 
            subject_id=subject_id,
            stay_id = stay_id,
            in_time= in_time,
            time_process_func= process_time_without_year
        )
        for temp in nhiet_do.itertuples(index=True, name='Pandas'):
            if (temp.itemid == 223761 and temp.valuenum > 100.4) or (temp.itemid == 223762 and temp.valuenum > 38):
                first_time1 = min(first_time1, temp.charttime)
                sot = True

        # Tieu gap
        tieugap = False
        tieu_gap = self.extractor.get_variable_data(
            variable_name="Tiểu gấp", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time = out_time
        )
        for sample in tieu_gap.itertuples(index=True, name='Pandas'):
                first_time1 = min(first_time1, sample.charttime)
                tieugap = True

        # Tieu nhieu
        tieunhieu = False
        tieu_nhieu = self.extractor.get_variable_data(
            variable_name="Tiểu nhiều lần", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time = out_time
        )
        for sample in tieu_nhieu.itertuples(index=True, name='Pandas'):
                first_time1 = min(first_time1, sample.charttime)
                tieunhieu = True
        
        # Tieu buot
        tieubuot = False
        tieu_buot = self.extractor.get_variable_data(
            variable_name="Tiểu buốt", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time = out_time
        )
        for sample in tieu_buot.itertuples(index=True, name='Pandas'):
                first_time1 = min(first_time1, sample.charttime)
                tieubuot = True
        
        # Dau hong suon
        dauhongsuon = False
        dau_hong_suon = self.extractor.get_variable_data(
            variable_name="Đau hông sườn", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time = out_time
        )
        for sample in dau_hong_suon.itertuples(index=True, name='Pandas'):
                first_time1 = min(first_time1, sample.charttime)
                dauhongsuon = True

        # Dau xuong mu
        dauxuongmu = False
        dau_xuong_mu = self.extractor.get_variable_data(
            variable_name="Đau/ấn đau vùng trên xương mu", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time = out_time
        )
        for sample in dau_xuong_mu.itertuples(index=True, name='Pandas'):
                first_time1 = min(first_time1, sample.charttime)
                dauxuongmu = True

        trieuchung1 = False
        if (sot + tieugap + tieubuot + tieunhieu + dauhongsuon + dauxuongmu) >= 2:
            trieuchung1 = True
        # Trieu chung 2
    
        # Cay nuoc tieu
        cay_nuoc_tieu_duong_tinh = False
        cay_nuoc_tieu = self.extractor.get_variable_data(
            variable_name="Cấy nước tiểu", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time = out_time
        )
        if not cay_nuoc_tieu.empty and 'charttime' in cay_nuoc_tieu.columns:
            cay_nuoc_tieu = cay_nuoc_tieu.sort_values('charttime', ascending=True)
            
            loai_vi_sinh_vat = set()
            for sample in cay_nuoc_tieu.itertuples():
                if pd.notna(sample.org_name):
                    if sample.org_name in loai_vi_sinh_vat:
                        cay_nuoc_tieu_duong_tinh = True
                        first_time1 = min(first_time1, sample.charttime)
                    else:
                        loai_vi_sinh_vat.add(sample.org_name)
            

        # Bach cau nieu
        bachcaunieu = False
        bach_cau_nieu = self.extractor.get_variable_data(
            variable_name="Bạch cầu niệu", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time= out_time
        )
        for sample in bach_cau_nieu.itertuples():
            if sample.valuenum > 5:
                bachcaunieu = True
                first_time1 = min(first_time1, sample.charttime)
        
        # Mu nieu
        munieu = False
        mu_nieu = self.extractor.get_variable_data(
            variable_name="Mủ niệu", 
            subject_id=subject_id,
            in_time= in_time,
            time_process_func= process_time_without_year,
            out_time= out_time
        )
        for sample in mu_nieu.itertuples():
            if sample.valuenum > 5:
                munieu = True
                first_time1 = min(first_time1, sample.charttime)

        # Nitrat nieu
        positive_values = ['POS', 'POSITIVE', 'P']
        nitratnieu = False
        nitrat_nieu = self.extractor.get_variable_data(
            variable_name="Nitrat niệu", 
            subject_id=subject_id,
            in_time= in_time, 
            out_time= out_time,
            time_process_func= process_time_without_year
        )
        for sample in nitrat_nieu.itertuples():
            val_str = str(sample.value).strip().upper()
            if val_str in positive_values:
                nitratnieu = True
                first_time1 = min(first_time1, sample.charttime)

        trieuchung2 = bachcaunieu or munieu or cay_nuoc_tieu_duong_tinh or nitratnieu
        
        thoi_gian_dat_ong_thong_tieu = False
        ong_thong_tieu = self.extractor.get_variable_data(
            variable_name="Thời gian đặt ống thông tiểu",
            subject_id = subject_id,
            in_time= in_time,
            out_time= out_time,
            time_process_func= process_time_without_year,
            check_48h = False
        )
        for sample in ong_thong_tieu.itertuples(index=True, name='Pandas'):
            if first_time1 != FAR_FUTURE and first_time1 - sample.starttime >= pd.Timedelta(days=2):
                thoi_gian_dat_ong_thong_tieu = True
                break
        
        r1 = False
        r2 =  trieuchung1 and trieuchung2 and thoi_gian_dat_ong_thong_tieu
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






