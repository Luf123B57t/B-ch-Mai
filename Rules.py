import pandas as pd
import numpy as np
from features_mapping import ClinicalDataExtractor


def process_time_without_year(date_input):
    dt = pd.to_datetime(date_input, errors='coerce')

    if isinstance(dt, pd.Series):
        return dt.apply(lambda x: x.replace(year=2000) if pd.notnull(x) else x)
    else:
        if pd.notnull(dt):
            return dt.replace(year=2000)
        return dt


def is_time_between(charttime, intime, outtime):
    t_check = process_time_without_year(charttime)
    t_in = process_time_without_year(intime)
    t_out = process_time_without_year(outtime)

    return (t_check >= t_in) & (t_check <= t_out)


class InfectionChecker:
    """Class quản lý việc kiểm tra các loại nhiễm trùng (HAIs) của bệnh nhân."""

    def __init__(self, icu_stays, extractor: ClinicalDataExtractor, verbose: bool = True):
        self.extractor = extractor
        self.verbose = verbose
        self.icu_stay = icu_stays

    # =========================================================
    # HELPERS
    # =========================================================

    def _get_stay_times(self, stay_id: int):
        stay_info = self.icu_stay[self.icu_stay["stay_id"] == stay_id]
        if stay_info.empty:
            return None, None

        in_time = process_time_without_year(stay_info["intime"]).iloc[0]
        out_time = process_time_without_year(stay_info["outtime"]).iloc[0]
        return in_time, out_time

    def _check_spo2_worsening(self, subject_id, stay_id, in_time):
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

    def _get_vap_symptoms(self, subject_id: int, stay_id: int, in_time):
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

    def _get_vap_imaging_positive(self, subject_id: int):
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
            in_time=None,
            time_process_func=None
        )

        ct = self.extractor.get_variable_data(
            variable_name="CT scan lồng ngực",
            subject_id=subject_id,
            in_time=None,
            time_process_func=None
        )

        pneumonia_keywords = [
            "pneumonia", "pna", "bronchopneumonia", "consolidation",
            "infiltrate", "infiltration", "opacity", "opacities",
            "ground glass", "ggo", "air bronchogram", "patchy",
            "cloudy", "hazy"
        ]

        for df_img in [xray, ct]:
            if df_img.empty:
                continue

            if "text" not in df_img.columns:
                continue

            for sample in df_img.itertuples(index=True, name="Pandas"):
                text = str(sample.text).lower()
                if any(k in text for k in pneumonia_keywords):
                    imaging_positive = True
                    break

            if imaging_positive:
                break

        return imaging_positive

    def _get_vap_micro_positive(self, subject_id: int, in_time):
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

    def _check_ventilation_before_first_time(self, subject_id: int, stay_id: int, in_time, first_time):
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
            time_process_func=process_time_without_year
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
    def check_vap_rule1(self, subject_id: int, stay_id: int):
        in_time, out_time = self._get_stay_times(stay_id)
        if in_time is None:
            return False

        symptom_flags, symptom_count, first_time, spo2_debug = self._get_vap_symptoms(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )

        imaging_positive = self._get_vap_imaging_positive(subject_id)
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

        if self.verbose:
            print("\n===== VAP RULE 1 =====")
            print("symptom_flags:", symptom_flags)
            print("symptom_count:", symptom_count)
            print("first_time:", first_time)
            print("imaging_positive:", imaging_positive)
            print("micro_positive:", micro_positive)
            print("ventilation_ok:", ventilation_ok)
            print("\nSpO2 debug:")
            print(spo2_debug)
            print("=> VAP RULE 1:", vap_rule1)

        return vap_rule1

    # =========================================================
    # VAP RULE 2
    # >= 3 triệu chứng
    # + vent >= 2 ngày trước first_time
    # =========================================================
    def check_vap_rule2(self, subject_id: int, stay_id: int):
        in_time, out_time = self._get_stay_times(stay_id)
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

        if self.verbose:
            print("\n===== VAP RULE 2 =====")
            print("symptom_flags:", symptom_flags)
            print("symptom_count:", symptom_count)
            print("first_time:", first_time)
            print("ventilation_ok:", ventilation_ok)
            print("\nSpO2 debug:")
            print(spo2_debug)
            print("=> VAP RULE 2:", vap_rule2)

        return vap_rule2

    # =========================================================
    # VAP RULE 3
    # >= 2 triệu chứng
    # + imaging
    # + vent >= 2 ngày trước first_time
    # =========================================================
    def check_vap_rule3(self, subject_id: int, stay_id: int):
        in_time, out_time = self._get_stay_times(stay_id)
        if in_time is None:
            return False

        symptom_flags, symptom_count, first_time, spo2_debug = self._get_vap_symptoms(
            subject_id=subject_id,
            stay_id=stay_id,
            in_time=in_time
        )

        imaging_positive = self._get_vap_imaging_positive(subject_id)

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

        if self.verbose:
            print("\n===== VAP RULE 3 =====")
            print("symptom_flags:", symptom_flags)
            print("symptom_count:", symptom_count)
            print("first_time:", first_time)
            print("imaging_positive:", imaging_positive)
            print("ventilation_ok:", ventilation_ok)
            print("\nSpO2 debug:")
            print(spo2_debug)
            print("=> VAP RULE 3:", vap_rule3)

        return vap_rule3

    # =========================================================
    # HÀM TỔNG HỢP
    # =========================================================
    def check_vap_subject(self, subject_id: int, stay_id: int):
        """
        Trả về dict tổng hợp cả 3 rule
        """
        r1 = self.check_vap_rule1(subject_id, stay_id)
        r2 = self.check_vap_rule2(subject_id, stay_id)
        r3 = self.check_vap_rule3(subject_id, stay_id)

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
    def check_vap_all_stays(self, subject_id: int):
        """
        Chạy VAP cho tất cả ICU stay của 1 subject.
        Trả về DataFrame:
        subject_id | stay_id | rule1 | rule2 | rule3 | final_vap
        """
        stays = self.icu_stay[self.icu_stay["subject_id"] == subject_id].copy()

        if stays.empty:
            return pd.DataFrame(columns=[
                "subject_id", "stay_id", "rule1", "rule2", "rule3", "final_vap"
            ])

        results = []

        for _, row in stays.iterrows():
            stay_id = row["stay_id"]

            vap_result = self.check_vap_subject(
                subject_id=subject_id,
                stay_id=stay_id
            )

            results.append({
                "subject_id": subject_id,
                "stay_id": stay_id,
                "rule1": vap_result["rule1"],
                "rule2": vap_result["rule2"],
                "rule3": vap_result["rule3"],
                "final_vap": vap_result["final_vap"]
            })

        return pd.DataFrame(results)