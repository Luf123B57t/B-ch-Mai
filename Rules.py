import pandas as pd
import numpy as np
from features_mapping import ClinicalDataExtractor

def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")

def _contains_any(series, keywords):
    pattern = "|".join(keywords)
    return series.astype(str).str.contains(pattern, case=False, na=False, regex=True)

class InfectionChecker:
    """Class quản lý việc kiểm tra các loại nhiễm trùng (HAIs) của bệnh nhân."""
    
    def __init__(self, icu_stays, extractor: ClinicalDataExtractor, verbose: bool = True):
        self.extractor = extractor
        self.verbose = verbose
        self.icu_stay = icu_stays

    def check_vap(self, subject_id: int, stay_id: int):
        # Sử dụng self.extractor ở đây
        pass

    def check_clabsi_subject(self, subject_id: int, stay_id: int):
        in_time = self.icu_stay[self.icu_stay['stay_id'] == stay_id]
        nhiet_do = self.extractor.get_variable_data(
            variable_name="Nhiệt độ", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        on_lanh = self.extractor.get_variable_data(
            variable_name="Ớn lạnh", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        huyet_ap = self.extractor.get_variable_data(
            variable_name="Huyết áp", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        cay_mau = self.extractor.get_variable_data(
            variable_name="Cấy máu", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        carthter = self.extractor.get_variable_data(
            variable_name="Thời gian đặt catheter TMTT", 
            subject_id=subject_id,
            stay_id = stay_id
        )

    def check_cauti_subject(self, subject_id: int, stay_id: int):
        nhiet_do = self.extractor.get_variable_data(
            variable_name="Nhiệt độ", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        tieu_gap = self.extractor.get_variable_data(
            variable_name="Tiểu gấp", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        tieu_nhieu = self.extractor.get_variable_data(
            variable_name="Tiểu nhiều lần", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        tieu_buot = self.extractor.get_variable_data(
            variable_name="Tiểu buốt", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        dau_hong_suon = self.extractor.get_variable_data(
            variable_name="Đau hông sườn", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        dau_xuong_mu = self.extractor.get_variable_data(
            variable_name="Đau/ấn đau vùng trên xương mu", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        cay_nuoc_tieu = self.extractor.get_variable_data(
            variable_name="Cấy nước tiểu", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        bach_cau_nieu = self.extractor.get_variable_data(
            variable_name="Bạch cầu niệu", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        mu_nieu = self.extractor.get_variable_data(
            variable_name="Mủ niệu", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        nitrat_nieu = self.extractor.get_variable_data(
            variable_name="Nitrat niệu", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        ong_thong_tieu = self.extractor.get_variable_data(
            variable_name="Thời gian đặt catheter TMTT", 
            subject_id=subject_id,
            stay_id = stay_id
        )







