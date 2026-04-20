import pandas as pd
import numpy as np
from features_mapping import ClinicalDataExtractor

def process_time_without_year(date_series):
    """
    Converts an object/string series to datetime and neutralizes the year.
    This allows you to calculate time differences (months, days, hours, mins) 
    while completely ignoring the original year.
    
    Parameters:
    date_series (pd.Series or list): The input dates as strings/objects.
    
    Returns:
    pd.Series: Datetime objects with the year standardized to 2000.
    """
    # 1. Convert the object to a proper datetime type
    dt_series = pd.to_datetime(date_series, errors='coerce')
    
    # 2. "Remove" the year by setting every year to 2000
    # The lambda function checks if the value is not null before replacing
    dt_no_year = dt_series.apply(lambda x: x.replace(year=2000) if pd.notnull(x) else x)
    
    return dt_no_year

def is_time_between(charttime, intime, outtime):
    """
    Kiểm tra xem một mốc thời gian có nằm giữa intime và outtime không.
    Trả về True nếu: intime <= check_time <= outtime
    """
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

    def check_vap(self, subject_id: int, stay_id: int):
        # Sử dụng self.extractor ở đây
        pass

    def check_clabsi_subject(self, subject_id: int, stay_id: int):
        stay_info = self.icu_stay[self.icu_stay['stay_id'] == stay_id]
        in_time = stay_info['intime']
        out_time = stay_info['outtime']

        # Cấy máu dương tính với tác nhân gây bệnh
        cay_mau = self.extractor.get_variable_data(
            variable_name="Cấy máu", 
            subject_id=subject_id,
            stay_id = stay_id
        )
        


        # Sot
        sot = False
        nhiet_do = self.extractor.get_variable_data(
            variable_name="Nhiệt độ", 
            subject_id=subject_id,
            stay_id = stay_id
        )

        for temp in nhietdo.itertuples(index=True, name='Pandas'):
            if (temp.itemid == 223761 and temp.valuenum > 100.4) or (temp.itemid == 223762 and temp.valuenum > 38):
                sot = True
                break
                


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







