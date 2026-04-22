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

class InfectionChecker:
    """Class quản lý việc kiểm tra các loại nhiễm trùng (HAIs) của bệnh nhân."""
    
    def __init__(self,extractor: ClinicalDataExtractor, verbose: bool = True):
        self.extractor = extractor
        self.verbose = verbose

    def check_vap(self, subject_id: int, stay_id: int):
        # Sử dụng self.extractor ở đây
        pass

    def check_clabsi_subject(self, subject_id: int, stay_id: int, intime: object, outtime: object):
        in_time = process_time_without_year(intime)
        out_time = process_time_without_year(outtime)

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
            if first_time1 != FAR_FUTURE and first_time1 - sample.charttime >= pd.Timedelta(days=2):
                thoi_gian_dat_catheter = True
                break
        
        if cay_mau_duong_tinh and thoi_gian_dat_catheter:
            return True

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

        
        df_skin = cay_mau[
            (cay_mau['spec_type_desc'] == "BLOOD CULTURE") & 
            (cay_mau['org_name'].str.upper().isin(skin_contaminants))
        ].copy()

        df_skin = df_skin.sort_values(by=['org_name', 'charttime'])

        cay_mau_duong_tinh_hop_le = False
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
            if first_time2 != FAR_FUTURE and first_time2 - sample.charttime >= pd.Timedelta(days=2):
                thoi_gian_dat_catheter = True
                break

        return trieu_chung and thoi_gian_dat_catheter and cay_mau_duong_tinh_hop_le


    def check_cauti_subject(self, subject_id: int, stay_id: int, intime: object, outtime: object):
        in_time = process_time_without_year(intime)
        out_time = process_time_without_year(outtime)

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

        if not trieuchung1:
            return False
    
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
        if not trieuchung2:
            return False
        
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
            if first_time1 != FAR_FUTURE and first_time1 - sample.charttime >= pd.Timedelta(days=2):
                thoi_gian_dat_ong_thong_tieu = True
                break

        return trieuchung1 and trieuchung2 and thoi_gian_dat_ong_thong_tieu






