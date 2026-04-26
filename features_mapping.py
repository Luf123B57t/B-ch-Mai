import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# 1. Định nghĩa Schema cho một luật Mapping
@dataclass
class MappingRule:
    variable_name: str
    source_table: str
    value_column: str
    time_column: Optional[str] = None
    lookup_column: Optional[str] = None
    lookup_value: Any = None
    lookup_operator: str = "=="
    secondary_lookup_column: Optional[str] = None
    secondary_lookup_value: Any = None
    endtime_column: Optional[str] = None
    subject_id_column: str = "subject_id"
    hadm_id_column: Optional[str] = "hadm_id"
    stay_id_column: Optional[str] = None
    note_id_column: Optional[str] = None

class ClinicalDataExtractor:
    """
    Class chuyên dụng để trích xuất dữ liệu lâm sàng dựa trên các luật mapping.
    """
    def __init__(self, data_tables: Dict[str, pd.DataFrame]):
        """
        Khởi tạo extractor với các bảng dữ liệu thực tế.
        :param data_tables: Dictionary chứa các Pandas DataFrames (VD: {"chartevents": df1, "labevents": df2})
        """
        self.tables = data_tables
        
        # Các từ khóa cố định
        self.keyword_map = {
            "Ớn lạnh": ["chills", "rigors", "shivering", "feeling cold", "cold"],
            "Tiểu gấp": ["urinary urgency", "urgency"],
            "Tiểu nhiều lần": ["urinary frequency", "frequent urination"],
            "Tiểu buốt": ["dysuria", "painful urination", "burning on urination"],
            "Đau hông sườn": ["flank pain", "cva tenderness"],
            "Đau/ấn đau vùng trên xương mu": ["suprapubic pain", "suprapubic tenderness"],
        }
        
        self.central_line_locations = [
            "Left IJ", "Right IJ", "Left Subclavian", "Right Subclavian",
            "Left Femoral", "Left Femoral.", "Right Femoral", "Right Femoral."
        ]
        
        self.mapping_rules = self._load_mapping_rules()

    def _load_mapping_rules(self) -> pd.DataFrame:
        """
        Định nghĩa và tải toàn bộ các luật mapping vào một DataFrame.
        """
        rules = [
            # ===== CHARTEVENTS =====
            MappingRule("Nhiệt độ", "chartevents", lookup_column="itemid", lookup_value=223761, value_column="valuenum", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Nhiệt độ", "chartevents", lookup_column="itemid", lookup_value=223762, value_column="valuenum", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Ho", "chartevents", lookup_column="itemid", lookup_value=223991, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Ho", "chartevents", lookup_column="itemid", lookup_value=223992, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Đờm mủ", "chartevents", lookup_column="itemid", lookup_value=224370, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Nhịp thở", "chartevents", lookup_column="itemid", lookup_value=220210, value_column="valuenum", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("SpO2", "chartevents", lookup_column="itemid", lookup_value=220277, value_column="valuenum", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Cần hỗ trợ oxy", "chartevents", lookup_column="itemid", lookup_value=226732, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("FiO2", "chartevents", lookup_column="itemid", lookup_value=223835, value_column="valuenum", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Rale", "chartevents", lookup_column="itemid", lookup_value=223986, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Rale", "chartevents", lookup_column="itemid", lookup_value=223987, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Rale", "chartevents", lookup_column="itemid", lookup_value=223988, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Rale", "chartevents", lookup_column="itemid", lookup_value=223989, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Tiếng thở phế quản", "chartevents", lookup_column="itemid", lookup_value=223986, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Tiếng thở phế quản", "chartevents", lookup_column="itemid", lookup_value=223987, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Tiếng thở phế quản", "chartevents", lookup_column="itemid", lookup_value=223988, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Tiếng thở phế quản", "chartevents", lookup_column="itemid", lookup_value=223989, value_column="value", time_column="charttime", stay_id_column="stay_id"),
            MappingRule("Huyết áp tâm thu", "chartevents", lookup_column="itemid", lookup_value=220179, value_column="valuenum", time_column="charttime", stay_id_column="stay_id"),

            # ===== LABEVENTS =====
            # Lưu ý: dataclass đã mặc định stay_id_column=None, nên không cần khai báo
            MappingRule("Bạch cầu niệu", "labevents", lookup_column="itemid", lookup_value=51516, value_column="valuenum", time_column="charttime"),
            MappingRule("Mủ niệu", "labevents", lookup_column="itemid", lookup_value=51516, value_column="valuenum", time_column="charttime"),
            MappingRule("Nitrat niệu", "labevents", lookup_column="itemid", lookup_value=51987, value_column="value", time_column="charttime"),
            MappingRule("Nitrat niệu", "labevents", lookup_column="itemid", lookup_value=51487, value_column="value", time_column="charttime"),

            # ===== MICROBIOLOGY =====
            MappingRule("Cấy dịch đường hô hấp", "microbiology", lookup_column="spec_type_desc", lookup_value="BRONCHIAL WASHINGS", value_column="org_name", time_column="charttime"),
            MappingRule("Cấy dịch đường hô hấp", "microbiology", lookup_column="spec_type_desc", lookup_value="BRONCHOALVEOLAR LAVAGE", value_column="org_name", time_column="charttime"),
            MappingRule("Cấy dịch đường hô hấp", "microbiology", lookup_column="spec_type_desc", lookup_value="PLEURAL FLUID", value_column="org_name", time_column="charttime"),
            MappingRule("Cấy dịch đường hô hấp", "microbiology", lookup_column="spec_type_desc", lookup_value="SPUTUM", value_column="org_name", time_column="charttime"),
            MappingRule("Cấy máu", "microbiology", lookup_column="spec_type_desc", lookup_value="BLOOD CULTURE", value_column="org_name", time_column="charttime"),
            MappingRule("Cấy nước tiểu", "microbiology", lookup_column="spec_type_desc", lookup_value="URINE", secondary_lookup_column="test_name", secondary_lookup_value="URINE CULTURE", value_column="org_name", time_column="charttime"),

            # ===== PROCEDUREEVENTS =====
            MappingRule("Thời gian thở máy", "procedureevents", lookup_column="itemid", lookup_value=225792, value_column="value", time_column="starttime", endtime_column="endtime", stay_id_column="stay_id"),
            MappingRule("Thời gian đặt catheter TMTT", "procedureevents", lookup_column="location", lookup_value="__CENTRAL_LINE__", lookup_operator="in_list", value_column="value", time_column="starttime", endtime_column="endtime", stay_id_column="stay_id"),
            MappingRule("Thời gian đặt ống thông tiểu", "procedureevents", lookup_column="itemid", lookup_value=229351, value_column="value", time_column="starttime", endtime_column="endtime", stay_id_column="stay_id"),

            # ===== DISCHARGE =====
            # Note: Đã ẩn các lookup_column/lookup_value vì keyword_search xử lý qua value_column
            MappingRule("Ớn lạnh", "discharge", lookup_operator="keyword_search", value_column="text", time_column="charttime", note_id_column="note_id"),
            MappingRule("Tiểu gấp", "discharge", lookup_operator="keyword_search", value_column="text", time_column="charttime", note_id_column="note_id"),
            MappingRule("Tiểu nhiều lần", "discharge", lookup_operator="keyword_search", value_column="text", time_column="charttime", note_id_column="note_id"),
            MappingRule("Tiểu buốt", "discharge", lookup_operator="keyword_search", value_column="text", time_column="charttime", note_id_column="note_id"),
            MappingRule("Đau hông sườn", "discharge", lookup_operator="keyword_search", value_column="text", time_column="charttime", note_id_column="note_id"),
            MappingRule("Đau/ấn đau vùng trên xương mu", "discharge", lookup_operator="keyword_search", value_column="text", time_column="charttime", note_id_column="note_id"),

            # ===== RADIOLOGY =====
            # hadm_id_column set thành None vì trong code gốc bạn ghi rõ "hadm_id_column": None
            MappingRule("X-quang ngực","radiology",lookup_column="note_type",lookup_value="Radiology",value_column="text",time_column="charttime",note_id_column="note_id"),
            MappingRule("CT scan lồng ngực","radiology",lookup_column="note_type",lookup_value="Radiology",value_column="text",time_column="charttime",note_id_column="note_id"),
        ]
        
        # Chuyển đổi thành DataFrame và loại bỏ trùng lặp
        df_rules = pd.DataFrame([vars(r) for r in rules])
        return df_rules.drop_duplicates().reset_index(drop=True)

    def _apply_lookup_logic(self, df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
        """Hàm private xử lý logic lọc (==, in_list, keyword_search)"""
        if row["lookup_operator"] == "==" and pd.notna(row["lookup_column"]):
            df = df[df[row["lookup_column"]] == row["lookup_value"]]

        elif row["lookup_operator"] == "in_list":
            if row["lookup_column"] == "location" and row["lookup_value"] == "__CENTRAL_LINE__":
                df = df[df["location"].isin(self.central_line_locations)]

        elif row["lookup_operator"] == "keyword_search":
            keywords = self.keyword_map.get(row["variable_name"], [])
            if "text" in df.columns and keywords:
                pattern = "|".join(keywords)
                df = df[df["text"].str.contains(pattern, case=False, na=False, regex=True)]
                
        # Áp dụng bộ lọc thứ 2 nếu có
        if pd.notna(row["secondary_lookup_column"]):
            col2 = row["secondary_lookup_column"]
            val2 = row["secondary_lookup_value"]
            if col2 in df.columns:
                df = df[df[col2] == val2]
                
        return df

    def get_variable_data(self, variable_name: str, time_process_func = None, in_time = None, out_time = None, subject_id: Optional[int] = None, stay_id: Optional[int] = None, check_48h: bool = True) -> pd.DataFrame:
        """
        Trích xuất dữ liệu cho một biến cụ thể.
        """
        # Lấy các luật áp dụng cho biến này
        target_mappings = self.mapping_rules[self.mapping_rules["variable_name"] == variable_name]
        results = []

        for _, row in target_mappings.iterrows():
            source_table = row["source_table"]
            if source_table not in self.tables:
                continue # Bỏ qua nếu bảng không được cung cấp

            # Lấy data và filter theo các ID
            df = self.tables[source_table]
            
            # Lọc theo ID (dùng pd.notna để tránh lỗi với giá trị None trong Pandas)
            if subject_id is not None and pd.notna(row["subject_id_column"]) and row["subject_id_column"] in df.columns:
                df = df[df[row["subject_id_column"]] == subject_id]

            if stay_id is not None and pd.notna(row["stay_id_column"]) and row["stay_id_column"] in df.columns:
                df = df[df[row["stay_id_column"]] == stay_id]

            # Áp dụng logic lookup nghiệp vụ
            df = self._apply_lookup_logic(df, row)

            if df.empty:
                continue
            
            if time_process_func is not None:
                time_col = row["time_column"]
                if pd.notna(time_col) and time_col in df.columns:
                    # Đảm bảo cột thời gian trong df là datetime
                    df[time_col] = time_process_func(df[time_col])

                    if stay_id is None and in_time is not None and out_time is not None:
                        # Sử dụng .between để lọc trong khoảng [in_time, outtime]
                        df = df[df[time_col].between(in_time, out_time)]
                    
                    if check_48h and in_time is not None: 
                        min_time_allowed = in_time + pd.Timedelta(hours=48)
                        df = df[df[time_col] >= min_time_allowed]
                    

            # Copy để gán meta-data mà không ảnh hưởng DataFrame gốc
            df_mapped = df.copy()
            df_mapped["mapped_variable_name"] = row["variable_name"]
            df_mapped["mapped_source_table"] = row["source_table"]
            df_mapped["mapped_value_column"] = row["value_column"]
            df_mapped["mapped_time_column"] = row["time_column"]
            df_mapped["mapped_endtime_column"] = row["endtime_column"]

            results.append(df_mapped)

        if not results:
            return pd.DataFrame()

        # Gộp kết quả
        final_result = pd.concat(results, ignore_index=True, sort=False)
        return final_result.drop_duplicates().reset_index(drop=True)
    