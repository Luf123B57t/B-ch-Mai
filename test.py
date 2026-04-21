import pandas as pd
import spacy
from negspacy.negation import Negex
from spacy.pipeline import EntityRuler

def label_medical_text(input_csv, output_csv):
    # 1. Khởi tạo mô hình NLP
    nlp = spacy.load("en_core_web_sm")

    # 2. Danh sách các từ khóa của bạn
    target_keywords = [
        "pneumonia", "pna", "bronchopneumonia", "consolidation", 
        "infiltrate", "infiltration", "opacity", "opacities", 
        "ground glass", "ggo", "air bronchogram", "patchy", "cloudy", "hazy"
    ]

    # 3. Thêm EntityRuler để định nghĩa các từ khóa này là thực thể (Entity)
    # Điều này bắt buộc vì negspacy chỉ kiểm tra phủ định trên các Entity
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "CONDITION", "pattern": word} for word in target_keywords]
    ruler.add_patterns(patterns)

    # 4. Thêm thành phần Negex vào pipeline
    # ent_types=["CONDITION"] giúp nó tập trung kiểm tra phủ định cho các từ khóa ta vừa định nghĩa
    nlp.add_pipe("negex", config={"ent_types": ["CONDITION"]})

    # 5. Đọc dữ liệu
    df = pd.read_csv(input_csv)

    def process_text(text):
        if pd.isna(text) or text.strip() == "":
            return 0
        
        doc = nlp(text.lower())
        
        for ent in doc.ents:
            # Nếu thực thể nằm trong danh sách từ khóa và KHÔNG bị phủ định (._.negex == False)
            if ent.label_ == "CONDITION" and ent._.negex == False:
                return 1
        return 0

    # 6. Chạy xử lý (có thể mất thời gian nếu dữ liệu lớn)
    print("Đang xử lý dữ liệu...")
    df['label'] = df['text'].apply(process_text)

    # 7. Lưu kết quả
    df.to_csv(output_csv, index=False)
    print(f"Hoàn thành! Kết quả đã lưu tại: {output_csv}")

# Chạy hàm
label_medical_text('bert_findings.csv', 'labeled_with_negspacy.csv')