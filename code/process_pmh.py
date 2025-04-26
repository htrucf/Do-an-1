import ijson
import pandas as pd
import re
import os

os.chdir('C:/Users/LENOVO/Downloads') # THAY ĐỔI ĐƯỜNG DẪN ĐẾN THƯ MỤC CỦA BẠN
filename = "./data/notes_sepsis.json"
disease_keywords = {
    'smoke': ['smoke', 'smoking', 'tobacco', 'cigarette'],
    'COPD': ['copd', 'chronic obstructive pulmonary disease'],
    'CKD': ['ckd', 'chronic kidney disease', 'esrd', 'end-stage renal disease', 'chronic renal failure', 'crf'],
    'tumor': ['tumor', 'malignancy', 'melanoma', 'cell carcinoma'],
    'hypertension': ['hypertension', 'high blood pressure', 'htn'],
    'CHD': ['chd', 'coronary heart disease', 'ischemic heart disease', 'cad', 'coronary artery disease','cabg'],
    "CHF": ["chf", "congestive heart failure", "chronic heart failure", "heart failure", "hfref", "hfpef", "systolic heart failure", "diastolic heart failure", "reduced ejection fraction", "preserved ejection fraction", " ef "],
    'diabetes': ['diabetes', 'dm', 'dm2', 'dm ii', 'dm1', 'dm i', 'type 2 diabetes', 'type ii diabetes', 'type 1 diabetes', 'type i diabetes', 'insulin dependent diabetes', 'non-insulin dependent diabetes', 'iddm', 'niddm', 'diabetic', 'diabetes mellitus', 'adult-onset diabetes', 'juvenile diabetes', 'latent autoimmune diabetes in adults', 'diabetes mellitus type 2', 'diabetes mellitus type 1', 'pancreatic diabetes', 'type 2 DM', 'type 1 DM', 'T2DM', 'T1DM', 'diabetes of pregnancy', 'gestational diabetes', 'maturity onset diabetes of the young', 'insulin resistance', 'metabolic syndrome'],
    'cirrhosis': ['cirrhosis', 'liver cirrhosis', "psc", "autoimmune hepatitis", "cirrhotic changes", "nodular liver"]
}

def extract_diseases(text):
    match = re.search(
        r"Past Medical History\s*:\s*\n*((?:.*\n)*?)(?=^\s*Social History\s*:)",
        text,
        re.IGNORECASE | re.MULTILINE
    )
    pmh_text = match.group(1).lower() if match else ""

    def has_disease(pmh_text, keywords):
        for kw in keywords:
            if kw in pmh_text:
                # Nếu có keyword, kiểm tra xem nó có bị phủ định không
                negation_patterns = [
                    rf"\(\-\)\s*{re.escape(kw)}",         # (-)diabetes
                    rf"no\s+{re.escape(kw)}",             # no diabetes
                    rf"denies\s+{re.escape(kw)}",         # denies diabetes
                    rf"without\s+{re.escape(kw)}",        # without diabetes
                ]
                if any(re.search(pat, pmh_text) for pat in negation_patterns):
                    continue  # bỏ qua nếu có biểu hiện phủ định
                return True  # chỉ return True nếu có keyword KHÔNG phủ định
        return False

    return {
        disease: int(has_disease(pmh_text, keywords))
        for disease, keywords in disease_keywords.items()
    }


# Đọc từng phần tử trong mảng JSON
results = []
with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        obj = eval(line.strip())  # Chuyển đổi dòng thành một đối tượng JSON
        data = extract_diseases(obj.get('text', ''))
        data['subject_id'] = obj.get('subject_id')
        data['hadm_id'] = obj.get('hadm_id')
        results.append(data)

# Chuyển sang DataFrame
df = pd.DataFrame(results)
df.to_csv("./data/past_medical_history_extracted.csv", index=False)
print("Đã lưu file past_medical_history_extracted.csv")
