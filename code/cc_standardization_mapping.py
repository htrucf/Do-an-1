import pandas as pd
from openpyxl import load_workbook
import os

os.chdir('C:/Users/LENOVO/Downloads') # THAY ĐỔI ĐƯỜNG DẪN ĐẾN THƯ MỤC CỦA BẠN
# Đọc dữ liệu từ file Excel
map_path = "./data/complaint_extracted.xlsx"
data_path = "./data/triage_data.xlsx"

# Sheet ccmap: chứa old_word → new_word
df_map = pd.read_excel(map_path, sheet_name='result')

# Sheet triage: chứa chiefcomplaint cần chuẩn hóa
df_triage = pd.read_excel(data_path)
sheet_name = 'triage_data'
# Chuyển đổi old_word → new_word từ df_map thành list of tuples
mapping_rules = list(zip(df_map['old_word'], df_map['new_word']))

# Hàm xử lý chuẩn hóa 1 dòng văn bản
def standardize_complaint(text):
    if pd.isna(text):
        return None

    text_lower = str(text).lower()
    phrases = [p.strip() for p in text_lower.split(",")]
    matched = set()

    for old_word, new_word in mapping_rules:
        if old_word.lower() == "pain":
            # Nếu chỉ có "pain" đơn lẻ, KHÔNG nằm trong các cụm cụ thể
            if "pain" in phrases and not any(specific_pain in phrases for specific_pain in [
                "abdominal pain", "chest pain", "back pain", "leg pain", "head pain",
                "flank pain", "shoulder pain", "knee pain", "hip pain", "arm pain",
                "hand pain", "neck pain", "eye pain", "dental pain"
            ]):
                matched.add("body pain")
        else:
            # Nếu old_word có mặt trong phrases (đã tách riêng từng cụm)
            if old_word.lower() in phrases:
                matched.add(new_word.strip())

    return ",".join(sorted(matched)) if matched else None

# Áp dụng lên toàn bộ cột chiefcomplaint
df_triage['chiefcomplaint_changed'] = df_triage['chiefcomplaint'].apply(standardize_complaint)

# Bước 2: Ghi lại vào file Excel gốc, sheet 'triage'
book = load_workbook(data_path)

# Tính vị trí bắt đầu ghi (cột cuối cùng)
startcol = len(book[sheet_name][1])  # Số cột hiện có trong sheet

# Dữ liệu muốn ghi
df_to_write = df_triage[['chiefcomplaint_changed']]
with pd.ExcelWriter(data_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    writer._book = book 
    
    df_to_write.to_excel(writer, sheet_name=sheet_name, startcol=startcol, startrow=0, index=False, header=True)

print("Đã thêm cột 'chiefcomplaint_changed'.")
