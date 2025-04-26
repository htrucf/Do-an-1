import pandas as pd
from collections import Counter
import os

os.chdir('C:/Users/LENOVO/Downloads') # THAY ĐỔI ĐƯỜNG DẪN ĐẾN THƯ MỤC CỦA BẠN

# Bước 1: Đọc file Excel (giả sử sheet chứa dữ liệu cần xử lý có tên là 'triage_data')
df = pd.read_excel('./data/triage_data.xlsx', sheet_name='chiefcomplaint')  # Thay đúng tên file và sheet

# Bước 2: Tách cụm từ bằng dấu phẩy, chuẩn hóa chữ thường, loại bỏ khoảng trắng
all_phrases = []

for entry in df['chiefcomplaint'].dropna():  # Giả sử cột chứa dữ liệu là 'chiefcomplaint'
    phrases = entry.split(',')
    cleaned = [phrase.strip().lower() for phrase in phrases if phrase.strip()]
    all_phrases.extend(cleaned)

# Bước 3: Thống kê số lần xuất hiện
counter = Counter(all_phrases)

# Bước 4: Lưu thống kê ra file CSV
counter_df = pd.DataFrame(counter.items(), columns=['Phrase', 'Count'])
counter_df.to_csv('./data/count_complaint.csv', index=False)

# In kết quả thống kê
print(counter_df.head())  # In ra một vài dòng đầu của thống kê
