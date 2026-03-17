import pandas as pd
import glob
import os

# CSV dosyalarının yolu
csv_files = glob.glob("../newLogs/*.csv")

total_rows = 0

print(f"{'Dosya Adı':<50} | {'Veri Satırı Sayısı':<20}")
print("-" * 75)

for file in csv_files:
    try:
        # Dosyayı oku
        df = pd.read_csv(file)
        
        # Satır sayısını al (Pandas zaten ilk satırı başlık sayıp veri olarak saymaz)
        row_count = len(df)
        total_rows += row_count
        
        # Sadece dosya adını göster (yolun tamamını değil)
        file_name = os.path.basename(file)
        print(f"{file_name:<50} | {row_count:<20}")
        
    except Exception as e:
        print(f"{file} okunurken hata oluştu: {e}")

print("-" * 75)
print(f"{'TOPLAM VERİ SATIRI':<50} | {total_rows:<20}")