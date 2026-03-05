import pandas as pd

# 1. Veriyi yükle
df = pd.read_csv('all_sensors_merged.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Zaman damgasına göre sıralandığından emin ol
df = df.sort_values('timestamp')

# 3. İki satır arasındaki zaman farkını hesapla (saniye cinsinden)
df['diff_seconds'] = df['timestamp'].diff().dt.total_seconds()

# 4. 15 saniyeden büyük olan boşlukları filtrele
# (İlk satırda fark NaN olacağı için onu görmezden gelir)
gaps = df[df['diff_seconds'] > 15].copy()

# 5. Sonuçları Terminale Yazdır
if gaps.empty:
    print("Harika! Veri setinde 15 saniyeden büyük bir boşluk bulunamadı.")
else:
    print(f"--- Toplam {len(gaps)} adet büyük boşluk tespit edildi ---\n")
    print(f"{'Satır No':<10} | {'Başlangıç (Önceki Veri)':<26} | {'Bitiş (Hatalı Veri)':<26} | {'Kayıp Süre (sn)'}")
    print("-" * 90)
    
    for index, row in gaps.iterrows():
        # index-1 önceki geçerli verinin olduğu satırdır
        start_time = df.loc[index - 1, 'timestamp']
        end_time = row['timestamp']
        duration = row['diff_seconds']
        
        print(f"{index:<10} | {str(start_time):<26} | {str(end_time):<26} | {duration:.2f} sn")

    print("\n" + "-" * 90)
    print(f"En uzun kesinti: {gaps['diff_seconds'].max():.2f} saniye")