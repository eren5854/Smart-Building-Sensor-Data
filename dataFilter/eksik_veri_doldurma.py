import pandas as pd
import numpy as np

# 1. Veriyi yükle
df = pd.read_csv('all_sensors_merged.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Eksik zaman aralıklarını belirle
# 10 saniyeden uzun (örneğin 15sn+) boşluk olan yerleri tespit ediyoruz
df = df.sort_values('timestamp')
diff = df['timestamp'].diff().dt.total_seconds()
gap_indices = diff[diff > 15].index # 15 saniyeden büyük boşlukları yakala

# 3. Boşlukları doldurmak için yeni satırlar oluştur
new_rows = []
for idx in gap_indices:
    start_time = df.loc[idx-1, 'timestamp']
    end_time = df.loc[idx, 'timestamp']
    
    # Aradaki 10'ar saniyelik adımları hesapla
    current_time = start_time + pd.Timedelta(seconds=10)
    while current_time < end_time - pd.Timedelta(seconds=5):
        new_rows.append({'timestamp': current_time})
        current_time += pd.Timedelta(seconds=10)

# 4. Yeni satırları ana tabloya ekle ve tekrar sırala
df_gap_filled = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
df_gap_filled = df_gap_filled.sort_values('timestamp').reset_index(drop=True)

# 5. İnterpolasyon (Sadece sayısal sütunlara)
numeric_cols = ['temperature', 'humidity', 'motion', 'ldr', 'outTemperature', 'outHumidity', 'outLdr']
df_gap_filled[numeric_cols] = df_gap_filled[numeric_cols].interpolate(method='linear')

# 6. FORMATI ESKİ HALİNE GETİRME (Kritik Adım)
# Tam sayı olması gereken sütunları tekrar integer yapıyoruz
int_cols = ['motion', 'ldr', 'outLdr']
df_gap_filled[int_cols] = df_gap_filled[int_cols].round().astype(int)

# Ondalıklı sayıları orijinaldeki gibi tek haneye çekiyoruz (Opsiyonel)
float_cols = ['temperature', 'humidity', 'outTemperature', 'outHumidity']
df_gap_filled[float_cols] = df_gap_filled[float_cols].round(1)

# 7. Kaydetme (Orijinal mikro saniye formatını koruyarak)
df_gap_filled.to_csv('doludurulmus_veri.csv', index=False)

print(df_gap_filled.head(10))