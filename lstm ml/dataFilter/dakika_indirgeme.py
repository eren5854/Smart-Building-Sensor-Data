import pandas as pd

# 1. Dosyayı oku
df = pd.read_csv("temizlenmis_hareket_veriler.csv")

# 2. Zaman damgasını işle
df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')
df.set_index("timestamp", inplace=True)

# 3. Yeniden örnekleme (Dakikalık ortalama)
resample_logic = {
    'temperature': 'mean',
    'humidity': 'mean',
    'ldr': 'mean',
    'outTemperature': 'mean',
    'outHumidity': 'mean',
    'outLdr': 'mean'
}
minute_df = df.resample('1T').agg(resample_logic)

# 4. Hareket (motion) verisi: En çok tekrar edeni (mod) bulma
motion_resampled = df['motion'].resample('1T').mean().apply(lambda x: 1 if x >= 0.5 else 0)
minute_df['motion'] = motion_resampled

# ---------------------------------------------------------
# 5. FORMATLAMA İŞLEMLERİ
# ---------------------------------------------------------

# Sıcaklık ve Nem: Virgülden sonra 2 basamak
float_columns = ['temperature', 'humidity', 'outTemperature', 'outHumidity']
minute_df[float_columns] = minute_df[float_columns].round(2)

# LDR ve outLdr: Ondalık kısımları at ve tam sayı yap
# Önce yuvarlayıp sonra int tipine çeviriyoruz
ldr_columns = ['ldr', 'outLdr']
minute_df[ldr_columns] = minute_df[ldr_columns].round(0).astype(int)

# Sütun sırasını orijinal formata getir
column_order = ['temperature', 'humidity', 'motion', 'ldr', 'outTemperature', 'outHumidity', 'outLdr']
minute_df = minute_df[column_order]

# 6. Saniye ve milisaniyeyi temizle
minute_df.index = minute_df.index.strftime('%Y-%m-%d %H:%M')

# 7. Kaydet
minute_df.to_csv("sensors_data.csv")

print("İşlem tamamlandı! Örnek satır:")
print(minute_df.head(1))