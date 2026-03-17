import pandas as pd

# 1. Dosyayı oku
input_file = 'doludurulmus_veri.csv'
output_file = 'temizlenmis_hareket_veriler.csv'

df = pd.read_csv(input_file, skipinitialspace=True)

# --- DEĞİŞİKLİK TAKİBİ İÇİN KOPYA ALALIM ---
original_motion = df['motion'].copy()

# 2. Hareket verisini düzelt
df['motion'] = df['motion'].rolling(window=5, center=True).median()

# 3. Boşlukları doldur
df['motion'] = df['motion'].ffill().bfill()

# 4. Tamsayıya çevir
df['motion'] = df['motion'].astype(int)

# --- KAÇ VERİ DEĞİŞTİ HESAPLAYALIM ---
# Orijinal seri ile yeni seri arasındaki farkları sayıyoruz
changed_count = (original_motion != df['motion']).sum()
total_count = len(df)
change_percentage = (changed_count / total_count) * 100

# 5. Kaydet
df.to_csv(output_file, index=False)

print(f"İşlem tamamlandı. '{output_file}' dosyasına bakabilirsiniz.")
print(f"Toplam Satır: {total_count}")
print(f"Düzeltilen Veri Sayısı: {changed_count}")
print(f"Değişim Oranı: %{change_percentage:.2f}")