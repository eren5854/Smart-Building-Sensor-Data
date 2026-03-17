import pandas as pd

# 1. Dosyayı oku
dosya_adi = 'sensors_data.csv'  # Kendi dosyanızın adını buraya yazın
df = pd.read_csv(dosya_adi)

# 'timestamp' sütununu zaman verisine dönüştürelim (sıralı olduğundan emin olmak için)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 2. Hata/Gürültü Düzeltme İşlemi (Rolling Median)
# window_size: Kaç satırlık bir bloğun çoğunluğuna bakılacağını belirler.
# Örn: window=5 ise, kendisi, 2 önceki ve 2 sonraki değere bakar. Çoğunluk neyse onu alır.
# Kısa süreli gürültüleriniz 3-4 dakika sürüyorsa, bu değeri 7 veya 9 yapabilirsiniz.
window_size = 21  # 21 dakikalık bir pencere, yani 10 dakika öncesi ve 10 dakika sonrası dahil

# Hareketli medyanı hesapla, center=True ile ortadaki veriyi baz al
df['motion_duzeltilmis'] = df['motion'].rolling(window=window_size, center=True).median()

# Uç noktalarda (ilk ve son satırlarda) NaN(boş) değerler oluşabilir, 
# buraları orijinal değerlerle dolduruyoruz.
df['motion_duzeltilmis'] = df['motion_duzeltilmis'].fillna(df['motion'])

# float'a dönüşmüş olabilen veriyi tekrar tam sayıya (0 veya 1) çeviriyoruz
df['motion'] = df['motion_duzeltilmis'].astype(int)

# İşlem için oluşturduğumuz geçici sütunu siliyoruz
df = df.drop(columns=['motion_duzeltilmis'])

# 3. Yeni veriyi yeni bir CSV dosyasına kaydet
yeni_dosya_adi = 'new_sensors_data.csv'
df.to_csv(yeni_dosya_adi, index=False)

print(f"Veriler başarıyla düzeltildi ve '{yeni_dosya_adi}' adlı dosyaya kaydedildi.")