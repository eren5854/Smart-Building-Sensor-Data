import pandas as pd

# 1. Veriyi yükle
df = pd.read_csv('doludurulmus_veri.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Yardımcı zaman sütunları oluştur
df['hour'] = df['timestamp'].dt.hour
df['date'] = df['timestamp'].dt.date

# 3. Gelişmiş Lamba Kontrol Mantığı
def check_lamp(row):
    ldr = row['ldr']
    out_ldr = row['outLdr']
    hour = row['hour']
    
    # Öğle saatleri filtresi (11:00 - 14:00 arası)
    if 11 <= hour <= 14:
        # Bu saatlerde güneş güçlüdür. 
        # Lambanın açık sayılması için iç ışığın dışarıdan çok daha baskın olması gerekir.
        # Ayrıca LDR'nin sizin belirttiğiniz 300-400 aralığında olmasını bekleriz.
        if (ldr >= 300 and ldr <= 450) and (out_ldr > ldr + 500): 
            return True
        else:
            return False
    else:
        # Diğer saatlerde (akşam, gece, sabah erken)
        # LDR 300-400 arasındaysa ve dışarısı içeriden daha karanlıksa (outLdr > ldr)
        if (ldr >= 300 and ldr <= 450):
            return True
        return False

# Mantığı uygula
df['is_light_on'] = df.apply(check_lamp, axis=1)

# 4. Günlük Hesaplama (Her satır 10 saniye)
daily_analysis = df.groupby('date')['is_light_on'].sum() * 10

# 5. Tabloyu Hazırla
summary_table = daily_analysis.reset_index()
summary_table.columns = ['Tarih', 'Toplam_Saniye']

def format_seconds(s):
    hours = s // 3600
    minutes = (s % 3600) // 60
    seconds = s % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

summary_table['Kullanım_Süresi (SS:DD:sn)'] = summary_table['Toplam_Saniye'].apply(format_seconds)

# 6. Sonuçları Yazdır
print("-" * 60)
print("GÜNCEL GÜNLÜK LAMBA KULLANIM RAPORU (Öğle Saati Filtreli)")
print("-" * 60)
print(summary_table[['Tarih', 'Kullanım_Süresi (SS:DD:sn)']])
print("-" * 60)