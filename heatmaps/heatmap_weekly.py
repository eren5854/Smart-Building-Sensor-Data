import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Veriyi Oku
df = pd.read_csv('../dataFilter/sensors_data.csv')

# 2. Timestamp sütununu datetime formatına çevir
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 3. Saat ve Gün bilgilerini çıkar
df['hour'] = df['timestamp'].dt.hour
df['day_name'] = df['timestamp'].dt.day_name()

# Hafta ve Yıl bilgisini çıkar (ISO takvimi pazartesiden başlar)
df['week'] = df['timestamp'].dt.isocalendar().week
df['year'] = df['timestamp'].dt.isocalendar().year

# İngilizce gün isimlerini Türkçe'ye çevir
gunler_map = {
    'Monday': 'Pazartesi', 'Tuesday': 'Salı', 'Wednesday': 'Çarşamba',
    'Thursday': 'Perşembe', 'Friday': 'Cuma', 'Saturday': 'Cumartesi', 'Sunday': 'Pazar'
}
df['day_name'] = df['day_name'].map(gunler_map)

# Y ekseninde (solda) günlerin doğru sırada çıkması için Categorical veri tipine çeviriyoruz
gunler_sira = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
df['day_name'] = pd.Categorical(df['day_name'], categories=gunler_sira, ordered=True)

# 4. Her yılın her haftası için döngü oluştur
for (year, week), group in df.groupby(['year', 'week']):
    
    # Klasör oluştur (Örn: Yil_2025_Hafta_49)
    folder_name = f'Yil_{year}_Hafta_{week}'
    os.makedirs(folder_name, exist_ok=True)
    
    # X ekseninde 0'dan 23'e kadar tüm saatlerin çıkmasını garantile
    all_hours = range(24)
    
    # Hareket için Yoğunluk (Sum) hesaplayan pivot tablo
    motion_pivot = group.pivot_table(index='day_name', columns='hour', values='motion', aggfunc='sum', dropna=False)
    motion_pivot = motion_pivot.reindex(columns=all_hours, fill_value=0) # Veri olmayan saatleri 0 ile doldur
    
    # Sıcaklık için Ortalama hesaplayan pivot tablo
    temp_pivot = group.pivot_table(index='day_name', columns='hour', values='temperature', aggfunc='mean', dropna=False)
    temp_pivot = temp_pivot.reindex(columns=all_hours) # Veri olmayan saatler NaN (boş) kalır
    
    # LDR için Ortalama hesaplayan pivot tablo
    ldr_pivot = group.pivot_table(index='day_name', columns='hour', values='ldr', aggfunc='mean', dropna=False)
    ldr_pivot = ldr_pivot.reindex(columns=all_hours) # Veri olmayan saatler NaN (boş) kalır
    
    # Grafik çizdirme ve kaydetme yardımcı fonksiyonu
    def plot_weekly_heatmap(data, title, filename, cmap, fmt):
        plt.figure(figsize=(18, 8))
        # Heatmap oluştur. annot=True ile kutuların içine değerler yazılır.
        sns.heatmap(data, cmap=cmap, annot=True, fmt=fmt, linewidths=.5, cbar=True,
                    cbar_kws={'label': 'Değer'})
        plt.title(title, pad=15, fontsize=14)
        plt.xlabel('Saatler (0-23)', fontsize=12)
        plt.ylabel('Günler', fontsize=12)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Grafikleri Çiz ve İlgili Hafta Klasörüne Kaydet
    
    # Hareket grafiği 
    plot_weekly_heatmap(motion_pivot, f'Haftalık Hareket Yoğunluğu ({year} - Hafta {week})', 
                        os.path.join(folder_name, 'hareket.png'), 'Blues', '.0f')
    
    # Sıcaklık grafiği (Maviden Kırmızıya geçiş için 'coolwarm' eklendi)
    plot_weekly_heatmap(temp_pivot, f'Haftalık Sıcaklık Ortalaması ({year} - Hafta {week})', 
                        os.path.join(folder_name, 'sicaklik.png'), 'coolwarm', '.1f')
    
    # LDR grafiği (Beyazdan Siyaha geçiş için 'Greys' eklendi)
    plot_weekly_heatmap(ldr_pivot, f'Haftalık LDR Ortalaması ({year} - Hafta {week})', 
                        os.path.join(folder_name, 'ldr.png'), 'Greys', '.1f')

print("İşlem başarıyla tamamlandı. Tüm haftalara ait klasörler ve ısı haritaları güncellenmiş renklerle oluşturuldu.")