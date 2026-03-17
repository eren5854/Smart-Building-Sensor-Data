import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Veriyi Oku
# Dosya adının sensors_data.csv olduğundan emin olun
df = pd.read_csv('../newLogs/sensors_data.csv')

# 2. Timestamp sütununu datetime formatına çevir
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 3. Tarih, saat ve İngilizce gün isimlerini çıkar
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_name'] = df['timestamp'].dt.day_name()

# İngilizce gün isimlerini Türkçe'ye çevir (Sistem dillerinden bağımsız hatasız çalışması için)
gunler_map = {
    'Monday': 'Pazartesi', 'Tuesday': 'Salı', 'Wednesday': 'Çarşamba',
    'Thursday': 'Perşembe', 'Friday': 'Cuma', 'Saturday': 'Cumartesi', 'Sunday': 'Pazar'
}
df['day_name'] = df['day_name'].map(gunler_map)

# 4. Her tarih (gün) için döngü oluşturup klasör ve grafikleri yarat
for current_date, group in df.groupby('date'):
    
    # Klasör oluştur (Örn: 2025-12-01)
    folder_name = str(current_date)
    os.makedirs(folder_name, exist_ok=True)
    
    # X ekseninde 0'dan 23'e kadar tüm saatlerin çıkmasını garantilemek için tam saat dilimini tanımlıyoruz
    all_hours = range(24)
    
    # Hareket için Yoğunluk (Sum/Toplam) hesaplayan pivot tablo
    motion_pivot = group.pivot_table(index='day_name', columns='hour', values='motion', aggfunc='sum')
    motion_pivot = motion_pivot.reindex(columns=all_hours, fill_value=0) # Veri olmayan saatleri 0 ile doldur
    
    # Sıcaklık için Ortalama hesaplayan pivot tablo
    temp_pivot = group.pivot_table(index='day_name', columns='hour', values='temperature', aggfunc='mean')
    temp_pivot = temp_pivot.reindex(columns=all_hours) # Veri olmayan saatler NaN (boş) kalır
    
    # LDR için Ortalama hesaplayan pivot tablo
    ldr_pivot = group.pivot_table(index='day_name', columns='hour', values='ldr', aggfunc='mean')
    ldr_pivot = ldr_pivot.reindex(columns=all_hours) # Veri olmayan saatler NaN (boş) kalır
    
    # Grafik çizdirme ve kaydetme yardımcı fonksiyonu
    def plot_heatmap(data, title, filename, cmap, fmt):
        plt.figure(figsize=(15, 3)) # Tek gün (tek satır) olduğu için yüksekliği az tutuyoruz
        # Heatmap oluştur. annot=True ile kutuların içine değerler yazılır.
        sns.heatmap(data, cmap=cmap, annot=True, fmt=fmt, linewidths=.5, cbar=True,
                    cbar_kws={'label': 'Değer'}, vmin=0)
        plt.title(title)
        plt.xlabel('Saatler (0-23)')
        plt.ylabel('Gün')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close() # Bellekte yer kaplamaması için grafiği kapat

    # Grafikleri Çiz ve İlgili Klasöre Kaydet
    
    # Hareket grafiği (Değerler tam sayı olduğu için fmt='.0f')
    plot_heatmap(motion_pivot, f'Hareket Yoğunluğu - {current_date}', 
                 os.path.join(folder_name, 'hareket.png'), 'Blues', '.0f')
    
    # Sıcaklık grafiği (Ondalıklı değerler için fmt='.1f')
    plot_heatmap(temp_pivot, f'Sıcaklık Ortalaması - {current_date}', 
                 os.path.join(folder_name, 'sicaklik.png'), 'OrRd', '.1f')
    
    # LDR grafiği (Ondalıklı/Tam sayı değerleri için fmt='.1f')
    plot_heatmap(ldr_pivot, f'LDR Ortalaması - {current_date}', 
                 os.path.join(folder_name, 'ldr.png'), 'YlGn', '.1f')

print("İşlem başarıyla tamamlandı. Tüm günlere ait klasörler ve ısı haritaları (heatmaps) oluşturuldu.")