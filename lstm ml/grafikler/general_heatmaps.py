import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Veriyi Yükleme ve Hazırlama
# -----------------------------------------------------------------------------
df = pd.read_csv('../dataFilter/sensors_data.csv')

# Timestamp sütununu datetime formatına çevirme
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Gün ve Saat bilgilerini çıkarma
df['Gun'] = df['timestamp'].dt.day_name()
df['Saat'] = df['timestamp'].dt.hour

# Gün isimlerini Türkçeye çevirme ve doğru sıralama için kategorik veri tipi oluşturma
gun_ceviri = {
    'Monday': 'Pazartesi', 'Tuesday': 'Salı', 'Wednesday': 'Çarşamba',
    'Thursday': 'Perşembe', 'Friday': 'Cuma', 'Saturday': 'Cumartesi', 'Sunday': 'Pazar'
}
df['Gun'] = df['Gun'].map(gun_ceviri)
gun_sirasi = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
df['Gun'] = pd.Categorical(df['Gun'], categories=gun_sirasi, ordered=True)

# -----------------------------------------------------------------------------
# 2. HAREKET YOĞUNLUK HARİTASI (% Formatında)
# -----------------------------------------------------------------------------
# Hareket verisinin ortalamasını alıp 100 ile çarparak % değerine çeviriyoruz
motion_pivot = df.pivot_table(index='Gun', columns='Saat', values='motion', aggfunc='mean')
motion_pct = motion_pivot * 100

# Kutucukların içerisine %100, %50 şeklinde yazdırmak için özel metin formatı oluşturma
annot_labels = motion_pct.apply(lambda col: col.map(lambda x: f"%{int(x)}" if pd.notnull(x) else ""))

plt.figure(figsize=(16, 6))

# fmt="" yapıyoruz çünkü kendi hazırladığımız string (metin) dizisini (annot_labels) veriyoruz
sns.heatmap(motion_pct, cmap='YlGnBu', annot=annot_labels, fmt="", linewidths=.5)

plt.title('Haftalık Personel Mevcut Durum (Yoğunluk) Haritası', fontsize=16)
plt.xlabel('Günün Saatleri', fontsize=12)
plt.ylabel('Haftanın Günleri', fontsize=12)
plt.tight_layout()
plt.savefig('1_hareket_yogunluk_haritasi.png')
plt.show()

# -----------------------------------------------------------------------------
# 3. İÇ VE DIŞ ORTAM SICAKLIK KARŞILAŞTIRMA HARİTASI
# -----------------------------------------------------------------------------
temp_in_pivot = df.pivot_table(index='Gun', columns='Saat', values='temperature', aggfunc='mean')
temp_out_pivot = df.pivot_table(index='Gun', columns='Saat', values='outTemperature', aggfunc='mean')

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# İç Sıcaklık
sns.heatmap(temp_in_pivot, ax=axes[0], cmap='YlOrRd', annot=True, fmt=".1f", linewidths=.5)
axes[0].set_title('Haftalık İÇ Ortam Sıcaklık Haritası (°C)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Saat')
axes[0].set_ylabel('Gün')

# Dış Sıcaklık
sns.heatmap(temp_out_pivot, ax=axes[1], cmap='coolwarm', annot=True, fmt=".1f", linewidths=.5)
axes[1].set_title('Haftalık DIŞ Ortam Sıcaklık Haritası (°C)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Saat')
axes[1].set_ylabel('Gün')

plt.tight_layout()
plt.savefig('2_sicaklik_karsilastirma.png')
plt.show()

# -----------------------------------------------------------------------------
# 4. İÇ VE DIŞ ORTAM IŞIK ŞİDDETİ (LDR) HARİTASI (Güncellenmiş Renk Skalası)
# -----------------------------------------------------------------------------
ldr_in_pivot = df.pivot_table(index='Gun', columns='Saat', values='ldr', aggfunc='mean')
ldr_out_pivot = df.pivot_table(index='Gun', columns='Saat', values='outLdr', aggfunc='mean')

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# İç LDR: Düşük=Aydınlık (Sarı/Açık), Yüksek=Karanlık (Siyah/Mor) için 'magma_r' kullanıldı.
sns.heatmap(ldr_in_pivot, ax=axes[0], cmap='magma_r', annot=True, fmt=".0f", linewidths=.5)
axes[0].set_title('İÇ Işık Şiddeti (LDR) - [Açık Renk=Aydınlık, Koyu Renk=Karanlık]', fontsize=14)
axes[0].set_xlabel('Saat')
axes[0].set_ylabel('Gün')

# Dış LDR: 'viridis_r' kullanıldı (Sarı/Açık yeşilden koyu mora doğru)
sns.heatmap(ldr_out_pivot, ax=axes[1], cmap='viridis_r', annot=True, fmt=".0f", linewidths=.5)
axes[1].set_title('DIŞ Işık Şiddeti (LDR)', fontsize=14)
axes[1].set_xlabel('Saat')
axes[1].set_ylabel('Gün')

plt.suptitle('Haftalık Işık Şiddeti Analizi', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('3_ldr_karsilastirma.png')
plt.show()