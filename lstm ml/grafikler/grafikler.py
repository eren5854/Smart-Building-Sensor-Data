import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv('../src3/train_processed.csv')

#Korelasyon Isı Haritası (Correlation Heatmap)
plt.figure(figsize=(12, 8))
# Sadece sayısal sütunları alalım
cols_to_corr = ['temperature', 'humidity', 'motion', 'ldr', 'outTemperature', 
                'outHumidity', 'outLdr', 'hour', 'target']
correlation_matrix = df[cols_to_corr].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Sensör Verileri Korelasyon Matrisi')
plt.savefig('1_korelasyon_heatmap.png')
plt.show()

#Zaman Serisi ve Etiket İlişkisi (Zaman Serisi Analizi)
# Örnek olarak ilk 500 dakikayı alalım (Görselin anlaşılır olması için)
sample_df = df.iloc[600:1100].copy() 

fig, ax1 = plt.subplots(figsize=(15, 6))

# Sol eksen: Sensörler
ax1.plot(sample_df.index, sample_df['temperature'], label='İç Sıcaklık', color='red', alpha=0.6)
ax1.plot(sample_df.index, sample_df['ldr'], label='LDR (Işık)', color='orange', alpha=0.6)
ax1.plot(sample_df.index, sample_df['motion'], label='Hareket', color='blue', linestyle='--')
ax1.set_ylabel('Normalize Değerler (0-1)')
ax1.legend(loc='upper left')

# Sağ eksen: Karar (Target)
ax2 = ax1.twinx()
ax2.fill_between(sample_df.index, 0, sample_df['target'], color='green', alpha=0.2, label='Kapatma Kararı (1)')
ax2.set_ylabel('Hedef Değişken (0: Açık, 1: Kapalı)')
ax2.set_ylim(-0.1, 1.1)
ax2.legend(loc='upper right')

plt.title('Sensör Değişimleri ve Otomatik Etiketleme İlişkisi')
plt.savefig('2_sensor_target_iliskisi.png')
plt.show()


#Model Eğitim Performansı
# Eğer eğitim sırasında 'history' değişkenini kaydettiysen bunu kullanabilirsin.
# Kaydetmediysen, eğitim loglarındaki değerlerle manuel bir sözlük oluşturabiliriz:

# Örnek veri (Senin loglarından alınmıştır)
history_dict = {
    'accuracy': [0.89, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.96],
    'val_accuracy': [0.99, 1.00, 1.00, 0.99, 0.99, 0.98, 0.95, 0.92],
    'loss': [0.26, 0.22, 0.20, 0.18, 0.15, 0.12, 0.09, 0.08],
    'val_loss': [0.02, 0.02, 0.03, 0.03, 0.05, 0.07, 0.05, 0.09]
}

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.figure(figsize=(14, 5))

# Accuracy Grafiği
plt.subplot(1, 2, 1)
plt.plot(epochs, history_dict['accuracy'], 'b', label='Eğitim Başarısı')
plt.plot(epochs, history_dict['val_accuracy'], 'r', label='Doğrulama Başarısı')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict['loss'], 'b', label='Eğitim Kaybı')
plt.plot(epochs, history_dict['val_loss'], 'r', label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('3_model_performans.png')
plt.show()


#Karışıklık Matrisi (Confusion Matrix)
from sklearn.metrics import confusion_matrix
import seaborn as sns

# test_model.py sonucundaki gerçek ve tahmin değerlerini buraya koymalısın
# Örnek (Senin raporundaki sayılar):
y_true = [0]*677 + [1]*2143
y_pred = [0]*568 + [1]*109 + [1]*2014 + [0]*129 # Raporundaki %92'ye göre simüle edilmiştir

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Dolu (0)', 'Boş (1)'], 
            yticklabels=['Dolu (0)', 'Boş (1)'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Durum')
plt.title('Karışıklık Matrisi (Confusion Matrix)')
plt.savefig('4_confusion_matrix.png')
plt.show()


#Enerji Tasarrufu Karşılaştırması (Bar Chart)
# Senin simülasyon sonuçların:
labels = ['İklimlendirme (HVAC)', 'Aydınlatma']
ai_tasarruf = [18, 7] # Dakika

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, ai_tasarruf, color=['skyblue', 'lightgreen'])

plt.ylabel('Kazanılan Süre (Dakika)')
plt.title('Yapay Zeka Destekli Günlük Zaman Tasarrufu (2 Günlük Test Verisi)')

# Değerleri barların üzerine yazalım
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom')

plt.savefig('5_tasarruf_ozeti.png')
plt.show()

#Veri İndirgeme ve Temizleme Analizi (Tablo ve Bar Chart)
import pandas as pd
import matplotlib.pyplot as plt

# Bu verileri kendi analiz sonuçlarına göre doldurmalısın
data_summary = {
    'Aşama': ['Ham Veri (10sn)', 'Hatalı/Gürültülü', 'Eksik Veri (Doldurulan)', 'Final Veri (1dk)'],
    'Veri Sayısı': [234886, 724, 312, 39321]
}

summary_df = pd.DataFrame(data_summary)

plt.figure(figsize=(10, 6))
colors = ['gray', 'red', 'orange', 'green']
bars = plt.bar(summary_df['Aşama'], summary_df['Veri Sayısı'], color=colors)

plt.yscale('log') # Veri sayıları arasındaki fark çoksa logaritmik eksen daha iyi görünür
plt.ylabel('Kayıt Sayısı (Logaritmik Ölçek)')
plt.title('Veri Hazırlama ve Temizleme Süreci İstatistikleri')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.savefig('6_veri_temizleme_istatistikleri.png')
plt.show()


#Personel Mevcut Durum Analizi (Activity Map)
import seaborn as sns

# Eğitim verinden saatlik yoğunluk haritası çıkaralım
# df = pd.read_csv('train_processed.csv') # Daha önce yüklediğimiz veri

gunler = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

# Saat ve gün bazında hareket (motion) ortalamasını alalım
pivot_table = df.pivot_table(index='day_of_week', columns='hour', values='motion', aggfunc='mean')

plt.figure(figsize=(15, 6))
sns.heatmap(pivot_table, annot=False, cmap='YlGnBu', yticklabels=gunler,xticklabels=range(0, 24))

plt.title('Haftalık Personel Mevcut Durum (Yoğunluk) Haritası', fontsize=15)
plt.xlabel('Günün Saatleri', fontsize=12)
plt.ylabel('Haftanın Günleri', fontsize=12)

# Eksen yazılarını daha düzgün durması için döndürelim
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout() # Grafiğin kenarlara sıkışmasını önler
plt.savefig('7_personel_yogunluk_haritasi.png')
plt.show()