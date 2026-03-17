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
colors = ['blue', 'red', 'orange', 'green']
bars = plt.bar(summary_df['Aşama'], summary_df['Veri Sayısı'], color=colors)

plt.yscale('log') # Veri sayıları arasındaki fark çoksa logaritmik eksen daha iyi görünür
plt.ylabel('Kayıt Sayısı (Logaritmik Ölçek)')
plt.title('Veri Hazırlama ve Temizleme Süreci İstatistikleri')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.savefig('6_veri_temizleme_istatistikleri.png')
plt.show()