import pandas as pd
import glob

# CSV dosyalarının bulunduğu klasör
csv_files = glob.glob("../newLogs/*.csv")

# Tüm dosyaları oku ve listeye al
df_list = [pd.read_csv(file) for file in csv_files]

# Hepsini birleştir
merged_df = pd.concat(df_list, ignore_index=True)

# Timestamp’e göre sırala (isteğe bağlı ama önerilir)
merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"], format='mixed')
merged_df = merged_df.sort_values("timestamp")

# Tek bir CSV olarak kaydet
merged_df.to_csv("all_sensors_merged.csv", index=False)

print("Birleştirme tamamlandı.")