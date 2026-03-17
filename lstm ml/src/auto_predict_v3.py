import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1. YARDIMCI FONKSİYONLAR ---

def clean_motion_data(motion_array, min_duration=10):
    m = motion_array.copy()
    changed = True
    while changed:
        changed = False
        runs = []
        start = 0
        for i in range(1, len(m)):
            if m[i] != m[i-1]:
                runs.append((start, i, m[start]))
                start = i
        runs.append((start, len(m), m[start]))
        
        for start, end, val in runs:
            if (end - start) < min_duration:
                m[start:end] = 1 - val
                changed = True
                break
    return m

def calculate_targets(motion_array):
    n = len(motion_array)
    mins_until_leave = np.zeros(n)
    leave_duration = np.zeros(n)
    
    current_leave_idx = -1
    
    for i in range(n-1, -1, -1):
        if motion_array[i] == 0:
            pass 
        elif motion_array[i] == 1:
            if i == n - 1 or motion_array[i+1] == 0:
                current_leave_idx = i + 1 if i != n-1 else -1
                
                if current_leave_idx != -1:
                    current_return_idx = -1
                    for j in range(current_leave_idx, n):
                        if motion_array[j] == 1:
                            current_return_idx = j
                            break
                    
                    if current_return_idx != -1:
                        dur = current_return_idx - current_leave_idx
                        leave_duration[i] = min(dur, 120) 
                    else:
                        leave_duration[i] = 120 
            
            if current_leave_idx != -1:
                mins_until_leave[i] = min(current_leave_idx - i, 120)
                if i < n-1 and motion_array[i+1] == 1:
                     leave_duration[i] = leave_duration[i+1]
        else:
            if current_leave_idx != -1:
                mins_until_leave[i] = min(current_leave_idx - i, 120)
                if i < n-1:
                    leave_duration[i] = leave_duration[i+1]
                    
    # Sınıflandırma Mantığı
    time_class = (mins_until_leave <= 15).astype(int) 
    
    dur_class = np.zeros_like(leave_duration, dtype=int)
    dur_class[(leave_duration > 15) & (leave_duration <= 60)] = 1 
    dur_class[leave_duration > 60] = 2                            
    
    return time_class, dur_class

# --- 2. ANA SİMÜLASYON FONKSİYONU ---

def run_simulation(csv_path, seq_len=60, grace_period=20, override_timeout=10, empty_timeout=30, light_timeout=10, ldr_threshold=500):
    print("Model ve veriler yükleniyor. Simülasyon başlıyor...")
    
    try:
        model = load_model('lstm_prof_model.keras')
        scaler_X = joblib.load('scaler_X.pkl')
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60.0)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60.0)

    # Veriden tarih ve gün bilgisini çıkartıyoruz
    tr_days = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    first_date = df['timestamp'].iloc[0]
    date_str = f"{first_date.strftime('%d.%m.%Y')} {tr_days[first_date.dayofweek]}"

    df['motion_clean'] = clean_motion_data(df['motion'].values, min_duration=1)
    actual_time_class, actual_dur_class = calculate_targets(df['motion_clean'].values)
    
    features = ['temperature', 'humidity', 'motion_clean', 'ldr', 'outTemperature', 
                'outHumidity', 'outLdr', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 
                'minute_sin', 'minute_cos']

    plot_times = []
    plot_actual_time = []
    plot_pred_time_prob = []
    plot_actual_dur = []
    plot_predicted_dur = []
    plot_ac_command = []
    
    plot_ldr_in = []
    plot_ldr_out = []
    plot_light_command = []
    
    continuous_time_in_room = 0 
    time_since_off = 0          
    continuous_time_empty = 0

    print(f"Günlük veri dakika dakika taranıyor (Time-lapse) - Tarih: {date_str}...")
    
    for i in range(seq_len, len(df)):
        current_time = df['timestamp'].iloc[i]
        current_motion = df['motion_clean'].iloc[i]
        
        current_ldr_in = df['ldr'].iloc[i]
        current_ldr_out = df['outLdr'].iloc[i]
        
        act_t_class = actual_time_class[i]
        act_d_class = actual_dur_class[i]
        
        # --- AYDINLATMA KARAR MEKANİZMASI ---
        is_daylight_bright = (current_ldr_out < ldr_threshold)
        
        if current_motion == 0:
            continuous_time_empty += 1 
            
            if continuous_time_empty >= light_timeout:
                plot_light_command.append("Kapat (Boş Oda)")
            elif is_daylight_bright:
                plot_light_command.append("Kapat (Aydınlık)")
            else:
                plot_light_command.append("Açık")
                
            # Klima Kararı (Mevcut yapı)
            if continuous_time_empty >= empty_timeout:
                plot_ac_command.append("Kapat") 
            else:
                plot_ac_command.append("-")
            
            plot_times.append(current_time)
            plot_actual_time.append(np.nan)    
            plot_pred_time_prob.append(np.nan) 
            plot_actual_dur.append(np.nan)
            plot_predicted_dur.append(np.nan)
            plot_ldr_in.append(current_ldr_in)
            plot_ldr_out.append(current_ldr_out)
            
            continuous_time_in_room = 0
            time_since_off = 0
            continue
        else:
            continuous_time_empty = 0
            if is_daylight_bright:
                plot_light_command.append("Kapat (Aydınlık)")
            else:
                plot_light_command.append("Açık")
            
        recent_data = df.iloc[i - seq_len + 1 : i + 1]
        recent_features = recent_data[features]
        recent_scaled = scaler_X.transform(recent_features)
        X_pred = np.array([recent_scaled])
        
        pred_time_scaled, pred_dur_scaled = model.predict(X_pred, verbose=0)
        
        pred_time_prob = pred_time_scaled[0][0]
        pred_time_class = int(pred_time_prob > 0.45) 
        pred_dur_class = int(np.argmax(pred_dur_scaled[0]))
        
        # --- KLİMA KARAR MEKANİZMASI ---
        continuous_time_in_room += 1 
        
        if continuous_time_in_room < grace_period:
            ac_command = "Açık"
            time_since_off = 0
        else:
            if pred_time_class == 1: 
                if time_since_off == 0:
                    ac_command = "Kapat"
                    time_since_off = 1
                else:
                    time_since_off += 1
                    if time_since_off >= override_timeout:
                        ac_command = "Açık"
                        time_since_off = 0
                        continuous_time_in_room = 0 
                    else:
                        ac_command = "Kapat"
            else:
                ac_command = "Açık"
                time_since_off = 0
                
        plot_ac_command.append(ac_command)

        plot_times.append(current_time)
        plot_actual_time.append(act_t_class)
        plot_pred_time_prob.append(pred_time_prob)
        plot_actual_dur.append(act_d_class)
        plot_predicted_dur.append(pred_dur_class)
        plot_ldr_in.append(current_ldr_in)
        plot_ldr_out.append(current_ldr_out)

    # --- 3. METRİKLER VE TABLO OLUŞTURMA ---
    correct_time_preds = 0
    correct_dur_preds = 0
    total_valid_preds = 0
    results_data = []
    
    dur_labels = {0: "Kısa (0-15dk)", 1: "Uzun (15-60dk)", 2: "Paydos (>60dk)"}

    for t, act_t, pred_prob, act_d, pred_d, ac_cmd, light_cmd in zip(plot_times, plot_actual_time, plot_pred_time_prob, plot_actual_dur, plot_predicted_dur, plot_ac_command, plot_light_command):
        if pd.notna(act_t) and pd.notna(pred_prob):
            total_valid_preds += 1
            pred_t_class = int(pred_prob > 0.5)
            if pred_t_class == act_t: correct_time_preds += 1
            if pred_d == act_d: correct_dur_preds += 1
            
            results_data.append({
                'Zaman': t.strftime('%H:%M'),
                'Tahmin (%)': f"%{pred_prob*100:.0f}",
                'Tahmin Süre': dur_labels.get(pred_d, '-'),
                'Klima': ac_cmd,
                'Işık': light_cmd
            })
        elif ac_cmd in ["Kapat", "-"]:
            results_data.append({
                'Zaman': t.strftime('%H:%M'),
                'Tahmin (%)': '-',
                'Tahmin Süre': '-',
                'Klima': 'Kapat (Boş Oda)' if ac_cmd == "Kapat" else '-',
                'Işık': light_cmd
            })
            
    df_results = pd.DataFrame(results_data)
    df_results.to_csv('simulation_detailed_results.csv', index=False)
    
    acc_time = (correct_time_preds / total_valid_preds) * 100 if total_valid_preds > 0 else 0
    acc_dur = (correct_dur_preds / total_valid_preds) * 100 if total_valid_preds > 0 else 0

    # --- 4. GRAFİK ÇİZİMİ VE KAYDETME ---
    print("Grafikler oluşturuluyor ve kaydediliyor...")
    
    output_dir = "grafikler"
    os.makedirs(output_dir, exist_ok=True)
    
    # Eksen için saat başlarını ve formatı belirliyoruz
    hour_locator = mdates.HourLocator(interval=1)  # Her 1 saatte bir (24 saat)
    time_fmt = mdates.DateFormatter('%H:%M')

    # --- GRAFİK 1: ÇIKIŞ İHTİMALİ VE KLİMA KONTROLÜ ---
    fig1, ax_plot = plt.subplots(figsize=(14, 6))
    
    ax_motion1 = ax_plot.twinx() 
    ax_motion1.fill_between(df['timestamp'], 0, df['motion_clean'], color='royalblue', alpha=0.08)
    ax_motion1.set_ylim(-0.1, 5) 
    ax_motion1.axis('off') 
    
    ax_plot.fill_between(plot_times, 0, plot_actual_time, color='green', alpha=0.2, label='Gerçekte 15dk İçinde Çıkıldı')
    ax_plot.plot(plot_times, plot_pred_time_prob, label=f'Modelin Çıkış Tahmini (% İhtimal)', color='red', linewidth=2)
    ax_plot.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5, label='Karar Eşiği (%45)')
    
    is_ac_off = np.array([cmd == "Kapat" for cmd in plot_ac_command])
    ax_plot.fill_between(plot_times, 0, 1, where=is_ac_off, color='orange', alpha=0.3, 
                         transform=ax_plot.get_xaxis_transform(), label=f'Klima KAPAT Komutu')

    ax_plot.set_title(f'[{date_str}] Akıllı Bina Simülasyonu: 1) Ofisten Çıkış İhtimali ve Otonom Klima', fontsize=14, fontweight='bold')
    ax_plot.set_xlabel('Zaman (Saatler)', fontsize=12)
    ax_plot.set_ylabel('Çıkış İhtimali (0.0 - 1.0)', fontsize=12)
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.grid(True, linestyle=':', alpha=0.7)
    ax_plot.legend(loc='upper left', fontsize=10)
    
    # 24 Saati yazdırmak için yeni eksen ayarları:
    ax_plot.xaxis.set_major_locator(hour_locator)
    ax_plot.xaxis.set_major_formatter(time_fmt)
    fig1.autofmt_xdate(rotation=45)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, '1_cikis_ihtimali_ve_klima.png'), dpi=300)
    plt.close(fig1)

    # --- GRAFİK 2: DIŞARIDA KALMA SÜRESİ KATEGORİSİ ---
    fig2, ax_dur = plt.subplots(figsize=(14, 6))
    
    ax_motion2 = ax_dur.twinx() 
    ax_motion2.fill_between(df['timestamp'], 0, df['motion_clean'], color='royalblue', alpha=0.08)
    ax_motion2.set_ylim(-0.1, 5) 
    ax_motion2.axis('off')
    
    ax_dur.scatter(plot_times, plot_actual_dur, label='Gerçek Süre Sınıfı', color='teal', s=60, alpha=0.7)
    ax_dur.step(plot_times, plot_predicted_dur, label='Tahmin Edilen Sınıf', color='darkorange', linewidth=2, where='mid')
    
    ax_dur.set_title(f'[{date_str}] 2) Odadan Uzak Kalma Süresi Tahmini (Sınıflandırma)', fontsize=14, fontweight='bold')
    ax_dur.set_xlabel('Zaman (Saatler)', fontsize=12)
    ax_dur.set_ylabel('Süre Kategorisi', fontsize=12)
    ax_dur.set_yticks([0, 1, 2])
    ax_dur.set_yticklabels(["Kısa Mola\n(0-15dk)", "Uzun Mola\n(15-60dk)", "Paydos\n(>60dk)"])
    ax_dur.grid(True, linestyle=':', alpha=0.7)
    ax_dur.legend(loc='upper left', fontsize=10)
    
    # 24 Saati yazdırmak için yeni eksen ayarları:
    ax_dur.xaxis.set_major_locator(hour_locator)
    ax_dur.xaxis.set_major_formatter(time_fmt)
    fig2.autofmt_xdate(rotation=45)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, '2_sure_tahmini.png'), dpi=300)
    plt.close(fig2)

    # --- GRAFİK 3: LDR VE AYDINLATMA ---
    fig3, ax_light = plt.subplots(figsize=(14, 6))
    
    ax_motion3 = ax_light.twinx() 
    ax_motion3.fill_between(df['timestamp'], 0, df['motion_clean'], color='royalblue', alpha=0.08)
    ax_motion3.set_ylim(-0.1, 5) 
    ax_motion3.axis('off')
    
    ax_light.plot(plot_times, plot_ldr_in, label='İç Mekan LDR Değeri', color='darkgoldenrod', linewidth=2)
    ax_light.plot(plot_times, plot_ldr_out, label='Dış Mekan LDR Değeri', color='sienna', linewidth=1.5, linestyle='--')
    
    ax_light.axhline(y=ldr_threshold, color='olive', linestyle='--', alpha=0.8, label=f'Gün Işığı Eşiği ({ldr_threshold} LDR)')
    
    is_light_off_empty = np.array([cmd == "Kapat (Boş Oda)" for cmd in plot_light_command])
    is_light_off_daylight = np.array([cmd == "Kapat (Aydınlık)" for cmd in plot_light_command])

    ax_light.fill_between(plot_times, 0, 1, where=is_light_off_empty, color='black', alpha=0.2, 
                         transform=ax_light.get_xaxis_transform(), label=f'KAPAT (10dk Boş Kuralı)')
    
    ax_light.fill_between(plot_times, 0, 1, where=is_light_off_daylight, color='gold', alpha=0.2, 
                         transform=ax_light.get_xaxis_transform(), label=f'KAPAT (Gün Işığı Tasarrufu)')

    ax_light.set_title(f'[{date_str}] 3) Otonom Aydınlatma: Yüksek Değer = Karanlık / Düşük Değer = Aydınlık', fontsize=14, fontweight='bold')
    ax_light.set_xlabel('Gün İçindeki Zaman (Saatler)', fontsize=12)
    
    ax_light.invert_yaxis()
    ax_light.set_ylabel('Işık Şiddeti (LDR)\n↑ Daha Aydınlık', fontsize=12)
    
    ax_light.grid(True, linestyle=':', alpha=0.7)
    ax_light.legend(loc='upper left', fontsize=10)
    
    # 24 Saati yazdırmak için yeni eksen ayarları:
    ax_light.xaxis.set_major_locator(hour_locator)
    ax_light.xaxis.set_major_formatter(time_fmt)
    fig3.autofmt_xdate(rotation=45) 
    
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, '3_aydinlatma_ve_ldr.png'), dpi=300)
    plt.close(fig3)

    # --- 5. ÖZET METİN VE TABLO (AYRI GÖRSEL) ---
    fig4, ax_text = plt.subplots(figsize=(10, 16)) 
    ax_text.axis('off') 
    
    stats_text = (
        f"SİMÜLASYON ÖZET RAPORU ({date_str})\n"
        "--------------------------------------------------\n"
        f"Geçerli Tahmin: {total_valid_preds}\n"
        f"Klima Oto Kapanma: {empty_timeout} dk (Boş)\n"
        f"Işık Oto Kapanma: {light_timeout} dk (Boş)\n"
        f"Gün Işığı Eşiği: < {ldr_threshold} LDR\n"
        f"Çıkış Tahmini Doğruluğu: %{acc_time:.2f}\n"
        f"Süre Kategorisi Doğruluğu: %{acc_dur:.2f}\n"
    )
    
    ax_text.text(0.5, 0.98, stats_text, transform=ax_text.transAxes, fontsize=12, weight='bold',
                 verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.5))

    if not df_results.empty:
        sample_step = max(1, len(df_results)//40)
        table_sample = df_results.iloc[::sample_step].head(40).copy() 
        table_sample.columns = ['Zaman', 'Tahmin(%)', 'T.Süre', 'Klima', 'Işık']
        
        table = ax_text.table(cellText=table_sample.values,
                              colLabels=table_sample.columns,
                              cellLoc='center',
                              loc='center',
                              bbox=[0.0, 0.0, 1.0, 0.82]) 
        
        table.auto_set_font_size(False)
        table.set_fontsize(9) 
        
        for (row, col), cell in table.get_celld().items():
            if row == 0: 
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4CAF50')
                cell.set_height(0.04)  
            else: 
                cell.set_height(0.02)  

    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, '4_ozet_ve_tablo.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("İşlem tamam! Tüm grafikler 'grafikler' klasörüne ayrı ayrı kaydedildi.")

if __name__ == "__main__":
    # ldr_threshold değeri dış mekanın aydınlık sayılacağı maksimum ldr değeridir.
    run_simulation('current_status_22_pzrts.csv', grace_period=20, override_timeout=25, empty_timeout=30, light_timeout=10, ldr_threshold=400)