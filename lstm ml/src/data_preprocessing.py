import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

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
                    
    # ==========================================
    # DÜZELTME: REGRESYONDAN SINIFLANDIRMAYA GEÇİŞ
    # ==========================================
    
    # 1. Hedef: Yakın zamanda (15 dk içinde) ofisten ayrılacak mı? (Evet: 1, Hayır: 0)
    time_class = (mins_until_leave <= 15).astype(int)
    
    # 2. Hedef: Çıkarsa ne kadar süre kalacak? (Kategori 0, 1, 2)
    dur_class = np.zeros_like(leave_duration, dtype=int)
    dur_class[(leave_duration > 15) & (leave_duration <= 60)] = 1 # Kategori 1: 15-60 Dk
    dur_class[leave_duration > 60] = 2                            # Kategori 2: > 60 Dk
    
    return time_class, dur_class

def load_and_preprocess(filepath='new_sensors_data.csv', sequence_length=60):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.set_index('timestamp').resample('1min').ffill().reset_index()
    df = df.bfill()
    
    df['motion_clean'] = clean_motion_data(df['motion'].values, min_duration=10)
    
    # Artık dakikalar değil, sınıflar dönüyor
    time_class, dur_class = calculate_targets(df['motion_clean'].values)
    df['time_class'] = time_class
    df['dur_class'] = dur_class
    
    day_of_week = df['timestamp'].dt.dayofweek
    hour = df['timestamp'].dt.hour
    minute = df['timestamp'].dt.minute
    
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['minute_sin'] = np.sin(2 * np.pi * minute / 60.0)
    df['minute_cos'] = np.cos(2 * np.pi * minute / 60.0)
    
    features = ['temperature', 'humidity', 'motion_clean', 'ldr', 'outTemperature', 
                'outHumidity', 'outLdr', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 
                'minute_sin', 'minute_cos']
    
    # Sadece X (girdiler) scale edilecek. Y (hedefler) zaten sınıf etiketleri (0, 1, 2)
    scaler_X = MinMaxScaler()
    scaled_X = scaler_X.fit_transform(df[features])
    joblib.dump(scaler_X, 'scaler_X.pkl')
    
    targets = ['time_class', 'dur_class']
    y_data = df[targets].values 
    
    X_seq, y_seq = [], []
    for i in range(len(scaled_X) - sequence_length):
        X_seq.append(scaled_X[i:(i + sequence_length)])
        y_seq.append(y_data[i + sequence_length])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print("Veri ön işleme tamamlandı!")
    print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")