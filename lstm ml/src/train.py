from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def train_model():
    print("Veriler hazırlanıyor...")
    X_train, X_test, y_train, y_test = load_and_preprocess(filepath='new_sensors_data.csv')
    
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    x = LSTM(128, activation='relu', return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64, activation='relu', return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    # ==========================================
    # GÖREV 1: 15 DK İÇİNDE ÇIKACAK MI? (İkili Sınıflandırma - Sigmoid)
    # ==========================================
    branch_time = Dense(32, activation='relu')(x)
    out_time = Dense(1, activation='sigmoid', name='time_output')(branch_time)
    
    # ==========================================
    # GÖREV 2: NE KADAR KALACAK? (Çoklu Sınıflandırma: 3 Sınıf - Softmax)
    # ==========================================
    branch_dur = Dense(32, activation='relu')(x)
    out_dur = Dense(3, activation='softmax', name='dur_output')(branch_dur)
    
    model = Model(inputs=inputs, outputs=[out_time, out_dur])
    
    # Hata fonksiyonları ve metrikler sınıflandırmaya uyarlandı
    model.compile(optimizer='adam', 
                  loss={'time_output': 'binary_crossentropy', 'dur_output': 'sparse_categorical_crossentropy'},
                  loss_weights={'time_output': 1.0, 'dur_output': 1.0},
                  metrics={'time_output': 'accuracy', 'dur_output': 'accuracy'})
    
    y_train_time = y_train[:, 0]
    y_train_dur = y_train[:, 1]
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint('lstm_prof_model.keras', monitor='val_loss', save_best_only=True)
    
    print("Model eğitimi başlıyor...")
    history = model.fit(
        X_train, {'time_output': y_train_time, 'dur_output': y_train_dur}, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.1, 
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    print("Model başarıyla kaydedildi.")

    # ==========================================
    # EĞİTİM PERFORMANSI GRAFİKLERİ (ACCURACY VE LOSS)
    # ==========================================
    plt.figure(figsize=(16, 6))
    
    # 1. Çıkış Zamanı Doğruluğu (Zamanlama Sınıflandırması)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['time_output_accuracy'], label='Eğitim Başarısı')
    plt.plot(history.history['val_time_output_accuracy'], label='Doğrulama Başarısı')
    plt.title('Yakın Zamanda Çıkış Tahmini Başarısı')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.legend()
    
    # 2. Dışarıda Kalma Süresi Doğruluğu (Süre Sınıflandırması)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dur_output_accuracy'], label='Eğitim Başarısı')
    plt.plot(history.history['val_dur_output_accuracy'], label='Doğrulama Başarısı')
    plt.title('Dışarıda Kalma Süresi (Kategori) Tahmini Başarısı')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sınıflandırma_model_performans.png')
    plt.show()

    # Modelin Test Setindeki Toplam Başarısı
    print("\nTest Verisi Değerlendirmesi:")
    y_test_time = y_test[:, 0]
    y_test_dur = y_test[:, 1]
    results = model.evaluate(X_test, {'time_output': y_test_time, 'dur_output': y_test_dur}, verbose=0)
    print(f"Test Zaman Tahmini Doğruluğu (Time Acc): %{results[3]*100:.2f}")
    print(f"Test Süre Tahmini Doğruluğu (Dur Acc): %{results[4]*100:.2f}")

    # ==========================================
    # KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) ÇİZİMİ
    # ==========================================
    print("\nKarmaşıklık Matrisleri Oluşturuluyor...")
    
    # Test verisi üzerinden tahminleri al (Olasılık değerleri döner)
    predictions = model.predict(X_test)
    
    # 1. Görev Tahminleri: Çıkış Zamanı (Sigmoid çıkışı olduğu için 0.5 eşiğini kullanıyoruz)
    pred_time_prob = predictions[0].flatten()
    pred_time_class = (pred_time_prob > 0.45).astype(int)
    
    # 2. Görev Tahminleri: Süre (Softmax çıkışı olduğu için en yüksek olasılıklı sınıfı alıyoruz)
    pred_dur_class = np.argmax(predictions[1], axis=1)
    
    # Gerçek test etiketleri
    y_test_time = y_test[:, 0].astype(int)
    y_test_dur = y_test[:, 1].astype(int)
    
    # Matrisleri hesapla
    cm_time = confusion_matrix(y_test_time, pred_time_class)
    cm_dur = confusion_matrix(y_test_dur, pred_dur_class)
    
    # Matrisleri görselleştir
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sol Grafik: Çıkış Zamanı
    disp_time = ConfusionMatrixDisplay(confusion_matrix=cm_time, display_labels=['Çıkmayacak (0)', 'Çıkacak (1)'])
    disp_time.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Çıkış Zamanı: Gerçek vs Tahmin')
    
    # Sağ Grafik: Dışarıda Kalma Süresi
    disp_dur = ConfusionMatrixDisplay(confusion_matrix=cm_dur, display_labels=['Kısa Mola (0)', 'Uzun Mola (1)', 'Paydos (2)'])
    disp_dur.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title('Dışarıda Kalma Süresi: Gerçek vs Tahmin')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    train_model()