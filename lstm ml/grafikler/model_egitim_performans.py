import matplotlib.pyplot as plt
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
