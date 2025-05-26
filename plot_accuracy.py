import pickle
import matplotlib.pyplot as plt

# Load history
with open('model/history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plot training accuracy
plt.plot(history['accuracy'], label='Training Accuracy')
plt.title('Grafik Akurasi Selama Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
