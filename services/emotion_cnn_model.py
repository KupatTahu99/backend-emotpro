import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import cv2

class EmotionCNNModel:
    def __init__(self, model_path='models/model.h5'):
        self.model = self._build_model()
        self.load_weights(model_path)
        # Urutan label sesuai dataset training (Alphabetical)
        self.emotion_labels = {0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang", 4: "Netral", 5: "Sedih", 6: "Terkejut"}

    def _build_model(self):
        """
        Membangun ulang arsitektur CNN persis seperti di main.py Anda.
        Jika struktur ini beda sedikit saja, load_weights akan error.
        """
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        # Compile dummy agar struktur lengkap (optimizer tidak berpengaruh untuk inferensi)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        
        return model

    def load_weights(self, path):
        # Cek path absolut agar aman
        base_dir = os.getcwd()
        full_path = os.path.join(base_dir, path)
        
        # Fallback cek path
        if not os.path.exists(full_path):
             # Coba cek langsung di root jika tidak ada di folder models/
             full_path = os.path.join(base_dir, 'model.h5')

        if os.path.exists(full_path):
            try:
                self.model.load_weights(full_path)
                print(f"✅ [EmotionCNN] Model loaded from {full_path}")
            except Exception as e:
                print(f"❌ [EmotionCNN] Error loading weights: {e}")
        else:
            print(f"❌ [EmotionCNN] File not found: {full_path}")

    def predict_emotion(self, face_image):
        """
        Menerima gambar wajah (crop) dan mengembalikan prediksi emosi.
        """
        try:
            # 1. Preprocessing: Resize ke 48x48
            roi = cv2.resize(face_image, (48, 48))
            
            # 2. Pastikan Grayscale (jika input masih BGR/RGB)
            if len(roi.shape) == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 3. Normalisasi (0-1) - Penting! Karena training pakai rescale=1./255
            roi = roi.astype('float32') / 255.0

            # 4. Reshape untuk input model: (1, 48, 48, 1)
            roi = np.expand_dims(roi, axis=-1) # Tambah channel
            roi = np.expand_dims(roi, axis=0)  # Tambah batch

            # 5. Prediksi
            prediction = self.model.predict(roi, verbose=0)
            max_index = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][max_index])

            return {
                "emotion": self.emotion_labels[max_index],
                "confidence": confidence,
                "all_probabilities": prediction[0].tolist()
            }
        except Exception as e:
            print(f"Error prediction: {e}")
            return {"emotion": "Error", "confidence": 0.0}