import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

class WearableDevicePredictor:
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.abspath(os.path.join(current_dir, "..", "model"))

        self.lstm_model = load_model(os.path.join(model_dir, "health_pattern_model.h5"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.knn = joblib.load(os.path.join(model_dir, "knn_model.pkl"))

    def preprocessing(self, input_data):
        try:
            data = input_data["week_data"]
            df = pd.DataFrame(data)
            if df.shape[1] != 4:
                raise ValueError("Ожидается 4 признака: steps, deepSleepTime, shallowSleepTime, heartRate")

            return df
        except Exception as e:
            raise ValueError(f"Ошибка препроцессинга: {str(e)}")

    def create_sequences(self, data, seq_length=7):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    def compute_prediction(self, input_data):
        try:
            df = self.preprocessing(input_data)
            if len(df) < 8:
                raise ValueError("Требуется минимум 8 дней данных для анализа")

            data_scaled = self.scaler.transform(df)
            X_seq = self.create_sequences(data_scaled)

            encoded = self.lstm_model.predict(X_seq)
            encoded_flat = encoded.reshape(encoded.shape[0], -1)
            clusters = self.knn.predict(encoded_flat)

            type_names = {0: "Норма", 1: "Стресс", 2: "Отдых", 3: "Неопределено"}
            risks = {0: 0.1, 1: 0.8, 2: 0.2, 3: 0.5}

            result = []
            for i, cl in enumerate(clusters):
                result.append({
                    "day_index": i + 7,
                    "day_type": type_names.get(cl, "Неизвестно"),
                    "cluster": int(cl),
                    "risk": round(risks.get(cl, 0.5) * 100, 1),
                    "is_risk_day": cl in [1, 3]  # Стресс или Неопределено
                })

            return {"status": "OK", "result": result}
        except Exception as e:
            return {"status": "Error", "message": str(e)}
