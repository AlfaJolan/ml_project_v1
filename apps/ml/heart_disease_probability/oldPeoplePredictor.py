import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # Загружаем Keras-модель

class KerasModelPredictor:
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.path_to_artifacts = os.path.abspath(
            os.path.join(current_dir, "..", "modelsBox", "oldPeople")
        )

        # Загружаем подготовленные файлы
        self.scaler = joblib.load(os.path.join(self.path_to_artifacts, "scaler.pkl"))
        self.numeric_features = joblib.load(os.path.join(self.path_to_artifacts, "numeric_features.pkl"))
        self.model = load_model(os.path.join(self.path_to_artifacts, "best_modelb2.h5"))

        # Определяем порядок столбцов, который ожидает модель

        self.expected_columns = [
            'health_score', 'med_visits', 'age', 'male', 'female', 'copd',
            'rheumatic', 'ulcers', 'liver_disease', 'diabetes', 'cancer',
            'hypertension', 'infectious_dis', 'depression', 'drug_use',
            'clean_room', 'yard_work', 'water_wood', 'laundry_sew', 'cook',
            'clean_house', 'shopping', 'still_working', 'community_work',
            'social_active', 'reading_tv', 'other_hobbies'
        ]

    def preprocessing(self, input_data):
        """Предобработка входных данных"""
        input_data = pd.DataFrame(input_data, index=[0])

        # Убедимся, что колонки идут в правильном порядке
        input_data = input_data[self.expected_columns]
        input_data[self.numeric_features] = self.scaler.transform(input_data[self.numeric_features])

        return input_data

    def predict(self, input_data):
        """Получаем предсказание от модели"""
        input_data = np.array(input_data)  # Преобразуем в numpy массив
        predictions = self.model.predict(input_data)
        return predictions

    def postprocessing(self, prediction):
        """Преобразование результата"""
        if len(prediction.shape) == 1:
            probability = prediction[0]  # Если массив одномерный
        else:
            probability = prediction[0][0]  # Если двумерный

        label = "Низкий риск"
        if probability > 0.5:
            label = "Высокий риск"

        return {"probability": float(probability), "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        """Общая функция предсказания"""
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            print("❌ Ошибка:", str(e))  # Покажет в логах точную ошибку
            return {"status": "Error", "message": str(e)}

        return prediction
