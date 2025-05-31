import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # Загружаем Keras-модель

class YoungPeoplePredictor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Путь до папки с артефактами
        self.path_to_artifacts = os.path.join(current_dir, "..", "..", "modelsBox", "youngPeople")
        self.path_to_artifacts = os.path.abspath(self.path_to_artifacts)

        # Загружаем признаки и модель
        self.model = joblib.load(os.path.join(self.path_to_artifacts, "heart_disease_model.pkl"))

        # Определяем порядок столбцов, который ожидает модель

        self.expected_columns = ['age', 'gender', 'high_BP', 'diabetes', 'smoking', 'alcohol',
       'coffee_cups', 'energy_drinks', 'bmi', 'Kidney_Disease',
       'Liver_Disease', 'Hypertension', 'Lung_Disease', 'Thyroid_Disease',
       'Arthritis', 'Cancer', 'Neurological_Disease']

    def preprocessing(self, input_data):
        """Предобработка входных данных"""
        input_data = pd.DataFrame(input_data, index=[0])

        # Убедимся, что колонки идут в правильном порядке
        input_data = input_data[self.expected_columns]
        #input_data[self.numeric_features] = self.scaler.transform(input_data[self.numeric_features])

        return input_data

    def predict(self, input_data):
        """Получаем предсказание от модели"""
        input_data = np.array(input_data)  # Преобразуем в numpy массив
        predictions = self.model.predict_proba(input_data)[:, 1][0]
        return predictions

    def postprocessing(self, prediction):
        """Преобразование результата"""
        probability = prediction  # Если двумерный

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
