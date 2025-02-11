import joblib
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # Загружаем Keras-модель

class KerasModelPredictor:
    def __init__(self):
        # Путь к артефактам модели
        self.path_to_artifacts = r"C:\Users\Nurkhan\diploma model"

        # Загружаем подготовленные файлы (кроме encoders)
        #self.values_fill_missing = joblib.load(os.path.join(self.path_to_artifacts, "train_mode.joblib"))

        # Загружаем Keras модель
        self.model = load_model(os.path.join(self.path_to_artifacts, "best_modelb.h5"))

        # Определяем порядок столбцов, который ожидает модель
        self.expected_columns = [
            'оценка качества здоровья', 'частота обращений к мед персоналу за год',
            'возраст', 'Мужчина', 'Женщина',
            'Обструктивные хронические болезни легких',
            'Ревматические (аутоимунные) заболевания',
            'Язвенные болезни желудка и 12 перстной кишки', 'Болезни печени',
            'Сахарный диабет', 'Злокачественные опухоли', 'Артериальная гипертония',
            'Инфекционные и паразитарные заболевания', 'Депрессия', 'Наркотическая',
            'Уборка в комнате', 'Работа во дворе, в палисаднике',
            'Заготовка воды и дров', 'Стирка белья, шитье', 'Готовка еды',
            'Уборка дома', 'Поход в продуктовый магазин и аптеку',
            'Продолжает работать', 'Участвует в общественных работах',
            'Активно общается с родственниками и соседями',
            'Чтение газет и журналов, просмотр телевизора',
            'Другие разные интересы'
        ]

    def preprocessing(self, input_data):
        """Предобработка входных данных"""
        input_data = pd.DataFrame(input_data, index=[0])

        # Заполняем пропущенные значения
        #input_data.fillna(self.values_fill_missing, inplace=True)

        # Убедимся, что колонки идут в правильном порядке
        input_data = input_data[self.expected_columns]

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
