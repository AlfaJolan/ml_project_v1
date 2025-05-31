import os
import pandas as pd
from apps.ml.modelsBox.neuroStimulators.neuroModel import NeurostimulatorRiskModel


class NeurostimulatorRiskPredictor:
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "..", "modelsBox", "neuroStimulators", "config.json")
        config_path = os.path.abspath(config_path)

        self.model = NeurostimulatorRiskModel(config_path)

    def preprocessing(self, input_data):
        try:
            age = input_data["age"]
            bmi = input_data["bmi"]
            has_heart_disease = input_data["has_heart_disease"]
            week_data = input_data["week_data"]

            df = pd.DataFrame(week_data)
            return age, bmi, has_heart_disease, df
        except Exception as e:
            raise ValueError(f"Ошибка препроцессинга: {str(e)}")

    def compute_prediction(self, input_data):
        try:
            age, bmi, has_heart_disease, df = self.preprocessing(input_data)
            result = self.model.calculate_risk(age, bmi, has_heart_disease, df)
            result["status"] = "OK"
            return result
        except Exception as e:
            return {"status": "Error", "message": str(e)}
