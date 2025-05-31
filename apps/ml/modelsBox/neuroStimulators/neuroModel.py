import pandas as pd
import json


class NeurostimulatorRiskModel:
    def __init__(self, config_path):
        # Загружаем конфиг
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get_limit(self, substance, age):
        """Получить лимит в зависимости от возраста"""
        limits = self.config['limits'][substance]
        if age < 18 and 'under_18' in limits:
            return limits['under_18']
        elif age >= 60 and 'over_60' in limits:
            return limits['over_60']
        else:
            return limits['default']

    def calculate_risk(self, age, bmi, has_heart_disease, data: pd.DataFrame):
        risk_score = 0
        explanations = []
        alerts = []

        # Флаги превышений
        exceed_daily = {'coffee': [], 'energy': [], 'alcohol': []}

        # Анализ по каждому веществу
        for substance in ['coffee', 'energy', 'alcohol']:
            daily_limit = self.get_limit(substance, age)
            weekly_limit = self.config['weekly_limits'][substance]

            # Проверка по дням
            for idx, value in enumerate(data[substance]):
                if value > daily_limit:
                    risk_score += self.config['penalties']['exceed_daily']
                    explanations.append(
                        f"{substance.capitalize()} Day {idx + 1}: {value} > {daily_limit}, +{self.config['penalties']['exceed_daily']}")
                    exceed_daily[substance].append(1)
                else:
                    exceed_daily[substance].append(0)

            # Сумма за неделю
            total_week = data[substance].sum()
            if total_week > weekly_limit:
                risk_score += self.config['penalties']['exceed_weekly']
                explanations.append(
                    f"{substance.capitalize()} weekly total {total_week} > {weekly_limit}, +{self.config['penalties']['exceed_weekly']}")
            else:
                explanations.append(f"{substance.capitalize()} weekly total {total_week} within limit {weekly_limit}")

            # Поиск серий превышений
            series = []
            count = 0
            for flag in exceed_daily[substance]:
                if flag:
                    count += 1
                else:
                    if count > 1:
                        series.append(count)
                    count = 0
            if count > 1:
                series.append(count)

            if series:
                penalty_series = len(series) * self.config['penalties']['consecutive_bonus']
                risk_score += penalty_series
                explanations.append(f"{substance.capitalize()} consecutive exceed series {series}, +{penalty_series}")
                alerts.append(
                    f"{substance.capitalize()}: {len(series)} consecutive exceed series, max {max(series)} days")

        # Возрастные риски
        for age_threshold, penalty in self.config['penalties']['age_risk'].items():
            if age >= int(age_threshold):
                risk_score += penalty
                explanations.append(f"Age {age} >= {age_threshold}: +{penalty}")
                break  # Применяем только один порог

        # BMI риск
        for bmi_threshold, penalty in self.config['penalties']['bmi_risk'].items():
            if bmi >= int(bmi_threshold):
                risk_score += penalty
                explanations.append(f"BMI {bmi} >= {bmi_threshold}: +{penalty}")
                break

        # Риск болезни сердца
        if has_heart_disease:
            risk_score += self.config['penalties']['heart_disease']
            explanations.append(f"Has heart disease: +{self.config['penalties']['heart_disease']}")

        # Ограничим максимум 100
        risk_score = min(risk_score, 100)

        # Вывод рекомендаций
        if risk_score < 30:
            recommendation = "Low risk. Maintain healthy lifestyle."
        elif risk_score < 70:
            recommendation = "Moderate risk. Reduce caffeine/energy drink intake."
        else:
            recommendation = "High risk! Consult a doctor for advice."

        return {
            'risk_score': risk_score,
            'recommendation': recommendation,
            'explanations': explanations,
            'alerts': alerts
        }

    def save_report(self, result_dict, filepath):
        """Сохранить результат в JSON"""
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=4)
        print(f"Saved report to {filepath}")
