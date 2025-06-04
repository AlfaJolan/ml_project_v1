import pandas as pd
import json


class NeurostimulatorRiskModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Сопоставление строковых уровней риска с числовыми штрафами
        self.risk_level_penalties = {
            "low": 0,
            "medium": self.config['penalties'].get('risk_level_medium', 10),
            "high": self.config['penalties'].get('risk_level_high', 25)
        }

    def get_limit(self, substance, age):
        limits = self.config['limits'][substance]
        if age < 18 and 'under_18' in limits:
            return limits['under_18']
        elif age >= 60 and 'over_60' in limits:
            return limits['over_60']
        else:
            return limits['default']

    def calculate_risk(self, age, bmi, risk_level_str, data: pd.DataFrame):
        risk_score = 0
        explanations = []
        alerts = []
        recommendations = []

        exceed_daily = {'coffee': [], 'energy': [], 'alcohol': []}

        for substance in ['coffee', 'energy', 'alcohol']:
            daily_limit = self.get_limit(substance, age)
            weekly_limit = self.config['weekly_limits'][substance]

            for idx, value in enumerate(data[substance]):
                if value > daily_limit:
                    risk_score += self.config['penalties']['exceed_daily']
                    explanations.append(
                        f"{substance.capitalize()} Day {idx + 1}: {value} > {daily_limit}, +{self.config['penalties']['exceed_daily']}")
                    exceed_daily[substance].append(1)
                else:
                    exceed_daily[substance].append(0)

            total_week = data[substance].sum()
            if total_week > weekly_limit:
                risk_score += self.config['penalties']['exceed_weekly']
                explanations.append(
                    f"{substance.capitalize()} weekly total {total_week} > {weekly_limit}, +{self.config['penalties']['exceed_weekly']}")
            else:
                explanations.append(f"{substance.capitalize()} weekly total {total_week} within limit {weekly_limit}")

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

            # Точечные советы по веществу
            if any(exceed_daily[substance]):
                if substance == 'coffee':
                    recommendations.append("Reduce coffee intake.")
                elif substance == 'energy':
                    recommendations.append("Limit energy drink consumption.")
                elif substance == 'alcohol':
                    recommendations.append("Cut down on alcohol.")

        for age_threshold, penalty in self.config['penalties']['age_risk'].items():
            if age >= int(age_threshold):
                risk_score += penalty
                explanations.append(f"Age {age} >= {age_threshold}: +{penalty}")
                break

        for bmi_threshold, penalty in self.config['penalties']['bmi_risk'].items():
            if bmi >= int(bmi_threshold):
                risk_score += penalty
                explanations.append(f"BMI {bmi} >= {bmi_threshold}: +{penalty}")
                break

        penalty_for_level = self.risk_level_penalties.get(risk_level_str, 0)
        risk_score += penalty_for_level
        explanations.append(f"Risk level '{risk_level_str}': +{penalty_for_level}")

        risk_score = min(risk_score, 100)

        if risk_score < 30:
            base_recommendation = "Low risk. Maintain a healthy lifestyle."
        elif risk_score < 70:
            base_recommendation = "Moderate risk."
        else:
            base_recommendation = "High risk! Consult a healthcare professional."

        if recommendations:
            full_recommendation = base_recommendation + " Recommendations: " + " ".join(recommendations)
        else:
            full_recommendation = base_recommendation

        alerts_text = "; ".join(alerts) if alerts else "No significant alerts."

        return {
            'risk_score': round(risk_score, 2),
            'recommendation': full_recommendation,
            'explanations': explanations,
            'alerts': alerts_text
        }

    def save_report(self, result_dict, filepath):
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=4)
        print(f"Saved report to {filepath}")
