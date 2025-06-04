import os
from django.core.wsgi import get_wsgi_application

from apps.endpoints.models import MLAlgorithm
from apps.ml.heart_disease_probability.WearableDevicePredictor import WearableDevicePredictor
from apps.ml.heart_disease_probability.oldPeoplePredictor import oldPeoplePredictor
from apps.ml.heart_disease_probability.neurostimulatorRiskPredictor import NeurostimulatorRiskPredictor
from apps.ml.heart_disease_probability.young_people_predictor import YoungPeoplePredictor

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.heart_disease_probability.random_forest import RandomForestClassifier
from apps.ml.heart_disease_probability.extra_trees import ExtraTreesClassifier # import ExtraTrees ML algorithm

try:
    registry = MLRegistry() # create ML registry
    MLAlgorithm.objects.filter(name="CNN", version="0.0.1").delete()
    # Не удаляем MLEndpoint
    MLAlgorithm.objects.all().delete()

    # Extra Trees classifier
    oldPeople = oldPeoplePredictor()
    # add to ML registry
    registry.add_algorithm(endpoint_name="old_statistical_prediction_v1",
                           algorithm_object=oldPeople,
                           algorithm_name="CNN",
                           algorithm_status="production",
                           algorithm_version="0.0.1",
                           owner="Nurkhan",
                           algorithm_description="Extra Trees with simple pre- and post-processing",
                           algorithm_code=inspect.getsource(oldPeoplePredictor))

    young_model = YoungPeoplePredictor()
    MLAlgorithm.objects.filter(name="random_tree", version="0.0.1").delete()
    registry.add_algorithm(endpoint_name="young_statistical_prediction",
                           algorithm_object=young_model,
                           algorithm_name="random_tree",
                           algorithm_status="production",
                           algorithm_version="0.0.1",
                           owner="Nurkhan",
                           algorithm_description="Random Tree Classifier probability prediction",
                           algorithm_code=inspect.getsource(YoungPeoplePredictor))
    neuro_model = NeurostimulatorRiskPredictor()
    registry.add_algorithm(
        endpoint_name="neurostimulator_risk",
        algorithm_object=neuro_model,
        algorithm_name="neuro_model",
        algorithm_status="production",
        algorithm_version="0.0.1",
        owner="Nurkhan",
        algorithm_description="Model for assessing neurostimulation risk using 7-day drink intake data",
        algorithm_code=inspect.getsource(NeurostimulatorRiskPredictor)
    )

    # === Инициализируем предиктор ===
    wearable_model = WearableDevicePredictor()

    # === Регистрируем в реестре ===
    registry.add_algorithm(
        endpoint_name="wearable_health_analysis",
        algorithm_object=wearable_model,
        algorithm_name="wearable_lstm_knn",
        algorithm_status="production",
        algorithm_version="0.0.1",
        owner="Nurkhan",
        algorithm_description="LSTM + KNN модель для анализа данных с носимых устройств и выявления потенциальных дней риска для сердца",
        algorithm_code=inspect.getsource(WearableDevicePredictor)
    )
except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))