import os
from django.core.wsgi import get_wsgi_application

from apps.endpoints.models import MLAlgorithm
from apps.ml.income_classifier.KerasModelPredictor import KerasModelPredictor

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.random_forest import RandomForestClassifier
from apps.ml.income_classifier.extra_trees import ExtraTreesClassifier # import ExtraTrees ML algorithm

try:
    registry = MLRegistry() # create ML registry
    MLAlgorithm.objects.filter(name="CNN", version="0.0.1").delete()

    # Extra Trees classifier
    ke = KerasModelPredictor()
    # add to ML registry
    registry.add_algorithm(endpoint_name="old_statistical_prediction",
                           algorithm_object=ke,
                           algorithm_name="CNN",
                           algorithm_status="production",
                           algorithm_version="0.0.1",
                           owner="Piotr",
                           algorithm_description="Extra Trees with simple pre- and post-processing",
                           algorithm_code=inspect.getsource(KerasModelPredictor))
except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))