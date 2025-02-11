from django.apps import AppConfig

class EndpointsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.endpoints'  # Должно совпадать с названием в INSTALLED_APPS!
