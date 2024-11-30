from django.urls import path
from . import views

urlpatterns = [
    # URL pour le formulaire d'upload
    path('', views.upload_excel, name='upload_excel'),
]
