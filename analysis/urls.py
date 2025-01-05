from django.urls import path
from . import views

urlpatterns = [
    # URL pour le formulaire d'upload
    path('upload/', views.upload_excel, name='upload_excel'),
    path('', views.dashboard_view, name='dashboard_view'),
    path('report/pdf/', views.generate_report_pdf, name='generate_report_pdf'),

]
