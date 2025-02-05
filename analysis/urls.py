from django.urls import path
from . import views

urlpatterns = [
    # URL pour le formulaire d'upload
    path('upload/', views.upload_excel, name='upload_excel'),
    path('', views.index, name='dashboard_view'),
    path('Intro', views.intro, name='intro'),
    path('report/pdf/', views.generate_report_pdf, name='generate_report_pdf'),
path('test_map/', views.map_view, name='test_map'),
 path('maps/', views.map_view, name='maps'),  # The main Bokeh map
    path('get_drill_hole_data/<str:hole_id>/', views.get_drill_hole_data, name='get_drill_hole_data'),
    path('get_3d_model_for_hole/<str:hole_id>/', views.get_3d_model_for_hole, name='get_3d_model_data'),
    path('charts/page/<str:page_id>/', views.bokeh_charts_page, name='bokeh_charts_page'),

]
