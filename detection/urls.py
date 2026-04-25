from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_sign,  name='predict_sign'),
    path('frame/',   views.predict_frame, name='predict_frame'),
]