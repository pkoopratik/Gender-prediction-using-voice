from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('', views.index,name="index"),
    path('upload', views.uploadfile,name="uploadfile"),
    path('male', views.male,name="male"),
    path('female', views.female,name="female"),




]