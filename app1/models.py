from django.db import models
from django.db.models.fields.files import FileField

# Create your models here.
class user(models.Model):
    fl=models.FileField(upload_to="recorded_audio")

