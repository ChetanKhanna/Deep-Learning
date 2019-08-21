from django.db import models

# Create your models here.
class FruitImage(models.Model):
	fruit_image = models.ImageField(upload_to='clf/Images/')
	uploaded_at = models.DateTimeField(auto_now_add=True)
