from django.db import models

class FaceShape(models.Model):
    image = models.ImageField(upload_to='images/')
    predicted_shape = models.CharField(max_length=20)
    oneri1 = models.ImageField(upload_to='images/', default="")
    oneri2 = models.ImageField(upload_to='images/', default="")
    oneri3 = models.ImageField(upload_to='images/', default="")
    oneri4 = models.ImageField(upload_to='images/', default="")
    oneri5 = models.ImageField(upload_to='images/', default="")