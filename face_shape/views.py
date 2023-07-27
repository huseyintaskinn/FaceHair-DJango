from django.shortcuts import render
from .forms import FaceShapeForm
from .models import FaceShape
from .utils import analyze_face_shape
from .utils import yuzBirlestir
import os
from django.conf import settings
from datetime import datetime

def home(request):
    url = ""

    if request.method == 'POST':
        form = FaceShapeForm(request.POST, request.FILES)
        if form.is_valid():
            face_shape = form.save(commit=False)
            face_shape.predicted_shape = analyze_face_shape(face_shape.image)
            
            if face_shape.predicted_shape == "Kare":
                url = os.path.join(settings.MEDIA_ROOT, 'oneri\\Kare\\')
            elif face_shape.predicted_shape == "Kalp":
                url = os.path.join(settings.MEDIA_ROOT, 'oneri\\Kalp\\')
            elif face_shape.predicted_shape == "Yuvarlak":
                url = os.path.join(settings.MEDIA_ROOT, 'oneri\\Yuvarlak\\')
            elif face_shape.predicted_shape == "Uzun":
                url = os.path.join(settings.MEDIA_ROOT, 'oneri\\Uzun\\')
            elif face_shape.predicted_shape == "Oval":
                url = os.path.join(settings.MEDIA_ROOT, 'oneri\\Oval\\')

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d%H%M%S")
            # Dosya adını oluşturun
            filename = f"orj_{timestamp}.jpg"
            face_shape.image.name = filename

            # Dosyayı kaydedin
            face_shape.save()

            url2 = os.path.join(settings.MEDIA_ROOT, 'images\\')

            face_shape.oneri1 = yuzBirlestir(url2 + filename, url + "1.jpg")
            face_shape.oneri2 = yuzBirlestir(url2 + filename, url + "2.jpg")
            face_shape.oneri3 = yuzBirlestir(url2 + filename, url + "3.jpg")
            face_shape.oneri4 = yuzBirlestir(url2 + filename, url + "4.jpg")
            face_shape.oneri5 = yuzBirlestir(url2 + filename, url + "5.jpg")

            face_shape.save()
    else:
        form = FaceShapeForm()

    return render(request, 'home.html', {'form': form})
