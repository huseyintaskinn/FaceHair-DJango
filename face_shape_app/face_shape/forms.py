from django import forms
from .models import FaceShape

class FaceShapeForm(forms.ModelForm):
    class Meta:
        model = FaceShape
        fields = ['image']
        labels = {
            'image': 'Yüzünüzün önden çekilmiş bir fotoğrafını yükleyiniz.'
        }
        widgets = {
            'image': forms.ClearableFileInput(attrs={'class': 'file'})
        }
