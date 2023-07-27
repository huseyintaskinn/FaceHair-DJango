from django import forms
from .models import FaceShape

class FaceShapeForm(forms.ModelForm):
    image = forms.ImageField(label='Yüzünüzün önden çekilmiş bir fotoğrafını yükleyiniz.', widget=forms.ClearableFileInput(attrs={'class': 'file'}), required=True)

    class Meta:
        model = FaceShape
        fields = ['image']
