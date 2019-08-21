from django import forms
from clf.models import FruitImage


class FruitImageForm(forms.ModelForm):

	class Meta:
		model = FruitImage
		fields = ['fruit_image']
