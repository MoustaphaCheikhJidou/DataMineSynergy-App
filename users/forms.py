from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm 
from .models import Profile
from django.contrib.auth import get_user_model


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()
    # is_superuser = forms.BooleanField(required=False)

    class Meta:
        model = get_user_model()
        fields = ['username', 'email', 'password1', 'password2']

    # def save(self, commit=True):
    #     user = super().save(commit=False)
    #     user.is_superuser = self.cleaned_data['is_superuser']
    #     if commit:
    #         user.save()
    #     if user.is_superuser:
    #         user.is_staff = True
    #         user.save()
    #     return user


class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email']


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['image']

