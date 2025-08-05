# optimizer/forms.py
from django import forms

class PortfolioOptimizerForm(forms.Form):
    # Define the choices for the dropdown
    MODEL_CHOICES = [
        ('black_litterman', 'Black-Litterman Model'),
        ('risk_parity', 'Risk Parity'),
        ('mean_variance', 'Mean Variance'),
    ]

    # Assume your file field is named 'returns_file'
    returns_file = forms.FileField(
        label="Select a returns CSV file",
        help_text="File must be in CSV format.",
        widget=forms.ClearableFileInput(attrs={'class': 'block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none'})
    )
    
    # Add the new model choice field
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="Select Optimization Model",
        widget=forms.Select(attrs={'class': 'block w-full mt-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50'})
    )