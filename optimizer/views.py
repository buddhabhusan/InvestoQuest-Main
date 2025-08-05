# optimizer/views.py

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import pandas as pd
from .forms import PortfolioOptimizerForm # <-- UPDATED

# This decorator ensures that only logged-in users can access this page
@login_required
def welcome_view(request):
    """
    Renders the main welcome page with navigation buttons.
    """
    return render(request, 'optimizer/welcome.html')

@login_required
def know_the_models_view(request):
    """
    Renders the page explaining the different portfolio models.
    """
    return render(request, 'optimizer/know_the_models.html')

@login_required
def portfolio_optimizer_view(request):
    """
    Handles file upload, model selection, and displays optimization results.
    """
    if request.method == 'POST':
        form = PortfolioOptimizerForm(request.POST, request.FILES)
    if form.is_valid():
        selected_model = form.cleaned_data['model_choice']
        returns_file = form.cleaned_data['returns_file']

        try:
            df = pd.read_csv(returns_file, index_col="Date", parse_dates=True)

            if selected_model == 'mean_variance':
                from .mean_variance_optimizer import analyze_portfolio
                results_html = analyze_portfolio(df)

            elif selected_model == 'risk_parity':
                results_html = df.head().to_html(classes='table-auto w-full text-left whitespace-no-wrap') + "<p class='mt-2 font-bold'>Processed with Risk Parity.</p>"

            else:
                results_html = "<p>Selected model not recognized.</p>"

            return render(request, 'optimizer/portfolio_optimizer.html', {
                'form': form,
                'optimization_results': results_html
            })

        except Exception as e:
            error_message = f"Error processing file: {e}"
            return render(request, 'optimizer/portfolio_optimizer.html', {
                'form': form,
                'error_message': error_message
            })
    else:
        form = PortfolioOptimizerForm()

    return render(request, 'optimizer/portfolio_optimizer.html', {'form': form})


@login_required
def about_us_view(request):
    """
    Renders the 'About Us' page.
    """
    return render(request, 'optimizer/about_us.html')

@login_required
def contact_us_view(request):
    """
    Renders the 'Contact Us' page.
    """
    return render(request, 'optimizer/contact_us.html')