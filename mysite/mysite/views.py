import os
import joblib
import pandas as pd
from django.shortcuts import render

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_yield_predictor.pkl')
model = joblib.load(MODEL_PATH)

def predict_yield(request):
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            region = request.POST.get('region')
            crop = request.POST.get('crop')
            soil_type = request.POST.get('soil_type')
            rainfall = float(request.POST.get('rainfall'))
            temperature = float(request.POST.get('temperature'))
            fertilizer_used = request.POST.get('fertilizer_used') == 'Yes'
            irrigation_used = request.POST.get('irrigation_used') == 'Yes'
            weather = request.POST.get('weather')
            days_to_harvest = float(request.POST.get('days_to_harvest'))

            input_data = pd.DataFrame([{
                'Region': region,
                'Crop': crop,
                'Soil_Type': soil_type,
                'Rainfall_mm': rainfall,
                'Temperature_Celsius': temperature,
                'Fertilizer_Used': fertilizer_used,
                'Irrigation_Used': irrigation_used,
                'Weather_Condition': weather,
                'Days_to_Harvest': days_to_harvest
            }])

            prediction = round(model.predict(input_data)[0], 2)

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render(request, 'predict.html', {
        'prediction': prediction,
        'error_message': error_message
    })
