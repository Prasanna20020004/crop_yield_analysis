import os
import joblib
import pandas as pd
import logging
from django.shortcuts import render
from groq import Groq  # ✅ Groq AI import

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_yield_predictor.pkl')

logger = logging.getLogger(__name__)
_model = None
_model_load_error = None

# Initialize Groq client (use env var; do NOT hardcode API keys)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set in environment; Groq client disabled.")
    client = None
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        logger.exception("Failed to initialize Groq client; continuing without it.")
        client = None

def _local_recommendations(region, crop, soil_type, rainfall, temperature, fertilizer_used, irrigation_used, weather, days_to_harvest, prediction):
    """Return 3 short fallback recommendations if Groq API is unavailable."""
    recs = []
    # Simple heuristic recommendations
    if not fertilizer_used:
        recs.append("Do a soil test and apply balanced fertilizer as per results.")
    else:
        recs.append("Ensure fertilizer timing and rates match crop needs.")

    if irrigation_used:
        recs.append("Optimize irrigation schedule to avoid water stress.")
    else:
        recs.append("Consider small-scale irrigation (drip/mulch) to conserve moisture.")

    if rainfall < 250:
        recs.append("Use mulching or cover crops to retain soil moisture during dry spells.")
    else:
        recs.append("Monitor drainage and avoid waterlogging; improve soil structure if needed.")

    return "\n".join(recs[:3])

def _get_model():
    global _model, _model_load_error
    if _model is None and _model_load_error is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            _model_load_error = e
            logger.exception("Failed to load ML model from %s", MODEL_PATH)
    return _model, _model_load_error


def predict_yield(request):
    prediction = None
    recommendations = None
    error_message = None

    model, load_error = _get_model()
    if load_error:
        error_message = f"Model load error: {load_error}"
        return render(request, 'predict.html', {
            'prediction': prediction,
            'recommendations': recommendations,
            'error_message': error_message
        })

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

            # ✅ Groq AI Recommendation Generation (focused on increasing yield)
            prompt = f"""
            You are an expert agricultural scientist and crop advisor with field experience in small-to-medium farms. 
            Your task is to generate a **thorough, step-by-step plan** to increase the yield of the crop described below.

            Guidelines:
            1. Provide exactly **5 numbered recommendations**, ordered by impact (most effective first).
            2. Each recommendation must be **3–5 sentences** long and include:
            - specific actions (e.g., apply 120 kg/ha NPK 15:15:15, irrigate every 6 days for 3 hours)
            - measurable or observable indicators (e.g., leaf greenness, moisture level, pest threshold)
            - recommended timing (days/weeks before harvest)
            - a low-cost or organic alternative if modern input isn’t available
            3. Each point should address a different domain: soil nutrition, irrigation, pest/disease management, weather adaptation, and harvest/storage.
            4. End with a clearly labeled section:
            **Immediate 48-hour Action Checklist:**
            - Include 3 fast checks or small actions the farmer should take right now to stabilize crop health or prevent loss.
            - Add one safety reminder for handling fertilizer, tools, or pesticides.
            5. Keep tone practical, farmer-friendly, and regionally adaptable. Use short, direct sentences and avoid jargon unless explained.

            Field Data:
            - Region: {region}
            - Crop: {crop}
            - Soil Type: {soil_type}
            - Rainfall: {rainfall} mm
            - Temperature: {temperature} °C
            - Fertilizer Used: {fertilizer_used}
            - Irrigation Used: {irrigation_used}
            - Weather: {weather}
            - Days to Harvest: {days_to_harvest}
            - Predicted Yield: {prediction} tons/ha

            Output format:
            1) Detailed numbered list (1–5)
            2) Section: "Immediate 48-hour Action Checklist"
            Ensure formatting and numbering are clean and clear for direct display on a web app.
            """


            if client is None:
                logger.info("Groq client unavailable — using local fallback recommendations.")
                recommendations = _local_recommendations(region, crop, soil_type, rainfall, temperature, fertilizer_used, irrigation_used, weather, days_to_harvest, prediction)
            else:
                try:
                    groq_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    recommendations = groq_response.choices[0].message.content.strip()
                except Exception:
                    logger.exception("Groq API call failed — returning local recommendations")
                    recommendations = _local_recommendations(region, crop, soil_type, rainfall, temperature, fertilizer_used, irrigation_used, weather, days_to_harvest, prediction)

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render(request, 'predict.html', {
        'prediction': prediction,
        'recommendations': recommendations,
        'error_message': error_message
    })
