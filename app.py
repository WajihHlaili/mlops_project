from flask import Flask, render_template, request, jsonify
import requests
import json


# Create Flask app
app = Flask(__name__, template_folder='client/templates', static_folder='client/static')


# External URL of the FastAPI service (to be used in GCP)
API_URL = "https://fastapi-app-180154123347.us-central1.run.app"
  # Replace with actual URL or IP

fuels = ['Diesel', 'Petrol', 'CNG', 'LPG']
owners = ['Un', 'Deux', 'Trois et plus']

@app.route('/')
def index():
    return render_template('index.html', Pred='0.00')

@app.route('/prediction', methods=['POST'])
def predict():
    json_data = request.get_json()

    # Debug log to check the received data
    print(f"Received data: {json_data}")

    # Get data from the Flask request
    transmission = json_data.get('transmission', '')
    fuel = json_data.get('fuel', '')
    owner = json_data.get('owner', '')
    year = int(json_data.get('year', 0))
    km_driven = float(json_data.get('km_driven', 0))
    engine = float(json_data.get('engine', 0))
    max_power = float(json_data.get('max_power', 0))

    # Convert the transmission, fuel, and owner values
    transmission = 2 if transmission == 'Manuel' else 1
    owner = owners.index(owner)
    fuel = fuels.index(fuel) + 1

    # Data to send to FastAPI prediction endpoint
    data = {
        "transmission": transmission,
        "fuel": fuel,
        "owner": owner,
        "year": year,
        "km_driven": km_driven,
        "engine": engine,
        "max_power": max_power
    }

    print(f"Sending data to FastAPI: {data}")

    # Send the data to FastAPI for prediction
    response = requests.post(f"{API_URL}/predict", json=data)

    # Debug log to check FastAPI response
    print(f"FastAPI response status: {response.status_code}")
    print(f"FastAPI response data: {response.text}")

    if response.status_code == 200:
        result = response.json()
        print(f"Prediction result from FastAPI: {result}")
        return jsonify({"Predict": result['Predict']})
    else:
        return jsonify({"error": "Unable to get a prediction"})


if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0', port=8080)