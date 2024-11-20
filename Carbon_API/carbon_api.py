from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('carbon_emission.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['id'], data['electriccity'], data['gas'], data['transportation'], 
                          data['food'], data['organic_waste'], data['inorganic_waste'],
                          data['user_id']]]) 
    
    prediction = model.predict(features)
    return jsonify({'predicted_carbon_footprint': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
