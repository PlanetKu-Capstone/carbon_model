from flask import Flask, request, jsonify
import pickle
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Memuat model dari file pickle
with open('carbon_emission.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Endpoint untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil data dari request
    data = request.json
    features = np.array([[data['id'], data['electriccity'], data['gas'], data['transportation'], 
                          data['food'], data['organic_waste'], data['inorganic_waste'],
                          data['user_id']]])  # Tambahkan fitur yang diperlukan
    
    # Melakukan prediksi
    prediction = model.predict(features)
    
    # Mengembalikan hasil prediksi
    return jsonify({'predicted_carbon_footprint': prediction[0]})

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)

