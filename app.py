import os
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Fungsi untuk memuat model berdasarkan algoritma
def load_model(algorithm):
    model_path = f"{algorithm}_fish_model.joblib"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as file:
                model = joblib.load(file)
            print(f"Model {algorithm} berhasil dimuat.")
            return model
        except Exception as e:
            print(f"Error saat memuat model {algorithm}: {e}")
            return None
    else:
        print(f"File model {algorithm} tidak ditemukan.")
        return None

# Fungsi untuk memuat scaler
def load_scaler(algorithm):
    scaler_path = f"{algorithm}_scaler.joblib"
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as file:
                scaler = joblib.load(file)
            print(f"Scaler untuk {algorithm} berhasil dimuat.")
            return scaler
        except Exception as e:
            print(f"Error saat memuat scaler untuk {algorithm}: {e}")
            return None
    else:
        print(f"File scaler untuk {algorithm} tidak ditemukan.")
        return None

# Fungsi untuk memuat label encoder
def load_label_encoder():
    encoder_path = 'label_encoder.joblib'
    if os.path.exists(encoder_path):
        try:
            with open(encoder_path, 'rb') as file:
                encoder = joblib.load(file)
            print("Label encoder berhasil dimuat.")
            return encoder
        except Exception as e:
            print(f"Error saat memuat label encoder: {e}")
            return None
    else:
        print("File label encoder tidak ditemukan.")
        return None

# Inisialisasi aplikasi Flask
app = Flask(__name__)
label_encoder = load_label_encoder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if label_encoder is None:
        return jsonify({"error": "Label encoder tidak tersedia"}), 500

    # Mendapatkan data input
    data = request.get_json()
    algorithm = data.get('algorithm', '').lower()
    try:
        length = float(data['length'])
        weight = float(data['weight'])
        w_l_ratio = float(data['w_l_ratio'])
        input_features = [[length, weight, w_l_ratio]]

        # Muat model dan scaler sesuai algoritma yang dipilih
        model = load_model(algorithm)
        if model is None:
            return jsonify({"error": f"Model untuk algoritma {algorithm} tidak tersedia"}), 500

        # Terapkan scaler jika algoritma menggunakan scaler
        if algorithm in ['svm', 'perceptron']:
            scaler = load_scaler(algorithm)
            if scaler is None:
                return jsonify({"error": f"Scaler untuk algoritma {algorithm} tidak tersedia"}), 500
            input_features = scaler.transform(input_features)

        # Prediksi menggunakan model yang dipilih
        prediction = model.predict(input_features)

        # Konversi label prediksi kembali ke nama spesies
        species = label_encoder.inverse_transform(prediction)[0]
        return jsonify({"species": species, "algorithm": algorithm})
    except ValueError as e:
        return jsonify({"error": f"Kesalahan tipe data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Kesalahan dalam prediksi: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
