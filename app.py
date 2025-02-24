import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16 MB
app.secret_key = 'your_secret_key'

# Pastikan folder upload ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load model CNN dan label encoder
try:
    model_path = os.path.join('C:\\xampp\\htdocs\\ta_project', 'skin_cancer_model.h5')
    model = tf.keras.models.load_model(model_path)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    label_encoder_path = os.path.join('C:\\xampp\\htdocs\\ta_project', 'label_encoder.pkl')
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Label encoder berhasil dimuat.")
except Exception as e:
    print(f"Error loading label encoder: {e}")

# Fungsi untuk prediksi dan pengambilan gejala serta artikel
def predict_cancer_type(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction, axis=1)
    cancer_type = label_encoder.inverse_transform(predicted_class)[0]

    return cancer_type

def get_article_content_and_links(cancer_type):
    if cancer_type == "melanoma":
        url = "https://www.alodokter.com/kanker-kulit-melanoma"
        symptoms = "Gejala melanoma termasuk perubahan pada tahi lalat yang ada, gatal, dan pendarahan."
        treatment = "Penanganan melanoma bisa mencakup operasi, terapi radiasi, dan kemoterapi."
    else:
        url = "https://www.halodoc.com/kesehatan/karsinoma-sel-basal"
        symptoms = "Gejala karsinoma sel basal termasuk benjolan merah atau putih pada kulit yang bisa berdarah."
        treatment = "Penanganan karsinoma sel basal meliputi operasi pengangkatan tumor, terapi radiasi, dan krio-surgery."

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        article_title = soup.title.get_text()
        paragraphs = soup.find_all('p')
        article_content = " ".join([p.get_text() for p in paragraphs[:5]])  # Ambil 5 paragraf pertama sebagai konten
    except Exception as e:
        print(f"Error fetching article: {e}")
        article_content, article_title = "Tidak dapat mengambil artikel", "Error"
    
    return article_content, symptoms, treatment, [{'title': article_title, 'url': url, 'content': article_content}]

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk upload gambar dan klasifikasi
@app.route('/', methods=['POST'])
def upload_and_classify():
    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Simpan file
        file.save(file_path)

        # Prediksi kanker dan pengambilan artikel
        cancer_type = predict_cancer_type(file_path)
        article_content, symptoms, treatment, articles = get_article_content_and_links(cancer_type)

        # Generate URL untuk gambar yang diupload
        image_url = url_for('static', filename=f'uploads/{filename}')

        return render_template('result.html', cancer_type=cancer_type, article_content=article_content, 
                               symptoms=symptoms, treatment=treatment, articles=articles, image_path=image_url)

if __name__ == '__main__':
    app.run(debug=True)
    