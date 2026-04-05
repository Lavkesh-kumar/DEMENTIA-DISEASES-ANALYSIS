import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.inference import load_model, predict

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
MODEL_PATH = os.path.join('model', 'resnet18_best.pth')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

app.model = load_model(MODEL_PATH)

if app.model is None:
    print(f"WARNING: Model not found at {MODEL_PATH}. Prediction will fail unless you insert the .pth file.")

severity_mapping = {
    0: "Severe Dementia (Worst Case)",
    1: "Moderate to Severe Dementia",
    2: "Moderate Dementia",
    3: "Mild Dementia",
    4: "Very Mild Dementia",
    5: "Normal (No signs of Dementia)"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in request.")
            
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No selected file.")
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if app.model is None:
                return render_template('index.html', error="Model weights not found. Please add resnet18_best.pth to the model/ folder and restart.")
            
            try:
                prediction_class = predict(filepath, app.model)
                prefix = ""
                prediction_text = severity_mapping.get(int(prediction_class), f"Class {prediction_class}")
                image_url = f"/static/uploads/{filename}"
                return render_template('index.html', prediction=prediction_text, image_url=image_url)
            except Exception as e:
                return render_template('index.html', error=f"Error predicting: {str(e)}")
                
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
