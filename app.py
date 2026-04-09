from flask import Flask, render_template, request, send_file, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler  # Sampling အတွက် လိုအပ်သည်

app = Flask(__name__)
app.secret_key = "tu_meiktila_weather_project_secret"

# ၁။ Upload နှင့် Model Folder များ တည်ဆောက်ခြင်း
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

MODEL_PATH = os.path.join(MODEL_FOLDER, 'final_model.pkl')
ENCODER_PATH = os.path.join(MODEL_FOLDER, 'encoders.pkl')

# --- Routes ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/preprocessing')
def preprocessing():
    if 'data_file' in session and os.path.exists(session['data_file']):
        df = pd.read_csv(session['data_file'])
        return render_template('preprocessing.html', 
                               tables=[df.head(10).to_html(classes='data')], 
                               msg="File loaded. Ready for preprocessing.")
    return render_template('preprocessing.html', tables=None, msg="Please Upload CSV File")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('preprocessing'))
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        session['data_file'] = filepath
        return redirect(url_for('preprocessing'))
    return "Invalid file format", 400

@app.route('/process', methods=['POST'])
def process_data():
    if 'data_file' not in session:
        return redirect(url_for('preprocessing'))
    
    filepath = session['data_file']
    df = pd.read_csv(filepath)
    action = request.form.get('action')
    msg = ""

    if action == 'missing':
        # Step 1: Label Encoding for Categorical Columns
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                mask = df[col].notnull()
                df.loc[mask, col] = le.fit_transform(df[col].loc[mask].astype(str))
        
        # Step 2: Fill Missing Values with Mean
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        msg = "Missing Values handled with Mean Imputation!"

    elif action == 'scaling':
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]
        
        num_cols = features.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            scaler = StandardScaler()
            features[num_cols] = scaler.fit_transform(features[num_cols])
            df = pd.concat([features, target], axis=1)
            joblib.dump(scaler, os.path.join(MODEL_FOLDER, 'scaler.pkl'))
            msg = "Standard Scaling applied to Features!"
        else:
            msg = "No numeric columns found for scaling."

    elif action == 'clear':
        if os.path.exists(filepath):
            os.remove(filepath)
        session.pop('data_file', None)
        return redirect(url_for('preprocessing'))

    df.to_csv(filepath, index=False)
    return render_template('preprocessing.html', tables=[df.head(10).to_html(classes='data')], msg=msg)

@app.route('/save')
def save_file():
    if 'data_file' in session and os.path.exists(session['data_file']):
        return send_file(session['data_file'], as_attachment=True)
    return "No file to download", 404

@app.route('/training', methods=['GET'])
def training():
    return render_template('training.html')

@app.route('/train_only', methods=['POST'])
def train_only():
    model_type = request.form.get('model_type')
    train_file = request.files.get('train_file') 

    if not train_file:
        return render_template('training.html', result="❌ Error: Please select a CSV file.")

    try:
        df = pd.read_csv(train_file)
        
        # Categorical columns များကို Encode လုပ်ပြီး Encoder များကို သိမ်းမည်
        encoders = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, ENCODER_PATH)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].round().astype(int)

        # ၂။ Random Under Sampling အသုံးပြုခြင်း
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)

        # ၃။ Model ရွေးချယ်ခြင်း (Max Depth ပါဝင်သည်)
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)

        model.fit(X_resampled, y_resampled)
        joblib.dump(model, MODEL_PATH)

        return render_template('training.html', result=f"✅ {model_type.upper()} Trained with Random Under Sampling & Max Depth!")

    except Exception as e:
        return render_template('training.html', result=f"❌ Train Error: {str(e)}")

@app.route('/test_only', methods=['POST'])
def test_only():
    test_file = request.files.get('test_file')
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return render_template('training.html', result="⚠️ Please train the model first!")

    if not test_file:
        return render_template('training.html', result="❌ Error: Please select a Test CSV file.")

    try:
        df_test = pd.read_csv(test_file)
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)

        # Test data ကို Train တုန်းက encoder များဖြင့် Encode လုပ်ခြင်း
        for col, le in encoders.items():
            if col in df_test.columns:
                # Train တုန်းက မပါခဲ့သော Value အသစ်များအတွက် Error မတက်အောင် -1 ထည့်မည်
                df_test[col] = df_test[col].map(lambda s: le.transform([str(s)])[0] if str(s) in le.classes_ else -1)

        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1].round().astype(int)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        return render_template('training.html', result=f"🎯 Test Accuracy Score: {acc*100:.2f}%")

    except Exception as e:
        return render_template('training.html', result=f"❌ Test Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)