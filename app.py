import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for, session
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = "tu_meiktila_weather_project_secret"

# ---------------------------------------------------------
# ၁။ Folder များ တည်ဆောက်ခြင်း
# ---------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_FOLDER, 'final_model.pkl')
ENCODER_PATH = os.path.join(MODEL_FOLDER, 'encoders.pkl')
TARGET_ENCODER_PATH = os.path.join(MODEL_FOLDER, 'target_encoder.pkl')
SCALER_PATH = os.path.join(MODEL_FOLDER, 'scaler.pkl')
IMPUTER_PATH = os.path.join(MODEL_FOLDER, 'imputer.pkl')

# Fake Noise Features များ ထည့်သွင်းပေးသည့် Helper Function
def add_fake_features(df_features, num_noise=20):
    np.random.seed(42)
    df_copy = df_features.copy()
    for i in range(1, num_noise + 1):
        df_copy[f'fake_noise_{i}'] = np.random.uniform(-100, 100, len(df_copy))
    return df_copy

# ---------------------------------------------------------
# Routes (လမ်းကြောင်းများ)
# ---------------------------------------------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/preprocessing')
def preprocessing():
    if 'data_file' in session and os.path.exists(session['data_file']):
        df = pd.read_csv(session['data_file'])
        return render_template('preprocessing.html', 
                               tables=[df.head(10).to_html(classes='data')], 
                               msg="ဖိုင်ကို လှမ်းယူပြီးပါပြီ။ Preprocessing ပြုလုပ်ရန် အသင့်ဖြစ်ပါသည်။")
    return render_template('preprocessing.html', tables=None, msg="ကျေးဇူးပြု၍ CSV ဖိုင် တင်ပါ (Upload)")

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
    return "မမှန်ကန်သော ဖိုင်အမျိုးအစားဖြစ်သည်", 400

# ---------------------------------------------------------
# Preprocessing အဆင့်တွင် Target Encoder ပါ တစ်ခါတည်း သိမ်းဆည်းခြင်း
# ---------------------------------------------------------
@app.route('/process', methods=['POST'])
def process_data():
    if 'data_file' not in session:
        return redirect(url_for('preprocessing'))
    
    filepath = session['data_file']
    df = pd.read_csv(filepath)
    action = request.form.get('action')
    msg = ""

    if action == 'preprocessing':
        encoders = {}
        target_col = df.columns[-1]  # နောက်ဆုံး Column ကို Target ဟု သတ်မှတ်သည်

        # ၁။ Target Column ကို LabelEncode လုပ်ပြီး သီးသန့် Save လုပ်မည်
        if df[target_col].dtype == 'object' or df[target_col].dtype == 'category':
            target_le = LabelEncoder()
            df[target_col] = target_le.fit_transform(df[target_col].astype(str))
            joblib.dump(target_le, TARGET_ENCODER_PATH)

        # ၂။ ကျန်တဲ့ Feature Columns များကို Encode လုပ်မည်
        for col in df.columns[:-1]:
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        
        joblib.dump(encoders, ENCODER_PATH)
        msg = "Data Preprocessing (Label Encoding) အောင်မြင်စွာ ပြုလုပ်ပြီး Target Encoder ပါ သိမ်းဆည်းပြီးပါပြီ!"

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
    return "ဒေါင်းလုဒ်ဆွဲရန် ဖိုင်မရှိပါ", 404

@app.route('/training', methods=['GET'])
def training():
    return render_template('training.html')

# ---------------------------------------------------------
# Model Training (Preprocessed Data ဖြင့် Train ခြင်း)
# ---------------------------------------------------------
@app.route('/train_only', methods=['POST'])
def train_only():
    model_type = request.form.get('model_type')
    train_file = request.files.get('train_file') 

    if not train_file:
        return render_template('training.html', result="❌ Error: ကျေးဇူးပြု၍ CSV ဖိုင်ကို ရွေးချယ်ပါ။")

    try:
        df = pd.read_csv(train_file)
        
        # ၁။ Features နှင့် Target ခွဲထုတ်ခြင်း (Data များသည် Encode လုပ်ပြီးသား ဂဏန်းများဖြစ်သည်)
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy().astype(int)

        # 🎯 Target Name များကို ရှာဖွေခြင်း
        # Preprocessing တုန်းက သိမ်းခဲ့သော Target Encoder ရှိလျှင် Original Class Names ကိုယူမည်
        target_names = None
        if os.path.exists(TARGET_ENCODER_PATH):
            target_le = joblib.load(TARGET_ENCODER_PATH)
            target_names = target_le.classes_.astype(str).tolist()
        else:
            # တကယ်လို့ target_encoder.pkl မရှိရင် ဂဏန်းအတိုင်းပဲ ပြမည်
            target_names = [str(c) for c in np.unique(y)]

        # ၂။ Fake Noise Features (၂၀) ခု ထည့်သွင်းခြင်း
        X = add_fake_features(X, num_noise=20)

        # ၃။ Data Split (70% Train, 30% Validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        num_cols = X_train.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            imputer = SimpleImputer(strategy='median')
            X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
            X_val[num_cols] = imputer.transform(X_val[num_cols])
            joblib.dump(imputer, IMPUTER_PATH)

            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_val[num_cols] = scaler.transform(X_val[num_cols])
            joblib.dump(scaler, SCALER_PATH)

        # ၄။ Model Train ပြုလုပ်ခြင်း
        if model_type == 'rf':
            model = RandomForestClassifier(
                criterion='gini',
                n_estimators=500,
                max_depth=2,
                max_features=2,
                random_state=42,
                max_leaf_nodes=5,
                class_weight='balanced'
            )
        else:
            model = LogisticRegression(max_iter=1000, C=0.05, random_state=42)

        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

        # ၅။ Validation Evaluation & Classification Report
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        # Target Class Names ဖြင့် Report ထုတ်ပေးခြင်း
        labels = list(range(len(target_names)))
        val_report = classification_report(y_val, val_preds, labels=labels, target_names=target_names)

        return render_template(
            'training.html', 
            result=f"✅ {model_type.upper()} Model လေ့ကျင့်ပြီးပါပြီ! (Validation Accuracy: {val_acc * 100:.2f}%)",
            report=val_report
        )

    except Exception as e:
        return render_template('training.html', result=f"❌ Train Error: {str(e)}")

# ---------------------------------------------------------
# Test File ဖြင့် စမ်းသပ်ခြင်း
# ---------------------------------------------------------
# ---------------------------------------------------------
# Test File ဖြင့် စမ်းသပ်ခြင်း (Actual vs Predicted Table ဖြင့်)
# ---------------------------------------------------------
@app.route('/test_only', methods=['POST'])
def test_only():
    test_file = request.files.get('test_file')
    
    if not os.path.exists(MODEL_PATH):
        return render_template('training.html', result="⚠️ ကျေးဇူးပြု၍ ပထမဦးစွာ Model ကို Train ပြုလုပ်ပါ။")

    if not test_file:
        return render_template('training.html', result="❌ Error: ကျေးဇူးပြု၍ Test CSV ဖိုင်ကို ရွေးချယ်ပါ။")

    try:
        df_test = pd.read_csv(test_file)
        model = joblib.load(MODEL_PATH)

        X_test = df_test.iloc[:, :-1].copy()
        y_test_orig = df_test.iloc[:, -1].copy() # Original Actual y values

        # 🎯 Target Class Names များကို ရယူခြင်း
        target_names = None
        if os.path.exists(TARGET_ENCODER_PATH):
            target_le = joblib.load(TARGET_ENCODER_PATH)
            target_names = target_le.classes_.astype(str).tolist()
            
            # y_test သည် String ဖြစ်နေပါက Numeric ပြောင်းမည်
            if y_test_orig.dtype == 'object' or y_test_orig.dtype == 'category':
                known_classes = set(target_le.classes_)
                y_test = y_test_orig.astype(str).map(
                    lambda val: target_le.transform([val])[0] if val in known_classes else -1
                )
            else:
                y_test = y_test_orig.astype(int)
        else:
            target_names = [str(c) for c in np.unique(y_test_orig)]
            y_test = y_test_orig.astype(int)

        y_test = np.array(y_test, dtype=int)

        # Feature Encoders ရှိလျှင် ပြန်သုံးမည်
        if os.path.exists(ENCODER_PATH):
            encoders = joblib.load(ENCODER_PATH)
            for col in X_test.columns:
                if col in encoders and (X_test[col].dtype == 'object' or X_test[col].dtype == 'category'):
                    le = encoders[col]
                    X_test[col] = X_test[col].astype(str).map(
                        lambda val: le.transform([val])[0] if val in le.classes_ else -1
                    )

        # Fake Noise + Imputer + Scaler
        X_test = add_fake_features(X_test, num_noise=20)

        num_cols = X_test.select_dtypes(include=[np.number]).columns
        if os.path.exists(IMPUTER_PATH) and not num_cols.empty:
            imputer = joblib.load(IMPUTER_PATH)
            X_test[num_cols] = imputer.transform(X_test[num_cols])

        if os.path.exists(SCALER_PATH) and not num_cols.empty:
            scaler = joblib.load(SCALER_PATH)
            X_test[num_cols] = scaler.transform(X_test[num_cols])

        # Prediction ပြုလုပ်ခြင်း
        predictions = model.predict(X_test)
        predictions = np.array(predictions, dtype=int)

        acc = accuracy_score(y_test, predictions)

        # 🎯 ၁။ Classification Report
        labels = list(range(len(target_names)))
        test_report = classification_report(y_test, predictions, labels=labels, target_names=target_names)

        # 🎯 ၂။ Numeric Prediction များကို Original Target Class Names စာသားများသို့ ပြန်ပြောင်းခြင်း
        if os.path.exists(TARGET_ENCODER_PATH):
            target_le = joblib.load(TARGET_ENCODER_PATH)
            pred_labels = [target_le.inverse_transform([p])[0] if p < len(target_le.classes_) else str(p) for p in predictions]
        else:
            pred_labels = predictions.tolist()

        # 🎯 ၃။ Actual vs Predicted Comparison DataFrame တည်ဆောက်ခြင်း
        df_comparison = pd.DataFrame({
            'Row No.': range(1, len(y_test) + 1),
            'Actual Weather': y_test_orig.values,
            'Predicted Weather': pred_labels,
                    })

        # HTML Table အဖြစ် ပြောင်းလဲခြင်း
        comparison_table = df_comparison.to_html(
            classes='table table-striped table-hover table-bordered text-center', 
            index=False
        )

        return render_template(
            'training.html', 
            result=f"🎯 Test Accuracy Score: {acc * 100:.2f}%",
            report=test_report,
            comparison_table=comparison_table
        )

    except Exception as e:
        return render_template('training.html', result=f"❌ Test Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)