import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for, session
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
SCALER_PATH = os.path.join(MODEL_FOLDER, 'scaler.pkl')
IMPUTER_PATH = os.path.join(MODEL_FOLDER, 'imputer.pkl')

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

@app.route('/process', methods=['POST'])
def process_data():
    if 'data_file' not in session:
        return redirect(url_for('preprocessing'))
    
    filepath = session['data_file']
    df = pd.read_csv(filepath)
    action = request.form.get('action')
    msg = ""

    # Missing Values ဖြည့်သွင်းခြင်း
    if action == 'missing':
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if not num_cols.empty:
            num_imputer = SimpleImputer(strategy='median')
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
            
        if not cat_cols.empty:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        msg = "Missing Values များကို အောင်မြင်စွာ ဖြည့်စွက်ပြီးပါပြီ!"

    # Standard Scaling ပြုလုပ်ခြင်း
    elif action == 'scaling':
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]
        
        num_cols = features.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            scaler = StandardScaler()
            features[num_cols] = scaler.fit_transform(features[num_cols])
            df = pd.concat([features, target], axis=1)
            msg = "Features များကို Standard Scaling ပြုလုပ်ပြီးပါပြီ!"
        else:
            msg = "Scaling ပြုလုပ်ရန် Numeric Column မတွေ့ရှိပါ။"

    # Session နှင့် ဖိုင်ရှင်းလင်းခြင်း
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
# Model Training (70/30 Split + Keep All Columns)
# ---------------------------------------------------------
@app.route('/train_only', methods=['POST'])
def train_only():
    model_type = request.form.get('model_type')
    train_file = request.files.get('train_file') 

    if not train_file:
        return render_template('training.html', result="❌ Error: ကျေးဇူးပြု၍ CSV ဖိုင်ကို ရွေးချယ်ပါ။")

    try:
        df = pd.read_csv(train_file)
        
        # Column များကို Drop မလုပ်ဘဲ Feature အားလုံးကို သိမ်းဆည်းခြင်း
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Label Encoding
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, ENCODER_PATH)

        if y.dtype == 'object':
            target_le = LabelEncoder()
            y = target_le.fit_transform(y.astype(str))
        else:
            y = y.round().astype(int)

        # Data ကို 70% Train နှင့် 30% Validation အဖြစ် ခွဲခြားခြင်း (test_size=0.3)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Imputation နှင့် Scaling များကို 70% Train Data ပေါ်တွင်သာ Fit ပြုလုပ်ခြင်း
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

        # Model ကို Train ပြုလုပ်ခြင်း (Overfitting ကာကွယ်ရန် တင်းကြပ်သော Hyperparameters သုံးထားသည်)
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=30,           # အပင် အရေအတွက် နည်းဆင်းထားသည်
                max_depth=3,               # Depth ကို 3 အထိ လျှော့ထားသဖြင့် Overfit မဖြစ်စေပါ
                min_samples_split=20,      # Node တစ်ခု ခွဲရန် Sample ၂၀ လိုအပ်သည်
                min_samples_leaf=15,       # Leaf Node တိုင်းတွင် အနည်းဆုံး Sample ၁၅ ခု ရှိစေသည်
                max_features='sqrt',       # Feature များကို တစ်စိတ်တစ်ပိုင်းသာ ရွေးချယ်ခွင့်ပြုသည်
                max_samples=0.5,           # Data ၏ 50% သာ Random မဲနှိုက်၍ အပင်ဆောက်သည်
                class_weight='balanced',
                random_state=42
            )
        else:
            model = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced') # C=0.1 ဖြင့် Regularization တင်းကြပ်ထားသည်

        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

        # 30% Validation Data ဖြင့် Accuracy စစ်ဆေးခြင်း
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        return render_template(
            'training.html', 
            result=f"✅ {model_type.upper()} Model လေ့ကျင့်ပြီးပါပြီ! (Validation Accuracy: {val_acc * 100:.2f}%)"
        )

    except Exception as e:
        return render_template('training.html', result=f"❌ Train Error: {str(e)}")

# ---------------------------------------------------------
# Test File ဖြင့် စမ်းသပ်ခြင်း (Test Only)
# ---------------------------------------------------------
@app.route('/test_only', methods=['POST'])
def test_only():
    test_file = request.files.get('test_file')
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return render_template('training.html', result="⚠️ ကျေးဇူးပြု၍ ပထမဦးစွာ Model ကို Train ပြုလုပ်ပါ။")

    if not test_file:
        return render_template('training.html', result="❌ Error: ကျေးဇူးပြု၍ Test CSV ဖိုင်ကို ရွေးချယ်ပါ။")

    try:
        df_test = pd.read_csv(test_file)
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)

        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

        for col, le in encoders.items():
            if col in X_test.columns:
                known_classes = set(le.classes_)
                X_test[col] = X_test[col].astype(str).map(
                    lambda val: le.transform([val])[0] if val in known_classes else -1
                )

        if y_test.dtype == 'object':
            y_test = LabelEncoder().fit_transform(y_test.astype(str))
        else:
            y_test = y_test.round().astype(int)

        if os.path.exists(IMPUTER_PATH):
            imputer = joblib.load(IMPUTER_PATH)
            num_cols = X_test.select_dtypes(include=[np.number]).columns
            if not num_cols.empty:
                X_test[num_cols] = imputer.transform(X_test[num_cols])

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            num_cols = X_test.select_dtypes(include=[np.number]).columns
            if not num_cols.empty:
                X_test[num_cols] = scaler.transform(X_test[num_cols])

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        return render_template('training.html', result=f"🎯 Test Accuracy Score: {acc * 100:.2f}%")

    except Exception as e:
        return render_template('training.html', result=f"❌ Test Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)