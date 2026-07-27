import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for, session
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
=======
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
>>>>>>> d72dff6 (4months accuracy commit)

app = Flask(__name__)
app.secret_key = "tu_meiktila_weather_project_secret"

<<<<<<< HEAD
=======
# ---------------------------------------------------------
# ၁။ Folder များ တည်ဆောက်ခြင်း
# ---------------------------------------------------------
>>>>>>> d72dff6 (4months accuracy commit)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ENCODER_PATH = os.path.join(MODEL_FOLDER, 'encoders.pkl')
<<<<<<< HEAD
DROPPED_COLS_PATH = os.path.join(MODEL_FOLDER, 'dropped_cols.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_FOLDER, 'feature_names.pkl')

TARGET_COLUMN_NAME = 'Rain'

def clean_column_names(df):
    """Column နာမည်များတွင် ပါဝင်သော Space များနှင့် Capitalization များကို ရှင်းလင်းပေးသည်"""
    df.columns = df.columns.str.strip().str.lower()
    return df

def get_pipeline_path(model_type_key):
    return os.path.join(MODEL_FOLDER, f'weather_pipeline_{model_type_key}.pkl')
=======
SCALER_PATH = os.path.join(MODEL_FOLDER, 'scaler.pkl')
IMPUTER_PATH = os.path.join(MODEL_FOLDER, 'imputer.pkl')

# ---------------------------------------------------------
# Routes (လမ်းကြောင်းများ)
# ---------------------------------------------------------
>>>>>>> d72dff6 (4months accuracy commit)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/preprocessing')
def preprocessing():
    if 'data_file' in session and os.path.exists(session['data_file']):
        df = pd.read_csv(session['data_file'])
        return render_template('preprocessing.html', 
                               tables=[df.head(10).to_html(classes='data')], 
<<<<<<< HEAD
                               msg="File loaded. Duplicates & Missing Values check is ready.")
    return render_template('preprocessing.html', tables=None, msg="Please Upload CSV File")
=======
                               msg="ဖိုင်ကို လှမ်းယူပြီးပါပြီ။ Preprocessing ပြုလုပ်ရန် အသင့်ဖြစ်ပါသည်။")
    return render_template('preprocessing.html', tables=None, msg="ကျေးဇူးပြု၍ CSV ဖိုင် တင်ပါ (Upload)")
>>>>>>> d72dff6 (4months accuracy commit)

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
<<<<<<< HEAD
        df = df.drop_duplicates()
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                mask = df[col].notnull()
                df.loc[mask, col] = le.fit_transform(df[col].loc[mask].astype(str))
        
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        msg = "Duplicates removed and Missing values handled!"
=======
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
>>>>>>> d72dff6 (4months accuracy commit)

    # Session နှင့် ဖိုင်ရှင်းလင်းခြင်း
    elif action == 'clear':
        if os.path.exists(filepath):
            os.remove(filepath)
        session.pop('data_file', None)
        return redirect(url_for('preprocessing'))

    df.to_csv(filepath, index=False)
    return render_template('preprocessing.html', tables=[df.head(10).to_html(classes='data')], msg=msg)

<<<<<<< HEAD
@app.route('/save', methods=['GET'])
def save_csv():
    if 'data_file' not in session or not os.path.exists(session['data_file']):
        return redirect(url_for('preprocessing'))
    try:
        return send_file(session['data_file'], mimetype='text/csv', as_attachment=True,
                         download_name=os.path.basename(session['data_file']))
    except TypeError:
        return send_file(session['data_file'], mimetype='text/csv', as_attachment=True,
                         attachment_filename=os.path.basename(session['data_file']))
    except Exception as e:
        return render_template('preprocessing.html', tables=None, msg=f"Save Error: {str(e)}")
=======
@app.route('/save')
def save_file():
    if 'data_file' in session and os.path.exists(session['data_file']):
        return send_file(session['data_file'], as_attachment=True)
    return "ဒေါင်းလုဒ်ဆွဲရန် ဖိုင်မရှိပါ", 404
>>>>>>> d72dff6 (4months accuracy commit)

@app.route('/training', methods=['GET'])
def training():
    return render_template('training.html')

# ---------------------------------------------------------
# Model Training (70/30 Split + Keep All Columns)
# ---------------------------------------------------------
@app.route('/train_only', methods=['POST'])
def train_only():
    raw_model_type = request.form.get('model_type', 'lr').lower().strip()
    train_file = request.files.get('train_file') 

    # Dynamic Model Detection
    if 'rf' in raw_model_type or 'random' in raw_model_type:
        model_key = 'rf'
        model_name = "RANDOM FOREST"
    else:
        model_key = 'lr'
        model_name = "LOGISTIC REGRESSION"

    if not train_file:
<<<<<<< HEAD
        return render_template('training.html', result=f"❌ Error: Please select a CSV file to train {model_name}.")

    try:
        if train_file.filename.endswith('.xlsx'):
            df = pd.read_excel(train_file)
        else:
            df = pd.read_csv(train_file)

        df = df.drop_duplicates()
        df = clean_column_names(df)

        target_clean = TARGET_COLUMN_NAME.strip().lower()
        matched_targets = [c for c in df.columns if target_clean in c]
        if matched_targets:
            target_col = matched_targets[0]
        else:
            target_col = df.columns[-1]

        # Categorical Encoding
=======
        return render_template('training.html', result="❌ Error: ကျေးဇူးပြု၍ CSV ဖိုင်ကို ရွေးချယ်ပါ။")

    try:
        df = pd.read_csv(train_file)
        
        # Column များကို Drop မလုပ်ဘဲ Feature အားလုံးကို သိမ်းဆည်းခြင်း
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Label Encoding
>>>>>>> d72dff6 (4months accuracy commit)
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, ENCODER_PATH)

<<<<<<< HEAD
        X = df.drop(columns=[target_col])
        y = df[target_col].round().astype(int)

        # Target Leakage ဖြစ်စေနိုင်သော Precipitation Drop ခြင်း
        cols_to_drop = [col for col in X.columns if 'precipitation' in col]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        joblib.dump(cols_to_drop, DROPPED_COLS_PATH)
        joblib.dump(list(X.columns), FEATURE_NAMES_PATH)

        # ⚡ MODEL PIPELINE BUILDING ⚡
        if model_key == 'rf':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    n_estimators=300, 
                    max_depth=10, 
                    random_state=42, 
                    n_jobs=-1
                ))
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('model', LogisticRegression(
                    C=10.0,
                    solver='lbfgs',
                    multi_class='multinomial',
                    class_weight='balanced',
                    max_iter=5000,
                    random_state=42
                ))
            ])

        pipeline.fit(X, y)
        
        pipeline_path = get_pipeline_path(model_key)
        joblib.dump(pipeline, pipeline_path)

        y_train_pred = pipeline.predict(X)
        train_acc = accuracy_score(y, y_train_pred)

        target_msg = f"\n🎯 Identified Target Column: '{target_col}'"
        dropped_msg = f"\n⚠️ Dropped Feature: {', '.join(cols_to_drop)}" if cols_to_drop else ""

        return render_template('training.html', 
                               result=f"✅ {model_name} Trained Successfully!\n"
                                      f"📈 Train Accuracy: {train_acc*100:.2f}%" 
                                      + target_msg + dropped_msg)
=======
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
>>>>>>> d72dff6 (4months accuracy commit)

    except Exception as e:
        return render_template('training.html', result=f"❌ Train Error ({model_name}): {str(e)}")

# ---------------------------------------------------------
# Test File ဖြင့် စမ်းသပ်ခြင်း (Test Only)
# ---------------------------------------------------------
@app.route('/test_only', methods=['POST'])
def test_only():
    raw_model_type = request.form.get('model_type', 'lr').lower().strip()
    test_file = request.files.get('test_file')

    if 'rf' in raw_model_type or 'random' in raw_model_type:
        model_key = 'rf'
        model_name = "RANDOM FOREST"
    else:
        model_key = 'lr'
        model_name = "LOGISTIC REGRESSION"
    
<<<<<<< HEAD
    pipeline_path = get_pipeline_path(model_key)
    if not os.path.exists(pipeline_path):
        return render_template('training.html', result=f"⚠️ Please train the {model_name} model first!")

    if not test_file:
        return render_template('training.html', result=f"❌ Error: Please select a Test file for {model_name}.")
=======
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return render_template('training.html', result="⚠️ ကျေးဇူးပြု၍ ပထမဦးစွာ Model ကို Train ပြုလုပ်ပါ။")

    if not test_file:
        return render_template('training.html', result="❌ Error: ကျေးဇူးပြု၍ Test CSV ဖိုင်ကို ရွေးချယ်ပါ။")
>>>>>>> d72dff6 (4months accuracy commit)

    try:
        if test_file.filename.endswith('.xlsx'):
            df_test = pd.read_excel(test_file)
        else:
            df_test = pd.read_csv(test_file)

        df_test = clean_column_names(df_test)

        pipeline = joblib.load(pipeline_path)
        encoders = joblib.load(ENCODER_PATH)
        cols_to_drop = joblib.load(DROPPED_COLS_PATH) if os.path.exists(DROPPED_COLS_PATH) else []
        train_features = joblib.load(FEATURE_NAMES_PATH) if os.path.exists(FEATURE_NAMES_PATH) else []

<<<<<<< HEAD
        for col, le in encoders.items():
            if col in df_test.columns:
                df_test[col] = df_test[col].map(lambda s: le.transform([str(s)])[0] if str(s) in le.classes_ else -1)

        target_clean = TARGET_COLUMN_NAME.strip().lower()
        matched_targets = [c for c in df_test.columns if target_clean in c]
        if matched_targets:
            target_col = matched_targets[0]
        else:
            target_col = df_test.columns[-1]

        X_test = df_test.drop(columns=[target_col], errors='ignore')
        y_test = df_test[target_col].round().astype(int)

        if cols_to_drop:
            X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns], errors='ignore')

        if train_features:
            X_test = X_test[train_features]

        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        actual_counts = pd.Series(y_test).value_counts().to_dict()
        pred_counts = pd.Series(predictions).value_counts().to_dict()
        cm = confusion_matrix(y_test, predictions)

        status_msg = f"✅ Excellent {model_name} Performance (>80%)" if acc >= 0.80 else "⚠️ Test Evaluation Complete"

        analysis = (
            f"\n\n🔍 --- {model_name} Breakdown ({len(y_test)} Rows) --- \n"
            f"• {status_msg}\n"
            f"• Actual Labels Count: {actual_counts}\n"
            f"• Predicted Labels Count: {pred_counts}\n"
            f"• Confusion Matrix:\n{cm}"
        )

        return render_template('training.html', result=f"🎯 Test Accuracy Score ({model_name}): {acc*100:.2f}%" + analysis)
=======
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
>>>>>>> d72dff6 (4months accuracy commit)

    except Exception as e:
        return render_template('training.html', result=f"❌ Test Error ({model_name}): {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)