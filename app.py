from flask import Flask, render_template, request, send_file, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.secret_key = "tu_meiktila_weather_project_secret"

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

PIPELINE_PATH = os.path.join(MODEL_FOLDER, 'weather_pipeline.pkl')
ENCODER_PATH = os.path.join(MODEL_FOLDER, 'encoders.pkl')
DROPPED_COLS_PATH = os.path.join(MODEL_FOLDER, 'dropped_cols.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_FOLDER, 'feature_names.pkl')

# 🎯 Target Column နာမည် (CSV ထဲရှိ Target Column နှင့် Auto Match ပြုလုပ်ပေးပါမည်)
TARGET_COLUMN_NAME = 'Rain'

def clean_column_names(df):
    """Column နာမည်များတွင် ပါဝင်သော Space များနှင့် Capitalization များကို ရှင်းလင်းပေးသည်"""
    df.columns = df.columns.str.strip().str.lower()
    return df

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/preprocessing')
def preprocessing():
    if 'data_file' in session and os.path.exists(session['data_file']):
        df = pd.read_csv(session['data_file'])
        return render_template('preprocessing.html', 
                               tables=[df.head(10).to_html(classes='data')], 
                               msg="File loaded. Duplicates & Missing Values check is ready.")
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
        df = df.drop_duplicates()
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                mask = df[col].notnull()
                df.loc[mask, col] = le.fit_transform(df[col].loc[mask].astype(str))
        
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        msg = "Duplicates removed and Missing values handled!"

    elif action == 'clear':
        if os.path.exists(filepath):
            os.remove(filepath)
        session.pop('data_file', None)
        return redirect(url_for('preprocessing'))

    df.to_csv(filepath, index=False)
    return render_template('preprocessing.html', tables=[df.head(10).to_html(classes='data')], msg=msg)

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
        encoders = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, ENCODER_PATH)

        X = df.drop(columns=[target_col])
        y = df[target_col].round().astype(int)

        # Target Leakage ဖြစ်စေနိုင်သော Precipitation Feature ကို Drop ပြုလုပ်ခြင်း
        cols_to_drop = [col for col in X.columns if 'precipitation' in col]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        joblib.dump(cols_to_drop, DROPPED_COLS_PATH)
        joblib.dump(list(X.columns), FEATURE_NAMES_PATH)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # 🌲 Multi-class Weather Prediction အတွက် Optimised ပြုလုပ်ထားသော Random Forest 🌲
        if model_type == 'rf':
            clf = RandomForestClassifier(
                n_estimators=300,        # Forest Size ကို တိုးမြှင့်ထားသည်
                max_depth=10,            # Decision Depth ကို ၁၀ အထိ တိုးမြှင့်၍ Class 3 Pattern ကို ခွဲထုတ်ပေးသည်
                min_samples_split=2,    
                min_samples_leaf=1,      
                max_features='log2',     # Feature Overlapping ပြဿနာကို လျှော့ချပေးသည်
                class_weight='balanced_subsample', # Minority/Overlapping Class များကို ပိုမို အလေးပေးသည်
                random_state=42,
                n_jobs=-1
            )
        else:
            clf = LogisticRegression(
                max_iter=3000, 
                C=1.0,
                class_weight='balanced',
                solver='lbfgs',
                random_state=42
            )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clf)
        ])

        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, PIPELINE_PATH)

        y_train_pred = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        feature_info = "\n\n📊 --- Feature Contribution --- \n"
        trained_model = pipeline.named_steps['model']

        if model_type == 'rf':
            importances = trained_model.feature_importances_
            for col, imp in zip(X.columns, importances):
                feature_info += f"• {col}: {imp*100:.2f}%\n"

        target_msg = f"\n🎯 Identified Target Column: '{target_col}'"
        dropped_msg = f"\n⚠️ Dropped Feature: {', '.join(cols_to_drop)}" if cols_to_drop else ""

        return render_template('training.html', 
                               result=f"✅ {model_type.upper()} Model Trained Successfully!\n"
                                      f"📈 Train Accuracy: {train_acc*100:.2f}%\n"
                                      f"🎯 Validation Accuracy: {val_acc*100:.2f}%" 
                                      + target_msg + dropped_msg + feature_info)

    except Exception as e:
        return render_template('training.html', result=f"❌ Train Error: {str(e)}")

@app.route('/test_only', methods=['POST'])
def test_only():
    test_file = request.files.get('test_file')
    
    if not os.path.exists(PIPELINE_PATH):
        return render_template('training.html', result="⚠️ Please train the model first!")

    if not test_file:
        return render_template('training.html', result="❌ Error: Please select a Test file.")

    try:
        if test_file.filename.endswith('.xlsx'):
            df_test = pd.read_excel(test_file)
        else:
            df_test = pd.read_csv(test_file)

        df_test = clean_column_names(df_test)

        pipeline = joblib.load(PIPELINE_PATH)
        encoders = joblib.load(ENCODER_PATH)
        cols_to_drop = joblib.load(DROPPED_COLS_PATH) if os.path.exists(DROPPED_COLS_PATH) else []
        train_features = joblib.load(FEATURE_NAMES_PATH) if os.path.exists(FEATURE_NAMES_PATH) else []

        # Categorical Encoding
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

        # Feature Alignment
        if train_features:
            X_test = X_test[train_features]

        # Prediction ပြုလုပ်ခြင်း
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # Prediction Analysis ရလဒ်များ
        actual_counts = pd.Series(y_test).value_counts().to_dict()
        pred_counts = pd.Series(predictions).value_counts().to_dict()
        cm = confusion_matrix(y_test, predictions)

        analysis = (
            f"\n\n🔍 --- Test Breakdown (31 Rows) --- \n"
            f"• Actual Labels Count: {actual_counts}\n"
            f"• Predicted Labels Count: {pred_counts}\n"
            f"• Confusion Matrix:\n{cm}"
        )

        return render_template('training.html', result=f"🎯 Test Accuracy Score: {acc*100:.2f}%" + analysis)

    except Exception as e:
        return render_template('training.html', result=f"❌ Test Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)