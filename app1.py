
from flask import Flask, render_template, request, send_file, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
#from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "tu_meiktila_weather_project_secret"

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

ENCODER_PATH = os.path.join(MODEL_FOLDER, 'encoders.pkl')
TARGET_ENCODER_PATH = os.path.join(MODEL_FOLDER, 'target_encoder.pkl')
DROPPED_COLS_PATH = os.path.join(MODEL_FOLDER, 'dropped_cols.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_FOLDER, 'feature_names.pkl')
IMPUTER_PATH = os.path.join(MODEL_FOLDER, 'imputer.pkl')

TARGET_COLUMN_NAME = 'Rain'

def clean_column_names(df):
    """Column နာမည်များတွင် ပါဝင်သော Space များနှင့် Capitalization များကို ရှင်းလင်းပေးသည်"""
    df.columns = df.columns.str.strip().str.lower()
    return df

def get_pipeline_path(model_type_key):
    return os.path.join(MODEL_FOLDER, f'weather_pipeline_{model_type_key}.pkl')

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

# ---------------------------------------------------------
# 🎯 MODEL TRAINING ROUTE
# ---------------------------------------------------------
#@app.route('/train_only', methods=['POST'])

# @app.route('/train_only', methods=['POST'])
# def train_only():

#     train_file = request.files.get("train_file")

#     if train_file is None:
#         return render_template(
#             "training.html",
#             result="❌ Please choose a training file."
#         )

#     try:

#         # --------------------------------------------------
#         # Read dataset
#         # --------------------------------------------------
#         if train_file.filename.endswith(".xlsx"):
#             df = pd.read_excel(train_file)
#         else:
#             df = pd.read_csv(train_file)

#         df = clean_column_names(df)
#         df = df.drop_duplicates()

#         # --------------------------------------------------
#         # Detect target column
#         # --------------------------------------------------
#         target_candidates = [
#             c for c in df.columns
#             if "weather type" in c.lower()
#             or c.lower() == "rain"
#         ]

#         if len(target_candidates) == 0:
#             return render_template(
#                 "training.html",
#                 result="❌ Target column not found."
#             )

#         target_col = target_candidates[0]

#         # --------------------------------------------------
#         # Features & Target
#         # --------------------------------------------------
#         X = df.drop(columns=[target_col])
#         y = df[target_col].astype(str)

#         # Save target encoder only
#         target_encoder = LabelEncoder()
#         y = target_encoder.fit_transform(y)

#         joblib.dump(target_encoder, TARGET_ENCODER_PATH)

#         # --------------------------------------------------
#         # Feature types
#         # --------------------------------------------------
#         categorical_cols = X.select_dtypes(
#             include=["object"]
#         ).columns.tolist()

#         numeric_cols = X.select_dtypes(
#             exclude=["object"]
#         ).columns.tolist()

#         # --------------------------------------------------
#         # Preprocessing
#         # --------------------------------------------------
#         numeric_transformer = Pipeline([
#             ("imputer", SimpleImputer(strategy="median"))
#         ])

#         categorical_transformer = Pipeline([
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("encoder", OneHotEncoder(handle_unknown="ignore"))
#         ])

#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", numeric_transformer, numeric_cols),
#                 ("cat", categorical_transformer, categorical_cols)
#             ]
#         )

#         # --------------------------------------------------
#         # Train/Test Split
#         # --------------------------------------------------
#         X_train, X_valid, y_train, y_valid = train_test_split(
#             X,
#             y,
#             test_size=0.20,
#             stratify=y,
#             random_state=42
#         )

#         # --------------------------------------------------
#         # Pipeline
#         # --------------------------------------------------
#         pipeline = Pipeline([
#             ("prep", preprocessor),
#             ("model", RandomForestClassifier(
#                 n_estimators=500,
#                 max_depth=15,
#                 min_samples_leaf=5,
#                 min_samples_split=10,
#                 max_features="sqrt",
#                 bootstrap=True,
#                 class_weight="balanced",
#                 random_state=42,
#                 n_jobs=-1
#             ))
#         ])

#         # --------------------------------------------------
#         # Grid Search
#         # --------------------------------------------------
#         params = {

#             "model__n_estimators":[300,500],

#             "model__max_depth":[10,20,None],

#             "model__min_samples_split":[2,5],

#             "model__min_samples_leaf":[1,2],

#             "model__max_features":["sqrt"]
#         }

#         grid = GridSearchCV(

#             pipeline,

#             params,

#             cv=5,

#             scoring="accuracy",

#             n_jobs=-1

#         )

#         grid.fit(X_train, y_train)

#         best_model = grid.best_estimator_

#         # --------------------------------------------------
#         # Validation Accuracy
#         # --------------------------------------------------
#         pred = best_model.predict(X_valid)

#         acc = accuracy_score(
#             y_valid,
#             pred
#         )

#         # --------------------------------------------------
#         # Save ONE model
#         # --------------------------------------------------
#         joblib.dump(
#             best_model,
#             get_pipeline_path("rf")
#         )

#         return render_template(

#             "training.html",

#             result=
#             f"✅ Random Forest trained successfully.\n\n"
#             f"Validation Accuracy : {acc*100:.2f}%\n\n"
#             #f"Best Parameters :\n{grid.best_params_}"

#         )

#     except Exception as e:

#         return render_template(
#             "training.html",
#             result=f"❌ {str(e)}"
#         )
@app.route('/train_only', methods=['POST'])
def train_only():

    train_file = request.files.get("train_file")

    if train_file is None:
        return render_template(
            "training.html",
            result="❌ Please choose a training file."
        )

    try:

        # --------------------------------------------------
        # Load dataset
        # --------------------------------------------------

        if train_file.filename.endswith(".xlsx"):
            df = pd.read_excel(train_file)
        else:
            df = pd.read_csv(train_file)

        df = clean_column_names(df)
        df = df.drop_duplicates()


        # --------------------------------------------------
        # Find target column
        # --------------------------------------------------

        target_candidates = [
            c for c in df.columns
            if "weather type" in c.lower()
            or c.lower() == "rain"
        ]

        if not target_candidates:
            return render_template(
                "training.html",
                result="❌ Target column not found."
            )


        target_col = target_candidates[0]


        # --------------------------------------------------
        # Features / Target
        # --------------------------------------------------

        X = df.drop(columns=[target_col])

        y = df[target_col].astype(str)


        # Encode target
        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(y)

        joblib.dump(
            target_encoder,
            TARGET_ENCODER_PATH
        )


        # --------------------------------------------------
        # Detect feature types
        # --------------------------------------------------

        categorical_cols = X.select_dtypes(
            include=["object"]
        ).columns.tolist()


        numeric_cols = X.select_dtypes(
            exclude=["object"]
        ).columns.tolist()



        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        numeric_transformer = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ])


        categorical_transformer = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])



        preprocessor = ColumnTransformer([

            (
                "num",
                numeric_transformer,
                numeric_cols
            ),

            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        ])



        # --------------------------------------------------
        # Train / Validation split
        # --------------------------------------------------

        X_train, X_valid, y_train, y_valid = train_test_split(

            X,
            y,

            test_size=0.2,

            stratify=y,

            random_state=42
        )



        # --------------------------------------------------
        # Regularized Random Forest
        # --------------------------------------------------

        rf = RandomForestClassifier(

            n_estimators=300,

            # prevent memorization
            max_depth=10,

            min_samples_split=15,

            min_samples_leaf=8,

            max_features="sqrt",

            bootstrap=True,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1

        )



        pipeline = Pipeline([

            (
                "prep",
                preprocessor
            ),

            (
                "model",
                rf
            )

        ])



        # --------------------------------------------------
        # Cross validation
        # --------------------------------------------------

        cv = StratifiedKFold(

            n_splits=5,

            shuffle=True,

            random_state=42
        )



        params = {


            "model__n_estimators":
                [200,300],


            "model__max_depth":
                [8,10,12],


            "model__min_samples_split":
                [10,15,20],


            "model__min_samples_leaf":
                [5,8,10],


            "model__max_features":
                ["sqrt"]

        }



        grid = GridSearchCV(

            pipeline,

            params,

            cv=cv,

            scoring="accuracy",

            n_jobs=-1

        )



        grid.fit(
            X_train,
            y_train
        )



        best_model = grid.best_estimator_



        # --------------------------------------------------
        # Evaluate
        # --------------------------------------------------

        train_pred = best_model.predict(
            X_train
        )

        valid_pred = best_model.predict(
            X_valid
        )


        train_acc = accuracy_score(
            y_train,
            train_pred
        )


        valid_acc = accuracy_score(
            y_valid,
            valid_pred
        )


        cv_acc = grid.best_score_



        # --------------------------------------------------
        # Save model
        # --------------------------------------------------

        joblib.dump(

            best_model,

            get_pipeline_path("rf")

        )



        return render_template(

            "training.html",

            result=(

                "✅ Random Forest trained successfully.\n\n"

                f"Training Accuracy : {train_acc*100:.2f}%\n"

                f"Validation Accuracy : {valid_acc*100:.2f}%\n"

                f"CV Accuracy : {cv_acc*100:.2f}%\n\n"

                f"Best Parameters:\n"

                f"{grid.best_params_}"

            )

        )



    except Exception as e:

        return render_template(

            "training.html",

            result=f"❌ {str(e)}"

        )
@app.route('/test_only', methods=['POST'])
def test_only():

    test_file = request.files.get("test_file")

    if test_file is None:
        return render_template(
            "training.html",
            result="❌ Please choose a test file."
        )

    try:

        # ------------------------------------------
        # Load test data
        # ------------------------------------------
        if test_file.filename.endswith(".xlsx"):
            df = pd.read_excel(test_file)
        else:
            df = pd.read_csv(test_file)

        df = clean_column_names(df)

        # ------------------------------------------
        # Load trained pipeline
        # ------------------------------------------
        pipeline = joblib.load(get_pipeline_path("rf"))

        target_encoder = joblib.load(TARGET_ENCODER_PATH)

        # ------------------------------------------
        # Find target column
        # ------------------------------------------
        target_col = None

        for c in df.columns:
            if "weather type" in c.lower() or c.lower() == "rain":
                target_col = c
                break

        if target_col is None:
            return render_template(
                "training.html",
                result="❌ Target column not found."
            )

        # ------------------------------------------
        # Split X and y
        # ------------------------------------------
        X_test = df.drop(columns=[target_col])

        y_true_text = df[target_col].astype(str)

        from sklearn.preprocessing import LabelEncoder

        all_labels = list(target_encoder.classes_)

        for lbl in y_true_text.unique():
            if lbl not in all_labels:
                all_labels.append(lbl)

        temp_encoder = LabelEncoder()
        temp_encoder.fit(all_labels)

        y_true = temp_encoder.transform(y_true_text)

        #y_true = target_encoder.transform(y_true_text)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------
        y_pred = pipeline.predict(X_test)

        y_pred_text = target_encoder.inverse_transform(y_pred)

        # ------------------------------------------
        # Accuracy
        # ------------------------------------------
        acc = accuracy_score(y_true_text, y_pred_text)

        report = classification_report(
            y_true_text,
            y_pred_text,
            zero_division=0
        )

        cm = confusion_matrix(y_true_text, y_pred_text)

        # ------------------------------------------
        # Comparison Table
        # ------------------------------------------
        result_df = X_test.copy()

        result_df["Actual"] = y_true_text.values
        result_df["Prediction"] = y_pred_text

        result_df["Correct"] = (
            result_df["Actual"] ==
            result_df["Prediction"]
        )

        table = result_df.to_html(
            index=False,
            classes="table table-striped table-bordered"
        )

        return render_template(
            "training.html",
            result=f"✅ Test Accuracy : {acc*100:.2f}%",
            report=report,
            confusion_matrix=cm,
            comparison_table=table
        )

    except Exception as e:

        return render_template(
            "training.html",
            result=f"❌ Test Error : {str(e)}"
        )
# def train_only():
#     raw_model_type = request.form.get('model_type', 'lr').lower().strip()
#     train_file = request.files.get('train_file') 

#     if 'rf' in raw_model_type or 'random' in raw_model_type:
#         model_key = 'rf'
#         model_name = "RANDOM FOREST"
#     else:
#         model_key = 'lr'
#         model_name = "LOGISTIC REGRESSION"

#     if not train_file:
#         return render_template('training.html', result=f"❌ Error: Please select a file to train {model_name}.")

#     try:
#         if train_file.filename.endswith('.xlsx'):
#             df = pd.read_excel(train_file)
#         else:
#             df = pd.read_csv(train_file)

#         # ၁။ Clean Column Names & Remove Duplicates
#         df = df.drop_duplicates()
#         df = clean_column_names(df)

#         # ၂။ Target Column အား ရှာဖွေခြင်း
#         target_clean = TARGET_COLUMN_NAME.strip().lower()
#         matched_targets = [c for c in df.columns if target_clean in c]
#         target_col = matched_targets[0] if matched_targets else df.columns[-1]

#         # ၃။ Categorical Feature Encoders
#         encoders = {}
#         for col in df.columns:
#             if col != target_col and df[col].dtype == 'object':
#                 le = LabelEncoder()
#                 df[col] = le.fit_transform(df[col].astype(str))
#                 encoders[col] = le
#         joblib.dump(encoders, ENCODER_PATH)

#         # ၄။ Target Column Label Encoding (Original Class Name သိမ်းရန်)
#         if df[target_col].dtype == 'object' or df[target_col].dtype == 'category':
#             target_le = LabelEncoder()
#             y = target_le.fit_transform(df[target_col].astype(str))
#             joblib.dump(target_le, TARGET_ENCODER_PATH)
#         else:
#             y = df[target_col].round().astype(int).values

#         X = df.drop(columns=[target_col])

#         # ၅။ Dropped Columns & Feature Names သိမ်းဆည်းခြင်း
#         cols_to_drop = [col for col in X.columns if 'precipitation' in col]
#         if cols_to_drop:
#             X = X.drop(columns=cols_to_drop)
#         joblib.dump(cols_to_drop, DROPPED_COLS_PATH)
#         joblib.dump(list(X.columns), FEATURE_NAMES_PATH)

#         # ၆။ Numerical Missing Value Imputer
#         imputer = SimpleImputer(strategy='mean')
#         X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
#         joblib.dump(imputer, IMPUTER_PATH)

#         # ၇။ Pipeline တည်ဆောက်ခြင်း (Scaler + Model)
#         if model_key == 'rf':
#             pipeline = Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('model', RandomForestClassifier(
#                     n_estimators=500,
#                     max_depth=15,
#                     min_samples_leaf=5,
#                     min_samples_split=10,
#                     max_features="sqrt",
#                     bootstrap=True,
#                     class_weight="balanced",
#                     random_state=42,
#                     n_jobs=-1
#                 ))
#             ])
#         else:
#             pipeline = Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#                 ('model', LogisticRegression(
#                     C=10.0,
#                     solver='lbfgs',
#                     multi_class='multinomial',
#                     class_weight='balanced',
#                     max_iter=5000,
#                     random_state=42
#                 ))
#             ])

#         pipeline.fit(X_imputed, y)
#         joblib.dump(pipeline, get_pipeline_path(model_key))

#         y_train_pred = pipeline.predict(X_imputed)
#         train_acc = accuracy_score(y, y_train_pred)

#         target_msg = f"\n🎯 Identified Target Column: '{target_col}'"
#         dropped_msg = f"\n⚠️ Dropped Feature: {', '.join(cols_to_drop)}" if cols_to_drop else ""

#         return render_template('training.html', 
#                                result=f"✅ {model_name} Trained Successfully!\n"
#                                       f"📈 Train Accuracy: {train_acc*100:.2f}%" 
#                                       + target_msg + dropped_msg)

#     except Exception as e:
#         return render_template('training.html', result=f"❌ Train Error ({model_name}): {str(e)}")


# # ---------------------------------------------------------
# # 🎯 TEST EVALUATION ROUTE (TRAIN အဆင့်အတိုင်း ၁၀၀% တူအောင် ပြင်ထားသည်)
# # ---------------------------------------------------------
# @app.route('/test_only', methods=['POST'])
# #@app.route('/test_only', methods=['POST'])
# def test_only():

#     raw_model_type = request.form.get('model_type', 'rf').lower().strip()
#     test_file = request.files.get('test_file')

#     model_key = 'rf' if 'rf' in raw_model_type else 'lr'
#     model_name = "RANDOM FOREST" if model_key == 'rf' else "LOGISTIC REGRESSION"

#     pipeline_path = get_pipeline_path(model_key)

#     if not os.path.exists(pipeline_path):
#         return render_template(
#             'training.html',
#             result=f"Please train {model_name} first."
#         )

#     if not test_file:
#         return render_template(
#             'training.html',
#             result="Please choose test file."
#         )

#     try:

#         # -----------------------------
#         # Load test data
#         # -----------------------------
#         if test_file.filename.endswith(".xlsx"):
#             df = pd.read_excel(test_file)
#         else:
#             df = pd.read_csv(test_file)

#         df = clean_column_names(df)

#         # -----------------------------
#         # Load saved objects
#         # -----------------------------
#         pipeline = joblib.load(pipeline_path)

#         train_features = joblib.load(FEATURE_NAMES_PATH)

#         cols_to_drop = joblib.load(DROPPED_COLS_PATH)

#         imputer = joblib.load(IMPUTER_PATH)

#         target_encoder = None
#         if os.path.exists(TARGET_ENCODER_PATH):
#             target_encoder = joblib.load(TARGET_ENCODER_PATH)

#         # -----------------------------
#         # Detect target column
#         # -----------------------------
#         target_col = None

#         for c in df.columns:
#             if "weather type" in c.lower() or "rain" == c.lower():
#                 target_col = c
#                 break

#         y_true = None

#         if target_col is not None:

#             y_true_text = df[target_col].astype(str)

#             if target_encoder is not None:
#                 y_true = target_encoder.transform(y_true_text)

#             df = df.drop(columns=[target_col])

#         # -----------------------------
#         # Remove dropped columns
#         # -----------------------------
#         df = df.drop(
#             columns=[c for c in cols_to_drop if c in df.columns],
#             errors="ignore"
#         )

#         # -----------------------------
#         # Match training columns
#         # -----------------------------
#         df = df.reindex(columns=train_features, fill_value=0)

#         # -----------------------------
#         # Missing value handling
#         # -----------------------------
#         X = pd.DataFrame(
#             imputer.transform(df),
#             columns=train_features
#         )

#         # -----------------------------
#         # Prediction
#         # -----------------------------
#         pred = pipeline.predict(X)

#         if target_encoder is not None:
#             pred_text = target_encoder.inverse_transform(pred)
#         else:
#             pred_text = pred

#         # -----------------------------
#         # Result table
#         # -----------------------------
#         result_df = pd.DataFrame({
#             "Prediction": pred_text
#         })

#         if y_true is not None:

#             acc = accuracy_score(y_true, pred)

#             report = classification_report(
#                 y_true,
#                 pred,
#                 zero_division=0
#             )

#             result_df.insert(
#                 0,
#                 "Actual",
#                 y_true_text.values
#             )

#             table = result_df.to_html(
#                 index=False,
#                 classes="table table-striped table-bordered"
#             )

#             return render_template(
#                 "training.html",
#                 result=f"Test Accuracy ({model_name}) : {acc*100:.2f}%",
#                 report=report,
#                 comparison_table=table
#             )

#         else:

#             table = result_df.to_html(
#                 index=False,
#                 classes="table table-striped table-bordered"
#             )

#             return render_template(
#                 "training.html",
#                 result="Prediction completed.",
#                 comparison_table=table
#             )

#     except Exception as e:

#         return render_template(
#             "training.html",
#             result=f"Test Error : {str(e)}"
#         )
# def test_only():
#     raw_model_type = request.form.get('model_type', 'lr').lower().strip()
#     test_file = request.files.get('test_file')

#     if 'rf' in raw_model_type or 'random' in raw_model_type:
#         model_key = 'rf'
#         model_name = "RANDOM FOREST"
#     else:
#         model_key = 'lr'
#         model_name = "LOGISTIC REGRESSION"
    
#     pipeline_path = get_pipeline_path(model_key)
#     if not os.path.exists(pipeline_path):
#         return render_template('training.html', result=f"⚠️ Please train the {model_name} model first!")

#     if not test_file:
#         return render_template('training.html', result=f"❌ Error: Please select a Test file for {model_name}.")

#     try:
#         if test_file.filename.endswith('.xlsx'):
#             df_test = pd.read_excel(test_file)
#         else:
#             df_test = pd.read_csv(test_file)

#         # ၁။ Clean Column Names (Train တုန်းက အတိုင်း)
#         df_test = clean_column_names(df_test)

#         pipeline = joblib.load(pipeline_path)
#         encoders = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else {}
#         cols_to_drop = joblib.load(DROPPED_COLS_PATH) if os.path.exists(DROPPED_COLS_PATH) else []
#         train_features = joblib.load(FEATURE_NAMES_PATH) if os.path.exists(FEATURE_NAMES_PATH) else []
#         imputer = joblib.load(IMPUTER_PATH) if os.path.exists(IMPUTER_PATH) else None

#         # ၂။ Feature Label Encoding (Train တုန်းက အတိုင်း)
#         for col, le in encoders.items():
#             if col in df_test.columns:
#                 df_test[col] = df_test[col].map(lambda s: le.transform([str(s)])[0] if str(s) in le.classes_ else -1)

#         # ၃။ Target Column ရှာဖွေခြင်း & Encoding
#         target_clean = TARGET_COLUMN_NAME.strip().lower()
#         matched_targets = [c for c in df_test.columns if target_clean in c]
#         target_col = matched_targets[0] if matched_targets else df_test.columns[-1]

#         y_test_orig = df_test[target_col].copy()

#         if os.path.exists(TARGET_ENCODER_PATH):
#             target_le = joblib.load(TARGET_ENCODER_PATH)
#             if y_test_orig.dtype == 'object' or y_test_orig.dtype == 'category':
#                 y_test = y_test_orig.astype(str).map(lambda s: target_le.transform([str(s)])[0] if str(s) in target_le.classes_ else -1).values
#             else:
#                 y_test = y_test_orig.round().astype(int).values
#         else:
#             y_test = y_test_orig.round().astype(int).values

#         X_test = df_test.drop(columns=[target_col], errors='ignore')

#         # ၄။ Drop Precipitation & Match Columns Order (Train တုန်းက အတိုင်း)
#         if cols_to_drop:
#             X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns], errors='ignore')

#         if train_features:
#             X_test = X_test.reindex(columns=train_features, fill_value=0)

#         # ၅။ Imputation (Train တုန်းက Fit ခဲ့သော Imputer ဖြင့် Transform သာလုပ်မည်)
#         if imputer:
#             X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
#         else:
#             X_test_imputed = X_test

#         # ၆။ Prediction (Pipeline က StandardScaler ကို အလိုအလျောက် တူညီစွာ Transform လုပ်ပေးသည်)
#         predictions = pipeline.predict(X_test_imputed)
#         acc = accuracy_score(y_test, predictions)

#         # ၇။ Numeric Predictions များကို Original Text Labels (Sunny, Rainy, etc.) သို့ ပြန်ပြောင်းခြင်း
#         if os.path.exists(TARGET_ENCODER_PATH):
#             target_le = joblib.load(TARGET_ENCODER_PATH)
#             pred_labels = [target_le.inverse_transform([p])[0] if 0 <= p < len(target_le.classes_) else str(p) for p in predictions]
#         else:
#             pred_labels = predictions.tolist()

#         # ၈။ Actual Weather vs Predicted Weather Table တည်ဆောက်ခြင်း
#         df_comparison = pd.DataFrame({
#             'Actual Weather': y_test_orig.values,
#             'Predicted Weather': pred_labels
#         })

#         comparison_table = df_comparison.to_html(
#             classes='table table-striped table-hover table-bordered text-center', 
#             index=False
#         )

#         # Classification Report ထုတ်ယူခြင်း
#         report = classification_report(y_test, predictions, zero_division=0)

#         return render_template(
#             'training.html', 
#             result=f"🎯 Test Accuracy Score ({model_name}): {acc*100:.2f}%",
#             report=report,
#             comparison_table=comparison_table
#         )

#     except Exception as e:
#         return render_template('training.html', result=f"❌ Test Error ({model_name}): {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)