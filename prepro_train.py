# Stevanus Gerald Marconus - 2802392500
from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, classification_report


class Cleaning(BaseEstimator, TransformerMixin):
    NUM_ERROR = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Changed_Credit_Limit'
    ]
    CLIP_COLS = [
        'Interest_Rate', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Num_of_Loan', 'Num_Credit_Inquiries', 'Delay_from_due_date'
    ]

    @staticmethod
    def _clean_numeric(x):
        if isinstance(x, str):
            x = x.replace('_', '').replace(',', '')
            try:    return float(x)
            except: return np.nan
        return x

    def _base_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'Age' in X.columns:
            X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
            X.loc[(X['Age'] < 18) | (X['Age'] > 100), 'Age'] = np.nan
        if 'Num_of_Loan' in X.columns:
            X['Num_of_Loan'] = pd.to_numeric(X['Num_of_Loan'], errors='coerce')
        for col in self.NUM_ERROR:
            if col in X.columns:
                X[col] = X[col].apply(self._clean_numeric)
        if 'Occupation' in X.columns:
            X['Occupation'] = X['Occupation'].replace('_______', 'Unknown')
        if 'Payment_Behaviour' in X.columns:
            X['Payment_Behaviour'] = X['Payment_Behaviour'].replace('!@9#%8', 'Unknown')
        return X

    def fit(self, X, y=None):
        X = X.copy()
        X = self._base_clean(X)
        self.clip_uppers_ = {}
        for col in self.CLIP_COLS:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                self.clip_uppers_[col] = X[col].quantile(0.99)
        self.cat_cols_ = list(X.select_dtypes(include='object').columns)
        self.num_cols_ = list(X.select_dtypes(exclude='object').columns)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = self._base_clean(X)
        for col, upper in self.clip_uppers_.items():
            if col in X.columns:
                X.loc[X[col] < 0, col] = 0
                X[col] = X[col].clip(upper=upper)
        for col in self.cat_cols_:
            if col in X.columns: X[col] = X[col].astype(str)
        for col in self.num_cols_:
            if col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
        return X

class Preprocess:
    def __init__(self):
        self.test_size = 0.2
        self.random_state = 42
        self.cleaner = Cleaning()
        self.le_target = LabelEncoder()
        self.smote = SMOTE(random_state=42)

    def clean_and_split(self, data_path: str | Path) -> Tuple:
        print("Preprocessing")
        df = pd.read_csv(Path(data_path))
        df = df.drop(columns=['Unnamed: 0', 'ID', 'Customer_ID', 'SSN', 'Name', 'Month', 'Type_of_Loan', 'Credit_History_Age'],errors='ignore')
        X = df.drop(['Credit_Score'], axis=1)
        y = df['Credit_Score']

        x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=self.test_size,random_state=self.random_state,stratify=y)

        y_train_enc = self.le_target.fit_transform(y_train)
        y_test_enc  = self.le_target.transform(y_test)
        return x_train, x_test, y_train_enc, y_test_enc

    def build_transformer(self, x_train: pd.DataFrame):
        self.cleaner.fit(x_train)
        cat_cols = self.cleaner.cat_cols_
        num_cols = self.cleaner.num_cols_

        numeric_pipeline = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median')),
            ('num_scaler',  RobustScaler())
        ])

        categorical_pipeline = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer(transformers=[
            ('numPreprocess', numeric_pipeline,     num_cols),
            ('catPreprocess', categorical_pipeline, cat_cols),
        ], remainder='drop')

    def Smote(self, x_train, col_transformer):
        x_cleaned     = self.cleaner.transform(x_train)
        x_transformed = col_transformer.fit_transform(x_cleaned)

        x_smote, y_smote = self.smote.fit_resample(x_transformed, self._y_train_enc)
        return x_smote, y_smote


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.pipeline_  = None
        self.metrics_ = {}
        self.y_pred_ = None
        self.y_test_ = None
        self.best_params_ = None

    @abstractmethod
    def train(self, X_train_smote, y_train_smote, cleaner, col_transformer):
        pass

    @abstractmethod
    def test(self, X_test, y_test): 
        pass

    def _compute_metrics(self, y_test, y_pred):
        self.y_pred_  = y_pred
        self.y_test_  = y_test
        self.metrics_ = {
            'accuracy' : accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        return self.metrics_

    def report(self, label_encoder: LabelEncoder):
        print(classification_report(self.y_test_, self.y_pred_,target_names=label_encoder.classes_))

    def _build_pipeline(self, clf, cleaner, col_transformer):
        return Pipeline([
            ('cleaner', cleaner),
            ('preprocess', col_transformer),
            ('classifier', clf)
        ])

class RandomForestModel1(BaseModel):
    def __init__(self, name: str, param_grid: dict = None, cv: int = 3):
        super().__init__(name)
        self.param_grid = param_grid or {
            'n_estimators': [100, 200],
            'max_depth' : [10, 20],
            'min_samples_split': [2, 5, 10]
        }
        self.cv = cv

    def train(self, X_train_smote, y_train_smote, cleaner, col_transformer):
        print(f" Train {self.name}")
        clf  = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(
            clf, self.param_grid,
            cv=self.cv, scoring='f1_weighted', n_jobs=-1, verbose=0
        )
        grid.fit(X_train_smote, y_train_smote)
        best_clf = grid.best_estimator_
        self.best_params_ = grid.best_params_

        self.pipeline_ = self._build_pipeline(best_clf, cleaner, col_transformer)
        return self

    def test(self, X_test, y_test):
        self._compute_metrics(y_test, self.pipeline_.predict(X_test))
        print(f" F1: {self.metrics_['f1']:.4f} | Acc: {self.metrics_['accuracy']:.4f} | Prec: {self.metrics_['precision']:.4f} | Rec: {self.metrics_['recall']:.4f}")
        return self

class XGBoostModel2(BaseModel):
    def __init__(self, name: str, param_grid: dict = None, cv: int = 3):
        super().__init__(name)
        self.param_grid = param_grid or {
            'n_estimators' : [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.05, 0.1]
        }
        self.cv = cv

    def train(self, X_train_smote, y_train_smote, cleaner, col_transformer):
        print(f" Train {self.name}")
        clf  = XGBClassifier(
            objective='multi:softmax', eval_metric='mlogloss',
            random_state=42, verbosity=0
        )
        grid = GridSearchCV(
            clf, self.param_grid,
            cv=self.cv, scoring='f1_weighted', n_jobs=-1, verbose=0
        )
        grid.fit(X_train_smote, y_train_smote)
        best_clf = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.pipeline_ = self._build_pipeline(best_clf, cleaner, col_transformer)
        print(f"best: {self.best_params_}")
        return self

    def test(self, X_test, y_test):
        self._compute_metrics(y_test, self.pipeline_.predict(X_test))
        print(f"F1: {self.metrics_['f1']:.4f} | Acc: {self.metrics_['accuracy']:.4f} | Prec: {self.metrics_['precision']:.4f} | Rec: {self.metrics_['recall']:.4f}")
        return self


class LightGBMModel3(BaseModel):
    def __init__(self, name: str, param_grid: dict = None, cv: int = 3):
        super().__init__(name)
        self.param_grid = param_grid or {
            'n_estimators' : [200, 300],
            'max_depth' : [6, 8],
            'learning_rate': [0.05, 0.1],
            'num_leaves' : [31, 63]
        } 
        self.cv = cv

    def train(self, X_train_smote, y_train_smote, cleaner, col_transformer):
        print(f" Train {self.name}")
        clf  = LGBMClassifier(
            objective='multiclass', random_state=42, verbose=-1
        )
        grid = GridSearchCV(
            clf, self.param_grid,
            cv=self.cv, scoring='f1_weighted', n_jobs=-1, verbose=0
        )
        grid.fit(X_train_smote, y_train_smote)
        best_clf = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.pipeline_ = self._build_pipeline(best_clf, cleaner, col_transformer)
        print(f"best: {self.best_params_}")
        return self

    def test(self, X_test, y_test):
        self._compute_metrics(y_test, self.pipeline_.predict(X_test))
        print(f"F1: {self.metrics_['f1']:.4f} | Acc: {self.metrics_['accuracy']:.4f} | Prec: {self.metrics_['precision']:.4f} | Rec: {self.metrics_['recall']:.4f}")
        return self

class CreditScoreTrainer:
    def __init__(self, experiment_name: str = "Credit_Score_Class Prediction", artifact_path: str = "artifacts"):
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_path)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)

    def run(self, data_path: str | Path) -> Tuple:
        print("Training")
        preprocessor = Preprocess()
        x_train, x_test, y_train_enc, y_test_enc = preprocessor.clean_and_split(data_path)

        col_transformer = preprocessor.build_transformer(x_train)
        cleaner = preprocessor.cleaner
        le_target = preprocessor.le_target

        preprocessor._y_train_enc = y_train_enc
        x_train_smote, y_train_smote = preprocessor.Smote(x_train, col_transformer)

        model_configs = [
            RandomForestModel1('Random Forest'),
            XGBoostModel2 ('XGBoost'),
            LightGBMModel3('LightGBM'),
        ]

        results = []
        for model in model_configs:
            model.train(x_train_smote, y_train_smote, cleaner, col_transformer)
            model.test(x_test, y_test_enc)
            results.append(model)

        best = max(results, key=lambda m: m.metrics_['f1'])
        print(f"Best model: {best.name} (F1: {best.metrics_['f1']:.4f}) (Recall: {best.metrics_['recall']:.4f}) (Precision: {best.metrics_['precision']:.4f}) (Accuracy: {best.metrics_['accuracy']:.4f})")

        with mlflow.start_run() as run:
            mlflow.log_param("best_model",best.name)
            mlflow.log_param("best_params", str(best.best_params_))
            mlflow.log_metric("accuracy", best.metrics_['accuracy'])
            mlflow.log_metric("precision",best.metrics_['precision'])
            mlflow.log_metric("recall",best.metrics_['recall'])
            mlflow.log_metric("f1_weighted", best.metrics_['f1'])

            artifacts = {
                'pipeline' : best.pipeline_,
                'le_target': le_target,
            }
            model_path = self.artifact_dir / "best_model.pkl"
            joblib.dump(artifacts, model_path)
            mlflow.sklearn.log_model(best.pipeline_, name="model")

            print(f"Artifacts saved → {model_path}")
            return run.info.run_id, x_test, y_test_enc, le_target