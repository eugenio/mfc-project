# Comprehensive Data Science Workflow Guide
## Data Collection and Preprocessing

Data collection forms the foundation of any successful data science project. Understanding your data sources and quality is crucial for downstream success.

### Data Sources

Modern data science projects typically integrate multiple data sources:

- **Structured databases**: SQL databases, data warehouses
- **Semi-structured data**: JSON, XML, API responses  
- **Unstructured data**: Text documents, images, audio files
- **Streaming data**: Real-time feeds, IoT sensors

### Data Quality Assessment

```python
import pandas as pd
import numpy as np

def assess_data_quality(df):
    """
    Comprehensive data quality assessment function
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Check for outliers using IQR method
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
    
    quality_report['outliers'] = outliers
    return quality_report
```

### Data Cleaning Strategies

Effective data cleaning requires systematic approaches tailored to your specific data challenges.

```python
def clean_dataset(df, strategy='comprehensive'):
    """
    Apply comprehensive data cleaning strategies
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in categorical_columns:
        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df
```
## Exploratory Data Analysis

EDA reveals patterns, relationships, and insights that guide modeling decisions and feature engineering.

### Statistical Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_eda(df, target_column=None):
    """
    Perform comprehensive exploratory data analysis
    """
    # Basic statistics
    print("Dataset Shape:", df.shape)
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Correlation analysis
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.show()
    
    # Distribution plots
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    n_cols = min(4, len(numeric_columns))
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(numeric_columns):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()
```

### Feature Relationships

Understanding relationships between features and the target variable guides feature selection and engineering.

| Analysis Type | Purpose | Method |
|---------------|---------|--------|
| Correlation | Linear relationships | Pearson correlation |
| Mutual Information | Non-linear dependencies | Information theory |
| Statistical Tests | Significance testing | Chi-square, ANOVA |
| Visualization | Pattern discovery | Scatter plots, boxplots |
## Feature Engineering

Feature engineering transforms raw data into meaningful representations that improve model performance.

### Automated Feature Engineering

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

class AutoFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
    
    def engineer_features(self, df, target_column, k_best=10):
        """
        Automated feature engineering pipeline
        """
        engineered_df = df.copy()
        
        # Separate features and target
        X = engineered_df.drop(columns=[target_column])
        y = engineered_df[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(X[[col]])
            self.scalers[col] = scaler
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        return X_selected, y
```

### Domain-Specific Features

Create features that capture domain knowledge and business logic specific to your problem.

```python
def create_time_features(df, datetime_column):
    """
    Extract comprehensive time-based features
    """
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Extract basic time components
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['day'] = df[datetime_column].dt.day
    df['hour'] = df[datetime_column].dt.hour
    df['day_of_week'] = df[datetime_column].dt.dayofweek
    df['day_of_year'] = df[datetime_column].dt.dayofyear
    df['week_of_year'] = df[datetime_column].dt.isocalendar().week
    
    # Create cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Create business logic features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    return df
```
## Model Development

Model development involves algorithm selection, hyperparameter tuning, and performance evaluation.

### Model Selection Strategy

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV

class ModelSelector:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42)
        }
        self.best_model = None
        self.best_score = 0
    
    def evaluate_models(self, X, y, cv=5):
        """
        Evaluate multiple models using cross-validation
        """
        results = {}
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            if scores.mean() > self.best_score:
                self.best_score = scores.mean()
                self.best_model = model
        
        return results
```

### Hyperparameter Optimization

```python
def optimize_hyperparameters(model, param_grid, X, y, cv=5):
    """
    Comprehensive hyperparameter optimization
    """
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }
```
## Model Evaluation and Validation

Rigorous evaluation ensures models generalize well to unseen data and meet business requirements.

### Comprehensive Evaluation Framework

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

def comprehensive_evaluation(model, X_test, y_test, X_train=None, y_train=None):
    """
    Comprehensive model evaluation with multiple metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Basic metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC AUC if probabilities available
    if y_pred_proba is not None:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC Score: {auc_score:.4f}")
    
    # Feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': range(len(model.feature_importances_)),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importances')
        plt.show()
```

### Cross-Validation Strategies

Different cross-validation strategies ensure robust model evaluation across various scenarios.

| Strategy | Use Case | Description |
|----------|----------|-------------|
| K-Fold | Standard evaluation | Splits data into k equal folds |
| Stratified K-Fold | Imbalanced datasets | Maintains class distribution |
| Time Series Split | Temporal data | Respects chronological order |
| Leave-One-Out | Small datasets | Uses single sample for validation |
## Production Deployment

Successful deployment requires careful consideration of infrastructure, monitoring, and maintenance.

### Model Serving Architecture

```python
import joblib
import flask
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
model = joblib.load('trained_model.pkl')
feature_engineer = joblib.load('feature_engineer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for model predictions
    """
    try:
        # Get input data
        data = request.json
        
        # Feature engineering
        processed_data = feature_engineer.transform(data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)
        
        return jsonify({
            'prediction': prediction.tolist(),
            'probability': probability.tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Monitoring and Maintenance

Production models require continuous monitoring to ensure performance remains optimal.

```python
import logging
import time
from datetime import datetime

class ModelMonitor:
    def __init__(self, model, baseline_metrics):
        self.model = model
        self.baseline_metrics = baseline_metrics
        self.prediction_logs = []
        
    def log_prediction(self, input_data, prediction, actual=None):
        """
        Log prediction for monitoring and analysis
        """
        log_entry = {
            'timestamp': datetime.now(),
            'input_data': input_data,
            'prediction': prediction,
            'actual': actual
        }
        self.prediction_logs.append(log_entry)
        
    def check_data_drift(self, recent_data, threshold=0.1):
        """
        Detect data drift in recent predictions
        """
        # Simple drift detection using statistical measures
        # In practice, use more sophisticated methods like KS test
        pass
        
    def generate_performance_report(self):
        """
        Generate comprehensive performance report
        """
        if not self.prediction_logs:
            return "No predictions logged yet"
        
        # Calculate recent performance metrics
        # Compare with baseline
        # Generate alerts if performance degrades
        pass
```
