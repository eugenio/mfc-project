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
