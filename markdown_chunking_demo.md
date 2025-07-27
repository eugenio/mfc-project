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
