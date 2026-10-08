"""
Kaggle Schema Credit Card Fraud Dataset Synthetic Generator
Generates realistic 28 PCA features (V1-V28), Time, Amount, and Class target.
"""

import numpy as np
import pandas as pd
import os

def generate_synthetic_kaggle_data(sample_count=50000, fraud_ratio=0.00172, random_state=42):
    np.random.seed(random_state)
    fraud_count = int(sample_count * fraud_ratio)
    normal_count = sample_count - fraud_count

    # Generate normal transactions
    normal_data = {
        'Time': np.random.uniform(0, 172800, normal_count),
        'Amount': np.random.exponential(scale=88.3, size=normal_count),
        'Class': np.zeros(normal_count, dtype=int)
    }
    for i in range(1, 29):
        normal_data[f'V{i}'] = np.random.normal(0.0, 1.0, normal_count)

    # Inject fraud patterns (strong mean shifts in V14, V12, V10, V4, V11, V17)
    fraud_data = {
        'Time': np.random.uniform(0, 172800, fraud_count),
        'Amount': np.random.choice([np.random.normal(1.25, 0.5), np.random.normal(850.0, 300)], size=fraud_count),
        'Class': np.ones(fraud_count, dtype=int)
    }
    for i in range(1, 29):
        fraud_data[f'V{i}'] = np.random.normal(0.0, 1.0, fraud_count)
    
    # Specific PCA anomaly signatures based on empirical Kaggle observations
    fraud_data['V14'] = np.random.normal(-4.5, 1.2, fraud_count)
    fraud_data['V12'] = np.random.normal(-3.8, 1.1, fraud_count)
    fraud_data['V10'] = np.random.normal(-3.2, 1.0, fraud_count)
    fraud_data['V4']  = np.random.normal(3.6, 1.1, fraud_count)
    fraud_data['V11'] = np.random.normal(2.9, 1.0, fraud_count)
    fraud_data['V17'] = np.random.normal(-3.1, 1.2, fraud_count)
    fraud_data['V2']  = np.random.normal(2.4, 1.0, fraud_count)

    df_normal = pd.DataFrame(normal_data)
    df_fraud = pd.DataFrame(fraud_data)

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df['id'] = [f'TX-{i:06d}' for i in range(len(df))]

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_kaggle_data()
    output_path = "data/creditcard_synthetic.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated Synthetic Kaggle Schema Dataset successfully at {output_path}")
    print(f"Shape: {df.shape}")
    print("Class Value Counts:\n", df['Class'].value_counts())
    print("Class Ratio:\n", df['Class'].value_counts(normalize=True))
