import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def build_feature_matrix_simple(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal feature builder: MinMax scale numeric columns (adds *_scaled)
    and one-hot encode selected categorical columns. Returns X.
    """
    # --- config ---
    num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    cat_cols = ['Geography', 'Gender', 'NumOfProducts']  # add/remove as needed
    drop_cols_extra = ['id', 'CustomerId', 'Surname', 'Exited']  # optional identifiers/target to drop

    # copy to avoid side effects
    df = df_raw.copy(deep=True).reset_index(drop=True)

    # sanity check
    required = set(num_cols + cat_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- scale numerics (fit here; for inference reuse the trained scaler) ---
    scaler = MinMaxScaler()
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    scaled = scaler.fit_transform(df[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in num_cols], index=df.index)

    # --- one-hot encode categoricals ---
    ohe_df = pd.get_dummies(df[cat_cols], columns=cat_cols, drop_first=False)

    # --- assemble X ---
    X = pd.concat([scaled_df, ohe_df], axis=1)

    # If you also want to keep any other passthrough columns, add them here before dropping.
    # Finally, ensure we’re not returning source columns:
    # (not strictly necessary since we built X from scaled_df + ohe_df only)
    # X = X.drop(columns=[c for c in X.columns if c in num_cols + cat_cols], errors='ignore')

    return X


import pandas as pd

# --- one sample with all required columns ---
df_raw = pd.DataFrame([{
    "id": 1,
    "CustomerId": 15634602,
    "Surname": "Hargrave",
    "Geography": "France",
    "Gender": "Male",
    "Age": 45,
    "Tenure": 3,
    "Balance": 65000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 120000.0,
    "CreditScore": 600,
    "Exited": 0,
}])

# build features
X = build_feature_matrix_simple(df_raw)

print("X shape:", X.shape)
print("Columns:", list(X.columns))
print("Row values:", X.iloc[0].to_dict())
