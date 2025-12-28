import polars as pl
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
import pickle 

# --- 1. SAMPLING FUNCTIONS (Giữ nguyên) ---

def positive_sampling(transaction_lf, q_val):
    print(">> Positive Sampling (Ground Truth)...")
    val_lf = transaction_lf.filter(pl.sql_expr(q_val))
    val_cust_ids = val_lf.select("customer_id").unique().collect()["customer_id"].to_list()
    
    pos_df = (
        val_lf
        .select(["customer_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int64).alias("target"))
        .collect()
    )
    return pos_df, val_cust_ids

def negative_sampling(transaction_lf, q_val, val_cust_ids, cfg):
    print(">> Negative Sampling (Random popular items)...")
    n_neg = cfg["n_neg"]
    n_top = cfg["top_n"]
    
    top_items = (
        transaction_lf
        .filter(pl.sql_expr(q_val))
        .group_by("item_id")
        .agg(pl.col("created_date").len().alias("cnt"))
        .sort("cnt", descending=True)
        .limit(n_top)
        .select("item_id")
        .collect()
        ["item_id"].to_list()
    )
    
    neg_data = []
    if len(top_items) > 0:
        for cust_id in tqdm(val_cust_ids, desc="Generating Negatives"):
            sampled_items = random.sample(top_items, min(len(top_items), n_neg))
            for item_id in sampled_items:
                neg_data.append((cust_id, item_id, 0))
    
    schema_map = transaction_lf.schema
    user_dtype = schema_map["customer_id"]
    item_dtype = schema_map["item_id"]
    
    neg_df = pl.DataFrame(
        neg_data, 
        schema={"customer_id": user_dtype, "item_id": item_dtype, "target": pl.Int64}, 
        orient="row"
    )
    return neg_df

def get_prediction_expr(model, feature_cols):
    """
    Chuyển trọng số Model Sklearn thành biểu thức Polars.
    Giúp tính điểm trực tiếp trong Polars mà không cần convert sang Numpy.
    """
    # 1. Lấy Bias (Intercept)
    if isinstance(model.intercept_, (list, np.ndarray)) or (hasattr(model.intercept_, 'ndim') and model.intercept_.ndim > 0):
        bias = model.intercept_[0]
    else:
        bias = model.intercept_
    
    # 2. Lấy Weights (Coefficients)
    if hasattr(model.coef_, 'ndim') and model.coef_.ndim > 1:
        weights = model.coef_[0]
    else:
        weights = model.coef_
    
    # 3. Xây dựng biểu thức: Score = Bias + (w1*col1) + (w2*col2) + ...
    score_expr = pl.lit(bias)
    
    for col_name, w in zip(feature_cols, weights):
        # Chỉ cộng những feature có trọng số đáng kể (Tối ưu cho Lasso)
        if abs(w) > 1e-9: 
            score_expr = score_expr + (pl.col(col_name) * w)
            
    # 4. Xử lý hàm kích hoạt (Activation Function)
    # Kiểm tra xem là Logistic (Classification) hay Lasso (Regression)
    is_logistic = hasattr(model, "classes_")
    
    if is_logistic:
        # Sigmoid: 1 / (1 + e^-x)
        score_expr = 1.0 / (1.0 + (-score_expr).exp())
    else:
        # Lasso: Clip về [0, 1] để không bị lỗi hiển thị
        score_expr = score_expr.clip(0.0, 1.0)
        
    return score_expr.alias("pred_score")
# --- 2. TRAINING FUNCTION ---

def train_model(df_train, feature_cols, model_name, cfg):
    print(f">> Training Model: {model_name}...")
    
    # [TỐI ƯU RAM QUAN TRỌNG]
    # 1. Chuyển trực tiếp từ Polars -> Numpy (không qua Pandas)
    # 2. Ép kiểu về float32 (Giảm 50% RAM so với float64 mặc định)
    print("   -> Converting Training Data to Numpy (Float32)...")
    
    try:
        # Lấy X: Fill null = 0, ép kiểu Float32
        X = df_train.select(feature_cols).fill_null(0).cast(pl.Float32).to_numpy()
        
        # Lấy y:
        y = df_train.select("target").to_numpy().ravel()
        
    except Exception as e:
        print(f"⚠️ Error converting data: {e}")
        # Fallback nếu cách trên lỗi (nhưng thường sẽ chậm hơn)
        X = df_train.select(feature_cols).fill_null(0).to_pandas().values.astype(np.float32)
        y = df_train.select("target").to_pandas().values.ravel()

    models = {
        "Linear Regression": LinearRegression(),
        # Lasso cần alpha nhỏ (ví dụ 0.01) vì nó phạt rất mạnh (L1 regularization)
        "Lasso": Lasso(alpha=cfg.get("alpha", 0.01)), 
        "Logistic Regression": LogisticRegression(
            C=cfg.get("alpha", 1.0), 
            solver='liblinear', 
            class_weight='balanced'
        ),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=20, max_depth=5, n_jobs=-1)
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} chưa hỗ trợ")
        
    model = models[model_name]
    
    print(f"   -> Fitting {model_name} on {X.shape} matrix...")
    model.fit(X, y)
    
    if hasattr(model, "coef_"):
        print(f"   Model Coef sample: {model.coef_.flatten()[:5]}")
        zero_feats = np.sum(model.coef_ == 0)
        print(f"   (Lasso Info) Số lượng Feature bị ép về 0: {zero_feats} / {len(feature_cols)}")
        
    return model

# --- 3. PREDICTION FUNCTION (Cập nhật Clip Score) ---

def predict(model, df_test, feature_cols):
    print(">> Infering/Scoring...")
    X_test = df_test.select(feature_cols).fill_null(0).to_pandas().values
    
    if hasattr(model, "predict_proba"):
        # Classification (Logistic, Random Forest) -> Trả về xác suất [0, 1]
        scores = model.predict_proba(X_test)[:, 1]
    else:
        # Regression (Lasso, Linear) -> Trả về giá trị thực (-inf, +inf)
        scores = model.predict(X_test)
        
        # [QUAN TRỌNG] Clip score về [0, 1] để App Streamlit không bị lỗi hiển thị Progress Bar
        # Lasso có thể dự đoán > 1 hoặc < 0
        scores = np.clip(scores, 0.0, 1.0)
        
    return df_test.with_columns(pl.Series(name="pred_score", values=scores))