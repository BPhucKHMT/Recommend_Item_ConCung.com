import polars as pl
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
import pickle 

# --- 1. SAMPLING FUNCTIONS (Đã sửa sang customer_id) ---

def positive_sampling(transaction_lf, q_val):
    """
    Lấy các cặp (Customer, Item) thực tế đã mua trong tập Validation -> Gán target = 1
    """
    print(">> Positive Sampling (Ground Truth)...")
    
    # Lọc dữ liệu Validation
    val_lf = transaction_lf.filter(pl.sql_expr(q_val))
    
    # [THAY ĐỔI] Lấy danh sách customer_id thay vì user_id
    val_cust_ids = val_lf.select("customer_id").unique().collect()["customer_id"].to_list()
    
    # Tạo bảng Positive
    pos_df = (
        val_lf
        .select(["customer_id", "item_id"]) # [THAY ĐỔI]
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int64).alias("target"))
        .collect()
    )
    
    return pos_df, val_cust_ids

def negative_sampling(transaction_lf, q_val, val_cust_ids, cfg):
    """
    Tạo các cặp (Customer, Item) user KHÔNG mua -> Gán target = 0.
    """
    print(">> Negative Sampling (Random popular items)...")
    
    n_neg = cfg["n_neg"]
    n_top = cfg["top_n"]
    
    # 1. Lấy Top Items (Trending) làm pool để random
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
    
    print(f"   -> Top items pool size: {len(top_items)}")
    
    # 2. Random Negative Samples
    neg_data = []
    if len(top_items) > 0:
        for cust_id in tqdm(val_cust_ids, desc="Generating Negatives"):
            sampled_items = random.sample(top_items, min(len(top_items), n_neg))
            for item_id in sampled_items:
                # [THAY ĐỔI] Lưu customer_id
                neg_data.append((cust_id, item_id, 0))
    
    # [THAY ĐỔI] Lấy schema của customer_id
    schema_map = transaction_lf.schema
    user_dtype = schema_map["customer_id"] # Int64
    item_dtype = schema_map["item_id"]
    
    neg_df = pl.DataFrame(
        neg_data, 
        schema={
            "customer_id": user_dtype,  # [THAY ĐỔI] Tên cột là customer_id
            "item_id": item_dtype, 
            "target": pl.Int64
        }, 
        orient="row"
    )
    
    return neg_df

# --- 2. TRAINING FUNCTION (Giữ nguyên logic, chỉ in log cho đẹp) ---

def train_model(df_train, feature_cols, model_name, cfg):
    """
    Huấn luyện model được chỉ định.
    """
    print(f">> Training Model: {model_name}...")
    
    # Chuyển qua Pandas/Numpy
    X = df_train.select(feature_cols).fill_null(0).to_pandas().values
    y = df_train.select("target").to_pandas().values.ravel()
    
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(alpha=cfg.get("alpha", 0.1)),
        "Logistic Regression": LogisticRegression(
            C=cfg.get("alpha", 1.0), 
            solver='liblinear', 
            class_weight='balanced'
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=20, 
            max_depth=5, 
            random_state=42,
            n_jobs=-1
        )
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} chưa hỗ trợ")
        
    model = models[model_name]
    model.fit(X, y)
    
    # In thông số model
    if hasattr(model, "coef_"):
        # In thử 5 hệ số đầu tiên để debug
        print(f"   Model Coef sample: {model.coef_.flatten()[:5]}")
        
    return model

# --- 3. PREDICTION FUNCTION (Giữ nguyên) ---

def predict(model, df_test, feature_cols):
    """
    Dự đoán điểm số.
    """
    print(">> Infering/Scoring...")
    X_test = df_test.select(feature_cols).fill_null(0).to_pandas().values
    
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.predict(X_test)
        
    return df_test.with_columns(pl.Series(name="pred_score", values=scores))