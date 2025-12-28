import polars as pl
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
# [NEW] Import LightGBM
import lightgbm as lgb 
from sklearn.linear_model import Lasso
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
def negative_sampling(candidates_df, pos_df, cfg):
    """
    Fast & safe negative sampling (Polars)
    Lấy mẫu Negative từ chính tập Candidates (Hard Negatives)
    thay vì lấy ngẫu nhiên toàn sàn.
    """
    print(">> Negative Sampling (From Candidates)...")
    
    # 1. Chuẩn bị Positives (Chỉ cần 2 cột để Anti-Join)
    # Đảm bảo pos_df đã collect (là DataFrame)
    pos_keys = pos_df.select(["customer_id", "item_id"])

    # 2. Oversample từ Candidates
    # Lấy gấp 3 lần n_neg để sau khi trừ đi Positive vẫn còn đủ
    n_sample = cfg["n_neg"] * 3
    
    # [TỐI ƯU] Nếu candidates quá lớn, sample trước khi join để nhẹ RAM
    sampled_lf = (
        candidates_df.lazy()  # Xử lý Lazy để tối ưu
        .group_by("customer_id")
        .agg(
            pl.col("item_id").sample(
                n=n_sample, 
                with_replacement=True, # Cho phép lặp nếu candidate < n_sample
                shuffle=True
            )
        )
        .explode("item_id")
    )

    # 3. Loại bỏ những món KHÁCH ĐÃ MUA (Positives)
    # Anti-join: Giữ lại những món trong Candidates mà KHÔNG nằm trong Positives
    # Đây chính là những món "Model gợi ý nhưng Khách không mua" (Hard Negatives)
    neg_df = (
        sampled_lf.collect() # Collect về RAM để join
        .join(
            pos_keys, 
            on=["customer_id", "item_id"], 
            how="anti"
        )
    )

    # 4. Downsample về đúng số lượng n_neg cần thiết
    final_neg_df = (
        neg_df
        .group_by("customer_id")
        .agg(
            pl.col("item_id").sample(
                n=cfg["n_neg"], 
                with_replacement=False,
                shuffle=True
            )
        )
        .explode("item_id")
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("target")) # Gán nhãn 0
        .select(["customer_id", "item_id", "target"])
    )
    
    return final_neg_df



# def negative_sampling(transaction_lf, q_val, val_cust_ids, cfg):
#     print(">> Negative Sampling (Random opular items)...")
#     n_neg = cfg["n_neg"]
#     n_top = cfg["top_n"]
    
#     # Lấy Top Items
#     top_items = (
#         transaction_lf
#         .filter(pl.sql_expr(q_val))
#         .group_by("item_id")
#         .agg(pl.col("created_date").len().alias("cnt"))
#         .sort("cnt", descending=True)
#         .limit(n_top)
#         .select("item_id")
#         .collect()
#         ["item_id"].to_list()
#     )
    
#     neg_data = []
#     if len(top_items) > 0:
#         # Vector hóa việc random sampling để nhanh hơn
#         for cust_id in tqdm(val_cust_ids, desc="Generating Negatives"):
#             sampled_items = random.sample(top_items, min(len(top_items), n_neg))
#             for item_id in sampled_items:
#                 neg_data.append((cust_id, item_id, 0))
    
#     schema_map = transaction_lf.schema
#     user_dtype = schema_map["customer_id"]
#     item_dtype = schema_map["item_id"]
    
#     neg_df = pl.DataFrame(
#         neg_data, 
#         schema={"customer_id": user_dtype, "item_id": item_dtype, "target": pl.Int64}, 
#         orient="row"
#     )
#     return neg_df

# --- 2. PREDICTION EXPR (Dành cho LightGBM - Native Inference) ---
# LightGBM có thể dùng trees_to_dataframe hoặc predict lá, nhưng phức tạp để convert sang Polars Expr thuần.
# Nên ta sẽ tắt tính năng "Native Inference" và dùng Sklearn fallback (vẫn rất nhanh với LGBM).
def get_prediction_expr(model, feature_cols):
    return None # Tắt Native Polars mode, dùng model.predict()

# --- 3. TRAINING FUNCTION (Switch to LightGBM) ---
def train_model(df_train, feature_cols, model_name, cfg):
    print(f">> Training Model: {model_name} (LightGBM Ranker)...")

    print("   -> Converting Training Data to Numpy (Float32)...")
    df_train = df_train.sort("customer_id")
    try:
        X = df_train.select(feature_cols).fill_null(0).cast(pl.Float32).to_numpy()
        y = df_train.select("target").to_numpy().ravel()
    except:
        X = (
            df_train
            .select(feature_cols)
            .fill_null(0)
            .to_pandas()
            .values
            .astype(np.float32)
        )
        y = df_train.select("target").to_pandas().values.ravel()

    # ===== GROUP (RANKING QUAN TRỌNG NHẤT) =====
    print("   -> Building group (by customer_id)...")
    try:
        group = (
        df_train
        .group_by("customer_id", maintain_order=True)
        .len()
        ["len"]
        .to_numpy()
    )
    except:
        group = (
            df_train
            .to_pandas()
            .groupby("customer_id")
            .size()
            .values
        )

    print(f"   -> Total groups: {len(group)}")

    # ===== CONFIG LIGHTGBM RANKER =====
    print("   -> Configuring LightGBM Ranker...")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=1000,       # Tăng estimator lên vì data lớn (10M dòng)
        learning_rate=0.05,      # Learning rate vừa phải
        num_leaves=63,           # Tăng độ phức tạp cho cây
        max_depth=12,            
        min_child_samples=50,    # Tránh overfit với user ít data
        subsample=0.8,           # Row sampling
        colsample_bytree=0.8,    # Feature sampling
        random_state=42,
        n_jobs=-1,
        importance_type='gain',  # Dùng gain chuẩn hơn
        verbose=-1
    )

    print(f"   -> Fitting ranker on {X.shape} matrix...")
    model.fit(X, y, group=group)

    # =====================================================
    # 📌 BIAS (BASELINE SCORE – THAM CHIẾU)
    # =====================================================
    print("\n📌 MODEL BIAS (Baseline Score – Reference):")
    preds = model.predict(X)
    bias = float(np.mean(preds))
    print(f"   -> Mean prediction score: {bias:.6f}")

    # =====================================================
    # 📊 FEATURE IMPORTANCE
    # =====================================================
    print("\n📊 FEATURE IMPORTANCE (LightGBM Ranker):")

    booster = model.booster_
    importance = booster.feature_importance(importance_type="gain")

    imp_df = (
        pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importance
        })
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    imp_df["Importance_norm"] = (
        imp_df["Importance"] / imp_df["Importance"].sum()
    )

    print(imp_df.to_string(
        index=False,
        formatters={
            "Importance_norm": "{:.4f}".format
        }
    ))

    print("\n📌 GỢI Ý DEBUG:")
    print(" - Feature < 1% → DROP")
    print(" - Feature top quá mạnh → kiểm tra leakage")
    print(" - Bias tăng nhưng precision giảm → candidate bị loãng")

    print("=" * 60)

    return model
