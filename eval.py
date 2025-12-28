
import polars as pl
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import gc
import os
import math

def evaluate(df_final, k=10, gt_path="data/final_groundtruth.pkl", history_path="data/table/purchase_data.parquet"):
    print(f"\n>>> EVALUATION (New Product Discovery Mode) @ K={k}")
    
    # --------------------------------------------------------------------------
    # BƯỚC 1: LOAD HISTORY (ĐỂ LỌC BỎ MÓN CŨ KHỎI GROUND TRUTH)
    # --------------------------------------------------------------------------
    print(">> Loading Training History (to filter re-purchases)...")
    hist_dict = {}
    try:
        # Load transaction log (chứa lịch sử train)
        # Chỉ lấy cột customer_id và item_id
        if os.path.exists(history_path):
            df_hist = pl.scan_parquet(history_path).select([
                pl.col("customer_id").cast(pl.String),
                pl.col("item_id").cast(pl.String).str.strip_chars()
            ])
            
            # Gom nhóm thành Dict {user: set(items)}
            # Dùng Polars groupby -> aggregation -> to_pandas -> zip để tạo dict cực nhanh
            # (Nhanh hơn iter_rows rất nhiều)
            hist_agg = (
                df_hist.group_by("customer_id")
                .agg(pl.col("item_id"))
                .collect()
            )
            
            # Convert to dict lookup
            # customer_id -> list[item_id]
            uids = hist_agg["customer_id"].to_list()
            items_list = hist_agg["item_id"].to_list()
            
            # Chuyển list thành set ngay lập tức để tra cứu O(1)
            hist_dict = {u: set(i) for u, i in zip(uids, items_list)}
            
            print(f"   -> Loaded History for {len(hist_dict)} users.")
            del df_hist, hist_agg, uids, items_list
            gc.collect()
        else:
            print(f"⚠️ Warning: History file {history_path} not found. Evaluation will act like standard Recall.")
    except Exception as e:
        print(f"❌ Error loading history: {e}")

    # --------------------------------------------------------------------------
    # BƯỚC 2: CHUẨN BỊ PREDICTIONS
    # --------------------------------------------------------------------------
    print(">> Aggregating predictions...")
    preds_lf = (
        df_final.lazy()
        .sort(["customer_id", "pred_score"], descending=[False, True])
        .group_by("customer_id")
        .agg(pl.col("item_id").cast(pl.String).head(k).alias("pred_items"))
    )
    
    # --------------------------------------------------------------------------
    # BƯỚC 3: LOAD GROUND TRUTH (FIXED & OPTIMIZED)
    # --------------------------------------------------------------------------
    print(">> Loading Ground Truth (Direct Pandas -> Polars)...")
    try:
        # Load trực tiếp file pickle (đang là pandas DataFrame)
        pd_gt = pd.read_pickle(gt_path)
        
        # Đổi tên cột 'item_id' trong GT thành 'gt_items' để tránh trùng tên khi join
        if "item_id" in pd_gt.columns:
            pd_gt.rename(columns={"item_id": "gt_items"}, inplace=True)
            
        # Convert thẳng sang Polars (cực nhanh, zero-copy nếu có thể)
        df_gt = pl.from_pandas(pd_gt)
        
        # Xóa biến pandas ngay lập tức để giải phóng RAM
        del pd_gt
        gc.collect()

    except Exception as e:
        print(f"❌ Error loading GT: {e}")
        return {}

    # Kiểm tra xem có dữ liệu không
    if df_gt.height == 0: return {}

    # Đồng bộ Schema ID với df_final (ép kiểu customer_id cho khớp)
    pred_schema_id = df_final.schema["customer_id"]
    if df_gt.schema["customer_id"] != pred_schema_id:
        df_gt = df_gt.with_columns(pl.col("customer_id").cast(pred_schema_id))
    # --------------------------------------------------------------------------
    # BƯỚC 4: JOIN DATA
    # --------------------------------------------------------------------------
    print(">> Joining Predictions with Ground Truth...")
    merged_df = preds_lf.join(df_gt.lazy(), on="customer_id", how="inner").collect()
    
    n_users_matched = merged_df.height
    if n_users_matched == 0:
        print("⚠️ Không có user nào trùng khớp!")
        return {"precision": 0, "recall": 0, "ndcg": 0}

    # --------------------------------------------------------------------------
    # BƯỚC 5: TÍNH METRICS (CÓ LỌC HISTORY)
    # --------------------------------------------------------------------------
    print("   -> Calculating metrics (Logic: Only evaluate on NEW items)...")
    
    total_precision = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    
    valid_users_count = 0 # Đếm số user thực sự mua món mới
    
    # Cache IDCG
    idcg_table = {}
    for n_rel in range(1, k + 1):
        idcg_table[n_rel] = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    
    # Cần lấy cả customer_id để tra cứu history
    rows_iter = merged_df.select(["customer_id", "pred_items", "gt_items"]).iter_rows()
    
    for uid, pred_items, gt_items in tqdm(rows_iter, total=n_users_matched, desc="Scoring"):
        if not gt_items: continue
        
        # 1. Lấy Ground Truth thô
        gt_set_raw = set(gt_items)
        
        # 2. Lấy History Items của user này
        # (Lưu ý: uid từ merged_df đang ở dạng đúng của pred_schema, cần cast về str để tra dict nếu dict key là str)
        hist_items = hist_dict.get(str(uid), set())
        
        # 3. [QUAN TRỌNG] Lọc bỏ món đã mua khỏi GT
        # Đây là tập hợp những món MỚI khách mua trong tương lai
        gt_set_new = gt_set_raw - hist_items
        
        # Nếu khách chỉ toàn mua lại đồ cũ -> Bỏ qua, không tính vào đánh giá
        # Vì model discovery không có nhiệm vụ đoán việc mua lại.
        if not gt_set_new: 
            continue
            
        valid_users_count += 1
            
        hits = 0
        dcg = 0.0
        
        # 4. Tính toán Metrics trên tập GT_NEW
        for i, item in enumerate(pred_items):
            # Item đoán đúng phải nằm trong tập MỚI
            if item in gt_set_new:
                hits += 1
                dcg += 1.0 / math.log2(i + 2)
        
        # Precision: Bao nhiêu % item gợi ý là đúng (và là món mới)
        total_precision += (hits / k)
        
        # Recall: Gợi ý được bao nhiêu % trong tổng số món MỚI khách đã mua
        # Mẫu số là len(gt_set_new), KHÔNG PHẢI len(gt_set_raw)
        total_recall += (hits / len(gt_set_new))
        
        # NDCG
        ideal_num = min(len(gt_set_new), k)
        if ideal_num > 0:
            idcg = idcg_table.get(ideal_num, 0.0)
            if idcg > 0:
                total_ndcg += (dcg / idcg)

    # Clean up
    del merged_df
    gc.collect()

    if valid_users_count == 0:
        print("⚠️ Warning: No users bought NEW items in the test set.")
        return {"precision": 0, "recall": 0, "ndcg": 0}

    avg_prec = total_precision / valid_users_count
    avg_recall = total_recall / valid_users_count
    avg_ndcg = total_ndcg / valid_users_count
    
    print("-" * 60)
    print(f"📊 DISCOVERY REPORT @ K={k}")
    print(f"   (Evaluated on {valid_users_count} users who bought NEW items)")
    print("-" * 60)
    print(f"   - Precision (New Items): {avg_prec:.4%}")
    print(f"   - Recall (New Items):    {avg_recall:.4%}")
    print(f"   - NDCG (New Items):      {avg_ndcg:.4%}")
    print("-" * 60)
    
    return {"precision": avg_prec, "recall": avg_recall, "ndcg": avg_ndcg}