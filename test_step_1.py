from config import load_config
import candidate_version4 as gc_module 
import os
import polars as pl
import pickle
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def calculate_recall(candidates_df, groundtruth_path, history_lf, K=10):
    """
    Tính Recall@K tối ưu hóa tốc độ và định dạng dữ liệu.
    """
    print("\n" + "="*60)
    print("CALCULATING RECALL@K (Optimized)")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. PREPARE CANDIDATES (Polars Long -> Dict List)
    # ---------------------------------------------------------
    print(">> Preparing Candidates...")
    # Gom nhóm customer_id và tạo list item_id
    preds_agg = (
        candidates_df
        .select([
            pl.col("customer_id"),
            pl.col("item_id").cast(pl.String)
        ])
        .group_by("customer_id")
        .agg(pl.col("item_id"))
    )
    # Convert sang Dict {user_id: set(items)} để tra cứu O(1)
    pred_dict = {
        row[0]: set(row[1]) 
        for row in preds_agg.iter_rows()
    }
    print(f"   Candidates ready for {len(pred_dict):,} users.")

    # ---------------------------------------------------------
    # 2. PREPARE HISTORY (Polars Long -> Dict Set)
    # ---------------------------------------------------------
    print(">> Preparing History...")
    hist_agg = (
        history_lf
        .select([
            pl.col("customer_id"),
            pl.col("item_id").cast(pl.String)
        ])
        .group_by("customer_id")
        .agg(pl.col("item_id"))
        .collect() # Collect từ LazyFrame
    )
    hist_dict = {
        row[0]: set(row[1]) 
        for row in hist_agg.iter_rows()
    }
    print(f"   History ready for {len(hist_dict):,} users.")

    # ---------------------------------------------------------
    # 3. LOAD GROUND TRUTH (Pandas Wide -> Dict List)
    # ---------------------------------------------------------
    print(f">> Loading Groundtruth from {groundtruth_path}...")
    with open(groundtruth_path, 'rb') as f:
        gt_pandas = pickle.load(f)
    
    # Đảm bảo column name đúng
    if "item_id" not in gt_pandas.columns and "list_items" in gt_pandas.columns:
        gt_pandas = gt_pandas.rename(columns={"list_items": "item_id"})
    
    # Convert sang Dict {user_id: list(items)}
    # Lưu ý: GT cần giữ là List để loop cắt Top K, nhưng item bên trong phải là String
    gt_dict = dict(zip(
        gt_pandas["customer_id"].astype(str), 
        gt_pandas["item_id"].apply(lambda x: [str(i).strip() for i in x])
    ))
    print(f"   Groundtruth loaded for {len(gt_dict):,} users.")

    # ---------------------------------------------------------
    # 4. CALCULATE METRICS
    # ---------------------------------------------------------
    print(f"\n>> Computing Recall@{K}...")
    recalls = []
    ncold_start = 0
    evaluated_count = 0
    
    for user_id, gt_items in gt_dict.items():
        # Lấy lịch sử mua của user (nếu có)
        user_history = hist_dict.get(user_id, set())
        
        # 1. LỌC HISTORY KHỎI GROUND TRUTH
        # Chúng ta chỉ quan tâm những món user mua trong tương lai mà CHƯA từng mua trong quá khứ
        # (Tùy logic bài toán, nhưng thường Recall tính trên New Items)
        relevant_items = [item for item in gt_items if item not in user_history]
        
        # Lấy Top K items trong GT (Giả sử GT đã sort theo time hoặc quan trọng)
        # Nếu GT không có thứ tự, bước này lấy ngẫu nhiên K items đúng
        target_items = relevant_items[:K]
        
        if not target_items:
            continue # Không còn item nào để dự đoán (user chỉ mua lại đồ cũ)

        # 2. LẤY CANDIDATES
        # Nếu user không có trong candidates (do cold start hoặc bị lọc), recall = 0
        if user_id not in pred_dict:
            ncold_start += 1
            recalls.append(0.0)
            continue
            
        pred_items_set = pred_dict[user_id] # Tập candidates (100-200 items)
        
        # 3. TÍNH RECALL
        # Recall = (Số item dự đoán trúng) / (Tổng số item thực tế cần dự đoán)
        hits = sum(1 for item in target_items if item in pred_items_set)
        
        recall = hits / len(target_items)
        recalls.append(recall)
        evaluated_count += 1

    # ---------------------------------------------------------
    # 5. REPORT
    # ---------------------------------------------------------
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    print(f"\n>> Results:")
    print(f"   Total Users in GT:   {len(gt_dict):,}")
    print(f"   Users evaluated:     {evaluated_count:,}")
    print(f"   Users w/o Cands:     {ncold_start:,}")
    print(f"   Recall@{K}:          {mean_recall:.4f} ({mean_recall*100:.2f}%)")
    
    return mean_recall, {
        "recall": mean_recall,
        "K": K,
        "evaluated": evaluated_count
    }

# --- GLOBAL DECLARATION ---
root_dir = 'data/table/'

try:
    if os.path.exists(root_dir):
        item_lf = pl.scan_parquet(os.path.join(root_dir, 'item_data.parquet'))
        user_lf = pl.scan_parquet(os.path.join(root_dir, 'user_data.parquet'))
        transaction_lf = pl.scan_parquet(os.path.join(root_dir, 'purchase_data.parquet'))
    else:
        # Fallback to CSV for testing
        print("⚠️  Parquet not found, using CSV files...")
        item_lf = pl.scan_csv('item_1000.csv')
        user_lf = pl.scan_csv('user_1000.csv')
        transaction_lf = pl.scan_csv('purchase_1000.csv')
except Exception as e:
    print(f"❌ Error loading data: {e}")
    transaction_lf = None
    item_lf = None
    user_lf = None

def main():
    try:
        print(">>> START PIPELINE TEST - GET_CANDIDATES >>>")
        
        cfg = load_config("params.json")
        final_path = "final_submission.parquet" 
        
        df_final = None

        if os.path.exists(final_path):
            print(f"\n✅ FOUND EXISTING RESULT: {final_path}")
        
        if df_final is None:
            if transaction_lf is None: 
                raise ValueError("❌ Data error - cannot load transaction data")
            
            print("\n>> Processing data...")
            
            # Xử lý transaction data
            if "created_date" not in transaction_lf.columns:
                if "timestamp" in transaction_lf.columns:
                    # Nếu có timestamp, convert sang created_date
                    transaction_lf_processed = (
                        transaction_lf
                        .with_columns([
                            pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("created_date")
                        ])
                        .drop_nulls(subset=["customer_id"])
                    )
                else:
                    # Nếu có created_year, created_month, created_day
                    transaction_lf_processed = (
                        transaction_lf
                        .with_columns(pl.date("created_year", "created_month", "created_day").alias("created_date"))
                        .drop_nulls(subset=["customer_id"])
                    )
            else:
                transaction_lf_processed = transaction_lf.drop_nulls(subset=["customer_id"])
            
            # Tạo queries từ config
            queries = cfg.create_query_string()
            
            print(f"\n>> Query History: {queries['history']}")
            print(f">> Query Validation: {queries['val']}")
            
            # Tạo query cho toàn bộ lịch sử từ 2024-01-01 đến TRƯỚC validation (không leak)
            val_date = queries['val_raw_date']
            q_all_hist = f'"created_date" >= date(\'2024-01-01\') AND "created_date" <= date(\'{val_date}\')'
            print(f">> Query All History (for filtering): {q_all_hist}")
            
            stage2_cfg = {
                "n_neg": 10, 
                "top_n": 50,  # Test với 50 items thay vì 80
                "alpha": 0.001, 
                "model_name": "LightGBM", 
                "session_window": 1, 
                "min_coo": 1
            }

            # --- STAGE 1: CANDIDATES ---
            print("\n" + "="*60)
            print("STAGE 1: CANDIDATE GENERATION")
            print("="*60)
            
            save_path_s1 = "candidates_stage_21recall.parquet"
            
            if os.path.exists(save_path_s1):
                print(f"✅ Found existing candidates: {save_path_s1}")
                df_candidates = pl.read_parquet(save_path_s1)
            else:
                print(f">> Generating new candidates...")
                df_candidates = gc_module.get_candidates(
                    transaction_lf=transaction_lf_processed, 
                    item_lf=item_lf,
                    q_hist=queries['history'],  # 120 ngày để train
                    q_val=queries['val'],
                    q_all_hist=q_all_hist,  # Toàn bộ lịch sử từ 2024-01-01 để lọc
                    total_target=200,
                  #  filter_active_only=True

                )
                df_candidates.write_parquet(save_path_s1)
                print(f"✅ Saved candidates to: {save_path_s1}")
            
            # Hiển thị kết quả
            print(f"\n>> Candidates generated:")
            print(f"   Total rows: {df_candidates.shape[0]:,}")
            print(f"   Unique customers: {df_candidates['customer_id'].n_unique():,}")
            print(f"   Unique items: {df_candidates['item_id'].n_unique():,}")
            
            # Sample output
            print(f"\n>> Sample candidates (first 20 rows):")
            print(df_candidates.head(20))
            '''
            # Group by customer để check format
            print(f"\n>> Checking output format...")
            result_dict = {}
            for row in df_candidates.iter_rows(named=True):
                cust_id = row['customer_id']
                item_id = row['item_id']
                if cust_id not in result_dict:
                    result_dict[cust_id] = []
                result_dict[cust_id].append(item_id)
            
            # Show 5 customers
            print(f"\n>> Final format preview (5 customers):")
            for i, (cust, items) in enumerate(list(result_dict.items())[:5]):
                print(f"   Customer {cust}: {items[:10]}... (total: {len(items)} items)")
            
            print(f"\n✅ SUCCESS! Total customers with candidates: {len(result_dict):,}")
            '''
            # --- CALCULATE RECALL ---
            groundtruth_path = "data/final_groundtruth.pkl"
            if os.path.exists(groundtruth_path):
                print("\n" + "="*60)
                print("EVALUATING CANDIDATES WITH RECALL")
                print("="*60)
                
                recall_score, metrics = calculate_recall(
                    candidates_df=df_candidates,
                    groundtruth_path=groundtruth_path,
                    history_lf=transaction_lf_processed,
                    K=10
                )
                
                # Save metrics
                import json
                with open('recall_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n✅ Metrics saved to: recall_metrics.json")
            else:
                print(f"\n⚠️  Groundtruth file not found: {groundtruth_path}")
                print("   Skipping recall calculation...")


    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()