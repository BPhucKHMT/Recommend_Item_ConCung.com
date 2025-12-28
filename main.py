from config import load_config
import candidate_version4 as gc_module 
import features as feat 
import reranking as rank 
import eval as eval_module
import os
import polars as pl
import pandas as pd
import pickle
import json
from tqdm import tqdm
import shutil
import gc as python_gc
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- GLOBAL DECLARATION ---
root_dir = 'data/table/'

try:
    if os.path.exists(root_dir):
        item_lf = pl.scan_parquet(os.path.join(root_dir, 'item_data.parquet'))
        user_lf = pl.scan_parquet(os.path.join(root_dir, 'user_data.parquet'))
        transaction_lf = pl.scan_parquet(os.path.join(root_dir, 'purchase_data.parquet'))
    else:
        transaction_lf = None; item_lf = None
except Exception:
    transaction_lf = None; item_lf = None

def main():
    try:
        print(">>> START PIPELINE (High Precision Strategy: 5-6%) >>>")
        
        cfg = load_config("params.json")
        final_path = "final_submission.parquet" 
        model_path = "lgbm_model.pkl"
        
        df_final = None

        if os.path.exists(final_path):
            print(f"\n✅ FOUND EXISTING RESULT: {final_path}")
            # df_final = pl.read_parquet(final_path) # Uncomment nếu muốn resume từ file kết quả
        
        if df_final is None:
            if transaction_lf is None: raise ValueError("Data error.")
            
            print(">> Processing data...")
            transaction_lf_processed = (
                transaction_lf
                .with_columns(pl.date("created_year", "created_month", "created_day").alias("created_date"))
                .drop_nulls(subset=["customer_id"])
            )
            
            queries = cfg.create_query_string()
            
            stage2_cfg = {
                "n_neg": 15, "top_n": 100, "alpha": 0.001, 
                "model_name": "LightGBM", "session_window": 1, "min_coo": 1
            }

            # --- STAGE 1: CANDIDATES ---
            print("\n--- STAGE 1: CANDIDATE GENERATION ---")
            save_path_s1 = "candidates_stage1.parquet"
            if os.path.exists(save_path_s1):
                print(f"✅ Found existing candidates: {save_path_s1}")
                df_candidates = pl.read_parquet(save_path_s1)
            else:
                print(">> Generating new candidates (Top 500)...")
                df_candidates = gc_module.get_candidates(
                    transaction_lf_processed, 
                    q_hist=queries['history'], 
                    #top_n=stage2_cfg['top_n'],
                    total_target= 200,
                    q_val=queries['val']
                )
                df_candidates.write_parquet(save_path_s1)

            # --- STAGE 2: TRAINING ---
            print("\n--- STAGE 2: TRAINING MODEL ---")  
            # [UPDATED] Full feature set với temporal + query-level + Word2Vec features
            feature_cols = [
                # --- 1. POPULARITY & TREND (Quan trọng nhất) ---
                "item_pop_log",      # Độ hot toàn lịch sử
                "trend_7d",          # Độ hot tuần này (Bắt trend)
                "trend_30d",         # Độ hot tháng này
                "trend_momentum",    # Tốc độ tăng trưởng (7d / 30d)

                # --- 2. COLLABORATIVE FILTERING (Cốt lõi) ---
                "cooc_max",          # Món này có hay đi kèm món user từng mua không?
                "cooc_mean",         
                "w2v_sim_max",       # Vector tương đồng nhất (Semantic Match)
                "w2v_sim_mean",
                
                # --- 3. USER PERSONALIZATION (Cá nhân hóa) ---
                "brand_match",       # Đúng brand ruột không?
                "cat_l4_match",      # Đúng loại hàng đang tìm không?
                "price_ratio",       # Giá có hợp túi tiền user không?
                "price_zscore",      # [Mới] User thích hàng "Xịn" hay hàng "Rẻ" trong danh mục này?
                
                # --- 4. BEHAVIOR (Hành vi) ---
                "days_since_last_brand_purchase", # Lâu rồi chưa mua brand này?
                "days_since_last_cat3_purchase",  # Lâu rồi chưa mua loại này?
                "purchase_velocity",              # User mua sắm điên cuồng hay thong thả?
                "item_repurchase_rate",           # Món này người ta có mua lại không?
                
                # --- 5. QUERY CONTEXT ---
                "candidate_rank",    # Thứ hạng từ Stage 1 (Top 100 hay Top 500?)
            ]

            if os.path.exists(model_path):
                print(f"✅ Found existing model: {model_path}")
                with open(model_path, 'rb') as f: model = pickle.load(f)
            else:
                print(">> Starting Training...")
                pos_df, val_cust_ids = rank.positive_sampling(transaction_lf_processed, queries['val'])
                # neg_df = rank.negative_sampling(transaction_lf_processed, queries['val'], val_cust_ids, stage2_cfg)
                print("   -> Using Candidates for Hard Negative Mining...")
                
                # Chúng ta cần lọc df_candidates chỉ lấy những user có trong tập val (val_cust_ids)
                # để tiết kiệm RAM và thời gian tính toán
                target_candidates = df_candidates.filter(pl.col("customer_id").is_in(val_cust_ids))
                
                neg_df = rank.negative_sampling(target_candidates, pos_df, stage2_cfg)
                
                # [FIX SCHEMA ERROR] Ép kiểu neg_df theo pos_df trước khi nối
                # Lấy schema chuẩn từ pos_df
                target_user_type = pos_df.schema["customer_id"]
                target_item_type = pos_df.schema["item_id"]
                target_target_type = pos_df.schema["target"]
                
                neg_df = neg_df.select([
                    pl.col("customer_id").cast(target_user_type),
                    pl.col("item_id").cast(target_item_type),
                    pl.col("target").cast(target_target_type)
                ])
                df_train_raw = pl.concat([pos_df, neg_df])
                
                # Xóa cache train cũ để đảm bảo feature khớp
                if os.path.exists("temp_features_train"): shutil.rmtree("temp_features_train")
                
                df_train_features_lazy = feat.generate_features(
                    candidates_df=df_train_raw,
                    transaction_lf=transaction_lf_processed,
                    item_lf=item_lf,
                    queries=queries,
                    cfg=cfg,
                    feature_cols=feature_cols
                )
                
                print(">> Collecting Training Data...")
                df_train_features = df_train_features_lazy.collect()
                #if df_train_features.height > 500_000:
                    #df_train_features = df_train_features.sample(n=500_000, seed=42)
                
                model = rank.train_model(df_train_features, feature_cols, stage2_cfg["model_name"], stage2_cfg)
                
                # if hasattr(model, "feature_importances_"):
                #     print("\n📊 LIGHTGBM IMPORTANCE:")
                #     try:
                #         imp = pd.DataFrame({"Feature": feature_cols, "Value": model.feature_importances_})
                #         print(imp.sort_values(by="Value", ascending=False).to_string(index=False))
                #     except: pass
                #     print("="*40)

                with open(model_path, 'wb') as f: pickle.dump(model, f)
                del df_train_features, df_train_raw
                python_gc.collect()
                if os.path.exists("temp_features_train"): shutil.rmtree("temp_features_train")

            # --- STAGE 3: INFERENCE & HYBRID RERANKING ---
            print("\n--- STAGE 3: INFERENCE (HYBRID RERANKING) ---")
            print(">> Checking Inference files...")
            _ = feat.generate_features(
                candidates_df=df_candidates, 
                transaction_lf=transaction_lf_processed,
                item_lf=item_lf,
                queries=queries,
                cfg=cfg,
                model=model,           
                feature_cols=feature_cols 
            )
            
            print(">> Sorting with Hybrid Logic...")
            inference_dir = "temp_features_inference"
            top10_dir = "temp_top10_chunks"
            if os.path.exists(top10_dir): shutil.rmtree(top10_dir)
            os.makedirs(top10_dir)
            
            pred_files = [os.path.join(inference_dir, f) for f in os.listdir(inference_dir) if f.endswith('.parquet') and f != "_SUCCESS"]
            
            # [UPDATED] Đọc các cột cần thiết cho diversity reranking
            cols_needed = [
                "customer_id", "item_id", "pred_score", 
                "brand_match", "cat_l4_match", "item_pop_log", "price_ratio",
                # [NEW] Thêm candidate_rank để tính diversity
                "candidate_rank"
            ]
            
            SCORE_THRESHOLD = 0 # Ngưỡng cao để lọc rác

            for i, fpath in enumerate(tqdm(pred_files, desc="Hybrid Ranking")):
                try:
                    df_chunk = pl.read_parquet(fpath, columns=cols_needed)
                    
                    # [UPDATED] DIVERSITY-AWARE RERANKING
                    df_chunk_top = (
                        df_chunk
                        .with_columns([
                            # 1. Phạt giá: Lệch quá xa -> Giảm điểm
                            pl.when((pl.col("price_ratio") > 1.5) | (pl.col("price_ratio") < 0.5))
                              .then(pl.lit(0.6)) 
                              .otherwise(pl.lit(1.0))
                              .alias("price_penalty"),
                            
                            # 2. Thưởng Brand/Cat/Trend
                            pl.when((pl.col("brand_match") == 1) | (pl.col("cat_l4_match") == 1))
                              .then(pl.lit(1.3)) # Ưu tiên đúng gu
                              .when(pl.col("item_pop_log") > 3.0) 
                              .then(pl.lit(1.1)) # Ưu tiên hàng Hot
                              .otherwise(pl.lit(1.0))
                              .alias("relevance_boost"),
                            
                            # 3. [NEW] Diversity penalty (giảm điểm nếu nhiều items cùng brand)
                            # Tính cumsum của brand_match theo thứ tự candidate_rank
                            (pl.col("brand_match").cum_sum().over("customer_id") / (pl.col("candidate_rank") + 1))
                              .alias("brand_diversity_score")
                        ])
                        .with_columns([
                            # Final score: Base * Price Penalty * Relevance Boost * Diversity Factor
                            (pl.col("pred_score") 
                             * pl.col("price_penalty") 
                             * pl.col("relevance_boost")
                             * (1.0 - 0.08 * pl.col("brand_diversity_score"))  # Giảm 8% cho mỗi item trùng brand
                            ).alias("final_score")
                        ])
                        
                        # Lọc theo threshold
                        .filter(pl.col("pred_score") >= SCORE_THRESHOLD)
                        
                        .sort(["customer_id", "final_score"], descending=[False, True])
                        .group_by("customer_id")
                        .head(10)
                        
                        .select(["customer_id", "item_id", pl.col("final_score").alias("pred_score")])
                    )
                    
                    if df_chunk_top.height > 0:
                        df_chunk_top.write_parquet(f"{top10_dir}/top10_part_{i}.parquet")
                    del df_chunk, df_chunk_top
                except Exception as e:
                    print(f"⚠️ Error reading {fpath}: {e}")
            
            print(">> Final Merge...")
            try:
                df_final = pl.scan_parquet(f"{top10_dir}/*.parquet").sort(["customer_id", "pred_score"], descending=[False, True]).group_by("customer_id").head(10).collect()
            except: 
                df_final = pl.DataFrame({"customer_id": [], "item_id": [], "pred_score": []}, schema={"customer_id": pl.Int64, "item_id": pl.String, "pred_score": pl.Float32})

            if os.path.exists(top10_dir): shutil.rmtree(top10_dir)
            '''
            # --- STAGE 3.5: SAFE FALLBACK ---
            print("\n--- STAGE 3.5: FALLBACK (Global Trending) ---")
            
            if user_lf is not None: all_custs = user_lf.select("customer_id").unique()
            else: all_custs = transaction_lf_processed.select("customer_id").unique()

            user_counts = df_final.group_by("customer_id").len()
            
            users_short = (
                all_custs
                .join(user_counts.lazy(), on="customer_id", how="left")
                .filter(pl.col("len").fill_null(0) < 10)
                .select("customer_id")
            ).collect()
            
            missing_custs_list = users_short["customer_id"].to_list()
            print(f"   -> Found {len(missing_custs_list)} users needing Fallback.")

            if len(missing_custs_list) > 0:
                print(f"   -> Filling with Top 10 Global Trending...")
                global_pop = list(gc_module.get_trending_items(transaction_lf_processed, item_lf,queries['history'], queries['val'], n_trend=10))
                n_pop = len(global_pop)
                
                batch_size = 50000
                fallback_dfs = []
                target_schema = df_final.schema 
                
                for i in range(0, len(missing_custs_list), batch_size):
                    chunk_u = missing_custs_list[i : i + batch_size]
                    
                    chunk_df = pl.DataFrame({
                        "customer_id": np.repeat(chunk_u, n_pop),
                        "item_id": np.tile(global_pop, len(chunk_u)),
                        "pred_score": np.zeros(len(chunk_u)*n_pop, dtype=np.float32)
                    })
                    
                    chunk_df = chunk_df.select([
                        pl.col("customer_id").cast(target_schema["customer_id"]),
                        pl.col("item_id").cast(target_schema["item_id"]),
                        pl.col("pred_score").cast(target_schema["pred_score"])
                    ])
                    
                    fallback_dfs.append(chunk_df)
                
                if fallback_dfs:
                    print("   -> Merging fallback...")
                    full_fallback = pl.concat(fallback_dfs)
                    df_final = pl.concat([df_final, full_fallback]).unique(subset=["customer_id", "item_id"])
                    
                    df_final = (
                        df_final
                        .sort(["customer_id", "pred_score"], descending=[False, True])
                        .group_by("customer_id")
                        .head(10)
                    )

            df_final.write_parquet(final_path)
            print(f"✅ Saved final submission to: {final_path}")
        '''
        # --- STAGE 3.5: ADVANCED SEGMENT FALLBACK ---
            print("\n--- STAGE 3.5: FALLBACK (Segment-Based) ---")
            
            # Gọi hàm siêu cấp vũ trụ mới viết
            df_final = gc_module.get_advanced_fallback(
                df_final,       # Kết quả hiện tại (bị thiếu)
                user_lf,        # Bảng User (để lấy Gender, Region)
                transaction_lf_processed, # Bảng Transaction (để tính trend)
                item_lf,        # Bảng Item (để lọc active)
                queries['history'],
                queries['val'],
                n_fill=10
            )
            
            # Lưu kết quả
            df_final.write_parquet(final_path)
            print(f"✅ Saved final submission to: {final_path}")
        # --- STAGE 4: EXPORT & EVALUATION ---
        print("\n--- STAGE 4: EXPORT & EVALUATION ---")
        json_path = "result.json"
        
        print(">> Grouping items...")
        df_grouped = df_final.sort(["customer_id", "pred_score"], descending=[False, True]).group_by("customer_id", maintain_order=True).agg(pl.col("item_id"))
        
        print(f">> Streaming results to {json_path}...")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                f.write("{") 
                iterator = df_grouped.select(["customer_id", "item_id"]).iter_rows()
                total_rows = df_grouped.height
                first = True
                for cust_id, items in tqdm(iterator, total=total_rows, desc="Writing JSON"):
                    if not first: f.write(",") 
                    else: first = False
                    line = f'"{cust_id}":{json.dumps(list(items))}'
                    f.write(line)
                f.write("}") 
            print(f"✅ Exported JSON successfully.")
        except Exception as e:
            print(f"⚠️ Error writing JSON: {e}")

        gt_path = "data/final_groundtruth.pkl"
        if os.path.exists(gt_path):
            print(">> Starting Evaluation...")
            eval_module.evaluate(df_final, k=10, gt_path=gt_path)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()