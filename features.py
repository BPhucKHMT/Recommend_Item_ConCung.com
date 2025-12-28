import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import gc
import os
import shutil
import datetime
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Thử import logic tối ưu từ reranking_lasso
try:
    from reranking import get_prediction_expr
except ImportError:
    get_prediction_expr = None

warnings.filterwarnings('ignore')

# --- 1. PREPARE LOOKUP TABLES (FULL + TIME FEATURES) ---
def prepare_lookup_tables(transaction_lf, item_lf, q_hist, cfg):
    print(">> Preparing Lookup Tables (Vectorized)...")

    print(">> Preparing Lookup Tables (Advanced)...")
    
    # 1. Base Filter
    hist_lf = transaction_lf.filter(pl.sql_expr(q_hist))
    
    # Xác định ngày cuối cùng trong tập train (Anchor Date)
    anchor_date = hist_lf.select(pl.col("created_date").max()).collect().item()
    print(f"   -> Anchor Date for Trends: {anchor_date}")

    # =========================================================
    # 🔥 [NEW] 1. TREND FEATURES (7D, 30D)
    # =========================================================
    print("   -> Computing Short-term Trends...")
    # Lọc transaction trong 30 ngày cuối
    trend_lf = hist_lf.filter(pl.col("created_date") >= (anchor_date - datetime.timedelta(days=30)))
    
    item_trends = (
        trend_lf
        .group_by("item_id")
        .agg([
            pl.col("created_date").filter(pl.col("created_date") >= (anchor_date - datetime.timedelta(days=7))).len().alias("trend_7d"),
            pl.col("created_date").len().alias("trend_30d")
        ])
        .collect()
    )

    # =========================================================
    # 🔥 [NEW] 2. ITEM RELATIVE PRICE (Z-Score in Category)
    # =========================================================
    print("   -> Computing Price Z-Score per Category...")
    df_price_stats = None
    if item_lf is not None:
        item_schema = item_lf.collect_schema().names()
        price_col = "price" if "price" in item_schema else "current_price"
        
        # Tính Mean và Std của giá trong từng Category L3
        cat_price_stats = (
            item_lf
            .group_by("category_l3")
            .agg([
                pl.col(price_col).mean().alias("cat_avg_price"),
                pl.col(price_col).std().fill_null(1).alias("cat_std_price")
            ])
        )
        
        # Join ngược lại để tính Z-Score cho từng Item
        df_price_stats = (
            item_lf
            .join(cat_price_stats, on="category_l3")
            .select([
                pl.col("item_id"),
                ((pl.col(price_col) - pl.col("cat_avg_price")) / (pl.col("cat_std_price") + 0.01)).alias("price_zscore")
            ])
            .collect()
        )

    # =========================================================
    # 🔥 [NEW] 3. REPURCHASE RATE
    # =========================================================
    print("   -> Computing Item Repurchase Rate...")
    item_repurchase = (
        hist_lf
        .group_by(["item_id", "customer_id"])
        .len() # Đếm số lần user mua item này
        .filter(pl.col("len") > 1) # Chỉ lấy những lần mua lại
        .group_by("item_id")
        .len().alias("repurchase_user_count") # Số user đã mua lại
        .join(
            hist_lf.group_by("item_id").n_unique("customer_id").alias("total_unique_users"),
            on="item_id"
        )
        .select([
            pl.col("item_id"),
            (pl.col("repurchase_user_count") / pl.col("total_unique_users")).alias("item_repurchase_rate")
        ])
        .collect()
    )
    
    # 1. Filter Data
    hist_lf = (
        transaction_lf
        .filter(pl.sql_expr(q_hist))
        .select(["customer_id", "item_id", "created_date"])
        .unique()
    )
    
    # [FEATURE 1] Global Popularity
    print("   -> Computing Global Item Popularity...")
    item_stats = hist_lf.group_by("item_id").agg(pl.len().alias("global_item_count"))
    df_item_stats = item_stats.collect()

    # 2. Lọc Spammer
    user_counts = hist_lf.group_by(["customer_id", "created_date"]).len().filter(pl.col("len") < 30)
    hist_clean = hist_lf.join(user_counts.select(["customer_id", "created_date"]), on=["customer_id", "created_date"], how="inner")

    # [FEATURE 2] User Spending Power
    print("   -> Computing User Spending Power...")
    user_spending = None
    item_prices = None
    if item_lf is not None:
        item_schema = item_lf.collect_schema().names()
        price_col = "price" if "price" in item_schema else ("current_price" if "current_price" in item_schema else None)
        if price_col:
            item_prices = item_lf.select([pl.col("item_id"), pl.col(price_col).cast(pl.Float32).alias("price")]).collect()
            user_spending = (
                hist_clean.join(item_prices.lazy(), on="item_id")
                .group_by("customer_id")
                .agg([pl.col("price").mean().alias("user_avg_spend")])
                .collect()
            )

    # [FEATURE 3 - MỚI] CATEGORY TIME CYCLES (Chu kỳ mua sắm)
    # Tính toán xem lần cuối user mua Category L3 này là bao lâu
    print("   -> Computing Category Time Cycles (RAM Safe)...")
    cat_time_stats = None
    if item_lf is not None:
        # Chỉ lấy cột cần thiết để nhẹ RAM
        item_cats = item_lf.select(["item_id", "category_l3"]).unique().collect()
        
        # Join transaction với category
        # Lưu ý: GroupBy theo (User, Cat L3) sẽ nhỏ hơn rất nhiều so với (User, Item)
        cat_time_stats = (
            hist_clean
            .join(item_cats.lazy(), on="item_id", how="inner")
            .group_by(["customer_id", "category_l3"])
            .agg([
                pl.col("created_date").max().alias("last_cat3_date"), # Ngày mua cuối
                pl.len().alias("user_cat3_buy_count") # Số lần mua Cat này
            ])
            .collect()
        )

    # 3. Item Co-occurrence
    print("   -> Computing Item-Item co-occurrence...")
    co_purchase = (
        hist_clean.lazy()
        .join(hist_clean.lazy(), on=["customer_id", "created_date"], suffix="_right")
        .filter(pl.col("item_id") != pl.col("item_id_right"))
        .group_by(["item_id", "item_id_right"])
        .agg(pl.len().alias("cooc_score"))
        .filter(pl.col("cooc_score") >= cfg.min_coo)
    )
    df_cooc = co_purchase.collect()
    df_cooc_rev = df_cooc.select([pl.col("item_id_right").alias("item_id"), pl.col("item_id").alias("item_id_right"), pl.col("cooc_score")])
    df_cooc = pl.concat([df_cooc, df_cooc_rev]).unique(subset=["item_id", "item_id_right"])

    # 4. Brand Co-occurrence
    print("   -> Computing Brand-Brand co-occurrence...")
    df_brand_cooc = None
    if item_lf is not None:
        item_brands = item_lf.select(["item_id", "brand"]).unique()
        hist_with_brand = hist_clean.join(item_brands, on="item_id")
        user_brand_unique = hist_with_brand.select(["customer_id", "brand"]).unique()
        
        brand_cooc_lazy = (
            user_brand_unique.lazy()
            .join(user_brand_unique.lazy(), on="customer_id", suffix="_right")
            .filter(pl.col("brand") != pl.col("brand_right"))
            .group_by(["brand", "brand_right"])
            .agg(pl.len().alias("brand_cooc_score"))
            .filter(pl.col("brand_cooc_score") >= 5)
        )
        df_brand_cooc = brand_cooc_lazy.collect()
        df_brand_cooc_rev = df_brand_cooc.select([pl.col("brand_right").alias("brand"), pl.col("brand").alias("brand_right"), pl.col("brand_cooc_score")])
        df_brand_cooc = pl.concat([df_brand_cooc, df_brand_cooc_rev]).unique(subset=["brand", "brand_right"])

    # 5. User History
    print("   -> Preparing user history...")
    df_hist_long = hist_clean.sort("created_date", descending=True).group_by("customer_id").head(30).select(["customer_id", "item_id"]).rename({"item_id": "hist_item_id"}).collect()

    # 6. User Profile
    print("   -> Preparing user profile summaries...")
    if item_lf is not None:
        cols_needed = ["item_id", "brand", "category_l3", "category"]
        available_cols = item_lf.collect_schema().names()
        selected_cols = [c for c in cols_needed if c in available_cols]
        item_small = item_lf.select(selected_cols)
        if "category" not in available_cols: item_small = item_small.with_columns(pl.lit("unknown").alias("category"))
        data_with_info = hist_clean.join(item_small, on="item_id", how="left")
    else:
        data_with_info = hist_clean.with_columns([pl.lit("u").alias("brand"), pl.lit("u").alias("category_l3"), pl.lit("u").alias("category")])

    df_user_profile = data_with_info.group_by("customer_id").agg([
        pl.col("brand").drop_nulls().unique().alias("hist_brands"),
        pl.col("category_l3").drop_nulls().unique().alias("hist_cats_l3"),
        pl.col("category").drop_nulls().unique().alias("hist_cats_l4")
    ]).collect()
    
    if item_lf is not None:
        df_hist_brands_long = data_with_info.select(["customer_id", "brand"]).unique().rename({"brand": "hist_brand_right"}).collect()
    else:
        df_hist_brands_long = None

    # 7. Brand Loyalty
    print("   -> Preparing Brand Loyalty...")
    if item_lf is not None:
        brand_info = item_lf.select(["item_id", "brand"])
        df_brand_loyalty = hist_clean.join(brand_info, on="item_id", how="left").group_by(["customer_id", "brand"]).agg([
            pl.len().alias("user_brand_count"),
            pl.col("created_date").max().alias("last_brand_purchase_date")
        ]).collect()
    else:
        df_brand_loyalty = None

    anchor_date = hist_clean.select(pl.col("created_date").max()).collect().item()
    
    # 8. Item Info Metadata
    df_item = None
    if item_lf is not None:
        df_item = item_lf.select(selected_cols).collect()
        if "category" not in df_item.columns: df_item = df_item.with_columns(pl.lit("unknown").alias("category"))

    # [NEW] 9. Purchase Velocity (TÍNH TRƯỚC temporal features)
    print("   -> Computing Purchase Velocity...")
    user_velocity_collected = None
    if item_lf is not None:
        # Cần category_l3 để tính velocity
        item_cats = item_lf.select(["item_id", "category_l3"]).unique().collect()
        
        user_velocity = (
            hist_clean
            .join(item_cats.lazy(), on="item_id", how="inner")
            .group_by(["customer_id", "category_l3"])
            .agg([
                pl.col("created_date").min().alias("first_purchase"),
                pl.col("created_date").max().alias("last_purchase"),
                pl.count().alias("total_purchases")
            ])
            .with_columns([
                ((pl.col("last_purchase") - pl.col("first_purchase")).dt.total_days() + 1).alias("active_days"),
                (pl.col("total_purchases") / ((pl.col("last_purchase") - pl.col("first_purchase")).dt.total_days() + 1))
                    .alias("purchase_velocity")
            ])
            .select(["customer_id", "category_l3", "purchase_velocity"])
        )
        user_velocity_collected = user_velocity.collect()
    
    # [NEW] 10. Word2Vec Item Embeddings
    print("   -> Training Word2Vec item embeddings...")
    w2v_model = None
    user_purchase_seqs = None
    try:
        # Get user purchase sequences ordered by time
        user_seqs_df = (
            hist_clean
            .sort(["customer_id", "created_date"])
            .group_by("customer_id")
            .agg(pl.col("item_id").alias("items"))
            .collect()
        )
        
        # Convert to list of sequences (each item as string)
        sequences = [[str(item) for item in seq] for seq in user_seqs_df["items"].to_list()]
        
        # Train Word2Vec (vector_size=64, window=10, min_count=2)
        w2v_model = Word2Vec(sentences=sequences, vector_size=64, window=10, min_count=2, 
                            workers=8, sg=1, epochs=5)
        print(f"      -> Trained embeddings for {len(w2v_model.wv)} items")
        
        # Store user purchase sequences for similarity calculation
        user_purchase_seqs = user_seqs_df.to_pandas().set_index('customer_id')['items'].to_dict()
        
    except Exception as e:
        print(f"      -> Warning: Word2Vec training failed: {e}")
    
    return df_cooc, df_brand_cooc, df_hist_long, df_hist_brands_long, df_user_profile, \
           df_item_stats, df_brand_loyalty, anchor_date, df_item, user_spending, item_prices, cat_time_stats, \
           user_velocity_collected, w2v_model, user_purchase_seqs , item_trends, df_price_stats, item_repurchase

# --- 2. VECTORIZED GENERATION ---
def generate_features(candidates_df, transaction_lf, item_lf, queries, cfg, model=None, feature_cols=None):
    print(f"   [Input] Candidates Size: {candidates_df.height} rows")
    
    mode_name = "inference" if model is not None else "train"
    temp_dir = f"temp_features_{mode_name}"
    success_flag = os.path.join(temp_dir, "_SUCCESS")
    
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    else:
        if os.path.exists(success_flag):
            print(f"   ✅ Found completion flag (_SUCCESS). Skipping generation step!")
            return pl.scan_parquet(f"{temp_dir}/part_*.parquet")
        print(f"   [Info] Directory {temp_dir} exists (but no success flag). Resuming...")
    
    if not hasattr(cfg, 'q_recent'): cfg.q_recent = queries['recent']
    
    # Unpack ALL tables (New: cat_time_stats, user_velocity, w2v_model, user_purchase_seqs)
    df_cooc, df_brand_cooc, df_hist_long, df_hist_brands_long, df_user_profile, \
    df_item_stats, df_brand_loyalty, anchor_date, df_item, user_spending, item_prices, cat_time_stats, \
    user_velocity, w2v_model, user_purchase_seqs , item_trends, df_price_stats, item_repurchase  = prepare_lookup_tables(transaction_lf, item_lf, queries['history'], cfg)

    batch_size = 200000 
    total_rows = candidates_df.height
    pbar = tqdm(total=total_rows, desc=f"Vectorized {mode_name}")
    cand_lazy = candidates_df.lazy()

    pred_expr = None
    if model is not None and get_prediction_expr is not None:
        try:
            pred_expr = get_prediction_expr(model, feature_cols)
        except: pass

    for offset in range(0, total_rows, batch_size):
        file_name = f"{temp_dir}/part_{offset}.parquet"
        if os.path.exists(file_name):
            remaining = total_rows - offset
            step = batch_size if remaining > batch_size else remaining
            pbar.update(step)
            continue 

        chunk = cand_lazy.slice(offset, batch_size).collect()
        
        # 1. Metadata Join
        if df_item is not None:
            chunk_final = chunk.join(df_item, on="item_id", how="left")
        else:
            chunk_final = chunk.with_columns([pl.lit("u").alias("brand"), pl.lit("u").alias("category_l3"), pl.lit("u").alias("category")])

            # [NEW JOIN] 1. Trends
        chunk_final = chunk_final.join(item_trends, on="item_id", how="left")
        
        # [NEW JOIN] 2. Price Z-Score
        if df_price_stats is not None:
            chunk_final = chunk_final.join(df_price_stats, on="item_id", how="left")
        
        # [NEW JOIN] 3. Repurchase Rate
        chunk_final = chunk_final.join(item_repurchase, on="item_id", how="left")
            # 2. Co-occurrence
        chunk_cooc = chunk.join(df_hist_long, on="customer_id", how="left")
        chunk_cooc = chunk_cooc.join(df_cooc, left_on=["item_id", "hist_item_id"], right_on=["item_id", "item_id_right"], how="left")
        feat_cooc = chunk_cooc.group_by(["customer_id", "item_id"]).agg([
            (pl.col("cooc_score").max().fill_null(0) + 1).log10().alias("cooc_max"),
            pl.col("cooc_score").mean().fill_null(0).alias("cooc_mean"),
            pl.col("cooc_score").sum().fill_null(0).alias("cooc_sum"),
            pl.col("cooc_score").drop_nulls().len().alias("cooc_len")
        ])
        chunk_final = chunk_final.join(feat_cooc, on=["customer_id", "item_id"], how="left")

        # 3. Brand Cross-sell
        if df_brand_cooc is not None and df_hist_brands_long is not None:
            chunk_brand_cross = chunk_final.select(["customer_id", "item_id", "brand"]).join(df_hist_brands_long, on="customer_id", how="left")
            chunk_brand_cross = chunk_brand_cross.join(df_brand_cooc, left_on=["brand", "hist_brand_right"], right_on=["brand", "brand_right"], how="left")
            feat_brand_cross = chunk_brand_cross.group_by(["customer_id", "item_id"]).agg([
                pl.col("brand_cooc_score").sum().fill_null(0).alias("brand_cross_score"), 
                pl.col("brand_cooc_score").max().fill_null(0).alias("brand_cross_max")    
            ])
            chunk_final = chunk_final.join(feat_brand_cross, on=["customer_id", "item_id"], how="left")
        else:
            chunk_final = chunk_final.with_columns([pl.lit(0).alias("brand_cross_score"), pl.lit(0).alias("brand_cross_max")])

        # 4. Stats, Profile, Popularity
        chunk_final = chunk_final.join(df_user_profile, on="customer_id", how="left")
        
        if df_item_stats is not None:
            chunk_final = chunk_final.join(df_item_stats, on="item_id", how="left")
        else:
            chunk_final = chunk_final.with_columns(pl.lit(0).alias("global_item_count"))

        if df_brand_loyalty is not None:
            chunk_final = chunk_final.join(df_brand_loyalty, on=["customer_id", "brand"], how="left")
        else:
            chunk_final = chunk_final.with_columns([pl.lit(0).alias("user_brand_count"), pl.lit(None).alias("last_brand_purchase_date")])

        if user_spending is not None and item_prices is not None:
            chunk_final = chunk_final.join(user_spending, on="customer_id", how="left")
            if "price" not in chunk_final.columns:
                chunk_final = chunk_final.join(item_prices, on="item_id", how="left")
        else:
            chunk_final = chunk_final.with_columns([pl.lit(0).alias("user_avg_spend"), pl.lit(0).alias("price")])

        # [NEW] 5. Time Features (Category Cycles)
        if cat_time_stats is not None:
            # Join theo (User, Category L3) - An toàn về RAM
            chunk_final = chunk_final.join(cat_time_stats, on=["customer_id", "category_l3"], how="left")
        else:
            chunk_final = chunk_final.with_columns([pl.lit(None).alias("last_cat3_date"), pl.lit(0).alias("user_cat3_buy_count")])
        
        # [NEW] 6. Purchase Velocity
        if user_velocity is not None:
            chunk_final = chunk_final.join(user_velocity, on=["customer_id", "category_l3"], how="left")
            chunk_final = chunk_final.with_columns([
                pl.col("purchase_velocity").fill_null(0).alias("purchase_velocity")
            ])
        else:
            chunk_final = chunk_final.with_columns([pl.lit(0.0).alias("purchase_velocity")])

        # 6. Calc Features
        chunk_final = chunk_final.with_columns([
            pl.col("cooc_len").fill_null(0),
            pl.col("user_brand_count").fill_null(0),
            pl.col("brand_cross_score").fill_null(0),
            pl.col("brand_cross_max").fill_null(0),
            pl.col("global_item_count").fill_null(0),
            pl.col("price").fill_null(0),
            pl.col("user_avg_spend").fill_null(0),
            
            (pl.col("global_item_count") + 1).log10().alias("item_pop_log"),
            (pl.col("price") - pl.col("user_avg_spend")).abs().alias("price_diff_abs"),
            (pl.col("price") / (pl.col("user_avg_spend") + 1)).alias("price_ratio"),

            (anchor_date - pl.col("last_brand_purchase_date")).dt.total_days().fill_null(999).alias("days_since_last_brand_purchase"),
            
            # [NEW] Days since cat purchase & Freq
            (anchor_date - pl.col("last_cat3_date")).dt.total_days().fill_null(999).alias("days_since_last_cat3_purchase"),
            (pl.col("user_cat3_buy_count").fill_null(0) + 1).log10().alias("user_cat3_freq_log"),

            pl.when(pl.col("brand").is_in(pl.col("hist_brands"))).then(1).otherwise(0).alias("brand_match"),
            pl.when(pl.col("category_l3").is_in(pl.col("hist_cats_l3"))).then(1).otherwise(0).alias("cat_l3_match"),
            pl.when(pl.col("category").is_in(pl.col("hist_cats_l4"))).then(1).otherwise(0).alias("cat_l4_match")
        ])
        
        # [NEW] 7. Query-level Features (for LightGBM Ranking)
        chunk_final = chunk_final.with_columns([
            # Position trong candidate list (rank)
            pl.col("item_id").rank("dense").over("customer_id").alias("candidate_rank"),
            
            # Normalized cooc_max trong nhóm user (để model học relative importance)
            (pl.col("cooc_max") / pl.col("cooc_max").max().over("customer_id")).fill_null(0).alias("cooc_max_normalized"),
            
            # Số lượng candidates của user (diversity indicator)
            pl.count().over("customer_id").alias("user_candidate_count")
        ])
        
      # [NEW] 8. Word2Vec Item Similarity (OPTIMIZED NUMPY VERSION)
        if w2v_model is not None and user_purchase_seqs is not None:
            print("   -> Calculating Word2Vec Sims (Vectorized)...")
            
            # 1. Pre-normalize Vectors (QUAN TRỌNG)
            # Chuẩn hóa vector về độ dài 1. Khi đó: Cosine Sim = Dot Product
            # Giúp bỏ qua bước chia cho norm (căn bậc 2) trong vòng lặp -> Cực nhanh
            w2v_model.wv.fill_norms(force=True)
            
            # Lấy toàn bộ vector item ra RAM để tra cứu nhanh hơn truy cập vào model class
            # key_to_index giúp map item_id string sang index của matrix
            try:
                # Gensim 4.x
                key_to_index = w2v_model.wv.key_to_index 
                vectors = w2v_model.wv.get_normed_vectors()
            except AttributeError:
                # Gensim 3.x fallback
                key_to_index = {k: i for i, k in enumerate(w2v_model.wv.index2word)}
                vectors = w2v_model.wv.vectors_norm

            # 2. Convert Data sang List để xử lý (Nhanh hơn Pandas iterrows)
            # Lấy 2 cột cần thiết
            data_pairs = chunk_final.select(["customer_id", "item_id"]).to_numpy()
            
            w2v_sims = []
            
            # 3. Loop với Numpy Operations (Thay vì Scikit-Learn)
            for cust_id, item_id in data_pairs:
                cand_item = str(item_id).strip()
                
                # Check Candidate có trong vocab không
                if cand_item not in key_to_index:
                    w2v_sims.append([0.0, 0.0, 0.0])
                    continue
                
                # Lấy vector candidate (O(1) lookup)
                cand_idx = key_to_index[cand_item]
                cand_vec = vectors[cand_idx] # Shape: (128,)
                
                # Lấy lịch sử user
                hist_items = user_purchase_seqs.get(cust_id, [])
                if len(hist_items) == 0: 
                    w2v_sims.append([0.0, 0.0, 0.0])
                    continue
                
                # Lấy indices của các item trong lịch sử (Filter item có trong vocab)
                hist_indices = [key_to_index[str(h).strip()] for h in hist_items if str(h).strip() in key_to_index]
                
                if not hist_indices:
                    w2v_sims.append([0.0, 0.0, 0.0])
                    continue
                
                # [THẦN CHÚ TỐC ĐỘ] Matrix Multiplication thay vì Loop
                # hist_vecs shape: (N_history, 128)
                # cand_vec shape:  (128,)
                # Dot product ->   (N_history,) chứa toàn bộ cosine similarity
                hist_vecs = vectors[hist_indices]
                sims = np.dot(hist_vecs, cand_vec)
                
                # Tính Mean/Max/Min bằng Numpy (C code)
                w2v_sims.append([
                    float(np.mean(sims)),
                    float(np.max(sims)),
                    float(np.min(sims))
                ])
            
            # 4. Gán ngược lại Polars
            sim_matrix = np.array(w2v_sims, dtype=np.float32)
            
            w2v_df = pl.DataFrame({
                "w2v_sim_mean": sim_matrix[:, 0],
                "w2v_sim_max":  sim_matrix[:, 1],
                "w2v_sim_min":  sim_matrix[:, 2]
            })
            
            chunk_final = pl.concat([chunk_final, w2v_df], how="horizontal")

        else:
            # Fallback
            chunk_final = chunk_final.with_columns([
                pl.lit(0.0).alias("w2v_sim_mean"),
                pl.lit(0.0).alias("w2v_sim_max"),
                pl.lit(0.0).alias("w2v_sim_min")
            ])
        chunk_final = chunk_final.with_columns([
            pl.col("price_diff_abs").fill_null(99999),
            pl.col("price_ratio").fill_null(1.0)
        ])

        chunk_final = chunk_final.with_columns([
        # Fill Null cho feature mới
        pl.col("trend_7d").fill_null(0),
        pl.col("trend_30d").fill_null(0),
        pl.col("price_zscore").fill_null(0), # 0 nghĩa là giá trung bình
        pl.col("item_repurchase_rate").fill_null(0),
        
        # [NEW] 4. USER AFFINITY SHARE (Tỷ trọng sở thích)
        # Công thức: (Số lần User mua Cat này) / (Tổng số lần mua của User + 1)
        # Giả sử 'user_cat3_buy_count' đã có từ code cũ
        # Cần thêm 'user_total_buys' (nếu chưa có thì xấp xỉ bằng user_velocity * active_days hoặc tính ở prepare)
        
        # Trend Ratio: Trend 7 ngày so với 30 ngày (Đang lên hay đang xuống?)
        (pl.col("trend_7d") / (pl.col("trend_30d") + 1)).alias("trend_momentum"),
    ])
        final_cols = [
            "item_pop_log",
            "cooc_max",
            "cooc_mean",
            "cooc_len",
            "brand_match",
            "cat_l4_match",
            "user_brand_count",
            "days_since_last_brand_purchase",
            "brand_cross_score",
            "price_ratio",
            # [NEW] Temporal Features
            "purchase_velocity",
            "days_since_last_cat3_purchase",
            "user_cat3_freq_log",
            # [NEW] Query-level Features
            "candidate_rank",
            "cooc_max_normalized",
            "user_candidate_count",
            # [NEW] Word2Vec Similarity Features
            "w2v_sim_mean",
            "w2v_sim_max",
            "w2v_sim_min",

            # [NEW]
            "trend_7d", 
            "trend_30d",
            "trend_momentum",
            "price_zscore",
            "item_repurchase_rate"
        ]

        
        if model is not None:
            if pred_expr is not None:
                result_df = (
                    chunk_final
                    .select(["customer_id", "item_id"] + final_cols)
                    .with_columns([pl.col(c).cast(pl.Float32) for c in final_cols])
                    .with_columns(pred_expr.cast(pl.Float32))
                )
                cols_to_save = ["customer_id", "item_id", "pred_score"] + final_cols
                result_df.select(cols_to_save).write_parquet(file_name)
            else:
                X_test = chunk_final.select(final_cols).to_numpy()
                np.nan_to_num(X_test, copy=False, nan=0.0)
                if hasattr(model, "predict_proba"): scores = model.predict_proba(X_test)[:, 1]
                else: scores = model.predict(X_test)
                scores = np.clip(scores, 0.0, 1.0)
                
                cols_to_save = ["customer_id", "item_id"] + final_cols
                result_df = chunk_final.select(cols_to_save).with_columns(pl.Series("pred_score", scores).cast(pl.Float32))
                result_df.write_parquet(file_name)
        else:
            cols_to_save = ["customer_id", "item_id", "target"] + final_cols
            if "created_date" in chunk_final.columns: cols_to_save.append("created_date")
            chunk_final.select(cols_to_save).write_parquet(file_name)

        pbar.update(batch_size if offset + batch_size <= total_rows else total_rows - offset)
        del chunk_cooc, feat_cooc, chunk_final, chunk
    
    pbar.close()
    if not os.path.exists(success_flag):
        with open(success_flag, 'w') as f: f.write("done")
    
    # Cleanup
    del df_cooc, df_hist_long, df_user_profile, df_item, df_brand_loyalty, df_item_stats
    if 'cat_time_stats' in locals(): del cat_time_stats
    if 'user_velocity' in locals(): del user_velocity
    gc.collect()

    return pl.scan_parquet(f"{temp_dir}/part_*.parquet")