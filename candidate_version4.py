"""
STATE-OF-THE-ART CANDIDATE GENERATION (H200 OPTIMIZED - LOGIC PRESERVED)
- ALS Matrix Factorization (GPU Batch Mode)
- BPR Ranking (GPU Batch Mode)
- Item2Vec Embeddings (Vectorized)
- Content-based (FAISS GPU)
- Logic & Filters: 100% Preserved from original
"""

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import warnings
import gc
import os
import shutil
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- HARDWARE CONFIG ---
N_THREADS = 10 
os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_THREADS)

# GPU Check
try:
    import cupy as cp
    import cupy.cuda.runtime as runtime 
    
    # Kiểm tra số lượng GPU
    n_devices = runtime.getDeviceCount()
    
    if n_devices > 0:
        HAS_GPU = True
        props = runtime.getDeviceProperties(0)
        gpu_name = props['name'].decode('utf-8')
        print(f"🚀 GPU DETECTED: {gpu_name} (H200 Optimization Enabled)")
    else:
        HAS_GPU = False
        print("⚠️ CuPy installed but NO GPU detected.")

except ImportError:
    HAS_GPU = False
    print("⚠️ No CuPy found. Install 'cupy-cuda12x' to unlock H200 speed!")
except Exception as e:
    HAS_GPU = False
    print(f"⚠️ GPU Check Failed: {e}")
    
try:
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    from implicit.gpu import HAS_CUDA
    HAS_IMPLICIT = True
    if HAS_CUDA: print("✓ IMPLICIT: GPU Acceleration ENABLED")
except ImportError:
    HAS_IMPLICIT = False

try:
    import faiss
    HAS_FAISS = True
    if hasattr(faiss, 'StandardGpuResources'): print("✓ FAISS: GPU Acceleration ENABLED")
except ImportError:
    HAS_FAISS = False


# --- 8. ADVANCED FALLBACK: SEGMENT-BASED TRENDING ---
def get_advanced_fallback(df_current_results, user_lf, transaction_lf, item_lf, q_hist, q_val, n_fill=10):
    """
    Chiến thuật Cold Start nâng cao:
    1. Chia user thành các Segment (Gender + Region + Membership).
    2. Tính Trending riêng cho từng Segment.
    3. Fill vào cho user bị thiếu.
    4. Ai không thuộc Segment nào thì dùng Global Trending.
    """
    print("\n>>> 🛡️ ADVANCED FALLBACK (SEGMENT-BASED) <<<")
    
    # ------------------------------------------------------------------
    # BƯỚC 1: XÁC ĐỊNH USER BỊ THIẾU (MISSING)
    # ------------------------------------------------------------------
    # Lấy toàn bộ user từ bảng User (Source of Truth)
    all_users = user_lf.select(pl.col("customer_id")).unique()
    
    # Lấy những user đã có đủ 10 items
    sufficient_users = (
        df_current_results.lazy()
        .group_by("customer_id")
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") >= n_fill)
        .select(pl.col("customer_id"))
    )
    
    # Tìm user thiếu: All - Sufficient
    missing_users_df = all_users.join(sufficient_users, on="customer_id", how="anti").collect()
    
    n_missing = missing_users_df.height
    if n_missing == 0:
        print("   ✅ No missing users! Skipping fallback.")
        return df_current_results

    print(f"   🚨 Found {n_missing:,} users needing fallback.")

    # ------------------------------------------------------------------
    # BƯỚC 2: CHUẨN BỊ GLOBAL TRENDING 
    # ------------------------------------------------------------------
    print("   -> Calculating Global Trending...")
    
    # Lọc item active
    active_items = item_lf.filter(pl.col("sale_status") == 1).select(pl.col("item_id").cast(pl.String)).collect()
    active_set = set(active_items["item_id"].to_list())
    
    # Tính Global Top N
    global_trend_df = (
        transaction_lf
        .filter(pl.sql_expr(f"({q_hist}) OR ({q_val})"))
        .group_by("item_id")
        .agg(pl.len().alias("score"))
        .sort("score", descending=True)
        .limit(n_fill * 2) # Lấy dư ra chút
        .select(pl.col("item_id").cast(pl.String))
        .collect()
        .filter(pl.col("item_id").is_in(active_set)) # Chỉ lấy active
        .head(n_fill)
    )
    
    # ------------------------------------------------------------------
    # BƯỚC 3: TÍNH SEGMENT TRENDING 
    # ------------------------------------------------------------------
    print("   -> Calculating Segment-Based Trending (Gender + Region)...")
    
    # a. Lấy thông tin Segment của user (Gender, Region)
    # Xử lý null: Nếu null thì điền 'unknown'
    user_segments = (
        user_lf
        .select([
            pl.col("customer_id"),
            pl.col("gender").fill_null("unknown").cast(pl.String),
            pl.col("region").fill_null("unknown").cast(pl.String)
        ])
    )
    
    # b. Join Transaction với User Segment để biết "Nhóm nào mua gì"
    # Logic: Transaction -> Join User -> GroupBy [Gender, Region, Item] -> Count
    segment_sales = (
        transaction_lf
        .join(user_segments, on="customer_id", how="inner") # Join với lazy frame user
        .group_by(["gender", "region", "item_id"])
        .agg(pl.len().alias("seg_score"))
        .filter(pl.col("item_id").cast(pl.String).is_in(active_set)) # Chỉ lấy item active
        .sort(["gender", "region", "seg_score"], descending=[False, False, True])
    )
    
    # c. Lấy Top N item cho MỖI nhóm (Window Function)
    # Đoạn này chạy cực nhanh trên Polars
    top_segment_items = (
        segment_sales
        .with_columns(
            pl.col("seg_score").rank("ordinal", descending=True).over(["gender", "region"]).alias("rank")
        )
        .filter(pl.col("rank") <= n_fill)
        .select(["gender", "region", pl.col("item_id").cast(pl.String), pl.col("seg_score").cast(pl.Float64)])
        .collect()
    )
    
    # ------------------------------------------------------------------
    # BƯỚC 4: FILL ITEM CHO USER THIẾU (VECTORIZED JOIN)
    # ------------------------------------------------------------------
    print("   -> Applying Segment Strategy...")
    
    # a. Lấy thông tin segment của các user bị thiếu
    missing_with_seg = (
        missing_users_df.lazy()
        .join(user_segments, on="customer_id", how="left")
        .with_columns([
            pl.col("gender").fill_null("unknown"),
            pl.col("region").fill_null("unknown")
        ])
        .collect()
    )
    
    # b. Join với bảng Top Segment Items
    # Đây là bước quyết định: User Nam-HN sẽ được join với Top Items của Nam-HN
    fallback_segment = (
        missing_with_seg.lazy()
        .join(top_segment_items.lazy(), on=["gender", "region"], how="left")
        .select(["customer_id", "item_id", pl.col("seg_score").alias("pred_score")])
        .collect()
    )
    
    # c. Những user nào join xong mà item_id bị null (do nhóm đó ko có data), hoặc score thấp
    # Thì dùng Global Trending đắp vào
    # Chiến thuật: Tạo Global cho TẤT CẢ missing, sau đó ưu tiên Segment
    
    # Tạo bảng Global cho tất cả missing user (Cross Join)
    # Global score set thấp (0.001) để ưu tiên segment score (thường là số nguyên > 1)
    fallback_global = (
        missing_users_df.lazy()
        .join(global_trend_df.lazy().with_columns(pl.lit(0.001).alias("pred_score")), how="cross")
        .collect()
    )
    
    # Gộp tất cả lại: Segment Recs + Global Recs
    # Deduplicate: Nếu item có trong cả 2, giữ cái nào score cao hơn (Segment)
    final_fallback = (
        pl.concat([fallback_segment, fallback_global])
        .drop_nulls(subset=["item_id"]) # Bỏ rác
        .unique(subset=["customer_id", "item_id"], keep="first") # Segment nằm trên nên sẽ được giữ
        .sort(["customer_id", "pred_score"], descending=[False, True])
        .group_by("customer_id")
        .head(n_fill)
    )
    
    print(f"   ✅ Generated fallback for {final_fallback['customer_id'].n_unique():,} users.")
    
    # ------------------------------------------------------------------
    # BƯỚC 5: MERGE VÀO KẾT QUẢ CHÍNH
    # ------------------------------------------------------------------
    # Cast đúng schema trước khi concat
    target_schema_id = df_current_results.schema["customer_id"]
    target_schema_item = df_current_results.schema["item_id"]
    target_schema_score = df_current_results.schema["pred_score"]
    
    final_fallback = final_fallback.select([
        pl.col("customer_id").cast(target_schema_id),
        pl.col("item_id").cast(target_schema_item),
        pl.col("pred_score").cast(target_schema_score)
    ])
    
    # Concat -> Unique -> Sort -> Head
    df_merged = (
        pl.concat([df_current_results, final_fallback])
        .unique(subset=["customer_id", "item_id"])
        .sort(["customer_id", "pred_score"], descending=[False, True])
        .group_by("customer_id")
        .head(n_fill)
    )
    
    return df_merged


# --- 2. TRENDING với filter năm ---
def get_trending_items(transaction_lf, item_lf, q_hist, q_val, n_trend=300):
    if n_trend == 0: return set()
    
    trend_raw = (transaction_lf.filter(pl.sql_expr(f"({q_hist}) OR ({q_val})"))
                 .group_by("item_id").agg(pl.col("created_date").len().alias("cnt"))
                 .sort("cnt", descending=True).limit(n_trend*2)
                 .select("item_id").collect().get_column("item_id").to_list())
    
    if item_lf is not None:
        filtered = (item_lf.select(["item_id","created_date_year","created_date_month"]).collect()
                   .filter(~pl.col("created_date_year").is_between(2011,2016))
                   .filter(~((pl.col("created_date_year")>=2025)&(pl.col("created_date_month")>=3)))
                   .select("item_id").to_series().to_list())
        filtered_str = {str(x).strip() for x in filtered}
        return set([x for x in trend_raw if str(x).strip() in filtered_str][:n_trend])
    return set(trend_raw[:n_trend])

# --- 1. EXTRACT MATRIX ---
def extract_user_item_matrix_with_time(transaction_lf, q_hist, decay_days=30):
    """Extract user-item matrix với time decay weighting (Optimized with Pandas)"""
    print(">> Extracting user-item matrix...")
    
    df = (transaction_lf.filter(pl.sql_expr(q_hist))
          .select(["customer_id", "item_id", "quantity", "created_date"])
          .collect())
    
    # Pandas xử lý datetime vectorization nhanh hơn loop
    df_pd = df.to_pandas()
    df_pd['created_date'] = pd.to_datetime(df_pd['created_date'])
    
    max_date = df_pd['created_date'].max()
    days_ago = (max_date - df_pd['created_date']).dt.days
    
    # Vectorized Weight Calculation
    df_pd['weighted_rating'] = df_pd['quantity'] * np.exp(-days_ago / decay_days)
    
    # GroupBy Sum
    df_agg = df_pd.groupby(['customer_id', 'item_id'], as_index=False).agg({
        'weighted_rating': 'sum',
        'created_date': 'max'
    })
    
    # Mappings
    user_ids = df_agg['customer_id'].unique()
    item_ids = df_agg['item_id'].unique()
    
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}
    
    rev_user = {i: uid for uid, i in user_map.items()}
    rev_item = {i: iid for iid, i in item_map.items()}
    
    # Build Matrix
    row = df_agg['customer_id'].map(user_map).values
    col = df_agg['item_id'].map(item_map).values
    data = df_agg['weighted_rating'].values.astype(np.float32)
    
    matrix_coo = coo_matrix((data, (row, col)), shape=(len(user_map), len(item_map)))
    
    print(f"   Matrix shape: {matrix_coo.shape}, nnz: {matrix_coo.nnz:,}")
    return matrix_coo, user_map, item_map, rev_user, rev_item, df_agg


# --- 2. TRAIN MODELS (GPU ENABLED) ---
def train_als_model(matrix_coo, factors=128, iterations=20, regularization=0.01):
    if not HAS_IMPLICIT: return None
    use_gpu = HAS_IMPLICIT and HAS_CUDA
    device = "GPU" if use_gpu else "CPU"
    print(f">> Training ALS on {device}...")
    
    # Force GPU usage for H200
    model = AlternatingLeastSquares(factors=factors, iterations=iterations, regularization=regularization, 
                                    use_gpu=use_gpu, num_threads=N_THREADS)
    model.fit(matrix_coo.tocsr())
    return model

def train_bpr_model(matrix_coo, factors=128, iterations=100):
    if not HAS_IMPLICIT: return None
    use_gpu = HAS_IMPLICIT and HAS_CUDA
    device = "GPU" if use_gpu else "CPU"
    print(f">> Training BPR on {device}...")
    
    model = BayesianPersonalizedRanking(factors=factors, iterations=iterations, learning_rate=0.01, 
                                        regularization=0.01, use_gpu=use_gpu, num_threads=N_THREADS)
    model.fit(matrix_coo.tocsr())
    return model


# --- 3. ITEM2VEC  ---
import pandas as pd
def train_item2vec(df_agg, user_map, item_map, vector_size=128, window=5):
    print(f">> Training Item2Vec (dim={vector_size})...")
    
    # [OPTIMIZATION] Dùng Pandas GroupBy thay vì vòng lặp Python
    # Bước này cực nhanh với 250GB RAM
    print("   -> Preparing sequences (Vectorized)...")
    
    # Đảm bảo df_agg là pandas (nó đã là pandas từ hàm extract, nhưng check cho chắc)
    if isinstance(df_agg, pl.DataFrame): df_agg = df_agg.to_pandas()
    
    # Sort theo thời gian
    df_sorted = df_agg.sort_values(['customer_id', 'created_date'])
    
    # Map sang string index (Word2Vec cần list of str)
    df_sorted['item_str'] = df_sorted['item_id'].map(item_map).astype(str)
    
    # GroupBy tạo list
    sequences = df_sorted.groupby('customer_id')['item_str'].apply(list).tolist()
    
    # Filter short sequences
    sequences = [s for s in sequences if len(s) >= 2]
    
    print(f"   Sequences: {len(sequences)}")
    if len(sequences) < 10: return None
    
    print("   -> Fitting Word2Vec...")
    model = Word2Vec(
        sentences=sequences,  # [FIX] Đã đổi sequences -> sentences
        vector_size=vector_size, window=window, min_count=2, 
        workers=N_THREADS, epochs=10, sg=1
    )
    print(f"   ✓ Vocab: {len(model.wv)}")
    return model


# --- 4. CONTENT SIMILARITY  ---
def build_enhanced_content_similarity(item_lf, item_mapping, top_k=50):
    print(f">> Building content similarity (FAISS)...")
    
    # Load data & Create features
    item_df = (item_lf.select([
        "item_id", "category_l1", "category_l2", "category_l3", "brand", "price_log"
    ]).collect().filter(pl.col("item_id").is_in(list(item_mapping.keys()))))
    
    df_pd = item_df.to_pandas().fillna("unknown")
    df_pd['price_bin'] = (df_pd['price_log'] * 2).round().astype(int).astype(str)
    df_pd['features'] = (df_pd['category_l1'] + " " + df_pd['category_l2'] + " " + 
                         df_pd['category_l3'] + " " + df_pd['brand'] + " " + df_pd['price_bin'])
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    feat_matrix = vectorizer.fit_transform(df_pd['features']).toarray().astype(np.float32)
    
    items = df_pd['item_id'].tolist()
    content_map = {}
    
    # --- FAISS LOGIC (AUTO SWITCH) ---
    if HAS_FAISS:
        d = feat_matrix.shape[1]
        index = faiss.IndexFlatIP(d) # Inner Product (Cosine nếu đã normalize)
        faiss.normalize_L2(feat_matrix)
        
        # Check xem có dùng được GPU không?
        use_gpu_index = False
        if hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                # Thử đẩy qua GPU
                index = faiss.index_cpu_to_gpu(res, 0, index)
                use_gpu_index = True
                print("   ✓ FAISS: Running on GPU")
            except:
                print("   ⚠️ FAISS: GPU failed, falling back to CPU")
        
        if not use_gpu_index:
            print("   ✓ FAISS: Running on CPU (Ram Mode)")

        # Add & Search
        index.add(feat_matrix)
        D, I = index.search(feat_matrix, top_k+1)
        
        # Map results
        valid_set = set(item_mapping.keys())
        for i, row_indices in enumerate(tqdm(I, desc="Mapping Content")):
            src_id = items[i]
            if src_id not in valid_set: continue
            
            sims = []
            for j, idx in enumerate(row_indices[1:]): # Skip self
                tgt_id = items[idx]
                score = D[i][j+1]
                if tgt_id in valid_set and score > 0.1:
                    sims.append((item_mapping[tgt_id], float(score)))
            content_map[item_mapping[src_id]] = sims
            
    else:
        print("   ⚠️ FAISS module not found. Skipping content sim.")
        
    return content_map


# --- 5. MMR DIVERSITY (LOGIC GIỮ NGUYÊN) ---
def maximal_marginal_relevance(candidates_with_scores, item_features, lambda_param=0.5, k=150):
    if len(candidates_with_scores) <= k: return [item for item, _ in candidates_with_scores]
    selected = []
    candidates = list(candidates_with_scores)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected.append(candidates[0][0])
    candidates.pop(0)
    
    while len(selected) < k and candidates:
        mmr_scores = []
        for item_idx, rel_score in candidates:
            relevance = rel_score
            max_sim = 0
            if item_idx in item_features:
                feat_i = item_features[item_idx]
                for sel_idx in selected:
                    if sel_idx in item_features:
                        feat_s = item_features[sel_idx]
                        # Simple category similarity
                        sim = int(feat_i.get('cat1') == feat_s.get('cat1')) * 0.5 + \
                              int(feat_i.get('cat2') == feat_s.get('cat2')) * 0.5
                        max_sim = max(max_sim, sim)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((item_idx, mmr_score))
        
        mmr_scores.sort(key=lambda x: x[1], reverse=True)
        selected.append(mmr_scores[0][0])
        candidates = [(i, s) for i, s in candidates if i != mmr_scores[0][0]]
    
    return selected


# --- 6. HELPER: GET CATEGORY RECS ---
def build_category_recommendations(item_lf, transaction_lf, q_hist, top_k=100):
    # Logic cũ: Lấy top item theo category
    item_cats = (item_lf.select(["item_id", "category_l1"]).collect())
    pop_items = (transaction_lf.filter(pl.sql_expr(q_hist))
                 .group_by("item_id").agg(pl.col("quantity").sum().alias("cnt"))
                 .collect().sort("cnt", descending=True))
    merged = item_cats.join(pop_items, on="item_id", how="inner")
    
    cat_recs = {}
    for cat1 in merged["category_l1"].unique():
        items = merged.filter(pl.col("category_l1")==cat1).head(top_k)["item_id"].to_list()
        cat_recs[cat1] = items
    return cat_recs


# --- 7. MAIN FUNCTION: GET CANDIDATES ---
def get_candidates(transaction_lf, item_lf, q_hist, q_val, q_all_hist=None, 
                   total_target=150, use_diversity=True, filter_active_only=True):
    
    temp_dir = "temp_candidates"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("\n" + "="*80)
    print("🚀 GENERATION START (BATCH MODE) | RAM: 250GB | GPU: H200")
    print("="*80)
    
    # 1. FILTER VALID ITEMS (LOGIC GỐC CỦA BẠN - KHÔNG SỬA)
    print("\n>> Filtering valid items (Preserving Original Logic)...")
    valid_items_str = set()
    item_features = {}
    
    if item_lf is not None:
        # Tái tạo chính xác bộ lọc của bạn
        filter_expr = ~pl.col("created_date_year").is_between(2011, 2016)
        filter_expr &= ~((pl.col("created_date_year") >= 2025) & (pl.col("created_date_month") >= 3))
        
        if filter_active_only:
            filter_expr &= (pl.col("sale_status") == 1)
            
        filtered = (item_lf.select([
            "item_id", "category_l1", "category_l2", "sale_status", 
            "created_date_year", "created_date_month"
        ]).collect().filter(filter_expr))
        
        valid_items_str = set(str(x).strip() for x in filtered["item_id"].to_list())
        
        # Store features for MMR Diversity
        for row in filtered.iter_rows(named=True):
            item_features[str(row['item_id']).strip()] = {
                'cat1': row['category_l1'], 'cat2': row['category_l2']
            }
        print(f"   Valid items count: {len(valid_items_str):,}")

    # 2. Trending
    print(">> Getting trending items...")
    trend_raw = (transaction_lf.filter(pl.sql_expr(f"({q_hist}) OR ({q_val})"))
                 .group_by("item_id").agg(pl.col("created_date").len().alias("cnt"))
                 .collect().sort("cnt", descending=True).head(500))
    trend_items_str = [str(x).strip() for x in trend_raw["item_id"].to_list()]
    if valid_items_str:
        trend_items_str = [x for x in trend_items_str if x in valid_items_str]

    # 3. Train Models
    mat_coo, user_map, item_map, rev_user, rev_item, df_agg = \
        extract_user_item_matrix_with_time(transaction_lf, q_hist, decay_days=30)
    
    als_model = train_als_model(mat_coo, factors=128)
    bpr_model = train_bpr_model(mat_coo, factors=128)
    item2vec_model = train_item2vec(df_agg, user_map, item_map)
    content_map = build_enhanced_content_similarity(item_lf, item_map)
    
    # 4. History Lookup
    if q_all_hist is None: q_all_hist = q_hist
    hist_df = (transaction_lf.filter(pl.sql_expr(q_all_hist))
               .select(["customer_id", "item_id"]).unique().collect())
    user_purchased_map = defaultdict(set)
    for r in hist_df.iter_rows():
        user_purchased_map[r[0]].add(str(r[1]).strip())
        
    mat_csr = mat_coo.tocsr()
    
    # 5. BATCH INFERENCE LOOP
    # Tăng Batch Size để tận dụng RAM/GPU, giảm overhead Python loop
    BATCH_SIZE = 100000
    users = list(user_map.keys())
    
    print(f">> Starting Inference for {len(users):,} users (Batch: {BATCH_SIZE})...")
    
    # [TQDM] Theo dõi tiến độ từng batch
    for start in tqdm(range(0, len(users), BATCH_SIZE), desc="🚀 Processing Batches"):
        end = min(start + BATCH_SIZE, len(users))
        batch_users = users[start:end]
        batch_indices = [user_map[u] for u in batch_users]
        
        # A. MATRIX INFERENCE (GPU) - Tính 1 lần cho cả batch
        als_res = [[], []]; bpr_res = [[], []]
        
        if als_model:
            ids, scs = als_model.recommend(batch_indices, mat_csr[batch_indices], N=200, filter_already_liked_items=False)
            als_res = [ids, scs]
        if bpr_model:
            ids, scs = bpr_model.recommend(batch_indices, mat_csr[batch_indices], N=200, filter_already_liked_items=False)
            bpr_res = [ids, scs]
            
        batch_output = []
        
        # B. COMBINE SCORES (Logic Python nhưng chạy trên tập nhỏ)
        for i, uid in enumerate(batch_users):
            candidates = defaultdict(float)
            purchased = user_purchased_map.get(uid, set())
            
            # 1. ALS (Weight 0.4)
            if als_model:
                for idx, sc in zip(als_res[0][i], als_res[1][i]):
                    iid = rev_item.get(idx)
                    if iid:
                        s_iid = str(iid).strip()
                        if s_iid not in purchased and (not valid_items_str or s_iid in valid_items_str):
                            candidates[s_iid] += sc * 0.4
                            
            # 2. BPR (Weight 0.3)
            if bpr_model:
                for idx, sc in zip(bpr_res[0][i], bpr_res[1][i]):
                    iid = rev_item.get(idx)
                    if iid:
                        s_iid = str(iid).strip()
                        if s_iid not in purchased and (not valid_items_str or s_iid in valid_items_str):
                            candidates[s_iid] += sc * 0.3
            
            # 3. Item2Vec (0.2) & Content (0.1) - Logic cũ: dựa trên history
            # Để nhanh, chỉ lấy 5 item gần nhất của user hiện tại
            u_idx = batch_indices[i]
            hist_idxs = mat_csr[u_idx].indices
            recent_idxs = hist_idxs[-10:] if len(hist_idxs) > 0 else []
            
            for h_idx in recent_idxs:
                # Item2Vec
                if item2vec_model:
                    h_key = str(h_idx)
                    if h_key in item2vec_model.wv:
                        for sim_key, sim_sc in item2vec_model.wv.most_similar(h_key, topn=10):
                            s_idx = int(sim_key)
                            iid = rev_item.get(s_idx)
                            if iid:
                                s_iid = str(iid).strip()
                                if s_iid not in purchased and (not valid_items_str or s_iid in valid_items_str):
                                    candidates[s_iid] += sim_sc * 0.2
                # Content
                if h_idx in content_map:
                    for s_idx, s_sc in content_map[h_idx][:10]:
                        iid = rev_item.get(s_idx)
                        if iid:
                            s_iid = str(iid).strip()
                            if s_iid not in purchased and (not valid_items_str or s_iid in valid_items_str):
                                candidates[s_iid] += s_sc * 0.1

            # 4. Fallback Trending
            if not candidates:
                for t in trend_items_str[:total_target]:
                    if t not in purchased: candidates[t] = 0.01
            
            # 5. Sort & MMR (Diversity)
            # Chuyển candidates về list tuples
            cand_list = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            
            if use_diversity and len(cand_list) > total_target:
                # [LOGIC] Chỉ chạy MMR trên Top 2*Target để không quá chậm
                # Vẫn đảm bảo tính chất diversity trên tập top đầu
                top_pool = cand_list[:total_target*2]
                final_items = maximal_marginal_relevance(top_pool, item_features, lambda_param=0.7, k=total_target)
            else:
                final_items = [x[0] for x in cand_list[:total_target]]
                
            # Fill thiếu
            if len(final_items) < total_target:
                rem = [t for t in trend_items_str if t not in purchased and t not in final_items]
                final_items.extend(rem[:total_target - len(final_items)])
            
            # Save format
            for item in final_items[:total_target]:
                batch_output.append({"customer_id": uid, "item_id": item})
        
        # Save Batch Parquet
        if batch_output:
            pl.DataFrame(batch_output).write_parquet(f"{temp_dir}/part_{start}.parquet")
            
    print(">> Cleaning up...")
    del als_model, bpr_model, item2vec_model, content_map, mat_coo, mat_csr
    gc.collect()
    
    print(">> Loading results...")
    result = pl.scan_parquet(f"{temp_dir}/*.parquet").collect()
    print(f"   Total candidates: {result.shape[0]:,}")
    
    return result