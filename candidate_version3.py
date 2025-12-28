import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import gc 
import os
import shutil

warnings.filterwarnings('ignore')

# --- 1. EXTRACT MATRIX ---
def extract_user_item_rating_coo_matrix(transaction_lf, user_mapping=None, item_mapping=None, 
                                        user_col="customer_id", item_col="item_id", 
                                        rating_col="quantity", time_col="created_date"):
    lf_filtered = (transaction_lf
                   .with_columns([(pl.col(rating_col)).alias("rating")])
                   .select([pl.col(user_col), pl.col(item_col), pl.col("rating"), pl.col(time_col)])
                   .group_by([user_col, item_col])
                   .agg([pl.col("rating").sum(), pl.col(time_col).max()]))
    df_all = lf_filtered.collect().to_pandas().dropna(subset=[user_col])

    if user_mapping is None: user_mapping = {id: idx for idx, id in enumerate(df_all[user_col].unique())}
    if item_mapping is None: item_mapping = {id: idx for idx, id in enumerate(df_all[item_col].unique())}
        
    df_all['user_idx'] = df_all[user_col].map(user_mapping)
    df_all['item_idx'] = df_all[item_col].map(item_mapping)
    
    df_all_coo = coo_matrix((df_all['rating'].values.astype(np.float32), 
                             (df_all['item_idx'].values, df_all['user_idx'].values)), 
                            shape=(len(item_mapping), len(user_mapping)))
    return df_all_coo, user_mapping, item_mapping, {v:k for k,v in user_mapping.items()}, {v:k for k,v in item_mapping.items()}

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

# --- 3. CONTENT-BASED ---
def build_content_similarity(item_lf, item_mapping, rev_item_mapping, top_k=20):
    item_df = (item_lf.select(["item_id","category_l1","category_l2","category_l3","brand","price_log"]).collect()
               .filter(pl.col("item_id").is_in(list(item_mapping.keys())))
               .with_columns([pl.col(c).fill_null("unknown") for c in ["category_l1","category_l2","category_l3","brand"]]+
                            [pl.col("price_log").fill_null(0.0)])
               .with_columns([(pl.col("price_log")*2).round().cast(pl.Int32).alias("price_bin")])
               .with_columns([(pl.col("category_l1").cast(pl.String)+"|"+pl.col("category_l2").cast(pl.String)+"|"+
                              pl.col("category_l3").cast(pl.String)+"|"+pl.col("brand").cast(pl.String)+"|"+
                              pl.col("price_bin").cast(pl.String)).alias("features")]))
    
    vectorizer = TfidfVectorizer(token_pattern=r'[^|]+')
    feat_matrix = vectorizer.fit_transform(item_df["features"].to_list())
    sim_matrix = cosine_similarity(feat_matrix, dense_output=False)
    
    content_map = {}
    items = item_df["item_id"].to_list()
    for i, item_id in enumerate(items):
        if item_id not in item_mapping: continue
        top_idx = np.argsort(sim_matrix[i].toarray().flatten())[::-1][1:top_k+1]
        content_map[item_mapping[item_id]] = [item_mapping[items[idx]] for idx in top_idx if items[idx] in item_mapping]
    return content_map

# --- 4. TRAIN CF ---
def train_CF_models(transaction_lf, q_hist):
    df_coo, user_map, item_map, rev_user, rev_item = extract_user_item_rating_coo_matrix(
        transaction_lf.filter(pl.sql_expr(q_hist)), user_col="customer_id", item_col="item_id", 
        rating_col="quantity", time_col="created_date")
    df_csr = df_coo.tocsr()
    
    cos_model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute', n_jobs=-1)
    cos_model.fit(df_csr)
    
    tfidf_matrix = TfidfTransformer().fit_transform(df_csr)
    tfidf_model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute', n_jobs=-1)
    tfidf_model.fit(tfidf_matrix)
    
    return [cos_model, tfidf_model], [df_csr, tfidf_matrix], user_map, item_map, rev_user, rev_item

# --- 5. PRECOMPUTE ---
def precompute_similarity_map(model, matrix, n_neighbors):
    sim_map = {}
    for i in range(0, matrix.shape[0], 1000):
        dists, indices = model.kneighbors(matrix[i:min(i+1000, matrix.shape[0])], n_neighbors=n_neighbors)
        for local_idx, neighbors in enumerate(indices):
            sim_map[i+local_idx] = [n for n in neighbors if n != i+local_idx]
    return sim_map

# --- 6. GET CANDIDATES ---
def get_candidates(transaction_lf, item_lf, q_hist, q_val, q_all_hist=None,
                   n_behavior=50, n_content=20, n_trend=80, total_target=150, filter_active_only=True):
    """
    n_behavior: 50 items từ Collaborative Filtering
    n_content: 20 items từ Content-based
    n_trend: 80 items từ Trending
    total_target: 150 tổng items
    q_all_hist: Query để lấy toàn bộ lịch sử từ 2024-01-01 (để lọc purchased)
    """
    temp_dir = "temp_candidates"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Lọc items hợp lệ
    valid_items_str = set()
    if item_lf is not None:
        filtered = (item_lf.select(["item_id","sale_status","created_date_year","created_date_month"]).collect()
                   #.filter((pl.col("sale_status")==1) if filter_active_only else True)
                   .filter(~pl.col("created_date_year").is_between(2011,2016))
                   .filter(~((pl.col("created_date_year")>=2025)&(pl.col("created_date_month")>=3))))
        valid_items_str = {str(x).strip() for x in filtered["item_id"].to_list()}

    trend_items_str = {str(x).strip() for x in get_trending_items(transaction_lf, item_lf, q_hist, q_val, 300)}
    if valid_items_str: trend_items_str &= valid_items_str

    models, matrices, user_map, item_map, rev_user, rev_item = train_CF_models(transaction_lf, q_hist)
    content_map = build_content_similarity(item_lf, item_map, rev_item, n_content)
    behavior_maps = [precompute_similarity_map(m, matrices[i], n_behavior) for i, m in enumerate(models)]
    
    # Load toàn bộ lịch sử mua hàng để lọc purchased (từ 2024-01-01)
    if q_all_hist is None:
        q_all_hist = q_hist  # Fallback về 120 ngày nếu không truyền
    
    all_purchased = (transaction_lf.filter(pl.sql_expr(q_all_hist))
                     .select(["customer_id", "item_id"])
                     .unique()
                     .collect()
                     .group_by("customer_id")
                     .agg(pl.col("item_id").alias("purchased_items")))
    
    user_purchased_map = {row["customer_id"]: {str(x).strip() for x in row["purchased_items"]} 
                          for row in all_purchased.iter_rows(named=True)}
    
    user_matrix = matrices[0].T.tocsr()
    ncold = 0
    
    for i in range(0, len(user_map), 50000):
        batch = []
        for cust_id in list(user_map.keys())[i:i+50000]:
            user_idx = user_map[cust_id]
            hist = user_matrix[user_idx].indices
            purchased = user_purchased_map.get(cust_id, set())  # Dùng toàn bộ lịch sử từ 2024-01-01
            hist_recent = hist[-min(10, len(hist)):]
            
            if len(hist_recent)==0:
                ncold+=1
                batch.extend([{"customer_id":cust_id,"item_id":item} for item in list(trend_items_str-purchased)[:n_trend]])
                continue
            
            # Behavior
            behav = set()
            for m_idx in range(len(models)):
                for item_idx in hist_recent:
                    if item_idx in behavior_maps[m_idx]:
                        for n_idx in behavior_maps[m_idx][item_idx][:n_behavior//2]:
                            if n_idx in rev_item:
                                s = str(rev_item[n_idx]).strip()
                                if not valid_items_str or s in valid_items_str: behav.add(s)
            behav_list = list(behav-purchased)[:n_behavior]
            
            # Content
            cont = set()
            for item_idx in hist_recent:
                if item_idx in content_map:
                    for sim_idx in content_map[item_idx][:n_content//2]:
                        if sim_idx in rev_item:
                            s = str(rev_item[sim_idx]).strip()
                            if not valid_items_str or s in valid_items_str: cont.add(s)
            cont_list = list(cont-purchased-set(behav_list))[:n_content]
            
            # Trending
            trend_list = list(trend_items_str-purchased-set(behav_list)-set(cont_list))[:n_trend]
            
            # Kết hợp
            final = behav_list + cont_list + trend_list
            if len(final) < total_target:
                final += list(trend_items_str-purchased-set(final))[:(total_target-len(final))]
            
            batch.extend([{"customer_id":cust_id,"item_id":item} for item in final[:total_target]])
        
        if batch: pl.DataFrame(batch).write_parquet(f"{temp_dir}/part_{i}.parquet")
        del batch; gc.collect()
    
    print(f"Cold start: {ncold}")
    del behavior_maps, content_map, models, matrices; gc.collect()
    return pl.scan_parquet(f"{temp_dir}/*.parquet").collect()