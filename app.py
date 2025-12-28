import streamlit as st
import polars as pl
import pandas as pd
import pickle
import os
import random
import numpy as np
import plotly.express as px 

# --- CẤU HÌNH ---
DATA_DIR = "data/table"
RES_PATH = "final_submission.parquet"
GT_PATH = "data/groundtruth.pkl"

if os.path.exists("lasso_model.pkl"):
    MODEL_PATH = "lasso_model.pkl"
else:
    MODEL_PATH = "logistic_model.pkl"

st.set_page_config(layout="wide", page_title="RecSys Audit (New Items Only)")

# --- 1. LOAD DATA ---
@st.cache_resource
def load_data():
    data = {}
    if os.path.exists(RES_PATH):
        try: data['preds'] = pl.read_parquet(RES_PATH)
        except Exception as e: st.error(f"Lỗi Recs: {e}"); return None
    else: return None

    if os.path.exists(GT_PATH):
        with open(GT_PATH, 'rb') as f: data['gt'] = pickle.load(f)
    else: return None

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f: data['model'] = pickle.load(f)
    else: data['model'] = None

    trans_path = os.path.join(DATA_DIR, "purchase_data.parquet")
    item_path = os.path.join(DATA_DIR, "item_data.parquet")
    
    if os.path.exists(trans_path):
        data['trans'] = pl.read_parquet(trans_path).with_columns(
            pl.date("created_year", "created_month", "created_day").alias("date")
        )
    
    if os.path.exists(item_path):
        try:
            lf = pl.scan_parquet(item_path)
            all_cols = lf.collect_schema().names()
            target_cols = ["item_id", "brand", "category_l1", "category_l3", "category_l4", "category"]
            cols_to_load = [c for c in target_cols if c in all_cols]
            data['items'] = pl.read_parquet(item_path).select(cols_to_load).fill_null("unknown")
        except: data['items'] = None
    
    return data

# --- MAIN APP ---
data_store = load_data()

if data_store:
    df_preds = data_store.get('preds')
    gt_dict = data_store.get('gt')
    df_trans = data_store.get('trans')
    df_items = data_store.get('items')
    model = data_store.get('model')

    # --- SIDEBAR ---
    valid_users = list(gt_dict.keys())
    if "current_user_id" not in st.session_state:
        st.session_state["current_user_id"] = random.choice(valid_users)

    def pick_random_user():
        st.session_state["current_user_id"] = random.choice(valid_users)

    st.sidebar.title("🔍 RecSys Audit")
    st.sidebar.button("🔀 Random User", on_click=pick_random_user, type="primary")
    
    selected_user_str = st.sidebar.selectbox(
        f"Users ({len(valid_users)}):", 
        options=valid_users, format_func=lambda x: f"ID: {x}", key="current_user_id"
    )
    try: selected_user = int(selected_user_str)
    except: selected_user = selected_user_str

    st.title(f"👤 User Analysis: {selected_user}")

    # ==============================================================================
    # 🧠 MODEL INSIGHTS (11 FEATURES)
    # ==============================================================================
    with st.expander("🧠 Model Internals (Bias & Weights)", expanded=True): 
        if model is None:
            st.warning("⚠️ Không tìm thấy file model.")
        else:
            # [CẬP NHẬT] Danh sách 11 Features (Đã bỏ 3 cái cũ)
            feature_cols = [
                "cooc_max", "cooc_mean", "cooc_sum", "cooc_len", 
                "brand_match", "cat_l3_match", "cat_l4_match",
                "user_brand_count", "days_since_last_brand_purchase",
                "brand_cross_score", "brand_cross_max"
            ]
            
            try:
                if isinstance(model.intercept_, (list, np.ndarray)) or (hasattr(model.intercept_, 'ndim') and model.intercept_.ndim > 0):
                    bias = model.intercept_[0]
                else: bias = model.intercept_

                if hasattr(model.coef_, 'ndim') and model.coef_.ndim > 1:
                    weights = model.coef_[0]
                else: weights = model.coef_
                
                st.metric("Model Bias (Intercept)", f"{bias:.4f}")
                
                w_df = pd.DataFrame({"Feature": feature_cols, "Weight": weights})
                w_df["Impact"] = w_df["Weight"].apply(lambda x: "Positive (+)" if x > 0 else "Negative (-)")
                w_df["AbsWeight"] = w_df["Weight"].abs()
                w_df = w_df.sort_values(by="AbsWeight", ascending=True)

                fig = px.bar(
                    w_df, x="Weight", y="Feature", orientation='h', color="Impact",
                    color_discrete_map={"Positive (+)": "#28a745", "Negative (-)": "#dc3545"},
                    title="Feature Importance (New Items Only)", text_auto='.4f', height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi hiển thị trọng số: {e}")

    # --- PREPARE DATA ---
    actual_data = gt_dict[selected_user]
    if isinstance(actual_data, dict) and 'list_items' in actual_data:
        actual_items = [str(x) for x in actual_data['list_items']]
    else: actual_items = [str(x) for x in actual_data]
    
    user_recs = df_preds.filter(pl.col("customer_id") == selected_user)

    # --- PHẦN 1: LỊCH SỬ ---
    st.markdown("---")
    st.header("1. 🕰️ Lịch sử mua hàng")
    if df_trans is not None:
        start_date = pd.to_datetime("2024-01-01").date()
        end_date = pd.to_datetime("2025-01-01").date()
        history = df_trans.filter(pl.col("customer_id")==selected_user).filter((pl.col("date")>=start_date)&(pl.col("date")<end_date)).sort("date", descending=True)
        if history.height > 0:
            if df_items is not None: history = history.join(df_items, on="item_id", how="left")
            disp_cols = ["date", "item_id", "quantity", "brand", "category_l1" , "category_l2""category_l3", "category_l4"]
            disp_cols = [c for c in disp_cols if c in history.columns]
            st.dataframe(history.select(disp_cols).to_pandas(), use_container_width=True, hide_index=True)
        else: st.info("⚠️ Không có giao dịch gần đây.")

    # --- PHẦN 2: RECOMMENDATION ---
    st.markdown("---")
    col_rec, col_gt = st.columns([1.8, 1])
    with col_rec:
        st.header("2. 🤖 Model Recommend")
        if user_recs.height == 0: st.warning("⚠️ Không có recommendation.")
        else:
            rec_df_show = user_recs.to_pandas()
            rec_df_show["Target?"] = rec_df_show["item_id"].astype(str).isin(actual_items)
            
            if df_items is not None:
                item_info = df_items.to_pandas()
                item_info["item_id"] = item_info["item_id"].astype(str)
                rec_df_show["item_id_str"] = rec_df_show["item_id"].astype(str)
                rec_df_show = rec_df_show.merge(item_info, left_on="item_id_str", right_on="item_id", how="left")
                cols_to_drop = ["item_id_str", "item_id_y"]
                rec_df_show = rec_df_show.drop(columns=[c for c in cols_to_drop if c in rec_df_show.columns]).rename(columns={"item_id_x": "Item ID"})
            else: rec_df_show = rec_df_show.rename(columns={"item_id": "Item ID"})

            # [CẬP NHẬT] Bỏ hiển thị các cột đã xóa
            main_cols = ["Item ID", "Target?", "pred_score", "brand", "category_l4"]
            feat_cols = [
                "cat_l4_match", "cooc_max", "cooc_len",
                "brand_cross_score", "user_brand_count"
            ]
            
            final_cols = []
            for group in [main_cols, feat_cols]: final_cols.extend([c for c in group if c in rec_df_show.columns])
            
            st.dataframe(
                rec_df_show[final_cols], use_container_width=True, hide_index=True,
                column_config={
                    "pred_score": st.column_config.ProgressColumn("Score", format="%.4f", min_value=0, max_value=1),
                    "Target?": st.column_config.CheckboxColumn("Hit?"),
                    "category_l4": st.column_config.TextColumn("Cat L4"),
                    "cat_l4_match": st.column_config.NumberColumn("L4 Match", format="%d"),
                    "user_brand_count": st.column_config.NumberColumn("Brand Count", format="%d"),
                    "brand_cross_score": st.column_config.NumberColumn("Br.X-Sell", format="%d"),
                    "cooc_max": st.column_config.ProgressColumn("Cooc", format="%.2f", min_value=0, max_value=5),
                }
            )

    # --- PHẦN 3: KẾT QUẢ ---
    with col_gt:
        st.header("3. 🛒 Thực tế")
        if not actual_items: st.info("Empty Ground Truth.")
        else:
            rec_list_str = [str(x) for x in user_recs["item_id"].to_list()]
            gt_data = []
            hits = 0
            for item in actual_items:
                is_hit = item in rec_list_str
                if is_hit: hits += 1
                gt_data.append({"Item ID": item, "Status": "🎯" if is_hit else "➖"})
            
            df_gt_show = pd.DataFrame(gt_data)
            if df_items is not None:
                item_info = df_items.to_pandas()
                item_info["item_id"] = item_info["item_id"].astype(str)
                df_gt_show = df_gt_show.merge(item_info, left_on="Item ID", right_on="item_id", how="left").drop(columns=["item_id"])
            
            st.dataframe(df_gt_show, use_container_width=True, hide_index=True)
            if len(rec_list_str)>0: st.metric("Precision", f"{hits/len(rec_list_str):.1%}")

else: st.error("❌ Error loading data.")