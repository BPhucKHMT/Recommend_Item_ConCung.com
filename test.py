import polars as pl
def check_candidate_recall(df_candidates, pos_df, k=None):
    """
    df_candidates: (customer_id, item_id, [score])
    pos_df: (customer_id, item_id)
    k: top_k candidate (None = all)
    """

    print("\n🔍 CHECK: Candidate Recall")

    # 1️⃣ Align schema (RẤT QUAN TRỌNG)
    df_candidates = df_candidates.select([
        pl.col("customer_id").cast(pl.Int32),
        pl.col("item_id").cast(pl.String)
    ])

    pos_df = pos_df.select([
        pl.col("customer_id").cast(pl.Int32),
        pl.col("item_id").cast(pl.String)
    ])

    # 2️⃣ Nếu có k → lấy TOP-K đúng nghĩa
    if k is not None:
        df_candidates = (
            df_candidates
            .with_row_count("_idx")  # giữ thứ tự ban đầu
            .sort(["customer_id", "_idx"])
            .group_by("customer_id", maintain_order=True)
            .head(k)
            .drop("_idx")
        )

    # 3️⃣ Hit = positive ∩ candidate
    hit = (
        pos_df
        .join(
            df_candidates,
            on=["customer_id", "item_id"],
            how="inner"
        )
    )

    recall = hit.height / pos_df.height if pos_df.height > 0 else 0.0

    print(f"   -> #Positive GT     : {pos_df.height}")
    print(f"   -> #Hit in candidate: {hit.height}")
    print(f"   -> Candidate Recall@{k if k else 'ALL'} = {recall:.4f}")

    return recall
df_candidates = pl.read_parquet("candidates_stage1.parquet")

# Toàn bộ candidate
check_candidate_recall(df_candidates, pos_df, k=None)

# Top-200
check_candidate_recall(df_candidates, pos_df, k=200)

# Top-100
check_candidate_recall(df_candidates, pos_df, k=100)


