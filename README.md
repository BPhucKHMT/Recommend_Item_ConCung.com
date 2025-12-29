# 🎯 Recommend_Item_ConCung.com

![RecSys](https://img.shields.io/badge/RecSys-Personalized-blue?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square) ![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

Hệ thống **Recommendation System** này được xây dựng để đề xuất sản phẩm cá nhân hóa cho khách hàng của ConCung.com, dựa trên lịch sử giao dịch, thông tin sản phẩm và các đặc trưng hành vi.

Về data do vấn đề bảo mật, cần liên hệ riêng để download
---

## 📁 Cấu trúc thư mục

```
.
├── app.py
├── candidate_version3.py
├── candidate_version4.py
├── config.py
├── eval.py
├── features.py
├── final_submission.json
├── get_candidates.py
├── item_data.csv
├── main.py
├── params.json
├── prediction.json
├── recall_metrics.json
├── reranking_lasso.py
├── reranking_logistic.py
├── reranking.py
├── result.json
├── test_step_1.py
├── test.py
├── unavailable_items.txt
├── data/
│   └── table/
│       ├── item_data.parquet
│       └── ...
├── notebook/
│   ├── check_date_item.ipynb
│   ├── check_groundtruth_availability.ipynb
│   ├── evaluate.ipynb
│   └── test.ipynb
└── ...
```

---

## 🚦 Pipeline & Chiến thuật chi tiết

### 1️⃣ Tiền xử lý dữ liệu

- Đọc dữ liệu giao dịch, sản phẩm từ `.parquet`.
- Chuẩn hóa ngày tháng, loại bỏ các cột thiếu thông tin.

---

### 2️⃣ Sinh candidates (Ứng viên sản phẩm)

**Các chiến thuật chính:**

- **ALS Matrix Factorization (Collaborative Filtering)**
  - Sử dụng mô hình ALS (Alternating Least Squares) để học latent factors từ ma trận user-item, khai thác lịch sử mua hàng để đề xuất sản phẩm tương tự với hành vi người dùng.
  - Chạy batch trên GPU nếu có (tối ưu tốc độ).

- **BPR Ranking (Bayesian Personalized Ranking)**
  - Học thứ tự ưu tiên sản phẩm cho từng user dựa trên lịch sử tương tác, tối ưu hóa cho implicit feedback.

- **Item2Vec Embedding**
  - Sử dụng Word2Vec để học embedding cho sản phẩm dựa trên chuỗi mua hàng, từ đó tìm sản phẩm tương tự.

- **Content-based Filtering**
  - Sử dụng đặc trưng sản phẩm (brand, category, mô tả...) để tính toán độ tương đồng (cosine similarity, TF-IDF, FAISS GPU nếu có).
  - Đề xuất sản phẩm tương tự về nội dung với các sản phẩm user từng mua.

- **Trending Items**
  - Đề xuất các sản phẩm đang bán chạy nhất trong khoảng thời gian gần đây, có thể lọc theo năm/tháng để loại trừ hàng cũ hoặc lỗi thời.

- **Segment-based Fallback (Cold Start)**
  - Nếu user chưa đủ lịch sử, chia user thành các segment (theo giới tính, khu vực, membership...) và đề xuất các sản phẩm trending riêng cho từng segment.
  - Nếu vẫn thiếu, fallback về global trending.

- **Loại trừ sản phẩm đã mua và sản phẩm không khả dụng**
  - Không đề xuất lại sản phẩm user đã mua hoặc nằm trong [`unavailable_items.txt`](unavailable_items.txt).

- **Diversity bằng Maximal Marginal Relevance (MMR)**
  - Đảm bảo danh sách đề xuất đa dạng về ngành hàng, thương hiệu, không bị trùng lặp quá nhiều sản phẩm tương tự.

**Tham khảo code:**  
- [`candidate_version4.py`](candidate_version4.py) - Hàm `get_candidates`, `get_advanced_fallback`, `maximal_marginal_relevance`, v.v.

---

### 3️⃣ Trích xuất đặc trưng (Feature Engineering)

- Module [`features.py`](features.py) tính toán đặc trưng cho từng cặp khách hàng - sản phẩm:
  - **Tương đồng lịch sử mua hàng** (co-occurrence, brand, category).
  - **Giá, xu hướng giá, độ phổ biến**.
  - **Tần suất mua lại, velocity, repurchase rate**.
  - **Các đặc trưng embedding (nếu có)**.
  - **Contextual features**: ngày, mùa vụ, v.v.

---

### 4️⃣ Reranking (Sắp xếp lại)

- Sử dụng [`reranking.py`](reranking.py), [`reranking_lasso.py`](reranking_lasso.py), [`reranking_logistic.py`](reranking_logistic.py) để xếp hạng lại danh sách đề xuất dựa trên nhiều đặc trưng.
- Áp dụng penalty cho sản phẩm lệch giá, tăng diversity, ưu tiên sản phẩm mới, thưởng cho đúng brand/loại hàng user yêu thích.

---

### 5️⃣ Đánh giá

- Module [`eval.py`](eval.py) đánh giá chất lượng đề xuất theo Precision@10, loại trừ các sản phẩm đã mua trong lịch sử.
- So sánh với groundtruth thực tế.

---

### 6️⃣ Xuất kết quả

- Lưu kết quả cuối cùng ra [`final_submission.json`](final_submission.json) hoặc [`result.json`](result.json).

---

## 🛠️ Hướng dẫn chạy

1. **Cài đặt thư viện:**
    ```sh
    pip install -r requirements.txt
    ```
    (Nếu chưa có file `requirements.txt`, cài đặt: `polars`, `pandas`, `numpy`, `tqdm`, ...)

2. **Chạy pipeline chính:**
    ```sh
    python main.py
    ```

3. **Chạy giao diện kiểm thử:**
    ```sh
    streamlit run app.py
    ```

4. **Chỉnh sửa tham số:**  
   Thay đổi các mốc thời gian, số lượng candidate, v.v. trong [`params.json`](params.json).

---


## 📒 Notebook hỗ trợ & Phân tích

| Notebook | Mục đích |
|---|---|
| [notebook/check_date_item.ipynb](notebook/check_date_item.ipynb) | Kiểm tra, xử lý dữ liệu ngày tháng của sản phẩm, phát hiện lỗi thời gian. |
| [notebook/check_groundtruth_availability.ipynb](notebook/check_groundtruth_availability.ipynb) | Phân tích độ phủ groundtruth, kiểm tra tính khả dụng của groundtruth cho từng user/item. |
| [notebook/EDA1.ipynb](notebook/EDA1.ipynb) | Phân tích dữ liệu tổng quan (EDA), thống kê số lượng user, item, phân phối giao dịch. |
| [notebook/EDA2.ipynb](notebook/EDA2.ipynb) | Phân tích sâu hơn về hành vi người dùng, sản phẩm, xu hướng mua sắm. |
| [notebook/evaluate.ipynb](notebook/evaluate.ipynb) | Đánh giá offline các kết quả đề xuất, so sánh các chiến thuật candidate/rerank. |
| [notebook/recommendation-preprocess.ipynb](notebook/recommendation-preprocess.ipynb) | Tiền xử lý dữ liệu, chuẩn hóa, tạo các đặc trưng đầu vào cho pipeline. |
| [notebook/test.ipynb](notebook/test.ipynb) | Notebook kiểm thử nhanh các hàm, module trong pipeline. |

---

## 💡 Một số lưu ý chiến thuật

- **Loại trừ sản phẩm không khả dụng**: Đảm bảo không recommend hàng đã ngừng kinh doanh.
- **Tối ưu precision cho sản phẩm mới**: Chỉ tính điểm với sản phẩm chưa từng mua.
- **Đa dạng hóa đề xuất**: Không để 1 user nhận toàn sản phẩm cùng ngành hàng/brand.
- **Tối ưu tốc độ**: Dùng Polars cho xử lý dữ liệu lớn, chia batch khi sinh đặc trưng.

---

## 📂 Một số file quan trọng

- [`main.py`](main.py): Pipeline chính của hệ thống.
- [`config.py`](config.py): Định nghĩa các tham số cấu hình và hàm tạo query thời gian.
- [`features.py`](features.py): Trích xuất đặc trưng cho mô hình.
- [`candidate_version4.py`](candidate_version4.py): Sinh candidate sản phẩm.
- [`reranking.py`](reranking.py): Các hàm reranking kết hợp nhiều đặc trưng.
- [`eval.py`](eval.py): Đánh giá kết quả đề xuất.

---

## 🤝 Kết quả

- Đạt được precision@10 **4.96%** trên dữ liệu giao dịch tháng 2/2025

---

<p align="center">
  <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" width="80"/>
</p>
