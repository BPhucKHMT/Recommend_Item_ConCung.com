
---

## 🚦 Pipeline & Chiến thuật

### 1️⃣ Tiền xử lý dữ liệu

- Đọc dữ liệu giao dịch, sản phẩm từ `.csv`/`.parquet`.
- Chuẩn hóa ngày tháng, loại bỏ các bản ghi thiếu thông tin.
- Lọc các sản phẩm không khả dụng dựa trên [`unavailable_items.txt`](unavailable_items.txt).

### 2️⃣ Sinh candidates (Ứng viên sản phẩm)

- Sử dụng [`candidate_version4.py`](candidate_version4.py) để sinh danh sách sản phẩm tiềm năng cho từng khách hàng.
- Áp dụng các chiến thuật:
  - **Lọc sản phẩm mới** (theo năm tạo, loại trừ hàng cũ).
  - **Đa dạng hóa**: Ưu tiên sản phẩm đa dạng ngành hàng, thương hiệu.
  - **Chỉ lấy sản phẩm active** nếu cần.

### 3️⃣ Trích xuất đặc trưng

- Module [`features.py`](features.py) tính toán đặc trưng cho từng cặp khách hàng - sản phẩm:
  - **Tương đồng lịch sử mua hàng** (co-occurrence, brand, category).
  - **Giá, xu hướng giá, độ phổ biến**.
  - **Tần suất mua lại, velocity, repurchase rate**.
  - **Các đặc trưng embedding (nếu có)**.

### 4️⃣ Reranking (Sắp xếp lại)

- Sử dụng [`reranking.py`](reranking.py), [`reranking_lasso.py`](reranking_lasso.py), [`reranking_logistic.py`](reranking_logistic.py) để xếp hạng lại danh sách đề xuất dựa trên nhiều đặc trưng.
- Áp dụng penalty cho sản phẩm lệch giá, tăng diversity, ưu tiên sản phẩm mới.

### 5️⃣ Đánh giá

- Module [`eval.py`](eval.py) đánh giá chất lượng đề xuất theo Recall@K, loại trừ các sản phẩm đã mua trong lịch sử.
- So sánh với groundtruth thực tế.

### 6️⃣ Xuất kết quả

- Lưu kết quả cuối cùng ra [`final_submission.json`](final_submission.json) hoặc [`result.json`](result.json).

---

## 🛠️ Hướng dẫn chạy

1. **Cài đặt thư viện:**
    ```sh
    pip install -r requirements.txt
    ```
    (Nếu chưa có file `requirements.txt`, cài đặt: [polars](http://_vscodecontentref_/16), [pandas](http://_vscodecontentref_/17), [numpy](http://_vscodecontentref_/18), [tqdm](http://_vscodecontentref_/19), ...)

2. **Chạy pipeline chính:**
    ```sh
    python main.py
    ```

3. **Chạy giao diện kiểm thử:**
    ```sh
    streamlit run app.py
    ```

4. **Chỉnh sửa tham số:**  
   Thay đổi các mốc thời gian, số lượng candidate, v.v. trong [params.json](http://_vscodecontentref_/20).

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

> 📌 **Lưu ý:** Các notebook này giúp kiểm tra, phân tích dữ liệu, debug pipeline và đánh giá hiệu quả các chiến thuật đề xuất.

---



## 🤝 Đóng góp & Liên hệ

- Nếu có thắc mắc hoặc cần hỗ trợ, vui lòng liên hệ nhóm phát triển qua GitHub Issues hoặc email.

---

<p align="center">
  <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" width="80"/>
</p>