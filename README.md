# recommendation_system

Hệ thống recommendation này được xây dựng để đề xuất sản phẩm cá nhân hóa cho khách hàng dựa trên lịch sử giao dịch, thông tin sản phẩm và các đặc trưng hành vi.

## Cấu trúc thư mục

```
recommendation_system/
│
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
│       ├── item_data.csv
│       └── ... (chỉ sử dụng các file trong data/table, không lấy toàn bộ thư mục data)
├── notebook/
│   ├── check_date_item.ipynb
│   ├── check_groundtruth_availability.ipynb
│   ├── evaluate.ipynb
│   └── test.ipynb
└── __pycache__/
```

## Quy trình chính

1. **Tiền xử lý dữ liệu**: Đọc dữ liệu giao dịch, dữ liệu sản phẩm từ các file `.csv` hoặc `.parquet`.
2. **Sinh candidates**: Sử dụng các module `candidate_version3.py` hoặc `candidate_version4.py` để sinh ra danh sách sản phẩm tiềm năng cho từng khách hàng.
3. **Trích xuất đặc trưng**: Module `features.py` dùng để tính toán các đặc trưng cho từng cặp khách hàng - sản phẩm.
4. **Reranking**: Sắp xếp lại danh sách đề xuất dựa trên các mô hình reranking trong `reranking.py`, `reranking_lasso.py`, `reranking_logistic.py`.
5. **Đánh giá**: Sử dụng `eval.py` để đánh giá chất lượng đề xuất.
6. **Kết quả cuối cùng**: Lưu kết quả ra file `final_submission.json`.

## Hướng dẫn chạy

1. Cài đặt các thư viện cần thiết:
	```sh
	pip install -r requirements.txt
	```
	(Nếu chưa có file `requirements.txt`, bạn có thể cần cài đặt các thư viện như: polars, pandas, numpy, tqdm, ...)

2. Chạy pipeline chính:
	```sh
	python main.py
	```

3. Các tham số cấu hình được lưu trong `params.json`. Bạn có thể chỉnh sửa file này để thay đổi các mốc thời gian, số lượng candidate, v.v.

## Một số file quan trọng

- `main.py`: Pipeline chính của hệ thống.
- `config.py`: Định nghĩa các tham số cấu hình và hàm tạo query thời gian.
- `features.py`: Trích xuất đặc trưng cho mô hình.
- `candidate_version4.py`: Sinh candidate sản phẩm.
- `reranking.py`: Các hàm reranking kết hợp nhiều đặc trưng.
- `eval.py`: Đánh giá kết quả đề xuất.

## Notebook hỗ trợ

- `notebook/check_date_item.ipynb`: Kiểm tra dữ liệu ngày tháng của sản phẩm.
- `notebook/evaluate.ipynb`: Đánh giá offline các kết quả đề xuất.

## Liên hệ

Nếu có thắc mắc hoặc cần hỗ trợ, vui lòng liên hệ nhóm phát triển.

---