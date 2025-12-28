import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

@dataclass
class AppConfig:
    date_col: str
    anchor_date: str
    len_hist: int
    len_recent: int
    len_val: int
    len_test: int
    N_trend: int = 100
    N_cand: int = 20
    session_window: int = 1
    min_coo: int = 1

    def get_anchor_date(self):
        """Chuyển string sang date object"""
        return datetime.strptime(self.anchor_date, "%Y-%m-%d").date()

    def create_query_string(self) -> Dict[str, str]:
        """
        Tạo câu query filter cho Polars & IN RA MÀN HÌNH ĐỂ DEBUG.
        """
        anchor = self.get_anchor_date()
        
        # 1. Tính toán các mốc cắt (Cut-off dates)
        # Test: [Anchor - len_test, Anchor]
        threshold_test = anchor - timedelta(days=self.len_test)
        
        # Validation: [threshold_test - len_val, threshold_test]
        threshold_val = threshold_test - timedelta(days=self.len_val)
        
        # Recent: [threshold_val - len_recent, threshold_val]
        threshold_rec = threshold_val - timedelta(days=self.len_recent)
        
        # History Start: [threshold_rec - len_hist]
        threshold_hist = threshold_rec - timedelta(days=self.len_hist)

        # 2. Tạo Query String
        c = f'"{self.date_col}"'
        
        queries = {
            "test":    f"{c} > date('{threshold_test}') AND {c} <= date('{anchor}')",
            "val":     f"{c} > date('{threshold_val}') AND {c} <= date('{threshold_test}')",
            "recent":  f"{c} > date('{threshold_rec}') AND {c} <= date('{threshold_val}')",
            # [LƯU Ý] History bao trùm cả Recent (Kết thúc tại threshold_val)
            "history": f"{c} > date('{threshold_hist}') AND {c} <= date('{threshold_val}')",
            "val_raw_date": threshold_val # Dùng để sampling nếu cần
        }

        # --- 3. IN DEBUG RA MÀN HÌNH ---
        print("\n" + "="*50)
        print(f"🕒 TIME SPLIT DEBUG (Anchor: {anchor})")
        print("="*50)
        print(f"1. TEST RANGE:   (> {threshold_test}) --> (<= {anchor})")
        print(f"2. VAL RANGE:    (> {threshold_val}) --> (<= {threshold_test})")
        print("-" * 50)
        print(f"3. RECENT RANGE: (> {threshold_rec}) --> (<= {threshold_val})")
        print(f"4. HIST RANGE:   (> {threshold_hist}) --> (<= {threshold_val})")
        print("   ✅ (Logic Check): History kết thúc cùng ngày với Recent.")
        print("="*50 + "\n")

        return queries

def load_config(file_path: str = "params.json") -> AppConfig:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    valid_keys = {k: v for k, v in data.items() if k in AppConfig.__annotations__}
    return AppConfig(**valid_keys)