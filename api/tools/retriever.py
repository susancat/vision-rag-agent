# api/tools/retriever.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Optional

import faiss
import numpy as np


class Retriever:
    """
    輕量檢索器（FAISS + metadata）
    - 支援文字 / 影像雙索引
    - 友善錯誤訊息（索引不存在 / 空索引）
    - 自動修正 k 值、查詢向量 shape/dtype
    - 支援基於 metadata 的過濾（predicate）
    """

    def __init__(self, base_dir: str):
        self.base = Path(base_dir)

        # --- 安全載入 ---
        self.text_idx = self._load_faiss(self.base / "text.faiss")
        self.img_idx = self._load_faiss(self.base / "image.faiss")

        self.text_meta = self._load_json(self.base / "text_meta.json")
        self.img_meta = self._load_json(self.base / "image_meta.json")

        # 一致性檢查（非致命）
        if self.text_idx is not None and len(self.text_meta) != self.text_idx.ntotal:
            print(f"[warn] text_meta({len(self.text_meta)}) != text_idx.ntotal({self.text_idx.ntotal})")
        if self.img_idx is not None and len(self.img_meta) != self.img_idx.ntotal:
            print(f"[warn] image_meta({len(self.img_meta)}) != image_idx.ntotal({self.img_idx.ntotal})")

    # ---------- public API ----------
    def search_text(
        self,
        q_vec: np.ndarray,
        k: int = 5,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        文字索引檢索。
        - q_vec: shape 可為 (d,) 或 (1, d)，dtype 任意；內部會轉成 (1, d) float32
        - predicate(meta) -> bool：可選過濾，例如只看某來源
        """
        if self.text_idx is None or self.text_idx.ntotal == 0:
            return []
        qv = self._prep_query(q_vec, self.text_idx.d)
        k = int(min(max(k, 1), self.text_idx.ntotal))

        D, I = self.text_idx.search(qv, k)
        pairs = [(float(D[0][i]), self.text_meta[int(I[0][i])]) for i in range(len(I[0]))]
        if predicate:
            pairs = [p for p in pairs if predicate(p[1])]
        return pairs

    def search_image(
        self,
        q_vec: np.ndarray,
        k: int = 5,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        影像索引檢索；介面同上。
        """
        if self.img_idx is None or self.img_idx.ntotal == 0:
            return []
        qv = self._prep_query(q_vec, self.img_idx.d)
        k = int(min(max(k, 1), self.img_idx.ntotal))

        D, I = self.img_idx.search(qv, k)
        pairs = [(float(D[0][i]), self.img_meta[int(I[0][i])]) for i in range(len(I[0]))]
        if predicate:
            pairs = [p for p in pairs if predicate(p[1])]
        return pairs

    # ---------- helpers ----------
    @staticmethod
    def _load_faiss(path: Path) -> Optional[faiss.Index]:
        if not path.exists():
            # 不拋錯，讓上層能顯示友善提示
            print(f"[warn] FAISS index not found: {path}")
            return None
        try:
            idx = faiss.read_index(str(path))
            return idx
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index {path}: {e}")

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            print(f"[warn] Meta not found: {path}")
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata {path}: {e}")

    @staticmethod
    def _prep_query(q_vec: np.ndarray, dim: int) -> np.ndarray:
        """
        將輸入向量轉成 (1, dim) 的 float32；若維度不匹配，拋出清楚錯誤。
        """
        q = np.asarray(q_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != dim:
            raise ValueError(f"Query dim {q.shape[1]} != index dim {dim}")
        # 假設外部已 normalize；保守起見再做一次 L2 normalize（對 inner product 也常見）
        faiss.normalize_L2(q)
        return q
