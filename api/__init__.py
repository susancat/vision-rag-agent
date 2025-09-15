# api/__init__.py
"""
Vision-RAG Agent - API package
包含 ingestion、query 等模組。
"""

__version__ = "0.1.0"

from .query import ask   # 讓外部可以直接 import api.ask
