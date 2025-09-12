from enum import Enum
from pydantic import BaseModel

class TaskType(str, Enum):
    TEXT_QA = "text_qa"
    VISION_QA = "vision_qa"
    TABLE_TO_CSV = "table_to_csv"
    CALC = "calc"

class Plan(BaseModel):
    task: TaskType
    steps: list[str]

KEYWORDS_VISION = ["圖", "figure", "arrow", "此圖", "上圖", "flow"]
KEYWORDS_TABLE = ["表格", "table", "csv", "欄位", "規格"]
KEYWORDS_CALC = ["加總", "平均", "mm", "換算", "數值"]

def router(question: str) -> Plan:
    ql = question.lower()
    if any(k in ql for k in ["figure", "diagram", "image"]) or any(k in question for k in KEYWORDS_VISION):
        return Plan(task=TaskType.VISION_QA, steps=["retrieve_image", "vision_parse", "synthesize", "cite"])
    if any(k in ql for k in ["table", "csv"]) or any(k in question for k in KEYWORDS_TABLE):
        return Plan(task=TaskType.TABLE_TO_CSV, steps=["table_extract", "synthesize", "cite"])
    if any(k in ql for k in ["avg", "sum"]) or any(k in question for k in KEYWORDS_CALC):
        return Plan(task=TaskType.CALC, steps=["retrieve", "python_calc", "synthesize", "cite"])
    return Plan(task=TaskType.TEXT_QA, steps=["retrieve", "synthesize", "cite"])
