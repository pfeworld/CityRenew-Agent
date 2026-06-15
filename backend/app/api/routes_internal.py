"""第12G：内部能力接口（仅供开发 / 测试，不在普通前台展示）。

POST /api/internal/regression/run-case-tests   基于华建 / 鲁商1992 案例做输入输出回归测试
GET  /api/internal/case-corpus/status           案例语料加载状态（统计量，不含原文）

红线：内部使用；不暴露语料原文；仅 train/val；不调外部 API；不使用大模型生成事实数字。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.services import case_learning_service as cases
from app.services import regression_service as reg

router = APIRouter(prefix="/api/internal", tags=["internal"])


@router.post("/regression/run-case-tests")
def run_case_tests(project_id: int = Query(default=1), db: Session = Depends(get_db)) -> dict:
    return reg.run_case_regression(db, project_id)


@router.get("/case-corpus/status")
def case_corpus_status() -> dict:
    return cases.case_corpus_status()
