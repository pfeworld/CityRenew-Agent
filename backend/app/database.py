"""数据库连接与会话管理（SQLAlchemy + SQLite）。

第1阶段：仅建立 engine / session / Base 与建表能力，不灌入任何语料数据。
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def _build_engine():
    url = settings.database_url
    connect_args: dict = {}
    if url.startswith("sqlite"):
        # SQLite 在多线程（FastAPI）下需要关闭同线程检查
        connect_args = {"check_same_thread": False}
        # 确保 sqlite 文件所在目录存在
        if ":///" in url:
            db_path = url.split(":///", 1)[1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args=connect_args, future=True)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


# 第3阶段：对第1阶段已建的空表做幂等补列（SQLite 不支持 create_all 自动加列）。
# 仅新增列，不删除/改名，保留第2阶段已导入的结构化数据与 split。
_ENSURE_COLUMNS: dict[str, dict[str, str]] = {
    # 第4阶段：为已建的 projects 表补齐项目管理新增列（仅新增，不删/改）
    "projects": {
        "city": "VARCHAR(128)",
        "street": "VARCHAR(128)",
        "address": "VARCHAR(512)",
        "land_use": "VARCHAR(128)",
        "project_area": "FLOAT",
        "building_area": "FLOAT",
        "build_year": "INTEGER",
        "update_demand": "TEXT",
        "expected_direction": "TEXT",
    },
    "knowledge_chunks": {
        "source_type": "VARCHAR(64)",
        "section": "VARCHAR(255)",
        "page_no": "INTEGER",
        "chunk_text": "TEXT",
        "chunk_summary": "TEXT",
        "keywords": "TEXT",
        "metadata_json": "TEXT",
        "is_sensitive": "BOOLEAN",
    },
    "evidence_chains": {
        "confidence": "FLOAT",
        "metadata_json": "TEXT",
    },
}


def _ensure_columns() -> None:
    """为已存在的表补齐缺失列（幂等，仅 SQLite 开发库）。"""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in _ENSURE_COLUMNS.items():
            if table not in existing_tables:
                continue
            present = {col["name"] for col in inspector.get_columns(table)}
            for col_name, col_type in columns.items():
                if col_name not in present:
                    conn.execute(
                        text(f'ALTER TABLE {table} ADD COLUMN {col_name} {col_type}')
                    )


def init_db() -> None:
    """创建全部表（若不存在），并对已存在的表做幂等补列。

    导入 models 以确保所有 ORM 类注册到 Base.metadata。
    """
    from app import models  # noqa: F401  (触发模型注册)

    Base.metadata.create_all(bind=engine)
    _ensure_columns()


def get_db() -> Generator:
    """FastAPI 依赖：提供数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """探测数据库连通性，供 /api/system/info 使用。"""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
