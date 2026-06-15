"""DataFile 资料文件元信息表（第2阶段导入时填充）。

仅记录文件级元信息（文件名 / 类型 / 记录数 / hash / split 概要），
不存储语料原文，符合涉密保护原则。
"""

from __future__ import annotations

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class DataFile(Base, TimestampMixin):
    __tablename__ = "data_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # 数据类型：poi / house_price / population / industry / policy / case / template ...
    data_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 来源：reference（参考资料） / corpus（训练语料）
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=True)
    # train/val/test 概要（JSON 字符串），第2阶段由 split_manager 写入
    split_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
