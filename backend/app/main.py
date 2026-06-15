"""CityRenew Agent 后端入口（FastAPI）。

第1阶段：项目骨架 + 数据库初始化 + 健康检查/系统信息接口。
第2阶段：新增本地资料导入、字段自适应与 split_manager（ingestion / splits 接口）。
第3阶段：新增 RAG 知识库与证据链（rag / evidence 接口），本地 BM25 检索。
第4阶段：新增项目管理与空间圈层分析（projects / spatial 接口），Haversine 归集。
第5阶段：新增四维核心分析（analysis 接口：POI/人口/房价/产业 + 一键四维 + 汇总）
与房价基线模型（仅 train/val 训练验证，test 不参与）。
不含 报告生成 / 外部大模型调用。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    agent_router,
    analysis_router,
    evaluation_router,
    evidence_router,
    external_router,
    features_router,
    health_router,
    ingestion_router,
    internal_router,
    models_router,
    projects_router,
    rag_router,
    report_router,
    reports_router,
    spatial_router,
    splits_router,
    system_router,
)
from app.config import settings
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时建表（仅创建 schema，不灌入任何语料数据）
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="城市更新前期策划智能体 - 后端服务（第5阶段：四维核心分析与房价基线模型）",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(system_router)
app.include_router(ingestion_router)
app.include_router(splits_router)
app.include_router(rag_router)
app.include_router(evidence_router)
app.include_router(projects_router)
app.include_router(spatial_router)
app.include_router(analysis_router)
app.include_router(evaluation_router)
app.include_router(reports_router)
app.include_router(report_router)
app.include_router(internal_router)
app.include_router(features_router)
app.include_router(external_router)
app.include_router(models_router)
app.include_router(agent_router)


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "system": settings.app_name,
        "version": settings.app_version,
        "mode": settings.app_mode,
        "docs": "/docs",
        "health": "/health",
        "system_info": "/api/system/info",
    }
