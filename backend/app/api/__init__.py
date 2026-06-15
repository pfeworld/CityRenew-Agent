"""API 路由包。"""

from app.api.health import router as health_router
from app.api.routes_agent import router as agent_router
from app.api.routes_analysis import router as analysis_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_evidence import router as evidence_router
from app.api.routes_external import router as external_router
from app.api.routes_features import router as features_router
from app.api.routes_ingestion import router as ingestion_router
from app.api.routes_internal import router as internal_router
from app.api.routes_models import router as models_router
from app.api.routes_projects import router as projects_router
from app.api.routes_rag import router as rag_router
from app.api.routes_report import router as report_router
from app.api.routes_reports import router as reports_router
from app.api.routes_spatial import router as spatial_router
from app.api.routes_splits import router as splits_router
from app.api.system import router as system_router

__all__ = [
    "health_router",
    "agent_router",
    "system_router",
    "ingestion_router",
    "splits_router",
    "rag_router",
    "evidence_router",
    "projects_router",
    "spatial_router",
    "analysis_router",
    "evaluation_router",
    "reports_router",
    "report_router",
    "internal_router",
    "features_router",
    "external_router",
    "models_router",
]
