"""ORM 模型汇总导出。

第1阶段建立 10 张基础表，仅建表预留字段，不灌入语料数据：
Project, DataFile, KnowledgeChunk, PoiPoint, PopulationProfile,
HousingRecord, IndustryPoint, AnalysisResult, EvidenceChain, EvaluationResult。
"""

from app.models.analysis_result import AnalysisResult
from app.models.data_file import DataFile
from app.models.evaluation_result import EvaluationResult
from app.models.evidence_chain import EvidenceChain
from app.models.housing_record import HousingRecord
from app.models.industry_point import IndustryPoint
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.poi_point import PoiPoint
from app.models.population_profile import PopulationProfile
from app.models.project import Project
from app.models.project_feature import ProjectFeature

__all__ = [
    "Project",
    "DataFile",
    "KnowledgeChunk",
    "PoiPoint",
    "PopulationProfile",
    "HousingRecord",
    "IndustryPoint",
    "AnalysisResult",
    "EvidenceChain",
    "EvaluationResult",
    "ProjectFeature",
]
