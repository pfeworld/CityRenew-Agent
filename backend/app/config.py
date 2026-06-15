"""应用配置。

通过 pydantic-settings 从环境变量 / .env 读取配置。
关键默认值对齐 docs/11 关键决策记录：
- APP_MODE 默认 eval（优先保证 train/val/test 隔离与自评可信度）。
- 圈层口径以报告模板为准：近邻 500m / 辐射 1500m（D4）。
- coordinate_system 预留，默认 WGS84（待第2/4阶段确认，D1）。
- 外部大模型第1阶段不启用，默认 mock 模式（D3）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "cityrenew.db"
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "训练语料"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "参考资料"


class Settings(BaseSettings):
    """全局配置项。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 应用基础 ----
    app_name: str = "CityRenew Agent"
    app_version: str = "0.1.0"
    app_env: str = "development"
    # 运行模式：eval（默认，严格 test 隔离） / demo（演示，可选）
    app_mode: str = "eval"

    # ---- 服务 ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # 允许的前端来源（CORS），逗号分隔
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---- 数据库 ----
    database_url: str = f"sqlite:///{DEFAULT_DB_PATH}"

    # ---- 本地涉密语料目录（仅本地读取，不入库、不外发；对齐 docs/06）----
    corpus_dir: str = str(DEFAULT_CORPUS_DIR)
    # 参考资料目录（政策/模板/脱敏报告/口径说明，第3阶段 RAG 知识源）
    reference_dir: str = str(DEFAULT_REFERENCE_DIR)
    # 第12G：SC 正式材料目录（报告模版.docx / 华建案例 / 鲁商1992 案例）。
    # 仅供智能体内部解析、样例学习与回归测试；已 gitignore，不入库、不外发。
    sc_dir: str = str(PROJECT_ROOT / "SC")

    # ---- 第3阶段 RAG ----
    # 接口/前端返回的片段最大字数（涉密：不返回原文整段）
    rag_snippet_max_chars: int = 120
    # 摘要最大字数
    rag_summary_max_chars: int = 160
    # 默认检索返回条数
    rag_default_top_k: int = 5
    # chunk 切块目标字数与重叠
    rag_chunk_max_chars: int = 500
    rag_chunk_overlap_chars: int = 80

    # ---- 数据集切分（第2阶段）----
    # POI/产业空间整组切分的网格 cell 尺寸（度）；~0.002°≈200m，防近邻泄露
    split_cell_size_deg: float = 0.002

    # ---- 坐标系与圈层口径（对齐 docs/11 D1/D4）----
    coordinate_system: str = "WGS84"
    nearby_buffer_m: int = 500
    radiation_buffer_m: int = 1500
    # 第4阶段：core_buffer_m=0 且无红线时的核心圈兜底半径（米）
    default_core_buffer_m: int = 150

    # ---- 外部大模型（第1-2阶段不启用，仅预留；对齐 docs/11 D3）----
    # 默认 mock：不调用任何外部 API，不读取语料，不生成事实数字。
    llm_provider: str = "mock"
    llm_base_url: str = ""
    llm_api_key: str = ""

    # ---- 第12F：DeepSeek 大模型思考层（key 仅从 .env 读取，从不写死/外发）----
    # 双模型：常规对话用 flash（快），深度思考用 pro（推理模型，慢但更深）。
    # 仅负责语言组织表达，事实数字一律来自自研模型结构化结果，不得编造。
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model_chat: str = "deepseek-v4-flash"
    deepseek_model_think: str = "deepseek-v4-pro"
    # 兼容旧单模型配置（仅当未提供 chat/think 时回退使用）
    deepseek_model: str = ""
    deepseek_thinking_enabled: bool = False
    deepseek_request_timeout_s: int = 60

    # ---- 第10B：合规外部数据源 API Key（一律从 .env 读取，不写入代码）----
    # 缺失时对应采集接口返回 not_configured，绝不伪造数据。
    amap_key: str = ""
    baidu_map_key: str = ""
    tencent_map_key: str = ""
    # 高德采集限流与配额保护（不绕配额、不突破分页；样例采集上限）
    amap_request_timeout_s: int = 8
    amap_rate_limit_qps: float = 3.0
    amap_max_pages: int = 3
    amap_page_size: int = 20

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ---- 数据派生目录（均在 backend/data/ 下，已 gitignore）----
    @property
    def corpus_path(self) -> Path:
        return Path(self.corpus_dir)

    @property
    def reference_path(self) -> Path:
        return Path(self.reference_dir)

    @property
    def sc_path(self) -> Path:
        return Path(self.sc_dir)

    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def quality_report_path(self) -> Path:
        return self.processed_dir / "quality_report.json"

    @property
    def split_manifest_path(self) -> Path:
        return self.splits_dir / "split_manifest.json"

    @property
    def bm25_index_path(self) -> Path:
        return self.index_dir / "bm25_index.pkl"

    @property
    def chunks_meta_path(self) -> Path:
        return self.index_dir / "chunks_meta.json"

    @property
    def data_security_notice(self) -> str:
        return (
            "本系统遵循涉密保护、测试集隔离、禁止编造数据三条红线："
            "参考资料/训练语料及数据派生目录不入库不外发；"
            "test split 仅用于评估；所有数字均来自本地确定性计算。"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
