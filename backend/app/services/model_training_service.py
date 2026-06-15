"""多模型训练服务（第10A 仅骨架，第11 阶段实现）。

本文件在第10A 仅提供训练任务编排骨架与契约，不执行真实多模型训练，
不暴露训练接口（路由在第11 阶段注册）。

第11 阶段计划任务（均严守 train 训练 / val 选模型 / test 仅最终评估）：
- 房价预测（监督）：median baseline / ridge / elasticnet / random forest /
  gradient boosting / hist gradient boosting / xgboost / lightgbm
  （xgboost、lightgbm 不可用时自动跳过并标 degraded=true + reason）。
- 项目类型识别（半监督）：以规则标签作 weak_label（weak_label=true，不伪装真标签）。
- 评分校准：基于 train/val 四维指标分布做 score calibration（不用 test）。
- 无监督聚类：POI 功能区 / 人口客群 / 房价分层 / 产业集聚 / 项目画像。

红线：不写假指标；样本不足写 degraded=true + 原因；test_used_for_training 恒为 false；
每个模型输出 data_usage_audit / model_card / training_log，结果可复现。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

PLANNED_TASKS: tuple[dict[str, Any], ...] = (
    {
        "model_name": "house_price_regressor",
        "task_type": "supervised_regression",
        "candidates": [
            "median_baseline", "ridge", "elasticnet", "random_forest",
            "gradient_boosting", "hist_gradient_boosting", "xgboost", "lightgbm",
        ],
        "test_used_for_training": False,
    },
    {
        "model_name": "project_type_weak_classifier",
        "task_type": "weak_supervised_classification",
        "candidates": ["decision_tree", "logistic_regression", "random_forest"],
        "weak_label": True,
        "test_used_for_training": False,
    },
    {
        "model_name": "score_calibrator",
        "task_type": "calibration",
        "test_used_for_training": False,
    },
    {
        "model_name": "unsupervised_clustering",
        "task_type": "clustering",
        "candidates": ["kmeans"],
        "targets": ["poi_zone", "population_segment", "housing_tier", "industry_cluster", "project_profile"],
        "test_used_for_training": False,
    },
)


def get_training_plan() -> dict[str, Any]:
    """返回第11 阶段训练计划契约（只读骨架，不训练）。"""
    return {
        "phase": "11 (planned)",
        "implemented": False,
        "degraded": True,
        "degraded_reason": "第10A 仅提供训练骨架；多模型训练在第11 阶段实现。",
        "test_used_for_training": False,
        "planned_tasks": list(PLANNED_TASKS),
        "policy": "train 训练 / val 选模型 / test 仅最终评估；不写假指标；样本不足标 degraded。",
    }


SUPERVISED_HOUSING_TASKS = frozenset({"housing_price_regression", "house_price_regressor"})


def guarded_training_entry(db: Session, request: dict[str, Any]) -> dict[str, Any]:
    """第11 真实训练的**强制入口护栏 + 分发**。

    任何监督训练在执行 fit() 之前**必须**先经过护栏：
    1. 房价监督训练（T3）→ housing_price_training_service.train（内部先 assert_training_allowed）；
    2. 其它任务尚未实现 → 仅跑护栏并返回未训练。
    """
    task = request.get("training_task", "housing_price_regression")
    if task in SUPERVISED_HOUSING_TASKS:
        from app.services import housing_price_training_service

        return housing_price_training_service.train(db, request)

    if task in {"project_type_classification", "project_type_weak_classifier"}:
        from app.services import project_type_training_service

        return project_type_training_service.train(db, request)

    if task in {"score_calibration", "score_calibrator"}:
        from app.services import score_calibration_service

        return score_calibration_service.train(db, request)

    from app.services import training_guard_service

    guard = training_guard_service.assert_training_allowed(db, request)
    return {
        "guard": guard,
        "trained": False,
        "reason": f"任务 {task} 的真实训练尚未实现（T4/T5 后续）；护栏已通过。",
    }
