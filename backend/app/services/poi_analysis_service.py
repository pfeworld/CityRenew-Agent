"""POI 区位配套分析（L 维度，第5阶段）。

目标：基于项目三圈层统计 POI 配套与功能短板，产出可解释的 L_score。

口径：
- category_name 为 "一级;二级;三级" 层级串，按**一级类目**归并为四大功能组：
  商业服务 / 公共服务 / 便民生活 / 交通。
- 圈层为累计圆（core ⊆ nearby ⊆ radiation），与 spatial_service 一致。
- 默认仅 train/val；include_test=true 才纳入 test。

红线：仅输出统计量/分类/评分/置信度/notes/evidence_id；
不返回 POI 名称、地址、坐标、raw_json。
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import analysis_common as ac
from app.services import spatial_service

logger = logging.getLogger("cityrenew.analysis.poi")

DIMENSION = "poi"

# 一级类目 -> 四大功能组
GROUP_COMMERCIAL = "commercial"
GROUP_PUBLIC = "public"
GROUP_CONVENIENCE = "convenience"
GROUP_TRANSPORT = "transport"
GROUP_OTHER = "other"

FIRST_LEVEL_GROUP: dict[str, str] = {
    "购物服务": GROUP_COMMERCIAL,
    "餐饮服务": GROUP_COMMERCIAL,
    "住宿服务": GROUP_COMMERCIAL,
    "商务住宅": GROUP_COMMERCIAL,
    "金融保险服务": GROUP_COMMERCIAL,
    "医疗保健服务": GROUP_PUBLIC,
    "科教文化服务": GROUP_PUBLIC,
    "政府机构及社会团体": GROUP_PUBLIC,
    "公共设施": GROUP_PUBLIC,
    "体育休闲服务": GROUP_PUBLIC,
    "生活服务": GROUP_CONVENIENCE,
    "交通设施服务": GROUP_TRANSPORT,
    "通行设施": GROUP_TRANSPORT,
    "道路附属设施": GROUP_TRANSPORT,
}

# 短板候选类别（一级类目）-> 推荐补充业态文案（脱敏，类别级，非具体商户）
SHORTBOARD_CANDIDATES: dict[str, str] = {
    "医疗保健服务": "社区卫生服务/诊所/药房",
    "科教文化服务": "幼托/教育培训/文化场馆",
    "购物服务": "综合商业/便利零售",
    "餐饮服务": "餐饮配套/特色餐饮",
    "交通设施服务": "公共停车/公交接驳",
    "生活服务": "家政/维修/便民服务点",
    "体育休闲服务": "健身/运动/休闲场所",
    "公共设施": "公共服务设施/便民驿站",
}

# L_score 经验上界（每 km² POI 密度归一化上界）
DENSITY_HI_PER_KM2 = 60.0


def _first_level(category_name: str | None) -> str | None:
    if not category_name:
        return None
    return category_name.split(";")[0].strip() or None


def _group_of(first_level: str | None) -> str:
    if first_level is None:
        return GROUP_OTHER
    return FIRST_LEVEL_GROUP.get(first_level, GROUP_OTHER)


def _ring_stat(rows: list[Any], radius_m: int, ring: str) -> dict[str, Any]:
    group_counts = Counter()
    first_counts = Counter()
    for r in rows:
        fl = _first_level(r.category_name)
        first_counts[fl or "未分类"] += 1
        group_counts[_group_of(fl)] += 1
    mix = ac.normalized_entropy(
        [group_counts.get(g, 0) for g in (GROUP_COMMERCIAL, GROUP_PUBLIC, GROUP_CONVENIENCE, GROUP_TRANSPORT)]
    )
    return {
        "ring": ring,
        "radius_m": radius_m,
        "total": len(rows),
        "commercial": group_counts.get(GROUP_COMMERCIAL, 0),
        "public": group_counts.get(GROUP_PUBLIC, 0),
        "convenience": group_counts.get(GROUP_CONVENIENCE, 0),
        "transport": group_counts.get(GROUP_TRANSPORT, 0),
        "other": group_counts.get(GROUP_OTHER, 0),
        "mix_index": mix,
        "_first_counts": first_counts,
    }


def analyze(db: Session, project: Project, include_test: bool = False) -> dict[str, Any]:
    collected = spatial_service.collect_ring_records(db, project, "poi", include_test)
    radii = collected["radii"]
    records = collected["records"]

    ring_stats = {
        ring: _ring_stat(records[ring], radii[ring], ring) for ring in ac.RING_ORDER
    }
    radiation = ring_stats[ac.RING_RADIATION]
    nearby = ring_stats[ac.RING_NEARBY]

    # ---- 配套短板 & 推荐 ----
    rad_first: Counter = radiation["_first_counts"]
    shortboards = sorted(
        SHORTBOARD_CANDIDATES.keys(),
        key=lambda c: rad_first.get(c, 0),
    )[:5]
    recommend = [SHORTBOARD_CANDIDATES[c] for c in shortboards]

    # ---- 评分（可解释经验权重）----
    rad_area = spatial_service.ring_area_km2(radii[ac.RING_RADIATION])
    density = ac.safe_div(radiation["total"], rad_area) or 0.0
    density_score = ac.minmax_score(density, 0.0, DENSITY_HI_PER_KM2)
    present = sum(1 for c in SHORTBOARD_CANDIDATES if rad_first.get(c, 0) > 0)
    coverage_score = present / len(SHORTBOARD_CANDIDATES) * 100.0
    mix_score = nearby["mix_index"] * 100.0
    accessibility = ac.safe_div(nearby["total"], radiation["total"])
    accessibility_score = (accessibility or 0.0) * 100.0

    score = ac.clamp(
        0.35 * density_score
        + 0.30 * coverage_score
        + 0.20 * mix_score
        + 0.15 * accessibility_score
    )

    confidence = round(min(1.0, radiation["total"] / 100.0), 3)
    notes: list[str] = [
        "圈层为累计圆（核心⊆近邻⊆辐射）；POI 一级类目归并为商业/公共/便民/交通四组。",
    ]
    if radiation["total"] == 0:
        notes.append("辐射范围内无 train/val POI 记录，评分置信度低（可能 splits 未构建或范围内无数据）。")
    if collected["skipped_no_coord"]:
        notes.append(f"{collected['skipped_no_coord']} 条 POI 因坐标不可用被跳过。")

    # ---- 落库（AnalysisResult + EvidenceChain）----
    ac.clear_dimension_results(db, project.id, DIMENSION)
    evidence_ids: list[str] = []
    for ring in ac.RING_ORDER:
        st = ring_stats[ring]
        for key in ("total", "commercial", "public", "convenience", "transport"):
            evidence_ids.append(
                ac.record_metric(
                    db, project_id=project.id, dimension=DIMENSION, ring=ring,
                    metric_key=f"poi_{key}", value=float(st[key]), unit="个",
                    summary=f"{ring} 圈层 POI {key} 计数",
                    confidence=confidence,
                    metadata={"allowed_splits": collected["allowed_splits"]},
                )
            )
        evidence_ids.append(
            ac.record_metric(
                db, project_id=project.id, dimension=DIMENSION, ring=ring,
                metric_key="poi_mix_index", value=float(st["mix_index"]), unit="ratio",
                summary=f"{ring} 圈层功能混合度（归一化香农熵）",
                confidence=confidence,
            )
        )
    evidence_ids.append(
        ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="L_score", value=float(round(score, 2)), unit="score",
            summary="区位配套 L 维度评分（密度/覆盖/混合/可达 可解释加权）",
            confidence=confidence,
            metadata={"shortboards_top5": shortboards},
        )
    )
    db.commit()

    rings_out = [
        {k: v for k, v in ring_stats[ring].items() if not k.startswith("_")}
        for ring in ac.RING_ORDER
    ]
    category_top = dict(rad_first.most_common(15))
    category_top.pop("未分类", None)

    logger.info(
        "poi analyze project_id=%s L_score=%.1f total_rad=%s splits=%s",
        project.id, score, radiation["total"], collected["allowed_splits"],
    )

    return {
        "project_id": project.id,
        "dimension": DIMENSION,
        "score": round(score, 2),
        "confidence": confidence,
        "allowed_splits": collected["allowed_splits"],
        "include_test": include_test,
        "used_test": include_test,
        "center_status": collected["center_status"],
        "evidence_ids": evidence_ids,
        "notes": notes,
        "rings": rings_out,
        "category_top": category_top,
        "shortboards_top5": shortboards,
        "recommend_top5": recommend,
    }
