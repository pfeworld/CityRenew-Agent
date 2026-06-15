"""项目管理服务（第4阶段）。

职责：项目的创建 / 列表 / 详情 / 更新 / 删除（CRUD）。
- 中心点经纬度按 D1 校验（[lng, lat]，上海范围，自动纠偏）。
- 不写死数据源格式；圈层口径缺省取 config（核心兜底 150 / 近邻 500 / 辐射 1500）。
- 仅处理项目自身录入字段，不触碰语料原文、不返回 raw_json。
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Project
from app.schemas.project import ProjectCreate, ProjectOut, ProjectUpdate
from app.utils import geo_utils

logger = logging.getLogger("cityrenew.project")


def _to_out(project: Project) -> ProjectOut:
    """ORM -> 响应模型；同时填充常用别名（project_name / lon / lat）。"""
    return ProjectOut(
        id=project.id,
        name=project.name,
        project_name=project.name,
        city=project.city,
        district=project.district,
        street=project.street,
        address=project.address,
        description=project.description,
        center_lng=project.center_lng,
        center_lat=project.center_lat,
        lon=project.center_lng,
        lat=project.center_lat,
        coordinate_system=project.coordinate_system,
        boundary_geojson=project.boundary_geojson,
        core_buffer_m=project.core_buffer_m,
        nearby_buffer_m=project.nearby_buffer_m,
        radiation_buffer_m=project.radiation_buffer_m,
        land_use=project.land_use,
        project_area=project.project_area,
        building_area=project.building_area,
        build_year=project.build_year,
        update_demand=project.update_demand,
        expected_direction=project.expected_direction,
        project_type=project.project_type,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def _normalized_center(lng, lat) -> tuple[float | None, float | None, str | None]:
    """对中心点做合法性校验与自动纠偏，返回 (lng, lat, note)。

    无坐标时返回 (None, None, None)；非法坐标抛 ValueError 由上层转 400。
    """
    if lng is None and lat is None:
        return None, None, None
    if lng is None or lat is None:
        raise ValueError("center_lng / center_lat 必须成对提供")
    result = geo_utils.validate_center(lng, lat)
    if not result.is_usable:
        raise ValueError(f"项目中心点不合法：{result.note or '超出上海合法范围'}")
    return result.lng, result.lat, result.note


def create_project(db: Session, payload: ProjectCreate) -> ProjectOut:
    center_lng, center_lat, note = _normalized_center(
        payload.center_lng, payload.center_lat
    )

    project = Project(
        name=payload.name,
        city=payload.city,
        district=payload.district,
        street=payload.street,
        address=payload.address,
        description=payload.description,
        center_lng=center_lng,
        center_lat=center_lat,
        coordinate_system=payload.coordinate_system or settings.coordinate_system,
        boundary_geojson=payload.boundary_geojson,
        core_buffer_m=payload.core_buffer_m if payload.core_buffer_m is not None else 0,
        nearby_buffer_m=(
            payload.nearby_buffer_m
            if payload.nearby_buffer_m is not None
            else settings.nearby_buffer_m
        ),
        radiation_buffer_m=(
            payload.radiation_buffer_m
            if payload.radiation_buffer_m is not None
            else settings.radiation_buffer_m
        ),
        land_use=payload.land_use,
        project_area=payload.project_area,
        building_area=payload.building_area,
        build_year=payload.build_year,
        update_demand=payload.update_demand,
        expected_direction=payload.expected_direction,
        status=payload.status or "created",
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    logger.info("project created id=%s coord_note=%s", project.id, note)
    return _to_out(project)


def list_projects(db: Session) -> tuple[int, list[ProjectOut]]:
    rows = db.query(Project).order_by(Project.id.desc()).all()
    return len(rows), [_to_out(p) for p in rows]


def get_project(db: Session, project_id: int) -> Project | None:
    return db.get(Project, project_id)


def get_project_out(db: Session, project_id: int) -> ProjectOut | None:
    project = get_project(db, project_id)
    return _to_out(project) if project else None


def update_project(
    db: Session, project_id: int, payload: ProjectUpdate
) -> ProjectOut | None:
    project = get_project(db, project_id)
    if project is None:
        return None

    data = payload.model_dump(exclude_unset=True)

    # 中心点成对校验：任一被更新即重新校验
    if "center_lng" in data or "center_lat" in data:
        new_lng = data.get("center_lng", project.center_lng)
        new_lat = data.get("center_lat", project.center_lat)
        center_lng, center_lat, _ = _normalized_center(new_lng, new_lat)
        project.center_lng = center_lng
        project.center_lat = center_lat
        data.pop("center_lng", None)
        data.pop("center_lat", None)

    for field, value in data.items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)
    logger.info("project updated id=%s fields=%s", project.id, list(data.keys()))
    return _to_out(project)


def delete_project(db: Session, project_id: int) -> bool:
    project = get_project(db, project_id)
    if project is None:
        return False
    db.delete(project)
    db.commit()
    logger.info("project deleted id=%s", project_id)
    return True
