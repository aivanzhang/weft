"""Detached substep telemetry; contact state precedes the integrated state."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class EquipmentState:
    object_id: str
    position: tuple[float, ...]
    quaternion: tuple[float, ...]
    velocity: tuple[float, ...]
    angular_velocity: tuple[float, ...]
    virtual_force: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HumanObjectContact:
    human_geom: str
    human_segment: str
    object_id: str
    position: tuple[float, ...]
    distance: float
    normal_force: float
    force_on_human_world: tuple[float, ...]
    normal_on_human_world: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SubstepSample:
    step_index: int
    substep_index: int
    interval_start_s: float
    interval_end_s: float | None
    contact_time_s: float
    valid: bool
    failure: str | None
    human_qpos: tuple[float, ...]
    human_qvel: tuple[float, ...]
    target_qpos: tuple[float, ...]
    human_action: tuple[float, ...]
    equipment: tuple[EquipmentState, ...]
    contacts: tuple[HumanObjectContact, ...]
    wrist_positions: Mapping[str, tuple[float, ...]]
    max_human_floor_penetration_model_units: float | None
