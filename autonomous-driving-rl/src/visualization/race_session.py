"""Race-mode choices and TRAIN presets. No change to the canonical registry."""

from dataclasses import dataclass, field, replace
from pathlib import Path

from src.agents.MPPI import MPPIAgent
from src.launcher.models import LaunchRequestV1, LauncherMode
from src.launcher.registry import build_default_agent_registry


RACE_AGENTS = (
    ("stage2_astar", "A*", "Chi phí đã đi + ước lượng tới đích"),
    ("stage2_bfs", "BFS", "Tìm kiếm theo chiều rộng"),
    ("stage2_dijkstra", "Dijkstra", "Tìm đường theo tổng chi phí"),
    ("stage2_hybrid_astar", "Hybrid A*", "Tìm đường có xét hướng xe"),
    (MPPIAgent.agent_id, "MPPI · thử nghiệm", "Lấy mẫu và tối ưu chuỗi điều khiển"),
)
ROAD_NAMES = ("Dễ", "Trung bình", "Khó", "Việt Nam")
HUMAN_INDEX = 5


def race_participant(index):
    if index == HUMAN_INDEX:
        return ('human_phone', 'Bạn · điện thoại', 'Người chơi điều khiển qua Wi-Fi')
    return RACE_AGENTS[index]


def build_race_registry():
    """Use the same policies and JSON configuration as CLI / Workbench."""
    return build_default_agent_registry()


def load_race_roads(project_root: Path, map_set='train'):
    """Verify each preset before the persistent native engine is opened."""
    if map_set == 'original':
        roads = load_original_roads(project_root)
        from src.visualization.vietnam_track import race_road
        return [*roads, race_road()]
    if map_set != 'train':
        raise ValueError('Unknown map set')
    from src.launcher.cases import load_locked_geometry_block_sequence, resolve_sandbox_case
    roads = []
    for tier in ("Easy", "Medium", "Hard"):
        request = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="stage2_astar", tier=tier)
        case = resolve_sandbox_case(request, project_root)
        blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict(), project_root)
        roads.append((case, blocks))
    from src.visualization.vietnam_track import race_road
    return [*roads, race_road()]


def load_original_roads(project_root: Path):
    """Restore the archived MapSuite showcase roads, retaining real provenance.

    The supplied reference minimap matches SCTCS/0. These are visual reference
    sessions, not new TRAIN samples or canonical benchmark results.
    """
    from src.launcher.cases import load_locked_geometry_block_sequence, load_manifest_csv
    from src.launcher.models import ResolvedCaseV1
    from src.platform import compute_route_aware_horizon
    from src.launcher.cases import LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256
    rows=load_manifest_csv(project_root/'results/audits/evaluation_protocol/geometry_split_manifest.csv',
                           LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256)
    roads=[]
    for tier,sequence,seed in (('Easy','SCS',11),('Medium','SCTCS',0),('Hard','SCXOCS',2)):
        row=next(r for r in rows if r['tier']==tier and r['sequence']==sequence
                 and int(r['geometry_generation_seed'])==seed)
        case=ResolvedCaseV1(case_id=f'original/{tier}/{sequence}/{seed}/0',case_index=1,
                           protocol_order_index=1,split=row['split'],tier=tier,sequence=sequence,
                           geometry_generation_seed=seed,geometry_sha256=row['geometry_sha256'],
                           environment_seed=0,traffic_density=float(row['traffic_density']),
                           horizon_steps=compute_route_aware_horizon(float(row['route_length_m'])))
        blocks,_,_=load_locked_geometry_block_sequence(case.to_dict(),project_root)
        roads.append((case,blocks))
    return roads


@dataclass
class RaceSelection:
    agent_index: int = 0
    road_index: int = 0
    traffic: bool = False
    speed: float = 1.0
    mode: str = "compare"
    comparison_indices: tuple = tuple(range(len(RACE_AGENTS)))
    player_model: str = 'lamborghini'
    vehicle_models: dict = field(default_factory=dict)
    test_case: object = None
    test_blocks: object = None

    @property
    def manual(self):
        return self.mode in ('human', 'practice')

    def model_for(self, index):
        if index == HUMAN_INDEX:
            return self.player_model
        return self.vehicle_models.get(index, ('lamborghini', 'tesla', 'vf3')[index % 3])

    @property
    def agent_id(self):
        return RACE_AGENTS[self.agent_index][0]

    def case(self, roads):
        if self.test_case is not None:
            return self.test_case
        case = roads[self.road_index][0]
        return replace(case, traffic_density=(.015 if case.sequence == 'VN' else .1) if self.traffic else 0.)

    def blocks(self, roads):
        return self.test_blocks if self.test_case is not None else roads[self.road_index][1]

    def select_test(self, case, blocks):
        if case.split!='TEST':
            raise ValueError('Track picker accepts only TEST dataset cases')
        self.test_case, self.test_blocks = case, blocks

    def clear_test(self):
        self.test_case = self.test_blocks = None

    @property
    def chosen_indices(self):
        if self.mode == 'practice':
            return ()
        return (self.agent_index,) if self.mode == "single" else self.comparison_indices

    def set_mode(self, mode):
        if mode not in ("single", "compare", "human", "practice"):
            raise ValueError("Unknown race mode")
        self.mode = mode
        if self.manual:
            self.speed = 1.
        if mode == "compare" and not self.comparison_indices:
            self.comparison_indices = (self.agent_index,)

    def choose(self, index):
        if type(index) is not int or not 0 <= index < len(RACE_AGENTS):
            raise ValueError("Unknown algorithm")
        if self.mode == "single":
            self.agent_index = index
        else:
            selected = set(self.comparison_indices)
            selected.remove(index) if index in selected else selected.add(index)
            self.comparison_indices = tuple(sorted(selected))

    def planned_indices(self):
        selected = self.chosen_indices
        if self.mode == "compare" and len(selected) < 2:
            raise ValueError("Chọn ít nhất 2 thuật toán hoặc bấm Chọn cả 5.")
        if self.mode == 'human' and not selected:
            raise ValueError('Chọn ít nhất 1 AI hoặc chuyển sang Lái tự do.')
        if self.manual:
            return (HUMAN_INDEX, *selected)
        return tuple(selected)


@dataclass
class RaceComparison:
    """One simultaneous race; cars may finish in any order."""
    indices: tuple
    case: dict
    comparison_id: int
    results: list = field(default_factory=list)
    status: str = "RUNNING"
    execution: str = 'shared_world_lockstep'
    control_dt_s: float = .1

    def record(self, summary):
        expected = {race_participant(i)[0] for i in self.indices}
        recorded = {row["agent_id"] for row in self.results}
        if self.status != "RUNNING" or summary["agent_id"] in recorded:
            raise ValueError("Race does not expect another result for this car")
        if summary["agent_id"] not in expected or summary["case"] != self.case:
            raise ValueError("Comparison agent or case changed")
        self.results.append(dict(summary))
        if len(self.results) == len(self.indices):
            self.status = "COMPLETE"

    def to_dict(self):
        from src.visualization.race_rules import rules_manifest
        recorded = {row["agent_id"] for row in self.results}
        finishers = [row for row in self.results if row.get('clean_success')]
        return {"kind": "NATIVE_SIMULTANEOUS_RACE", "canonical_run": False,
                "race_rules": rules_manifest(speed_limit_kmh=80. if self.case.get('sequence') == 'VN' else 20.,
                                             cruise_kmh=60. if self.case.get('sequence') == 'VN' else 15.),
                "finish_ranks": {row['agent_id']:1+sum(other['simulation_time_s'] < row['simulation_time_s'] for other in finishers)
                                 for row in finishers},
                "execution": self.execution, "control_dt_s":self.control_dt_s,"inter_vehicle_collisions": True,
                "comparison_id": self.comparison_id, "status": self.status,
                "case": self.case, "agent_seed": 42,
                "agent_ids": [race_participant(i)[0] for i in self.indices],
                "results": self.results,
                "pending_agent_ids": [race_participant(i)[0] for i in self.indices
                                      if race_participant(i)[0] not in recorded]}


def prepare_race_reset(env, case, blocks):
    """Pinned PGMapManager API: replace geometry on reset without closing its window.

    Maps are deliberately not cached. before_reset() detaches/destroys the old
    current map in manager order, after the old minimap has been destroyed.
    """
    from metadrive.component.map.pg_map import MapGenerateMethod
    env.config["start_seed"] = int(case.environment_seed)
    env.config["traffic_density"] = float(case.traffic_density)
    env.config["horizon"] = int(case.horizon_steps)
    env.config["map_config"].update({"type": MapGenerateMethod.PG_MAP_FILE, "config": blocks})
    env.start_index = env.start_seed = int(case.environment_seed)
    env.engine.map_manager.start_seed = int(case.environment_seed)
    env.engine.map_manager.maps = {int(case.environment_seed): None}


OUTCOME_LABELS = {
    "SOLID_LINE": "Bị loại: đè vạch liền",
    "SUCCESS": "Đã về đích", "OUT_OF_ROAD": "Xe đã ra khỏi đường",
    "TIMEOUT": "Hết thời gian", "CRASH_VEHICLE": "Va chạm xe khác",
    "CRASH_OBJECT": "Va chạm vật cản", "CRASH_BUILDING": "Va chạm công trình",
    "CRASH_HUMAN": "Va chạm người", "CRASH_SIDEWALK": "Va chạm lề đường",
    "PREVIEW_LIMIT": "Đã chạy đủ bước kiểm tra", "ABORTED": "Đã dừng lượt chạy",
    "TECHNICAL_ERROR": "Lỗi kỹ thuật · xem thông báo bên dưới",
}
