"""Read-only TEST dataset catalog. Browsing never creates a simulator engine."""
from copy import deepcopy
import csv
from dataclasses import dataclass
import json
from pathlib import Path

from src.launcher.cases import resolve_test_cases, load_manifest_csv, LOCKED_TEST_CASE_MANIFEST_SHA256
from src.platform import canonical_json_sha256


TIERS = ('Easy', 'Medium', 'Hard', 'Extreme')
TIER_LABELS = {'Easy':'Dễ', 'Medium':'Trung bình', 'Hard':'Khó', 'Extreme':'Cực khó'}


@dataclass(frozen=True)
class TestTrack:
    geometry_id: str
    tier: str
    sequence: str
    geometry_seed: int
    geometry_sha256: str
    length_m: float
    curves: int
    intersections: int
    preview: Path
    cases: tuple
    _blocks: tuple

    def blocks(self):
        return deepcopy(list(self._blocks))


class TestTrackCatalog:
    def __init__(self, root):
        root = Path(root)
        dataset_cases = load_manifest_csv(root/'datasets/mapsuite_v1/evaluation_cases/test_cases.csv',
                                          LOCKED_TEST_CASE_MANIFEST_SHA256)
        locked = {case.case_id:case for case in resolve_test_cases(root)}
        self.cases = tuple(locked[row['case_id']] for row in dataset_cases)
        by_geometry = {}
        for case in self.cases:
            by_geometry.setdefault(case.geometry_sha256, []).append(case)
        with (root/'datasets/mapsuite_v1/splits/test_geometries.csv').open(encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
        if len(rows)!=12 or {r['geometry_sha256'] for r in rows}!=set(by_geometry):
            raise ValueError('TEST map dataset does not match the locked TEST cases')
        archived = json.loads((root/'configs/maps/mapsuite_v1_candidates.json').read_text(encoding='utf-8'))
        exact = {}
        for definitions in archived['tier_definitions'].values():
            for row in [definitions['primary_canonical'], *definitions['alternate_canonicals']]:
                exact[canonical_json_sha256(row['exact_block_sequence'])] = row
        tracks = []
        for row in rows:
            sha = row['geometry_sha256']
            if row['split']!='TEST' or sha not in exact:
                raise ValueError('Missing verified TEST map geometry')
            cases = tuple(sorted(by_geometry[sha], key=lambda case:case.environment_seed))
            signature = (row['tier'],row['sequence'],int(row['geometry_generation_seed']))
            if any((c.tier,c.sequence,c.geometry_generation_seed)!=signature for c in cases):
                raise ValueError('TEST map identity differs between dataset and case manifest')
            if len(cases)!=5 or {c.environment_seed for c in cases}!=set(range(9101,9106)):
                raise ValueError('TEST map must contain the five locked scenarios')
            source = exact[sha]
            preview = root/'datasets/mapsuite_v1/previews'/Path(source['preview_image']).name
            if not preview.is_file():
                raise FileNotFoundError(preview)
            tracks.append(TestTrack(row['geometry_id'],*signature,sha,float(row['route_length_m']),
                                    int(row['curve_count']),int(row['intersection_count'])+int(row['t_intersection_count']),
                                    preview,cases,tuple(source['exact_block_sequence'])))
        self.tracks = tuple(tracks)
        self.by_id = {track.geometry_id:track for track in tracks}
        self.by_case = {case.case_id:track for track in tracks for case in track.cases}

    def in_tier(self, tier):
        if tier not in TIERS:
            raise ValueError('Unknown TEST map difficulty')
        return tuple(track for track in self.tracks if track.tier==tier)

    def resolve(self, case_id):
        track = self.by_case[case_id]
        return next(case for case in track.cases if case.case_id==case_id),track.blocks()


class TrackPickerDraft:
    """Draft choices are committed only by the Apply button."""
    def __init__(self, catalog):
        self.catalog = catalog
        self.track = catalog.in_tier('Medium')[0]
        self.scenario = 0

    def reset(self, current_case):
        self.track = next((t for t in self.catalog.tracks if t.geometry_sha256==current_case.geometry_sha256),
                          self.catalog.in_tier(current_case.tier)[0])
        self.scenario = next((i for i,c in enumerate(self.track.cases)
                              if c.environment_seed==current_case.environment_seed),0)

    def tier(self, tier):
        self.track = self.catalog.in_tier(tier)[0]

    def map(self, geometry_id):
        self.track = self.catalog.by_id[geometry_id]

    def choose_scenario(self, index):
        if type(index) is not int or not 0<=index<len(self.track.cases):
            raise ValueError('Unknown TEST scenario')
        self.scenario = index

    @property
    def case(self):
        return self.track.cases[self.scenario]
