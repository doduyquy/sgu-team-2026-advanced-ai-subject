"""Dataset integrity, draft/apply semantics and no simulation while browsing."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QProcess  # noqa:F401 - Windows DLL loading order.
from src.launcher.cases import resolve_test_cases
from src.platform import canonical_json_sha256
from src.visualization.race_session import RaceSelection
from src.visualization.test_track_catalog import TestTrackCatalog,TrackPickerDraft,TIERS
import test_race_mode as fixtures


ROOT=Path(__file__).resolve().parents[1]


class TestTrackPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=TestTrackCatalog(ROOT)

    def test_catalog_contains_exactly_the_locked_test_cases_and_dataset_previews(self):
        with patch('src.launcher.cases.MetaDriveEnv') as environment:
            catalog=TestTrackCatalog(ROOT)
        environment.assert_not_called()
        self.assertEqual(len(catalog.tracks),12)
        self.assertEqual({c.case_id:c.to_dict() for c in catalog.cases},
                         {c.case_id:c.to_dict() for c in resolve_test_cases(ROOT)})
        for tier in TIERS:self.assertEqual(len(catalog.in_tier(tier)),3)
        for track in catalog.tracks:
            self.assertTrue(track.preview.is_file())
            self.assertEqual(track.preview.parent,ROOT/'datasets/mapsuite_v1/previews')
            self.assertEqual({c.environment_seed for c in track.cases},set(range(9101,9106)))
            self.assertTrue(all(c.split=='TEST' for c in track.cases))
            self.assertEqual(canonical_json_sha256(track.blocks()),track.geometry_sha256)

    def test_resolved_geometry_is_defensive_copy_and_case_is_unmodified(self):
        case_id=self.catalog.cases[0].case_id
        case,blocks=self.catalog.resolve(case_id)
        blocks[1]['id']='corrupted'
        again,original=self.catalog.resolve(case_id)
        self.assertEqual(case,again)
        self.assertEqual(canonical_json_sha256(original),case.geometry_sha256)
        with self.assertRaises(KeyError):self.catalog.resolve('not-a-test-case')

    def test_browsing_extreme_and_cancel_preserves_committed_map_and_conditions(self):
        selection=RaceSelection()
        track=self.catalog.in_tier('Medium')[1]
        committed=track.cases[3]
        selection.select_test(*self.catalog.resolve(committed.case_id))
        draft=TrackPickerDraft(self.catalog);draft.reset(committed)
        self.assertEqual(draft.scenario,3)
        draft.tier('Extreme');draft.choose_scenario(4)
        self.assertEqual(draft.case.tier,'Extreme')
        self.assertIs(selection.case([]),committed)
        draft.reset(committed)
        self.assertEqual(draft.case,committed)
        for invalid in (-1,5,True,'1'):
            with self.assertRaises(ValueError):draft.choose_scenario(invalid)

    def test_selected_test_traffic_seed_and_horizon_are_exact_even_with_preset_toggle(self):
        selection=RaceSelection()
        case,blocks=self.catalog.resolve(self.catalog.in_tier('Extreme')[0].cases[2].case_id)
        selection.select_test(case,blocks)
        for toggle in (True,False):
            selection.traffic=toggle
            self.assertEqual(selection.case([]).to_dict(),case.to_dict())
            self.assertEqual(canonical_json_sha256(selection.blocks([])),case.geometry_sha256)
        with self.assertRaises(ValueError):
            selection.select_test(SimpleNamespace(split='TRAIN'),[])
        selection.clear_test()
        preset=self.catalog.in_tier('Easy')[0].cases[0]
        self.assertEqual(selection.case([(preset,[])]).sequence,'SCS')

    def app_with_picker(self):
        app=fixtures.TestRaceMode().app()
        app.test_track_catalog=self.catalog
        picker=SimpleNamespace(is_open=True,draft=TrackPickerDraft(self.catalog),hide=MagicMock(),handle=MagicMock())
        app.ui.track_picker=picker
        return app,picker

    def test_apply_commits_exact_dataset_case_without_resetting_or_starting_engine(self):
        app,picker=self.app_with_picker()
        picker.draft.tier('Extreme');picker.draft.choose_scenario(4)
        app.handle('track_apply')
        self.assertEqual(app.selection.test_case.to_dict(),picker.draft.case.to_dict())
        self.assertEqual(canonical_json_sha256(app.selection.test_blocks),picker.draft.case.geometry_sha256)
        app.env.reset.assert_not_called();app.env.step.assert_not_called();app._start.assert_not_called()
        picker.hide.assert_called_once();app.save_manifest.assert_called_once()

    def test_modal_blocks_background_shortcuts_and_close_discards_draft(self):
        app,picker=self.app_with_picker()
        app.handle('start');app.handle('mode','human');app.handle('road',2)
        self.assertEqual(app.selection.mode,'compare');self.assertEqual(app.selection.road_index,0)
        self.assertIsNone(app.selection.test_case)
        app._start.assert_not_called()
        app.handle('pause')
        picker.hide.assert_called_once()
        app.env.reset.assert_not_called();app.env.step.assert_not_called()


if __name__=='__main__':unittest.main()
