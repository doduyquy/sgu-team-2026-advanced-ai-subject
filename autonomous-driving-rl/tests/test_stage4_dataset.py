"""Dataset source selection, portability, history isolation and tamper checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import build_stage4_dataset as builder
from src.datasets import stage4_bc as data
from src.imitation.audit import load_json
from tests.test_rule_based_agent import make_input

RULES=load_json(builder.ROOT/"experiments/stage4/g1_quality_rules_v1.json")


def episode(eid, side="FIT", timeout=False):
    n=300 if timeout else 20
    intervals=[{"start":0,"end":n,"decision":"KEEP","destination":"BC_A_AND_B","context_before_start":0}]
    windows=[{"start":0,"end":n,"label":"PROGRESS"}]
    if timeout:
        intervals=[{"start":0,"end":100,"decision":"EXCLUDE"},
                   {"start":100,"end":150,"decision":"KEEP","destination":"BC_B_ONLY","context_before_start":91},
                   {"start":150,"end":300,"decision":"EXCLUDE"}]
        windows=[{"start":0,"end":50,"label":"UNCERTAIN"},{"start":50,"end":200,"label":"PROGRESS"},
                 {"start":200,"end":300,"label":"UNCERTAIN"}]
    geo="geo-"+eid
    return {"source_run_id":"run","episode_id":eid,"partition":side,"geometry_id":geo,"steps":n,
            "case":{"geometry_sha256":geo,"split":"TRAIN","case_id":eid,"tier":"Medium"},
            "outcome":"TIMEOUT" if timeout else "SUCCESS","whole_success_approved":not timeout,
            "trace_sha256":"trace","transitions_sha256":"transitions","map_sha256":"map",
            "intervals":intervals,"windows":windows,"eligible_steps":50 if timeout else n,
            "window_label_steps":{"PROGRESS":n}}


def build_fixture(base):
    episodes=[episode("success"),episode("dev",side="DEV"),episode("timeout",timeout=True)]
    cache={}
    for e in episodes:
        cache[("run",e["episode_id"])]= {}
        for step in range(e["steps"]):
            inp=make_input().to_dict()
            inp["step_index"]=step
            cache[("run",e["episode_id"])][step]={"agent_input":inp,"canonical_action":[0.,.5]}
    report={"episodes":episodes,"rules_version":"1.0.0"}
    for variant in ("BC-A","BC-B"):
        builder.materialize_variant(base/variant,variant,report,RULES,cache,[],{},"fixture")


def rewrite_rows(path, rows):
    path.unlink()
    data.write_rows(path,rows)


def reseal(folder):
    manifest=data.read_json(folder/"manifest.json")
    manifest["files"]={p.name:data.digest(p) for p in folder.iterdir() if p.name not in ("manifest.json","CHECKSUMS.sha256")}
    (folder/"manifest.json").write_text(json.dumps(manifest),encoding="utf8")
    checks={**manifest["files"],"manifest.json":data.digest(folder/"manifest.json")}
    (folder/"CHECKSUMS.sha256").write_bytes("".join(f"{h}  {n}\n" for n,h in sorted(checks.items())).encode())


class TestDatasetSelection(unittest.TestCase):
    def test_bc_a_excludes_reviewed_timeout(self):
        self.assertEqual(builder.selected_intervals(episode("t",timeout=True),"BC-A",RULES),[])

    def test_bc_b_keeps_exact_approved_timeout_interval(self):
        result=builder.selected_intervals(episode("t",timeout=True),"BC-B",RULES)
        self.assertEqual([(s["start"],s["end"]) for s in result],[(100,150)])

    def test_success_not_approved_is_rejected(self):
        e=episode("s")
        e["whole_success_approved"]=False
        with self.assertRaises(data.DatasetError): builder.selected_intervals(e,"BC-A",RULES)

    def test_crash_marked_keep_is_rejected(self):
        e=episode("s")
        e["outcome"]="CRASH_VEHICLE"
        with self.assertRaises(data.DatasetError): builder.selected_intervals(e,"BC-B",RULES)

    def test_overlap_or_gap_in_annotations_is_rejected(self):
        for start in (99,101):
            e=episode("t",timeout=True)
            e["intervals"][1]["start"]=start
            with self.assertRaises(data.DatasetError): builder.selected_intervals(e,"BC-B",RULES)

    def test_timeout_guard_cannot_be_bypassed(self):
        e=episode("t",timeout=True)
        e["windows"][0]["end"]=100
        e["windows"][1]["start"]=100
        with self.assertRaises(data.DatasetError): builder.selected_intervals(e,"BC-B",RULES)

    def test_history_cannot_start_after_required_context(self):
        e=episode("t",timeout=True)
        e["intervals"][1]["context_before_start"]=100
        with self.assertRaises(data.DatasetError): builder.selected_intervals(e,"BC-B",RULES)

    def test_non_float32_labels_rejected(self):
        with self.assertRaises(data.DatasetError): data.validate_action([.1,0.])
        data.validate_action([0.,.5])

    def test_gzip_payload_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a.gz",Path(tmp)/"b.gz"
            data.write_rows(a,[{"x":1}])
            data.write_rows(b,[{"x":1}])
            self.assertEqual(a.read_bytes(),b.read_bytes())


class TestPortableDataset(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        build_fixture(self.base)
        self.a,self.b=self.base/"BC-A",self.base/"BC-B"

    def test_bc_b_is_identical_superset(self):
        self.assertEqual(data.compare_variants(self.a,self.b),{"common_samples":40,"timeout_added_samples":50,"bc_a_subset_of_bc_b":True})

    def test_context_frames_are_not_training_labels(self):
        result=data.verify_dataset(self.b)
        self.assertEqual(result["FIT"]["samples"],70)
        self.assertEqual(result["FIT"]["context_only_frames"],9)

    def test_history_is_past_only_and_padding_is_masked(self):
        first=next(data.iter_examples(self.a,"FIT",10))
        self.assertEqual(first["history_mask"],[False]*9+[True])
        self.assertEqual(first["observations"][:9],[None]*9)
        rows=list(data.iter_examples(self.b,"FIT",10))
        self.assertEqual([x["step_index"] for x in rows[20]["observations"]],list(range(91,101)))

    def test_reader_does_not_expose_source_metadata(self):
        example=next(data.iter_examples(self.a,"FIT"))
        self.assertEqual(set(example),{"observations","history_mask","canonical_action"})
        self.assertNotIn("outcome",example["observations"][0])

    def test_reader_defensively_copies_shared_frames(self):
        reader=data.iter_examples(self.a,"FIT",10)
        first=next(reader)
        first["observations"][-1]["core_observation"]["features"][0]=999
        second=next(reader)
        self.assertNotEqual(second["observations"][-2]["core_observation"]["features"][0],999)

    def test_payload_tampering_detected(self):
        with (self.a/"samples_FIT.jsonl.gz").open("ab") as stream: stream.write(b"tamper")
        with self.assertRaises(data.DatasetError): data.verify_dataset(self.a)

    def test_future_history_rejected_even_if_file_rehashed(self):
        path=self.a/"samples_FIT.jsonl.gz"
        rows=list(data.read_rows(path))
        rows[0]["history_frame_ids"][-1]="run/success/1"
        rewrite_rows(path,rows)
        reseal(self.a)
        with self.assertRaisesRegex(data.DatasetError,"History"): data.verify_dataset(self.a)

    def test_missing_approved_sample_rejected_even_if_file_rehashed(self):
        path=self.a/"samples_FIT.jsonl.gz"
        rewrite_rows(path,list(data.read_rows(path))[1:])
        reseal(self.a)
        with self.assertRaisesRegex(data.DatasetError,"approved intervals"): data.verify_dataset(self.a)

    def test_privileged_fields_rejected_inside_public_input(self):
        path=self.a/"frames_FIT.jsonl.gz"
        rows=list(data.read_rows(path))
        rows[0]["agent_input"]["task_context"]["reward"]=1
        rewrite_rows(path,rows)
        reseal(self.a)
        with self.assertRaisesRegex(data.DatasetError,"Private metadata"): data.verify_dataset(self.a)

    def test_geometry_leakage_rejected(self):
        meta_path=self.a/"sample_index.jsonl.gz"
        rows=list(data.read_rows(meta_path))
        for row in rows:
            if row["partition"]=="DEV": row["geometry_sha256"]="geo-success"
        rewrite_rows(meta_path,rows)
        p=self.a/"provenance.json"
        provenance=data.read_json(p)
        next(e for e in provenance["episodes"] if e["partition"]=="DEV")["case"]["geometry_sha256"]="geo-success"
        p.write_text(json.dumps(provenance))
        reseal(self.a)
        with self.assertRaisesRegex(data.DatasetError,"Geometry leaks"): data.verify_dataset(self.a)

    def test_existing_destination_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            builder.materialize_variant(self.a,"BC-A",{},RULES,{},[],{},"another")
        self.assertEqual(data.verify_dataset(self.a)["FIT"]["samples"],20)

    def test_checksum_file_and_extra_files_are_checked(self):
        (self.a/"extra").write_bytes(b"unknown")
        with self.assertRaisesRegex(data.DatasetError,"Unexpected or missing"): data.verify_dataset(self.a)

    def test_bad_history_length_or_partition_rejected(self):
        with self.assertRaises(data.DatasetError): next(data.iter_examples(self.a,"FIT",2))
        with self.assertRaises(data.DatasetError): next(data.iter_examples(self.a,"TEST",verify=False))


class TestBuildAttemptRecords(unittest.TestCase):
    def test_missing_inputs_preserve_failed_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            with self.assertRaises(FileNotFoundError):
                builder.build(root,"missing","experiments/stage4/missing.json","artifacts/stage4/data")
            attempt=root/"runs/missing"
            self.assertEqual(data.read_json(attempt/"run_state.json")["status"],"FAILED")
            self.assertEqual(data.read_json(attempt/"run_integrity.json")["status"],"FAILED")

    def test_existing_run_id_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            attempt=root/"runs/existing"
            attempt.mkdir(parents=True)
            (attempt/"original").write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                builder.build(root,"existing","missing","artifacts/stage4/data")
            self.assertEqual((attempt/"original").read_bytes(),b"keep")


if __name__=="__main__":
    unittest.main()
