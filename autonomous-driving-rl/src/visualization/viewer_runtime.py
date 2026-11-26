"""Playback pacing and episode-level recovery for noncanonical search viewing."""

import json
import math
from pathlib import Path


class FrameRate:
    """Count completed native render calls per wall-clock second."""

    def __init__(self, now):
        self.started, self.frames, self.fps = now, 0, 0.

    def record(self, now):
        self.frames += 1
        elapsed = now - self.started
        if elapsed >= 1.:
            self.fps = self.frames / elapsed
            self.frames, self.started = 0, now


def set_native_view_mode(engine, mode, target_fps):
    """Smooth renders all five physics substeps; fast allows frame decimation."""
    if mode == "smooth":
        engine.force_fps.enable()
        engine.force_fps.fps = target_fps
    elif mode == "fast":
        engine.force_fps.disable()
    else:
        raise ValueError("View mode must be smooth or fast")


class PlaybackSpeed:
    """Change wall-clock waiting only; zero means no artificial waiting."""

    def __init__(self, speed=5.):
        self.set_speed(speed)

    def set_speed(self, speed):
        speed = float(speed)
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Playback speed must be finite and nonnegative (0 = maximum)")
        self.speed = speed

    @property
    def label(self):
        return "MAX" if self.speed == 0 else f"{self.speed:g}×"

    def delay(self, work_seconds):
        return max(0., .1 / self.speed - work_seconds) if self.speed else 0.


def load_viewer_progress(directory, schedule, source):
    """Resume a stopped session with the same cases and planner source.

    Only complete or explicitly skipped episodes are carried over. An interrupted
    episode must restart because a full physics-engine snapshot was not saved.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "NONCANONICAL_VISUAL_RERUN":
        raise ValueError("Can only resume a search visual session")
    if manifest.get("status") == "RUNNING":
        raise ValueError("Stop the previous viewer before resuming its progress")
    if manifest["schedule"] != schedule:
        raise ValueError("Resume schedule differs; use the same agents, splits and case selection")
    if manifest["source"] != source:
        raise ValueError("Planner source/config changed; start a new visual session")
    finished = {int(key): status for key, status in manifest.get("carried_episode_statuses", {}).items()}
    path = directory / "episodes.jsonl"
    if path.exists():
        previous = None
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            index = row["episode"]
            if not 1 <= index <= len(schedule):
                raise ValueError("Episode index outside the resume schedule")
            expected = schedule[index - 1]
            if row["agent_id"] != expected["agent_id"] or row["case"] != {k: v for k, v in expected.items() if k != "agent_id"}:
                raise ValueError("Episode provenance does not match the resume schedule")
            # Early viewers logged A (next agent) only on the active case, then
            # jumped over its remaining cases. Recover that explicit group skip.
            if previous is not None and previous["status"] == "SKIPPED" and index > previous["episode"] + 1:
                group = (previous["agent_id"], previous["case"]["split"])
                gap = range(previous["episode"] + 1, index)
                if (row["agent_id"], row["case"]["split"]) != group and all(
                    (schedule[i - 1]["agent_id"], schedule[i - 1]["split"]) == group for i in gap
                ):
                    finished.update({i: "SKIPPED" for i in gap})
            if row["status"] in ("COMPLETE", "SKIPPED"):
                finished[index] = row["status"]
            previous = row
    if any(not 1 <= index <= len(schedule) or status not in ("COMPLETE", "SKIPPED")
           for index, status in finished.items()):
        raise ValueError("Invalid carried episode progress")
    return finished
