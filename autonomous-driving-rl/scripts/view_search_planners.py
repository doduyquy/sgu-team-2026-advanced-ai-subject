"""View Stage 2 search on the locked VALIDATION/TEST cases with decision logs.

This is an explicitly noncanonical visual rerun, not a benchmark certification.
It does not change registry eligibility, planner parameters or existing results.
Default: BFS, Dijkstra, A*, Hybrid A* on all 96 VALIDATION + 60 TEST cases (624).
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AGENTS = {"bfs": "stage2_bfs", "dijkstra": "stage2_dijkstra", "astar": "stage2_astar",
          "hybrid_astar": "stage2_hybrid_astar"}
HISTORICAL_COMMIT = "63b36c851f7a4bb2c246705573ba5268c61fa8b1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", nargs="+", choices=tuple(AGENTS), default=list(AGENTS))
    p.add_argument("--splits", nargs="+", choices=("VALIDATION", "TEST"), default=["VALIDATION", "TEST"])
    p.add_argument("--limit-cases", type=int, help="Optional preview subset per split (never a full benchmark)")
    p.add_argument("--smoke-steps", type=int, default=0, help="Stop the entire viewer after N decisions")
    p.add_argument("--preview-steps", type=int, default=0, help="Preview N steps per case, then advance (marked PREVIEW_LIMIT)")
    p.add_argument("--speed", type=float, default=5., help="Target playback speed; 0 = maximum (keys 1/2/5/0)")
    p.add_argument("--render-fps", type=float, default=60., help="Target native FPS (smooth mode renders physics substeps)")
    p.add_argument("--view-mode", choices=("smooth", "fast"), default="smooth", help="Prioritize smooth movement or simulation throughput (M toggles)")
    p.add_argument("--resume-from", type=Path, help="Stopped session directory; carry finished cases into a new output")
    p.add_argument("--output", type=Path)
    p.add_argument("--capture-step", type=int, default=30)
    p.add_argument("--plan-only", action="store_true")
    return p


def source_provenance():
    paths = [f"src/agents/{name}.py" for name in (
        "graph_search", "graph_search_agents", "planning_graph", "astar_agent",
        "path_following", "hybrid_astar", "hybrid_astar_agent")]
    paths += ["configs/experiments/astar_learning_v1.json", "configs/experiments/search_planners_v1.json"]
    records = []
    for path in paths:
        current = (PROJECT_ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        previous = subprocess.run(
            ["git", "show", f"{HISTORICAL_COMMIT}:autonomous-driving-rl/{path}"],
            cwd=PROJECT_ROOT, capture_output=True, check=True,
        ).stdout.replace(b"\r\n", b"\n")
        records.append({"path": path, "sha256": hashlib.sha256(current).hexdigest(),
                        "matches_historical_source": current == previous})
    return records


def main():
    args = parser().parse_args()
    if (args.limit_cases is not None and args.limit_cases <= 0) or args.smoke_steps < 0 or args.preview_steps < 0:
        raise ValueError("limit-cases must be positive; smoke/preview-steps must be nonnegative")
    # The local Windows runtime needs Qt loaded before Panda3D/MetaDrive.
    from PySide6.QtCore import QProcess  # noqa: F401
    from PySide6.QtWidgets import QApplication  # noqa: F401
    from src.visualization.viewer_runtime import PlaybackSpeed, FrameRate, load_viewer_progress, set_native_view_mode
    playback = PlaybackSpeed(args.speed)
    if not 0 < args.render_fps <= 120:
        raise ValueError("render-fps must be in (0, 120]")
    from src.launcher.cases import (resolve_validation_cases, resolve_test_cases,
                                    load_locked_geometry_block_sequence, make_jsonable)
    from src.training.runtime import local_training_runtime

    with local_training_runtime(PROJECT_ROOT):
        cases = {split: (resolve_validation_cases(PROJECT_ROOT) if split == "VALIDATION"
                         else resolve_test_cases(PROJECT_ROOT)) for split in args.splits}
        jobs = [(AGENTS[key], case) for split in args.splits for key in args.agents
                for case in cases[split][:args.limit_cases]]
        provenance = source_provenance()
    schedule = [{"agent_id": a, **c.to_dict()} for a, c in jobs]
    carried = load_viewer_progress(args.resume_from, schedule, provenance) if args.resume_from else {}
    if args.plan_only:
        print(json.dumps({"episodes": len(jobs), "splits": {k: len(v) for k, v in cases.items()},
                          "canonical_run": False, "source": provenance,
                          "carried_finished_episodes": len(carried), "remaining_episodes": len(jobs) - len(carried),
                          "first_case": jobs[0][1].to_dict()}, indent=2))
        return 0

    from metadrive.constants import TerrainProperty
    from metadrive.engine.core import engine_core
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from direct.showbase.DirectObject import DirectObject
    from panda3d.core import Filename, WindowProperties, loadPrcFileData
    from src.launcher.executor import build_metadrive_case_config, read_vehicle_speed_kmh
    from src.launcher.registry import build_default_agent_registry
    from src.platform import AgentPublicEpisodeContext, InputProfileId, build_agent_input, classify_episode_outcome, get_action_adapter
    from src.visualization.hud_manager import RacingHudManager
    from src.visualization.planner_trace import SEARCH_LABELS, SearchTraceOverlay, explain_search_decision

    # Reduce only terrain texture memory, preserving geometry, physics and policy.
    TerrainProperty.get_semantic_map_pixel_per_meter = classmethod(lambda cls: 4)
    loadPrcFileData("search_viewer", "sync-video false")
    # MetaDrive hardcodes 16x MSAA on Windows. Reduce only visual sampling,
    # process-locally, so native viewing does not pay for that very high setting.
    original_pbr_init = engine_core.init

    def viewer_pbr_init(*positional, **kwargs):
        kwargs["msaa_samples"] = 2
        return original_pbr_init(*positional, **kwargs)

    engine_core.init = viewer_pbr_init
    output = (args.output or PROJECT_ROOT / "artifacts" / "search_visual" / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"kind": "NONCANONICAL_VISUAL_RERUN", "canonical_run": False,
                "historical_commit": HISTORICAL_COMMIT, "source": provenance,
                "planned_episodes": len(jobs), "args": {**vars(args), "output": str(output)},
                "schedule": schedule,
                "resumed_from": str(args.resume_from.resolve()) if args.resume_from else None,
                "carried_episode_statuses": carried,
                "playback_speed": playback.speed,
                "view_mode": args.view_mode,
                "graphics_settings": {"msaa_samples": 2, "pssm_shadows": False,
                                      "window_size": [1280, 800], "target_fps": args.render_fps},
                "control_dt_s": .1, "terrain_pixels_per_meter": 4,
                "status": "RUNNING", "completed_episodes": sum(s == "COMPLETE" for s in carried.values())}
    manifest_path = output / "session.json"

    def save_manifest():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    controls = {"quit": False, "next": False, "next_agent": False, "view_mode": args.view_mode}
    registry = build_default_agent_registry()
    adapter = get_action_adapter("discrete9_lowbranch_v1")
    total_decisions, skip_agent = 0, None
    text_log = (output / "decisions.log").open("w", encoding="utf-8", buffering=1)
    json_log = (output / "decisions.jsonl").open("w", encoding="utf-8", buffering=1)
    results = (output / "episodes.jsonl").open("w", encoding="utf-8", buffering=1)

    def log(message):
        print(message, flush=True)
        text_log.write(message + "\n")

    def set_control(key):
        controls[key] = True
        if env is not None:
            env.in_stop = False

    def set_speed(value):
        playback.set_speed(value)
        manifest["playback_speed"] = playback.speed
        save_manifest()
        log(f"PLAYBACK {playback.label} | physics dt remains 0.02 s; control dt remains 0.1 s")

    def toggle_view_mode():
        controls["view_mode"] = "fast" if controls["view_mode"] == "smooth" else "smooth"
        set_native_view_mode(env.engine, controls["view_mode"], args.render_fps)
        manifest["view_mode"] = controls["view_mode"]
        save_manifest()
        log(f"VIEW_MODE {controls['view_mode']} | target FPS {args.render_fps:g}")

    try:
        log(f"VISUAL RERUN | {len(jobs)} episodes | logs: {output}")
        log("P: pause/resume | N: next case | A: next agent | Q: quit | I: input details")
        log(f"PLAYBACK {playback.label} | 1/2/5: target speed | 0: maximum | M: smooth/fast | view {controls['view_mode']}")
        if carried:
            log(f"RESUME | carrying {len(carried)} finished episodes from {args.resume_from}; unfinished case restarts")
        log("Explanations report actual diagnostics; every decision is saved in decisions.jsonl and decisions.log.")
        if not all(p["matches_historical_source"] for p in provenance):
            log("NOTICE: current planner source differs from the historical evaluation; see session.json.")
        for index, (agent_id, case) in enumerate(jobs, 1):
            if controls["quit"] or (output / "STOP").exists():
                break
            if index in carried:
                continue
            if skip_agent == (case.split, agent_id):
                results.write(json.dumps({"episode": index, "agent_id": agent_id, "case": case.to_dict(),
                                          "status": "SKIPPED", "steps": 0, "route_completion": 0.,
                                          "outcome": None, "clean_success": False, "reason": "NEXT_AGENT"}) + "\n")
                continue
            skip_agent = None
            controls["next"] = controls["next_agent"] = False
            label = SEARCH_LABELS[agent_id][0]
            log(f"START {index}/{len(jobs)} | {label} | {case.split} | {case.case_id} | geometry={case.geometry_generation_seed} env={case.environment_seed}")
            manifest.update(current_episode=index, agent_id=agent_id, case=case.to_dict())
            save_manifest()
            env = hud = overlay = events = agent = None
            step, route, speed = 0, 0., 0.
            flags, outcome = {}, None
            result_status = "INTERRUPTED"
            try:
                blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict(), PROJECT_ROOT)
                config = build_metadrive_case_config(case, blocks, render_mode="NATIVE")
                config["window_size"] = (1280, 800)
                env = MetaDriveEnv(config)
                obs, info = env.reset(seed=case.environment_seed)
                set_native_view_mode(env.engine, controls["view_mode"], args.render_fps)
                env.engine.pssm.toggle_shadows_mode()
                env.engine.pssm.buffer.setActive(False)
                gsg = env.engine.win.getGsg()
                graphics = {"vendor": gsg.getDriverVendor(), "renderer": gsg.getDriverRenderer(),
                            "driver": gsg.getDriverVersion()}
                manifest["graphics_device"] = graphics
                save_manifest()
                log(f"GPU {graphics['vendor']} | {graphics['renderer']} | {graphics['driver']}")
                props = WindowProperties()
                props.setTitle(f"Search Viewer | {label} | {case.split} | {index}/{len(jobs)}")
                env.engine.win.requestProperties(props)
                hud = RacingHudManager(env)
                hud.agent_io.root.hide()
                overlay = SearchTraceOverlay(env.engine, f"{label} · {case.split} · {index}/{len(jobs)} · {case.case_id}")
                render_tasks = env.engine.taskMgr.getTasksNamed("igLoop")
                if len(render_tasks) != 1:
                    raise RuntimeError("Expected exactly one Panda3D render task")
                render_task = render_tasks[0]
                original_render = render_task.getFunction()
                draw_state = {"due": True}
                frame_rate = FrameRate(time.perf_counter())

                def paced_render(task):
                    if controls["view_mode"] == "smooth" or draw_state["due"] or env.in_stop:
                        result = original_render(task)
                        frame_rate.record(time.perf_counter())
                        return result
                    return task.cont

                render_task.setFunction(paced_render)
                events = DirectObject()
                events.accept("n", set_control, ["next"])
                events.accept("a", set_control, ["next_agent"])
                events.accept("q", set_control, ["quit"])
                events.accept("m", toggle_view_mode)
                for key, value in (("1", 1.), ("2", 2.), ("5", 5.), ("0", 0.)):
                    events.accept(key, set_speed, [value])
                def watch_stop(task):
                    if (output / "STOP").exists():
                        set_control("quit")
                    return task.cont
                env.engine.taskMgr.add(watch_stop, "search_viewer_stop_file")
                agent = registry.get_factory(agent_id)()
                agent.reset(AgentPublicEpisodeContext(horizon_steps=case.horizon_steps,
                            input_profile_id="STATE_DECISION_V1", action_adapter_id="discrete9_lowbranch_v1"))
                last_hud_update = -float("inf")
                last_draw = -float("inf")
                last_draw_step = -1000
                rate_started, rate_step, observed_speed = time.perf_counter(), 0, 0.
                while step < case.horizon_steps:
                    if controls["quit"] or controls["next"] or controls["next_agent"] or (output / "STOP").exists():
                        result_status = "SKIPPED" if controls["next"] or controls["next_agent"] else "INTERRUPTED"
                        break
                    started = time.perf_counter()
                    draw_stride = max(1, int(playback.speed or 10))
                    draw = (controls["view_mode"] == "smooth" or (step - last_draw_step >= draw_stride and started - last_draw >= 1. / args.render_fps)
                            or step + 1 == args.capture_step)
                    # Keep Panda's event/tasks and every physics step running, while
                    # avoiding a costly GPU draw for every accelerated decision.
                    draw_state["due"] = draw
                    if draw:
                        last_draw = started
                        last_draw_step = step
                    traffic = list(getattr(env.engine.traffic_manager, "traffic_vehicles", []))
                    inp = build_agent_input(raw_obs=obs, ego_vehicle=env.agent, traffic_vehicles=traffic,
                                            navigation=env.agent.navigation, road_network=env.current_map.road_network,
                                            step_index=step, profile_id=InputProfileId.STATE_DECISION_V1)
                    t0 = time.perf_counter()
                    decision = agent.act(inp)
                    latency = (time.perf_counter() - t0) * 1000
                    action = adapter.to_canonical(decision.action_payload)
                    trace = explain_search_decision(agent_id, decision, action)
                    obs, _, terminated, truncated, info = env.step(action.to_numpy())
                    step += 1
                    total_decisions += 1
                    for name in ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "max_step"):
                        if info.get(name, False):
                            flags[name] = True
                    outcome = classify_episode_outcome(raw_flags=flags, terminated=terminated,
                                                      truncated=truncated or step >= case.horizon_steps)
                    speed, route = read_vehicle_speed_kmh(env.agent), float(info.get("route_completion", 0.))
                    now = time.perf_counter()
                    if now - rate_started >= 1.:
                        observed_speed = (step - rate_step) * .1 / (now - rate_started)
                        rate_started, rate_step = now, step
                    if now - last_hud_update >= .1 or step == args.capture_step or outcome.terminated or outcome.truncated:
                        last_hud_update = now
                        hud.update(step_idx=step, control_dt_s=.1, canonical_steering=action.steering,
                               canonical_throttle_brake=action.throttle_brake, step_info=info,
                               outcome=outcome if outcome.terminated or outcome.truncated else None,
                               agent_input=inp, agent_decision=decision, agent_id=agent_id, act_latency_ms=latency)
                        overlay.update(inp.step_index, trace, speed, route,
                                       playback_label=playback.label, observed_speed=observed_speed,
                                       render_fps=frame_rate.fps, view_mode=controls["view_mode"])
                    record = {"episode": index, "case_id": case.case_id, "split": case.split,
                              "agent_id": agent_id, "input_step": inp.step_index, "input_time_s": inp.step_index * .1,
                              "post_step": step, "public_traffic_count": inp.traffic_context.active_count,
                              "diagnostics": decision.diagnostics, "action": action.to_dict(),
                              "explanation": trace, "act_latency_ms": latency,
                              "playback_speed": playback.speed, "observed_speed": observed_speed,
                              "rendered_frame": draw,
                              "render_fps": frame_rate.fps, "view_mode": controls["view_mode"],
                              "post_speed_kmh": speed, "post_route_completion": route}
                    json_log.write(json.dumps(make_jsonable(record), ensure_ascii=False) + "\n")
                    line = f"[{case.split} {label} {case.case_id} t={inp.step_index * .1:.1f}s] {trace['text']} | {speed:.1f} km/h | route={route:.1%}"
                    text_log.write(line + "\n")
                    if step % 10 == 0 or step == 1:
                        print(line, flush=True)
                    if step == args.capture_step:
                        env.engine.graphicsEngine.renderFrame()
                        env.engine.graphicsEngine.renderFrame()
                        env.engine.win.saveScreenshot(Filename.fromOsSpecific(str(output / f"episode_{index:03d}.png")))
                    if args.smoke_steps and total_decisions >= args.smoke_steps:
                        controls["quit"] = True
                        break
                    if outcome.terminated or outcome.truncated:
                        result_status = "COMPLETE"
                        break
                    if args.preview_steps and step >= args.preview_steps:
                        result_status = "PREVIEW_LIMIT"
                        break
                    time.sleep(playback.delay(time.perf_counter() - started))
                if controls["next_agent"]:
                    skip_agent = (case.split, agent_id)
            except (KeyboardInterrupt, SystemExit):
                controls["quit"] = True
            except Exception:
                result_status = "ERROR"
                raise
            finally:
                result = {"episode": index, "agent_id": agent_id, "case": case.to_dict(),
                          "status": result_status, "steps": step, "route_completion": route,
                          "outcome": str(outcome.primary_reason.value) if outcome else None,
                          "clean_success": outcome.clean_success if outcome else False}
                results.write(json.dumps(result, ensure_ascii=False) + "\n")
                log(f"END {index} | {label} | {result_status} | {result['outcome']} | route={route:.1%}")
                if result_status == "COMPLETE":
                    manifest["completed_episodes"] += 1
                if events is not None:
                    events.ignoreAll()
                if overlay is not None:
                    overlay.destroy()
                if hud is not None:
                    hud.destroy()
                if agent is not None:
                    agent.close()
                if env is not None:
                    env.close()
                save_manifest()
        manifest["status"] = "COMPLETE" if manifest["completed_episodes"] == len(jobs) else "INTERRUPTED"
    except Exception as error:
        manifest["status"] = "ERROR"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["total_decisions"] = total_decisions
        save_manifest()
        text_log.close()
        json_log.close()
        results.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
