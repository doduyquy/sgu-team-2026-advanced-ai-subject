"""Open the Vietnamese algorithm garage directly inside MetaDrive."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Exercise simultaneous 5/2/1/3-car races, 3 roads, pause, focus and persistent window, then exit")
    parser.add_argument("--steps", type=int, default=12, help="Decisions per smoke-test run (>= 4)")
    parser.add_argument("--output", type=Path, help="New log directory (must not exist)")
    parser.add_argument('--phone', action='store_true', help='Open the phone QR host in human racing mode')
    parser.add_argument('--mode', choices=('single','compare','human','practice'), help='Initial garage mode')
    parser.add_argument('--agent',choices=('astar','bfs','dijkstra','hybrid_astar','mppi'),help='Initial algorithm in single mode')
    parser.add_argument('--car',choices=('lamborghini','tesla','vf3'),help='Initial selected/player car model')
    parser.add_argument('--port', type=int, default=8765, help='LAN phone HTTP port')
    parser.add_argument('--advertise-host', help='LAN IP shown in QR when automatic routing chooses another adapter')
    parser.add_argument('--encoder',choices=('auto','nvenc','qsv','cpu'),default='nvenc',help='H.264 backend; nvenc strictly requires NVIDIA, auto permits CPU fallback')
    parser.add_argument('--legacy-phone',action='store_true',help='Use old lockstep phone physics for comparison')
    parser.add_argument('--lockstep',action='store_true',help='Use synchronous 0.1 s diagnostic execution instead of smooth interactive mode')
    parser.add_argument('--map-set', choices=('original','train'), default='original',
                        help='Archived showcase maps (default) or previous TRAIN presets')
    parser.add_argument('--road', choices=('easy','medium','hard','vietnam'), help='Initial road; includes the 10 km Vietnam OSM corridor')
    parser.add_argument('--vehicle', choices=('lamborghini','tesla','vf3'),
                        help='Initial vehicle displayed in the garage')
    parser.add_argument('--graphics', choices=('lite', 'balanced'), default='lite',
                        help='Local GPU budget (default: lite, native-resolution HUD with a lighter 3D scene)')
    args = parser.parse_args()
    if args.steps < 4:
        parser.error("--steps must be >= 4")
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    # Import Qt first on this Windows environment, before Panda3D DLLs.
    from PySide6.QtCore import QProcess, QLockFile  # noqa: F401
    from PySide6.QtWidgets import QApplication  # noqa: F401
    # A second full 3D world can exhaust this laptop's 4 GB VRAM. Hold the
    # lock for the entire run; QLockFile recovers it if its owner has exited.
    lock_path=PROJECT_ROOT/'artifacts'/'race-mode.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    instance_lock=QLockFile(str(lock_path))
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(0):
        print('Race Mode dang mo. Hay dung cua so game hien tai; khong mo them phien 3D.',flush=True)
        return 0
    from src.training.runtime import local_training_runtime
    from src.visualization.race_mode import RaceApp, configure_race_graphics
    configure_race_graphics(args.graphics)
    with local_training_runtime(PROJECT_ROOT):
        app = RaceApp(PROJECT_ROOT, args.output, args.steps if args.smoke_test else 0)
        app.map_set=args.map_set
        app.selection.road_index=('easy','medium','hard','vietnam').index(args.road) if args.road else (1 if args.map_set=='original' else 0)
        app.selection.set_mode(args.mode or ('human' if args.phone else 'compare'))
        if args.agent:
            app.selection.agent_index=('astar','bfs','dijkstra','hybrid_astar','mppi').index(args.agent)
        if args.car:
            app.selection.vehicle_models[app.selection.agent_index]=args.car
            app.selection.player_model=args.car
        if args.vehicle:
            app.selection.player_model=args.vehicle
            app.selection.vehicle_models[app.selection.agent_index]=args.vehicle
        app.phone_port,app.phone_address = args.port,args.advertise_host
        app.startup_phone = args.phone
        app.phone_encoder=args.encoder
        app.realtime_enabled=not (args.legacy_phone or args.lockstep)
        app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
