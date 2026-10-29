"""
Pick which StarCraft II install burnysc2 launches, before `sc2` is imported.

burnysc2 reads SC2PF / SC2PATH when `sc2.paths` is first imported, so entry
scripts call apply_sc2_runtime_args() as their very first import:

    from sc2_rl_agent.starcraftenv_test.sc2_runtime import apply_sc2_runtime_args
    apply_sc2_runtime_args()

--sc2_platform Linux runs the headless Linux build (no window, nothing
rendered); WSL2 / Windows run the Windows client with its game window.
Without the flags, whatever SC2PF / SC2PATH the shell exports is used, and
on a plain Linux server burnysc2 defaults to Linux and ~/StarCraftII.
"""

import argparse
import glob
import os
import platform
import sys

PLATFORMS = ("Linux", "WSL1", "WSL2", "Windows", "Darwin", "WineLinux")


def add_sc2_runtime_args(parser):
    parser.add_argument('--sc2_platform', type=str, default=None, choices=PLATFORMS,
                        help='SC2 build to launch. "Linux" = headless Linux build. '
                             'Default: $SC2PF, else auto-detected by burnysc2.')
    parser.add_argument('--sc2_path', type=str, default=None,
                        help='SC2 install folder. Default: $SC2PATH, else burnysc2 default '
                             '(~/StarCraftII on Linux).')
    return parser


def add_game_run_args(parser):
    """Per-game options passed to burnysc2's run_game (Protoss launchers)."""
    parser.add_argument('--game_time_limit', type=int, default=None,
                        help='End the game as a Tie after this many in-game seconds. Default: no limit.')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='SC2 random seed for reproducible games. Default: SC2 picks one.')
    parser.add_argument('--bot_rules', type=str, default='dynamic', choices=['dynamic', 'legacy'],
                        help='Protoss bot strategy rules. dynamic (default): supply-driven pylons, no '
                             'time-window building limits, no scripted 12:00 attack. legacy: the original '
                             'hard-coded rules, for comparing with older games.')
    return parser


def apply_sc2_runtime_args(argv=None):
    """Read only the SC2 flags from argv and export them for burnysc2."""
    parser = add_sc2_runtime_args(argparse.ArgumentParser(add_help=False))
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if 'sc2' in sys.modules and (known.sc2_platform or known.sc2_path):
        print("[sc2_runtime] warning: sc2 was imported before apply_sc2_runtime_args(); "
              "--sc2_platform/--sc2_path may be ignored")
    if known.sc2_platform:
        os.environ['SC2PF'] = known.sc2_platform
        if known.sc2_platform == 'Linux':
            # burnysc2 turns "Linux" into "WSL2" when it detects WSL and then
            # launches the Windows client; this opt-out keeps the headless build.
            os.environ['SC2_WSL_DETECT'] = '0'
            # The headless build runs on this machine. A WSL setup for the Windows
            # client exports the Windows host IP here, which would never connect.
            os.environ['SC2CLIENTHOST'] = '127.0.0.1'
            os.environ['SC2SERVERHOST'] = '127.0.0.1'
    if known.sc2_path:
        os.environ['SC2PATH'] = os.path.expanduser(known.sc2_path)
    return runtime_info()


def runtime_info():
    """What will be launched, for logging and the telemetry run config."""
    pf = os.environ.get('SC2PF', platform.system())
    path = os.environ.get('SC2PATH')
    if not path and pf == 'Linux':
        path = os.path.expanduser('~/StarCraftII')
    builds = sorted(glob.glob(os.path.join(path, 'Versions', 'Base*'))) if path else []
    return {
        'sc2_platform': pf,
        'sc2_path': path,
        'sc2_base_build': os.path.basename(builds[-1])[len('Base'):] if builds else None,
        'sc2_headless': pf == 'Linux',
    }


def check_install(map_name=None):
    """Fail early with a readable message instead of a burnysc2 traceback."""
    info = runtime_info()
    path = info['sc2_path']
    problems = []
    if not path or not os.path.isdir(path):
        problems.append(f"SC2 folder not found: {path!r} (set --sc2_path or $SC2PATH)")
    elif info['sc2_platform'] == 'Linux':
        if not glob.glob(os.path.join(path, 'Versions', 'Base*', 'SC2_x64')):
            problems.append(f"no Linux SC2 binary under {path}/Versions/Base*/SC2_x64 "
                            f"(run scripts/setup_linux_sc2.sh)")
    if map_name and path and os.path.isdir(path):
        maps = glob.glob(os.path.join(path, 'Maps', '**', f'{map_name}.SC2Map'), recursive=True)
        if not maps:
            problems.append(f"map {map_name!r} not found under {path}/Maps "
                            f"(copy it from the repo's Maps/ folder)")
    if problems:
        raise SystemExit("[sc2_runtime] " + "\n[sc2_runtime] ".join(problems))
    return info
