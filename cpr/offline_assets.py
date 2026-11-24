"""Alle Isaac-Sim-Asset-Wurzeln auf das lokale Assetpack auf BIGWORK umbiegen.

Verwendung in JEDEM Trainings-/Play-Skript - die Reihenfolge ist kritisch:

    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    import cpr.offline_assets as offline_assets
    offline_assets.patch_launcher_args(args_cli)      # VOR AppLauncher!

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    offline_assets.apply()                            # nach App-Start

    import gymnasium as gym                           # erst JETZT isaaclab-Module
    import isaaclab.sim as sim_utils
"""

import os

DEFAULT_ROOT = "/bigwork/nhkbrc2a/isaac/assets/Assets/Isaac/5.1"


def asset_root() -> str:
    root = os.environ.get("ISAACSIM_ASSET_ROOT", DEFAULT_ROOT)
    if not os.path.isdir(root):
        raise RuntimeError(f"[offline_assets] local asset root not found: {root}")
    return root


def patch_launcher_args(args_cli):
    """Asset-Root-Overrides als kit-CLI-Argumente anhaengen.

    Muss VOR dem AppLauncher passieren: isaaclab.utils.assets loest die
    Cloud-Wurzel bereits beim Import waehrend des App-Starts auf. Ein
    nachtraegliches Setzen der Carb-Settings kommt zu spaet.
    """
    root = asset_root()
    extra = " ".join(
        f"--/persistent/isaac/asset_root/{key}={root}" for key in ("default", "cloud", "nvidia")
    )
    prev = getattr(args_cli, "kit_args", None) or ""
    args_cli.kit_args = (prev + " " + extra).strip()
    return args_cli


def apply():
    """Guertel und Hosentraeger: dieselben Settings nach dem App-Start nochmal."""
    import carb

    root = asset_root()
    settings = carb.settings.get_settings()
    for key in ("default", "cloud", "nvidia"):
        settings.set(f"/persistent/isaac/asset_root/{key}", root)
        settings.set(f"/isaac/asset_root/{key}", root)
    print(f"[offline_assets] asset roots -> {root}")
    return root
