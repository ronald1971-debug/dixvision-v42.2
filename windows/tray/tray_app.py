"""
windows/tray/tray_app.py
DIX VISION v42.2 — System Tray Application

Shows status icon in Windows system tray.
Right-click menu: Status | Kill Switch | Open Dashboard
"""
from __future__ import annotations


def run_tray() -> None:
    """Run the tray application. Requires pystray + Pillow."""
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        print("Tray: pystray/Pillow not installed. Skipping tray icon.")
        return

    def create_icon(color: str) -> Image.Image:
        img = Image.new("RGB", (64, 64), color)
        d = ImageDraw.Draw(img)
        d.text((10, 20), "DV", fill="white")
        return img

    def on_status(icon, item):
        from system.state import get_state
        state = get_state()
        print(f"Status: mode={state.mode} health={state.health:.2f}")

    def on_kill(icon, item):
        from immutable_core.kill_switch import trigger_kill_switch
        trigger_kill_switch("operator_tray_kill", "tray_app")

    def on_quit(icon, item):
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Status", on_status),
        pystray.MenuItem("Kill Switch", on_kill),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon("DIX_VISION", create_icon("#1a73e8"), "DIX VISION v42.2", menu)
    icon.run()

if __name__ == "__main__":
    run_tray()
