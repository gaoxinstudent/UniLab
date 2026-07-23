"""MuJoCo GL backend selection helpers.

This module intentionally does not import ``mujoco`` at module import time.
MuJoCo reads ``MUJOCO_GL`` during import, so backend owners must configure the
environment first.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable

_GL_PROBE_SCRIPT = textwrap.dedent(
    '''
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <geom type="box" size="0.1 0.1 0.1" rgba="0 1 0 1"/>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=8, width=8)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    renderer.render()
    renderer.close()
    '''
)


def gl_backend_runtime_usable(backend: str) -> bool:
    """Return True if *backend* can create and render an off-screen scene."""
    if not backend:
        return False

    env = os.environ.copy()
    env["MUJOCO_GL"] = backend
    if backend == "egl":
        env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    try:
        subprocess.run(
            [sys.executable, "-c", _GL_PROBE_SCRIPT],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    if backend == "egl":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", env["MUJOCO_EGL_DEVICE_ID"])
    return True


def egl_runtime_usable() -> bool:
    """Probe the EGL backend specifically."""
    return gl_backend_runtime_usable("egl")


def resolve_mujoco_gl_backend(
    *,
    platform: str | None = None,
    egl_runtime_usable_fn: Callable[[], bool] | None = None,
) -> str:
    """Pick a valid ``MUJOCO_GL`` backend before importing MuJoCo.

    Linux off-screen playback prefers EGL, then OSMesa. GLFW is preserved only
    when explicitly configured because unstable X11 forwarding can terminate the
    process with fatal Xlib errors that Python cannot catch.
    """
    current = os.environ.get("MUJOCO_GL", "")
    target_platform = sys.platform if platform is None else platform

    if target_platform == "darwin":
        return current if current in {"glfw", "disabled"} else "glfw"

    if target_platform == "win32":
        return current if current in {"glfw", "disabled"} else "glfw"

    if current in {"glfw", "osmesa", "disabled"}:
        return current

    probe_egl = egl_runtime_usable if egl_runtime_usable_fn is None else egl_runtime_usable_fn
    if probe_egl():
        return "egl"

    return "osmesa"


def ensure_mujoco_gl_backend() -> str:
    """Resolve and set ``MUJOCO_GL`` before importing MuJoCo."""
    backend = resolve_mujoco_gl_backend()
    os.environ["MUJOCO_GL"] = backend
    return backend
