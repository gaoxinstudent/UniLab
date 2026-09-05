# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Adapted from wheeled-legged_RL/scripts/utils/{realtime_plotter,
# velocity_trace_html,export_velocity_trace_html}.py and the V14 trace owner at
# the pinned revision recorded in THIRD_PARTY_NOTICES.md.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================
"""WheelBipe play telemetry, bounded live data, and offline trace export."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

WHEELBIPE_TRACE_FIELDS: tuple[str, ...] = (
    "sim_time_s",
    "episode_time_s",
    "env_id",
    "terrain",
    "cmd_x",
    "cmd_y",
    "cmd_yaw",
    "vel_x_b",
    "vel_y_b",
    "yaw_rate_b",
    "height_cmd",
    "height_obs",
    "height_relative",
    "height_reward_ref",
    "airborne",
    "reward_total",
)
_TRACE_STRING_FIELDS = frozenset({"terrain"})
_TRACE_INTEGER_FIELDS = frozenset({"env_id", "airborne"})
_REWARD_COLUMN_RE = re.compile(r"^reward_[A-Za-z0-9_.-]+$")


def _reward_column_name(value: str) -> str:
    name = str(value)
    if not name.startswith("reward_"):
        name = f"reward_{name}"
    if name == "reward_total" or not _REWARD_COLUMN_RE.fullmatch(name):
        raise ValueError(f"invalid WheelBipe reward trace column {value!r}")
    return name


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"WheelBipe trace {name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WheelBipe trace {name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"WheelBipe trace {name} must be finite, got {value!r}")
    return result


def normalize_wheelbipe_trace_row(
    row: Mapping[str, Any],
    *,
    reward_columns: Sequence[str] | None = None,
) -> dict[str, float | int | str]:
    """Validate and normalize one source-compatible trace row.

    Only the fixed V14 schema and syntactically safe ``reward_*`` extensions
    are accepted.  This keeps CSV headers stable and prevents malformed values
    from reaching the offline JavaScript renderer.
    """

    if not isinstance(row, Mapping):
        raise ValueError("WheelBipe trace row must be a mapping")
    missing = [name for name in WHEELBIPE_TRACE_FIELDS if name not in row]
    if missing:
        raise ValueError("WheelBipe trace row is missing fields: " + ", ".join(missing))
    discovered = tuple(
        key
        for key in row
        if isinstance(key, str) and key.startswith("reward_") and key != "reward_total"
    )
    requested = (
        tuple(_reward_column_name(name) for name in reward_columns)
        if reward_columns is not None
        else discovered
    )
    allowed = set(WHEELBIPE_TRACE_FIELDS) | set(requested)
    unexpected = [str(key) for key in row if key not in allowed]
    if unexpected:
        raise ValueError("WheelBipe trace row has unsupported fields: " + ", ".join(unexpected))

    result: dict[str, float | int | str] = {}
    for name in WHEELBIPE_TRACE_FIELDS:
        value = row[name]
        if name in _TRACE_STRING_FIELDS:
            result[name] = str(value)
        elif name in _TRACE_INTEGER_FIELDS:
            number = _finite_number(value, name=name)
            if not number.is_integer():
                raise ValueError(f"WheelBipe trace {name} must be an integer, got {value!r}")
            integer = int(number)
            if name == "env_id" and integer < 0:
                raise ValueError("WheelBipe trace env_id must be non-negative")
            if name == "airborne" and integer not in (0, 1):
                raise ValueError("WheelBipe trace airborne must be zero or one")
            result[name] = integer
        else:
            result[name] = _finite_number(value, name=name)
    for name in requested:
        result[name] = _finite_number(row.get(name, 0.0), name=name)
    return result


def normalize_wheelbipe_trace_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int | str]]:
    """Validate rows with one stable reward-column schema and time ordering."""

    raw_rows = list(rows)
    reward_columns: list[str] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise ValueError("WheelBipe trace rows must contain mappings")
        for key in row:
            if isinstance(key, str) and key.startswith("reward_") and key != "reward_total":
                column = _reward_column_name(key)
                if column not in reward_columns:
                    reward_columns.append(column)
    normalized = [
        normalize_wheelbipe_trace_row(row, reward_columns=reward_columns) for row in raw_rows
    ]
    previous = -math.inf
    for row in normalized:
        current = float(row["sim_time_s"])
        if current < previous:
            raise ValueError("WheelBipe trace sim_time_s must be non-decreasing")
        previous = current
    return normalized


def build_wheelbipe_reward_signs(
    reward_scales: Mapping[str, float] | None = None,
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Build heatmap signs from reward scales with trace-column fallback."""

    signs = {"reward_total": 0}
    if reward_scales:
        for key, weight in reward_scales.items():
            column = _reward_column_name(str(key))
            value = _finite_number(weight, name=f"reward scale {key}")
            signs[column] = 1 if value > 0.0 else -1 if value < 0.0 else 0
    if rows:
        for row in rows:
            for key in row:
                if isinstance(key, str) and key.startswith("reward_"):
                    if key != "reward_total":
                        _reward_column_name(key)
                    signs.setdefault(key, 0)
    return signs


def _script_safe_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


_TRACE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WheelBipe velocity and reward trace</title>
<style>
:root { color-scheme: dark; font-family: system-ui,sans-serif; }
body { margin:0; background:#050607; color:#e8edf2; }
#bar { position:sticky; top:0; z-index:2; display:flex; gap:14px; flex-wrap:wrap;
  align-items:center; padding:10px 14px; background:#0d1117; border-bottom:1px solid #263241; }
label { display:inline-flex; gap:5px; align-items:center; }
button { color:#e8edf2; background:#1f2937; border:1px solid #3b4656; border-radius:4px;
  padding:4px 9px; }
#plot { display:block; width:100%; height:calc(100vh - 54px); min-height:620px; cursor:grab; }
#plot.drag { cursor:grabbing; }
</style>
</head>
<body>
<div id="bar"><strong>WheelBipe trace</strong><span id="meta"></span>
<span id="toggles"></span><button id="reset">Reset zoom</button></div>
<canvas id="plot"></canvas>
<script>
"use strict";
const rows=__ROWS__;
const rewardSigns=__SIGNS__;
const velocityKeys=["cmd_x","vel_x_b","cmd_yaw","yaw_rate_b"];
const heightKeys=["height_cmd","height_reward_ref","height_obs","height_relative"];
const rewardKeys=Object.keys(rewardSigns).filter(k=>k.startsWith("reward_"));
const colors={cmd_x:"#f05252",vel_x_b:"#4b91e2",cmd_yaw:"#f6ad55",yaw_rate_b:"#55c983",
  height_cmd:"#c084fc",height_reward_ref:"#f4d35e",height_obs:"#ff7ab6",height_relative:"#8bd3ff"};
const visible=Object.fromEntries([...velocityKeys,...heightKeys].map(k=>[k,true]));
const canvas=document.getElementById("plot"),ctx=canvas.getContext("2d");
const full=[rows.length?rows[0].sim_time_s:0,rows.length?rows.at(-1).sim_time_s:1];
if(full[1]<=full[0]) full[1]=full[0]+1;
let xMin=full[0],xMax=full[1],dragging=false,lastX=0,hover=null;
document.getElementById("meta").textContent=rows.length
  ? `rows=${rows.length} env=${rows.at(-1).env_id} terrain=${rows.at(-1).terrain}`:"no data";
const toggles=document.getElementById("toggles");
for(const key of [...velocityKeys,...heightKeys]){
  const label=document.createElement("label"),box=document.createElement("input");
  box.type="checkbox";box.checked=true;box.addEventListener("change",()=>{visible[key]=box.checked;draw();});
  label.append(box,document.createTextNode(key));toggles.append(label);
}
function resize(){const d=devicePixelRatio||1;canvas.width=Math.max(1,canvas.clientWidth*d);
  canvas.height=Math.max(1,canvas.clientHeight*d);draw();}
function selected(){return rows.filter(r=>r.sim_time_s>=xMin&&r.sim_time_s<=xMax);}
function range(rs,keys){const values=[];for(const r of rs)for(const k of keys)
  if(visible[k]&&Number.isFinite(Number(r[k])))values.push(Number(r[k]));
  if(!values.length)return[-1,1];let lo=Math.min(...values),hi=Math.max(...values);
  if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1;}const pad=(hi-lo)*.08;return[lo-pad,hi+pad];}
function linePanel(title,keys,rs,x,y,w,h){const [lo,hi]=range(rs,keys),d=devicePixelRatio||1;
  const sx=t=>x+(t-xMin)/Math.max(xMax-xMin,1e-9)*w,sy=v=>y+(hi-v)/(hi-lo)*h;
  ctx.strokeStyle="#344154";ctx.strokeRect(x,y,w,h);ctx.fillStyle="#d7dee8";
  ctx.font=`${13*d}px sans-serif`;ctx.fillText(title,x+8*d,y+18*d);
  for(let i=0;i<=4;i++){const yy=y+h*i/4;ctx.strokeStyle="#17202c";ctx.beginPath();
    ctx.moveTo(x,yy);ctx.lineTo(x+w,yy);ctx.stroke();ctx.fillStyle="#9aa7b5";
    ctx.fillText((hi-(hi-lo)*i/4).toFixed(3),8*d,yy+4*d);}
  let lx=x+8*d;for(const key of keys){if(!visible[key])continue;ctx.strokeStyle=colors[key];
    ctx.lineWidth=2*d;ctx.beginPath();let started=false;for(const r of rs){const v=Number(r[key]);
      if(!Number.isFinite(v))continue;const xx=sx(r.sim_time_s),yy=sy(v);started?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);started=true;}
    ctx.stroke();ctx.fillStyle=colors[key];ctx.fillText(key,lx,y+37*d);lx+=Math.max(92,key.length*8+20)*d;}
  if(hover){const xx=sx(hover.sim_time_s);ctx.strokeStyle="#ffffff99";ctx.beginPath();
    ctx.moveTo(xx,y);ctx.lineTo(xx,y+h);ctx.stroke();let ty=y+56*d;for(const key of keys){
      if(!visible[key])continue;ctx.fillStyle=colors[key];ctx.fillText(`${key}=${Number(hover[key]).toFixed(4)}`,x+8*d,ty);ty+=15*d;}}
}
function heatmap(rs,x,y,w,h){if(!rewardKeys.length)return;const d=devicePixelRatio||1;
  ctx.fillStyle="#d7dee8";ctx.fillText("reward heatmap",x,y-7*d);const rh=h/rewardKeys.length;
  rewardKeys.forEach((key,j)=>{let max=0;for(const r of rs)max=Math.max(max,Math.abs(Number(r[key])||0));
    ctx.fillStyle="#a9b4c2";ctx.fillText(key.replace("reward_",""),8*d,y+(j+.7)*rh);
    const cw=w/Math.max(rs.length,1);rs.forEach((r,i)=>{const v=Number(r[key])||0,t=max?Math.sqrt(Math.abs(v)/max):0;
      const sign=rewardSigns[key]||Math.sign(v);ctx.fillStyle=sign<0?`rgb(${70+150*t},${35-20*t},${35-20*t})`
        :`rgb(${20-12*t},${65+90*t},${40-22*t})`;ctx.fillRect(x+i*cw,y+j*rh,Math.ceil(cw),Math.max(1,rh-1));});});
  ctx.strokeStyle="#344154";ctx.strokeRect(x,y,w,h);
}
function nearest(t){if(!rows.length)return null;let best=rows[0];for(const r of rows){
  if(Math.abs(r.sim_time_s-t)<Math.abs(best.sim_time_s-t))best=r;}return best;}
function draw(){const d=devicePixelRatio||1,w=canvas.width,h=canvas.height,padL=112*d,padR=22*d;
  ctx.fillStyle="#050607";ctx.fillRect(0,0,w,h);const rs=selected(),pw=w-padL-padR;
  const heatH=rewardKeys.length?Math.max(90*d,rewardKeys.length*20*d):0,gap=28*d;
  const lineH=(h-52*d-heatH-(heatH?gap:0))/2;linePanel("velocity / yaw",velocityKeys,rs,padL,18*d,pw,lineH);
  linePanel("height",heightKeys,rs,padL,30*d+lineH,pw,lineH);
  if(heatH)heatmap(rs,padL,44*d+2*lineH+gap,pw,heatH);}
function pointerTime(e){const rect=canvas.getBoundingClientRect(),ratio=(e.clientX-rect.left)/rect.width;
  return xMin+Math.max(0,Math.min(1,ratio))*(xMax-xMin);}
canvas.addEventListener("mousemove",e=>{hover=nearest(pointerTime(e));if(dragging){const dx=e.clientX-lastX;
  lastX=e.clientX;const shift=-dx/canvas.clientWidth*(xMax-xMin);xMin+=shift;xMax+=shift;}draw();});
canvas.addEventListener("mouseleave",()=>{if(!dragging){hover=null;draw();}});
canvas.addEventListener("mousedown",e=>{dragging=true;lastX=e.clientX;canvas.classList.add("drag");});
window.addEventListener("mouseup",()=>{dragging=false;canvas.classList.remove("drag");});
canvas.addEventListener("wheel",e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();const center=pointerTime(e),
  ratio=(center-xMin)/(xMax-xMin),span=Math.max(.02,(xMax-xMin)*(e.deltaY<0?.82:1.2));
  xMin=center-span*ratio;xMax=center+span*(1-ratio);draw();},{passive:false});
document.getElementById("reset").onclick=()=>{xMin=full[0];xMax=full[1];draw();};
window.addEventListener("resize",resize);resize();
</script>
</body>
</html>
"""


def build_wheelbipe_trace_html(
    rows: Sequence[Mapping[str, Any]],
    reward_signs: Mapping[str, int] | None = None,
) -> str:
    """Render a self-contained, pan/zoom/hover interactive trace HTML."""

    normalized = normalize_wheelbipe_trace_rows(rows)
    signs = dict(reward_signs or build_wheelbipe_reward_signs(rows=normalized))
    for key, raw_sign in tuple(signs.items()):
        if key != "reward_total":
            _reward_column_name(key)
        try:
            sign = int(raw_sign)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"WheelBipe reward sign for {key!r} must be -1, 0, or 1") from exc
        if sign not in (-1, 0, 1):
            raise ValueError(f"WheelBipe reward sign for {key!r} must be -1, 0, or 1")
        signs[key] = sign
    for key in build_wheelbipe_reward_signs(rows=normalized):
        signs.setdefault(key, 0)
    return _TRACE_HTML.replace("__ROWS__", _script_safe_json(normalized)).replace(
        "__SIGNS__", _script_safe_json(signs)
    )


def load_wheelbipe_trace_csv(path: str | Path) -> list[dict[str, float | int | str]]:
    """Load and strictly validate a WheelBipe velocity/reward CSV."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"WheelBipe trace CSV has no header: {source}")
        missing = [name for name in WHEELBIPE_TRACE_FIELDS if name not in reader.fieldnames]
        if missing:
            raise ValueError("WheelBipe trace CSV is missing columns: " + ", ".join(missing))
        reward_columns = [
            name
            for name in reader.fieldnames
            if name.startswith("reward_") and name != "reward_total"
        ]
        for name in reward_columns:
            _reward_column_name(name)
        rows = [normalize_wheelbipe_trace_row(row, reward_columns=reward_columns) for row in reader]
    return normalize_wheelbipe_trace_rows(rows)


def load_wheelbipe_reward_signs(
    path: str | Path | None,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Load either reward-column signs or owner reward scales from JSON."""

    if path is None:
        return build_wheelbipe_reward_signs(rows=rows)
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"WheelBipe reward-sign JSON is invalid: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError("WheelBipe reward-sign JSON must contain an object")
    if raw and all(str(key).startswith("reward_") for key in raw):
        signs: dict[str, int] = {}
        for key, value in raw.items():
            name = str(key)
            if name != "reward_total":
                _reward_column_name(name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"WheelBipe reward sign for {name!r} must be -1, 0, or 1")
            try:
                sign = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"WheelBipe reward sign for {name!r} must be -1, 0, or 1") from exc
            if sign not in (-1, 0, 1):
                raise ValueError(f"WheelBipe reward sign for {name!r} must be -1, 0, or 1")
            signs[name] = sign
        for key in build_wheelbipe_reward_signs(rows=rows):
            signs.setdefault(key, 0)
        return signs
    return build_wheelbipe_reward_signs(
        {str(key): _finite_number(value, name=f"reward scale {key}") for key, value in raw.items()},
        rows=rows,
    )


def export_wheelbipe_trace_html(
    csv_path: str | Path,
    *,
    html_path: str | Path | None = None,
    reward_signs_path: str | Path | None = None,
) -> Path:
    """Export a validated trace CSV to a standalone HTML file."""

    source = Path(csv_path).expanduser().resolve()
    destination = (
        source.with_suffix(".html") if html_path is None else Path(html_path).expanduser().resolve()
    )
    rows = load_wheelbipe_trace_csv(source)
    signs = load_wheelbipe_reward_signs(reward_signs_path, rows=rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_wheelbipe_trace_html(rows, signs), encoding="utf-8")
    return destination


@dataclass(frozen=True)
class WheelbipeRealtimeSample:
    """One display-neutral sample matching the four upstream live panels."""

    step: int
    target_height: float
    actual_height: float
    jump_phase: float
    wheel_power: tuple[float, float]
    leg_torques: tuple[float, ...]
    spring_forces: tuple[float, float]

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or int(self.step) != self.step or int(self.step) < 0:
            raise ValueError("WheelBipe realtime step must be a non-negative integer")
        numeric = (
            self.target_height,
            self.actual_height,
            self.jump_phase,
            *self.wheel_power,
            *self.leg_torques,
            *self.spring_forces,
        )
        if not numeric or any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("WheelBipe realtime samples must contain only finite values")
        if len(self.wheel_power) != 2 or len(self.spring_forces) != 2:
            raise ValueError("WheelBipe realtime wheel_power and spring_forces must have width two")


class WheelbipeRealtimeBuffer:
    """Bounded, display-neutral live telemetry stream with subscribers."""

    def __init__(self, *, max_points: int = 200, num_leg_joints: int = 4) -> None:
        if isinstance(max_points, bool) or int(max_points) < 1:
            raise ValueError("WheelBipe realtime max_points must be positive")
        if isinstance(num_leg_joints, bool) or int(num_leg_joints) < 1:
            raise ValueError("WheelBipe realtime num_leg_joints must be positive")
        self.max_points = int(max_points)
        self.num_leg_joints = int(num_leg_joints)
        self._samples: deque[WheelbipeRealtimeSample] = deque(maxlen=self.max_points)
        self._subscribers: list[Callable[[WheelbipeRealtimeSample], None]] = []

    def subscribe(self, callback: Callable[[WheelbipeRealtimeSample], None]) -> None:
        if not callable(callback):
            raise TypeError("WheelBipe realtime subscriber must be callable")
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[WheelbipeRealtimeSample], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def append(self, sample: WheelbipeRealtimeSample) -> None:
        if not isinstance(sample, WheelbipeRealtimeSample):
            raise TypeError("WheelBipe realtime buffer accepts WheelbipeRealtimeSample values")
        if len(sample.leg_torques) != self.num_leg_joints:
            raise ValueError(
                "WheelBipe realtime leg_torques width does not match buffer contract: "
                f"expected {self.num_leg_joints}, got {len(sample.leg_torques)}"
            )
        self._samples.append(sample)
        for callback in tuple(self._subscribers):
            callback(sample)

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return copied arrays suitable for a renderer or remote UI."""

        samples = tuple(self._samples)
        return {
            "step": np.asarray([sample.step for sample in samples], dtype=np.int64),
            "target_height": np.asarray(
                [sample.target_height for sample in samples], dtype=np.float64
            ),
            "actual_height": np.asarray(
                [sample.actual_height for sample in samples], dtype=np.float64
            ),
            "jump_phase": np.asarray([sample.jump_phase for sample in samples], dtype=np.float64),
            "wheel_power": np.asarray(
                [sample.wheel_power for sample in samples], dtype=np.float64
            ).reshape(-1, 2),
            "leg_torques": np.asarray(
                [sample.leg_torques for sample in samples], dtype=np.float64
            ).reshape(-1, self.num_leg_joints),
            "spring_forces": np.asarray(
                [sample.spring_forces for sample in samples], dtype=np.float64
            ).reshape(-1, 2),
        }


class WheelbipeRealtimePlotter:
    """Optional lazy Matplotlib view over :class:`WheelbipeRealtimeBuffer`.

    No backend is selected at import time.  Headless callers can use the
    buffer alone; enabling this class requires a usable Matplotlib GUI backend.
    """

    def __init__(self, buffer: WheelbipeRealtimeBuffer, *, update_interval: int = 5) -> None:
        if isinstance(update_interval, bool) or int(update_interval) < 1:
            raise ValueError("WheelBipe realtime update_interval must be positive")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional display dependency
            raise RuntimeError("Matplotlib is required for WheelBipe realtime plotting") from exc
        self._plt = plt
        self.buffer = buffer
        self.update_interval = int(update_interval)
        self._updates = 0
        plt.ion()
        self.figure, self.axes = plt.subplots(4, 1, figsize=(10, 10))
        self.figure.suptitle("WheelBipe real-time telemetry")
        self._height_lines = self.axes[0].plot([], [], label="target") + self.axes[0].plot(
            [], [], label="actual"
        )
        self._phase_axis = self.axes[0].twinx()
        (self._phase_line,) = self._phase_axis.plot([], [], "--", label="phase")
        self._wheel_lines = self.axes[1].plot([], [], label="left") + self.axes[1].plot(
            [], [], label="right"
        )
        self._leg_lines = [
            self.axes[2].plot([], [], label=f"leg_{index}")[0]
            for index in range(buffer.num_leg_joints)
        ]
        self._spring_lines = self.axes[3].plot([], [], label="left") + self.axes[3].plot(
            [], [], label="right"
        )
        for axis, title, ylabel in zip(
            self.axes,
            ("height / jump phase", "wheel mechanical power", "leg torque", "spring force"),
            ("height (m)", "power (W)", "torque (N m)", "force (N)"),
            strict=True,
        ):
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
            axis.legend(loc="upper right")
        self.axes[-1].set_xlabel("control step")
        self.buffer.subscribe(self.update)
        self.figure.tight_layout()

    def update(self, _sample: WheelbipeRealtimeSample) -> None:
        self._updates += 1
        if self._updates % self.update_interval:
            return
        values = self.buffer.snapshot()
        x = values["step"]
        for line, key in zip(self._height_lines, ("target_height", "actual_height"), strict=True):
            line.set_data(x, values[key])
        self._phase_line.set_data(x, values["jump_phase"])
        for index, line in enumerate(self._wheel_lines):
            line.set_data(x, values["wheel_power"][:, index])
        for index, line in enumerate(self._leg_lines):
            line.set_data(x, values["leg_torques"][:, index])
        for index, line in enumerate(self._spring_lines):
            line.set_data(x, values["spring_forces"][:, index])
        for axis in (*self.axes, self._phase_axis):
            axis.relim()
            axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self) -> None:
        self.buffer.unsubscribe(self.update)
        self._plt.close(self.figure)


@dataclass(frozen=True)
class WheelbipePlaybackTelemetry:
    """Trace row plus its live-visualization sample."""

    trace_row: dict[str, float | int | str]
    realtime_sample: WheelbipeRealtimeSample


def _batch_row(value: Any, env_id: int, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 1 or env_id >= array.shape[0]:
        raise ValueError(f"WheelBipe telemetry {name} must be batched for env {env_id}")
    row = np.asarray(array[env_id])
    if width is not None and row.shape != (width,):
        raise ValueError(
            f"WheelBipe telemetry {name} row must have shape ({width},), got {row.shape}"
        )
    if row.dtype.kind not in "biufc" or not np.all(np.isfinite(row)):
        raise ValueError(f"WheelBipe telemetry {name} must contain finite numeric values")
    return row


def capture_wheelbipe_playback_telemetry(
    env: Any,
    *,
    sim_time_s: float,
    env_id: int = 0,
    terrain: str = "unknown",
) -> WheelbipePlaybackTelemetry:
    """Capture one sample through WheelBipe's public env/state contracts.

    Exact per-term reward arrays are included only when an owner explicitly
    exposes ``info['reward_terms']``.  The current UniLab owner exposes the
    selected row's total reward but not per-env term arrays; global/cadenced
    log means are intentionally not mislabeled as per-env rewards.
    """

    index = int(env_id)
    if index < 0:
        raise ValueError("WheelBipe telemetry env_id must be non-negative")
    state = env.state
    if state is None or not isinstance(state.info, dict):
        raise ValueError("WheelBipe telemetry requires an initialized env state")
    info = state.info
    commands = _batch_row(info.get("commands"), index, name="commands", width=3)
    heights = _batch_row(info.get("height_commands"), index, name="height_commands")
    if heights.shape != ():
        raise ValueError("WheelBipe telemetry height_commands must have one scalar per env")
    observed = _batch_row(info.get("observed_height"), index, name="observed_height")
    if observed.shape != ():
        raise ValueError("WheelBipe telemetry observed_height must have one scalar per env")
    policy_obs = _batch_row(state.obs.get("obs"), index, name="obs", width=35)
    linvel = _batch_row(env.get_local_linvel(), index, name="local linear velocity", width=3)
    dof_vel = _batch_row(env.get_full_dof_vel(), index, name="full dof velocity")
    torques = _batch_row(info.get("torques"), index, name="torques")
    if dof_vel.size < 8 or torques.size < 8:
        raise ValueError("WheelBipe realtime power requires at least eight native joint channels")
    leg_slots = np.asarray((0, 1, 4, 5), dtype=np.intp)
    wheel_slots = np.asarray((2, 6), dtype=np.intp)
    wheel_power = torques[wheel_slots] * dof_vel[wheel_slots]
    spring_forces_value = info.get("spring_forces")
    if spring_forces_value is None:
        spring_forces = np.zeros((2,), dtype=np.float64)
    else:
        spring_forces = _batch_row(
            spring_forces_value, index, name="spring_forces", width=2
        ).astype(np.float64, copy=False)
    steps = _batch_row(info.get("steps"), index, name="steps")
    if steps.shape != ():
        raise ValueError("WheelBipe telemetry steps must have one scalar per env")
    reward = _batch_row(state.reward, index, name="reward")
    if reward.shape != ():
        raise ValueError("WheelBipe telemetry reward must have one scalar per env")
    airborne_value = info.get("state_machine_airborne", np.zeros_like(state.reward, dtype=bool))
    airborne = _batch_row(airborne_value, index, name="state_machine_airborne")
    if airborne.shape != ():
        raise ValueError("WheelBipe telemetry airborne flag must have one scalar per env")
    phase_value = info.get("state_machine_jump_phase", np.zeros_like(state.reward))
    phase = _batch_row(phase_value, index, name="state_machine_jump_phase")
    if phase.shape != ():
        raise ValueError("WheelBipe telemetry jump phase must have one scalar per env")
    reward_ref = float(env.cfg.reward_config.base_height_target)
    ctrl_dt = float(env.cfg.ctrl_dt)
    row: dict[str, Any] = {
        "sim_time_s": _finite_number(sim_time_s, name="sim_time_s"),
        "episode_time_s": float(steps) * ctrl_dt,
        "env_id": index,
        "terrain": str(terrain),
        "cmd_x": float(commands[0]),
        "cmd_y": float(commands[1]),
        "cmd_yaw": float(commands[2]),
        "vel_x_b": float(linvel[0]),
        "vel_y_b": float(linvel[1]),
        # Normal policy observation stores the body gyro at slots 4:7,
        # scaled by 0.5.  Reading that public policy packet also preserves any
        # configured observation-delay semantics.
        "yaw_rate_b": float(policy_obs[6]) / 0.5,
        "height_cmd": float(heights),
        "height_obs": float(observed),
        "height_relative": float(observed),
        "height_reward_ref": reward_ref,
        "airborne": int(bool(airborne)),
        "reward_total": float(reward),
    }
    reward_terms = info.get("reward_terms")
    if isinstance(reward_terms, Mapping):
        for name, values in reward_terms.items():
            column = _reward_column_name(str(name))
            term = _batch_row(values, index, name=column)
            if term.shape != ():
                raise ValueError(f"WheelBipe telemetry {column} must have one scalar per env")
            row[column] = float(term)
    normalized = normalize_wheelbipe_trace_row(row)
    realtime = WheelbipeRealtimeSample(
        step=int(steps),
        target_height=float(heights),
        actual_height=float(observed),
        jump_phase=float(phase),
        wheel_power=(float(wheel_power[0]), float(wheel_power[1])),
        leg_torques=tuple(float(value) for value in torques[leg_slots]),
        spring_forces=(float(spring_forces[0]), float(spring_forces[1])),
    )
    return WheelbipePlaybackTelemetry(trace_row=normalized, realtime_sample=realtime)


class WheelbipeTraceRecorder:
    """Streaming CSV writer with bounded HTML memory and periodic export."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        html_path: str | Path | None = None,
        reward_scales: Mapping[str, float] | None = None,
        sample_dt: float = 0.0,
        html_update_interval_s: float = 1.0,
        max_rows: int = 20_000,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.html_path = (
            self.csv_path.with_suffix(".html")
            if html_path is None
            else Path(html_path).expanduser().resolve()
        )
        self.sample_dt = _finite_number(sample_dt, name="sample_dt")
        self.html_update_interval_s = _finite_number(
            html_update_interval_s, name="html_update_interval_s"
        )
        if self.sample_dt < 0.0 or self.html_update_interval_s < 0.0:
            raise ValueError("WheelBipe trace intervals must be non-negative")
        if isinstance(max_rows, bool) or int(max_rows) < 1:
            raise ValueError("WheelBipe trace max_rows must be positive")
        self.max_rows = int(max_rows)
        self.reward_scales = dict(reward_scales or {})
        self.reward_columns = tuple(_reward_column_name(str(name)) for name in self.reward_scales)
        if len(set(self.reward_columns)) != len(self.reward_columns):
            raise ValueError("WheelBipe trace reward scales produce duplicate columns")
        self.rows: deque[dict[str, float | int | str]] = deque(maxlen=self.max_rows)
        self._last_sample_time = -math.inf
        self._last_html_time = -math.inf
        self._closed = False
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.html_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = self.csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=[*WHEELBIPE_TRACE_FIELDS, *self.reward_columns],
            extrasaction="raise",
        )
        self._writer.writeheader()
        self._stream.flush()

    def append(self, row: Mapping[str, Any]) -> bool:
        """Append if the sampling interval elapsed; return whether written."""

        if self._closed:
            raise RuntimeError("WheelBipe trace recorder is closed")
        normalized = normalize_wheelbipe_trace_row(row, reward_columns=self.reward_columns)
        sim_time = float(normalized["sim_time_s"])
        if sim_time < self._last_sample_time:
            raise ValueError("WheelBipe trace sim_time_s must be non-decreasing")
        if sim_time - self._last_sample_time + 1.0e-12 < self.sample_dt:
            return False
        self._writer.writerow(normalized)
        self._stream.flush()
        self.rows.append(normalized)
        self._last_sample_time = sim_time
        if sim_time - self._last_html_time + 1.0e-12 >= self.html_update_interval_s:
            self.write_html()
            self._last_html_time = sim_time
        return True

    def write_html(self) -> None:
        """Atomically replace the interactive HTML snapshot."""

        html = build_wheelbipe_trace_html(
            list(self.rows),
            build_wheelbipe_reward_signs(self.reward_scales, rows=list(self.rows)),
        )
        temporary = self.html_path.with_name(f".{self.html_path.name}.tmp-{os.getpid()}")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(self.html_path)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.write_html()
            self._stream.flush()
        finally:
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "WheelbipeTraceRecorder":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


__all__ = [
    "WHEELBIPE_TRACE_FIELDS",
    "WheelbipePlaybackTelemetry",
    "WheelbipeRealtimeBuffer",
    "WheelbipeRealtimePlotter",
    "WheelbipeRealtimeSample",
    "WheelbipeTraceRecorder",
    "build_wheelbipe_reward_signs",
    "build_wheelbipe_trace_html",
    "capture_wheelbipe_playback_telemetry",
    "export_wheelbipe_trace_html",
    "load_wheelbipe_reward_signs",
    "load_wheelbipe_trace_csv",
    "normalize_wheelbipe_trace_row",
    "normalize_wheelbipe_trace_rows",
]
