#!/usr/bin/env python3
"""独立单脚本：CREST GFN-FF 构象搜索 + MACE-OMol 计算与能量排序。

合并自 ts_conformers/crest_search.py、crest_mlip_config.py、crest_mlip_xyz.py。
仅此文件即可运行，不需要安装 ts_conformers；科学计算依赖仍须单独安装。
使用 Linux/WSL2 + NVIDIA CUDA GPU。配置原子编号从 1 开始。

    python crest_mace.py --sample > config_mlip.yaml
    python crest_mace.py config_mlip.yaml

具体安装、配置和输出说明见同目录 README.md。
"""

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime
from pathlib import Path

VERSION = "1.0-standalone"
EV_TO_HARTREE = 1.0 / 27.211386245988
EV_TO_KCAL = 23.060549
HARTREE_TO_KCAL = 627.5094740631
ELEMENTS = set(('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
                'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce '
                'Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn '
                'Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split())

SAMPLE_CONFIG = """\
# CREST (GFN-FF) 搜索 -> MACE-OMol 单点/受约束优化 -> MACE 能量排序
# 所有相对路径以本 YAML 文件所在目录为基准。
# 电荷、自旋、原子编号沿用原包示例，不是自动识别值，必须按实际体系修改。
input: ./input/
output: ./results_sp/

# CPU 并行：首个结构先用 workers=stage1_workers=1 验证。
threads: 4
workers: 1
stage1_workers: 1

# 体系与约束（原子从 1 编号；下列编号要求输入至少有 47 个原子）。
chrg: 1
uhf: 0
freeze: [1, 35, 36, 37, 47]
freeze_by_structure: {}
cmetal: 0.2
notopo: null
hardfreeze: null
hardfreeze_tol_A: 0.05    # 硬冻结片段内部距离允许的最大变化 (Å)；整体刚体漂移只记录不剔除
quick: null
crest_mdlen: null         # 采样深度: null=CREST 自动; 60=每条 MTD 60 ps; "x2.0"=默认长度×2
fc_pos: 10.0

# true = 只做 Stage1 GFN-FF 构象采样, 完全跳过 MACE (无需 model / GPU)。
# 输出 crest_conformers.xyz、crest_ensemble.xyz、crest_best.xyz, 能量为 GFN-FF 级。
stage1_only: false

mlip:
  model: ./models/MACE-omol.model
  gpu_ids: [0]
  enable_cueq: false       # 先关闭可选加速；确认 CUDA/CuEq 兼容后再启用。
  spin: null               # null -> uhf+1；MACE-OMol 使用多重度 2S+1。
  mode: sp                 # sp=单点；opt=ASE FIRE 受约束优化。
  fmax: 0.04               # opt 有效，eV/Å。
  maxsteps: 600            # opt 的步数上限，不保证收敛。
  ewin_kcal_mol: 6.0       # 相对本结构最低 MACE 能量的输出窗口。
  max_conformers: 500      # 先按 GFN-FF 能量取前 N 个；null=全部。
  include_initial_ts: false

resume: true              # 只续跑配置/输入/模型/脚本全部匹配的结果。
cleanup: false            # 首次运行保留轨迹等排错文件。
dry_run: true             # 先预检；要真实计算时改为 false。
full_log: true
hb: 2                     # 终端面板刷新间隔，秒；0=不定时刷新。
"""


def detect_crest_version():
    """Probe only the program header; no sampling or model construction."""
    try:
        result = subprocess.run(["crest", "--version"], capture_output=True,
                                text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\bVersion\s*:?\s*(\d+)\.(\d+)",
                      (result.stdout or "") + "\n" + (result.stderr or ""), re.I)
    return (int(match[1]), int(match[2])) if match else None


def load_config(path):
    try:
        import yaml
    except ImportError:
        print("ERROR: 需要 PyYAML. 请运行: pip install pyyaml")
        sys.exit(1)
    try:
        with open(path, encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 格式错误: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError("配置必须是 YAML 键值表")
    defaults = {
        "input": None, "output": None,
        "threads": 8, "workers": 1,
        "chrg": 1, "uhf": 0,
        "freeze": [1, 35, 36, 37, 47],
        "cmetal": 0.2,
        "notopo": None, "hardfreeze": None, "quick": None, "crest_mdlen": None, "fc_pos": 10.0,
        "hardfreeze_tol_A": 0.05, "stage1_only": False,
        "resume": False, "cleanup": False, "dry_run": False,
        "full_log": False, "hb": 2,
        "mlip": {},
    }
    unknown = set(cfg) - set(defaults) - {"stage1_workers", "freeze_by_structure"}
    if unknown:
        raise ValueError(f"未知配置参数: {sorted(unknown, key=str)}")
    defaults.update(cfg)
    for key in ("threads", "workers"):
        if type(defaults[key]) is not int or defaults[key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    if type(defaults["chrg"]) is not int or type(defaults["uhf"]) is not int or defaults["uhf"] < 0:
        raise ValueError("chrg 必须为整数，uhf 必须为非负整数")
    for key in ("resume", "cleanup", "dry_run", "full_log", "stage1_only"):
        if type(defaults[key]) is not bool:
            raise ValueError(f"{key} 必须为 true 或 false（不要加引号）")
    for key in ("cmetal", "hb"):
        value = defaults[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} 必须为有限非负数")
    tol = defaults["hardfreeze_tol_A"]
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0:
        raise ValueError("hardfreeze_tol_A 必须为有限正数（Å，冻结片段内部距离允许的最大变化）")
    defaults["hardfreeze_tol_A"] = float(tol)
    stage1_workers = defaults.get("stage1_workers", defaults["workers"])
    if (type(stage1_workers) is not int or type(defaults["workers"]) is not int
            or not 1 <= stage1_workers <= defaults["workers"]):
        raise ValueError("stage1_workers must be a positive integer <= workers")
    defaults["stage1_workers"] = stage1_workers
    fr = defaults["freeze"]
    if isinstance(fr, str):
        defaults["freeze"] = [int(x) for x in fr.split(",")]
    elif isinstance(fr, list) and all(type(x) is int for x in fr):
        defaults["freeze"] = fr
    else:
        print(f"ERROR: freeze 格式错误: {fr}")
        raise ValueError("freeze 必须为整数原子编号列表或逗号分隔字符串")
    if len(set(defaults["freeze"])) != len(defaults["freeze"]):
        raise ValueError("freeze 原子编号不能重复")
    for key in ("notopo", "hardfreeze"):
        atoms = defaults[key]
        if atoms is None or atoms == "":
            atoms = []
        elif isinstance(atoms, str):
            atoms = [int(atom.strip()) for atom in atoms.split(",") if atom.strip()]
        if not isinstance(atoms, list) or any(type(atom) is not int or atom < 1 for atom in atoms):
            raise ValueError(f"{key} must be a list or CSV string of positive 1-based atom indices")
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"{key} atom indices must be unique")
        defaults[key] = atoms or None
    quick = defaults["quick"]
    if quick is None or quick is False or quick in ("", "none", "None"):
        defaults["quick"] = None
    elif quick in ("quick", "squick", "mquick"):
        defaults["quick"] = "--" + quick
    else:
        raise ValueError("quick must be null, quick, squick, or mquick")
    # Stage1 采样深度：CREST --len（每 ps 为绝对值，或 "x2.0" 表示默认长度的倍数）。
    mdlen = defaults["crest_mdlen"]
    if mdlen is None or mdlen is False or mdlen in ("", "none", "None"):
        defaults["crest_mdlen"] = None
    elif isinstance(mdlen, bool):
        raise ValueError("crest_mdlen 不能是 true/false；请用正数（ps）或 x2.0 形式、或 null")
    elif isinstance(mdlen, (int, float)):
        if not math.isfinite(mdlen) or mdlen <= 0:
            raise ValueError("crest_mdlen 必须为有限正数（ps）")
        defaults["crest_mdlen"] = f"{mdlen:g}"
    elif isinstance(mdlen, str):
        text = mdlen.strip()
        factor = re.fullmatch(r"[xX]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", text)
        absolute = re.fullmatch(r"([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", text)
        if factor:
            value = float(factor.group(1))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("crest_mdlen 的倍数必须为正数，例如 x2.0")
            defaults["crest_mdlen"] = "x" + factor.group(1)
        elif absolute:
            value = float(absolute.group(1))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("crest_mdlen 必须为有限正数（ps）")
            defaults["crest_mdlen"] = f"{value:g}"
        else:
            raise ValueError('crest_mdlen 只能写正数（ps，如 60）或倍数（如 "x2.0"），或 null')
    else:
        raise ValueError('crest_mdlen 只能写正数（ps）、"x2.0" 形式的字符串，或 null')
    fc_pos = defaults["fc_pos"]
    if isinstance(fc_pos, bool) or not isinstance(fc_pos, (int, float)) or not math.isfinite(fc_pos) or fc_pos <= 0:
        raise ValueError("fc_pos must be a finite positive number")
    defaults["fc_pos"] = float(fc_pos)
    mlip = defaults.get("mlip")
    if mlip is None:
        mlip = {}
    allowed_mlip = {"model", "gpu_ids", "enable_cueq", "mode", "spin", "fmax", "maxsteps",
                    "ewin_kcal_mol", "max_conformers", "include_initial_ts"}
    if not isinstance(mlip, dict):
        raise ValueError("mlip 必须是 YAML 键值表")
    unknown = set(mlip) - allowed_mlip
    if unknown:
        raise ValueError(f"未知 mlip 参数: {sorted(unknown, key=str)}")
    mlip.setdefault("model", None)
    mlip.setdefault("gpu_ids", [0])
    mlip.setdefault("enable_cueq", True)
    mlip.setdefault("mode", "sp")
    mlip.setdefault("fmax", 0.04)
    mlip.setdefault("maxsteps", 600)
    mlip.setdefault("ewin_kcal_mol", 6.0)
    mlip.setdefault("max_conformers", 500)
    mlip.setdefault("include_initial_ts", False)
    # MACE-OMol uses multiplicity (2S+1); set spin explicitly for another convention.
    if mlip.get("spin") is None:
        mlip["spin"] = defaults["uhf"] + 1
    if type(mlip["spin"]) is not int or mlip["spin"] < 1:
        raise ValueError("mlip.spin 必须为正整数多重度，或 null（自动取 uhf+1）")
    for key in ("enable_cueq", "include_initial_ts"):
        if type(mlip[key]) is not bool:
            raise ValueError(f"mlip.{key} 必须为 true 或 false")
    for key in ("fmax", "ewin_kcal_mol"):
        value = mlip[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (key == "fmax" and value == 0)):
            raise ValueError(f"mlip.{key} 必须为有限{'正' if key == 'fmax' else '非负'}数")
    if mlip["mode"] not in ("sp", "opt"):
        print(f"ERROR: mlip.mode 只能是 'sp' 或 'opt', 当前: {mlip['mode']}")
        sys.exit(1)
    if not mlip["model"] and not defaults["stage1_only"]:
        print("ERROR: mlip.model 未指定 (MACE 模型路径; stage1_only: true 时可不填)")
        sys.exit(1)
    if not mlip["gpu_ids"] and not defaults["stage1_only"]:
        print("ERROR: mlip.gpu_ids 未指定 (stage1_only: true 时可不填)")
        sys.exit(1)
    if (not mlip["gpu_ids"] and defaults["stage1_only"]):
        mlip["gpu_ids"] = []
    elif (not isinstance(mlip["gpu_ids"], list)
            or any(type(g) is not int or g < 0 for g in mlip["gpu_ids"])
            or len(set(mlip["gpu_ids"])) != len(mlip["gpu_ids"])):
        raise ValueError("mlip.gpu_ids 必须是非负整数列表，编号不能重复")
    if not defaults["freeze"] or any(a < 1 for a in defaults["freeze"]):
        raise ValueError("freeze 必须包含正的 1-based 原子索引")
    if type(mlip["maxsteps"]) is not int or mlip["maxsteps"] < 1:
        raise ValueError("mlip.maxsteps must be a positive integer")
    if mlip["max_conformers"] is not None and (type(mlip["max_conformers"]) is not int or mlip["max_conformers"] < 1):
        raise ValueError("mlip.max_conformers 必须为正整数或 null")
    mlip["freeze_0based"] = [a - 1 for a in defaults["freeze"]]  # ASE FixAtoms indices.
    mlip["hardfreeze_tol_A"] = defaults["hardfreeze_tol_A"]  # Stage1 frozen-fragment tolerance.
    mlip["stage1_only"] = defaults["stage1_only"]  # Skip Stage 2 entirely (no model / GPU needed).
    defaults["mlip"] = mlip
    overrides = defaults.get("freeze_by_structure")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("freeze_by_structure must map XYZ stems to atom lists")
    for name, atoms in overrides.items():
        if not isinstance(name, str) or not name:
            raise ValueError("freeze_by_structure keys must be nonempty XYZ stems")
        if isinstance(atoms, str):
            try:
                atoms = [int(atom.strip()) for atom in atoms.split(",") if atom.strip()]
            except ValueError as exc:
                raise ValueError(f"freeze_by_structure[{name}] must contain integer atom indices") from exc
        if (not isinstance(atoms, list) or not atoms
                or any(type(atom) is not int or atom < 1 for atom in atoms)):
            raise ValueError(f"freeze_by_structure[{name}] must be a nonempty list of positive 1-based indices")
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"freeze_by_structure[{name}] atom indices must be unique")
        overrides[name] = atoms
    defaults["freeze_by_structure"] = overrides
    if not defaults["input"] or not defaults["output"]:
        print("ERROR: 配置中必须指定 input 和 output")
        sys.exit(1)
    base = Path(path).resolve().parent
    for key in ("input", "output"):
        defaults[key] = str((base / defaults[key]).resolve())
    mlip["model"] = str((base / mlip["model"]).resolve()) if mlip["model"] else None
    return defaults


def structure_atom_lists(cfg, name):
    """Apply an explicit per-structure override only to enabled atom lists."""
    override = cfg.get("freeze_by_structure", {}).get(name)
    return {
        key: override if override is not None and cfg[key] else cfg[key]
        for key in ("freeze", "notopo", "hardfreeze")
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(out, cfg, xyzs):
    """Only resume when this single script and all scientific inputs match."""
    # shortcut: runtime versions are not fingerprinted; add them before cross-environment resume.
    identity = {
        "config": cfg,
        "script_sha256": file_sha256(__file__),
        "model_sha256": (file_sha256(cfg["mlip"]["model"])
                         if not cfg["stage1_only"] and cfg["mlip"].get("model") else None),
        "inputs_sha256": {path.name: file_sha256(path) for path in xyzs},
    }
    manifest = out / "crest_mlip_manifest.json"
    if out.exists() and any(out.iterdir()):
        if not cfg["resume"] or not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"拒绝覆盖或复用不匹配的输出目录: {out}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_conformers(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            h = h.strip()
            if not h:
                continue
            try:
                nat = int(h.split()[0])
            except (ValueError, IndexError):
                break
            f.readline()
            for _ in range(nat):
                if not f.readline():
                    return n
            n += 1
    return n


def read_lowest_energy(path: Path) -> float:
    """Read crest_best.xyz's second line (comment), which stores Hartree."""
    if not path.exists():
        return float("nan")
    with open(path) as f:
        f.readline()
        c = f.readline().strip()
    try:
        return float(c.split()[0])
    except (ValueError, IndexError):
        return float("nan")


def read_xyz_frames(path: Path):
    """Read multi-frame XYZ as ``[(symbols, positions), ...]`` with Nx3 lists."""
    frames = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    i, n = 0, len(lines)
    while i < n:
        h = lines[i].strip()
        if not h:
            i += 1
            continue
        try:
            nat = int(h)
        except ValueError as exc:
            raise ValueError(f"{path}: 第 {i + 1} 行原子数无效") from exc
        if nat < 1 or i + nat + 2 > n:
            raise ValueError(f"{path}: 第 {i + 1} 行开始的 XYZ 帧不完整或原子数无效")
        syms = []
        pos = []
        for j in range(nat):
            parts = lines[i + 2 + j].split()
            try:
                if len(parts) < 4 or not re.fullmatch(r"[A-Za-z]{1,2}", parts[0]):
                    raise ValueError("invalid atom row")
                if parts[0].capitalize() not in ELEMENTS:
                    raise ValueError("unknown element")
                coords = [float(value) for value in parts[1:4]]
                if not all(math.isfinite(value) for value in coords):
                    raise ValueError("nonfinite coordinates")
            except ValueError as exc:
                raise ValueError(f"{path}: 第 {i + 3 + j} 行元素/坐标无效") from exc
            syms.append(parts[0].capitalize())  # CREST 3.1 writes e.g. RH; ASE expects Rh.
            pos.append(coords)
        frames.append((syms, pos))
        i += 2 + nat
    return frames


def write_ensemble_xyz(path: Path, frames_with_eh):
    """Write multi-frame XYZ with Hartree energy comments (CREST-compatible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for energy_eh, syms, pos in frames_with_eh:
            f.write(f"{len(syms)}\n")
            f.write(f"{energy_eh:.8f}\n")
            for s, (x, y, z) in zip(syms, pos):
                f.write(f" {s:<2s} {x:16.8f} {y:14.8f} {z:14.8f}\n")


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ────────────────── ANSI / TTY 面板 (与原版一致) ──────────────────
_ANSI_UP = "\033[{}A"
_ANSI_CLR = "\033[J"
_ANSI_HIDE = "\033[?25l"
_ANSI_SHOW = "\033[?25h"


class BatchUI:
    def __init__(self, log_path: Path):
        self.lock = threading.Lock()
        self.log = open(log_path, "w", buffering=1)
        self.is_tty = sys.stdout.isatty()
        self.recent = deque(maxlen=6)
        self.prev_lines = 0
        self.start = time.time()
        if self.is_tty:
            sys.stdout.write(_ANSI_HIDE)
            sys.stdout.flush()

    def print_line(self, msg: str):
        line = f"[{now()}] {msg}"
        with self.lock:
            self.log.write(line + "\n")
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def event(self, msg: str, status=None, stats=None):
        line = f"[{now()}] {msg}"
        with self.lock:
            self.log.write(line + "\n")
            self.recent.append(line)
            if self.is_tty:
                self._redraw(status or {}, stats or {})
            else:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()

    def tick(self, status, stats):
        if not self.is_tty:
            return
        with self.lock:
            self._redraw(status or {}, stats or {})

    def _redraw(self, status, stats):
        if self.prev_lines > 0:
            sys.stdout.write(_ANSI_UP.format(self.prev_lines) + _ANSI_CLR)
        dash = self._dashboard(status, stats)
        sys.stdout.write(dash + "\n")
        sys.stdout.flush()
        self.prev_lines = dash.count("\n") + 1

    def _dashboard(self, status, stats):
        el = int(time.time() - self.start)
        el_s = f"{el // 3600:02d}:{(el % 3600) // 60:02d}:{el % 60:02d}"
        total = stats.get("total", 0); done = stats.get("done", 0)
        ok = stats.get("ok", 0); err = stats.get("err", 0); sk = stats.get("skipped", 0)
        # 表头须 < 80 列, 否则终端换行会破坏 ANSI 原地刷新 (按 \n 计行, 不计换行)
        L = [f"GFN-FF->MACE | 进度 {done}/{total} | ✓{ok} ✗{err} 跳{sk} | "
             f"{len(status)}运行 | {el_s}",
             "-" * 64]
        if status:
            for w, s in sorted(status.items()):
                st = int(time.time() - s.get("t0", time.time()))
                mm = f"{st // 60:02d}:{st % 60:02d}"
                L.append(f"  [W{w}] {s['name']:<10} {s['stage']:<22} {mm}")
        else:
            L.append("  (无运行中任务)")
        L += ["-" * 64, "recent:"]
        L += ["  " + e for e in self.recent]
        return "\n".join(L)

    def finish(self):
        with self.lock:
            if self.is_tty:
                if self.prev_lines > 0:
                    sys.stdout.write(_ANSI_UP.format(self.prev_lines) + _ANSI_CLR)
                sys.stdout.write(_ANSI_SHOW)
                sys.stdout.flush()
                self.prev_lines = 0


# ────────────────── CREST 命令运行 (Stage 1) ──────────────────
_KEY_LINE_RE = re.compile(
    r"ERROR STOP|Error termination|coord file does not exist|safety termination|"
    r"SIGSEGV|Segmentation fault|parse_xtbinput|get_xtb_constraint_block|"
    r"Version|commit|Parsing xtb-type input file|FROZEN ATOMS|"
    r"Trial MTD|Iteration [0-9]|starting MTD|MTD.*completed|MTD.*finished|"
    r"CREGEN|E lowest|conformers within|lowest energy|Containing [0-9]+ structures|"
    r"ENSEMBLE SCREENING|Multilevel Ensemble|Initial Geometry Optimization|"
    r"Geometry successfully optimized|GEOMETRY OPTIMIZATION CONVERGED|"
    r"finished|Total wall|wall-time|wall time|"
    r"Program call|Command line input|User-defined|--gfn2|--gfnff|--chrg|cmetal|constrain|"
    r"Conformer-Rotamer Ensemble Sampling Tool|╔|║|╚|═",
    re.IGNORECASE)


def run_cmd(cmd, cwd, log_file, env, full=False) -> int:
    if full:
        with open(log_file, "w") as f:
            p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
        return p.returncode
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(log_file, "w") as f:
        for line in p.stdout:
            if _KEY_LINE_RE.search(line):
                f.write(line)
        p.wait()
    return p.returncode


# ────────────────── MLIP (MACE) Stage 2 ──────────────────
# 每个 GPU 进程的全局状态 (由 initializer 设置)
_worker_state = {}
_stage1_slots = None


def _init_gpu_worker(gpu_id, model_path, enable_cueq, charge, spin, result_queue):
    """Pool worker 初始化: 锁定 GPU + 存配置; MACE 模型懒加载 (首次计算时才加载).

    Stage 1 (CPU, 数小时) 期间不 import mace/torch、不占 GPU 显存;
    首个 Stage 2 构象触发加载, 之后复用.

    charge: 总电荷 (atoms.info["charge"])
    spin:   total_spin = 自旋多重度 2S+1 (MACE-OMol 约定, 与 ASE IdealGasThermo/UMA 一致)
            单重态=1, 双重态=2, 三重态=3 (= xtb uhf + 1)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _worker_state["gpu_id"] = gpu_id
    _worker_state["model_path"] = str(model_path)
    _worker_state["enable_cueq"] = bool(enable_cueq)
    _worker_state["charge"] = int(charge)
    _worker_state["spin"] = float(spin)
    _worker_state["rq"] = result_queue
    _worker_state["calc"] = None  # 懒加载: 首次 _compute_conformer 时才 import mace + 加载模型


def _get_calc():
    """懒加载 MACE 模型: 首次调用 import mace + 加载模型 (占 GPU 显存), 后续直接复用."""
    if _worker_state.get("calc") is None:
        from mace.calculators import MACECalculator
        gid = _worker_state["gpu_id"]
        try:
            calc = MACECalculator(
                model_paths=_worker_state["model_path"],
                enable_cueq=_worker_state["enable_cueq"],
                device="cuda:0",
            )
        except Exception as e:
            if not _worker_state["enable_cueq"]:
                raise
            import sys as _sys
            _sys.stderr.write(f"[GPU {gid}] CUEQ 不可用 ({e}), 回退 enable_cueq=False\n")
            calc = MACECalculator(
                model_paths=_worker_state["model_path"],
                enable_cueq=False,
                device="cuda:0",
            )
        _worker_state["calc"] = calc
    return _worker_state["calc"]


def _compute_conformer(task):
    """计算单个构象的 MACE 能量 (SP 或 opt). 返回结果 push 到 result_queue."""
    idx, syms, pos, mode, fmax, maxsteps, freeze_0b = task
    try:
        from ase import Atoms
        from ase.constraints import FixAtoms

        atoms = Atoms(symbols=syms, positions=pos)
        atoms.info["charge"] = _worker_state["charge"]
        atoms.info["spin"] = _worker_state["spin"]
        atoms.calc = _get_calc()
        if freeze_0b and mode == "opt":
            atoms.set_constraint(FixAtoms(indices=freeze_0b))
        if mode == "sp":
            energy_ev = float(atoms.get_potential_energy())
            nsteps = 0
            converged = True
            out_pos = pos
        else:  # opt
            from ase.optimize import FIRE
            opt = FIRE(atoms, logfile=None)
            converged = bool(opt.run(fmax=float(fmax), steps=int(maxsteps)))
            energy_ev = float(atoms.get_potential_energy())
            nsteps = int(opt.nsteps)
            out_pos = atoms.get_positions().tolist()
        _worker_state["rq"].put((idx, ("ok", energy_ev, out_pos, nsteps, converged)))
    except Exception as e:
        _worker_state["rq"].put((idx, ("error", str(e)[:200], None, 0, False)))


def validate_stage2_result(result, symbols):
    if not isinstance(result, (list, tuple)) or len(result) != 5:
        raise ValueError("Stage2 checkpoint contains an invalid result")
    state, energy, positions, steps, converged = result
    if state == "error":
        if not isinstance(energy, str) or positions is not None or steps != 0 or converged is not False:
            raise ValueError("Stage2 checkpoint contains an invalid error result")
        return
    if (state != "ok" or isinstance(energy, bool) or not isinstance(energy, (int, float))
            or not math.isfinite(energy) or type(steps) is not int or steps < 0
            or type(converged) is not bool or not isinstance(positions, list)
            or len(positions) != len(symbols)):
        raise ValueError("Stage2 checkpoint contains an invalid result")
    for xyz in positions:
        if (not isinstance(xyz, list) or len(xyz) != 3
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in xyz)):
            raise ValueError("Stage2 checkpoint contains invalid coordinates")


def init_gpu_pools(gpu_ids, model_path, enable_cueq, charge, spin):
    """为每个 GPU 创建一个单进程 Pool, 共享一个 result_queue."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")  # Linux 默认 fork, 确保 CUDA 隔离
    result_queue = ctx.Queue()
    pools = []
    for gpu_id in gpu_ids:
        p = ctx.Pool(processes=1,
                     initializer=_init_gpu_worker,
                     initargs=(gpu_id, str(model_path), enable_cueq, charge, spin, result_queue))
        pools.append(p)
    return pools, result_queue


def wait_gpu_result(result_queue, workers):
    """A native/CUDA worker crash must not leave the parent waiting forever."""
    while True:
        try:
            return result_queue.get(timeout=1.0)
        except queue.Empty:
            if any(worker.exitcode is not None for worker in workers):
                raise RuntimeError("Stage2 GPU worker exited before returning a result")


def run_mlip_stage2(frames, mlip_cfg, gpu_pools, result_queue, stage2_lock,
                    event_q, wid, name):
    """对一批构象跑 MACE SP/opt, 多 GPU 轮询分配, 返回 [(idx, status, energy_ev, pos, nsteps)]."""
    mode = mlip_cfg["mode"]
    fmax = mlip_cfg["fmax"]
    maxsteps = mlip_cfg["maxsteps"]
    freeze_0b = mlip_cfg["freeze_0based"]
    n_gpu = len(gpu_pools)
    n_conf = len(frames)

    tasks = [(i, frames[i][0], frames[i][1], mode, fmax, maxsteps, freeze_0b)
             for i in range(n_conf)]

    Path(mlip_cfg["work"]).mkdir(parents=True, exist_ok=True)
    checkpoint = Path(mlip_cfg["work"]) / "stage2_checkpoint.json"
    frames_sha256 = hashlib.sha256(
        json.dumps(frames, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    results = [None] * n_conf
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if (not isinstance(saved, dict) or saved.get("frames_sha256") != frames_sha256
                or not isinstance(saved.get("results"), list)
                or len(saved["results"]) != n_conf):
            raise ValueError("Stage2 checkpoint does not match the current conformers")
        results = saved["results"]
        for i, result in enumerate(results):
            if result is not None:
                validate_stage2_result(result, frames[i][0])
    done = sum(r is not None and r[0] == "ok" for r in results)
    pending = {i for i, result in enumerate(results)
               if result is None or result[0] != "ok"}
    report_step = max(1, n_conf // 10)
    with stage2_lock:  # 串行化 Stage 2, 单结构独占全部 GPU
        workers = [worker for pool in gpu_pools for worker in getattr(pool, "_pool", [])]
        for i, task in enumerate(tasks):
            if results[i] is not None and results[i][0] == "ok":
                continue
            gpu_pools[i % n_gpu].apply_async(_compute_conformer, (task,))
        while done < n_conf:
            idx, payload = wait_gpu_result(result_queue, workers)
            if type(idx) is not int or idx not in pending:
                raise ValueError(f"Stage2 returned an unexpected conformer index: {idx}")
            validate_stage2_result(payload, frames[idx][0])
            pending.remove(idx)
            results[idx] = payload
            done += 1
            tmp = checkpoint.with_suffix(".tmp")
            tmp.write_text(json.dumps({"frames_sha256": frames_sha256, "results": results}) + "\n")
            tmp.replace(checkpoint)
            if done % report_step == 0 or done == n_conf:
                event_q.put(f"[W{wid}] {name}: Stage2 MACE {done}/{n_conf}")
        # Compute the original TS SP before releasing this structure's GPU slot.
        # Taking the lock again after optimization lets large queued jobs starve it.
        if mlip_cfg.get("initial_frame") is not None:
            initial_path = Path(mlip_cfg["work"]) / "initial_sp" / "stage2_checkpoint.json"
            initial_syms, initial_pos = mlip_cfg["initial_frame"]
            if initial_path.exists():
                initial_result = json.loads(initial_path.read_text())[0]
            else:
                initial_result = None
            if initial_result is None or initial_result[0] != "ok":
                gpu_pools[0].apply_async(_compute_conformer, ((0, initial_syms, initial_pos, "sp", fmax, maxsteps, []),))
                initial_idx, initial_result = wait_gpu_result(result_queue, workers)
                if type(initial_idx) is not int or initial_idx != 0:
                    raise ValueError("Unexpected initial TS result index")
                validate_stage2_result(initial_result, initial_syms)
                initial_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = initial_path.with_suffix(".tmp")
                tmp.write_text(json.dumps([initial_result]) + "\n")
                tmp.replace(initial_path)
            validate_stage2_result(initial_result, initial_syms)
            mlip_cfg["initial_result"] = initial_result
    return results


# ────────────────── 单结构处理 ──────────────────
def process_structure(wid, xyz_path, out_dir, threads, charge, uhf,
                      frozen_atoms, cmetal_fc, resume, cleanup, dry_run, full_log,
                      mlip_cfg, gpu_pools, result_queue, stage2_lock,
                      emitter, event_q, status, status_lock,
                      crest_notopo=None, crest_quick=None, hard_freeze=None, fc_pos=10.0,
                      crest_mdlen=None):
    """单个结构: Stage1 GFN-FF -> Stage2 MACE."""
    name = Path(xyz_path).stem
    work = Path(out_dir) / name

    def set_stage(s):
        with status_lock:
            status[wid] = {"name": name, "stage": s, "t0": time.time()}
        event_q.put(f"[W{wid}] {name}: {s}")

    # ---- resume: Stage2 是否已完成 ----
    s2_log = work / "stage2_mlip.log"
    ensemble = work / "crest_ensemble.xyz"

    def stage2_ok():
        if not s2_log.exists() or not (work / "crest_best.xyz").exists() or not ensemble.exists():
            return False
        try:
            txt = s2_log.read_text(errors="ignore")
        except Exception:
            return False
        if (not mlip_cfg.get("stage1_only") and mlip_cfg.get("include_initial_ts")
                and not (work / "ts_ensemble.xyz").exists()):
            return False
        marker = "STAGE1_ONLY_DONE" if mlip_cfg.get("stage1_only") else "STAGE2_MLIP_DONE"
        return (marker in txt and "ERROR" not in txt
                and count_conformers(ensemble) > 0)

    s1_backup = work / "crest_conformers_gfnff.xyz"

    if resume and stage2_ok():
        e = read_lowest_energy(work / "crest_best.xyz")
        n_fin = count_conformers(ensemble)
        completed_stage = "Stage1 仅采样" if mlip_cfg.get("stage1_only") else "Stage2"
        event_q.put(f"[W{wid}] SKIP {name} ({completed_stage} 已完成, E={e:.5f}, {n_fin} 构象)")
        with status_lock:
            status.pop(wid, None)
        return {"name": name, "status": "skipped", "lowest_energy": e,
                "n_final": n_fin, "work": str(work)}

    # Stage1 已完成但 Stage2 未成功 -> 复用 GFN-FF 系综
    skip_stage1 = (resume and not dry_run and s1_backup.exists()
                   and (work / "constraints.inp").exists()
                   and (work / "coord.ref").exists() and not stage2_ok())

    work.mkdir(parents=True, exist_ok=True)
    struc = work / "struc.xyz"
    if not struc.exists():
        shutil.copy(xyz_path, struc)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)

    # ============ Stage 1: GFN-FF 构象搜索 ============
    with _stage1_slots:
        if skip_stage1:
            event_q.put(f"[W{wid}] {name}: 复用已有 GFN-FF 系综, 跳过 Stage 1")
            shutil.copy(s1_backup, work / "crest_conformers.xyz")
            n_gfnff = count_conformers(work / "crest_conformers.xyz")
        else:
            need_const = (not resume) or not (work / "constraints.inp").exists() \
                         or not (work / "coord.ref").exists()
            if need_const:
                set_stage("gen_const")
                atoms_str = ",".join(str(a) for a in frozen_atoms)
                rc = run_cmd(["crest", "struc.xyz", "--constrain", atoms_str,
                              "--chrg", str(charge), "--gfn2"],
                             work, work / "constraints_setup.log", env, full=full_log)
                xctrl = work / ".xcontrol.sample"
                if rc != 0 or not xctrl.exists():
                    event_q.put(f"[W{wid}] ERROR {name}: 约束生成失败 (rc={rc})")
                    with status_lock:
                        status.pop(wid, None)
                    return {"name": name, "status": "error", "reason": "constraint setup",
                            "work": str(work)}
                shutil.copy(xctrl, work / "constraints.inp")
                for ff in [".xcontrol.sample", "coord", "crest_input_copy.xyz",
                           ".CHRG", ".UHF", "bondlengths", "gfnff_topo", "gfnff_charges"]:
                    (work / ff).unlink(missing_ok=True)

            # Finite positional restraints are not equivalent to CREST --freeze.
            cinp = work / "constraints.inp"
            text = cinp.read_text()
            pattern = r"(?im)^(\s*force\s+constant\s*=\s*)[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?(?=\s|$)"
            updated, count = re.subn(pattern, lambda match: f"{match[1]}{fc_pos:g}", text)
            if count != 1:
                raise ValueError("fc_pos requires exactly one numeric force constant in constraints.inp")
            if updated != text:
                cinp.write_text(updated)
                event_q.put(f"[W{wid}] {name}: position force constant={fc_pos:g}")

            if dry_run:
                event_q.put(f"[W{wid}] DRY-RUN {name}: 环境就绪, 跳过搜索")
                with status_lock:
                    status.pop(wid, None)
                return {"name": name, "status": "dry_run", "work": str(work)}

            set_stage("Stage1 GFN-FF")
            # Explicit TOML loading defers constraint parsing to CREST's calculator
            # setup. Some 3.0.2 builds fail in the early CLI --cinp parser.
            # Do not rely on an automatically discovered xcontrol file.
            sampling_input = (
                f'input = "struc.xyz"\nruntype = "imtd-gc"\nthreads = {threads}\n'
                'constraints = "constraints.inp"\n[calculation]\n')
            if hard_freeze:
                freeze_csv = ",".join(map(str, hard_freeze))
                sampling_input += f'freeze = "{freeze_csv}"\n'
            sampling_input += (
                f'[[calculation.level]]\nmethod = "gfnff"\nchrg = {charge}\nuhf = {uhf}\n')
            (work / "sampling.toml").write_text(sampling_input)
            cmd1 = ["crest", "sampling.toml", "--v3", "--gfnff", "--chrg", str(charge), "--uhf", str(uhf),
                    "--cmetal", str(cmetal_fc),
                    "--subrmsd", "--noreftopo", "--T", str(threads)]
            if crest_notopo:
                cmd1 += ["--notopo", ",".join(map(str, crest_notopo))]
            if crest_quick:
                cmd1.append(crest_quick)
            if crest_mdlen:
                # 采样深度：显式覆盖 CREST 按分子柔性自动估算的 MD/MTD 长度。
                cmd1 += ["--len", str(crest_mdlen)]
            rc1 = run_cmd(cmd1, work, work / "stage1_gfnff.log", env, full=full_log)
            conf1 = work / "crest_conformers.xyz"
            if rc1 != 0 or not conf1.exists():
                event_q.put(f"[W{wid}] ERROR {name}: Stage1 失败 (rc={rc1})")
                with status_lock:
                    status.pop(wid, None)
                return {"name": name, "status": "error", "reason": "stage1", "work": str(work)}
            n_gfnff = count_conformers(conf1)
            shutil.copy(conf1, s1_backup)
            event_q.put(f"[W{wid}] {name}: Stage1 完成 -> {n_gfnff} 个 GFN-FF 构象")

    # ============ Stage 2: MACE-omol SP/opt + 排序 ============
    stage1_only = bool(mlip_cfg.get("stage1_only"))
    conf_path = work / "crest_conformers.xyz"
    frames = read_xyz_frames(conf_path)
    if not frames:
        event_q.put(f"[W{wid}] ERROR {name}: Stage2 无构象可算")
        with status_lock:
            status.pop(wid, None)
        return {"name": name, "status": "error", "reason": "no conformers", "work": str(work)}
    if not stage1_only:
        set_stage(f"Stage2 MACE({mlip_cfg['mode']})")

    # Explicit GFN-FF energy ordering; never assume input file order.
    raw = conf_path.read_text().splitlines()
    energies = []
    cursor = 0
    while cursor < len(raw):
        if not raw[cursor].strip():
            cursor += 1
            continue
        nat = int(raw[cursor])
        energy = float(raw[cursor + 1].split()[0])
        if not math.isfinite(energy):
            raise ValueError("Stage1 contains a nonfinite GFN-FF energy")
        energies.append(energy)
        cursor += nat + 2
    order = sorted(range(len(frames)), key=energies.__getitem__)
    frames = [frames[i] for i in order]
    sorted_energies = [energies[i] for i in order]  # 与排序后的 frames 一一对应
    t0 = time.time()
    # CREGEN can translate/rotate frames: align frozen sites to the actual TS,
    # verify hard freezing, then restore their original coordinates exactly.
    #
    # CREST --freeze only zeroes the gradient on the frozen atoms (see CREST source
    # src/calculator/calc_type.f90: calculation_freezegrad). It never pins them to a
    # fixed Cartesian position, so the frozen fragment may still translate/rotate as a
    # rigid body with the surrounding molecule. What it does guarantee is that the
    # frozen fragment does not deform. Validation therefore checks the internal
    # geometry (max |d_ij - d_ij_ref|) and only records the rigid-body shift.
    import itertools
    import numpy as np
    reference_syms, reference_pos = read_xyz_frames(Path(xyz_path))[0]
    reference = np.asarray(reference_pos)
    frozen_idx = np.asarray([a - 1 for a in frozen_atoms])
    hard_idx = np.asarray([a - 1 for a in hard_freeze], dtype=int) if hard_freeze else None
    freeze_tol = float(mlip_cfg.get("hardfreeze_tol_A", 0.05))
    anchor_idx = hard_idx if hard_freeze else frozen_idx
    reference_pairs = list(itertools.combinations(hard_idx, 2)) if hard_freeze else []
    reference_dists = {pair: float(np.linalg.norm(reference[pair[0]] - reference[pair[1]]))
                       for pair in reference_pairs}
    aligned = []
    max_drift = 0.0
    max_internal = 0.0
    max_rigid_shift = 0.0
    rejected = []
    kept_ranks = []
    for gfnff_rank, (syms, pos) in enumerate(frames, 1):
        if syms != reference_syms:
            raise ValueError("Stage1 atom order differs from initial TS")
        arr = np.asarray(pos)
        rigid_shift = float(np.linalg.norm(
            arr[hard_idx].mean(axis=0) - reference[hard_idx].mean(axis=0))) if hard_freeze else 0.0
        max_rigid_shift = max(max_rigid_shift, rigid_shift)
        center_a = arr[anchor_idx].mean(axis=0)
        center_b = reference[anchor_idx].mean(axis=0)
        u, _, vt = np.linalg.svd((arr[anchor_idx] - center_a).T @ (reference[anchor_idx] - center_b))
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        arr = (arr - center_a) @ (u @ correction @ vt) + center_b
        drift = float(np.linalg.norm(arr[frozen_idx] - reference[frozen_idx], axis=1).max())
        max_drift = max(max_drift, drift)
        internal = 0.0
        if hard_freeze:
            internal = max((abs(float(np.linalg.norm(arr[i] - arr[j])) - reference_dists[(i, j)])
                            for i, j in reference_pairs), default=0.0)
        max_internal = max(max_internal, internal)
        if internal > freeze_tol:  # only the frozen fragment's own geometry is enforced
            rejected.append({"gfnff_rank": gfnff_rank, "internal_diff_A": internal,
                             "rigid_shift_A": rigid_shift})
            continue
        arr[frozen_idx] = reference[frozen_idx]
        aligned.append((syms, arr.tolist()))
        kept_ranks.append(gfnff_rank)
    if not aligned:
        raise ValueError("No conformers pass hard frozen site validation")
    max_conf = mlip_cfg.get("max_conformers")
    if rejected:
        event_q.put(f"[W{wid}] {name}: 剔除 {len(rejected)} 个冻结片段变形帧 (>{freeze_tol:g} Å)，"
                    f"保留 {len(aligned)} 个合格构象")
    if hard_freeze:
        event_q.put(f"[W{wid}] {name}: 冻结片段内部形变≤{max_internal:.1e} Å，整体漂移≤{max_rigid_shift:.3f} Å "
                    f"(整体漂移已由坐标写回消除)")
    else:
        event_q.put(f"[W{wid}] {name}: 未启用 hardfreeze，Stage1 仅为软约束，冻结位点漂移≤{max_drift:.3f} Å 且已写回参考坐标")

    if max_conf and len(aligned) > max_conf:
        event_q.put(f"[W{wid}] {name}: 取冻结核验合格且 GFN-FF 能量最低的 {max_conf}/{len(aligned)} 个构象")
        aligned = aligned[:max_conf]
        kept_ranks = kept_ranks[:max_conf]
    frames = aligned
    mlip_cfg = dict(mlip_cfg, work=str(work), freeze_0based=frozen_idx.tolist())
    (work / "freeze_validation.json").write_text(json.dumps({
        "atoms_1based": frozen_atoms,
        "hardfreeze_1based": hard_freeze,
        "hardfreeze_tol_A": freeze_tol,
        "criterion": ("hardfreeze: max |d_ij - d_ij_ref| within the frozen fragment; "
                      "rigid-body shift is recorded, not rejected"),
        "max_aligned_stage1_drift_A": max_drift,
        "max_internal_diff_A": max_internal,
        "max_rigid_shift_A": max_rigid_shift,
        "rejected_frames": rejected,
        "selected_gfnff_ranks": kept_ranks,
    }, indent=2, ensure_ascii=False))

    if stage1_only:
        # 仅采样模式: 不做 MACE 计算, 直接输出校验并写回坐标后的 GFN-FF 集合。
        ens_frames = []
        for i, (syms, pos) in enumerate(frames):
            ens_frames.append((sorted_energies[kept_ranks[i] - 1], syms, pos))
        write_ensemble_xyz(work / "crest_ensemble.xyz", ens_frames)
        write_ensemble_xyz(work / "crest_best.xyz", ens_frames[:1])
        (work / "stage2_mlip.log").write_text(
            "STAGE1_ONLY_DONE\n"
            "Stage2 已按 stage1_only 关闭；能量为 GFN-FF 级 (Hartree)。\n"
            f"input_conformers: {len(frames)}\n"
            f"selected: {len(ens_frames)}\n"
            f"hardfreeze_tol_A: {freeze_tol:g}  max_internal_diff_A: {max_internal:.6g}\n")
        e_min_eh = min(f[0] for f in ens_frames)
        n_final = len(ens_frames)
        event_q.put(f"[W{wid}] DONE {name}: 仅采样, 输出 {n_final} 个 GFN-FF 构象, 最低={e_min_eh:.5f} Eh")
        if cleanup:
            for d in ["MDFILES", "tmp", "TMPDIR", "exmundus"]:
                shutil.rmtree(work / d, ignore_errors=True)
            for f in work.glob("*.trj"):
                f.unlink(missing_ok=True)
        with status_lock:
            status.pop(wid, None)
        return {"name": name, "status": "ok", "reason": "",
                "lowest_energy": e_min_eh,
                "n_gfnff": n_gfnff, "n_final": n_final, "work": str(work)}

    if mlip_cfg.get("include_initial_ts"):
        mlip_cfg["initial_frame"] = (reference_syms, reference_pos)
    results = run_mlip_stage2(frames, mlip_cfg, gpu_pools, result_queue, stage2_lock,
                              event_q, wid, name)
    dt = time.time() - t0

    # 整理: (energy_ev, pos, nsteps), 跳过 error
    ok = []
    n_err = 0
    for i, payload in enumerate(results):
        if payload is None:
            n_err += 1
            continue
        status_str, energy_ev, out_pos, nsteps, converged = payload
        if status_str == "ok":
            ok.append((i, energy_ev, out_pos, nsteps))
        else:
            n_err += 1

    if not ok:
        event_q.put(f"[W{wid}] ERROR {name}: Stage2 全部失败 ({n_err} 个)")
        with status_lock:
            status.pop(wid, None)
        return {"name": name, "status": "error", "reason": "stage2 all failed", "work": str(work)}

    ok.sort(key=lambda x: x[1])  # 按 MACE 能量升序
    e_min_ev = ok[0][1]
    ewin_kcal = mlip_cfg["ewin_kcal_mol"]

    # 选窗内构象, 至少保留最低 1 个
    selected = []
    for idx, energy_ev, out_pos, nsteps in ok:
        rel_kcal = (energy_ev - e_min_ev) * EV_TO_KCAL
        if rel_kcal <= ewin_kcal or not selected:
            selected.append((idx, energy_ev, out_pos, nsteps, rel_kcal))

    # 写 crest_ensemble.xyz (Hartree) + crest_best.xyz
    ens_frames = []
    for idx, energy_ev, out_pos, nsteps, rel_kcal in selected:
        syms = frames[idx][0]
        ens_frames.append((energy_ev * EV_TO_HARTREE, syms, out_pos))
    write_ensemble_xyz(ensemble, ens_frames)
    write_ensemble_xyz(work / "crest_best.xyz", ens_frames[:1])
    # Preserve the original true TS without any optimization or coordinate changes.
    if mlip_cfg.get("include_initial_ts", False):
        initial_result = mlip_cfg["initial_result"]
        if initial_result[0] != "ok":
            raise ValueError(f"Initial TS single point failed: {initial_result[1]}")
        initial_eh = initial_result[1] * EV_TO_HARTREE
        target = work / "ts_ensemble.xyz"
        original_lines = Path(xyz_path).read_text().splitlines(keepends=True)
        original_lines[1] = f"{initial_eh:.12f} source=initial_true_TS name={name}\n"
        if not original_lines[-1].endswith("\n"):
            original_lines[-1] += "\n"
        target.write_text("".join(original_lines) + ensemble.read_text())
        shutil.copyfile(xyz_path, work / "initial_true_ts.xyz")
    (work / "optimization_status.json").write_text(json.dumps([
        {"gfnff_rank": kept_ranks[i], "status": payload[0], "steps": payload[3],
         "converged": payload[4], "energy_eV": payload[1] if payload[0] == "ok" else None}
        for i, payload in enumerate(results)], indent=2))

    # stage2 日志
    with open(s2_log, "w") as f:
        f.write("STAGE2_MLIP_DONE\n" if not n_err else "STAGE2_MLIP_PARTIAL\n")
        f.write(f"model: {mlip_cfg['model']}\n")
        f.write(f"mode: {mlip_cfg['mode']}  fmax: {mlip_cfg['fmax']}  maxsteps: {mlip_cfg['maxsteps']}\n")
        f.write(f"gpus: {mlip_cfg['gpu_ids']}\n")
        f.write(f"input_conformers: {len(frames)}  ok: {len(ok)}  error: {n_err}\n")
        f.write(f"selected (ewin {ewin_kcal} kcal/mol): {len(selected)}\n")
        f.write(f"wall_time_s: {dt:.1f}\n")
        f.write(f"\n{'rank':>4s} {'idx':>5s} {'E_eV':>14s} {'E_Hartree':>14s} "
                f"{'rel_kcal':>10s} {'nsteps':>7s}\n")
        for rank, (idx, energy_ev, out_pos, nsteps, rel_kcal) in enumerate(selected):
            f.write(f"{rank:4d} {idx:5d} {energy_ev:14.6f} {energy_ev*EV_TO_HARTREE:14.8f} "
                    f"{rel_kcal:10.4f} {nsteps:7d}\n")
        if n_err:
            f.write(f"\n错误构象 ({n_err}):\n")
            for i, payload in enumerate(results):
                if payload and payload[0] == "error":
                    f.write(f"  idx {i}: {payload[1]}\n")

    e_min_eh = e_min_ev * EV_TO_HARTREE
    n_final = len(selected)
    event_q.put(f"[W{wid}] DONE {name}: 最低={e_min_eh:.5f} Eh, {n_final} 个 MLP 构象 "
                f"({mlip_cfg['mode']}, {dt:.0f}s, {n_err} 错误)")

    if cleanup:
        for d in ["MDFILES", "tmp", "TMPDIR", "exmundus"]:
            shutil.rmtree(work / d, ignore_errors=True)
        for f in work.glob("*.trj"):
            f.unlink(missing_ok=True)

    with status_lock:
        status.pop(wid, None)
    return {"name": name, "status": "ok" if not n_err else "error",
            "reason": f"{n_err} Stage2 conformers failed" if n_err else "",
            "lowest_energy": e_min_eh,
            "n_gfnff": n_gfnff, "n_final": n_final, "work": str(work)}


# ────────────────── 主流程 ──────────────────
def main():
    ap = argparse.ArgumentParser(
        description="批量两阶段构象搜索 (GFN-FF -> MACE-omol MLIP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("config", nargs="?", default="config_mlip.yaml",
                    help="YAML 配置文件路径 (默认: config_mlip.yaml)")
    ap.add_argument("--sample", action="store_true", help="打印示例 YAML 并退出")
    args = ap.parse_args()

    if args.sample:
        print(SAMPLE_CONFIG, end="")
        return 0

    if not Path(args.config).exists():
        print(f"ERROR: 配置文件 {args.config} 不存在. 用 --sample 查看示例.")
        return 1

    try:
        cfg = load_config(args.config)
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    # 检查 crest (Stage 1 需要); mace/ase 在子进程中 import
    if not shutil.which("crest"):
        print("ERROR: crest 不在 PATH 中. 请先激活对应 conda 环境.")
        return 1
    version = detect_crest_version()
    if version is None or version < (3, 0):
        print(f"ERROR: explicit TOML constraint loading requires CREST >= 3.0; "
              f"detected {version or 'unknown'}.")
        return 1
    if not cfg["stage1_only"] and not Path(cfg["mlip"]["model"]).is_file():
        print(f"ERROR: MACE 模型不存在: {cfg['mlip']['model']}")
        return 1

    inp = Path(cfg["input"]).resolve()
    out = Path(cfg["output"]).resolve()
    xyzs = sorted(inp.glob("*.xyz"))
    if not xyzs:
        print(f"ERROR: {inp} 中没有 .xyz 文件")
        return 1
    unknown_overrides = set(cfg["freeze_by_structure"]) - {xyz.stem for xyz in xyzs}
    if unknown_overrides:
        print(f"ERROR: freeze_by_structure 包含不存在的 XYZ 文件名: {sorted(unknown_overrides)}")
        return 1
    if out == inp or out.is_relative_to(inp):
        print("ERROR: 输出目录不能位于输入目录内")
        return 1
    for xyz in xyzs:
        try:
            frames = read_xyz_frames(xyz)
            if len(frames) != 1:
                raise ValueError("每个输入 XYZ 必须恰好包含一帧结构")
            atom_count = len(frames[0][0])
        except (OSError, ValueError) as exc:
            print(f"ERROR: XYZ 文件无效: {xyz}: {exc}")
            return 1
        atom_lists = structure_atom_lists(cfg, xyz.stem)
        for key in ("freeze", "notopo", "hardfreeze"):
            indices = atom_lists[key]
            if indices and (min(indices) < 1 or max(indices) > atom_count):
                print(f"ERROR: {key} 原子索引超出 {xyz} 的 {atom_count} 个原子")
                return 1
    if cfg["dry_run"]:
        print(f"DRY-RUN: 已检查 {len(xyzs)} 个 XYZ、配置、CREST 版本和模型文件路径；未验证模型加载、GPU 或约束解析，未启动计算")
        return 0
    try:
        prepare_output(out, cfg, xyzs)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    threads = cfg["threads"]; workers = cfg["workers"]
    global _stage1_slots
    _stage1_slots = threading.BoundedSemaphore(cfg.get("stage1_workers", workers))
    chrg = cfg["chrg"]; uhf = cfg["uhf"]
    frozen = cfg["freeze"]; cmetal = cfg["cmetal"]
    resume = cfg["resume"]; cleanup = cfg["cleanup"]
    dry_run = cfg["dry_run"]; full_log = cfg["full_log"]; hb = cfg["hb"]
    mlip_cfg = cfg["mlip"]

    ui = BatchUI(out / "batch_run.log")
    ui.print_line("=" * 72)
    ui.print_line("批量两阶段构象搜索 (GFN-FF -> MACE-omol MLIP)")
    ui.print_line(f"  配置:   {args.config}")
    ui.print_line(f"  输入:   {inp}  ({len(xyzs)} 个结构)")
    ui.print_line(f"  输出:   {out}")
    ui.print_line(f"  Stage1: threads={threads}  stage1_workers={cfg['stage1_workers']}  pipeline_workers={workers}")
    if mlip_cfg.get("stage1_only"):
        ui.print_line("  Stage2: 已关闭 (stage1_only: true) —— 只做 GFN-FF 构象采样, 不计算 MACE")
    else:
        ui.print_line(f"  Stage2: model={Path(mlip_cfg['model']).name}  charge={chrg}  spin(多重度)={mlip_cfg['spin']}")
    ui.print_line(f"          gpus={mlip_cfg['gpu_ids']}  cueq={mlip_cfg['enable_cueq']}  "
                  f"mode={mlip_cfg['mode']}  fmax={mlip_cfg['fmax']}  "
                  f"maxsteps={mlip_cfg['maxsteps']}  ewin={mlip_cfg['ewin_kcal_mol']} kcal/mol")
    ui.print_line(f"  电荷: {chrg}  UHF: {uhf}  冻结原子: {frozen}  cmetal: {cmetal}")
    ui.print_line(f"  Stage1 额外: notopo={cfg['notopo']}  hardfreeze={cfg['hardfreeze']}  "
                  f"fc_pos={cfg['fc_pos']}  quick={cfg['quick'] or 'standard'}  "
                  f"hardfreeze_tol={cfg['hardfreeze_tol_A']:g} Å")
    ui.print_line(f"  采样深度: crest_mdlen={cfg['crest_mdlen'] or 'CREST 自动（按分子柔性估算）'}")
    if cfg["hardfreeze"]:
        ui.print_line(f"  Stage1 冻结: 硬冻结 {cfg['hardfreeze']}（梯度清零）；校验冻结片段内部形变 < {cfg['hardfreeze_tol_A']:g} Å，"
                      "整体漂移只记录并在坐标写回时消除")
        if set(cfg["freeze"]) - set(cfg["hardfreeze"]):
            ui.print_line(f"  警告: freeze 中的 {sorted(set(cfg['freeze']) - set(cfg['hardfreeze']))} 不在 hardfreeze 内，"
                          "这些原子在 Stage1 只受软约束，位点会漂移")
    else:
        ui.print_line("  警告: 未设置 hardfreeze —— Stage1 仅有位置软约束，反应位点会漂移；"
                      "需要固定时请设为与 freeze 相同的原子表")
    ui.print_line(f"  resume={resume}  cleanup={cleanup}  dry_run={dry_run}")
    ui.print_line("=" * 72)

    # 创建 GPU 进程池 (每 GPU 一个进程); MACE 模型懒加载, Stage1 期间不占显存
    if mlip_cfg.get("stage1_only"):
        gpu_pools, result_queue = [], None
        ui.print_line("  仅采样模式: 不创建 GPU 进程")
    else:
        ui.print_line(f"创建 {len(mlip_cfg['gpu_ids'])} 个 GPU 进程 (MACE 将在首次 Stage2 时懒加载, Stage1 期间不占显存)")
        t0 = time.time()
        gpu_pools, result_queue = init_gpu_pools(
            mlip_cfg["gpu_ids"], mlip_cfg["model"],
            mlip_cfg["enable_cueq"], chrg, mlip_cfg["spin"])
        ui.print_line(f"  GPU 进程就绪 ({time.time()-t0:.0f}s, 模型未加载)")

    event_q = queue.Queue()
    status = {}
    status_lock = threading.Lock()
    stage2_lock = threading.Lock()  # 串行化 Stage 2 (单结构独占全部 GPU)
    results = []
    progress = {"done": 0, "ok": 0, "err": 0, "skipped": 0}
    total = len(xyzs)
    stop = threading.Event()

    def snap():
        with status_lock:
            return dict(status), dict(progress)

    def display():
        last_tick = time.time()
        while not (stop.is_set() and event_q.empty()):
            try:
                evt = event_q.get(timeout=1.0)
                if evt:
                    st, pg = snap()
                    ui.event(evt, st, {"total": total, **pg})
            except queue.Empty:
                pass
            if hb > 0 and time.time() - last_tick > hb:
                st, pg = snap()
                ui.tick(st, {"total": total, **pg})
                last_tick = time.time()
        while not event_q.empty():
            try:
                evt = event_q.get_nowait()
                st, pg = snap()
                ui.event(evt, st, {"total": total, **pg})
            except queue.Empty:
                break

    disp = threading.Thread(target=display, daemon=True)
    disp.start()

    def cached_complete(xyz):
        checkpoint = out / xyz.stem / "stage2_checkpoint.json"
        if not checkpoint.exists():
            return False
        try:
            saved = json.loads(checkpoint.read_text())
        except (OSError, ValueError):
            return False
        values = saved.get("results") if isinstance(saved, dict) else None
        return (isinstance(values, list) and bool(values)
                and all(isinstance(v, list) and v and v[0] == "ok" for v in values))

    # Finalize existing complete batches before dispatching new long GPU work.
    xyzs.sort(key=lambda xyz: (not cached_complete(xyz), xyz.name))
    def submit_structure(pool, xyz, func, *call_args):
        if not cached_complete(xyz):
            return pool.submit(func, *call_args)
        future = Future()
        try:
            future.set_result(func(*call_args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {}
            for i, xyz in enumerate(xyzs):
                wid = i + 1
                atom_lists = structure_atom_lists(cfg, xyz.stem)
                per_atoms = atom_lists["freeze"]
                per_mlip = dict(mlip_cfg, freeze_0based=[a - 1 for a in per_atoms])
                f = submit_structure(pool, xyz, process_structure, wid, str(xyz), str(out), threads,
                                chrg, uhf, per_atoms, cmetal, resume,
                                cleanup, dry_run, full_log,
                                per_mlip, gpu_pools, result_queue, stage2_lock,
                                ui, event_q, status, status_lock,
                                atom_lists["notopo"], cfg["quick"],
                                atom_lists["hardfreeze"], cfg["fc_pos"], cfg["crest_mdlen"])
                futs[f] = xyz.stem
            for f in as_completed(futs):
                name = futs[f]
                try:
                    r = f.result()
                    results.append(r)
                except Exception as e:
                    event_q.put(f"[ERR] {name} 崩溃: {e}")
                    r = {"name": name, "status": "error", "reason": str(e)}
                    results.append(r)
                with status_lock:
                    progress["done"] += 1
                    st = r["status"]
                    if st == "ok":
                        progress["ok"] += 1
                    elif st == "error":
                        progress["err"] += 1
                    elif st == "skipped":
                        progress["skipped"] += 1
    except KeyboardInterrupt:
        event_q.put("!! 用户中断 (Ctrl+C), 等待运行中的任务结束...")
    finally:
        stop.set()
        disp.join(timeout=5)
        # 关闭 GPU 池
        for p in gpu_pools:
            p.terminate()
            p.join()

    # ---- 汇总 ----
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    sk = [r for r in results if r["status"] == "skipped"]
    dr = [r for r in results if r["status"] == "dry_run"]

    ui.finish()
    ui.print_line("=" * 72)
    ui.print_line(f"完成: {len(ok)} 成功, {len(err)} 失败, {len(sk)} 跳过, "
                  f"{len(dr)} dry-run, 共 {len(results)}")
    if ok:
        level = "GFN-FF 级" if cfg["stage1_only"] else "MACE 级"
        ui.print_line(f"最低能量构象 ({level}, 按能量升序):")
        for r in sorted(ok, key=lambda x: x["lowest_energy"]):
            ui.print_line(f"  {r['name']:20s}  {r['lowest_energy']:.5f} Eh  "
                          f"({r['n_final']} 构象)  -> {r['work']}/crest_best.xyz")
    if err:
        ui.print_line("失败:")
        for r in err:
            ui.print_line(f"  {r['name']:20s}  {r.get('reason', '?')}")
    ui.print_line("=" * 72)

    with open(out / "batch_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["name", "status", "lowest_energy_Eh", "n_gfnff", "n_final", "work"])
        for r in results:
            w.writerow([r["name"], r["status"], r.get("lowest_energy", ""),
                        r.get("n_gfnff", ""), r.get("n_final", ""), r.get("work", "")])

    print(f"\n主日志: {out / 'batch_run.log'}")
    print(f"汇总CSV: {out / 'batch_summary.csv'}")
    return 0 if not err else 1


if __name__ == "__main__":
    sys.exit(main())
