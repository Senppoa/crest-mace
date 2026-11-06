# CREST 构象搜索 + MACE 能量排序：独立脚本新手教程

这个目录可以整体复制到另一台机器使用。计算入口只有 **`crest_mace.py` 一个 Python 脚本**：原包的主流程、配置解析、XYZ 读写和配置模板已经合并进去，不需要安装 `ts_conformers`，也不需要把原包里的其他脚本一起复制。

| 文件 | 用途 |
| --- | --- |
| `crest_mace.py` | 独立计算脚本，包含全部流程和内嵌配置模板 |
| `config_mlip_example.yaml` | 可编辑的配置示例，初始为 `dry_run: true` |
| `input/iso_321.xyz` | 132 原子的示例输入，电荷 0、单重态 |
| `examples/` | 三种可迁移配置：仅采样、MACE 单点、受约束优化 |
| `validation/` | 真实计算案例、原始日志、验证报告及回归证据 |
| `test_crest_mace.py` | 19 项回归测试 |
| `requirements*.txt` | Python 依赖清单 |
| `README.md` | 本教程 |

**先准备自己的 XYZ 结构和本地 MACE-OMol 模型。模板中的电荷、自旋、冻结原子号来自原包示例，必须根据自己的体系修改。** 本仓库包含示例分子结构；模型权重需自行准备。

## 仓库快速开始

本仓库已包含修复脚本、示例分子、可迁移配置和真实案例。模型权重未包含。

```bash
cd /path/to/crest-mace
pip install -r requirements.txt
# 另行安装 CREST >= 3.0，确保 crest 在 PATH 中。
python -m unittest -v test_crest_mace
python crest_mace.py examples/config_iso321_stage1.yaml
```

最后一条命令默认只预检。将该配置的 `dry_run` 改为 `false` 后再次运行，即可做无需模型/GPU的 GFN-FF 短时案例，结果写入 `runs/iso321_stage1/`。

要运行 MACE，先按下文安装适配 CUDA 的 PyTorch，再执行 `pip install -r requirements-mace.txt`，将 OMol 权重放到 `models/MACE-omol.model`。随后使用 `examples/config_iso321_sp.yaml` 或 `examples/config_iso321_opt.yaml`，同样先预检再将 `dry_run` 改为 `false`。两个配置默认只使用 GPU 0。所有路径相对 YAML 所在目录解析。

已有结果可直接查看 [真实单点案例](validation/final_sp/iso_321/)、[真实优化案例](validation/final_opt/iso_321/)、[仅采样案例](validation/final_stage1/iso_321/) 和 [验证报告](validation/REPORT.md)。其中优化只执行 5 步，短时采样用于验证程序，不代表搜索充分或优化收敛。

无需模型/GPU即可检查打包案例：

```bash
python validation/verify_real.py --check-only
```

独立重新求能需模型和 GPU 0、1：`OMP_NUM_THREADS=2 python validation/verify_real.py --model models/MACE-omol.model`。该工具按当前仓库位置读取案例；历史日志和 manifest 中的路径保持原样。`validation/crest_mace.before.py` 是修复前审计备份，计算入口始终是根目录的 `crest_mace.py`。

仓库提供 `.gitignore`，本地计算输出、模型权重与 Python 缓存不会加入版本管理。

## 1. 这个脚本计算什么

对 `input` 目录中的每个 `.xyz` 文件，依次执行：

1. 用 CREST 的 `--constrain` 生成约束文件，然后用 **GFN-FF** 做 CREST iMTD-GC 构象搜索。候选池备份为 `crest_conformers_gfnff.xyz`。
2. 按候选池的 GFN-FF 能量升序整理构象，处理反应位点的对齐与坐标恢复；若启用硬冻结，先剔除硬冻结位点几何不合格的帧，再取至多 `max_conformers` 个。
3. 用 **MACE-OMol** 对这些构象做单点计算，或者进行冻结指定原子的 ASE FIRE 优化。
4. 按 **MACE 能量** 从低到高重新排序，按相对最低能量的窗口保留构象，输出 `crest_ensemble.xyz` 和最低能结构 `crest_best.xyz`。

GFN-FF 在这里负责采样和进入 MACE 前的数量筛选；最终排序与输出能窗使用 MACE 能量。准备约束时命令中出现的 `--gfn2` 用于生成约束模板，不代表又做了一轮 GFN2-xTB 构象排序。

这条流程沿用原包的受约束过渡态候选构象用途。`opt` 是固定部分原子的结构优化，并不执行过渡态鞍点搜索、频率分析或 IRC。所得构象仍需结合研究目的做后续验证。

## 2. 安装环境

### 2.1 平台与软件

此版本使用 Linux 的 `multiprocessing` **fork** 和 NVIDIA CUDA。Windows 用户在 **WSL2 的 Linux 终端**中运行以下命令；原生 Windows、macOS、CPU-only 和 Apple MPS 计算路径未实现。

需要：

- Python：脚本本身要求 3.10 或以上；新建环境建议使用 3.11，并以所选 MACE/PyTorch 版本要求为准。
- CREST：至少 3.0，支持 TOML 输入及所需约束功能；实际二进制还须通过单结构试跑。
- xTB：按 CREST 安装要求准备，建议与 CREST 一起安装。
- Python 包：PyTorch、MACE、ASE、NumPy、PyYAML。
- NVIDIA GPU、可用驱动，以及能够访问 GPU 的 CUDA 版 PyTorch。

### 2.2 新建环境的步骤

已有可运行 CREST + MACE-OMol 环境时，激活现有环境即可。新建环境可以按下面的顺序操作：

```bash
conda create -n crest-mace -c conda-forge python=3.11 pip crest xtb
conda activate crest-mace
```

先打开 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)，选择 Linux、Pip、Python 和适合当前驱动的 CUDA 版本，执行页面生成的安装命令。之后安装其他 Python 依赖：

```bash
python -m pip install "mace-torch==0.3.14" ase numpy pyyaml
```

这里固定 MACE 0.3.14，是因为它已明确支持 MACE-OMol 的 charge/spin 输入；使用其他版本时先确认模型与计算器接口兼容。当前 MACE 主分支的环境要求与旧版本可能不同，安装前可查看 [MACE 安装说明](https://github.com/ACEsuit/mace#installation) 和 [0.3.14 发布说明](https://github.com/ACEsuit/mace/releases/tag/v0.3.14)。

检查环境：

```bash
crest --version
xtb --version
nvidia-smi
python -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda); print("GPU available:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count())'
python -c 'import yaml, numpy, ase; from mace.calculators import MACECalculator; print("Python dependencies OK")'
```

`GPU available` 应为 `True`。最后一条命令只检查依赖能否导入，不会加载你的模型。

CREST 的安装和 PATH 设置见 [官方安装文档](https://crest-lab.github.io/crest-docs/page/installation/install_basic.html)。原包记录过部分 CREST 3.0.2 构建在约束解析时段错误的情况；版本检查通过，并不能证明这个二进制可以完成本流程。

### 2.3 准备 MACE 模型

使用能由 `MACECalculator(model_paths=...)` 加载、适用于当前体系的本地 MACE-OMol 模型。官方模型入口见 [MACE-OMol 模型发布页](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omol_0)。下载后把实际文件路径写入 `mlip.model`，文件不必改名。

脚本不会自动下载模型。把其他 MACE 模型改名为 `MACE-omol.model` 不会让它获得 OMol 的电荷/自旋支持；模型覆盖的元素、化学空间及其适用精度需要根据模型说明核对。

## 3. 第一次运行：逐步操作

### 3.1 准备目录和输入

进入本目录，创建输入目录和模型目录：

```bash
cd /path/to/crest-mace
mkdir -p input models
```

放入自己的文件后，目录例如：

```text
crest-mace/
├── crest_mace.py
├── config_mlip_example.yaml
├── README.md
├── input/
│   └── ts1.xyz
└── models/
    └── MACE-omol.model
```

首次只放 **一个实际结构**。每个 XYZ 必须恰好一帧：第一行是原子总数，第二行是注释，随后每行是元素符号和三个坐标，坐标单位为 Å。输入注释无需包含能量。脚本只读取 `input` 目录顶层、扩展名为小写 `.xyz` 的文件，不递归读取子目录。

同一批使用相同电荷、自旋、MACE 模型及全局设置。不同组成、不同电荷或不同自旋的体系应分开配置；仅原子编号不同时，可以使用 `freeze_by_structure`。对应构象必须保留相同的原子顺序。

### 3.2 查看帮助或重新生成模板

```bash
python crest_mace.py --help
python crest_mace.py --sample
```

若要另存一份模板：

```bash
python crest_mace.py --sample > my_config.yaml
```

`--help` 和 `--sample` 不需要安装科学计算依赖。模板已经内嵌在脚本里，复制单个 `.py` 文件到别处仍能生成配置。

### 3.3 编辑配置

把 `config_mlip_example.yaml` 复制成自己的配置（例如 `config_mlip.yaml`），再用文本编辑器打开，至少核对这些字段：

| 字段 | 必须确认的内容 |
| --- | --- |
| `input` | 自己的单帧 XYZ 所在目录 |
| `output` | 一个新的结果目录，不能放在输入目录里面 |
| `chrg`、`uhf` | 实际总电荷及未成对电子数 |
| `freeze` | 自己体系的反应位点原子号，从 1 开始 |
| `mlip.model` | 真实模型文件路径 |
| `mlip.gpu_ids` | 当前机器可使用的 GPU 编号 |
| `mlip.mode` | 首次可以先用 `sp`，确认流程后再决定是否用 `opt` |

模板的 `freeze: [1, 35, 36, 37, 47]` 要求至少有 47 个原子，且这五个编号必须对应你想约束的位点。**不要为了让预检通过而随意选几个原子。** XYZ 原子编号 1 对应文件的第三行，原子编号 2 对应第四行；注释行不计入原子编号。

YAML 用空格缩进；`mlip` 下的参数前保留两个空格。布尔值写 `true` / `false`，不加引号。`null` 表示不启用或使用指定的自动规则。拼错参数名、重复原子号、非整数 GPU 编号等会直接报错。

**所有相对路径以 YAML 文件所在目录为基准**，与从哪个目录启动命令无关。例如配置在 `/work/task/config_mlip.yaml`，`input: ./input` 就对应 `/work/task/input`。路径需要使用当前 Linux/WSL 环境可访问的写法。

### 3.4 先预检

保持配置中的：

```yaml
dry_run: true
```

执行：

```bash
python crest_mace.py config_mlip.yaml
```

预检检查配置格式、XYZ 帧与坐标格式、原子编号范围、CREST 版本以及模型文件路径，不启动搜索、不加载 MACE、不创建结果目录。它不会检验模型内容、CUDA 可用性、化学合理性或 CREST 的真实约束解析。

### 3.5 做第一个真实任务

确认配置后，将 `dry_run` 改为 `false`，其余先保留模板的 `workers: 1`、`stage1_workers: 1`、`enable_cueq: false`、`full_log: true`，再运行同一命令：

```bash
python crest_mace.py config_mlip.yaml
```

如果没有给配置路径，默认读取当前工作目录下的 `config_mlip.yaml`。真实运行会启动 CPU 搜索并在 Stage2 使用 GPU；MACE 首次加载可能较慢，Stage1 期间不会加载模型。

完成后先查看 `batch_summary.csv`、两个阶段的日志、最低能结构和约束位点几何。确认单结构任务符合预期，再向输入目录添加同类结构，使用一个新的输出目录做批量计算。运行期间不要向同一个输出目录同时启动第二份脚本。

## 4. 全部配置参数的含义

下面“默认值”指 **省略该字段时脚本使用的值**。随附模板为了便于首次排错，显式设置了较小并行数、`enable_cueq: false`、`dry_run: true`、`resume: true` 和完整日志；不要把模板值与省略字段的默认值混为一谈。

### 4.1 输入、输出与 CPU 资源

| 参数 | 脚本默认值 | 含义与用法 |
| --- | --- | --- |
| `input` | 必填 | 输入目录，包含一个或多个单帧 `.xyz` 文件 |
| `output` | 必填 | 本批次输出目录。新任务用新目录；已有非空目录受 manifest 保护 |
| `threads` | `8` | 每个 CREST 搜索作业的 CPU 线程数，正整数；不是 GPU 数量 |
| `workers` | `1` | 同时在整条管线中的结构数，包括搜索、等待 MACE 和执行 MACE 的结构 |
| `stage1_workers` | 等于 `workers` | 同时执行 Stage1 的作业数，必须为正整数且不超过 `workers` |

CPU 搜索线程需求大致为 `threads × stage1_workers`，另给主进程及其他程序留出资源。例如 `threads: 8`、`workers: 3`、`stage1_workers: 2`，表示最多两个 CREST 作业同时搜索，约使用 16 个搜索线程，同时允许第三个结构等待或执行 Stage2。

Stage2 每次只有一个结构使用 GPU 池；该结构的构象轮流分配给各张 GPU，每 GPU 一个进程。它不是“每个 workers 都占一整套 GPU”。增大 `workers` 可以使 CPU 与 GPU 阶段更好重叠，但并不保证加速。

### 4.2 体系、电荷与约束

| 参数 | 脚本默认值 | 含义与用法 |
| --- | --- | --- |
| `chrg` | `1` | 整个体系的整数总电荷；`0` 中性，`1` 表示 +1，`-1` 表示 −1。不是某个金属的局部形式电荷 |
| `uhf` | `0` | CREST/xTB 的未成对电子数，非负整数；通常单重态 `0`、双重态 `1`、三重态 `2` |
| `freeze` | `[1, 35, 36, 37, 47]` | 非空、无重复的 **1-based** 原子列表，或 `"1,35,36,37,47"`。Stage1 用于生成软约束；Stage2 对齐/恢复这些原子坐标，在 `opt` 中用 ASE FixAtoms 硬固定 |
| `freeze_by_structure` | `{}` | 按 XYZ 文件名（去掉 `.xyz`）覆盖位点列表，见下文 |
| `fc_pos` | `10.0` | 写入 `constraints.inp` 中唯一的 `force constant=` 数值，必须大于 0；软约束强度，不是距离容差 |
| `hardfreeze` | `null` | 可选的 Stage1 真正硬冻结原子列表，通过 TOML `[calculation].freeze` 传入；不替代 Stage2 的 `freeze` |
| `hardfreeze_tol_A` | `0.05` | 硬冻结片段**内部距离**允许的最大变化（Å）。超过则剔除该帧；片段整体的平移/转动只记录，不剔除 |
| `cmetal` | `0.2` | 传给 CREST `--cmetal` 的金属 M-X 键约束强度，用于初始结构检测到的过渡金属相关键；并非冻结金属坐标。**它会额外生成 `bondlengths` 金属配位距离约束，可能与 `constraints.inp` 叠加干扰**；不需要金属配位保护时建议设 `0` |
| `notopo` | `null` | 可选的 CREGEN 拓扑检查忽略原子列表；不固定坐标、不产生约束 |
| `quick` | `null` | `null` 为标准采样；`quick` / `squick` / `mquick` 逐步缩减采样预算。不要写 `--quick`；脚本自动加前缀 |
| `crest_mdlen` | `null` | Stage1 采样深度，对应 CREST `--len`。`null` 由 CREST 按分子柔性自动估算；数字表示**每条 MTD 的长度**（ps），如 `60`；字符串 `"x2.0"` / `"x0.5"` 表示相对默认长度的倍数。与 `quick` 同时设置时，`quick` 先缩减档位、`crest_mdlen` 再覆盖长度（以 CREST 实际行为为准，见日志的 `Σ(t(MTD))`） |
| `stage1_only` | `false` | `true` 时只做 Stage1 GFN-FF 构象采样，跳过 MACE 计算、能量窗排序与 GPU 进程；此时 `mlip.model` / `mlip.gpu_ids` 可不填 |

`freeze`、`hardfreeze`、`notopo` 和逐结构覆盖中的原子号都从 **1** 开始，支持整数列表或逗号分隔字符串；此脚本不解析 `1-5` 范围表达式。

**软约束和硬冻结的区别：**

- `freeze` 在 Stage1 通过 `crest --constrain` 生成 xTB 风格约束，保持选定原子相互几何接近参考值，允许有限偏移；不是笛卡尔坐标精确锁定。
- `fc_pos` 改变这项软约束的强度。xTB 对 `force constant=` 使用原子单位 Hartree/Bohr²；原包采用的 `10.0` 是很强的软约束，旧生成器值 `0.5` 也不等于硬冻结。数值需结合体系和约束能/位点偏移检查，不能当作普适最优值。参见 [xTB 约束说明](https://xtb-docs.readthedocs.io/en/latest/xcontrol.html)。
- `hardfreeze` 只控制 Stage1 的精确冻结。若目标是两个阶段固定相同反应位点，可把它设为与 `freeze` 相同的列表，并确认目标 CREST 支持 TOML 冻结。
- **硬冻结的实际语义**：CREST 的 `freeze` 只是把冻结原子的**梯度置零**（见 CREST 源码 `src/calculator/calc_type.f90: calculation_freezegrad`），并**不**把它们钉在固定笛卡尔坐标上。因此冻结片段内部几何会被保持，但整个片段仍可能随分子平动/转动。
- **核验判据**：脚本在刚体对齐后，只检查冻结片段的**内部距离变化** `max |d_ij − d_ij_ref|`，超过 `hardfreeze_tol_A`（默认 `0.05 Å`）才剔除该帧；整体漂移量写入 `freeze_validation.json` 的 `max_rigid_shift_A`，仅作记录。随后把冻结原子坐标直接写回输入 TS 的原值，消除残余平移/转动。
- 未设置 `hardfreeze` 时，Stage1 只有位置软约束，位点会明显漂移；脚本启动时会打印警告。
- Stage2 `opt` 始终固定 `freeze` 列表；单独出现在 `hardfreeze` 中的原子不会因此在 Stage2 固定。

`cmetal` 的数值原样传给 CREST，CLI 文档将此力常数参数以 Eh 标示；原包用的 `0.2` 不是 CREST 手册给出的默认 `0.02`。`fc_pos` 与 `cmetal` 作用于不同的约束设置，应分别核对实际生成的约束和搜索日志。金属键约束、拓扑忽略和 quick 选项的定义见 [CREST 命令行参数](https://crest-lab.github.io/crest-docs/page/documentation/keywords.html)。

**即使选择 `mode: sp`，MACE 也不是直接读取未经处理的 CREST 坐标：** 脚本先按参考位点做刚体对齐，再把 `freeze` 原子的坐标恢复为原始输入值，随后计算单点。只做刚体旋转/平移不会改变内部几何，但恢复有偏移的原子会改变内部几何。因此，Stage1 软约束偏移很大时，要检查 `freeze_validation.json` 和处理后的构象，不能只看最终能量。

`stage1_only: true` 时没有能量重算：`crest_ensemble.xyz` / `crest_best.xyz` 的注释保留原始 CREST 候选的 GFN-FF 能量，而坐标已经对齐并恢复 `freeze` 原子。这些能量用于追溯和排序，不能视为恢复坐标后几何的重新求值；原始坐标与对应能量保存在 `crest_conformers_gfnff.xyz`。`max_rigid_shift_A` 记录对齐前硬冻结片段质心的平移距离，不表示旋转角，也不等同于对齐后的残余形变。

逐结构原子编号示例：

```yaml
freeze: [1, 35, 36, 37, 47]
freeze_by_structure:
  ts_A: [1, 35, 36, 37, 47]
  ts_B: [1, 34, 35, 36, 46]
hardfreeze: [1, 35, 36, 37, 47]
notopo: null
```

当输入文件是 `ts_B.xyz` 时，实际 `freeze` 使用 `[1,34,35,36,46]`。**覆盖规则沿用原包：同一覆盖列表还会替换所有已经启用的 `hardfreeze` 和 `notopo` 列表；`null` 的项保持不启用。** 它不支持为三类列表分别定义不同的逐结构覆盖。如果需要这样的设置，拆成不同配置运行。

脚本还固定传入 `--subrmsd` 和 `--noreftopo`，并用 `sampling.toml` 显式加载 `constraints.inp`；这些不是 YAML 可调参数。采样深度通过 `crest_mdlen` 透传为 `--len`（可选）。`--noreftopo` 关闭初始参考拓扑检查，`notopo` 作用于后续 CREGEN 原子拓扑筛选，二者作用阶段不同。放宽拓扑筛选之后，仍需检查候选构象的键连关系。

### 4.3 MACE 计算、优化和输出筛选

以下字段全部写在 `mlip:` 下。

| 参数 | 脚本默认值 | 含义与用法 |
| --- | --- | --- |
| `model` | 必填 | 本地模型文件路径；相对 YAML 所在目录解析 |
| `gpu_ids` | `[0]` | 无重复的非负整数 GPU 列表，例如 `[0]` 或 `[0,1]`；不支持 `cpu` |
| `enable_cueq` | `true` | 是否尝试 cuEquivariance 推理加速；模板先设 `false`，排除可选加速依赖问题 |
| `spin` | `null` → `uhf+1` | MACE-OMol 的自旋多重度 `2S+1`，正整数；单重态 `1`、双重态 `2`、三重态 `3` |
| `mode` | `sp` | `sp`：只计算处理后几何的单点能量；`opt`：先用 ASE FIRE 优化非冻结原子，再计算末态能量 |
| `fmax` | `0.04` | `opt` 的最大原子力收敛阈值，单位 **eV/Å**，必须大于 0；数值越小要求越严格；`sp` 不使用 |
| `maxsteps` | `600` | `opt` 的优化步数上限，正整数；`sp` 不使用 |
| `ewin_kcal_mol` | `6.0` | 相对于本输入结构中最低 MACE 能量的输出能窗，单位 **kcal/mol**，非负数；`0` 只保留最低能及等能构象 |
| `max_conformers` | `500` | 进入 MACE 的构象数量上限，正整数或 `null`；按 GFN-FF 能量取前 N 个，`null` 表示全部合格帧 |
| `include_initial_ts` | `false` | 为未优化的原始输入结构另算 MACE 单点，并额外生成 `ts_ensemble.xyz`，详见结果说明 |

MACE-OMol 将 `charge` 和 `spin` 作为模型输入；脚本分别传入 `chrg` 和上述多重度。**`uhf: 1` 与 `mlip.spin: 1` 含义不同**：前者通常为双重态未成对电子数，后者是单重态多重度。自动映射只做 `uhf+1`，不会从 XYZ 推断电子态。MACE 的约定见 [官方 0.3.14 示例](https://github.com/ACEsuit/mace/releases/tag/v0.3.14)。

优化时的 `fmax` 作用于可移动自由度；冻结原子的受约束力不用于判断自由度收敛。`maxsteps` 耗尽时，脚本仍会保留末态能量并参与排序，`optimization_status.json` 中写 `converged: false`，不会自动排除它。**命令成功不等于所有优化都收敛**；用 `opt` 时必须检查这个文件。ASE 优化算法和收敛定义见 [ASE 结构优化文档](https://ase.gitlab.io/ase/ase/optimize.html)。

`max_conformers` 与 `ewin_kcal_mol` 控制两个不同阶段。例如先有 900 个合格 GFN-FF 构象，`max_conformers: 200` 只让其中 GFN-FF 最低的 200 个进入 MACE；随后 `ewin_kcal_mol: 3.0` 仅输出这 200 个中处于 MACE 最低能量以上 3 kcal/mol 内的构象。未进入这 200 个的构象不会再被 MACE“找回来”；较低的 GFN-FF 能量不保证较低的 MACE 能量。

每个 GPU 子进程会把 `CUDA_VISIBLE_DEVICES` 设置为配置中的单个编号，内部统一用 `cuda:0`。它会覆盖继承的可见设备设置；在作业调度器或容器中使用时，需按分配策略确认编号。CuEq 初始化失败时会尝试回退到普通 MACE CUDA 推理，**不会回退到 CPU**；需要加速时按 [MACE CUDA 加速说明](https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html) 匹配 torch/CUDA/CuEq。

### 4.4 运行、日志与续跑

| 参数 | 脚本默认值 | 含义与用法 |
| --- | --- | --- |
| `resume` | `false` | `true` 时，仅在配置、输入文件、模型文件和脚本 SHA256 均匹配的前提下续跑；已完成结构跳过，Stage1 已完成时可复用候选池 |
| `cleanup` | `false` | 写出结果后删除工作目录内 `MDFILES`、`tmp`、`TMPDIR`、`exmundus` 及 `*.trj`，保留主要系综、日志和检查点 |
| `dry_run` | `false` | `true` 只预检，不执行计算；模板显式设为 `true` |
| `full_log` | `false` | `true` 保留 CREST 完整输出；`false` 只保留关键行。首次运行建议完整日志 |
| `hb` | `2` | 交互终端面板定时刷新间隔，单位秒；`0` 关闭定时刷新，不影响计算或事件日志 |

续跑仍使用原命令：

```bash
python crest_mace.py config_mlip.yaml
```

要续跑，第一次真实计算时就设 `resume: true`。它也会记录在配置指纹中：原任务若为 `false`，事后改成 `true` 并不能通过原目录的匹配检查。

Stage2 的检查点逐构象保存成功结果；未完成项及计算报错项在续跑时重新提交。优化未收敛但成功返回能量的项视为已完成，不会自动重算。修改模型、模式、能窗、步数、并行参数、日志开关等任一有效配置，或修改脚本/输入内容，都应指定一个新的输出目录。不同版本原包的结果也不能直接当成本独立脚本的续跑目录。

manifest 没有记录 CREST 可执行文件或 MACE、ASE、PyTorch/CUDA 等运行环境的指纹。更换或升级这些软件时，即使其他文件都没变，也应使用新的输出目录；环境一致性需要自行保持。

## 5. 常用配置调整

以下片段用于修改完整配置中对应的字段，不是完整 YAML 文件。

**只做 MACE 单点排序：**

```yaml
mlip:
  mode: sp
```

**固定反应位点、优化其他原子后排序：**

```yaml
freeze: [1, 35, 36, 37, 47]  # 必须换成自己的位点
hardfreeze: [1, 35, 36, 37, 47]
mlip:
  mode: opt
  fmax: 0.04
  maxsteps: 600
```

`hardfreeze` 是可选设置；启用后要求 CREST 支持实际硬冻结（3.1+ 为正式特性，3.0.x 部分构建可用，日志中应出现 `FROZEN ATOMS`）。
**两张表必须一致**：只写 `freeze` 时 Stage1 只有距离软约束，反应位点会明显漂移（实测可达 0.5–1.4 Å）；
`hardfreeze` 缺失时脚本会在启动日志打印警告。模式改变时换一个新的 `output`。

**减少首次排错的 MACE 工作量：**

```yaml
workers: 1
stage1_workers: 1
mlip:
  max_conformers: 20
```

`20` 仅用于流程试跑，不是科研采样充分性的建议。需要减少 Stage1 预算时可显式使用 `quick: mquick`，并为正式标准采样另建输出目录。

**只做 CREST 构象采样，不跑 MACE（无需 GPU 与模型）：**

```yaml
stage1_only: true
mlip:
  model: null
  gpu_ids: []
```

开启后脚本不创建 GPU 进程、不加载 MACE，也不需要 `model` 存在。每个结构输出校验通过的 GFN-FF 构象池，
`crest_ensemble.xyz` / `crest_best.xyz` 的注释行能量是 **GFN-FF 级 Hartree**（不是 MACE 能量），
不会按 MACE 能量排序，也不受 `ewin_kcal_mol` 影响。`max_conformers` 仍会按 GFN-FF 能量截断。
适合先用 CPU 大规模采样、再拿构象池做后续高精度计算的工作流。

**调整采样深度：**

```yaml
quick: null            # 标准预算
crest_mdlen: "x2.0"    # 每条 MTD 长度 = CREST 自动估算值 ×2；也可写数字如 60（ps）
```

`crest_mdlen: null` 时完全由 CREST 按分子柔性估算。实际生效值看 `stage1_gfnff.log` 里的
`Σ(t(MTD)) / ps` 与 `MD simulation time`。注意采样加深不保证构象池线性变大，
CREGEN 的拓扑过滤与能量窗会再筛一轮。

**增加覆盖范围：**

```yaml
mlip:
  max_conformers: null
  ewin_kcal_mol: 6.0
```

这会计算全部合格候选帧，时间显著取决于候选数和 `sp`/`opt`。它不会扩大 CREST 已经生成的候选池，也不会恢复 CREST 自身筛掉的构象。

**两张 GPU 配合两个 CPU 搜索作业：**

```yaml
threads: 8
workers: 3
stage1_workers: 2
mlip:
  gpu_ids: [0, 1]
```

先确认有相应的 CPU 线程和两张可用 GPU。单张 GPU 上的模型或单个结构显存不足，不能靠添加另一张 GPU 自动拆分模型解决。

## 6. 结果文件怎么读

例如输入是 `ts1.xyz`，输出结构如下；某些文件由 CREST 版本或选项决定：

```text
results_sp/
├── crest_mlip_manifest.json
├── batch_run.log
├── batch_summary.csv
└── ts1/
    ├── struc.xyz
    ├── constraints.inp
    ├── coord.ref
    ├── sampling.toml
    ├── constraints_setup.log
    ├── stage1_gfnff.log
    ├── crest_conformers_gfnff.xyz
    ├── crest_conformers.xyz
    ├── freeze_validation.json
    ├── stage2_checkpoint.json
    ├── stage2_mlip.log
    ├── optimization_status.json
    ├── crest_ensemble.xyz
    └── crest_best.xyz
```

| 文件 | 读取重点 |
| --- | --- |
| `batch_summary.csv` | 每个输入结构的状态、最低能量（Eh）、GFN-FF 构象数及最终构象数；UTF-8 BOM，可用 Excel 打开中文名字。`stage1_only: true` 时该能量为 **GFN-FF 级** |
| `batch_run.log` | 整批任务的进度、失败和输出位置；续跑会重写本次批日志/汇总 |
| `crest_mlip_manifest.json` | 有效配置及脚本、模型、输入文件 SHA256，用于拒绝错误复用 |
| `constraints.inp`、`sampling.toml` | 实际交给 CREST 的约束和计算设置；排查约束是否载入时检查 |
| `constraints_setup.log`、`stage1_gfnff.log` | 约束生成与 CREST 搜索日志 |
| `crest_conformers_gfnff.xyz` | 原始 Stage1 候选池备份；保留 GFN-FF 能量与几何 |
| `crest_conformers.xyz` | Stage1 原始工作系综，**不是**最终结果；其中冻结原子**尚未**写回输入坐标（含 CREGEN 带来的整体平移/旋转）。需要精确冻结坐标时请用 `crest_ensemble.xyz` |
| `freeze_validation.json` | `max_internal_diff_A`：冻结片段内部距离最大变化（用于剔除）；`max_rigid_shift_A`：整体漂移（仅记录）；`max_aligned_stage1_drift_A`：`freeze` 位点对齐后位移；以及硬冻结原子表、`rejected_frames`、`selected_gfnff_ranks`（排序后序号，从 1 开始） |
| `stage2_checkpoint.json` | 逐构象计算结果及几何指纹，用于续跑；不要手动改写 |
| `optimization_status.json` | 每个进入 Stage2 的构象的状态、步数、`converged` 和能量（eV） |
| `stage2_mlip.log` | MACE 模型/模式、成功失败数、筛选后排名、能量、相对能量和失败信息；其中 `rank`、`idx` 从 0 开始，`idx` 是筛选后 Stage2 数组索引 |
| `crest_ensemble.xyz` | 最终系综，每帧第二行是 **Hartree** 能量。正常模式按 MACE 能量从低到高并受 `ewin_kcal_mol` 限制；`stage1_only: true` 时按 **GFN-FF 能量**排序、不做能窗筛选。冻结原子坐标均已写回输入值 |
| `crest_best.xyz` | 最低能量构象，即最终系综第一帧 |
| `stage2_mlip.log` | 正常模式记录 MACE 结果；`stage1_only: true` 时首行为 `STAGE1_ONLY_DONE` 并说明能量级别 |

能量转换采用：

$$
E_{\mathrm{Hartree}} = \frac{E_{\mathrm{eV}}}{27.211386245988}
$$

$$
\Delta E_i\; (\mathrm{kcal/mol}) = (E_i - E_{\min})\; (\mathrm{eV}) \times 23.060549
$$

能窗只在 **同一个输入结构的候选池内**计算，窗口内至少保留最低能构象。不要根据不同组成或不同电子态结构的绝对能量大小直接做构象优劣比较。

`include_initial_ts: true` 还会生成 `initial_true_ts.xyz` 和 `ts_ensemble.xyz`。前者保留原始输入文件；后者第一帧为原始输入几何，其注释行换成 MACE 单点能，之后才接排序后的系综。**`ts_ensemble.xyz` 整体不按能量排序**，原始第一帧也不受上述能窗限制，可能与后续帧重复。该开关不会改变 `crest_best.xyz` 的定义，也不会验证原始结构是否为真实 TS。

MACE 优化后没有额外的 RMSD 去重或拓扑复检；多个候选可能收敛到相同构象。结果用于后续分析前要检查重复、键连关系及约束位点。

## 7. 常见问题

| 现象 | 处理办法 |
| --- | --- |
| `crest 不在 PATH` | 激活安装 CREST 的环境；在同一终端运行 `crest --version` 检查 |
| 模型不存在 | 检查 `mlip.model`，注意相对路径以 YAML 所在目录为基准 |
| 原子索引越界 | 根据自己的 XYZ 重填原子号，模板不是通用编号；不要把 0-based 编号直接复制进来 |
| 没有 `.xyz` 文件 | 确认 `input` 是目录、文件在顶层且扩展名小写 |
| `拒绝覆盖或复用不匹配的输出目录` | 计算条件或文件指纹有变化，使用新的 `output`；不要靠删除 manifest 绕过检查 |
| `crest 不在 PATH` 或 `detected (2, 12)` 等旧版本 | 脚本要求 CREST ≥ 3.0；`which -a crest` 确认先命中的是 3.x。conda 环境里若同时装了旧版 `crest` 包，需卸载或用 PATH 让新版优先 |
| 只设了 `freeze`、位点却漂移 | `freeze` 只是距离软约束；要固定位点必须同时设 `hardfreeze`，并在 `stage1_gfnff.log` 里确认出现 `FROZEN ATOMS` |
| 预检通过但运行时 `mlip.model` 报错 | `dry_run: true` 只检查路径格式；确认文件真实存在且能被 `MACECalculator(model_paths=...)` 加载（含 CUDA 序列化的模型在无 GPU 机器上会失败） |
| Stage1 跑完构象数很少（1–3 个） | 构象先经 CREGEN 拓扑过滤与 5 kcal/mol 能窗筛选；冻结位点越多、`quick` 档位越低，剩余构象越少。加深采样用 `quick: null` + `crest_mdlen`，不要靠改 `max_conformers` |
| CREST 段错误或约束解析失败 | 查看完整 `constraints_setup.log`、`stage1_gfnff.log` 和实际 TOML/约束；更换已验证构建或排查程序兼容性后，用新目录试跑 |
| `No conformers pass hard frozen site validation` | 查看硬冻结设置和位点几何；核查目标 CREST 是否真实执行冻结及输入原子顺序，不要直接删除检查逻辑 |
| CUDA 不可用 | 检查驱动、WSL2 GPU 支持、CUDA 版 torch 和 GPU 编号；这个脚本没有 CPU 回退 |
| CuEq 报错 | 先设 `enable_cueq: false`，用新目录验证普通 MACE CUDA 推理，再配置加速依赖 |
| GPU 显存不足 | 换适合可用显存的兼容模型或更大显存 GPU；减少 `max_conformers` 主要减少总工作量，不会缩小单构象模型的显存需求 |
| `converged: false` | 该末态仍可能出现在最终系综；检查几何和优化需求，必要时调整阈值/步数，用新目录重算 |
| 部分 MACE 构象失败 | 可有部分输出，但批任务返回非零；检查 `stage2_mlip.log` 与检查点，条件不变且原配置为 `resume: true` 时可以续跑 |
| Stage2 看似没有进展 | 模型首次加载可能耗时；若 GPU 子进程被系统杀死，现有队列等待可能持续，需结合进程、显存及终端错误判断后中断续跑 |

正常完整计算返回退出码 `0`，计算失败返回非零；`status: ok` 只表示没有构象计算异常，不承诺所有优化都收敛，也不承诺采样已经充分。

## 8. 提取范围与验证记录

脚本合并自原包的 `crest_search.py`、`crest_mlip_config.py` 和 `crest_mlip_xyz.py`，内嵌了配置模板。保留 GFN-FF 采样、MACE 单点/受约束优化、多 GPU、原始结构附加输出、检查点、严格续跑身份检查、清理和批量日志；原包的 MMMC、xTB 描述符和审计功能不在这个独立目录中。

提取版补充了配置类型/未知键和完整 XYZ 格式检查，并修正了 `hardfreeze` 与 `freeze` 列表不同时的几何核验：硬冻结现在按实际 `hardfreeze` 原子表对齐、检查，而不是误用软约束位点列表。`fc_pos` 的所有取值（包括 `0.5`）都会显式写入约束，避免依赖 CREST 生成器的默认值。模型/输入/脚本身份检查仅依赖这一个 `.py` 文件，不再读取原包辅助脚本。

### 8.1 已实际验证的部分

2026-11-05 使用本目录的 `input/iso_321.xyz`（132 个原子，注释为电荷 0、单重态）完成了真实 CREST → MACE 验证。使用 CREST 3.0.2、NVIDIA L20、MACE-OMol extra-large 模型；本次配置使用 `chrg: 0`、`uhf: 0`、`freeze` / `hardfreeze: [1,2,3,5]`。示例模板中的电荷和冻结编号仍需按实际体系修改。

| 真实任务 | 结果与证据 |
| --- | --- |
| GFN-FF → MACE 单点 | [配置](validation/final_sp.yaml)、[输出](validation/final_sp/iso_321/)、[日志](validation/final_sp.console.log) |
| GFN-FF → MACE 受约束优化 | [配置](validation/final_opt.yaml)、[输出](validation/final_opt/iso_321/)、[日志](validation/final_opt.console.log)；限制 5 步，正确记录未收敛 |
| 仅 Stage1，无模型/GPU | [配置](validation/final_stage1.yaml)、[输出](validation/final_stage1/iso_321/)、[日志](validation/final_stage1.console.log)；故意填写不存在的模型路径，仍可成功 |
| 三种任务续跑 | `validation/final_*.resume.log` 均确认跳过已完成计算 |
| 双 GPU、输出能量独立复算 | [验证程序](validation/verify_real.py)、[核验结果](validation/verification.json) |

这些真实任务使用 `quick: mquick`、`crest_mdlen: 1`，用于验证功能，不代表构象搜索充分或优化收敛。CREST 含随机采样，不同运行得到不同构象数和能量是正常现象。`crest_mdlen: 0.1` 在本机 CREST 3.0.2 下生成空轨迹并失败，失败证据保存在 `validation/real_sp/`；短时测试应检查轨迹和 CREST 日志，不能假定任何正数都能产生可用构象。

详细结论、修复说明、验证边界和历史复现步骤见 [本次验证报告](validation/REPORT.md)。本仓库收录本次验证案例；原工作目录的历史 `results_frozen/` 未打包。

### 8.2 回归检查

安装 NumPy、PyYAML、ASE 后执行：

```bash
python -m unittest -v test_crest_mace
```

19 项测试覆盖配置类型、XYZ 格式与元素、GFN-FF 排序和能窗、冻结核验与坐标恢复、GFN-FF 独立采样续跑、原始输入附加、ASE 单点/受约束 FIRE、检查点续算与失败重试、损坏结果拒绝、GPU 子进程退出检测及 CLI 预检。[测试日志](validation/unittest.log) 保存本次运行结果。测试中的部分科学计算使用可控计算器或模拟结果；真实 MACE 验证单独记录在 8.1 节。

脚本修订后身份哈希改变，旧输出目录会按设计拒绝续跑。请用新输出目录验证新版，不要修改旧 manifest 绕过身份检查。`validation/*.yaml` 与原始日志保存历史机器的绝对路径，仅供追溯；请使用 `examples/` 下的相对路径配置重新计算，输出写入 `runs/`。案例的 manifest 保留原值，不应用来在新位置续跑。

## 9. 许可证

本项目以 [MIT License](LICENSE) 发布，可自由使用、修改、分发和商用，请保留原始版权声明。

Copyright (c) 2026 Tang Kun
