# crest_mace.py 功能验证报告

日期：2026-11-05。输入：`input/iso_321.xyz`，132 个原子，含 Rh，文件注释标明 charge=0、multiplicity=1。

## 结论

在修复下述问题后，已验证配置与输入校验、CREST 约束生成和 GFN-FF 搜索、冻结核验及恢复、MACE 单点、受约束 FIRE 优化、能量换算与排序、输出能窗、原始输入附加、双 GPU 分配、检查点和续跑的主要路径。验证同时使用真实 CREST/MACE 和可控的回归测试；测试方式和证据分别列出，不能将模拟计算的覆盖范围理解为全部真实体系都经过验证。

当前脚本适合继续做实际体系计算。此结论限于程序行为和输出一致性，不证明模型对该 Rh 体系的量化精度，也不证明短时采样足够。

## 环境及配置

- Python：`/home/user/miniconda3/envs/mace/bin/python`，Python 3.12。
- CREST 3.0.2；mace-torch 0.3.15；PyTorch 2.9.1+cu128；ASE 3.27.0；NumPy 2.2.6；PyYAML 6.0.3。
- NVIDIA L20；真实单点配置 GPU 0、1，优化配置 GPU 0；关闭 CuEq。
- 模型：`../Negishi/model/MACE-omol-0-extra-large-1024.model`。模型哈希记录在各真实任务的 `crest_mlip_manifest.json` 中，没有复制或修改模型。
- `chrg: 0`、`uhf: 0`、`mlip.spin: 1`；`freeze` 与 `hardfreeze` 均为 `[1,2,3,5]`；`fc_pos: 10`，`cmetal: 0`，冻结内部形变容差 0.05 Å。
- 功能验证预算：`quick: mquick`、`crest_mdlen: 1`、4 CPU 线程、单结构流水线；MACE 最多取 3 个候选，输出窗口 6 kcal/mol；优化最多 5 步。
- 检查了本机 MACECalculator 的实现，其默认 info 映射为 `total_charge -> charge`、`total_spin -> spin`，与脚本传参一致。

没有修改原输入和已有 `results_frozen/`。新增结果集中放在 `validation/`；原脚本备份为 `crest_mace.before.py`，改动可查 `changes.diff`。

## 发现与修复

| 问题 | 影响 | 修复和核验 |
| --- | --- | --- |
| 仅采样写入 `STAGE1_ONLY_DONE`，续跑只认 `STAGE2_MLIP_DONE` | 计算完成后仍重新整理结果，不能直接跳过 | 按模式识别完成标记；仅采样忽略 `include_initial_ts` 的输出要求；三种真实模式重复运行均跳过 |
| 仅采样仍尝试读取模型文件计算哈希 | 配置保留不存在的模型路径时，即使不做 MACE 也启动失败 | 仅采样不哈希未使用模型；真实 `final_stage1.yaml` 故意填写缺失模型并通过 |
| `mlip: []` / `false` 被 `or {}` 吞掉 | 非法类型被静默当作默认配置 | 仅将 null 转为空映射，保留严格类型检查 |
| XYZ 只检查元素字符格式 | `Xx` 等假元素能通过预检 | 校验 118 个元素符号，保留 `RH -> Rh` 兼容性 |
| 未拒绝 GFN-FF 注释中的 NaN/Inf | 排序、能窗和输出可能携带非有限能量 | 读取候选能量时要求有限值 |
| 对齐后再计算冻结片段质心位移 | `max_rigid_shift_A` 被对齐操作压到约零，丢失真实平移信息 | 改为对齐前记录；平移 `[4,-2,1]` 的测试得到 `sqrt(21)` Å |
| 原输入末尾没有换行时直接拼接系综 | 末行坐标与下一帧原子数粘连，`ts_ensemble.xyz` 损坏 | 拼接前补换行，原始备份仍逐字节保留；严格重新解析附加文件 |
| 仅采样汇总仍标为 MACE 级 | 容易误读计算层级 | 按模式标注 GFN-FF/MACE，跳过日志也标注对应阶段 |
| GPU 子进程原生崩溃时无期限等待结果队列 | 任务可能一直等待已死亡进程 | 周期检查任务开始时的 Pool worker 退出状态，异常退出时报错；有针对性的回归测试 |

初版 14 项回归测试在原脚本上复现了 5 个失败和 1 个异常（见 `before_regression.log`）。修复后扩展为 19 项，全部通过（见 `unittest.log`）。GPU 退出测试使用模拟已退出 worker，未在生产 GPU 上注入驱动故障。

## 验证覆盖与证据

### 真实计算

1. `final_sp.yaml`：从原始输入重新生成约束、运行 CREST GFN-FF、对齐与冻结恢复、运行 MACE 单点，并额外计算原始输入单点。
2. `final_opt.yaml`：独立重新采样后做最多 5 步 FIRE 优化；冻结原子保持输入坐标，未收敛状态写入 `optimization_status.json`，末态按当前设计保留。
3. `final_stage1.yaml`：不创建 GPU 计算进程，使用无效模型路径仍能完成纯 GFN-FF 路径。
4. 每种配置紧接着原样运行一次，`final_*.resume.log` 均记录 1 个跳过。
5. `verify_real.py` 检查最终文件可完整解析、元素顺序、冻结原子、升序能量、最低结构、能窗、原始输入附加顺序、输入备份逐字节一致性，以及 manifest 与最终脚本、输入哈希一致。
6. 极短搜索可能只产生一个候选，无法据此证明两个 GPU 都承担计算。因此另用原始结构和一个移动第 11 个原子 0.02 Å 的结构，调用真实双 GPU Stage2，确保 GPU 0、1 各接收一个任务。随后通过全新 ASE/MACECalculator 独立复算两个结构的能量。
7. 同一个独立计算器复算单点和优化输出的 `crest_best.xyz` 及原始输入；容许输出 XYZ 的 8 位小数舍入误差，要求最低结构能量偏差小于 `1e-5 eV`。机器可读结果见 `verification.json`，完整日志见 `verify_real.log`。

真实任务的配置、终端日志、CREST 原始日志、约束文件、采样 TOML、候选备份、检查点、输出系综和优化状态均保留在对应目录，便于逐项追溯。

### 自动回归

`test_crest_mace.py` 使用真实 132 原子输入，覆盖以下分支：

- XYZ 单帧/多帧往返、截断、非法坐标/元素；配置错误类型、重复索引、无效参数；路径与原子覆盖规则、spin 映射。
- 严格 manifest 拒绝输入变化；仅 Stage1 忽略未使用模型。
- 无序 GFN-FF 候选排序；刚体平移/旋转消除；恢复冻结原子；拒绝内部形变；`hardfreeze` 与 `freeze` 不同时使用正确核验原子表；过滤后才执行数量限制；全部拒绝和原子顺序改变时报错。
- MACE 输出排序、能窗、原始输入追加与无末尾换行；部分计算失败标记 PARTIAL 并返回错误；未收敛状态保留。
- ASE 二次势计算器验证单点不改坐标、FIRE 降低能量并固定指定原子，以及 charge/spin 传参。
- 检查点成功项复用、失败项重试、候选哈希不匹配拒绝；无效能量/坐标/状态拒绝；GPU worker 退出检测；CLI dry-run 不写输出、超界原子预检失败。

CREST 搜索和部分 Stage2 测试使用模拟结果以精确控制分支；ASE 优化测试使用可解析的二次势，真实 MACE 则由上述真实任务核验。`--help`、`--sample`、语法编译也已检查，生成模板与 `config_mlip_example.yaml` 一致。

## 使用时仍需注意的边界

- **不能把 1 ps 搜索当成充分采样**。本次故意缩短预算来验证流程；真实构象覆盖需要扩大预算并比较重复运行。
- **5 步优化未收敛**。`status: ok` 表示能量计算成功，收敛与否应读 `converged`；脚本当前会将未收敛末态参与排序。
- **仅 Stage1 输出的能量是原始候选能量**。恢复冻结坐标后没有重新算 GFN-FF；能量与输出几何不是严格重新求值的一对。原始一对一数据在 `crest_conformers_gfnff.xyz`。MACE 阶段则会在处理后的几何上重新求值，独立复算已验证一致性。
- **0.1 ps 是已复现的外部程序失败案例**。`real_sp/iso_321/stage1_gfnff.log` 显示空 `crest_dynamics.trj` 后 CREST `ERROR STOP`；脚本返回失败，没有误报成功。1 ps 的实际任务通过。没有人为添加通用最小时长限制，因为可用时长还取决于 CREST 的轨迹输出设置。
- **GFN-FF 与 MACE 绝对能量不可直接比较**。不同随机搜索的结果也不能用最低能量差简单衡量代码正确性。
- **未验证** CuEq 加速、其他 MACE/CREST 版本、多个输入结构同时运行的资源竞争、长时间 GPU 驱动故障或存活进程卡死。新增退出检测覆盖子进程死亡，未添加任意计算时限。
- 模型对该体系的精度、真实过渡态性质、频率和 IRC 均不在本次程序功能验证范围内。
- 修改脚本后旧输出 manifest 应拒绝续跑；请使用新目录，不要改哈希强行复用。`results_frozen/` 是历史结果，不能作为本次新脚本的续跑目录。

## 复现

在本机激活可用环境，确认配置中的绝对路径仍有效，再选新输出目录：

```bash
conda activate mace
python -m unittest -v test_crest_mace
python crest_mace.py validation/final_sp.yaml
python crest_mace.py validation/final_opt.yaml
python crest_mace.py validation/final_stage1.yaml
```

上述配置保持不变且脚本身份匹配时会直接续跑跳过。要从头再算，先复制配置并改为新的输出目录。独立核验可运行 `OMP_NUM_THREADS=2 python validation/verify_real.py`，需要模型和 GPU 0、1；它读取对应真实结果并生成 `verification.json`。

## 最终实测数值

| 模式 | 原始候选数 | 最终构象数 | 最低能量 / Eh | 最大冻结内部形变 / Å |
| --- | ---: | ---: | ---: | ---: |
| final_sp | 3 | 3 | -2955.12432198 | 0.026921 |
| final_opt | 4 | 3 | -2955.14159760 | 0.019097 |
| final_stage1 | 1 | 1 | -24.25680331 | 0.014403 |

双 GPU 两项能量与独立计算器的差均为 0.0 eV。单点和优化最低结构的独立复算误差分别为 8.47e-08 eV, 9.97e-09 eV，均低于 1e-5 eV；差异与 XYZ 舍入精度相符。上述三种模式来自独立随机搜索，绝对能量不用于互相比较优化增益。

## 仓库打包说明（2026-11-06）

本目录原始 YAML、日志及 manifest 保留验证时的绝对路径，以保留证据。迁移后请使用仓库 `examples/` 下的相对路径配置重新计算，输出到 `runs/`；不要直接运行本报告上文的历史绝对路径配置。根目录脚本与通过验证的版本逐字节相同。

打包后的 `verify_real.py` 按当前目录查找案例，新增 `--check-only`，无需 GPU 和模型；独立复算可通过 `--model` 指定当前模型，仍需要 GPU 0、1。历史 `results_frozen/` 未收录。
