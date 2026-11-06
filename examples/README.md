# 可迁移的示例配置

从仓库根目录运行 `python crest_mace.py examples/config_iso321_stage1.yaml`。三个配置均默认 `dry_run: true`；预检通过后改为 `false` 开始计算。

| 配置 | 功能 | 依赖 |
| --- | --- | --- |
| `config_iso321_stage1.yaml` | 仅 GFN-FF 采样 | CREST、NumPy、PyYAML |
| `config_iso321_sp.yaml` | GFN-FF → MACE 单点 | 另需模型、MACE、CUDA GPU 0 |
| `config_iso321_opt.yaml` | GFN-FF → MACE 受约束优化，最多 5 步 | 同上 |

输入固定为 `../input/iso_321.xyz` 所在目录，输出各写入 `../runs/` 下的新目录。使用电荷 0、单重态、冻结原子 `[1,2,3,5]`；这些设置仅适用于本案例。所有相对路径以 YAML 所在目录为基准。

这些是短时功能检查配置：`mquick`、每条 MTD 1 ps。优化 5 步通常未收敛，正式计算需调整采样深度、候选数量和优化步数。
