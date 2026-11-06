# π0.5 四任务阶段 MoE 全参数微调

配置：`pi05_maniskill_phase_moe_full`，JAX / Flax。基于本 fork 的 RECAP
版本加入四个残差 FFN 专家，沿用四任务 `pd_joint_pos` 数据、两路图像、10 步动作块、
前七个关节 delta + 绝对夹爪目标及 mean/std 归一化。

**这是全参数微调。** 视觉编码器、语言模型、原 action expert、输出头、路由器和新增专家
均可训练；`freeze_filter=nnx.Nothing`，未使用 LoRA 冻结配方。
新增模块采用低维瓶颈，但不意味着只训练新增模块。

## 结构与训练约定

- 原 action expert 的 18 层 FFN 保持原权重路径，每层并联四个 `1024 → 16 → 1024`
  SiLU 残差专家。新增输出矩阵为零，输入矩阵随机初始化；加载 dense 权重时新增分支不改变输出。
- 一个共享路由器，输入为带 mask 的图像/语言 **prefix embeddings（LLM 前）** 的均值和
  归一化机器人状态。两者分别 LayerNorm，再经 128 维 MLP 输出四个概率。
  未额外执行一遍 VLM，也不把标签、未来动作或 flow 时间输入路由器。
- 专家和路由器合计新增 2,630,340 个参数，完整模型参数量为 3,356,064,212。
- 一次观测只计算一次路由。所有 action FFN 层、动作 token 和该动作块的所有去噪步
  共用这一组权重；各层专家参数独立。下一次观测可以改变路由。
- 默认四专家 soft routing。`--model.action-moe.top-k 2` 可开启 Top-2，训练和服务必须一致。
  当前实现仍计算所有小专家，再加权，所以 Top-2 不代表已经实现稀疏算力加速。
- 损失为原 flow-matching loss + 阶段弱监督 CE + 很小的防闲置约束。
  CE 权重前 1,000 步为零，1,000–2,000 步线性升至 0.02；防闲置权重 0.001，
  只惩罚平均概率低于 2.5% 的专家，不强求四专家各占 25%。
- 默认 batch 8、30,000 步、AdamW、梯度裁剪 1.0、EMA 0.999；学习率
  1,000 步 warmup 至 `2e-5`，30,000 步衰减至 `2e-6`。支持 FSDP。
- `MoECheckpointWeightLoader` 仅允许新专家和路由器缺失，原模型缺失参数仍会使初始化失败。
  从 MoE checkpoint 启动会加载已有专家；断点续训使用 `--resume` 恢复优化器、EMA 和步数。
- 原 `pi05_maniskill` 配置默认关闭 MoE。此实现不支持 PyTorch，相关入口会明确报错。

## 弱标签

脚本读取 LeRobot parquet 中的 `state=qpos(9)+qvel(9)+tcp_pose(7)`、8 维绝对动作、
episode/frame 索引和 FPS，生成独立 NPZ 及 JSON 时间线，不改动数据集。
Panda 夹爪命令必须是 `-1` 关闭、`+1` 打开。

| 编号 | 弱阶段 | 主要依据 |
| --- | --- | --- |
| 0 | 接近 | 开爪且不在抓取前邻域或刚释放的窗口 |
| 1 | 对准 / 抓取 | 闭爪前 TCP 距离闭爪位置 4 cm 内的连续片段，以及闭爪后的短暂稳定窗口 |
| 2 | 约束操作 | 持续闭爪；结合 TCP 平移、旋转和实测开口变化调节置信度 |
| 3 | 释放 / 回撤 | 闭爪后重新开爪的短窗口，遇下一次闭爪截断 |

这些是**动作阶段的启发式代理**，不能证明接触、抓取成功或任务完成。
特别是阀门的重抓、静止等待和接触失败，需要结合 JSON 时间线抽查视频。
跨阶段动作块使用置信度加权的四类软分布；不跨 episode，也不把末尾 padding 当成证据。
标签可以利用示范的未来片段；它们仅用于 loss，推理不需要标签文件。

NPZ 绑定 repo ID、动作长度、FPS、完整 state/actions 和索引的 SHA-256。
加载时会核对原始数据，防止重建数据后误用旧标签。更改数据、动作长度或规则后须重新生成。
同一实验续训应保留同一份标签及 JSON，不中途改规则。

本地 `luna_cleaned_20260908` 初次规则运行：217 episodes、48,271 帧；接近 16,011、
对准/抓取 5,585、约束操作 25,592、释放/回撤 1,083。最后一类只有约 2.2%，
暂不能据此宣称四个专家已学到均衡或跨任务的技能分工。

## 在已部署的 A800 上启动

先将 `third_party/openpi` 切到 fork 的 `feat/pi05-phase-moe-full` 分支：

```bash
cd ~/maniskill_myws
git -C third_party/openpi fetch https://github.com/bitqjy/openpi.git feat/pi05-phase-moe-full
git -C third_party/openpi switch --detach FETCH_HEAD
```

沿用已部署的 Python/JAX/CUDA 环境，不必更新依赖或重新转换数据。
在主项目目录操作：

```bash
cd ~/maniskill_myws
source env_a800.sh
export MANISKILL_PROJECT_ROOT="$PWD"
export PI05_BASE_PARAMS=/mnt/data/base_model/pi05_base/params

# 检查最终命令；此操作不会训练或生成标签。
bash third_party/openpi/examples/maniskill/train_phase_moe.sh --dry-run

# 选空闲 GPU，先跑独立的两步验收，避免覆盖正式实验。
CUDA_VISIBLE_DEVICES=4 EXP_NAME=luna_phase_moe_smoke NUM_TRAIN_STEPS=2 \
  bash third_party/openpi/examples/maniskill/train_phase_moe.sh

# 正式四任务全参数微调。
CUDA_VISIBLE_DEVICES=4 \
  bash third_party/openpi/examples/maniskill/train_phase_moe.sh

# 同一实验断点续训。
CUDA_VISIBLE_DEVICES=4 \
  bash third_party/openpi/examples/maniskill/train_phase_moe.sh --resume
```

启动器自动生成缺失的弱标签，并复用已验收的
`assets_openpi/pi05_maniskill/local/luna_cleaned_20260908/norm_stats.json`。
它只复用归一化资产；训练配置仍是新的 MoE 配置。checkpoint 会保存实际使用的归一化统计。
复用成立的前提是数据及动作变换与 dense 训练相同。

多卡示例：`CUDA_VISIBLE_DEVICES=4,5 FSDP_DEVICES=2 BATCH_SIZE=8 bash .../train_phase_moe.sh`。
batch 为全局值。`EXP_NAME`、`NUM_TRAIN_STEPS`、`NUM_WORKERS`、`CHECKPOINT_BASE`、
`NORM_ASSETS_DIR`、`PHASE_LABELS` 均可通过环境变量覆盖。其余参数直接追加给 OpenPI CLI，
例如 `--wandb-enabled`；启动器默认关闭 W&B，也不会默认覆盖实验。

两步验收只确认执行链路，尚未进入阶段监督的 warmup 后区间。正式训练日志包含
`flow_loss`、`moe/phase_loss`、`moe/phase_ramp`、`moe/phase_accuracy`、
`moe/label_confidence`、`moe/router_entropy` 和四个 `moe/expert_*_prob`。

## 单独生成标签或在独立 OpenPI checkout 中运行

```bash
cd /path/to/openpi
export HF_LEROBOT_HOME=/path/to/maniskill_myws/dataset/lerobot
.venv/bin/python scripts/build_maniskill_phase_labels.py \
  --dataset-dir "$HF_LEROBOT_HOME/local/luna_cleaned_20260908" \
  --output assets/phase_labels/luna_cleaned_20260908_h10.npz

# 没有现成统计时，按此配置重新计算；已有经过验收的相同数据统计可复用。
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_maniskill_phase_moe_full
.venv/bin/python scripts/train.py pi05_maniskill_phase_moe_full \
  --exp-name luna_phase_moe_full \
  --weight-loader.params-path /mnt/data/base_model/pi05_base/params
```

推理必须选 `pi05_maniskill_phase_moe_full`，并保持训练时的 rank / Top-k 等结构设置。
普通动作推理、action latent、temporal latent、SAFE pre-velocity 入口均复用同一套 MoE 路由。
不要用 dense 配置加载 MoE checkpoint，否则其额外权重可能被过滤。

## 验证与效果边界

2026-11-06 本地针对性回归 **34 项通过**，包含新增的 16 项 float32 / bfloat16
与弱标签测试，以及原配置、LoRA、数据变换回归。真实四任务 batch 加载也已通过。
新增 CPU 测试使用缩小的 Gemma 和测试视觉编码器，覆盖：原参数加载后的输出一致性、
四专家及原模型梯度、推理不读取阶段标签、各推理入口一致性、原参数缺失检测、
阶段软标签边界和指纹检查，以及 JIT 更新、checkpoint 保存/恢复和续训。
真实数据加载另外检查归一化后的 `(B,32)` 状态、`(B,10,32)` 动作和 `(B,4)` 阶段目标。

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  src/openpi/models/action_moe_test.py src/openpi/training/phase_labels_test.py
```

2026-11-06 已在单张 **A800-SXM4-80GB** 上完成完整模型验收，代码提交
`34d763285145e3367a9ac1726aad9071397d7b96`，batch 8、2 个数据 worker、FSDP 1：

- 真实四任务数据完成两步全参数更新，再用 `--resume` 恢复模型、优化器、EMA 和步数，
  从 train step 2 继续更新到 3。三步 loss 和梯度范数均有限。
- 验收专用 CLI 覆盖 `--model.action-moe.phase-warmup-steps 0`
  `--model.action-moe.phase-ramp-steps 1 --log-interval 1`，确认第 1 步阶段监督实际启用：
  flow loss 0.3129、phase loss 1.3918、总 loss 0.3408。正式配置的 1,000 步 warmup
  和 1,000 步 ramp 保持不变；不要从这个短验收实验续跑正式训练。
- 从最终 checkpoint 的 EMA 参数重新加载，每个任务选一条真实观测，均能在
  **不加载阶段标签、不提供未来动作**时输出有限的 `(10, 8)` 绝对关节/夹爪目标。
  四任务的关节 delta、mean/std 归一化及逆变换检查均通过。
- 18 层四专家输出权重形状为 `(18, 4, 16, 1024)`；四个专家均已从全零初始化更新。
  此时 EMA 输出权重范数仍仅约 `6.5e-8–7.1e-8`，不能据此宣称已学到动作分工。
- `nvidia-smi` 每 5 秒采样的训练/续训显存峰值为 **74,825 MiB（约 73.1 GiB）**。
  这包含 JAX 90% 显存预分配，不能作为新增模块实际显存开销的测量。
  四条观测的同步重复推理耗时约 82–104 ms；含首次编译的首条推理约 33.5 秒。
  这些是短验收数据，尚未测量长期训练吞吐或与 dense 的等条件性能差异。

主项目中的验收脚本、日志和报告位于 `outputs/moe_a800_validation_20261106/`。
最终 checkpoint 位于
`checkpoints_openpi/pi05_maniskill_phase_moe_full/luna_phase_moe_accept_20261106/2`；
目录编号为保存时的循环索引，内部 train step 为 3。
服务器源数据指纹与本地审计一致，仍为 217 episodes、48,271 帧。

零初始化保持初始动作函数，不保证全参数训练后的成功率。
最终应按 episode/seed 留出验证，分别评估四任务成功率、动作连续性、延迟，
并通过强制切换/屏蔽专家验证动作分工。路由分类准确率本身不等于技能已经形成。
