# Multi-Platform Crowdsourcing Simulator (MPCS)

MPCS 是多平台空间众包仿真环境，内置电动车包裹取送场景。环境在同一物理帧收集全部平台动作，再统一推进任务分配、跨平台匹配、路线、结算和指标。数据准备、算法和实验流程各有独立接口。

## 快速上手

需要 Python 3.11 或更高版本。在项目根目录运行：

```powershell
python -m pip install -e ".[dev]"
python -m mpcs datasets
python -m mpcs algorithms
python -m mpcs mechanisms
python -m mpcs mixed --scenario examples/mixed-four-platform.json `
  --output output/first-mixed
```

根目录的 `setup.py` 是安装兼容入口；依赖、包发现和 `mpcs` 命令由 `pyproject.toml` 管理。synthetic 场景不需要外部数据。完成后查看 `output/first-mixed/summary.json`、`training/episodes.csv`、`validation/summary.json` 和 `comparison/mixed/metrics.png`。

## 配置四平台混合训练

这条命令让 P1 独自学习 PPO，P2/P3/P4 分别用 RL-CAPA、MRA、IMPGTA 的分池策略选择任务去向。所有平台的本地池统一使用 KM 匹配：

```powershell
python -m mpcs mixed --dataset synthetic --platforms 4 --learner P1 `
  --platform-policy P2=rl-capa --platform-policy P3=mra `
  --platform-policy P4=impgta --local-matcher km `
  --cross-mechanism regional-fixed `
  --episodes 20 --seed 11 `
  --output output/my-mixed --tensorboard
```

[场景 JSON 示例](examples/mixed-four-platform.json)保存了相同阵容和匹配规则，可通过 `--scenario` 重复运行。命令行的学习者、平台策略、本地匹配器、跨平台机制、平台数、轮数、种子和数据集覆盖 JSON 中的对应值。每个非学习平台都要指定策略。synthetic 支持 `--platforms 2` 到 `--platforms 16`；真实数据集的平台数由预设或完整配置决定。训练每轮仅更新指定 PPO 平台；验证和测试沿用相同阵容，PPO 使用确定性推理。

混合场景有两个算法选择阶段：每个平台的策略对待处理任务选择 `LOCAL`（进入本地池）、`RELEASE`（进入跨平台池）或 `WAIT`（留待后续帧）；环境随后用一个全局选定的 `local_matcher` 匹配各平台本地池，并用一个全局 `cross_mechanism` 处理跨平台池。`local_matcher` 可选 `greedy`（默认，同一辆车一帧可连续接多个任务）或 `km`（每辆车一帧最多接一个新任务，先最大化匹配数，再最小化新增路线距离）。跨平台机制可选 `paper`、`regional-fixed`、`pool-random`，也可注册插件。混合场景中的 baseline 名称只代表其**分池策略**；`run` 单独比较 baseline 时保留其完整参考算法。混合指标代表整场阵容，`profit_by_platform` 给出各平台收益。

## 接入自己的算法

分池算法只需为当前平台每个待处理任务返回 `LOCAL`、`RELEASE` 或 `WAIT`。MPCS 负责本地匹配和跨平台服务。无状态策略可实现一个决策函数；可运行的插件见 [examples/custom_policy.py](examples/custom_policy.py)：

```python
from mpcs.core.Domain import ParcelAction

def local_first(config, platform_id, observation):
    return {
        pickup.parcel_id: ParcelAction.LOCAL
        for pickup in observation.waiting_pickups
    }

def register(runner):
    runner.register_policy("local-first", local_first)
```

从项目根目录加载插件，并把它放入 P4：

```powershell
python -m mpcs mixed --scenario examples/mixed-four-platform.json `
  --plugin examples.custom_policy --platform-policy P4=local-first `
  --output output/custom-policy
```

同一个函数策略也可用 `run --methods local-first localsum` 比较。需要保存策略状态时，用 `runner.register_pool_policy(name, factory)` 注册分池策略；跨平台规则用 `runner.register_cross_mechanism(name, factory)` 注册。两处扩展接口及其参数见 [扩展指南](docs/extending.md)。

## 数据集和其他运行方式

内置预设：`synthetic`、`chengdu`、`shanghai`、`shanghai16`。原项目的 Chengdu、Shanghai、New York 数据和地图已复制到本地 `dataset/`，不提交 Git。Chengdu 支持本地 parcel-v2 数据；内置 Shanghai 预设只有测试任务，训练需自定义场景提供器。新克隆仓库可立即运行 synthetic。

成都原始订单可用 [DataUtils 预处理工具](mpcs/utils/DataUtils.py) 转换为仿真所需的 parcel-v2 文件：

```powershell
python -m mpcs.utils.DataUtils `
  --source-root dataset/Didichuxing/Chengdu/dataset `
  --output-root dataset/Didichuxing/Chengdu/parcel_v2 `
  --seed 20250308
```

成都混合训练可用 [按平台分配日期的场景](examples/chengdu-days.json)：

```powershell
python -m mpcs mixed --scenario examples/chengdu-days.json `
  --output output/chengdu-days
```

`platform_days` 对每个平台分别指定 `train`、`validation`、`test`，值可为 `YYYYMMDD` 字符串或日期列表。例如 `"P1": {"train": ["20161105", "20161109"], "validation": "20161111", "test": "20161121"}`。所有平台都需指定三个 split；同一天只能属于一个平台的一个 split，重复分配在读取数据前报错。`MPCSRunner.run_mixed(platform_days=...)` 接受相同映射。示例使用 `dataset/Didichuxing/Chengdu/parcel_v2/` 下的本地文件；新克隆仓库需先准备数据和地图。

```powershell
python -m mpcs run --dataset chengdu --split test `
  --methods localsum mra --output output/chengdu
python -m mpcs pipeline --dataset synthetic --episodes 20 --output output/all-methods
python -m mpcs sweep --dataset synthetic --methods localsum mra `
  --seeds 11 29 --max-workers 2 --output output/sweep
```

`run` 使用内置 baseline 的完整参考实现比较选定算法；`run --local-matcher` 仅在同时提供 `--ppo-checkpoint` 时设置 PPO 的本地匹配器。`pipeline` 训练独立 PPO，并与五种 baseline 在测试集比较；`sweep` 并行比较多个种子；`train-ppo` 只训练独立 PPO，默认按轮次轮换学习平台。交互终端用一个动态 Rich 面板展示数据读取、图与区域构建、任务和车队准备、环境构建、训练、验证与比较；完成的阶段保留耗时和结果，例如图的 `nodes`、`routing_edges`、任务数与车辆数。物理帧在面板内刷新，非交互终端仅输出阶段完成摘要。`--no-progress` 关闭阶段显示，`--tensorboard` 生成 TensorBoard 事件，JSONL、CSV 和图表仍会输出。

自定义数据格式用插件注册 `config_factory(output_dir)` 和 `scenario_provider(config, split)`。提供器解析数据与地图后调用 `mpcs.data.prepare_scenario`，传入路网、区域、站点、各平台任务和初始车辆。完整实验配置可用 `--config path/to/config.json` 指定；混合阵容 JSON 用 `--scenario` 指定。详见 [扩展指南](docs/extending.md)。

## 项目架构

| 模块 | 职责 |
| --- | --- |
| `mpcs/config.py` | 类型化实验配置、平台数、训练参数和路径 |
| `mpcs/utils/DataUtils.py`、`mpcs/data/Adapters.py` | 原始订单预处理、内置数据准备和外部场景构建 |
| `mpcs/core/Domain.py`、`Framework.py` | 观测与动作协议、全局状态、同步物理帧和结算 |
| `mpcs/core/GraphUtils.py`、`TaskUtils.py`、`LocalMatching.py` 等 | 路网、任务、统一本地匹配、路线及状态推进 |
| `mpcs/utils/Economics.py`、`Performance.py` | 通用经济计算和运行时间工具 |
| `mpcs/algorithms/baseline/` | 五种 baseline，各在独立文件中 |
| `mpcs/algorithms/PPOTraining.py` | 独立与混合 PPO 训练、验证和检查点 |
| `mpcs/experiments/Runner.py` | 算法注册、同场景比较和并行 sweep |
| `mpcs/experiments/Workflow.py` | 数据集注册、训练流程和混合阵容 |
| `mpcs/cli.py` | `mixed`、`pipeline`、`run`、`sweep` 等命令 |

场景准备产生可复用初始状态；各算法在隔离的路网运行时副本中运行。混合训练时，各平台策略只提供分池动作；环境统一处理本地匹配、跨平台匹配、状态推进与结算。独立 baseline 比较保留原算法的完整实现。模块边界见 [架构文档](docs/architecture.md)。

## 输出和本地开发

混合训练生成 `training/`、`validation/`、`comparison/mixed/` 和根目录 `summary.json`。训练包含 `episodes.jsonl`、`episodes.csv`、`training.png`、`checkpoints/`；比较包含事件日志、逐帧进度、CSV、图表和摘要。`mpcs/utils/DataUtils.py` 纳入 Git；原始数据、转换结果、`tests/`、`output/`、缓存和检查点均被忽略。
