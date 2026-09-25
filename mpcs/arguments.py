"""实验可调默认参数。

本文件只放实验输入的正式默认值；类型、合法性校验、序列化和
算法推导关系仍由 ``config.py`` 负责。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 路径与产物
DATASET_ROOT = PROJECT_ROOT / "dataset/Didichuxing/Chengdu/parcel_v2"  # 成都派生包裹目录。
GRAPH_PATH = (
    PROJECT_ROOT / "dataset/Didichuxing/Chengdu/roadnetwork/map_ChengDu"
)  # 成都正式 OSM 地图文件。
OUTPUT_ROOT = PROJECT_ROOT / "output/ppo_alternating_mc_1330_40ev_600p"  # 实验输出根目录。
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"  # 检查点目录。
MANIFEST_DIR = OUTPUT_ROOT / "manifests"  # 配置与结果清单目录。
TRAINING_LOG_PATH = OUTPUT_ROOT / "logs/training.jsonl"  # 训练 JSONL 日志路径。
TENSORBOARD_LOG_DIR = OUTPUT_ROOT / "tensorboard"  # TensorBoard 日志目录。

# 路网、区域与站点
SHORTEST_PATH_SOURCE_CACHE_SIZE = 512  # 最短路源点缓存上限。
# 旧项目 Station 网格分界点数；产生 (parts-1)^2 个格子。Region 是纯空间容器：
# 任务按坐标归属 Region，EV 初始放置按平台自身任务分布加权，之后全城工作。
# 真实数据统计（6000 记录/平台，0.415 inset）：parts=9 时每 Region 中位 pickup
# 25-32；parts=11 时 18-24；parts=13 时 16-17。默认保持 parts=11（10x10），
# 兼顾空间分辨率与每 Region 任务密度；EV 规模显著增大时可考虑 parts=13。
STATION_GRID_PARTS = 11
STATION_BOUNDS_INSET_RATIO = 0.415  # 订单范围两端各收缩比例；保留中心约 17% 服务域。
STATION_GRID_REFERENCE_SOURCE_FILE = "order_20161101"  # 固定用于生成网格的 train 参考源。
REGION_COUNT = (STATION_GRID_PARTS - 1) ** 2  # 全局区域数量，由旧网格推导。
REGION_GENERATION_METHOD = "legacy_station_grid"  # 区域生成方式。
REGION_BOUNDS = None  # 可选区域边界；预计算区域时保持为空。
STATION_COUNT = (STATION_GRID_PARTS - 1) ** 2  # 全局站点数量，由旧网格推导。
STATION_GENERATION_METHOD = "legacy_grid_midpoint"  # 站点生成方式。

# 数据集与需求采样
DATASET_NAME = "chengdu"  # 数据集名称。
DATASET_ADAPTER = "parcel_v2"  # 包裹数据适配器。
DATASET_SCHEMA_NAME = "didi_chengdu_parcel_v2"  # 输入包裹数据 schema。
TRAIN_SOURCE_FILES = tuple(f"order_201611{day:02d}" for day in range(1, 5))  # 训练集源文件。
VALIDATION_SOURCE_FILES = tuple(
    f"order_201611{day:02d}" for day in range(11, 15)
)  # 验证集源文件。
TEST_SOURCE_FILES = tuple(f"order_201611{day:02d}" for day in range(21, 25))  # 测试集源文件。
# 每个平台在各 split 内绑定一天，训练、验证和测试日期互不重叠。
PLATFORM_SOURCE_FILE_MAPPINGS = (
    (
        "P1",
        ("order_20161101",),
        ("order_20161111",),
        ("order_20161121",),
    ),
    (
        "P2",
        ("order_20161102",),
        ("order_20161112",),
        ("order_20161122",),
    ),
    (
        "P3",
        ("order_20161103",),
        ("order_20161113",),
        ("order_20161123",),
    ),
    (
        "P4",
        ("order_20161104",),
        ("order_20161114",),
        ("order_20161124",),
    ),
)  # 平台、训练、验证、测试源文件的显式映射。
SOURCE_CRS = "GCJ-02"  # 原始订单坐标系。
GRAPH_CRS = "EPSG:4326"  # 路网坐标系。
COORDINATE_TRANSFORM = "gcj02_to_wgs84"  # 订单到路网的坐标转换。
MAX_MAP_MATCH_DISTANCE_M = 1_000.0  # 地图匹配最大距离（米）。
PICKUP_COUNT_PER_PLATFORM = 500  # 每个平台抽取的取件订单数。
DROPOFF_COUNT_PER_PLATFORM = 100  # 每个平台抽取的送件订单数。
MAX_SOURCE_RECORDS = 6_000  # 通用候选池上限；平台配额路径按类型扫描整天源文件后采样。
ARRIVAL_WINDOW_START_S = 13 * 60 * 60  # 统一到达窗口起始秒数。
ARRIVAL_WINDOW_END_S = 13 * 60 * 60 + 30 * 60  # 统一到达窗口结束秒数（半开区间）。
DATASET_TIMEZONE = "Asia/Shanghai"  # 订单时间戳时区。

# 包裹与车辆
PICKUP_DEADLINE_POLICY = "fixed"  # 取件截止时间生成策略。
PICKUP_DEADLINE_MIN_S = 720  # 取件截止最小秒数。
PICKUP_DEADLINE_MAX_S = 720  # 取件截止最大秒数。
SYNTHETIC_PICKUP_FARE_AMOUNT = 15.0  # synthetic 取件收入。
SYNTHETIC_DROPOFF_FARE_AMOUNT = 0.0  # synthetic 送件收入。
FARE_NORMALIZATION_SCALE = 20.0  # 收益特征尺度 F_max；必须不小于最大单票 fare。
SYNTHETIC_CAPACITY_UNITS_PER_PARCEL = 1  # synthetic 包裹容量单位。
VEHICLES_PER_PLATFORM = 40  # 每个平台默认车辆数。
EV_SPEED_KM_PER_S = 0.02  # 车辆行驶速度（千米/秒）。
VEHICLE_CAPACITY = 30  # 单车容量单位上限。
SERVICE_RADIUS_KM = 4.0  # 车辆服务半径（千米）。
DROPOFF_LOAD_TARGET = 30  # 每次站点事务的送件装载容量单位上限。
# 车辆初始空间配置由各平台自身在各 Region 的任务数量分布决定
# （Region 是全局空间容器）；车辆随后可在全城工作。
EV_PLATFORM_OVERRIDES = ()  # 平台级完整车队覆盖。

# 路线、贪心与仿真
CANDIDATE_EV_LIMIT = None  # 候选车辆上限；None 为精确全量，正整数启用近似预筛选。
INSERTION_CANDIDATE_LIMIT = None  # 插入位置上限；None 为精确全量，正整数启用近似预筛选。
SHORTCUT_MODE = "balanced-shortcut-v1"  # 与 formal_cd 共用的路线候选策略。
SHORTCUT_CANDIDATE_EV_LIMIT = 64  # Shortcut 首轮 EV 上限。
SHORTCUT_RESCUE_EV_LIMIT = 128  # 首轮无精确项时的确定性救援上限。
MIN_LOCAL_NET_UTILITY_AMOUNT = 0.0  # 本地贪心接受的最小净效用。
RELEASE_DEADLINE_SLACK_THRESHOLD_S = 300  # 转交前所需最小截止富余秒数。
PLATFORM_NUM = 4  # 参与仿真的平台数量。
SIMULATION_START_TIME_S = ARRIVAL_WINDOW_START_S  # 仿真起始秒数。
SIMULATION_END_TIME_S = ARRIVAL_WINDOW_END_S  # 13:30 终止 episode 并结算剩余任务。
SIMULATION_STEP_SIZE_S = 20  # 全局逻辑时钟步长（秒）。
PROFIT_INTERVAL_S = 2 * 60  # 利润数据与曲线的固定记录间隔（秒）。
WAIT_MASK_ADVANCE_STEPS = 5  # WAIT 合法需至少再存活 N 个决策步，为截止前强制 RELEASE 预留窗口。

# 奖励与拍卖
AUCTION_MECHANISM = "paper_baseline"  # 拍卖机制；主流程使用论文反向 Vickrey 基线。
# 合同配置字段；实际 LOCAL/CROSS 结算成本采用 TRAVEL_COST_PER_KM × 额外距离。
ECONOMICS_COST_MODEL = "fare_ratio"  # 合同配置校验使用的模型标识。
EXECUTION_COST_RATIO = 0.30  # 合同配置校验使用的成本比例。
ORIGIN_MIN_MARGIN_RATIO = 0.10  # 原平台合作费的最低严格正利润比例。
SERVING_MIN_MARGIN_RATIO = 0.10  # 承接平台的最低严格正利润比例。
ECONOMICS_CONTRACT_VERSION = "reverse-vickrey-v1"  # 反向 Vickrey 净收益合同版本。
OFFER_BASE_RATIO = 0.20  # 模糊合作费 offer 的基础比例。
OFFER_QUALITY_WEIGHT_RATIO = 0.20  # 静态服务质量对 offer 的全局权重。
STATIC_SERVICE_QUALITY = 0.50  # 主模式全平台共享的静态服务质量。
AUCTION_TIE_POLICY = "seeded_rotation"  # 同价候选的可复现轮转规则。
TRAVEL_COST_PER_KM = 0.5  # 每千米行驶成本。
# PPO 每个物理批次使用排除 serving 的实际利润增量。
DROPOFF_OPERATIONAL_UTILITY_AMOUNT = 0.0  # 送件操作效用。
REWARD_NORMALIZATION_SCALE = 20.0  # 平台净利润奖励的归一化尺度。
BASIC_FARE_AMOUNT = 2.0  # 拍卖基础运费。
REVENUE_SHARING_RATIO = 0.7  # 中标平台的收入分成比例。
SHARING_RATE = 0.3  # 反向 Vickrey 报价中的收入分成比例 mu。
BASIC_PAYMENT_AMOUNT = 2.5  # 反向 Vickrey 报价的基础支付 f_b（调高以保证承接净收益非负）。
POTENTIAL_QUALITY_WEIGHT = 0.5  # 潜在服务质量权重。
HISTORICAL_QUALITY_WEIGHT = 0.5  # 历史服务质量权重。

# 隐私与局部观测
EPSILON_A = 1.0  # 出价差分隐私预算。
POTENTIAL_COMPONENT_SENSITIVITY = 1.0  # 潜在质量分量敏感度。
HISTORICAL_COMPONENT_SENSITIVITY = 1.0  # 历史质量分量敏感度。
LSH_HASH_COUNT = 8  # 位置 LSH 哈希次数。
LSH_BUCKET_WIDTH = 0.02  # 位置 LSH 桶宽。
LOCATION_LANDMARK_COUNT = REGION_COUNT  # 位置嵌入地标数，必须与全局 Region 数一致。
LOCATION_OBFUSCATION_TIME_THRESHOLD_S = 120  # 位置模糊化时间阈值（秒）。
DP_RNG_PROVIDER_NAME = "runtime-private"  # 运行时 DP 随机源名称。
LOCAL_FEATURE_DIM = 6  # 平台本地观测特征数。
RELEASE_HISTORY_DIM = 0  # RELEASE 历史仅用于审计和联邦标签。
RELEASE_HISTORY_WINDOW_SIZE = 100  # RELEASE 历史窗口长度。
FEDERATED_EMBEDDING_DIM = 1  # 联邦跨平台价值标量维度。

# PPO 自回归包裹策略与集中式价值网络
PPO_LOCAL_FEATURE_DIM = 6
PPO_FEDERATED_EMBEDDING_DIM = 1
PPO_BATCH_CONTEXT_DIM = 9
PPO_DECISION_CONTEXT_DIM = 8
PPO_CENTRAL_CONTEXT_DIM = 19
PPO_HIDDEN_DIMS = (64, 64)
PPO_GAMMA = 1.0  # 最大化平台完整 episode 利润，不按等待时长折扣。
PPO_GAE_LAMBDA = 1.0  # 完整物理时间回报，避免等待收益依赖远期 bootstrap。
PPO_INITIAL_ACTION_PROBABILITIES = (0.98, 0.019, 0.001)  # LOCAL / WAIT / RELEASE；合法掩码后重新归一化。
PPO_LEARNING_RATE = 0.0003
PPO_CRITIC_LEARNING_RATE = 0.001
PPO_CRITIC_UPDATE_EPOCHS = 8
PPO_CLIP_RATIO = 0.2
PPO_TARGET_KL = 0.02  # 多轮 rollout 更新期间限制行为策略漂移。
PPO_VALUE_LOSS_COEFFICIENT = 0.5
PPO_VALUE_NORMALIZATION_SCALE = 20.0  # critic 拟合平台剩余收益；GAE 恢复奖励原单位。
PPO_ENTROPY_COEFFICIENT = 0.0  # 已有随机采样探索，目标不额外奖励动作熵。
PPO_UPDATE_EPOCHS = 4
PPO_ROLLOUT_EPISODES = 5  # 同一行为策略采集多轮独立动作轨迹，再统一估计优势。
PPO_MINIBATCH_SIZE = 64  # 物理批次数；每批完整保留自回归包裹动作。
PPO_GRADIENT_CLIP_NORM = 0.5

# 独立 DDQN 模块参数
DDQN_HIDDEN_DIMS = (64, 64)  # DDQN 隐藏层维度。
DDQN_GAMMA = 1.0  # 负成本项（-p / -f）不允许被延迟折扣套利。
DDQN_LEARNING_RATE = 0.001  # DDQN 学习率。
DDQN_BATCH_SIZE = 64  # 每次 DDQN 更新均匀采样的已完成物理批次数。
REPLAY_CAPACITY = 100_000  # 每个平台保留的已完成物理批次数上限。
WARMUP_TRANSITIONS = 1_000  # 开始训练前至少积累的已完成物理批次数。
EPSILON_START = 1.0  # epsilon-greedy 初始探索率。
EPSILON_END = 0.05  # epsilon-greedy 最终探索率。
EPSILON_SCHEDULE_UNIT = "nonempty_batch"  # DDQN 按非空物理决策批次推进探索时钟。
EXPLORATION_DECAY_BATCHES = 35_000  # 探索衰减非空批次数（约 200 episode 到下限）。
EPSILON_DECAY_BATCHES = EXPLORATION_DECAY_BATCHES
TARGET_SYNC_INTERVAL_STEPS = 500  # 目标网络同步间隔步数。
GRADIENT_CLIP_NORM = 5.0  # 梯度裁剪范数上限。
MAX_REPLAY_VERSION_AGE = 2  # 可训练回放的联邦版本最大滞后。

# 联邦学习
FEDERATED_ENABLED = True  # 是否启用联邦学习。
SECURE_AGGREGATION_ENABLED = False  # 是否启用安全聚合；论文主流程默认使用清文 FedAvg。
PARTICIPANT_PLATFORM_IDS = ("P1", "P2", "P3", "P4")  # 联邦参与平台 ID。
ROUND_INTERVAL_EPISODES = 10  # 联邦轮次间隔 episode 数。
LOCAL_EPOCHS = 1  # 每轮联邦本地训练 epoch 数。
SHARED_INPUT_DIM = 4  # 联邦共享模型输入维度；粗粒度区域、时间、收益和紧迫度。
SHARED_HIDDEN_DIMS = (64,)  # 联邦共享模型隐藏层维度。
SHARED_EMBEDDING_DIM = 1  # 联邦共享模型输出的跨平台价值标量维度。
AUTHORIZED_TIME_BUCKET_SIZE_S = 300  # 跨平台授权时间桶大小（秒）。
FEDERATED_LEARNING_RATE = 0.001  # 联邦共享模型学习率。
FUSION_BETA = 1.0  # 联邦知识融合系数 beta。
FUSION_DELTA = 2.0  # 联邦知识融合系数 delta。
SECURE_MIN_CONTRIBUTORS = 2  # 安全聚合最少贡献平台数。
SECURE_QUANTIZATION_SCALE = 1_000_000_000  # 安全聚合定点量化尺度。
PAIRWISE_MASK_PROVIDER_NAME = "runtime-client-private"  # 运行时成对掩码提供器名称。

# 训练与终端进度
MASTER_SEED = 20260717  # 总随机种子。
RANDOM_MODE = "research_reproducible"  # 运行时随机源；研究实验默认可复现。
FLTA_MODE = "region_non_dp_fl_clear"  # 主流程：区域公开、出价无 DP、清文联邦聚合。
TOTAL_EPISODES = 500  # 总训练 episode 数。
MAX_STEPS_PER_EPISODE = None  # 每个 episode 的可选最大步数。
CHECKPOINT_INTERVAL_EPISODES = 50  # 检查点保存间隔。
EVALUATION_INTERVAL_EPISODES = 50  # 验证评估间隔。
EVALUATION_SEED_INDICES = (0, 1, 2)  # 每个 checkpoint 共用的固定验证场景面板。
TRAINING_DEVICE = "cuda"  # 训练设备。
TENSORBOARD_ENABLED = True  # 是否记录 TensorBoard 指标。
TENSORBOARD_FLUSH_SECS = 10  # TensorBoard 刷盘间隔（秒）。
PLOT_ENABLED = True  # 是否输出训练/评估 CSV 与 PNG 曲线。
PLOT_SMOOTHING_WINDOW = 10  # 曲线移动平均和真实波动阴影窗口。
CPU_WORKERS = 4  # 数据与训练 CPU 工作线程数。
TORCH_CPU_THREADS = 1  # Torch CPU 计算线程数。
TORCH_INTEROP_THREADS = 1  # Torch 算子间并行线程数。
PROGRESS_ENABLED = True  # 是否显示终端进度。
PROGRESS_MIN_INTERVAL_S = 0.2  # 进度刷新最小间隔（秒）。
