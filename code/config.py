"""Shared constants for the fixed experiment and local output builders.
固定实验与本地结果生成脚本共用的常量。

The executable analysis stages are the ``modal_*.py`` files. Some stage-specific
task lists remain beside their Modal entry points, so this file is not a command
and is not the sole source of every parameter.
可执行分析阶段均为 ``modal_*.py`` 文件。部分阶段专用任务列表仍放在相应的 Modal
入口旁，因此本文件不是运行入口，也不是所有参数的唯一来源。
"""

from pathlib import Path

# Paths / 路径
CODE_DIR = Path(__file__).resolve().parent
PKG_DIR = CODE_DIR.parent
RESULTS_DIR = PKG_DIR / "results"
PKL_DIR = PKG_DIR / "lake_pkls"

# Study systems / 研究水系
LAKES = [
    # Okanagan system / 奥肯那根水系
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    # Nelson–Winnipeg system / 尼尔森—温尼伯水系
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]

VARIABLES = ["WL", "T", "P", "R", "SWE", "Evap", "RegFlow"]

# Seven directly connected lake pairs used for connectivity groups. / 用于连通性分组的 7 对直接水道连接湖泊。
WATERWAY_CONNECTED_PAIRS = [
    ("Kalamalka_Lake", "Okanagan_Lake"),
    ("Okanagan_Lake", "Skaha_Lake"),
    ("Skaha_Lake", "Vaseux_Lake"),
    ("Rainy_Lake", "Lake_of_the_Woods"),
    ("Playgreen_Lake", "Sipiwesk_Lake"),
    ("Kiskitto_Lake", "Sipiwesk_Lake"),
    ("Sipiwesk_Lake", "Split_Lake"),
]

# Train/test split / 训练与测试切分
FORECAST_HORIZON = 37          # Approximate 9:1 split; CCM uses training data only. / 约按 9:1 切分；CCM 只使用训练期。
ROLLING_HORIZONS = [1, 3, 6, 12]

# Embedding selection uses fixed tau and univariate Simplex-selected E. / 嵌入参数使用固定 tau，并由单变量 Simplex 选择 E。
EMBED_TAU = 1
EMBED_E_CANDIDATES = range(2, 11)   # E candidates 2–10. / E 候选值为 2–10。

# Statistical tests / 统计检验
N_SURROGATES = 500
FDR_ALPHA = 0.05

# Missing-data rule / 缺测规则
MAX_FILLABLE_GAP_MONTHS = 6    # Exclude variables with gaps longer than six months. / 连续缺测超过 6 个月则排除变量。
OUTLIER_Z_THRESH = 6
OUTLIER_LEVEL_Z_THRESH = 5
