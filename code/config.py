"""数据生成阶段使用的共享配置。

分析脚本仍保留部分自己的湖泊列表和滞后范围；修改本文件不会自动同步到
所有脚本。验证脚本会检查这些配置是否仍然一致。
"""

from pathlib import Path

# 路径
CODE_DIR = Path(__file__).resolve().parent
PKG_DIR = CODE_DIR.parent
RESULTS_DIR = PKG_DIR / "results"
PKL_DIR = PKG_DIR / "lake_pkls"

# 研究系统
LAKES = [
    # 奥肯那根水系
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    # 尼尔森—温尼伯水系
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]

VARIABLES = ["WL", "T", "P", "R", "SWE", "Evap", "RegFlow"]

# 7 对直接水道连接的湖泊，用于连通性分组。
WATERWAY_CONNECTED_PAIRS = [
    ("Kalamalka_Lake", "Okanagan_Lake"),
    ("Okanagan_Lake", "Skaha_Lake"),
    ("Skaha_Lake", "Vaseux_Lake"),
    ("Rainy_Lake", "Lake_of_the_Woods"),
    ("Playgreen_Lake", "Sipiwesk_Lake"),
    ("Kiskitto_Lake", "Sipiwesk_Lake"),
    ("Sipiwesk_Lake", "Split_Lake"),
]

# 训练集与测试集
FORECAST_HORIZON = 37          # 约按 9:1 切分；CCM 也只使用训练期。
ROLLING_HORIZONS = [1, 3, 6, 12]

# 嵌入参数
# tau 固定为 1，以避免测试期信息进入参数选择，并缩短缺测序列的嵌入跨度。
# E 仍由单变量 simplex 自预测选择，不用待检验的 CCM 技巧选择。
EMBED_TAU = 1
EMBED_E_CANDIDATES = range(2, 11)   # E 取 2 到 10。

# 统计检验
N_SURROGATES = 500
FDR_ALPHA = 0.05

# 缺测处理
MAX_FILLABLE_GAP_MONTHS = 6    # 连续缺测超过 6 个月则排除变量。
OUTLIER_Z_THRESH = 6
OUTLIER_LEVEL_Z_THRESH = 5
