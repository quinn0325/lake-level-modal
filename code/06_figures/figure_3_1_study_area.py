"""图 3.1 研究区域图：加拿大定位图 + 两个水系的放大面板。

设计说明
--------
两个水系相距约 2000 km（BC 与 Manitoba/Ontario），画在同一比例尺的单幅地图上，
两簇湖泊各只占几个像素。因此采用「定位图 + 分区放大」的四面板布局，这是多站点
分散的水文研究的常规做法：

    (a) 加拿大定位图，标出三个研究区的范围框
    (b) Okanagan 链（4 湖）
    (c) Rainy Lake 与 Lake of the Woods
    (d) Nelson 河链（4 湖）

Nelson–Winnipeg 系统内部也要拆开：Rainy/Lake of the Woods 位于北纬 49°，
Nelson 河链位于 54–56°，相距约 700 km，合画一个面板会重蹈"两头都看不清"的覆辙。

湖泊用 HydroLAKES 的真实多边形绘制，而非点标记——Okanagan 四湖的串联形态本身
就传达了水系结构。投影统一用 EPSG:3347（Canada Atlas Lambert）；不要用等经纬
（PlateCarree），在北纬 50–56° 上经度方向会被拉伸约 40%，湖形明显变扁。

水位站与调控站用不同符号区分。这一区分不是装饰：Lake of the Woods 由 4 个水位站
监测（故需多站合成，见 §3.2.2），Playgreen 与 Kiskitto 共用同一调控站（Jenpeg
大坝 05UB009），Sipiwesk 自身无坝、其调控站位于下游（Kelsey 发电站 05UE005）。
这些在图上一眼可见，正文因而不必逐条罗列。

依赖
----
    geopandas, shapely, matplotlib, pyogrio    （不需要 cartopy）

外部数据（路径见下方常量，需按本机情况调整）
    HydroLAKES_polys_v10.shp                   湖泊多边形
    ne_50m_admin_0_countries.geojson           国界（Natural Earth）
    ne_50m_admin_1_states_provinces_lines.geojson  省界（可选）

跑法
----
    python code/06_figures/figure_3_1_study_area.py

输出
----
    results/figures/figure_3_1_study_area.pdf   （矢量，投稿用）
    results/figures/figure_3_1_study_area.png   （300 dpi，预览用）
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely.geometry import Point, box
from shapely.ops import nearest_points

# --------------------------------------------------------------------------
# 路径与常量
# --------------------------------------------------------------------------
# 这张图需要两份本地下载的地理数据，体积过大不随仓库分发（HydroLAKES 约 1 GB）。
# 用环境变量指向你自己的下载位置：
#
#   HydroLAKES v1.0 多边形     https://www.hydrosheds.org/products/hydrolakes
#     export CCM_HYDROLAKES_SHP=/path/to/HydroLAKES_polys_v10.shp
#
#   Natural Earth 1:50m 国界与省界已裁好随仓库分发，无需下载；如要用完整版：
#     https://www.naturalearthdata.com/downloads/50m-cultural-vectors/
#     export CCM_NATURALEARTH_DIR=/path/to/dir/holding/the/two/geojson
#
# 首次运行会把这十个湖裁切出来缓存到 results/figures/_cache_study_lakes.gpkg，
# 之后重画就不再需要原始文件。仓库里已带该缓存，因此只有想从头重建时才需要它们。
HYDROLAKES = Path(os.environ.get(
    "CCM_HYDROLAKES_SHP", "HydroLAKES_polys_v10.shp"))
# Natural Earth 的两层已裁到北美范围随仓库分发（合计约 670 KB），
# 因此默认不需要任何下载；设了环境变量就用你自己的完整版本。
ROOT = Path(__file__).resolve().parents[2]
_NE_BUNDLED = ROOT / "figures"
if not _NE_BUNDLED.exists():
    _NE_BUNDLED = ROOT / "reference" / "figures"
NE_DIR = Path(os.environ.get("CCM_NATURALEARTH_DIR", _NE_BUNDLED))
NE_COUNTRIES = NE_DIR / "ne_50m_admin_0_countries.geojson"
NE_PROVINCES = NE_DIR / "ne_50m_admin_1_states_provinces_lines.geojson"

OUT_DIR = ROOT / "results" / "figures"
# 裁切好的十个湖泊多边形随仓库分发，因此没有 HydroLAKES 原始文件也能重画此图。
CACHE = ROOT / "figures" / "_cache_study_lakes.gpkg"
if not CACHE.exists():
    CACHE = ROOT / "reference" / "figures" / "_cache_study_lakes.gpkg"
if not CACHE.exists():                            # 没有就退回到旧位置并现场生成
    CACHE = OUT_DIR / "_cache_study_lakes.gpkg"

CRS = 3347          # Canada Atlas Lambert
FIG_WIDTH_CM = 16.0  # 论文正文宽

# 湖泊：名称 → (HydroLAKES id, 面板, 图上标签)
LAKES = {
    "Kalamalka_Lake":    (7878,   "okanagan", "Kalamalka"),
    "Okanagan_Lake":     (695,    "okanagan", "Okanagan"),
    "Skaha_Lake":        (8117,   "okanagan", "Skaha"),
    "Vaseux_Lake":       (101700, "okanagan", "Vaseux"),
    "Rainy_Lake":        (710,    "woods",    "Rainy"),
    "Lake_of_the_Woods": (57,     "woods",    "Lake of\nthe Woods"),
    "Playgreen_Lake":    (578,    "nelson",   "Playgreen"),
    "Kiskitto_Lake":     (564,    "nelson",   "Kiskitto"),
    "Sipiwesk_Lake":     (520,    "nelson",   "Sipiwesk"),
    "Split_Lake":        (479,    "nelson",   "Split"),
}

WL_STATIONS = {
    "Kalamalka_Lake": ["08NM143"], "Okanagan_Lake": ["08NM083"],
    "Skaha_Lake": ["08NM084"], "Vaseux_Lake": ["08NM243"],
    "Rainy_Lake": ["05PB024", "05PB007"],
    "Lake_of_the_Woods": ["05PD029", "05PD008", "05PE014", "05PD011"],
    "Playgreen_Lake": ["05UB005"], "Kiskitto_Lake": ["05UB013"],
    "Sipiwesk_Lake": ["05UD006"], "Split_Lake": ["05UF003"],
}

REG_STATIONS = {
    "Kalamalka_Lake": ["08NM065"], "Okanagan_Lake": ["08NM050"],
    "Skaha_Lake": ["08NM002"], "Vaseux_Lake": ["08NM247"],
    "Rainy_Lake": ["05PC019"],
    "Lake_of_the_Woods": ["05PE011", "05PE006"],
    "Playgreen_Lake": ["05UB009"], "Kiskitto_Lake": ["05UB009"],
    "Sipiwesk_Lake": ["05UE005"], "Split_Lake": ["05UF006"],
}
# 与 00_data_generation/ccm_modal_app.py 的 REGULATION_STATIONS 保持一致。
# 曾把 Split_Lake 误写为 05UF003——那是该湖的**水位站**，调控站是下游的
# Kettle 发电站 05UF006，抄错会使调控站三角形叠到水位站圆点上、看起来少一个。
# 下面在导入时校验一次，不一致即报错。
def _verify_reg_stations():
    import re
    src = (Path(__file__).resolve().parents[1] /
           "00_data_generation" / "ccm_modal_app.py").read_text(encoding="utf-8")
    blk = src[src.index("REGULATION_STATIONS = {"):]
    blk = blk[:blk.index("\n}") + 2]
    truth = {k: eval(v) for k, v in
             re.findall(r'"(\w+)":\s*(\[[^\]]*\])', blk)}
    bad = {k: (v, truth.get(k)) for k, v in REG_STATIONS.items()
           if truth.get(k) != v}
    if bad:
        raise SystemExit(f"REG_STATIONS 与流程代码不一致: {bad}")

STATION_COORDS = {
    "05PB007": (-93.32068, 48.64912), "05PB024": (-92.95836, 48.70047),
    "05PC019": (-93.40344, 48.60853), "05PD008": (-94.28349, 49.13280),
    "05PD011": (-94.81025, 49.71264), "05PD029": (-94.85311, 49.32847),
    "05PE006": (-94.50308, 49.77253), "05PE011": (-94.52447, 49.77197),
    "05PE014": (-94.55372, 49.76294), "05UB005": (-97.97500, 53.90278),
    "05UB009": (-98.04806, 54.49806), "05UB013": (-98.43833, 54.30306),
    "05UD006": (-97.50000, 55.09444), "05UE005": (-96.52500, 56.03889),
    "05UF003": (-96.08729, 56.24386), "05UF006": (-94.63389, 56.38028),
    "08NM002": (-119.58041, 49.34205), "08NM050": (-119.61536, 49.49852),
    "08NM065": (-119.26689, 50.23847), "08NM083": (-119.49966, 49.88613),
    "08NM084": (-119.57508, 49.42642), "08NM143": (-119.27442, 50.22992),
    "08NM243": (-119.52551, 49.27319), "08NM247": (-119.52803, 49.25684),
}

# 7 对直接水道连接（与 03_inter_lake_ccm/run_inter_lake_ccm.py 一致）；
# 箭头按水流方向绘制，上游 → 下游。
FLOW = [
    ("Kalamalka_Lake", "Okanagan_Lake"),
    ("Okanagan_Lake", "Skaha_Lake"),
    ("Skaha_Lake", "Vaseux_Lake"),
    ("Rainy_Lake", "Lake_of_the_Woods"),
    ("Playgreen_Lake", "Sipiwesk_Lake"),
    ("Kiskitto_Lake", "Sipiwesk_Lake"),
    ("Sipiwesk_Lake", "Split_Lake"),
]

# 逐湖标注偏移（面板宽/高的比例），用于避开彼此与比例尺
LABEL_OFFSETS = {
    "Kalamalka_Lake":    (-0.17, 0.03),
    "Okanagan_Lake":     (-0.16, 0.00),
    "Skaha_Lake":        (0.14, 0.01),
    "Vaseux_Lake":       (0.15, 0.00),
    "Rainy_Lake":        (0.00, -0.09),
    "Lake_of_the_Woods": (-0.17, 0.06),
    "Playgreen_Lake":    (-0.16, -0.03),
    "Kiskitto_Lake":     (-0.15, 0.05),
    "Sipiwesk_Lake":     (0.15, 0.00),
    "Split_Lake":        (0.00, 0.07),
}

C_LAKE = "#A8C6DC"       # 湖体填充
C_LAKE_EDGE = "#4A7C99"
C_LAND = "#F2F0EA"
C_OTHER_LAKE = "#DCE7EF"  # 非研究湖泊
C_WL = "#1F1F1F"
C_REG = "#C1452B"
C_FLOW = "#4A7C99"


def load_lakes() -> gpd.GeoDataFrame:
    """读取 10 个研究湖泊的多边形；首次读 1 GB 源文件后缓存。"""
    if CACHE.exists():
        return gpd.read_file(CACHE)
    ids = ",".join(str(v[0]) for v in LAKES.values())
    g = gpd.read_file(HYDROLAKES, where=f"Hylak_id IN ({ids})", engine="pyogrio")
    id2name = {v[0]: k for k, v in LAKES.items()}
    g["lake"] = g["Hylak_id"].map(id2name)
    g = g[["lake", "Hylak_id", "Lake_area", "geometry"]]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(CACHE, driver="GPKG")
    return g


CONTEXT_CACHE = CACHE.parent / "_cache_context_lakes.gpkg"


def load_context_lakes(bounds_wgs84) -> gpd.GeoDataFrame:
    """面板范围内其余较大湖泊，作为浅色背景，避免研究湖泊像孤岛。

    与研究湖泊同理：结果随仓库分发，没有 HydroLAKES 原始文件也能画。
    缓存里存的是三个面板范围的并集，这里按当前范围再裁一次。
    """
    minx, miny, maxx, maxy = bounds_wgs84
    if CONTEXT_CACHE.exists():
        g = gpd.read_file(CONTEXT_CACHE, bbox=(minx, miny, maxx, maxy))
        return g[["Hylak_id", "geometry"]]
    g = gpd.read_file(HYDROLAKES, bbox=(minx, miny, maxx, maxy),
                      where="Lake_area > 20", engine="pyogrio")
    return g[["Hylak_id", "geometry"]]


def station_gdf(mapping, panel) -> gpd.GeoDataFrame:
    rows = []
    for lake, sids in mapping.items():
        if LAKES[lake][1] != panel:
            continue
        for sid in sids:
            lon, lat = STATION_COORDS[sid]
            rows.append({"sid": sid, "lake": lake, "geometry": Point(lon, lat)})
    g = gpd.GeoDataFrame(rows, crs=4326).to_crs(CRS)
    # 同一站点服务多个湖泊时（如 Jenpeg 大坝）只画一次
    return g.drop_duplicates(subset="sid")


def add_scalebar(ax, length_km, label=None, pad=0.055, side="left"):
    """比例尺。投影单位为米，故直接按米作图。side 用于避开该面板的标注。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    L = length_km * 1000
    bx = (x1 - pad * w - L) if side == "right" else (x0 + pad * w)
    by = y0 + pad * h
    ax.plot([bx, bx + L], [by, by], color="black", lw=1.4,
            solid_capstyle="butt", zorder=8)
    for xx in (bx, bx + L):
        ax.plot([xx, xx], [by, by + 0.012 * h], color="black", lw=1.4, zorder=8)
    ax.text(bx + L / 2, by + 0.02 * h, label or f"{length_km} km",
            ha="center", va="bottom", fontsize=6.5, zorder=8)


def add_north_arrow(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x1 - 0.075 * (x1 - x0)
    y = y1 - 0.075 * (y1 - y0)
    dy = 0.085 * (y1 - y0)
    ax.annotate("", xy=(x, y), xytext=(x, y - dy),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1), zorder=8)
    ax.text(x, y + 0.012 * (y1 - y0), "N", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", zorder=8)


def draw_panel(ax, lakes_gdf, panel, title, pad_frac=0.14, scale_km=25,
               label_offsets=None, scale_side="left"):
    sub = lakes_gdf[lakes_gdf["lake"].map(lambda n: LAKES[n][1]) == panel]
    # 面板范围要同时容纳湖体与调控站。Sipiwesk 与 Split 自身出口无坝，其调节
    # 流量取自下游的发电站（Kelsey 05UE005、Kettle 05UF006），后者距 Split
    # 约 90 km；只按湖体算边界会把它挤出画面，恰好漏掉 §3.2.1 所说
    # 「调节流量对该湖自身出流的代表程度不一」最需要看到的一例。
    st = station_gdf(REG_STATIONS, panel)
    bounds = gpd.GeoSeries(list(sub.geometry) + list(st.geometry),
                           crs=CRS).total_bounds
    minx, miny, maxx, maxy = bounds
    px, py = (maxx - minx) * pad_frac, (maxy - miny) * pad_frac
    pad = max(px, py)
    ext = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    # 背景：陆地、其余湖泊
    ext_wgs = (gpd.GeoSeries([box(*ext)], crs=CRS).to_crs(4326)
               .total_bounds)
    ax.set_facecolor(C_LAND)
    try:
        ctx = load_context_lakes(ext_wgs).to_crs(CRS)
        ctx[~ctx["Hylak_id"].isin([v[0] for v in LAKES.values()])].plot(
            ax=ax, facecolor=C_OTHER_LAKE, edgecolor="none", zorder=1)
    except Exception as exc:            # 背景湖泊缺失不影响主图
        print(f"  [{panel}] 背景湖泊读取失败，跳过: {type(exc).__name__}")

    # 边线必须细：Rainy 与 Lake of the Woods 的岸线极破碎，线宽 0.6 时
    # 边线本身的面积超过湖体，整个湖会显示成深色块。
    sub.plot(ax=ax, facecolor=C_LAKE, edgecolor=C_LAKE_EDGE, linewidth=0.22, zorder=2)

    # 水流方向箭头：由上游湖泊质心指向下游湖泊质心
    geom = {r["lake"]: r.geometry for _, r in sub.iterrows()}
    cent = {k: g.representative_point() for k, g in geom.items()}
    # 箭头连接两个湖体的**最近点**而非质心：质心连线会斜穿陆地，
    # 在 Okanagan 这种狭长串联湖上尤其失真。
    # 串联紧邻的湖泊（Okanagan 链）最近点相距仅数百米，按原样绘制箭头长度
    # 接近零、在面板尺度上不可见。低于最小长度时，沿两湖质心方向在最近点
    # 中点处补足一段可见的箭头，方向仍由上游指向下游。
    min_len = 0.055 * max(ext[2] - ext[0], ext[3] - ext[1])
    for up, dn in FLOW:
        if up in geom and dn in geom:
            a, b = nearest_points(geom[up], geom[dn])
            ax_, ay_, bx_, by_ = a.x, a.y, b.x, b.y
            if math.hypot(bx_ - ax_, by_ - ay_) < min_len:
                ux, uy = cent[dn].x - cent[up].x, cent[dn].y - cent[up].y
                norm = math.hypot(ux, uy) or 1.0
                ux, uy = ux / norm, uy / norm
                mx, my = (ax_ + bx_) / 2, (ay_ + by_) / 2
                ax_, ay_ = mx - ux * min_len / 2, my - uy * min_len / 2
                bx_, by_ = mx + ux * min_len / 2, my + uy * min_len / 2
            ax.annotate("", xy=(bx_, by_), xytext=(ax_, ay_),
                        arrowprops=dict(arrowstyle="-|>", color=C_FLOW, lw=1.0,
                                        alpha=0.9, shrinkA=1, shrinkB=1),
                        zorder=4)

    station_gdf(REG_STATIONS, panel).plot(ax=ax, color=C_REG, marker="^",
                                          markersize=20, edgecolor="white",
                                          linewidth=0.4, zorder=5)
    station_gdf(WL_STATIONS, panel).plot(ax=ax, color=C_WL, marker="o",
                                         markersize=11, edgecolor="white",
                                         linewidth=0.4, zorder=6)

    # 湖泊标注
    offs = label_offsets or LABEL_OFFSETS
    for name, pt in cent.items():
        dx, dy = offs.get(name, (0.0, 0.03))
        ax.annotate(LAKES[name][2],
                    xy=(pt.x, pt.y),
                    xytext=(pt.x + dx * (ext[2] - ext[0]),
                            pt.y + dy * (ext[3] - ext[1])),
                    fontsize=6.8, ha="center", va="center", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.set_title(title, fontsize=8, pad=4, loc="left")
    add_scalebar(ax, scale_km, side=scale_side)
    add_north_arrow(ax)
    return ext


def draw_locator(ax, extents):
    countries = gpd.read_file(NE_COUNTRIES).to_crs(CRS)
    canada = countries[countries.get("ADMIN", countries.columns[0]).astype(str)
                       .str.contains("Canada", case=False, na=False)]
    ax.set_facecolor("white")
    countries.plot(ax=ax, facecolor="#FBFAF7", edgecolor="#BBBBBB", linewidth=0.3, zorder=0)
    canada.plot(ax=ax, facecolor=C_LAND, edgecolor="#7A7A7A", linewidth=0.6, zorder=1)
    if NE_PROVINCES.exists():
        gpd.read_file(NE_PROVINCES).to_crs(CRS).plot(
            ax=ax, color="#CFCFCF", linewidth=0.3, zorder=2)

    # 方框设最小可见边长：Okanagan 链只有约 100 km，按真实范围画会小到看不见
    MIN_SIDE = 4.2e5
    for tag, ext in extents.items():
        cx, cy = (ext[0] + ext[2]) / 2, (ext[1] + ext[3]) / 2
        w = max(ext[2] - ext[0], MIN_SIDE)
        h = max(ext[3] - ext[1], MIN_SIDE)
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                               edgecolor=C_REG, linewidth=1.1, zorder=5))
        ax.annotate(tag, xy=(cx, cy + h / 2), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=C_REG, fontweight="bold", zorder=6)

    # 裁去高纬群岛与美国大部，让研究区在图中占更大比重。
    # x 下界不能高于 Okanagan 链（EPSG:3347 下 x≈4.26e6），否则 (b) 框会被裁掉。
    ax.set_xlim(3.85e6, 8.00e6)
    ax.set_ylim(0.90e6, 3.60e6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.set_title("(a) Study regions within Canada", fontsize=8, pad=4, loc="left")


def main():
    _verify_reg_stations()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lakes = load_lakes().to_crs(CRS)
    print(f"读入 {len(lakes)} 个研究湖泊多边形")

    w_in = FIG_WIDTH_CM / 2.54
    fig = plt.figure(figsize=(w_in, w_in * 0.92))
    gs = fig.add_gridspec(2, 2, wspace=0.05, hspace=0.16,
                          left=0.012, right=0.988, top=0.955, bottom=0.075)

    ax_loc = fig.add_subplot(gs[0, 0])
    ax_ok = fig.add_subplot(gs[0, 1])
    ax_wd = fig.add_subplot(gs[1, 0])
    ax_ne = fig.add_subplot(gs[1, 1])

    ext_ok = draw_panel(ax_ok, lakes, "okanagan",
                        "(b) Okanagan chain, British Columbia", scale_km=20,
                        scale_side="right")
    ext_wd = draw_panel(ax_wd, lakes, "woods",
                        "(c) Rainy Lake and Lake of the Woods", scale_km=50)
    ext_ne = draw_panel(ax_ne, lakes, "nelson",
                        "(d) Nelson River chain, Manitoba", scale_km=100)
    draw_locator(ax_loc, {"(b)": ext_ok, "(c)": ext_wd, "(d)": ext_ne})

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=C_WL,
               markeredgecolor="white", markersize=4.5, label="Water-level gauge"),
        Line2D([], [], marker="^", color="none", markerfacecolor=C_REG,
               markeredgecolor="white", markersize=5.5,
               label="Regulation control station"),
        Line2D([], [], color=C_FLOW, lw=1.1, label="Direction of flow"),
        Line2D([], [], marker="s", color="none", markerfacecolor=C_LAKE,
               markeredgecolor=C_LAKE_EDGE, markersize=6, label="Study lake"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, 0.028), handletextpad=0.4,
               columnspacing=1.3)

    for path, fmt in [(OUT_DIR / "figure_3_1_study_area.pdf", "pdf"),
                      (OUT_DIR / "figure_3_1_study_area.png", "png")]:
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor="white", format=fmt)
        print(f"  已写出 {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
