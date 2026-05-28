# opencv-ui-cli — 图像匹配 CLI 工具

## Context

用户需要自动识别 UI 设计稿中各个组件的位置。输入一张完整设计图 `design.png` 和一个 `components/` 文件夹，使用 OpenCV 模板匹配找到每个组件在 design 中的位置。需考虑组件被其他控件遮挡的情况，以信任度标记。输出 TOML 位置列表 + 每个组件一张 `{组件名}-matches-N.png` 可视化确认图。

## 技术选型

- **语言**：Rust
- **图像库**：`opencv` crate（模板匹配 + 图像合成）
- **SVG 渲染**：`resvg` + `usvg`（SVG → 像素，用作蒙版叠加层）
- **输出**：TOML（位置数据） + 每组件一张 `{name}-matches-N.png`

## 项目结构

```
opencv-ui-cli/
├── Cargo.toml
├── src/
│   └── main.rs
├── design.png
└── components/
    ├── button.png
    └── icon.png
```

输出示例：

```
match_result.toml
button-matches-N.png      # design + button 位置蒙版
icon-matches-N.png        # design + icon 位置蒙版
```

## 动态阈值降级匹配

乐观假设：components 目录下的每张图在 design 中**至少存在一次**。

匹配时从高阈值开始，逐步降低直到找到匹配：

```
threshold = 0.95
while threshold >= 0.5:
    matches = matchTemplate(threshold)
    if matches 非空 → 记录结果，结束本组件
    threshold -= 0.05
```

### 信任度等级

信任度基于**单次匹配的置信度分数**，与匹配时的阈值无关：

| 等级     | 条件               | 含义             | 蒙版颜色   |
| -------- | ------------------ | ---------------- | ---------- |
| `high`   | confidence >= 0.90 | 高度可信         | 绿色半透明 |
| `medium` | confidence >= 0.75 | 基本可信         | 黄色半透明 |
| `low`    | confidence < 0.75  | 存疑，需人工确认 | 红色半透明 |

## CLI 接口

```
opencv-ui-cli <design> <components-dir> [options]

Options:
  -o, --output <path>          输出 TOML 路径，默认 match_result.toml
  --start-threshold <float>    起始阈值，默认 0.95
  --min-threshold <float>      最低阈值，默认 0.5
  --threshold-step <float>     每次降低步长，默认 0.05
  --nms-threshold <float>      NMS IoU 阈值，默认 0.3
  --no-mask                    不生成 {name}-matches-N.png
```

## 核心流程

1. 设计图用 `IMREAD_COLOR` 加载（不需要 alpha）
2. 遍历 components/ 下所有图片，用 `load_with_mask` 加载（见下方 Alpha Mask 处理）
3. **动态降阈** — 从 start_threshold 开始 matchTemplate(mask=alpha)，匹配到则记录后跳出，否则降 step 重试，直到 min_threshold
4. NMS 去重，按置信度分 trust 等级
5. 汇总 → 输出 match_result.toml
6. **生成可视化**（每组件一张）：
   a. 根据该组件的匹配位置，用 `<rect>` 构建 SVG 蒙版（不同信任度不同颜色）
   b. `resvg` 将 SVG 渲染为 RGBA 像素缓冲区
   c. OpenCV 将蒙版图层 alpha 混合到 design 副本上
   d. 保存为 `{组件名}-matches-N.png`
7. **生成重构设计图** — `try-implement-design.png`：
   a. 创建与 design 等大的白色画布
   b. 将每个组件的 opaque 像素（通过 alpha mask）绘制到匹配位置上
   c. 用 SVG 绘制各组件对应的半透明虚线框 + 文件名/置信度标签（不同组件不同颜色）
   d. `resvg` 渲染 SVG 图层，alpha 混合到画布上
   e. 保存为 `try-implement-design.png`

## try-implement-design.png

白底重构图为每个组件预留了不同颜色的标识：

| 组件序号 | 框体颜色 | hex     |
| -------- | -------- | ------- |
| 1        | 红       | #e74c3c |
| 2        | 绿       | #2ecc71 |
| 3        | 蓝       | #3498db |
| 4        | 橙       | #f39c12 |
| 5        | 紫       | #9b59b6 |
| 6        | 青       | #1abc9c |
| 7        | 深橙     | #e67e22 |
| 8        | 深蓝     | #2980b9 |

## Alpha Mask 处理

### 问题背景

Figma、Sketch 等设计工具导出的透明 PNG，透明区域的 RGB 值通常为 `(0,0,0)`（纯黑），alpha = 0。如果用 `IMREAD_COLOR` 加载（丢弃 alpha 通道），这些黑色像素会参与 `matchTemplate` 的相似度计算，严重拉低匹配分数，导致本该匹配的组件返回 0 结果。

```
透明 PNG 实际数据：            IMREAD_COLOR 加载后：
┌──────────────────┐          ┌──────────────────┐
│ 透明区 RGBA=0,0,0,0  │    →    │ 黑色 RGB=0,0,0      │  ← 干扰匹配！
│ 控件区 RGBA=*  ,255  │    →    │ 控件 RGB=*          │
└──────────────────┘          └──────────────────┘
```

### 解决方案

`load_with_mask` 函数分三步处理：

1. **`IMREAD_UNCHANGED`** 加载，保留 4 通道 BGRA 数据
2. **`cvtColor(COLOR_BGRA2BGR)`** → 得到 3 通道 BGR 像素（供模板匹配使用）
3. **`extract_channel(3)`** → 提取 alpha 通道作为 1 通道 mask（0=透明/忽略, 255=不透明/参与匹配）

将 mask 传入 `matchTemplate` 的 mask 参数，OpenCV 在计算相似度时只考虑 mask 非零的像素。

### 对比

| 场景                          | 无 mask                             | 有 mask              |
| ----------------------------- | ----------------------------------- | -------------------- |
| Figma 导出 PNG（透明区=纯黑） | 置信度 0.35，位置错误，**匹配失败** | 置信度 1.0，位置正确 |
| 精确裁剪 PNG（无透明区）      | 正常                                | 正常                 |
| 带透明边框的组件              | 置信度低，可能失败                  | 正确匹配             |

### 其他受影响的逻辑

- **`template_variance`**：计算模板像素方差时也跳过透明像素，避免透明区的黑色像素影响方差判断（方差用于决定用 TM_CCOEFF_NORMED 还是 TM_SQDIFF_NORMED）
- **匹配算法自适应**：方差 < 1.0 的模板（纯色矩形等）自动切换为 `TM_SQDIFF_NORMED`，因为 `TM_CCOEFF_NORMED` 在纯色模板上分子分母均为 0，返回无意义结果

## SVG 蒙版生成细节

对每个组件生成如下 SVG：

```xml
<svg width="1920" height="1080" xmlns="http://www.w3.org/2000/svg">
  <!-- high trust -->
  <rect x="100" y="200" width="80" height="40" fill="rgba(0,200,0,0.3)" stroke="rgb(0,180,0)" stroke-width="2"/>
  <text x="100" y="195" font-size="12" fill="green">button.png 0.95</text>
  <!-- low trust -->
  <rect x="500" y="300" width="80" height="40" fill="rgba(200,0,0,0.3)" stroke="rgb(180,0,0)" stroke-width="2"/>
  <text x="500" y="295" font-size="12" fill="red">button.png 0.62</text>
</svg>
```

渲染为 RGBA 像素后，用 OpenCV `addWeighted` 或逐像素 alpha blend 叠加到 design 副本上，输出 PNG。

## TOML 输出格式

```toml
[design]
file = "design.png"
width = 1920
height = 1080

[[matches]]
component = "button.png"
count = 2
positions = [
  { x = 100, y = 200, width = 80, height = 40, confidence = 0.95, trust = "high" },
  { x = 500, y = 300, width = 80, height = 40, confidence = 0.62, trust = "low" },
]

[[matches]]
component = "icon.png"
count = 1
positions = [
  { x = 50, y = 60, width = 24, height = 24, confidence = 0.88, trust = "high" },
]
```

## 依赖

```toml
[dependencies]
opencv = "0.92"
clap = { version = "4", features = ["derive"] }
toml = "0.8"
serde = { version = "1", features = ["derive"] }
anyhow = "1"
resvg = "0.42"
usvg = "0.42"
```

## 验证方式

1. 准备测试 design（几何图形拼接），截取组件，额外做一个被遮挡的变体
2. 运行 `cargo run -- design.png components/`
3. 检查 `match_result.toml` 坐标与信任度是否正确
4. 打开各 `{name}-matches-N.png`，目视确认蒙版位置、颜色是否准确

## YOLO 候补检测 — low-trust 组件

### 动机

模板匹配在组件半透明、被遮挡、变形或颜色偏移时，置信度显著下降，标为 `low` 信任度。这些 low-trust 位置可能完全不准确。引入 YOLO 作为独立第二检测器：用 `components/` 下的图片自动生成训练数据并微调模型，让 YOLO 学会"这些组件长什么样"，然后直接在 design 上检测。

与模板匹配不同，YOLO 是语义理解——对遮挡、半透明、轻微变形的容忍度更高。

### 触发条件

只有同时满足以下条件的组件才走 YOLO 候补：

1. **trust = "low"** — 模板匹配置信度 < 0.75（不论从哪个 pass 来的）
2. **confidence < `--yolo-threshold`** — 默认 0.5，低于此阈值才触发

medium / high 信任度的组件跳过 YOLO，直接用模板匹配结果。

### 自动训练

无需用户手动标注。从 `components/` 自动生成训练数据：

```
components/
├── button.png       → 自动标为 class "button"
├── icon.png         → 自动标为 class "icon"
└── text_field.png   → 自动标为 class "text_field"

训练数据生成（scripts/train_yolo.py）：
1. 取每张 component 图，随机缩放 ±20%，旋转 ±5°
2. 随机贴到多张背景图上（纯色、渐变、design 局部裁剪）
3. 记录贴图位置 → 自动生成 YOLO 标注文件
4. 用 yolov8n.pt 预训练权重 fine-tune（～50 epochs，几分钟）
5. 输出 components/.yolo-cache/ui-detect.pt
```

已训练的模型**缓存复用**：若 `components/.yolo-cache/ui-detect.pt` 已存在且比 `components/` 下所有文件都新，跳过训练直接推理。

### 检测流程

Python 侧边车 `scripts/yolo_detect.py` 负责两件事：

1. **训练**（按需）：`python3 scripts/yolo_detect.py train --components components/ --output components/.yolo-cache/ui-detect.pt --epochs 50`
2. **推理**：`python3 scripts/yolo_detect.py detect --model components/.yolo-cache/ui-detect.pt --source design.png --output /tmp/yolo_result.json`

推理输出是对 design 的全图检测，包含所有找到的组件位置和类别。

### 结果合并

对每个 low-trust 组件，查找 YOLO 检测结果中**同类别（文件名匹配）且 IoU > 0.3** 的结果：

| 场景                                | 处理                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| YOLO 找到同类组件，IoU ≥ 0.3      | 用 YOLO 的 bbox 坐标和置信度**替换**模板匹配结果，trust 提升为 `medium` |
| YOLO 找到同类组件，但 IoU < 0.3   | 保留 YOLO 结果作为**新 position 追加**                       |
| YOLO 未找到该类组件                 | 保留模板匹配原始结果，trust 维持 `low`（不提升）             |

**YOLO 置信度到 trust 的映射**：

| YOLO confidence | 映射 trust |
| --------------- | ---------- |
| ≥ 0.75          | medium     |
| < 0.75           | low        |

YOLO 结果不会跃升到 `high` ——只有模板匹配能达到 high。

### 架构：Python 子进程桥接

- **主流程**：Rust 模板匹配（两轮）→ 收集 low-trust 组件 → 调用 Python 侧边车（训练 + 推理）→ 合并结果
- **Python 侧边车**：`scripts/yolo_detect.py`，使用 `ultralytics` 库
- **通信方式**：临时文件（JSON 输出）
- **失败策略**：fail-open。Python 未安装 / ultralytics 缺失 / 训练失败 → 模板匹配结果保持不变，输出 warning

### CLI 接口新增

```
--yolo                  启用 YOLO 候补检测（默认关闭，零性能开销）
--yolo-threshold <FLOAT>  触发 YOLO 的置信度阈值，低于此值的 low-trust 组件走 YOLO。默认 0.5
--yolo-conf <FLOAT>       YOLO 推理置信度阈值，默认 0.25
--yolo-epochs <INT>       fine-tune 训练轮数，默认 50
--yolo-python <PATH>      侧边车脚本路径，默认 scripts/yolo_detect.py
--yolo-cache <PATH>        模型缓存路径，默认 components/.yolo-cache/ui-detect.pt
```

### 数据格式

**YOLO 推理输出 JSON**：

```json
{
  "detections": [
    {
      "class_name": "button",
      "confidence": 0.87,
      "bbox": [x, y, width, height]
    }
  ],
  "count": 1,
  "error": null
}
```

**Position 结构新增字段**：

```toml
[[matches.positions]]
x = 100
y = 200
width = 80
height = 40
confidence = 0.87
trust = "medium"
source = "yolo"         # 新增："template" | "yolo"，标记来源
```

### 执行时机

```
1. 第一轮模板匹配（原始 design）
2. 第二轮模板匹配（mask working copy，替换更好的结果）
3. 收集 trust="low" 且 confidence < yolo_threshold 的组件
4. YOLO 训练（按需）+ 推理
5. 结果合并（替换/追加 low-trust 位置）
6. 生成可视化 + TOML 输出
```

### 文件变更

| 文件                      | 变更                                                           |
| ------------------------- | -------------------------------------------------------------- |
| `src/main.rs`             | 新增 YOLO CLI 参数、`Position.source` 字段、合并逻辑、可视化   |
| `src/yolo.rs`             | **新建** — YoloConfig、`run_yolo_fallback()`                   |
| `scripts/yolo_detect.py`  | **新建** — train + detect 子命令                               |
| `Cargo.toml`              | 新增 `serde_json = "1"`                                        |
| `.gitignore`              | 新增 `**/.yolo-cache/`、`datasets/`                            |

### 环境要求

```bash
pip3 install ultralytics
# 首次运行自动下载 yolov8n.pt 预训练权重（ultralytics 缓存到 ~/.cache/）
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# 微调后的模型缓存到 components/.yolo-cache/ui-detect.pt
```

### 验证方式

1. 不加 `--yolo` 运行，确认行为零变化
2. 加 `--yolo`，准备含遮挡/半透明组件的 design，确认 low-trust 位置被 YOLO 替换，trust 提升为 medium
3. 临时重命名 `python3`，确认工具输出 warning 但正常完成，位置保持 `low`
4. 检查 `match_result.toml` 中 `source = "yolo"` 的记录，目视确认坐标准确
