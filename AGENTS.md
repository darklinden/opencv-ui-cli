# opencv-ui-cli — 图像匹配 CLI 工具

## Context

用户需要自动识别 UI 设计稿中各个组件的位置。输入一张完整设计图 `design.png` 和一个 `components/` 文件夹，使用 OpenCV 模板匹配找到每个组件在 design 中的位置。需考虑组件被其他控件遮挡的情况，以信任度标记。输出 TOML 位置列表 + 每个组件一张 `{组件名}-match.png` 可视化确认图。

## 技术选型

- **语言**：Rust
- **图像库**：`opencv` crate（模板匹配 + 图像合成）
- **SVG 渲染**：`resvg` + `usvg`（SVG → 像素，用作蒙版叠加层）
- **输出**：TOML（位置数据） + 每组件一张 `{name}-match.png`

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
button-match.png      # design + button 位置蒙版
icon-match.png        # design + icon 位置蒙版
```

## 信任度模型

两阶段匹配：

| 阶段 | 阈值 | 说明 |
|------|------|------|
| 阶段 1 | 0.85 | 清晰可见组件 → trust = `high`，标记区域为"已占用" |
| 阶段 2 | 0.5 | 剩余区域搜索，与 high 区域重叠 → trust = `low`，不重叠 → trust = `medium` |

### 信任度等级与蒙版颜色

| 等级 | 含义 | 蒙版颜色 |
|------|------|----------|
| `high` | 清晰匹配，高度可信 | 绿色半透明 rgba(0,200,0,0.3) |
| `medium` | 存在但匹配模糊 | 黄色半透明 rgba(200,200,0,0.3) |
| `low` | 可能被遮挡 | 红色半透明 rgba(200,0,0,0.3) |

## CLI 接口

```
opencv-ui-cli <design> <components-dir> [options]

Options:
  -o, --output <path>         输出 TOML 路径，默认 match_result.toml
  --high-threshold <float>    高置信度阈值，默认 0.85
  --low-threshold <float>     低置信度阈值，默认 0.5
  --nms-threshold <float>     NMS IoU 阈值，默认 0.3
  --no-mask                   不生成 {name}-match.png
```

## 核心流程

1. 设计图用 `IMREAD_COLOR` 加载（不需要 alpha）
2. 遍历 components/ 下所有图片，用 `load_with_mask` 加载（见下方 Alpha Mask 处理）
3. **阶段 1** — matchTemplate(threshold=0.85, mask=alpha) → NMS → trust=high，记录已占用矩形列表
4. **阶段 2** — matchTemplate(threshold=0.5, mask=alpha) → 排除已被占用的候选 → NMS →
   - 与 high 区域重叠 → trust=low
   - 无重叠 → trust=medium
5. 汇总 → 输出 match_result.toml
6. **生成可视化**（每组件一张）：
   a. 根据该组件的匹配位置，用 `<rect>` 构建 SVG 蒙版（不同信任度不同颜色）
   b. `resvg` 将 SVG 渲染为 RGBA 像素缓冲区
   c. OpenCV 将蒙版图层 alpha 混合到 design 副本上
   d. 保存为 `{组件名}-match.png`

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

| 场景 | 无 mask | 有 mask |
|------|---------|---------|
| Figma 导出 PNG（透明区=纯黑） | 置信度 0.35，位置错误，**匹配失败** | 置信度 1.0，位置正确 |
| 精确裁剪 PNG（无透明区） | 正常 | 正常 |
| 带透明边框的组件 | 置信度低，可能失败 | 正确匹配 |

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
4. 打开各 `{name}-match.png`，目视确认蒙版位置、颜色是否准确
