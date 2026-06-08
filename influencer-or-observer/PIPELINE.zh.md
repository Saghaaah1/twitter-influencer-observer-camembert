# 流水线实现说明（influencer-or-observer）

本仓库是一个**自包含**的 PyTorch 深度 MLP 流水线，用于 Kaggle 上的一个 Twitter 二分类任务（法语语料的推文 / 账号分类，`label` 为按用户恒定的标签）。整套设计只围绕**一个核心目标**：**避免身份泄漏（identity leakage）**。

> 历史教训：早期提交 `sub10` 交叉验证到 94.4%、`sub21` 到 97.48%，但公榜只有 0.839 —— 原因是某些特征在按行交叉验证下沦为"用户身份代理"，把成绩虚高了。本仓库的所有设计都是为了**让 OOF 分数可信、与公榜对齐**。

---

## 一、文件结构

| 文件 | 职责 |
|------|------|
| `config.py` | 唯一配置源：路径、随机种子、超参数、禁用列 |
| `features.py` | "诚实"特征工程 + fold-safe 来源目标编码 |
| `model.py` | `TabularMLP` 网络结构 |
| `train.py` | 总编排：加载 → 交叉验证 → 训练 → 融合 → 平滑 → 生成提交 |

---

## 二、整体 Pipeline（`train.py`）

```
加载 jsonl ─► 展平嵌套字段 ─► 构建诚实特征(一次) ─► 解析 source(一次)
   ─► 方差守卫 + 泄漏守卫
   ─► 5 折交叉验证循环：
        ├─ fold-safe 来源目标编码（只用本折训练行）
        ├─ fold-safe 中位数填补（引用推文列）
        ├─ fold-safe StandardScaler（只 fit 训练行，供 MLP）
        ├─ 训练 MLP（AdamW + OneCycleLR + 早停）
        └─ 可选：同折训练 LightGBM
   ─► 阈值调优(只用 OOF) ─► 可选融合 ─► 可选按用户平滑 ─► 写出 submission.csv
```

### 1. 加载数据 — `train.py:53-56`
读取 `train.jsonl` / `kaggle_test.jsonl`，用 `json_normalize` 把嵌套对象（`user.created_at`、`quoted_status.user.followers_count` 等）展平成带点号的列。

### 2. 一次性构建基础特征 — `train.py:209-217`
约 74 个诚实的逐推文特征；并**只解析一次** `source`（HTML 列，最慢的一步），按折复用。

### 3. 两道安全守卫
- **方差守卫** `train.py:219-228`：自动剔除仍为常量的列（数据刷新后可自愈）。
- **泄漏守卫** `train.py:230-232`：硬 `assert`，确保 `config.BANNED_COLUMNS`（个人资料颜色 / 背景图 URL 等身份代理列）一个都没进入特征矩阵。

### 4. 5 折交叉验证循环 — `train.py:257-307`（核心）
对**每一折**，所有"依赖标签 / 依赖分布"的步骤都只在**本折训练行**上拟合，再应用到验证集 + 测试集：
- **fold-safe 来源目标编码** `train.py:259`
- **fold-safe 中位数填补** `train.py:269-275`（引用推文列先留 NaN，避免 `-1` 哨兵值污染标准化）
- **fold-safe StandardScaler** `train.py:286-289`（供 MLP）
- **训练一个 MLP** `train.py:291-293`，累加 OOF 概率与平均测试概率

---

## 三、关键方法

### 1. 交叉验证策略 —— 诚实性的关键（`--group`）
核心洞见（`train.py:59-72`、`train.py:237-246`）：
- 数据里**直接的用户 ID 被剥离了**，但 `user.created_at` 在同一账号内**完全恒定**、跨账号近乎唯一。把它和 `user.name` 拼起来就构成了**代理用户键（proxy user key）**。
- **默认 `StratifiedKFold`**：同一用户的多条推文会落在不同折 → 用户恒定特征（tweets_per_day、log_age、source_te……）被当作身份"背"下来 → **OOF 偏乐观（约 89–91%）**。
- **`--group` 的 `StratifiedGroupKFold`**：把每个用户的所有行锁在同一折内（折间用户互不相交）→ 身份无法被记忆 → **OOF 更低但诚实**，能对齐"用户不相交"的公榜。两者之差量化了行级 CV 的虚高程度。
- 关键：这个键**只用于切分，绝不进入特征**。

### 2. 特征工程（`features.py`）
- **只产出"诚实"特征**：逐推文合法可得的信息 —— 用户资料标志位、账号年龄、行为比率（发推/天、listed/推文）、文本表层统计（长度、话题标签数、大写比例、type-token ratio）、引用推文统计、时间戳的周期编码（`hour_sin`/`hour_cos`）。
- **不做任何跨行用户聚合**：因为用户 ID 已被剥离，"按用户 group by"实际会跨整个数据集聚合 = 泄漏（这正是早期虚高的原因）。
- **唯一依赖标签的特征**是 fold-safe **来源目标编码**（`features.py:325-343`）：按 source app 计算平滑后的标签均值，smoothing=20。
- **来源类别 one-hot**（`features.py:74-85`）：用子串关键词把约 270 个 app 分到 official / scheduler / automation / other / unknown 五类（此步与标签无关，不泄漏）。
- 丢弃恒定的死列（`features.py:43-48`）：`lang` 100% 是法语、所有互动计数在这份新抓取语料里都是 0。

### 3. 模型（`model.py`）
`TabularMLP`：`256 → 128 → 64 → 1`，每个块为 `Linear → BatchNorm1d → GELU → Dropout(0.30/0.30/0.20)`，单 logit 输出 + `BCEWithLogitsLoss`。
- 训练：**AdamW + OneCycleLR**，按验证准确率**早停**（patience=6），恢复验证最优的权重。
- 文件里坦诚说明：在这类表格元数据上 GBDT 通常能追平甚至超过 MLP；MLP 存在只是因为需求方要求"深度学习"。

### 4. 三个可选增强（命令行开关）
1. **`--blend`**（`train.py:299-348`）：在**相同的折**上训练 **LightGBM**（不缩放，树天然尺度不变，能捕捉 MLP 漏掉的轴对齐交互）。融合权重 `alpha·MLP + (1-alpha)·LGBM` 与阈值在**对齐的 OOF** 上联合调优 —— 诚实，因为不碰测试标签。
2. **`--smooth`**（`train.py:350-363`）：由于标签**按用户恒定**，在阈值化前把每个用户的逐推文概率取平均（按代理键分组），可去除逐推文噪声。据项目记忆，这是单项提升最大的一步。
3. **阈值调优**（`train.py:313-325`）：仅在 OOF 标签上做 0.40–0.60 网格搜索。

### 5. 自我监控
- TensorBoard 记录每折 train loss / val acc / lr（`train.py:132-135`），仅观测、不影响训练。
- **泄漏警报**（`train.py:371-372`）：若 OOF > 95% 就打印"可能泄漏"的告警 —— 把 sub21 的教训直接写进代码。

---

## 四、推荐运行方式

```bash
python train.py --group --blend --smooth
```

含义：诚实的"用户不相交"交叉验证 + MLP/LightGBM 集成 + 按用户平滑。

| 配置 | 诚实准确率 |
|------|-----------|
| `--group --blend` | 84.31% |
| `--group --blend --smooth` | **≈ 84.78%**（平滑约 +0.47） |
| 理论上限 | 约 85% |

其它命令：

```bash
python train.py                  # 默认行级 StratifiedKFold，仅 MLP（OOF 偏乐观）
python train.py --group          # 诚实 CV：按代理用户键的 StratifiedGroupKFold
python train.py --smoke          # 小型冒烟测试（5k 行，2 epoch，2 折）
```

---

## 五、一句话总结

这**不是**一个"压榨准确率"的流水线，而是一个**"不要自欺"**的流水线。group CV、处处 fold-safe、禁用列、>95% 告警 —— 每一个设计都为了让 OOF 数字**可信、并与真实公榜对齐**，因为早期提交曾在 CV 上拿到 97%、却在用户不相交的真实场景里崩盘。当前特征工程已接近饱和，诚实上限约 85%。
