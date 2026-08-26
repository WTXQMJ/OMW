# ERCPSP 质量风险多智能体强化学习 - 核心逻辑文档

> **版本**: v2.5
> **更新日期**: 2026-03-30
> **算法**: MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
> **更新说明**: v2.5 - 新增生产智能体同质子群参数共享（两条生产线共享同一Actor网络）；同质子群loss均值汇总与WandB日志记录；v2.4 - 新增每步完成进度奖励（每步+500*完成数 + 500*完成比例），解决冷启动困境；修复production_pool初始化错误和环境逻辑bug；v2.3 - 根据fixfix.ini重构奖励函数：改为成本最小化机制；各智能体奖励=负成本；新增模具成本/预制件生产成本的显式累计；引入可学习权重+Adam优化器自动优化权重比例
> **问题域**: 带质量风险的预制构件协同调度 (ERCPSP)

---

## 1. 问题背景与算法概述

### 1.1 任务背景

**ERCPSP-MMARL** (Multi-Agent Reinforcement Learning for ERCPSP with Quality Risk)

本项目针对装配式混凝土预制构件生产-运输-施工全流程协同调度问题，引入三节点质量风险机制，采用多智能体强化学习（MARL）算法进行优化求解。

**核心挑战**：
- 生产-运输-施工三阶段强耦合
- 质量风险导致的不确定性状态转移（ε₁=5%, ε₃=3%, ε₄=2%，实际失败概率随抽检数量动态变化）
- 多资源（生产线、车辆、塔吊）的协调调度
- 订单规模的动态性要求算法具有泛化能力
- **吊装顺序约束**：每批预制构件均有对应的吊装顺序要求
- **时间不确定性**：生产时间和运输时间存在随机波动
- **质量检验决策**：检验数量可调，跳过检验则概率累计；检验成本与数量成正比

### 1.2 预制构件工序流程

单个预制构件从生产到完工的全流程如下：

```
构件生产养护 → 入库存储 → 构件运输 → 卸货(现场) → 吊装准备(虚拟) → 吊装 → 安装(虚拟)
```

**工序说明**：

| 工序名称 | 类型 | 说明 |
|---------|------|------|
| 构件生产养护 | 实体工序 | 模具可用时直接生产；模具不可用时先生产模具再生产构件；生产时间含±5%随机波动 |
| 入库存储 | 实体工序 | 生产完成并通过节点1质检后入库养护 |
| 构件运输 | 实体工序 | 车辆从库存装载，运输时间含随机波动(8h±1h)；通过节点2质检后到达现场 |
| 卸货(现场) | 实体工序 | 塔吊执行卸货操作，卸货时间=0.167h（10min） |
| 吊装准备 | **虚拟工序** | 塔吊选择"直接吊装"或"先堆放再吊装" |
| 吊装 | 实体工序 | 塔吊执行吊装操作；特种构件45min，常规构件26min，均含±5min随机波动 |
| 安装 | **虚拟工序** | 通过节点3质检，完成安装 |

**关键说明**：
- 吊装准备和安装为**虚拟工序**，不占用真实时间，但影响塔吊的决策分支
- 模具可用性：特种构件（Type-1/Type-2/Type-3）有对应模具；常规构件（Type-4/Type-5）模具已就绪，无需生产
- 堆放决策：若塔吊选择"堆放"，构件先卸货入库存储（加入现场存储池），后续再执行吊装；若选择"直接吊装"，到达后立即执行吊装
- **时间随机性**：生产时间在基准值±5%内随机波动；运输时间为8h±1h；吊装时间在基准值±5min内随机波动

### 1.3 单构件生命周期决策逻辑

对于单个预制构件，完整的决策链条如下：

```
Step 1: [生产Agent] 是否有模具可用？
  ├─ 模具不可用 → 先生产模具（附加时间成本）
  └─ 模具可用 → 直接生产构件

Step 2: [生产Agent] 生产 → 节点1质检
  ├─ 合格 → 入库养护存储
  └─ 不合格 → 回退至生产池，触发重产（+1该类型）

Step 3: [运输Agent] 选择车辆装载 → 运输(8h±1h) → 节点2质检
  ├─ 合格 → 到达现场，进入卸货
  └─ 不合格 → 检查库存：
       ├─ 工厂库存有该类型 → 重装载+重运输
       └─ 工厂库存无该类型 → 触发重产（+1该类型）→ 重运输

Step 4: [施工Agent] 塔吊执行卸货(10min)
  └─ 卸货后：塔吊决策：
       ├─ 选择"直接吊装" → 立即执行吊装 → **H3检验（先检验，通过再完工）**
       │                          ├─ 合格 → 完工池+1 ✓
       │                          └─ 不合格 → 生产池+1，触发重产（+1该类型）+重运输
       └─ 选择"先堆放" → 卸货入库现场存储区 → 后续批次再吊装
```

### 1.4 与原MMARL的关键区别

| 设计维度 | 原MMARL | ERCPSP-MMARL |
|---------|---------|-------------|
| 生产Agent | 1个管理所有生产线 | **N个，每条线=1个Agent** |
| 施工Agent | 1个管理所有塔吊 | **M个，每台塔吊=1个Agent** |
| 订单表示 | 列举P001-P026 | **池化，水位+特征向量** |
| 状态空间 | 含订单编号 | **不含编号，只含池水位** |
| 动作空间 | 具体构件选择 | **类型选择，无编号** |
| 质量机制 | 无 | **三节点质检** |
| 模具管理 | 无 | **模具可用性作为输入状态** |
| 堆放决策 | 无 | **施工Agent可选择先堆放后吊装** |
| 吊装顺序 | 无 | **每批次有对应吊装顺序约束** |
| 训练架构 | 3-Agent MADDPG | **(N+M+1)-Agent MADDPG** |

### 1.5 质量风险三节点

```
生产完成 ──[节点1]──▶ 入库养护存储 ──[运输: 8h±1h]──▶ 到达现场 ──[节点2]──▶ 卸货(10min)
    │                              │                              │
    │ 失败→重产(+1)                  │ 失败→库存检查→重产+重运输        │ 失败→重产+重运输
    ▼                              ▼                              ▼
Production_Pool              Transit_Pool              Arrival_Pool

吊装 ──[节点3]──▶ 完工
          │
          │ 失败→重产+重运输
          ▼
Arrival_Pool

> **关键变化**：运输和施工阶段的质量失败均会触发重产（+1）以及重新运输，形成级联成本。
> **重要**：节点3（H3）质检的完工判断顺序为：**先进行H3检验，通过后再加入完工池**，不通过则不加入完工池，而是触发重产。```

> **关键变化**：运输和施工阶段的质量失败均会触发重产（+1）以及重新运输，形成级联成本。

### 1.6 时间不确定性参数

|| 参数 | 基准值 | 随机波动 |
|------|-------|---------|---------|
| 生产时间 | Type-1=96h, Type-2/3=72h, Type-4=48h, Type-5=36h | ±5% |
| 运输时间 | 8h | ±1h (7~9h均匀分布) |
| 吊装时间 | 特种=0.75h(45min), 常规=0.43h(26min) | ±5min |
| 卸货时间 | 0.167h (10min) | 固定 |

---

## 2. 池化订单设计

### 2.1 池架构

采用**池（Pool）**作为订单和中间件的统一抽象，每个池用类型维度的向量描述。

```
┌─────────────────────────────────────────────────────────────┐
│                    需求池 (Demand Pool)                      │
│         各类型构件待生产数量（水位）                           │
│         [Type-1:1, Type-2:7, Type-3:2, Type-4:12, Type-5:4] │
│         原始订单: 4批共26块预制构件                          │
└─────────────────────────────────────────────────────────────┘
                              │ 生产线Agent决策
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    生产池 (Production Pool)                   │
│         各类型构件待生产数量（随生产动态减少）                  │
│         初始: [1, 7, 2, 12, 4]                             │
└─────────────────────────────────────────────────────────────┘
                              │ 生产线Agent + 质检节点1
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               工厂库存池 (Factory Inventory Pool)              │
│         各类型构件库存数量（通过节点1质检后增加）               │
└─────────────────────────────────────────────────────────────┘
                              │ 运输Agent决策
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    在途池 (Transit Pool)                     │
│         各类型构件在途数量 + 预计到场时间分布                   │
└─────────────────────────────────────────────────────────────┘
                              │ 运输Agent到达 + 质检节点2
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 现场到达池 (Arrival Pool)                      │
│         各类型构件待卸数量（通过节点2质检后增加）             │
└─────────────────────────────────────────────────────────────┘
                              │ 塔吊卸货
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               现场存储池 (Site Storage Pool)                    │
│         各类型构件现场暂存数量（塔吊选择"先堆放"时入库）       │
│         选择直接吊装时跳过此池                                │
└─────────────────────────────────────────────────────────────┘
                              │ 塔吊Agent决策 + 质检节点3
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   完工池 (Completion Pool)                    │
│         各类型构件已完工数量（通过节点3质检后增加）             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 池化表示的优势

| 维度 | 列举订单 | 池化水位 |
|------|---------|---------|
| 状态空间维度 | O(N)，随订单数增长 | O(K)，K=类型数=5，固定 |
| 泛化能力 | 每个订单独立，无法泛化 | 同类型订单共享特征 |
| 计算复杂度 | 指数级增长 | O(K)，线性固定 |
| 协调难度 | 高（订单级依赖） | 低（池级依赖） |
| 决策粒度 | 构件级（过细） | 类型级（适中） |

### 2.3 池的数据结构

```python
class PoolState:
    """池状态基类"""
    levels: np.ndarray      # 各类型构件数量 (5维)
    total: int              # 总数量
    max_level: int          # 最大水位（用于归一化）

class DemandPool(PoolState):
    """需求池 - 初始为4批订单的总量"""
    pass

class ProductionPool(PoolState):
    """生产池 - 待生产构件水位"""
    pass

class FactoryInventoryPool(PoolState):
    """工厂库存池 - 库存持有成本计算基础"""
    pass

class TransitPool(PoolState):
    """在途池 - 运输中构件水位"""
    eta_distribution: np.ndarray  # 各类型预计到场时间

class ArrivalPool(PoolState):
    """到达池 - 等待卸货的构件"""
    waiting_time: float          # 平均等待时间

class SiteStoragePool(PoolState):
    """现场存储池 - 先堆放后吊装的构件"""
    pass

class CompletionPool(PoolState):
    """完工池 - 已完工构件水位"""
    pass
```

---

## 3. 多智能体架构

### 3.1 Agent总览

> **设计原则**：观测空间采用**泛化特征设计**，使用各类型实际需求量归一化，并增加完成率、利用率等特征，使模型不依赖于固定的构件总数。动作空间采用**离散策略选择**，Actor输出策略概率分布（Softmax），由启发式规则执行具体决策。
>
> **动作模式自动选择**：当动作维度 ≤10 时自动使用离散模式（ε-greedy+Softmax）；>10 时使用连续模式（Tanh+OU噪声）。

| Agent类型 | 数量 | 职责 | 观测维度 | 动作维度 | 动作类型 |
|-----------|------|------|---------|---------|---------|
| Production Agent | N_prod（每条线1个） | 生产线排程+节点1质检+模具状态 | 48 | **5**（等待/最小生产成本/最小库存/最小总成本/抽检决策） | 离散 |
| Transport Agent | 1 | 装载+发车+节点2质检+压车成本优化 | **82**（含10辆×4=40维车队感知） | **6**（等待/选新车/最小总车数/最小压车/最小压车+新车/抽检决策） | 离散 |
| Construction Agent | N_const（每台塔吊1个） | 卸货+安装+节点3质检+堆放决策 | 43+N_const | **5**（等待/直接吊装/先到场车辆/最先到场/抽检决策） | 离散 |

### 3.2 离散策略动作空间

各Agent的动作空间为**离散策略选择**，Actor输出各策略的Softmax概率分布，由对应的启发式规则执行具体决策。

#### 3.2.1 生产智能体动作空间 A_prod

```
A^prod = {a_0, a_1, a_2, a_3, a_4}

a_0: 等待不投产
     - 生产线保持空闲，不启动任何生产
     - 质检触发: 无

a_1: 按最小生产成本（最少换模次数）优先生产
     - 选择生产成本最低的类型（生产价格 + 换模成本）
     - 质检触发: 启动节点1抽检

a_2: 按最小库存优先生产
     - 选择当前工厂库存最低的类型
     - 质检触发: 启动节点1抽检

a_3: 按最小（生产成本与库存成本之和）优先生产
     - 选择（生产价格 + 换模成本 + 库存持有成本）之和最低的类型
     - 质检触发: 启动节点1抽检

a_4: 抽检决策
     - 选择最需要质量检验的类型（库存最高且未通过抽检的类型优先）
     - 触发节点1抽检，检验数量=1，失败概率=ε₁×1/n
     - 检验成本=5%×1×累计成本

Actor输出: Softmax概率分布 [P(a_0), P(a_1), P(a_2), P(a_3), P(a_4)]
```

#### 3.2.2 运输智能体动作空间 A_trans

```
A^trans = {b_0, b_1, b_2, b_3, b_4, b_5}

b_0: 等待
     - 不执行装载和发车操作

b_1: 选新车
     - 选择工厂中空闲车辆（共10辆）
     - 根据启发式规则装载

b_2: 按最小总车数发运
     - 选择装载量最大的方案（减少发车次数）

b_3: 按最小压车发运
     - 选择能使最早等待车辆满载的方案

b_4: 按最小（压车成本+新车投入）之和发运
     - 选择（等待时间×压车费率 + 新增车辆成本）之和最小的方案
     - 工厂有空闲车时新车投入为0，无空闲车时新增一辆

b_5: 抽检决策
     - 触发节点2抽检，检验数量=1，失败概率=ε₃×1/n
     - 检验成本=5%×1×累计成本

装载方案（由 b_1~b_4 触发）:
  方案1: 特种构件1块
  方案2: Type-4×1 + Type-5×2
  方案3: Type-4×2
  方案4: Type-5×6

Actor输出: Softmax概率分布 [P(b_0), P(b_1), P(b_2), P(b_3), P(b_4), P(b_5)]
```

#### 3.2.3 施工智能体动作空间 A_cons

```
A^cons = {c_0, c_1, c_2, c_3, c_4}

c_0: 等待
     - 塔吊保持空闲

c_1: 直接吊装
     - 选择到达池中的构件，优先顺序：Type-1 > Type-2 > ... > Type-5
     - 立即执行卸货+吊装，触发节点3质检
     - 完工构件加入完工池

c_2: 优先吊装先到场车辆上的预制件
     - 找出到达时间最早的车辆
     - 吊装该车辆上的构件类型
     - 若到达池为空，先卸货（入库现场存储区）后吊装

c_3: 优先吊装最先到场的预制件
     - 选择到达池中数量最多的类型（近似先到场）
     - 若到达池为空，先卸货后吊装

c_4: 抽检决策
     - 触发节点3抽检，选择最需要检验的构件类型（到达池/现场存储池中数量最多的类型优先）
     - 检验数量=1，失败概率=ε₄×1/n
     - 检验成本=5%×1×累计成本

注：c_2/c_3中"先卸货"指将构件存入现场存储池（site_storage_pool），后续再执行吊装。

Actor输出: Softmax概率分布 [P(c_0), P(c_1), P(c_2), P(c_3), P(c_4)]
```

### 3.3 生产智能体群 (Production Fleet)

**设计原则：每条生产线 = 1个独立的Production Agent**

#### 观测空间 O_prod[i]（48维）

```
O_prod[i] = [
    # ===== 生产线自身状态 =====
    self_line_status,        # 自身生产线状态 (1维): 0=占用, 1=空闲
    self_line_remaining,     # 自身生产线剩余时间 (1维)

    # ===== 模具可用性 =====
    mold_available,          # 模具可用性 (5维): 各类型模具是否可用 {0,1}

    # ===== 泛化池状态（typedemand归一化）=====
    demand_pool,             # 需求池 (5维): 各类型剩余需求 / 各类型总需求
    production_pool,         # 生产池 (5维): 各类型待生产数 / 各类型总需求
    factory_inventory,       # 工厂库存池 (5维): 各类型库存 / 各类型总需求

    # ===== 泛化特征 =====
    completion_rate,         # 各类型完成率 (5维): completion_pool[i] / total_demand[i]
    qc_pass_rate,           # 全局质检通过率 (1维): pass / (pass + fail)
    pool_utilization,        # 各池容量利用率 (7维): 各池总量 / max_level
                              # 顺序: demand, production, factory_inv, transit, arrival, completion, site_storage
    storage_cost_rate,       # 库存持有成本率 (2维): [factory_rate, site_rate]
    bottleneck_type,         # 瓶颈类型标识 (5维): one-hot编码（占用量最大的类型）

    # ===== 质检状态 =====
    qc_node1_status,         # 节点1质检状态 (5维): 各类型是否已抽检

    # ===== 时间 =====
    time_normalized,         # 时间归一化 (1维): time / max_steps
]  # 总计: 2+5+5+5+5+5+1+7+2+5+5+1 = 48维
```

#### 动作空间 A_prod[i]（4维，离散策略）

```
A_prod[i] = [
    strategy,  # 策略索引 ∈ {0, 1, 2, 3} → {等待, 最小生产成本, 最小库存, 最小总成本}
]  # 输出: 标量索引（Softmax概率分布）
```

#### 动作解码策略（含模具可用性检查）

```python
def decode_action(strategy: int, state: ERCPSPState, line) -> Dict:
    if strategy == 0:
        return {'type_idx': -1, 'qc_trigger': False}  # 等待
    elif strategy == 1:
        type_idx = select_by_min_production_cost(state, line)
        return {'type_idx': type_idx, 'qc_trigger': True}
    elif strategy == 2:
        type_idx = select_by_min_inventory(state, line)
        return {'type_idx': type_idx, 'qc_trigger': True}
    elif strategy == 3:
        type_idx = select_by_min_total_cost(state, line)
        return {'type_idx': type_idx, 'qc_trigger': True}
```

#### 有效动作约束

- 自身生产线空闲时才可选择生产
- 该类型模具已就绪，或需要先生产模具（附加时间成本）
- 仅当该类型在生产池中数量>0时才可选

---

### 3.3 运输智能体 (Transport Agent)

#### 观测空间 O_trans（54维）

```
O_trans = [
    # ===== 工厂库存池（typedemand归一化）=====
    factory_inventory,    # 工厂库存池 (5维): 各类型库存 / 各类型总需求

    # ===== 车辆状态（车队感知）=====
    vehicle_fleet_status, # 车队状态 (N_vehicles×4维): per vehicle [location_onehot×3 + load_ratio×1]
                          # location: FACTORY=0, TRANSIT=1, SITE=2

    # ===== 池状态（typedemand归一化）=====
    transit_pool,          # 在途池 (5维): 各类型在途数量 / 各类型总需求
    arrival_pool,          # 到达池 (5维): 各类型待卸数量 / 各类型总需求

    # ===== 泛化特征 =====
    completion_rate,        # 各类型完成率 (5维)
    qc_pass_rate,          # 全局质检通过率 (1维)
    pool_utilization,      # 各池容量利用率 (7维)
    storage_cost_rate,     # 库存持有成本率 (2维): [factory_rate, site_rate]
    bottleneck_type,       # 瓶颈类型标识 (5维): one-hot编码

    # ===== 塔吊和质检状态 =====
    crane_fleet_status,    # 塔吊群可用性 (N_const维): 每台塔吊 0=占用, 1=空闲
    qc_node2_status,       # 节点2质检状态 (5维): 各类型是否已抽检

    # ===== 时间 =====
    time_normalized,       # 时间归一化 (1维): time / max_steps
]
# 总计: 5 + 12(3×N_vehicles=3×4) + 5 + 5 + 5 + 1 + 7 + 2 + 5 + 1 + 5 + 1 = 54维
```

#### 动作空间 A_trans（5维，离散策略）

```
A_trans = [
    strategy,  # 策略索引 ∈ {0, 1, 2, 3, 4} → {等待, 选新车, 最小总车数, 最小压车, 最小压车+新车}
]  # 输出: 标量索引（Softmax概率分布 → ε-greedy选择）
```

#### 装载方案触发

b_1~b_4 均触发装载操作，从以下方案中选择（按动作类型决定装载优化目标）：

| 方案 | 特种构件 | Type-4 | Type-5 | 最大装载 |
|------|---------|--------|--------|---------|
| 方案1 | 1块 | 0 | 0 | 1 |
| 方案2 | 0 | 1 | 2 | 3 |
| 方案3 | 0 | 2 | 0 | 2 |
| 方案4 | 0 | 0 | 6 | 6 |

---

### 3.4 施工智能体群 (Construction Fleet)

**设计原则：每台塔吊 = 1个独立的Construction Agent**

#### 观测空间 O_const[j]（43+N_const 维）

```
O_const[j] = [
    # ===== 塔吊自身状态 =====
    crane_status[j],       # 自身塔吊状态 (1维): 1=空闲, 0=占用
    crane_remaining[j],    # 自身塔吊剩余时间 (1维): remaining / 1.0, 上限1.0

    # ===== 到达池（typedemand归一化）=====
    arrival_pool,          # 到达池 (5维): 各类型待安装数量 / 各类型总需求

    # ===== 现场存储池（typedemand归一化）=====
    site_storage,          # 现场存储池 (5维): 各类型在场内数量 / 各类型总需求

    # ===== 泛化特征 =====
    completion_rate,       # 各类型完成率 (5维)
    qc_pass_rate,         # 全局质检通过率 (1维)
    pool_utilization,     # 各池容量利用率 (7维)
    storage_cost_rate,    # 库存持有成本率 (2维): [factory_rate, site_rate]
    bottleneck_type,      # 瓶颈类型标识 (5维): one-hot编码

    # ===== 节点3质检状态 =====
    qc_node3_status,      # 节点3质检状态 (5维): 各类型是否已抽检

    # ===== 其他塔吊状态 =====
    other_crane_status,   # 其他塔吊状态 (N_const-1维): 无其他塔吊时=0维

    # ===== 在途池（typedemand归一化）=====
    transit_pool,          # 在途池 (5维): 各类型在途数量 / 各类型总需求

    # ===== 时间 =====
    time_normalized,       # 时间归一化 (1维)
]
# 总计: 2 + 5 + 5 + 5 + 1 + 7 + 2 + 5 + 5 + (N_const-1) + 5 + 1 = 43+N_const 维
# 默认配置(N_prod=2, N_const=1, N_veh=3): 48维观测
```

#### 动作空间 A_const[j]（4维，离散策略）

```
A_const[j] = [
    strategy,  # 策略索引 ∈ {0, 1, 2, 3} → {等待, 直接吊装, 先到场车辆, 最先到场}
]  # 输出: 标量索引（Softmax概率分布 → ε-greedy选择）
```

**堆放决策语义**：
- `c_1`（直接吊装）：构件到达后立即执行吊装，触发节点3质检
- `c_2`（先到场车辆）：找出到达时间最早的车辆，吊装该车辆上的构件；若到达池为空则先卸货后吊装
- `c_3`（最先到场）：选择到达池中数量最多的类型；若到达池为空则先卸货后吊装

#### 多施工Agent协同机制

- **环境级互斥**：塔吊资源通过环境约束强制互斥
- **中心化Critic协调**：所有施工Agent的观测+动作输入共享Critic
- **共享到达池**：所有施工Agent观测相同的到达池，池水位作为协调信号

---

## 4. MADDPG算法设计

### 4.1 算法概述

MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 采用**中心化训练 + 去中心化执行**范式：

- **中心化Critic**：训练时使用所有Agent的观测和动作，学习全局Q值
- **去中心化Actor**：执行时每个Agent仅使用自身观测输出动作
- **同质子群参数共享**：两条生产线（prod_0, prod_1）共享同一 Actor 网络和优化器；多台塔吊共享另一 Actor 网络；运输Agent使用独立网络

### 4.2 网络架构

#### Actor网络（同质子群参数共享）

```
输入: 观测向量 o_i
  │
  ▼
全连接层 (obs_dim → 256) → ReLU → LayerNorm
  │
  ▼
全连接层 (256 → 256) → ReLU → LayerNorm
  │
  ▼
全连接层 (256 → 128) → ReLU → LayerNorm
  │
  ▼
全连接层 (128 → action_dim) → [Softmax(离散) | Tanh(连续)]
  │
  ▼
输出: 动作向量 a_i（离散=one-hot索引，连续=[-1,1]^action_dim）
```

> **同质子群参数共享**：生产群（prod_0, prod_1）和施工群（const_0, ...）各自共享一个 Actor 网络实例。两条生产线使用完全相同的网络权重，梯度在共享优化器中累加后统一更新。运输 Agent 使用独立网络。
>
> **动作模式自动选择**：当 `action_dim ≤ 10` 时使用离散模式（Softmax输出→ε-greedy选择）；`action_dim > 10` 时使用连续模式（Tanh输出→OU噪声探索）。本项目所有Agent动作维度均为4~6，全部为离散模式。

#### 中心化Critic网络（全局唯一）

```
输入1: 所有Agent观测拼接
  [o_prod[0] || ... || o_prod[N-1] || o_trans || o_const[0] || ...]
  
输入2: 所有Agent动作拼接
  [a_prod[0] || ... || a_trans || a_const[0] || ...]
  │
  ▼
全连接层 (total_obs + total_action → 512) → ReLU → LayerNorm
  │
  ▼
全连接层 (512 → 256) → ReLU → LayerNorm
  │
  ▼
全连接层 (256 → 128) → ReLU → LayerNorm
  │
  ▼
全连接层 (128 → 1)
  │
  ▼
输出: 联合Q值 Q(s, a_1, ..., a_N)
```

### 4.3 核心公式

#### 策略梯度（离散动作，代码级实现）

$$
\nabla_{\theta_i} J \approx \mathbb{E}_{s,a\sim D}\left[ \nabla_{\theta_i} \mu(o_i|\theta_i) \cdot \nabla_{a_i} Q(s, a_1,...,a_N|\phi) \bigg|_{a_i=\mu(o_i)} \right]
$$

**代码级实现**（离散动作，Actor输出 Softmax 概率）：

```python
# 获取当前策略的Softmax概率（可微）
action_logits = actor(obs_tensor)           # shape: (batch, action_dim)
probs = F.softmax(action_logits, dim=-1)    # shape: (batch, action_dim)

# 用当前策略概率替换第i个agent的动作，输入Critic（其他agent用replay buffer中的动作）
alt_actions = list(old_actions_onehot)
alt_actions[i] = probs                      # 第i个agent用策略概率
q_val = critic(obs_tensors, alt_actions)     # Q(s, π(a_i), a_{-i})

# 策略梯度 = -E[Q]（梯度下降最大化Q）
loss_i = -q_val.mean()
loss_i.backward(retain_graph=(i < n-1))      # 保留计算图直到最后一个agent
```

#### Q值更新（Target网络）

$$
y_i = r_{norm} + \gamma (1 - d) \cdot Q'(s', a'_1,...,a'_N|\phi'_i)
$$

其中 $r_{norm} = (r - \mu_r) / (\sigma_r + 10^{-8})$ 为奖励归一化。

#### 软更新（Polyak更新）

$$
\theta^{target} \leftarrow \tau \cdot \theta^{source} + (1 - \tau) \cdot \theta^{target}
$$

**代码级实现**：

```python
def soft_update(target_net, source_net, tau=0.005):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
```

#### 同质子群梯度累加（生产群示例）

两条生产线共享同一 Actor 网络，梯度在共享优化器中累加：

```python
# prod_0 和 prod_1 共用同一个 actor_net 和 actor_opt
loss_0.backward()   # 梯度存入共享优化器
loss_1.backward()   # 梯度累加到同一优化器
actor_opt.step()    # 一次更新同时影响两个智能体
```

### 4.4 维度计算示例

配置：2条生产线 + 3辆运输车 + 1台塔吊

| 网络 | 观测维度 | 动作维度 | 计算说明 |
|------|---------|---------|---------|
| Prod_Agent[0] | 48 | **5** | 2(自身)+5(模具)+5(需求)+5(生产)+5(工厂库存)+5(完成率)+1(QC通过率)+7(池利用率)+2(库存成本率)+5(瓶颈)+5(QC)+1(时间) |
| Prod_Agent[1] | 48 | **5** | 同上 |
| Transport_Agent | **82** | **6** | 5(工厂库存)+40(10×4车辆)+5(在途)+5(到达)+5(完成率)+1(QC通过率)+7(池利用率)+2(库存成本率)+5(瓶颈)+1(塔吊)+5(QC)+1(时间) |
| Const_Agent[0] | 43 | **5** | 2(自身)+5(到达)+5(存储)+5(完成率)+1(QC通过率)+7(池利用率)+2(库存成本率)+5(瓶颈)+5(QC)+0(其他塔吊)+5(在途)+1(时间) |
| **中心化Critic观测输入** | **48+48+82+43=221** | - | 4个Agent观测拼接（Transport由54→82因车辆数增加） |
| **中心化Critic动作输入** | - | **5+5+6+5=21** | 4个Agent离散动作one-hot编码拼接（21维，prod_0+prod_1各5维，transport=6维，const_0=5维） |

> **说明**：
> - 实际维度因生产线数N_prod、塔吊数N_const、车辆数N_veh配置而变化
> - 泛化设计使模型可泛化至不同构件总数的场景
> - 所有Agent均为离散动作（action_dim ≤ 10），Critic输入为one-hot编码后的动作向量
> - ε=0.1（离散模式探索率），Actor输出Softmax概率分布

### 4.5 探索策略

采用 **ε-greedy 探索**（离散动作）和 **Ornstein-Uhlenbeck (OU) 噪声**（连续动作）：

```python
if action_mode == 'discrete':
    # ε-greedy 探索（所有Agent动作维度≤10，均为离散模式）
    if random() < epsilon:
        action = random_choice(action_dim)  # 随机探索
    else:
        action = argmax(policy_probs)  # 利用
else:
    # OU噪声探索（连续动作，当前项目不启用）
    noise = OUNoise.sample()
    action = clip(policy_output + noise_scale * noise, -1, 1)
```

参数：ε=0.1（离散模式探索率），OU噪声 θ=0.15, σ=0.2，衰减率=0.999（本项目所有Agent均为离散模式，OU噪声不启用）。

### 4.6 训练流程

```
MADDPG 训练循环（动作维度≤10自动进入离散模式）
---------------------------------------------------------------------
输入：训练参数（episodes=500, max_steps=2000, batch_size=256, lr_actor=1e-4, lr_critic=3e-4, gamma=0.95, tau=0.005）
---------------------------------------------------------------------

for episode = 1 to N:
    重置环境
    重置噪声（OU噪声/ε计数器）

    for step = 1 to max_steps:
        每个Agent独立获取观测 o_i
        每个Agent的Actor输出策略概率分布
            action_dim ≤ 10: Softmax -> ε-greedy选择动作索引（训练ε=0.1，评估ε=0）
            action_dim > 10: Tanh -> OU噪声扰动
        拼接所有观测+动作（离散动作one-hot编码），输入中心化Critic
        执行动作，获取(s', r, done)
        存储经验到ReplayBuffer（容量100000，batch_size=256）

        if 经验足够 (warmup=1000步后开始更新):
            从ReplayBuffer采样经验
            奖励归一化（零均值、单位方差）
            Critic: TD误差更新 Q网络（梯度裁剪max_norm=0.5）
            Actor: 策略梯度更新（离散=softmax概率引导，连续=DDPG）
            Polyak软更新目标网络（tau=0.005）

    if episode % eval_interval == 0:
        评估策略（确定性动作，无探索，ε=0）
        记录奖励、完成数、Q值、成本分解

    if episode % save_interval == 0:
        保存检查点
```

### 4.7 同质子群参数共享

本项目包含两组异质智能体群：

|| 智能体群 | 成员 | 同质子群 | 共享策略 |
||---------|------|---------|---------|
|| 生产群 | prod_0, prod_1 | **是**（两条生产线同构） | 共享同一Actor网络、同一个优化器 |
|| 施工群 | const_0, ... | **是**（多台塔吊同构） | 共享同一Actor网络、同一个优化器 |
|| 运输群 | transport | 否（单一Agent） | 独立网络 |

#### 4.7.1 生产智能体同质子群（两条生产线）

`maddpg_trainer.py` 中，通过以下方式实现参数共享：

```python
# 识别同质子群
self.prod_names = [n for n in self.sorted_names if n.startswith('prod_')]

# 创建一个共享网络实例
self._shared_prod_net = ActorNetwork(
    prod_obs_d, prod_act_d,
    action_mode='discrete', qc_action_dim=2, device=self.device
)
shared_prod_target = ActorNetwork(...)
shared_prod_target.load_state_dict(self._shared_prod_net.state_dict())
shared_prod_opt = optim.Adam(self._shared_prod_net.parameters(), lr=1e-4)

# 让 prod_0 / prod_1 都指向同一个网络实例（不复制参数，只共享引用）
for name in self.prod_names:
    self.actors[name] = self._shared_prod_net
    self.actors_target[name] = shared_prod_target
    self.actor_optimizers[name] = shared_prod_opt
```

**效果**：
- 两条生产线使用完全相同的网络权重
- 梯度在共享优化器中**累加**：两个智能体各自计算梯度后，累加到同一个Adam优化器，一次 `step()` 同时更新
- 训练时：`loss_0.backward()` + `loss_1.backward()` → 共享优化器一次 `step()`
- 软更新：只对共享目标网络执行一次 Polyak 更新

#### 4.7.2 施工智能体同质子群（多台塔吊）

施工群采用相同的参数共享模式：

```python
self.const_names = [n for n in self.sorted_names if n.startswith('const_')]
# 创建 shared_const_net，所有 const_i 指向同一网络实例
```

#### 4.7.3 异质智能体（运输Agent）

运输智能体 `transport` 是唯一智能体，无同质子群，始终使用独立网络。

#### 4.7.4 训练阶段梯度累积逻辑

```python
# 先对所有独立优化器做zero_grad；共享网络只做一次
_seen_opt_ids = set()
for name in self.sorted_names:
    opt_id = id(self.actor_optimizers[name])
    if opt_id not in _seen_opt_ids:
        self.actor_optimizers[name].zero_grad()
        _seen_opt_ids.add(opt_id)

# 遍历每个agent计算loss并反向传播
for i, name in enumerate(self.sorted_names):
    actor = self.actors[name]
    # ... 计算 loss_i ...
    # retain_graph：下一个agent如果不是共享网络则retain_graph=False
    retain = (self.actors.get(self.sorted_names[i+1]) is actor) if i < n-1 else False
    loss_i.backward(retain_graph=retain)

# 每个独立优化器只做一次step；共享网络只做一次
_seen_opt_ids = set()
for name in self.sorted_names:
    opt_id = id(self.actor_optimizers[name])
    if opt_id not in _seen_opt_ids:
        torch.nn.utils.clip_grad_norm_(self.actors[name].parameters(), 0.5)
        self.actor_optimizers[name].step()
        _seen_opt_ids.add(opt_id)

# 软更新：每个共享组只更新一次
_seen_target_ids = set()
for name in self.sorted_names:
    target_id = id(self.actors_target[name])
    if target_id not in _seen_target_ids:
        self._soft_update(self.actors[name], self.actors_target[name])
        _seen_target_ids.add(target_id)
```

#### 4.7.5 策略梯度推导

同质子群的策略梯度等价于两个智能体各自策略梯度的**加和**：

$$
\nabla_{\theta} J \approx \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \mathbb{E}_{s \sim D}\left[ \nabla_{\theta} \mu(o_i|\theta) \cdot \nabla_{a_i} Q(s, a_1,...,a_N|\phi) \bigg|_{a_i=\mu(o_i)} \right]
$$

其中 $\mathcal{S}$ 为同质子群（prod = {prod_0, prod_1}）。共享优化器实际执行的是两个梯度向量的加权平均（由Adam自适应学习率调制）。

### 4.8 Loss计算公式与训练细节

#### 4.8.1 Critic Loss（全局Q值网络）

$$
\mathcal{L}_{critic} = \mathbb{E}_{s,a,r,s'}\left[ (Q(s, a_1,...,a_N|\phi) - y)^2 \right]
$$

其中 TD target：

$$
y = r_{norm} + \gamma (1 - d) \cdot Q'(s', a'_1,...,a'_N|\phi')
$$

- $r_{norm}$：奖励归一化（零均值，单位方差）
- $Q'$：目标Critic网络（Polyak平均）
- $d$：done标志

#### 4.8.2 Actor Policy Loss（策略梯度）

对于离散动作空间，每个智能体的策略梯度为：

$$
\mathcal{L}_{actor,i} = -\mathbb{E}_{s \sim D}\left[ Q(s, a_1,...,a_N|\phi) \right]
$$

其中 $a_i$ 用当前策略的 Softmax 概率分布代入 Critic（其他智能体动作用 replay buffer 中保存的动作），通过 `loss_i.backward(retain_graph=True)` 逐个智能体反向传播。

**同质子群均值**：

$$
\overline{\mathcal{L}}_{actor}^{prod} = \frac{1}{|\mathcal{S}_{prod}|} \sum_{i \in \mathcal{S}_{prod}} \mathcal{L}_{actor,i}
$$

#### 4.8.3 Soft Update（Polyak软更新）

$$
\theta^{target} \leftarrow \tau \cdot \theta^{source} + (1 - \tau) \cdot \theta^{target}, \quad \tau = 0.005
$$

#### 4.8.4 WandB日志记录

每个 episode 结束时记录以下 loss 指标到 WandB：

|| WandB 键 | 说明 |
|---------|---------|
| `train/critic_loss` | 全局Critic loss |
| `train/policy_loss_prod_mean` | 生产子群policy loss均值 |
| `train/policy_loss_transport` | 运输Agent policy loss |
| `train/policy_loss_const_mean` | 施工子群policy loss均值 |
| `train/policy_loss_prod_0` | prod_0独立policy loss（调试用） |
| `train/policy_loss_prod_1` | prod_1独立policy loss（调试用） |
| `q/q_prod_mean` | 生产子群Q值均值 |
| `q/q_transport` | 运输Agent Q值 |
| `q/q_const_mean` | 施工子群Q值均值 |
| `q/prod_0`, `q/prod_1`, ... | 各Agent独立Q值（调试用） |

---

## 5. 质量风险机制

### 5.1 三节点质检概述

| 质检节点 | 触发时机 | 负责Agent | 失败概率 | 检验成本 | 失败后果 | 是否重生产 | 通过奖励 | 失败惩罚 |
|---------|---------|----------|---------|---------|---------|-----------|---------|---------|
| 节点1 | 生产完成入库 | Production Fleet | ε₁×x/n（x=已检验数） | 5%×n×累计成本 | 回退至Production Pool | **是**（+1该类型） | +20元/件 | 累计成本 |
| 节点2 | 运输到达现场 | Transport Agent | ε₃×x/n | 5%×n×累计成本 | 检查库存→重运输 | **是**（+1该类型） | +30元/件 | 累计成本 |
| 节点3 | 吊装完成 | Construction Fleet | ε₄×x/n | 5%×n×累计成本 | 重吊装+重运输 | **是**（+1该类型） | +300元/件 | 累计成本 |

> **累计成本说明**：生产失败惩罚=（生产价格+模具成本）；运输失败惩罚=基础+（工厂存储成本+运输成本）；施工失败惩罚=基础+（工厂存储+运输+现场存储+吊装成本）。

### 5.2 质量检验机制（按logic_fix.ini重写）

每个节点采用**可变数量抽检**机制（由Agent决策是否触发检验）：

```
检验数量范围: [1, type_demand_i]  （最少1件，最多该类型总需求）

检验成本 = 5% × n × 累计成本（n=检验数量）

失败概率 = ε × x / n
  - x: 已完成检验的该类型构件数量
  - n: 该类型总需求量
  - ε: 基础缺陷率（节点1=5%, 节点2=3%, 节点3=2%）

跳过检验: 该阶段失败概率累计到下一节点
  - 例如：生产阶段未检验（ε₁不降低）→ 运输阶段概率=ε₁+ε₃
  - 全程未检验：ε_total = ε₁ + ε₃ + ε₄ = 10%

级联失败处理:
  - 节点1失败 → production_pool +1（重产）
  - 节点2失败 → 检查factory_inventory：
      - 有库存 → 重装载+重运输
      - 无库存 → production_pool +1（重产）→ 重运输
  - 节点3失败 → production_pool +1（重产）→ 重运输
```

### 5.3 质量状态转移

```
Pool_Demand ──[生产决策]──▶ Production_Pool
                                 │
                                 │ 生产完成
                                 ▼
                         [节点1质检: 概率ε₁×x/n]
                                 │
                    ┌────────────┴────────────┐
                    │ 合格                      │ 不合格
                    ▼                          ▼
            Factory_Inventory_Pool      Production_Pool (+1该类型→重产)
                    │
                    │ 运输决策+发车
                    ▼
              Transit_Pool
                    │
                    │ 运输到达
                    ▼
            [节点2质检: 概率ε₃×x/n]
                    │
           ┌────────┴────────┐
           │ 合格              │ 不合格
           ▼                  ▼
      Arrival_Pool      [检查factory_inventory]
                              ├─ 有库存 → 重装载+重运输
                              └─ 无库存 → Production_Pool(+1)→重生产→重运输
           │
           │ 塔吊卸货
           ▼
      [节点3质检: 概率ε₄×x/n  取决于施工决策]
           │
    ┌──────┴──────┐
    │             │
 ▼卸载入库         ▼直接吊装
Site_Storage_Pool  [节点3: 先检验，通过才加入完工池]
 (无QC)                 │
                  ┌─────┴─────┐
                  │ 合格       │ 不合格
                  ▼            ▼
           Completion    Production_Pool(+count-pass_count)→重产→重运输
           (仅通过时)     （不加入完工池）
```

**说明**：
- **节点1失败**：生产池+1（重产），原生产构件作废
- **节点2失败**：检查工厂库存，有则重装载+重运输；无则生产池+1（重产）后重运输，原运输构件作废
- **节点3失败**：生产池+1（重产）后重运输，原施工构件作废（需重吊装）
- **跳过检验**：该阶段失败概率不降低，累计到后续节点
- 选择"先堆放"：构件进入现场存储池（无质检），后续再执行吊装

### 5.4 吊装顺序约束

每批次预制构件均对应有吊装顺序要求，即构件需按特定顺序安装。以下为批次吊装顺序示例（具体顺序由输入数据确定）：

```
批次1吊装顺序: Type-2(1) → Type-2(2) → Type-1 → Type-2(3) → ...
批次2吊装顺序: Type-4(1) → Type-3 → Type-4(2) → Type-4(3) → ...
批次3吊装顺序: Type-5(1) → Type-5(2) → Type-5(3) → Type-5(4)
批次4吊装顺序: Type-4(1) → Type-4(2) → Type-3 → Type-4(3) → ...
```

**状态空间中的吊装顺序表示**：
- 吊装顺序不直接作为状态维度（保持池化设计的简洁性）
- 当前待安装构件的顺序优先级通过**到达池中的构件类型和顺序标记**隐式编码
- 施工Agent可选择先堆放（不立即安装），从而保留后续按顺序安装的灵活性

**堆放决策的作用**：当某类型构件已到达但顺序上暂不需要安装时，施工Agent选择"先卸货后吊装"，将构件存入现场存储区，待顺序到达时再执行吊装。

---

## 6. 奖励函数设计

### 6.1 奖励计算结构（按fixfix.ini重构）

奖励函数的设计目标：**使总成本最小化**。

每个智能体的奖励函数设计为对应成本项的负值，激励各智能体最小化其可控成本：

$$
R_{total} = w_1 \cdot R_{prod} + w_2 \cdot R_{trans} + w_3 \cdot R_{const} - C_{quality} - C_{qc} - C_{time} + R_{progress} + R_{cumulative}
$$

其中 $w_1, w_2, w_3$ 为**可学习权重**，通过 Adam 算法自动优化；各智能体奖励函数为：

$$
R_{prod} = -(C_{mold} + C_{production} + C_{factory\_storage} + C_{transport\_by\_prod})
$$

$$
R_{trans} = -(C_{transport} + C_{waiting})
$$

$$
R_{const} = -(C_{waiting\_by\_cons} + C_{site\_storage} + C_{construction})
$$

各智能体分项奖励在传入策略网络前做 **裁剪（clipping）**，裁剪范围 $[-\text{clip\_bound}, +\text{clip\_bound}]$，$\text{clip\_bound}=10.0$。

### 6.2 各智能体成本项（每步计算）

#### 6.2.1 生产智能体成本

| 成本项 | 计算公式 | 说明 |
|--------|---------|------|
| 模具成本 | \$\sum(\text{模具数量} \times \text{模具种类} \times \text{模具单价})\$ | 每次换模时计入 |
| 预制构件生产成本 | \$\sum(\text{生产数量} \times \text{生产种类} \times \text{生产单价})\$ | 每次生产入库时计入 |
| 工厂仓储成本 | \$\text{工厂库存体积} \times \Delta t \times 0.5\$ | 元/小时/m³ |
| 运输成本（生产端） | \ \times \text{车次}\$ | 每次工厂发车时计入 |

**生产智能体奖励 = 负成本合计**：

\$\$
R_{prod} = -(C_{mold} + C_{production} + C_{factory\\_storage} + C_{transport\\_by\\_prod})
\$\$

#### 6.2.2 运输智能体成本

| 成本项 | 计算公式 | 说明 |
|--------|---------|------|
| 运输成本 | \ \times \text{累计车次}\$ | 到达工厂→现场的构件数×单价 |
| 压车等待成本 | \$\sum \max(\text{waiting\\_time} - 1.0, 0) \times 36\$ | 元/小时/辆 |

**运输智能体奖励 = 负成本合计**：

\$\$
R_{trans} = -(C_{transport} + C_{waiting})
\$\$

#### 6.2.3 施工智能体成本

| 成本项 | 计算公式 | 说明 |
|--------|---------|------|
| 压车等待成本 | \$\sum \max(\text{waiting\\_time} - 1.0, 0) \times 36\$ | 元/小时/辆（车辆在施工现场等待） |
| 现场仓储成本 | \$\text{现场存储体积} \times \Delta t \times 1.0\$ | 元/小时/m³ |
| 施工成本 | \$\text{塔吊使用时间} \times 2800\$ | 元/小时 |

**施工智能体奖励 = 负成本合计**：

\$\$
R_{const} = -(C_{waiting\\_by\\_cons} + C_{site\\_storage} + C_{construction})
\$\$

#### 6.2.4 共享成本（质量相关）

| 成本项 | 计算公式 | 说明 |
|--------|---------|------|
| 质量惩罚 | 累计成本惩罚（按阶段） | Node1/2/3失败时根据失败节点计算 |
| 质量检验成本 | \$\Delta \text{inspected\\_count} \times 5\% \times \text{cumulative\\_cost}\$ | 每次检验时计入 |
| 时间惩罚 | 仅超期时 \$\Delta t \times 20 \times 0.01\$ | 权重降低 |

#### 6.2.5 **完成进度奖励（新增：引导信号，解决冷启动困境）**

为解决MADDPG在随机初始化后缺乏成功样本导致的冷启动问题，新增两项进度奖励，为策略网络提供**每步中间引导信号**：

| 奖励项 | 计算公式 | 说明 | 作用 |
|--------|---------|------|------|
| **每步完成奖励** | $\Delta \text{completed} \times 500$ | 每新完成1件构件奖励+500元 | 激励生产-运输-施工链高效协作 |
| **累计完成比例奖励** | $\frac{\text{completed\_count}}{\text{n\_components}} \times 500$ | 当前完成率越高奖励越高 | 持续引导向完成更多构件的方向 |

> **设计原理**：原奖励函数仅基于成本（负值），在初期探索阶段几乎没有正向信号引导智能体学习。进度奖励在每步提供**密集的中间奖励**，使策略网络能够在有限的成功 episode 中快速学到"完成更多构件 = 更高奖励"的因果关系。

### 6.3 可学习奖励权重与总奖励（核心设计）

三个智能体族（生产/运输/施工）各自拥有独立的成本最小化奖励，通过可学习权重加权求和得到总奖励，加上新增的**完成进度奖励**：

$$
R_{total} = w_1 \cdot R_{prod} + w_2 \cdot R_{trans} + w_3 \cdot R_{const} - C_{quality} - C_{qc} - C_{time} + R_{progress} + R_{cumulative}
$$

**权重优化**：$w_1, w_2, w_3$ 为可学习参数（初始均匀值 $1/3$），通过 Adam 算法在每个 episode 结束时自动优化：

1. 根据当前权重计算总奖励 $R_{total}$
2. 计算损失函数：$\mathcal{L} = -R_{total}$（最大化奖励）
3. 反向传播更新权重
4. 通过 softmax 归一化得到最终权重分布

**奖励裁剪（clipping）**：每个智能体族的分项奖励在传入 Critic 之前做裁剪：

$$
R_{X}^{clipped} = \text{clip}(R_X, [-\text{clip\_bound}, +\text{clip\_bound}])
$$

其中 $\text{clip\_bound} = 10.0$，防止极端奖励信号干扰训练。

### 6.4 质量惩罚累计成本计算

质量惩罚按失败节点计算**累计到该节点的所有成本**：

| 节点 | 失败惩罚计算公式 | 说明 |
|------|----------------|------|
| 节点1（生产） | $-(\text{生产价格} + \text{模具成本})$ | 仅包含生产阶段成本 |
| 节点2（运输） | $-(\text{生产价格} + \text{模具成本} + \text{工厂存储} + \text{运输成本})$ | 包含生产+工厂+运输 |
| 节点3（施工） | $-(\text{生产价格} + \text{模具成本} + \text{工厂存储} + \text{运输成本} + \text{现场存储} + \text{施工成本})$ | 包含全部累计成本 |

### 6.5 奖励分配策略

`RewardBreakdown` 记录各成本项后，通过加权求和计算总奖励，再分配给各 Agent：

```python
def get_agent_rewards(self, breakdown, weights):
    # weights: (w1, w2, w3) - softmax归一化后的可学习权重
    clip_bound = 10.0

    # 1. 各智能体奖励 = 负成本
    prod_r  = -(breakdown.mold_cost + breakdown.production_price_cost
                + breakdown.factory_storage_cost + breakdown.transport_cost)
    trans_r = -(breakdown.transport_cost + breakdown.waiting_cost)
    const_r = -(breakdown.waiting_cost + breakdown.site_storage_cost
                + breakdown.construction_cost)

    # 2. 裁剪
    prod_r  = np.clip(prod_r,  -clip_bound, clip_bound)
    trans_r = np.clip(trans_r, -clip_bound, clip_bound)
    const_r = np.clip(const_r, -clip_bound, clip_bound)

    # 3. 加权求和
    total = weights[0] * prod_r + weights[1] * trans_r + weights[2] * const_r

    # 4. 分配给各Agent（均分）
    agent_rewards = {}
    for i in range(n_prod):
        agent_rewards[f"prod_{i}"]  = total / n_prod
    agent_rewards["transport"]      = total  # 运输Agent独占
    for j in range(n_const):
        agent_rewards[f"const_{j}"] = total / n_const
    return agent_rewards
```

> **注意**：运输 Agent 独占总奖励（运输是全局调度决策），而生产/施工 Agent 族均分各自族内的奖励。

### 6.6 RewardBreakdown 数据结构

```python
@dataclass
class RewardBreakdown:
    # === 各智能体奖励（成本最小化）===
    prod_agent_reward:  float  # -(模具成本+生产价格+工厂仓储+运输成本)
    trans_agent_reward: float  # -(运输成本+压车成本)
    const_agent_reward: float  # -(压车成本+现场仓储+施工成本)

    # === 各成本项明细 ===
    mold_cost:              float  # 模具成本
    production_price_cost:   float  # 预制构件生产成本
    factory_storage_cost:    float  # 工厂仓储成本
    transport_cost:          float  # 运输成本
    waiting_cost:            float  # 压车等待成本
    site_storage_cost:       float  # 现场仓储成本
    construction_cost:       float  # 施工成本（塔吊使用时间×2800）
    quality_penalty:         float  # 质量惩罚（累计成本）
    qc_inspection_cost:     float  # 质量检验成本
    time_penalty:           float  # 时间惩罚

    # === 完成进度奖励（新增：每步引导信号）===
    progress_reward:           float  # delta_completed * 500（每新完成1件+500）
    cumulative_progress_bonus: float  # completed_count/n_components * 500（完成比例奖励）

    @property
    def total(self) -> float:
        return (self.prod_agent_reward + self.trans_agent_reward
                + self.const_agent_reward
                - self.quality_penalty - self.qc_inspection_cost
                - self.time_penalty
                + self.progress_reward + self.cumulative_progress_bonus)
```

### 6.7 MADDPG Trainer 中的权重优化

```python
class MADDPGTrainer:
    def __init__(self, ...):
        # 可学习奖励权重（初始均匀）
        self.reward_weights = nn.Parameter(torch.ones(3) / 3.0)
        self.reward_weight_optimizer = optim.Adam(
            [self.reward_weights], lr=1e-3
        )
        self.reward_clip_bound = 10.0

    def get_reward_weights(self) -> torch.Tensor:
        """Softmax归一化权重"""
        return F.softmax(self.reward_weights, dim=0)

    def optimize_reward_weights(self, breakdown: RewardBreakdown):
        """基于奖励信号优化权重（每个episode结束时调用）"""
        w = self.get_reward_weights()
        prod_r  = torch.tensor(breakdown.prod_agent_reward)
        trans_r = torch.tensor(breakdown.trans_agent_reward)
        const_r = torch.tensor(breakdown.const_agent_reward)

        loss = -(w[0]*prod_r + w[1]*trans_r + w[2]*const_r)

        self.reward_weight_optimizer.zero_grad()
        loss.backward()
        self.reward_weight_optimizer.step()
```

---## 7. 环境配置

### 7.1 构件类型

| 类型编号 | 尺寸规格 | 生产周期 | 吊装时间 | 混凝土(m³) | 构件属性 |
|---------|----------|---------|---------|-----------|---------|
| Type-1 | 9.5×3.35m | 4天 | 45min | 9.45 | 特种 |
| Type-2 | 9.5×2.0m | 3天 | 45min | 5.64 | 特种 |
| Type-3 | 9.4×2.75m | 3天 | 45min | 7.84 | 特种 |
| Type-4 | 9.4×2.0m | 2天 | 26min | 5.70 | 常规 |
| Type-5 | 2.8×2.0m | 1.5天 | 26min | 1.60 | 常规 |

### 7.2 批次订单

| 批次 | Type-1 | Type-2 | Type-3 | Type-4 | Type-5 | 合计 |
|------|--------|--------|--------|--------|--------|------|
| 批次1 | 1 | 7 | 0 | 0 | 0 | 8 |
| 批次2 | 0 | 0 | 1 | 7 | 0 | 8 |
| 批次3 | 0 | 0 | 0 | 0 | 4 | 4 |
| 批次4 | 0 | 0 | 1 | 5 | 0 | 6 |
| **合计** | **1** | **7** | **2** | **12** | **4** | **26** |

### 7.3 资源参数

| 资源类型 | 参数值 |
|---------|-------|
| 生产线数量 | 2条 |
| 运输车辆 | 10辆（充足配置） |
| 塔吊数量 | 1台 |
| 运输时间 | 8小时（±1小时随机波动，7~9h均匀分布） |
| 卸货时间 | 10分钟/件（0.167小时/件） |
| 工厂内存放自然养护时间 | 3天（养护期结束后方可出库运输） |
| 模具生产时间（Type-1/Type-3大型特种盖板） | 15天 |
| 模具生产时间（Type-2中型特种盖板） | 12天 |
| 模具生产时间（Type-4/Type-5常规标准盖板） | 1天 |

### 7.4 成本参数

| 成本项 | 参数值 | 单位 |
|--------|-------|------|
| 施工成本费率 ω | 2800 | 元/小时 |
| 运输成本 | 1500 | 元/车次 |
| 压车费率 μ | 36 | 元/小时/辆 |
| 压车免费等待阈值 | 1.0 | 小时 |
| 工厂存储费率 θ | 0.3 | 元/件/小时 |
| 现场存储费率 σ | 1.0 | 元/件/分钟 |
| 模具制作费（Type-1/Type-3大型特种盖板） | 12万元/套 |
| 模具制作费（Type-2中型特种盖板） | 5万元/套 |
| 模具费（Type-4/Type-5常规标准盖板） | 2,000元/套 |
| 质量缺陷率 ε₁(生产) | 5% | — |
| 质量缺陷率 ε₃(运输) | 3% | — |
| 质量缺陷率 ε₄(施工) | 2% | — |


### 7.5 装载方案

| 方案 | 特种构件 | Type-4 | Type-5 | 最大装载 |
|------|---------|--------|--------|---------|
| 方案1 | 1块 | 0 | 0 | 1 |
| 方案2 | 0 | 1 | 2 | 3 |
| 方案3 | 0 | 2 | 0 | 2 |
| 方案4 | 0 | 0 | 6 | 6 |

---

## 8. 文件结构

```
ERCPSP_MMARL/
├── ERCPSP_Core_Logic.md              # 本文档
│
├── config/
│   ├── __init__.py
│   ├── ercpsp_config.py              # 主配置
│   └── qc_config.py                  # 质检节点配置
│
├── state/
│   ├── __init__.py
│   ├── ercpsp_state.py               # 状态定义（池化）
│   └── pool_encoder.py               # 池→向量编码器
│
├── env/
│   ├── __init__.py
│   ├── pools.py                      # 池管理器
│   ├── qc_checker.py                 # 三节点质检逻辑
│   ├── ercpsp_env.py                 # 主环境（Gym风格）
│   └── reward_ercpsp.py             # ERCPSP奖励函数
│
├── agents/
│   ├── __init__.py
│   ├── production_agent.py           # 单生产线Agent
│   ├── production_fleet.py           # 生产Agent群管理器
│   ├── transport_agent.py           # 运输Agent
│   ├── construction_agent.py         # 单塔吊Agent
│   └── construction_fleet.py        # 施工Agent群管理器
│
├── network/
│   ├── __init__.py
│   ├── maddpg_actor.py              # Actor网络
│   ├── maddpg_critic.py             # 中心化Critic网络
│   └── noise.py                      # OU噪声
│
├── trainer/
│   ├── __init__.py
│   ├── maddpg_trainer.py            # MADDPG主训练器
│   └── replay_buffer.py              # 经验回放
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                    # 日志工具
│   ├── result_visualizer.py         # 训练曲线与甘特图可视化
│   └── result_logger.py              # 训练结果分析文档生成
│
├── run_training.py                  # 训练入口（含实时进度）
├── run_training_gpu.py               # GPU训练脚本（500轮）
├── run_evaluation.py                 # 评估入口
├── requirements.txt                  # 依赖
└── README.md                        # 项目说明
```

## 9. 训练输出说明

每次训练完成后，`results/{run_id}/` 目录生成以下文件：

| 文件 | 内容 |
|------|------|
| `生产智能体Actor损失曲线.png` | 生产智能体（生产线Agent）独立的Actor loss曲线 |
| `运输智能体Actor损失曲线.png` | 运输智能体独立的Actor loss曲线 |
| `施工智能体Actor损失曲线.png` | 施工智能体（塔吊Agent）独立的Actor loss曲线 |
| `生产智能体累计reward曲线.png` | 生产智能体族累计reward随训练进度变化 |
| `运输智能体累计reward曲线.png` | 运输智能体累计reward随训练进度变化 |
| `施工智能体累计reward曲线.png` | 施工智能体族累计reward随训练进度变化 |
| `总的累计奖励图.png` | 训练奖励与评估奖励曲线（含误差带） |
| `动作热力图.png` | 各Agent动作选择频率分布 |
| `多智能体协同.png` | 生产/运输/施工三行协同时序图 |
| `甘特图.png` | 完整甘特图（生产+运输+施工） |
| `结果分析.md` | 含统计汇总 + 6项成本分解的完整报告 |

**成本分解**（结果分析.md）：
- 工厂仓储成本、现场仓储成本、压车等待成本、运输成本、质量惩罚、时间惩罚

---

## 10. 关键技术点

### 10.1 离散策略动作空间的实现

MADDPG通过以下策略适配离散策略选择动作空间：

1. **Actor输出**：Softmax概率分布 over {a_0, a_1, ..., a_n}
2. **ε-greedy探索**：以ε概率随机探索，以1-ε概率利用
3. **动作模式自动选择**：action_dim ≤ 10 → 离散模式；action_dim > 10 → 连续模式（本项目均≤10，全部为离散）
4. **Critic输入**：离散动作用one-hot编码拼接，维度 = Σ action_dim
5. **策略梯度更新**：用当前策略的Softmax概率替换critic输入中的动作，引导梯度

```python
# Actor前向传播
logits = self.output_net(features)  # 输出logits
probs = F.softmax(logits, dim=-1)   # 转概率

# ε-greedy动作选择
if random() < epsilon:
    action = random_choice(n_actions)     # 随机（按概率分布采样）
else:
    action = argmax(probs)                # 利用（argmax）

# Critic输入：one-hot编码
act_onehot = torch.zeros(batch_size, action_dim)
indices = act_batch.long().clamp(0, action_dim - 1)
act_onehot.scatter_(1, indices.unsqueeze(1), 1.0)

# 策略梯度：当前agent用新策略（Softmax概率），其他agent也用新策略
alt_actions = list(old_actions)
alt_actions[i] = probs  # 第i个agent用当前策略概率
q_val = critic(obs_tensors, alt_actions)
policy_loss = -q_val.mean()
```

### 10.2 泛化状态空间的特征设计

池化状态采用**泛化特征设计**，使模型不依赖于固定的构件总数：

```python
class PoolEncoder:
    """池→向量编码器 - 泛化特征设计"""

    def encode_pool_typedemand(self, pool: PoolState) -> np.ndarray:
        """
        使用各类型实际需求量作为分母归一化 (5维)。
        关键：使模型泛化至不同构件总数的场景。
        """
        demands = np.array(self.cfg.get_type_demands(), dtype=np.float32)
        return pool.levels / demands

    def encode_completion_rates(self, state: ERCPSPState) -> np.ndarray:
        """各类型完成率 (5维): completion / total_demand"""
        return state.get_type_completion_rates()

    def encode_pool_utilizations(self, state: ERCPSPState) -> np.ndarray:
        """各池容量利用率 (7维): 各池总量 / max_level"""
        return state.get_pool_utilizations()

    def encode_bottleneck_type(self, state: ERCPSPState) -> np.ndarray:
        """瓶颈类型标识 (5维): one-hot 编码"""
        return state.get_bottleneck_type()
```

### 10.3 多Agent中心化Critic设计

每个Agent有独立的Actor网络，但共享一个中心化Critic：

```python
class CentralizedCritic(nn.Module):
    def __init__(self, agent_obs_dims, agent_action_dims):
        # 拼接所有Agent的观测和动作
        total_obs = sum(agent_obs_dims)
        total_action = sum(agent_action_dims)
        # 3层全连接网络
        self.net = nn.Sequential(
            nn.Linear(total_obs + total_action, 512),
            nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
    
    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)
```

### 10.4 池化状态的编码（归一化统一）

> **重要变更（v2.2）**：`encode_pool`（使用固定max_level=26归一化）和`PoolState.normalized()`已**废弃**，统一使用`encode_pool_typedemand`（按各类型实际需求量归一化）。

```python
class PoolEncoder:
    """池→向量编码器 - 统一使用typedemand归一化（泛化设计）"""

    def encode_pool_typedemand(self, pool: PoolState) -> np.ndarray:
        """
        使用各类型实际需求量作为分母归一化池状态 (5维)。
        泛化设计：使模型不依赖于固定的构件总数。
        """
        demands = np.array(self.cfg.get_type_demands(), dtype=np.float32)
        return np.divide(
            pool.levels,
            demands,
            out=np.zeros_like(pool.levels, dtype=np.float32),
            where=demands > 0
        )

    # ========== 已废弃的方法（保留仅用于兼容性）==========

    def encode_pool(self, pool: PoolState, max_level: int) -> np.ndarray:
        """
        [废弃] 使用固定max_level归一化，与泛化设计冲突。
        请使用 encode_pool_typedemand 代替。
        """
        return np.array(pool.levels) / max_level

# PoolState.normalized() 也已废弃：
# def normalized(self) -> np.ndarray:
#     """[废弃] 请使用 PoolEncoder.encode_pool_typedemand()"""
#     return self.levels / max(self.max_level, 1)
```

### 10.5 目标网络与软更新

```python
def soft_update(target_net, source_net, tau=0.005):
    """Polyak软更新"""
    for target_param, source_param in zip(
        target_net.parameters(), source_net.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data + (1 - tau) * target_param.data
        )
```

### 10.6 同质子群参数共享实现

同质子群（生产群：prod_0/prod_1；施工群：const_0/...）使用参数共享，核心实现如下：

```python
# maddpg_trainer.py __init__

# 识别同质子群
self.prod_names = [n for n in self.sorted_names if n.startswith('prod_')]
self.const_names = [n for n in self.sorted_names if n.startswith('const_')]

# 创建共享网络（以生产群为例）
self._shared_prod_net = ActorNetwork(prod_obs_d, prod_act_d,
    action_mode='discrete', qc_action_dim=2, device=self.device)
shared_prod_target = ActorNetwork(...)
shared_prod_target.load_state_dict(self._shared_prod_net.state_dict())
shared_prod_opt = optim.Adam(self._shared_prod_net.parameters(), lr=1e-4)

# 让 prod_0 / prod_1 指向同一个网络实例（字典引用共享，非参数复制）
for name in self.prod_names:
    self.actors[name] = self._shared_prod_net
    self.actors_target[name] = shared_prod_target
    self.actor_optimizers[name] = shared_prod_opt
```

**梯度累加效果**：两个智能体各自计算 policy loss 后，分别调用 `backward()`，梯度被累加到共享的 Adam 优化器中，一次 `step()` 同时更新两个智能体的策略。这等价于对两个智能体的经验进行联合优化，学习一个共享的最优策略。

**同质子群 loss 汇总**（`update()` 返回值）：

```python
# 同质子群 policy loss 均值
prod_loss_vals = [per_agent_policy_loss.get(n, 0.0) for n in self.prod_names]
prod_loss_mean = float(np.mean(prod_loss_vals))
const_loss_vals = [per_agent_policy_loss.get(n, 0.0) for n in self.const_names]
const_loss_mean = float(np.mean(const_loss_vals))

# 同质子群 Q 值均值
prod_q_vals = [agent_q_estimates.get(n, 0.0) for n in self.prod_names]
prod_q_mean = float(np.mean(prod_q_vals))
```

**save/load 兼容性**：保存检查点时，通过 `id()` 去重，每个共享网络只保存一次参数；加载时同样按 `id()` 去重处理，兼容参数共享架构。

---

## 11. 参考文献

1. Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. NeurIPS.
2. ERCPSP Mathematical Formulation - Quality Risk Extension (本项目文档)
3. 预制构件供应链多智能体强化学习 - 输入数据清单 (本项目文档)
4. config/qc_config.py - 三节点质检配置（节点级奖励/惩罚参数）
