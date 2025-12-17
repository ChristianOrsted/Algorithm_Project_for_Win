# 项目核心代码分析

## 核心要素：构建强化学习的基石
在代码开始运转之前，我们首先要定义这个“世界”里的角色和规则。这些要素是强化学习（RL）的地基，直接对应了论文中 DEC-POMDP 模型的定义。 

### 1. 智能体 (Agent) —— 这里的“大脑”

**代码对应**：weak_tie_agent.py 中的 WeakTieAgent 类。

**详细解析**：在多智能体强化学习（MARL）中，Agent 指的是整个智能体团队的控制中心。你的代码中，WeakTieAgent 管理着两套网络：

**Actor (演员)**：即 self.actors，对应 WeakTieNet，负责根据观测做出动作。
```python
self.actors = nn.ModuleList([
            WeakTieNet(obs_dim, act_dim, hidden_dim, act_dim, use_actions=False).to(self.device)
            for _ in range(n_agents)
        ])
```
**Critic (评论家)**：即self.critic，也对应 WeakTieNet（但 use_actions=True），负责站在上帝视角给动作打分。
```python
self.critic = WeakTieNet(obs_dim, act_dim, hidden_dim, 1, use_actions=True).to(self.device)
```
**WeakTieNet**：定义如下：
```python
class WeakTieNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, output_dim, use_actions=False):
        super(WeakTieNet, self).__init__()
        self.fusion = WeakTieFusionLayer(obs_dim, act_dim, hidden_dim, use_actions=use_actions)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, local_obs, global_obs, mask, key_idx, hidden_state, local_act=None, global_act=None):
        """
                前向传播函数
                参数:
                    local_obs: 当前智能体的局部观测
                    global_obs: 所有智能体的观测 (用于提取弱联系和 Key Agent 信息)
                    mask: 弱联系掩码 (决定哪些队友的信息被保留，哪些被过滤)
                    key_idx: 关键智能体的索引
                    hidden_state: 上一时刻的记忆状态 (GRU 的 h_{t-1})
                    local_act: (仅Critic模式) 当前智能体的动作
                    global_act: (仅Critic模式) 所有智能体的动作
        """
        # 1. 特征融合 (Feature Fusion)
        # 调用 WeakTieFusionLayer，将 [自身信息, 弱联系聚合信息, Key Agent信息] 拼接并映射到 hidden_dim
        # 输入: 各种观测和动作 -> 输出: 形状为 (Batch, hidden_dim) 的特征向量 x
        x = self.fusion(local_obs, global_obs, mask, key_idx, local_act, global_act)
        x = self.relu(x)

        # 2. GRU 记忆
        # 输入: 当前特征 x 和 上一步记忆 hidden_state
        # 输出: 新的记忆 h_new (它既是这一层的输出，也是下一步的 hidden_state)
        # 形状: (Batch, hidden_dim)
        h_new = self.gru(x, hidden_state)

        # 3. 输出
        # 返回: (网络输出, 新的隐藏状态)
        # 新的隐藏状态需要在外部保存，并在下一个时间步传回来
        out = self.fc_out(h_new)
        return out, h_new
```

### 2. 环境 (Environment) —— 这里的“战场”

**代码对应**：weak_tie_env.py 中的 WeakTieStarCraft2Env 类。

**详细解析**：这是智能体生存和交互的场所。它继承自 SMAC（星际争霸2多智能体挑战），负责反馈游戏画面和判定胜负。我对其进行了关键修改，增加了 get_all_unit_positions 接口，以便获取物理坐标来构建“弱联系图”。
```python
class WeakTieStarCraft2Env(StarCraft2Env):
    """
    继承自 SMAC 环境，增加获取单位绝对坐标的接口，供 WeakTieGraph 模块使用。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_tags = []

    def reset(self):
        """重置环境并记录所有智能体的 Tag，保证ID顺序与 WeakTieAgent 一致"""
        obs, state = super().reset()
        # 必须记录所有智能体的 tag，即使它们在 reset 时被创建
        self.agent_tags = sorted(list(self.agents.keys()))
        return obs, state

    def get_all_unit_positions(self):
        """
        [核心功能] 获取所有智能体的绝对坐标 (x, y)。
        即使智能体死亡，也返回 [0.0, 0.0] 以保持数组形状 (n_agents, 2)。

        Returns:
            np.array: shape (n_agents, 2), dtype=float32
        """
        positions = []
        for tag in self.agent_tags:
            if tag in self.agents:  #如果单位活着
                unit = self.agents[tag]
                # 直接访问 PySC2 Unit 对象的 pos 属性
                positions.append([unit.pos.x, unit.pos.y])
            else:
                # 死亡单位返回 [0.0, 0.0]
                positions.append([0.0, 0.0])

        # 强制转换为 float32，确保与 PyTorch tensor 转换时无警告
        return np.array(positions, dtype=np.float32)
```

### 3. 状态 (State) —— 上帝视角的联合信息
**代码对应**：weak_tie_agent.py 中 update_batch 函数里传给 self.critic 的组合参数：global_obs (拼接观测) + mask (弱联系图) + key_idx。

**详细解析**：应论文中的全局状态S。在代码中，我弃用了环境原始的扁平 state 向量，改为用“所有智能体局部观测的拼接”加上的“弱联系图特征（包含 Mask 和 Key Agent）”来重组 State。在 CTDE（集中训练，分布执行） 架构中，这个包含图拓扑结构的全局信息仅在训练时传给 Critic 网络用于计算优势 (A_t)，Actor 在实战中是看不到的。
```python
q, h_new = self.critic(
                        local_obs=pad_obs[:, t, i], global_obs=pad_obs[:, t],
                        mask=pad_masks[:, t, i], key_idx=pad_keys[:, t],
                        hidden_state=critic_hiddens[i],
                        local_act=pad_acts_onehot[:, t, i], global_act=pad_acts_onehot[:, t]
                    )
```
### 4. 观测 (Observation) —— 士兵视角的局部信息

**代码对应**：train_smac.py 中的 obs 变量。

**详细解析**：对应论文中的O。每个智能体只能看到自己视野范围内的东西。这是 Actor 做决策的唯一依据。在我的代码中，obs 的维度包含了地形特征和视野内的单位信息。
```python
def run_episode(env, agent, wt_graph, train_mode=True):
    obs, state = env.reset()
    terminated = False
    episode_reward = 0
    raw_episode_reward = 0

    actor_hidden = agent.init_hidden(batch_size=1)

    episode_buffer = {'obs': [], 'acts': [], 'rewards': [], 'dones': [],
                      'avails': [], 'probs': [], 'masks': [], 'keys': []}

    step_count = 0
    last_mask_beta = None
    last_key_agent_idx = None

    while not terminated:
        avail_actions = env.get_avail_actions()
        positions = env.get_all_unit_positions()
        alive_mask = np.any(positions != 0, axis=1)

        # 动态图更新
        if step_count % GRAPH_UPDATE_INTERVAL == 0:
            mask_beta, key_agent_idx = wt_graph.compute_graph_info(positions, alive_mask)
            last_mask_beta = mask_beta
            last_key_agent_idx = key_agent_idx
        else:
            mask_beta = last_mask_beta
            key_agent_idx = last_key_agent_idx

        step_count += 1

        actions, probs, next_hidden = agent.select_action(
            obs, avail_actions, mask_beta, key_agent_idx, actor_hidden,
            deterministic=(not train_mode)
        )

        reward, terminated, info = env.step(actions)
        next_obs = env.get_obs()

        shaped_reward = reward / 5.0

        if train_mode:
            episode_buffer['obs'].append([obs])
            episode_buffer['acts'].append([actions])
            episode_buffer['rewards'].append([[shaped_reward] * len(actions)])
            episode_buffer['dones'].append([[float(terminated)] * len(actions)])
            episode_buffer['avails'].append([avail_actions])
            episode_buffer['probs'].append([probs])
            episode_buffer['masks'].append([mask_beta])
            episode_buffer['keys'].append([key_agent_idx])

        obs = next_obs
        actor_hidden = next_hidden
        episode_reward += shaped_reward
        raw_episode_reward += reward

    is_win = info.get('battle_won', False)
    return episode_reward, raw_episode_reward, is_win, episode_buffer, None
```

### 5. 动作 (Action) —— 具体的指令

**代码对应**：train_smac.py 中的 actions 变量（由 agent.select_action 产生）。

**详细解析**：对应论文中的A。这是智能体输出的离散指令，比如“向北移动”、“攻击ID为3的敌人”或“停止”。代码中使用 F.softmax 输出动作概率分布，然后采样得到具体的动作索引。
```python
def select_action(self, obs, avail_actions, mask, key_idx, actor_hidden, deterministic=False):
    obs = np.array(obs)
    avail_actions = np.array(avail_actions)
    mask = np.array(mask)
    key_idx = np.array(key_idx)

    if obs.ndim == 2:
        obs = obs[None, ...]
        avail_actions = avail_actions[None, ...]
        mask = mask[None, ...]
        key_idx = key_idx[None, ...]

    # 数据预处理
    obs = self.normalize_obs(obs, update_stats=True)

    obs_t = torch.FloatTensor(obs).to(self.device)
    avail_t = torch.FloatTensor(avail_actions).to(self.device)
    mask_t = torch.FloatTensor(mask).to(self.device)
    key_t = torch.LongTensor(key_idx).to(self.device)

    actions_list = []
    probs_list = []
    new_hidden_list = []

    with torch.no_grad():
        for i in range(self.n_agents):
            # 前向传播
            logits, h_new = self.actors[i](
                local_obs=obs_t[:, i, :],
                global_obs=obs_t,
                mask=mask_t[:, i, :],
                key_idx=key_t,
                hidden_state=actor_hidden[:, i, :],
                local_act=None, global_act=None
            )

            # 选取可用动作
            avail_i = avail_t[:, i, :]
            logits[avail_i == 0] = -1e10
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            actions_list.append(action)
            probs_list.append(probs)
            new_hidden_list.append(h_new)

    actions = torch.stack(actions_list, dim=1)
    probs = torch.stack(probs_list, dim=1)
    new_hidden = torch.stack(new_hidden_list, dim=1)

    return actions.cpu().numpy()[0], probs.cpu().numpy()[0], new_hidden
```

### 6. 奖励 (Reward) —— 行为的反馈信号

**代码对应**：train_smac.py 中的 shaped_reward。

**详细解析**：对应论文中的 R 。环境对智能体行为的即时反馈。

**原始奖励**：reward 来自 env.step()，通常是造成伤害或击杀敌人得分。

**奖励塑形 (Reward Shaping)**：你在代码中做了 shaped_reward = reward / 5.0 的处理。这是为了将数值缩放到神经网络合适的范围，防止梯度爆炸。
```python
reward, terminated, info = env.step(actions)
        next_obs = env.get_obs()

        shaped_reward = reward / 5.0
```

# 训练阶段

## 第一阶段：构建世界与弱联系图 (The Weak Tie Graph)

在定义好角色后，每一步决策开始前，我的代码做了一件传统算法不做的事情：**基于位置构建社交网络**。

### 1. 获取物理坐标

在 weak_tie_env.py 中，我重写了环境类，增加了 get_all_unit_positions() 方法。

**逻辑**：它不仅仅返回智能体的视野（Obs），还直接读取了游戏引擎内部所有单位的 x, y 坐标。即使某个单位阵亡了，也会返回 [0,0] 来保持数据对齐。

**对应原理**：这是为了后续计算“谁和谁离得近”提供原始数据。

### 2.	计算联系强度 (Tie Strength)
在 weak_tie_module.py 的 WeakTieGraph 类中，逻辑开始运转。
连边 **(Definition 1 & 2)**：代码计算所有智能体之间的距离矩阵 dists。如果距离小于 obs_range（代码设为 15.0），则认为两者有边相连。代码还包含了一个特殊的步骤来处理“孤岛”，强制连接不连通的子图，确保图的连通性。
强度公式 (Eq. 8)：这是最关键的数学映射。代码遍历每一对连接的智能体，计算：

(8)
$$
s_{strength} = 1/(D_i + D_j + W_{i,j} - 2)
$$

**代码体现**：denominator = D_i + D_j + W_ij - 2，然后取倒数 1.0 / denominator。这完全对应论文中的公式 (8)。
```python
for i in range(n):
    for j in range(n):
        if i == j:
            H[i][j] = 1.0   # 自己对自己强度为 1
            continue
        if i not in valid_indices or j not in valid_indices:
            H[i][j] = 0.0   # 死人强度为 0
            continue

        D_i, D_j = degrees[i], degrees[j]   # 度
        W_ij = path_lengths[i].get(j, 9999)  # 路径长度 (无穷大用 9999 代替)
        denominator = D_i + D_j + W_ij - 2   # 论文公式
        strength = 1.0 / denominator if denominator > 0 else 0.0  # 强度倒数，因为距离越大，联系越弱
        H[i][j] = strength
        tie_values.append(strength)
```

### 3.	筛选弱联系与关键智能体 (Filtering & Key Agent)

**弱联系掩码 (Mask Beta)**：论文理论认为“弱联系带来新颖信息”。代码计算所有强度的 alpha_quantile（30% 分位数）。凡是强度小于这个阈值的边，被标记为 1 (保留)，强度大的反而被标记为 0 (过滤)。这生成了 mask_beta 矩阵。

**关键智能体 (Definition 5)**：代码统计所有存活节点的度，选出度最大的那个节点索引作为 key_agent_idx。
```python
# 5. 动态阈值划分
        # 算出强度的 30% 分位数 (alpha_quantile)
        current_alpha = np.quantile(tie_values, self.alpha_quantile) if tie_values else 0.0

        # 生成 Mask：H <= Alpha 为弱联系(保留), 否则为强联系(过滤)
        # 加上对角线(保留自己)
        mask_beta = (H <= current_alpha).astype(np.float32)
        np.fill_diagonal(mask_beta, 1.0)
```

## 第二阶段：信息融合与决策 (Fusion & Decision)
现在，每个智能体不仅有了自己的视野，还有了刚才计算出的“社交关系”。在 weak_tie_module.py 的 WeakTieNet 网络中，发生了信息的深度融合。

### 1. 三路信息流
在 WeakTieFusionLayer 的 forward 函数中，输入数据被分成了三路：

**Local (自身)**：智能体自己看到的局部观测 local_obs。

**Weak (弱联系)**：这部分最特别。代码执行了 global_obs * mask。因为 mask 里只有弱联系是 1，强联系是 0，所以这步操作直接屏蔽了身边亲密队友的信息，只聚合了远处（弱联系）队友的信息。这对应论文中提到的“减少信息冗余” 7。

**Key (关键)**：直接提取 key_agent_idx 对应的那一份观测数据。
```python
def forward(self, local_obs, global_obs, mask_beta, key_idx, local_act=None, global_act=None):
    batch_size = local_obs.shape[0]

    # mask 维度调整: (B, N) -> (B, N, 1)，为了能和 (B, N, Dim) 的 obs 相乘
    mask = mask_beta.unsqueeze(-1)  # (B, N, 1)
    sum_mask = mask.sum(dim=1) + 1e-6  # 分母，用于求平均，加 1e-6 防止除以 0

    # --- 1. 弱联系聚合 (Weak Tie Aggregation) ---
    # 严格执行过滤：只聚合 mask=1 (弱联系) 的邻居，乘以 mask 后，强联系队友的信息变成 0
    weak_obs_masked = global_obs * mask
    # 求和并除以数量，得到弱联系队友的平均信息
    weak_obs_agg = weak_obs_masked.sum(dim=1) / sum_mask

    # Key Agent (Obs)
    feat_dim = global_obs.shape[2]
    idx_exp = key_idx.view(batch_size, 1, 1).expand(-1, 1, feat_dim)
    key_obs = torch.gather(global_obs, 1, idx_exp).squeeze(1)

    # --- 2. 弱联系聚合 (Act) ---
    if self.use_actions:
        # critic模式 输入 (Obs + Act)
        if local_act is None or global_act is None:
            raise ValueError("Critic mode requires actions!")

        # m_i 包含 beta * a^{-i}
        weak_act_masked = global_act * mask
        weak_act_agg = weak_act_masked.sum(dim=1) / sum_mask   # 求和并除以数量，得到弱联系队友的平均信息

        # Key Agent (Act)
        act_dim = global_act.shape[2]
        idx_act_exp = key_idx.view(batch_size, 1, 1).expand(-1, 1, act_dim)
        key_act = torch.gather(global_act, 1, idx_act_exp).squeeze(1)

        # 拼接: [Local(O+A), Weak(O+A), Key(O+A)]
        # 1. 处理自己的信息：把自己看到的(Obs)和自己做的(Act)拼起来
        local_feat = torch.cat([local_obs, local_act], dim=-1)
        # 2. 处理弱联系队友的信息：把队友看到的和队友做的拼起来，然后取平均
        weak_feat = torch.cat([weak_obs_agg, weak_act_agg], dim=-1)
        # 3. keyagent同理
        key_feat = torch.cat([key_obs, key_act], dim=-1)
    else:
        # Actor 模式 输入 (Obs)
        local_feat = local_obs
        weak_feat = weak_obs_agg
        key_feat = key_obs

    # --- 3. 融合 ---
    #把 [自己信息, 弱联系队友信息, Key Agent信息] 拼在一起
    combined = torch.cat([local_feat, weak_feat, key_feat], dim=-1)
    out = self.fc(combined)
    return self.layer_norm(out)
```

### 2. 拼接与记忆
代码将上述三部分 torch.cat（拼接）在一起，形成一个巨大的特征向量。
这个向量通过全连接层 (fc) 后，被送入 **GRU (门控循环单元)**。GRU 的作用是赋予智能体“记忆”，让它能理解时间序列（比如“我上一秒在挨打，这一秒应该跑”）。
```python
class WeakTieNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, output_dim, use_actions=False):
        super(WeakTieNet, self).__init__()
        self.fusion = WeakTieFusionLayer(obs_dim, act_dim, hidden_dim, use_actions=use_actions)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, local_obs, global_obs, mask, key_idx, hidden_state, local_act=None, global_act=None):
        """
                前向传播函数
                参数:
                    local_obs: 当前智能体的局部观测
                    global_obs: 所有智能体的观测 (用于提取弱联系和 Key Agent 信息)
                    mask: 弱联系掩码 (决定哪些队友的信息被保留，哪些被过滤)
                    key_idx: 关键智能体的索引
                    hidden_state: 上一时刻的记忆状态 (GRU 的 h_{t-1})
                    local_act: (仅Critic模式) 当前智能体的动作
                    global_act: (仅Critic模式) 所有智能体的动作
        """
        # 1. 特征融合 (Feature Fusion)
        # 调用 WeakTieFusionLayer，将 [自身信息, 弱联系聚合信息, Key Agent信息] 拼接并映射到 hidden_dim
        # 输入: 各种观测和动作 -> 输出: 形状为 (Batch, hidden_dim) 的特征向量 x
        x = self.fusion(local_obs, global_obs, mask, key_idx, local_act, global_act)
        x = self.relu(x)

        # 2. GRU 记忆
        # 输入: 当前特征 x 和 上一步记忆 hidden_state
        # 输出: 新的记忆 h_new (它既是这一层的输出，也是下一步的 hidden_state)
        # 形状: (Batch, hidden_dim)
        h_new = self.gru(x, hidden_state)

        # 3. 输出
        # 返回: (网络输出, 新的隐藏状态)
        # 新的隐藏状态需要在外部保存，并在下一个时间步传回来
        out = self.fc_out(h_new)
        return out, h_new
```

### 3. 输出动作概率
在 weak_tie_agent.py 的 select_action 中，网络的输出经过 F.softmax 变成了概率分布 probs。例如：[向左: 10%, 向右: 80%, 攻击: 10%]。注意self.actors[i]，调用各个agent的actor网络中的forward函数，此时，**网络只进行前向推理来生成动作，参数是冻结的**。

```python
with torch.no_grad():
    for i in range(self.n_agents):
        # 前向传播
        logits, h_new = self.actors[i](
            local_obs=obs_t[:, i, :],
            global_obs=obs_t,
            mask=mask_t[:, i, :],
            key_idx=key_t,
            hidden_state=actor_hidden[:, i, :],
            local_act=None, global_act=None
        )

        # 选取可用动作
        avail_i = avail_t[:, i, :]
        logits[avail_i == 0] = -1e10
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
```

## 第三阶段：交互与数据收集 (Interaction)

在 train_smac.py 的 run_episode 函数中，智能体带着大脑进入游戏。

### 1. 探索与利用
**训练模式**：代码使用 dist.sample() 根据概率进行随机采样。即使“向右”概率最高，智能体偶尔也会尝试“向左”，这是为了发现新策略（探索）。

**熵衰减**：在 train_smac.py 中，ENTROPY_START 到 ENTROPY_END 的衰减逻辑控制了这种随机性。初期熵高，随机性大；后期熵低，趋于稳定。在SAC中也有所体现

```python
def get_current_entropy(episode):
    if episode > ENTROPY_DECAY_EPISODES:
        return ENTROPY_END
    frac = 1.0 - (episode / ENTROPY_DECAY_EPISODES)
    return max(ENTROPY_END, ENTROPY_END + frac * (ENTROPY_START - ENTROPY_END))
```

### 2. 数据缓存 (Buffer)
由于 PPO 是 On-Policy 算法，它不能使用很久以前的经验。train_smac中run_episode函数的 episode_buffer 严格按时间顺序记录了这一局的所有 obs, actions, rewards, masks, keys。

```python
if train_mode:
    episode_buffer['obs'].append([obs])
    episode_buffer['acts'].append([actions])
    episode_buffer['rewards'].append([[shaped_reward] * len(actions)])
    episode_buffer['dones'].append([[float(terminated)] * len(actions)])
    episode_buffer['avails'].append([avail_actions])
    episode_buffer['probs'].append([probs])
    episode_buffer['masks'].append([mask_beta])
    episode_buffer['keys'].append([key_agent_idx])
```

**注意**：为了节省计算，wt_graph 每隔 3 步 (GRAPH_UPDATE_INTERVAL) 才更新一次图结构，中间步复用上一次的结果。

```python
while not terminated:
        avail_actions = env.get_avail_actions()
        positions = env.get_all_unit_positions()
        alive_mask = np.any(positions != 0, axis=1)

        # 动态图更新
        if step_count % GRAPH_UPDATE_INTERVAL == 0:
            mask_beta, key_agent_idx = wt_graph.compute_graph_info(positions, alive_mask)
            last_mask_beta = mask_beta
            last_key_agent_idx = key_agent_idx
        else:
            mask_beta = last_mask_beta
            key_agent_idx = last_key_agent_idx

        step_count += 1
```

## 第四阶段：反思与进化 (PPO Update)

当收集够了 BATCH_SIZE (64局) 的数据后，weak_tie_agent.py 中的 update_batch 函数开始工作。这是最复杂的数学部分。

### 1. 计算优势 (Advantage)

智能体需要知道：“我刚才做的动作，比‘平均水平’好多少？”

**Critic 网络**：它输入的是全局状态，负责打分。代码中 q_taken 计算了实际采取动作的价值。

**反事实基线 (Counterfactual Baseline)**：这是为了解决多智能体背锅问题。代码逻辑是：保持队友动作不变，把我的动作替换成所有可能的动作 (batch_all_actions)，再算一遍价值，然后加权平均得到 baseline_v。

**优势值**：advantages = q_taken - baseline_v。如果结果是正的，说明这一步做对了。
```python
# ==========================
# 2. 计算优势函数 (GAE / Advantage)
# ==========================
with torch.no_grad():  # 计算 Advantage 不需要梯度
    current_probs_list = []
    h_init = torch.zeros(BatchSize, self.hidden_dim).to(self.device)

    # 计算当前所有 Actor 的动作概率分布
    for i in range(self.n_agents):
        # 使用 Time-Folding 函数批量推理
        logits_seq = self._forward_network_sequence(
            self.actors[i], pad_obs[:, :, i], pad_obs, pad_masks[:, :, i], pad_keys, h_init
        )
        logits_seq[pad_avails[:, :, i] == 0] = -1e10
        current_probs_list.append(F.softmax(logits_seq, dim=-1))
    current_probs = torch.stack(current_probs_list, dim=2)  # [B, T, N, ActDim]

    # 准备计算 Critic 值
    q_taken_list = []
    critic_hiddens = [torch.zeros(BatchSize, self.hidden_dim).to(self.device) for _ in range(self.n_agents)]
    # 需要保存 Critic 的 hidden state 历史，因为后面算 Counterfactual Baseline 还要用到
    critic_history_tensor = torch.zeros(BatchSize, MaxTime, self.n_agents, self.hidden_dim).to(self.device)

    # 循环时间步计算 Q(s, a) - 实际采取动作的 Q 值，即采取某动作后，智能体期望获得的未来累积折扣奖励。
    # 动作价值函数
    for t in range(MaxTime):
        q_t_list = []
        for i in range(self.n_agents):
            critic_history_tensor[:, t, i] = critic_hiddens[i]  # 存下来
            # 调用 Critic 网络进行前向传播
            q, h_new = self.critic(
                local_obs=pad_obs[:, t, i], global_obs=pad_obs[:, t],
                mask=pad_masks[:, t, i], key_idx=pad_keys[:, t],
                hidden_state=critic_hiddens[i],
                local_act=pad_acts_onehot[:, t, i], global_act=pad_acts_onehot[:, t]
            )
            critic_hiddens[i] = h_new
            q_t_list.append(q)
        q_taken_list.append(torch.stack(q_t_list, dim=1))
    # 最终的输出张量，形状是 [BatchSize, MaxTime, N_Agents]。它存储了当前 Batch中所有 Episode、所有时间步、所有智能体的Q_taken值。
    q_taken = torch.stack(q_taken_list, dim=1).squeeze(-1)

    # Counterfactual Baseline 反事实基线核心逻辑
    # 论文中为了解决 Credit Assignment (信度分配) 问题，使用了 Counterfactual Baseline。
    # V(s) = sum_a [ pi(a|s) * Q(s, (a, a_{-i})) ]  V 是 Q 的期望！！！
    # 状态价值函数
    # 即: 保持其他队友动作不变，遍历当前 Agent 所有可能的动作，计算期望 Q 值作为 Baseline。
    baseline_v = torch.zeros_like(q_taken)
    all_actions_eye = torch.eye(self.act_dim).to(self.device)  # 生成单位矩阵，代表所有可能的动作
    batch_all_actions = all_actions_eye.unsqueeze(0).repeat(BatchSize, 1, 1).view(-1, self.act_dim)

    # 遍历 Time 和 Agent
    for t in range(MaxTime):
        base_global_act = pad_acts_onehot[:, t]  # 当前时刻所有人的动作
        for i in range(self.n_agents):
            # 数据复制 (Repeat)，因为我们要并行计算 act_dim 个动作的情况
            obs_rep = pad_obs[:, t, i].repeat_interleave(self.act_dim, dim=0)
            g_obs_rep = pad_obs[:, t].repeat_interleave(self.act_dim, dim=0)
            mask_rep = pad_masks[:, t, i].repeat_interleave(self.act_dim, dim=0)
            key_rep = pad_keys[:, t].repeat_interleave(self.act_dim, dim=0)
            h_rep = critic_history_tensor[:, t, i].repeat_interleave(self.act_dim, dim=0)

            # 构建 Counterfactual Global Action
            # 复制一份全局动作
            global_act_rep = base_global_act.repeat_interleave(self.act_dim, dim=0).clone()
            # 强行修改第 i 个 Agent 的动作为所有可能动作 (batch_all_actions)
            global_act_rep[:, i, :] = batch_all_actions

            # 批量输入 Critic，一次性计算 B x A 种反事实情况的 Q 值
            q_values_flat, _ = self.critic(
                local_obs=obs_rep, global_obs=g_obs_rep,
                mask=mask_rep, key_idx=key_rep,
                hidden_state=h_rep,
                local_act=batch_all_actions, global_act=global_act_rep
            )
            q_values = q_values_flat.view(BatchSize, self.act_dim)

            # 加权求平均: V = sum_a ( pi(a) * Q(a) )
            baseline_v[:, t, i] = (current_probs[:, t, i] * q_values).sum(dim=-1)

    # 计算优势: A(s, a) = Q(s, a) - V(s)
    advantages = q_taken - baseline_v

    # 计算目标回报 Returns (用于 Critic Loss)
    # Bellman 方程变体: r + gamma * V(s')
    # baseline_v[:, 1:] 是下一时刻的 V
    baseline_v_next = torch.cat([baseline_v[:, 1:], torch.zeros_like(baseline_v[:, :1])], dim=1)
    returns = pad_rews.squeeze(-1) + self.gamma * baseline_v_next * (1 - pad_dones.squeeze(-1))
```

### 2. 限制更新幅度 (Clipping)
PPO 的精髓在于“稳”。
代码计算了新旧策略的比率 ratio = exp(new_log_prob - old_log_prob)。

(6)
$$r_i = \frac{\pi_i(a_i | o_i)}{\pi_i^{old}(a_i | o_i)}$$

(7)
$$\eta^{MAPPO} = \frac{1}{N} \sum_{i}^{N} E_{\vec{a}^t, s^t \sim \pi^{old}}[\min(r_i A^{old}, \text{clip}(r_i, 1 - \epsilon, 1 + \epsilon) A^{\pi^{old}})]$$

$$= \frac{1}{N} \sum_{i}^{N} E_{\vec{a}^t, s^t \sim \pi^{old}}[f(r(\theta_i)), A^{\pi^{old}}(s, a)]$$

其中 $f(r(\theta_i)), A^{\pi^{old}}(s, a) = \min(r_i A^{old}, \text{clip}(r_i, 1 - \epsilon, 1 + \epsilon) A^{\pi^{old}})$。

**Clip 操作**：torch.clamp(ratio, 1-clip, 1+clip)。如果新策略比旧策略变了太多（比如概率从 0.1 变成了 0.9），PPO 会强制截断这个更新幅度。这保证了训练过程不会因为一次坏数据而崩盘。

```python
new_log_prob = dist.log_prob(acts_flat)
old_dist_probs = torch.distributions.Categorical(old_probs_flat)
old_log_prob = old_dist_probs.log_prob(acts_flat)

ratio = torch.exp(new_log_prob - old_log_prob)
surr1 = ratio * adv_flat
surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_flat

act_loss = -torch.min(surr1, surr2) * valid_flat
ent_loss = dist.entropy() * valid_flat
```

### 3. 总损失函数 (Total Loss)
代码最终优化的目标是最小化以下三者的组合：

**Actor Loss**：让优势大的动作概率变大。

**Critic Loss**：让裁判（Critic）打分更准 (MSE Loss)。

**Entropy Loss**：减去熵（鼓励探索，防止过早陷入死板的策略）。

```python
# [Critic Loss]
            # 计算 Critic 预测值
            # 这是 Critic 网络在当前参数下，对特定状态-动作组合（s, a）所做出的长期价值预测
            q_pred_seq = self._forward_network_sequence(
                self.critic,
                mb_obs[:, :, i], mb_obs, mb_masks[:, :, i], mb_keys, h_init_mb,
                local_act=mb_acts_oh[:, :, i], global_act=mb_acts_oh
            )
            q_flat = q_pred_seq.reshape(-1)
            # Q_target是使用贝尔曼方程构造出来的、一个更可靠的、相对准确的价值估计
            ret_flat = mb_ret[:, :, i].reshape(-1)

            # 均方误差 MSE Loss: (Q_pred - Q_target)^2
            crit_loss = F.mse_loss(q_flat, ret_flat, reduction='none') * valid_flat

            # 总 Loss = Policy Loss + Critic Loss - Entropy
            loss_scalar += (act_loss.sum() + 0.5 * crit_loss.sum() - entropy_coef * ent_loss.sum())

        # 计算平均 Loss (除以有效样本数)
        valid_sum = mb_valid.sum() + 1e-8
        final_loss = loss_scalar / valid_sum

        # 反向传播
        self.optimizer.zero_grad()  # 清空旧梯度
        final_loss.backward()  # 计算新梯度

        # torch.nn.utils.clip_grad_norm_: 梯度裁剪
        # 如果梯度向量的范数超过 10.0，则按比例缩小，防止梯度爆炸 (Exploding Gradients)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        for actor in self.actors:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)

        self.optimizer.step()  # 更新参数，根据梯度调整Actor和Critic网络中的所有权重，即根据loss的数值改变各个动作的概率，核心！！！
        total_loss_log += final_loss.item()

# 返回平均 Loss 用于日志
return total_loss_log / (self.ppo_epoch * (BatchSize / self.mini_batch_size))
```