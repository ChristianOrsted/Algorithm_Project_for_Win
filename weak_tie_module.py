import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.spatial.distance import cdist


class WeakTieGraph:
    """
    [论文 3.3 节] 弱联系图构建模块
    负责计算 mask_beta 和 key_agent_idx
    """

    def __init__(self, n_agents, obs_range=15.0, alpha_quantile=0.3):
        self.n_agents = n_agents   # 智能体数量
        self.obs_range = obs_range   # 视野范围
        self.alpha_quantile = alpha_quantile   # 弱联系阈值 α

    def compute_graph_info(self, agent_positions, alive_mask=None):
        n = self.n_agents
        G = nx.Graph()
        G.add_nodes_from(range(n))

        valid_indices = [i for i in range(n) if (alive_mask is None or alive_mask[i])]

        # 1. 基础建边 (Definition 1)
        if len(valid_indices) > 0:
            pos_valid = agent_positions[valid_indices]
            dists = cdist(pos_valid, pos_valid)
            for i_idx, i in enumerate(valid_indices):
                for j_idx, j in enumerate(valid_indices):
                    if i < j and dists[i_idx, j_idx] <= self.obs_range:
                        G.add_edge(i, j)

        # 2. 保证连通性 (算法 1 修正：最近邻连接)
        if len(valid_indices) > 1:
            subgraph = G.subgraph(valid_indices).copy()
            if not nx.is_connected(subgraph):
                comps = list(nx.connected_components(subgraph)) # 获取所有孤岛(分量)
                for k in range(len(comps) - 1):
                    comp1 = list(comps[k])
                    comp2 = list(comps[k + 1])
                    pos1 = agent_positions[comp1]
                    pos2 = agent_positions[comp2]
                    # 计算两个分量间的最短距离
                    d_mat = cdist(pos1, pos2)
                    min_idx = np.unravel_index(np.argmin(d_mat), d_mat.shape)
                    G.add_edge(comp1[min_idx[0]], comp2[min_idx[1]])

        # 3. Key Agent (Definition 5)
        degrees = dict(G.degree())   # 计算每个点的度 (连接数)
        valid_degrees = {k: v for k, v in degrees.items() if k in valid_indices}   # 过滤掉死人的度
        key_agent_idx = max(valid_degrees, key=valid_degrees.get) if valid_degrees else 0  # 找度最大的那个 ID，就是 Key Agent

        # 4. 计算联系强度 (Eq. 8)
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))  # 算出所有点对的最短路径跳数
        H = np.zeros((n, n))  # 初始化强度矩阵
        tie_values = []   # 存所有强度值，用于后续算分位数

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

        # 5. 动态阈值划分
        # 算出强度的 30% 分位数 (alpha_quantile)
        current_alpha = np.quantile(tie_values, self.alpha_quantile) if tie_values else 0.0

        # 生成 Mask：H <= Alpha 为弱联系(保留), 否则为强联系(过滤)
        # 加上对角线(保留自己)
        mask_beta = (H <= current_alpha).astype(np.float32)
        np.fill_diagonal(mask_beta, 1.0)

        # 再次确保死人被 Mask 掉
        if alive_mask is not None:
            dead_indices = np.where(alive_mask == 0)[0]
            mask_beta[dead_indices, :] = 0
            mask_beta[:, dead_indices] = 0

        return mask_beta, key_agent_idx


class WeakTieFusionLayer(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, use_actions=False):
        super(WeakTieFusionLayer, self).__init__()
        self.use_actions = use_actions

        # 输入维度:
        # Actor模式 (仅Obs): Local + Weak(Obs) + Key(Obs) -> 3 * obs_dim
        # Critic模式 (Obs+Act): Local(O+A) + Weak(O+A) + Key(O+A) -> 3 * (obs_dim + act_dim)
        if self.use_actions:
            input_dim = 3 * (obs_dim + act_dim)
        else:
            input_dim = 3 * obs_dim

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

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