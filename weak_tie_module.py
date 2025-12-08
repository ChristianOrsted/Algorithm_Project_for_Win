import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.spatial.distance import cdist


class WeakTieGraph:

    def __init__(self, n_agents, obs_range=15.0, alpha_quantile=0.3):
        self.n_agents = n_agents
        self.obs_range = obs_range
        self.alpha_quantile = alpha_quantile

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
                comps = list(nx.connected_components(subgraph))
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
        degrees = dict(G.degree())
        valid_degrees = {k: v for k, v in degrees.items() if k in valid_indices}
        key_agent_idx = max(valid_degrees, key=valid_degrees.get) if valid_degrees else 0

        # 4. 计算联系强度 (Eq. 8)
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        H = np.zeros((n, n))
        tie_values = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i][j] = 1.0
                    continue
                if i not in valid_indices or j not in valid_indices:
                    H[i][j] = 0.0
                    continue

                D_i, D_j = degrees[i], degrees[j]
                W_ij = path_lengths[i].get(j, 9999)
                denominator = D_i + D_j + W_ij - 2
                strength = 1.0 / denominator if denominator > 0 else 0.0
                H[i][j] = strength
                tie_values.append(strength)

        # 5. 动态阈值划分
        current_alpha = np.quantile(tie_values, self.alpha_quantile) if tie_values else 0.0

        # H <= Alpha 为弱联系(保留), 否则为强联系(过滤)
        # 加上对角线(保留自己)
        mask_beta = (H <= current_alpha).astype(np.float32)
        np.fill_diagonal(mask_beta, 1.0)

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
        mask = mask_beta.unsqueeze(-1)  # (B, N, 1)
        sum_mask = mask.sum(dim=1) + 1e-6  # 归一化因子

        # --- 1. 弱联系聚合 (Obs) ---
        # 严格执行过滤：只聚合 mask=1 (弱联系) 的邻居
        weak_obs_masked = global_obs * mask
        weak_obs_agg = weak_obs_masked.sum(dim=1) / sum_mask

        # Key Agent (Obs)
        feat_dim = global_obs.shape[2]
        idx_exp = key_idx.view(batch_size, 1, 1).expand(-1, 1, feat_dim)
        key_obs = torch.gather(global_obs, 1, idx_exp).squeeze(1)

        # --- 2. 弱联系聚合 (Act) - 仅 Critic ---
        if self.use_actions:
            if local_act is None or global_act is None:
                raise ValueError("Critic mode requires actions!")

            # Eq 11: m_i 包含 beta * a^{-i}
            weak_act_masked = global_act * mask
            weak_act_agg = weak_act_masked.sum(dim=1) / sum_mask

            # Key Agent (Act)
            act_dim = global_act.shape[2]
            idx_act_exp = key_idx.view(batch_size, 1, 1).expand(-1, 1, act_dim)
            key_act = torch.gather(global_act, 1, idx_act_exp).squeeze(1)

            # 拼接: [Local(O+A), Weak(O+A), Key(O+A)]
            local_feat = torch.cat([local_obs, local_act], dim=-1)
            weak_feat = torch.cat([weak_obs_agg, weak_act_agg], dim=-1)
            key_feat = torch.cat([key_obs, key_act], dim=-1)
        else:
            # Actor 模式
            local_feat = local_obs
            weak_feat = weak_obs_agg
            key_feat = key_obs

        # --- 3. 融合 (Eq 13) ---
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
        # 1. 特征融合
        x = self.fusion(local_obs, global_obs, mask, key_idx, local_act, global_act)
        x = self.relu(x)

        # 2. GRU 记忆
        h_new = self.gru(x, hidden_state)

        # 3. 输出
        out = self.fc_out(h_new)
        return out, h_new