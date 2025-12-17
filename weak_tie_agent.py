import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from weak_tie_module import WeakTieNet


# RunningMeanStd & init_weights 工具类
class RunningMeanStd:
    """
    动态标准化工具
    RL 算法对输入数据的尺度非常敏感。这个类实现了 Welford 在线算法，
    可以在不保存所有历史数据的情况下，动态计算并更新观测值的均值 (mean) 和方差 (var)。
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')  # 初始化均值，float64 保证精度
        self.var = np.ones(shape, 'float64')  # 初始化方差为 1，避免除以 0
        self.count = epsilon  # 计数器，设为极小值防止分母为 0

    def update(self, x):
        # np.mean(axis=0): 计算当前 batch 数据在每个特征维度上的均值
        batch_mean = np.mean(x, axis=0)
        # np.var(axis=0): 计算当前 batch 数据在每个特征维度上的方差
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]  # 当前 batch 的样本数量
        # 调用核心更新函数
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # 这是一个标准的 Welford 算法合并两个高斯分布的公式
        # 1. 计算新旧均值的差值
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count  # 更新总样本数

        # 2. 更新全局均值
        new_mean = self.mean + delta * batch_count / tot_count

        # 3. 更新全局方差
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        # 计算合并后的二阶中心矩
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # 4. 保存新状态
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def init_weights(m):
    """
    权重初始化函数
    递归地对网络层进行初始化。
    为什么用正交初始化 (Orthogonal)?
    在 RNN 和 深度强化学习中，正交初始化能保持梯度的模长在传播过程中相对稳定，
    有效防止梯度消失或爆炸。
    """
    if isinstance(m, nn.Linear):  # 如果是全连接层
        # nn.init.orthogonal_: 构建正交矩阵作为权重
        # gain=np.sqrt(2): 配合 ReLU 激活函数使用，因为 ReLU 会砍掉一半的激活值，所以需要放大权重以保持方差
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)  # 偏置项初始化为 0
    elif isinstance(m, nn.GRUCell):  # 如果是 GRU 单元
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


class WeakTieAgent:
    """
    WeakTieAgent 核心类
    实现了基于 PPO (Proximal Policy Optimization) 的多智能体算法。
    """

    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=64, lr=5e-4,
                 gamma=0.99, gae_lambda=0.95, clip_param=0.2,
                 ppo_epoch=10, mini_batch_size=8):
        # === 保存超参数 ===
        self.n_agents = n_agents  # 智能体数量
        self.obs_dim = obs_dim  # 观测维度
        self.act_dim = act_dim  # 动作空间维度 (Discrete)
        self.hidden_dim = hidden_dim  # RNN 隐藏层维度
        self.gamma = gamma  # 折扣因子 (Discount Factor)
        self.gae_lambda = gae_lambda  # GAE 参数 (用于平衡偏差和方差)
        self.clip_param = clip_param  # PPO 截断参数 (通常 0.1 ~ 0.2)
        self.ppo_epoch = ppo_epoch  # 每次更新迭代次数
        self.mini_batch_size = mini_batch_size  # 小批量大小

        # 检查是否有 GPU，优先使用 GPU 训练
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"WeakTieAgent running on device: {self.device}")

        # 实例化观测值标准化工具
        self.obs_norm = RunningMeanStd(shape=(obs_dim,))

        # === 初始化 Actor 网络 ===
        # nn.ModuleList: 将多个子网络存在一个列表中，并注册为模型参数，方便后续 .parameters() 调用
        # 这里为每个智能体创建一个独立的 WeakTieNet (独立参数)
        self.actors = nn.ModuleList([
            # use_actions=False: Actor 只根据观测选动作，不需要输入动作
            WeakTieNet(obs_dim, act_dim, hidden_dim, act_dim, use_actions=False).to(self.device)
            for _ in range(n_agents)
        ])

        # === 初始化 Critic 网络 ===
        # Shared Critic: 这是一个中心化的 Critic，所有智能体共享
        # use_actions=True: Critic 这是一个 Q-Critic 或 V-Critic 变体，这里设计为输入动作
        # output_dim=1: 输出一个标量值 (Value)
        self.critic = WeakTieNet(obs_dim, act_dim, hidden_dim, 1, use_actions=True).to(self.device)

        # 对网络进行权重初始化
        for actor in self.actors:
            actor.apply(init_weights)  # .apply() 会递归调用 init_weights 函数
        self.critic.apply(init_weights)

        # === 初始化优化器 ===
        # 将 Actor 和 Critic 的所有参数打包给一个 Adam 优化器
        all_params = list(self.critic.parameters())
        for actor in self.actors:
            all_params += list(actor.parameters())
        # eps=1e-5: 防止分母为 0 的数值稳定参数
        self.optimizer = optim.Adam(all_params, lr=lr, eps=1e-5)

    def init_hidden(self, batch_size=1):
        """生成全 0 的初始隐藏状态 h0"""
        # 形状: [batch_size, n_agents, hidden_dim]
        # .to(self.device): 确保 Tensor 在 GPU 上
        return torch.zeros(batch_size, self.n_agents, self.hidden_dim).to(self.device)

    def normalize_obs(self, obs, update_stats=True):
        """对观测值进行标准化: (x - mean) / std"""
        # 将 obs 展平为 [N, obs_dim] 以便统计
        flat_obs = obs.reshape(-1, self.obs_dim)
        if update_stats:
            self.obs_norm.update(flat_obs)  # 更新均值和方差
        # 1e-8 是 epsilon，防止除以 0
        norm_obs = (obs - self.obs_norm.mean) / (np.sqrt(self.obs_norm.var) + 1e-8)
        return norm_obs

    def select_action(self, obs, avail_actions, mask, key_idx, actor_hidden, deterministic=False):
        """
        [推理/采样函数]
        在与环境交互时调用。
        参数:
            obs: 当前观测
            avail_actions: 动作掩码 (1表示可用，0表示不可用)
            mask: 弱联系掩码 (Graph Mask)
            key_idx: 关键智能体索引
            actor_hidden: 上一步的 RNN 隐状态
            deterministic: True为贪婪策略(评估)，False为随机采样(训练)
        """
        # 将输入数据转换为 numpy 数组
        obs = np.array(obs)
        avail_actions = np.array(avail_actions)
        mask = np.array(mask)
        key_idx = np.array(key_idx)

        # 如果输入维度是 [n_agents, dim] (单局游戏)，手动增加 Batch 维度 -> [1, n_agents, dim]
        if obs.ndim == 2:
            # obs[None, ...] 等价于 np.expand_dims(obs, axis=0)
            obs = obs[None, ...]
            avail_actions = avail_actions[None, ...]
            mask = mask[None, ...]
            key_idx = key_idx[None, ...]

        # 归一化观测值，并更新统计数据 (训练时)
        obs = self.normalize_obs(obs, update_stats=True)

        # 将 numpy 转为 PyTorch Tensor 并移至 GPU
        obs_t = torch.FloatTensor(obs).to(self.device)
        avail_t = torch.FloatTensor(avail_actions).to(self.device)
        mask_t = torch.FloatTensor(mask).to(self.device)
        key_t = torch.LongTensor(key_idx).to(self.device)

        actions_list = []  # 存每个 agent 的动作
        probs_list = []  # 存每个 agent 的动作概率
        new_hidden_list = []  # 存每个 agent 新的 hidden state

        # torch.no_grad(): 上下文管理器，在此块内计算不构建计算图，节省显存并加速 (推理模式)
        with torch.no_grad():
            for i in range(self.n_agents):
                # 前向传播: 调用 WeakTieNet
                # global_obs 传入完整 obs_t，网络内部会处理
                logits, h_new = self.actors[i](
                    local_obs=obs_t[:, i, :],  # Agent i 自己的观测
                    global_obs=obs_t,  # 全局观测
                    mask=mask_t[:, i, :],  # Agent i 对应的 Graph Mask
                    key_idx=key_t,  # Key Agent Index
                    hidden_state=actor_hidden[:, i, :],  # Agent i 的隐状态
                    local_act=None, global_act=None
                )

                # [Action Masking] 处理不可用动作
                avail_i = avail_t[:, i, :]
                # 将不可用动作的 Logits 设为负无穷大
                logits[avail_i == 0] = -1e10

                # F.softmax: 将 Logits 转化为概率分布 (归一化，和为1)
                probs = F.softmax(logits, dim=-1)

                if deterministic:
                    # 贪婪模式: 直接选概率最大的动作索引
                    action = probs.argmax(dim=-1)
                else:
                    # 随机模式: 根据概率分布进行采样
                    # torch.distributions.Categorical: 创建分类分布对象
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()  # 按概率采样

                actions_list.append(action)
                probs_list.append(probs)
                new_hidden_list.append(h_new)

        # torch.stack: 将列表中的 Tensor 沿着新维度拼接 (dim=1 即 Agent 维度)
        # 结果形状: [Batch, n_agents, ...]
        actions = torch.stack(actions_list, dim=1)
        probs = torch.stack(probs_list, dim=1)
        new_hidden = torch.stack(new_hidden_list, dim=1)

        # 转回 numpy 返回给环境 (cpu().numpy() 将 GPU Tensor 转为 CPU numpy)
        return actions.cpu().numpy()[0], probs.cpu().numpy()[0], new_hidden

    def _forward_network_sequence(self, network, local_obs, global_obs, mask, key, hidden_init, local_act=None,
                                  global_act=None):
        """
        [Time-Folding 优化函数]
        由于 RNN 无法并行处理时间步 (t 依赖 t-1)，如果直接写 python 循环会很慢。
        优化思路: 将 (Batch, Time) 两个维度合并为 (Batch*Time)，先通过所有非 RNN 层 (MLP/CNN)，
        这些层是时间独立的，可以利用 GPU 并行加速。最后再 Reshape 回来进 GRU 循环。
        """
        B, T, _ = local_obs.shape  # 获取 Batch 和 Time 维度

        # 1. 维度折叠 (Merge Batch and Time)
        # .reshape(B*T, -1): 将前两维合并，-1 表示自动推导剩余维度
        flat_local_obs = local_obs.reshape(B * T, -1)
        flat_global_obs = global_obs.reshape(B * T, self.n_agents, -1)
        flat_mask = mask.reshape(B * T, self.n_agents)
        flat_key = key.reshape(B * T)

        flat_local_act = local_act.reshape(B * T, -1) if local_act is not None else None
        flat_global_act = global_act.reshape(B * T, self.n_agents, -1) if global_act is not None else None

        # 2. 批量并行特征提取 (WeakTieFusionLayer)
        # 此时 batch size 为 B*T，非常大，能跑满 GPU
        feat = network.fusion(
            flat_local_obs, flat_global_obs, flat_mask, flat_key, flat_local_act, flat_global_act
        )
        feat = network.relu(feat)  # 激活函数

        # 3. 维度恢复 (Reshape back for RNN)
        feat_seq = feat.view(B, T, -1)

        # 4. GRU 循环
        h = hidden_init
        h_seq = []

        for t in range(T):
            # feat_seq[:, t] 是 t 时刻所有 Batch 的输入
            h = network.gru(feat_seq[:, t], h)
            h_seq.append(h)  # 记录每个时刻的隐状态

        # stack: 拼接成 [B, T, Hidden]
        h_stack = torch.stack(h_seq, dim=1)

        # 5. 输出层 (再次折叠)
        flat_h = h_stack.view(B * T, -1)
        out = network.fc_out(flat_h)

        # 最终输出 [B, T, OutputDim]
        return out.view(B, T, -1)

    def update_batch(self, buffer_list, entropy_coef=0.01):
        """
        [PPO 训练主函数]
        参数:
            buffer_list: 包含多个 Episode 数据的列表
            entropy_coef: 熵正则化系数 (鼓励探索)
        """
        # ==========================
        # 1. 数据对齐 (Padding)
        # ==========================
        # 不同的 Episode 长度可能不同，需要 Pad 成矩阵才能并行计算
        batch_obs_list, batch_acts_list, batch_rews_list, batch_dones_list = [], [], [], []
        batch_avails_list, batch_masks_list, batch_keys_list, batch_old_probs_list = [], [], [], []
        batch_lens = []

        for episode_data in buffer_list:
            buf = episode_data[0]  # 取出 buffer 字典
            t_len = len(buf['obs'])  # 获取当前 episode 长度

            # 归一化 Obs ，让数据的平均值接近 0，标准差接近 1
            # 注意 update_stats=False，训练时不再更新均值方差，只用冻结的统计量
            obs = self.normalize_obs(np.array(buf['obs']).reshape(t_len, self.n_agents, -1), update_stats=False)

            # 转换为 Tensor 并加入列表
            batch_obs_list.append(torch.FloatTensor(obs))
            batch_acts_list.append(torch.LongTensor(np.array(buf['acts']).reshape(t_len, self.n_agents)))
            batch_rews_list.append(torch.FloatTensor(np.array(buf['rewards']).reshape(t_len, self.n_agents, 1)))
            batch_dones_list.append(torch.FloatTensor(np.array(buf['dones']).reshape(t_len, self.n_agents, 1)))
            batch_avails_list.append(torch.FloatTensor(np.array(buf['avails']).reshape(t_len, self.n_agents, -1)))
            batch_masks_list.append(
                torch.FloatTensor(np.array(buf['masks']).reshape(t_len, self.n_agents, self.n_agents)))
            batch_keys_list.append(torch.LongTensor(np.array(buf['keys']).reshape(t_len, 1)))
            batch_old_probs_list.append(torch.FloatTensor(np.array(buf['probs']).reshape(t_len, self.n_agents, -1)))
            batch_lens.append(t_len)

        BatchSize = len(batch_lens)
        MaxTime = max(batch_lens)  # 找出最长的 Episode 长度作为基准

        # 初始化全 0 的 Tensor (Padding)
        # .to(self.device): 创建在 GPU 上
        pad_obs = torch.zeros(BatchSize, MaxTime, self.n_agents, self.obs_dim).to(self.device)
        pad_acts = torch.zeros(BatchSize, MaxTime, self.n_agents, dtype=torch.long).to(self.device)
        pad_rews = torch.zeros(BatchSize, MaxTime, self.n_agents, 1).to(self.device)
        pad_dones = torch.zeros(BatchSize, MaxTime, self.n_agents, 1).to(self.device)
        pad_avails = torch.zeros(BatchSize, MaxTime, self.n_agents, self.act_dim).to(self.device)
        pad_masks = torch.zeros(BatchSize, MaxTime, self.n_agents, self.n_agents).to(self.device)
        pad_keys = torch.zeros(BatchSize, MaxTime, 1, dtype=torch.long).to(self.device)
        pad_old_probs = torch.zeros(BatchSize, MaxTime, self.n_agents, self.act_dim).to(self.device)

        # valid_mask: 用于标记哪些是真实数据(1)，哪些是填充数据(0)
        valid_mask = torch.zeros(BatchSize, MaxTime).to(self.device)

        # 填充默认值，防止除0或概率为0的计算错误
        pad_avails[..., 0] = 1.0  # 默认第一个动作可用
        pad_old_probs[..., 0] = 1.0  # 默认概率不为0

        # 将数据填入 Tensor
        for i, t_len in enumerate(batch_lens):
            pad_obs[i, :t_len] = batch_obs_list[i].to(self.device)
            pad_acts[i, :t_len] = batch_acts_list[i].to(self.device)
            pad_rews[i, :t_len] = batch_rews_list[i].to(self.device)
            pad_dones[i, :t_len] = batch_dones_list[i].to(self.device)
            pad_avails[i, :t_len] = batch_avails_list[i].to(self.device)
            pad_masks[i, :t_len] = batch_masks_list[i].to(self.device)
            pad_keys[i, :t_len] = batch_keys_list[i].to(self.device)
            pad_old_probs[i, :t_len] = batch_old_probs_list[i].to(self.device)  #依据旧策略输出的动作概率
            valid_mask[i, :t_len] = 1  # 设置有效位

        # valid_mask 扩展维度以匹配 agent
        valid_mask_agent = valid_mask.unsqueeze(-1).expand(-1, -1, self.n_agents)
        # F.one_hot: 将整数动作索引转换为独热编码 (One-Hot Encoding)
        # 例如 动作2 -> [0, 0, 1, 0...]，供 Critic 网络作为输入
        pad_acts_onehot = F.one_hot(pad_acts, num_classes=self.act_dim).float()

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

        # ==========================
        # 3. PPO 更新循环
        # ==========================
        total_loss_log = 0
        indices = np.arange(BatchSize)  # 生成索引数组用于打乱

        for _ in range(self.ppo_epoch):  # 重复利用数据更新 ppo_epoch 次
            np.random.shuffle(indices)  # 打乱数据

            # Mini-Batch 循环
            for start_idx in range(0, BatchSize, self.mini_batch_size):
                mb_idx = indices[start_idx: start_idx + self.mini_batch_size]
                curr_mb_size = len(mb_idx)

                # 切片取出当前 Mini-Batch 的数据
                mb_obs = pad_obs[mb_idx]
                mb_acts = pad_acts[mb_idx]
                mb_avails = pad_avails[mb_idx]
                mb_masks = pad_masks[mb_idx]
                mb_keys = pad_keys[mb_idx]
                mb_old_probs = pad_old_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_valid = valid_mask_agent[mb_idx]
                mb_acts_oh = pad_acts_onehot[mb_idx]

                loss_scalar = 0
                h_init_mb = torch.zeros(curr_mb_size, self.hidden_dim).to(self.device)

                for i in range(self.n_agents):
                    # 前向传播 (这次带梯度 Gradients)
                    logits_seq = self._forward_network_sequence(
                        self.actors[i],
                        mb_obs[:, :, i], mb_obs, mb_masks[:, :, i], mb_keys, h_init_mb
                    )

                    # 展平所有维度为 [N, ...] 方便计算 Loss
                    logits_flat = logits_seq.reshape(-1, self.act_dim)
                    acts_flat = mb_acts[:, :, i].reshape(-1)
                    avails_flat = mb_avails[:, :, i].reshape(-1, self.act_dim)
                    old_probs_flat = mb_old_probs[:, :, i].reshape(-1, self.act_dim)
                    adv_flat = mb_adv[:, :, i].reshape(-1)
                    valid_flat = mb_valid[:, :, i].reshape(-1)

                    # Action Masking
                    logits_flat[avails_flat == 0] = -1e10
                    probs_flat = F.softmax(logits_flat, dim=-1)
                    dist = torch.distributions.Categorical(probs_flat)

                    # [PPO Loss 核心公式]
                    # 1. 计算新动作 Log 概率: log pi_new(a|s)
                    new_log_prob = dist.log_prob(acts_flat)

                    # 2. 计算旧动作 Log 概率 (这里简化了，直接用保存的 prob 算，或者应该从 buffer 里取 log_prob)
                    old_dist_probs = torch.distributions.Categorical(old_probs_flat)
                    old_log_prob = old_dist_probs.log_prob(acts_flat)

                    # 3. 比率 Ratio = exp(log_new - log_old) = pi_new / pi_old
                    # 计算当前策略与数据采集时的旧策略之间在实际动作上的概率比率
                    ratio = torch.exp(new_log_prob - old_log_prob)

                    # 4. Surrogate Objectives
                    # surr1是用新策略来评估旧数据，能获得的期望优势
                    surr1 = ratio * adv_flat
                    # torch.clamp: 将 ratio 截断在 [1-eps, 1+eps] 之间
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_flat

                    # PPO Policy Loss = -min(surr1, surr2) (取负因为是梯度下降)
                    # 我们希望最大化优势。但如果策略变化太大，函数会选择被截断的surr2，从而限制更新的幅度
                    act_loss = -torch.min(surr1, surr2) * valid_flat

                    # Entropy Bonus: 鼓励策略分布具有较高的熵，防止过早收敛
                    ent_loss = dist.entropy() * valid_flat

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

    # === 保存与加载 ===
    def save_model(self, path, episode, win_rate=None):
        """保存模型状态字典"""
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir) and save_dir != '':
            os.makedirs(save_dir)
        # 获取所有 actor 的 state_dict
        actors_state = [actor.state_dict() for actor in self.actors]

        # 构建保存字典
        save_dict = {
            'episode': episode,
            'actors': actors_state,
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 也要保存 RunningMeanStd 的统计量，否则加载后归一化会出错
            'obs_norm': {'mean': self.obs_norm.mean, 'var': self.obs_norm.var, 'count': self.obs_norm.count}
        }

        if win_rate is not None:
            save_dict['win_rate'] = win_rate

        torch.save(save_dict, path)  # 序列化到磁盘
        print(f"模型已保存到: {path} (胜率记录: {win_rate if win_rate else '无'})")

    def load_model(self, path):
        """加载模型"""
        start_episode = 0
        if os.path.exists(path):
            try:
                # map_location: 确保加载到正确的设备 (CPU/GPU)
                # weights_only=False: 允许加载复杂的字典结构 (新版 PyTorch 安全特性)
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                if 'actors' in ckpt:
                    for i, actor_state in enumerate(ckpt['actors']):
                        self.actors[i].load_state_dict(actor_state)
                self.critic.load_state_dict(ckpt['critic'])
                if 'optimizer' in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt['optimizer'])
                    except:
                        pass
                if 'obs_norm' in ckpt:
                    self.obs_norm.mean = ckpt['obs_norm']['mean']
                    self.obs_norm.var = ckpt['obs_norm']['var']
                    self.obs_norm.count = ckpt['obs_norm']['count']
                if 'episode' in ckpt:
                    start_episode = ckpt['episode']
                print(f"成功加载: {path}")
            except Exception as e:
                print(f"加载失败: {e}")
        return start_episode