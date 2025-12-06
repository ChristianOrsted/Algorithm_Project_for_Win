import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from weak_tie_module import WeakTieNet


# --- RunningMeanStd & init_weights 工具类 ---
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.GRUCell):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


class WeakTieAgent:
    """
    [论文算法 2 复现 - 极速版]
    - 优化 1: 向量化 Counterfactual Baseline 计算
    - 优化 2: 向量化 PPO Update Loop (Time-Folding)，消除 Python 循环开销
    - 更新: 支持保存和加载胜率 (win_rate)
    """

    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=64, lr=5e-4,
                 gamma=0.99, gae_lambda=0.95, clip_param=0.2,
                 ppo_epoch=10, mini_batch_size=8):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"WeakTieAgent running on device: {self.device}")

        self.obs_norm = RunningMeanStd(shape=(obs_dim,))

        # Independent Actors
        self.actors = nn.ModuleList([
            WeakTieNet(obs_dim, act_dim, hidden_dim, act_dim, use_actions=False).to(self.device)
            for _ in range(n_agents)
        ])

        # Shared Critic
        self.critic = WeakTieNet(obs_dim, act_dim, hidden_dim, 1, use_actions=True).to(self.device)

        for actor in self.actors:
            actor.apply(init_weights)
        self.critic.apply(init_weights)

        all_params = list(self.critic.parameters())
        for actor in self.actors:
            all_params += list(actor.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr, eps=1e-5)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.n_agents, self.hidden_dim).to(self.device)

    def normalize_obs(self, obs, update_stats=True):
        flat_obs = obs.reshape(-1, self.obs_dim)
        if update_stats:
            self.obs_norm.update(flat_obs)
        norm_obs = (obs - self.obs_norm.mean) / (np.sqrt(self.obs_norm.var) + 1e-8)
        return norm_obs

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
                logits, h_new = self.actors[i](
                    local_obs=obs_t[:, i, :],
                    global_obs=obs_t,
                    mask=mask_t[:, i, :],
                    key_idx=key_t,
                    hidden_state=actor_hidden[:, i, :],
                    local_act=None, global_act=None
                )
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

    def _forward_network_sequence(self, network, local_obs, global_obs, mask, key, hidden_init, local_act=None,
                                  global_act=None):
        B, T, _ = local_obs.shape

        flat_local_obs = local_obs.reshape(B * T, -1)
        flat_global_obs = global_obs.reshape(B * T, self.n_agents, -1)
        flat_mask = mask.reshape(B * T, self.n_agents)
        flat_key = key.reshape(B * T)

        flat_local_act = local_act.reshape(B * T, -1) if local_act is not None else None
        flat_global_act = global_act.reshape(B * T, self.n_agents, -1) if global_act is not None else None

        feat = network.fusion(
            flat_local_obs, flat_global_obs, flat_mask, flat_key, flat_local_act, flat_global_act
        )
        feat = network.relu(feat)

        feat_seq = feat.view(B, T, -1)
        h = hidden_init
        h_seq = []

        for t in range(T):
            h = network.gru(feat_seq[:, t], h)
            h_seq.append(h)

        h_stack = torch.stack(h_seq, dim=1)

        flat_h = h_stack.view(B * T, -1)
        out = network.fc_out(flat_h)

        return out.view(B, T, -1)

    def update_batch(self, buffer_list, entropy_coef=0.01):
        batch_obs_list, batch_acts_list, batch_rews_list, batch_dones_list = [], [], [], []
        batch_avails_list, batch_masks_list, batch_keys_list, batch_old_probs_list = [], [], [], []
        batch_lens = []

        for episode_data in buffer_list:
            buf = episode_data[0]
            t_len = len(buf['obs'])
            obs = self.normalize_obs(np.array(buf['obs']).reshape(t_len, self.n_agents, -1), update_stats=False)
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
        MaxTime = max(batch_lens)
        pad_obs = torch.zeros(BatchSize, MaxTime, self.n_agents, self.obs_dim).to(self.device)
        pad_acts = torch.zeros(BatchSize, MaxTime, self.n_agents, dtype=torch.long).to(self.device)
        pad_rews = torch.zeros(BatchSize, MaxTime, self.n_agents, 1).to(self.device)
        pad_dones = torch.zeros(BatchSize, MaxTime, self.n_agents, 1).to(self.device)
        pad_avails = torch.zeros(BatchSize, MaxTime, self.n_agents, self.act_dim).to(self.device)
        pad_masks = torch.zeros(BatchSize, MaxTime, self.n_agents, self.n_agents).to(self.device)
        pad_keys = torch.zeros(BatchSize, MaxTime, 1, dtype=torch.long).to(self.device)
        pad_old_probs = torch.zeros(BatchSize, MaxTime, self.n_agents, self.act_dim).to(self.device)
        valid_mask = torch.zeros(BatchSize, MaxTime).to(self.device)
        pad_avails[..., 0] = 1.0
        pad_old_probs[..., 0] = 1.0

        for i, t_len in enumerate(batch_lens):
            pad_obs[i, :t_len] = batch_obs_list[i].to(self.device)
            pad_acts[i, :t_len] = batch_acts_list[i].to(self.device)
            pad_rews[i, :t_len] = batch_rews_list[i].to(self.device)
            pad_dones[i, :t_len] = batch_dones_list[i].to(self.device)
            pad_avails[i, :t_len] = batch_avails_list[i].to(self.device)
            pad_masks[i, :t_len] = batch_masks_list[i].to(self.device)
            pad_keys[i, :t_len] = batch_keys_list[i].to(self.device)
            pad_old_probs[i, :t_len] = batch_old_probs_list[i].to(self.device)
            valid_mask[i, :t_len] = 1

        valid_mask_agent = valid_mask.unsqueeze(-1).expand(-1, -1, self.n_agents)
        pad_acts_onehot = F.one_hot(pad_acts, num_classes=self.act_dim).float()

        with torch.no_grad():
            current_probs_list = []
            h_init = torch.zeros(BatchSize, self.hidden_dim).to(self.device)
            for i in range(self.n_agents):
                logits_seq = self._forward_network_sequence(
                    self.actors[i], pad_obs[:, :, i], pad_obs, pad_masks[:, :, i], pad_keys, h_init
                )
                logits_seq[pad_avails[:, :, i] == 0] = -1e10
                current_probs_list.append(F.softmax(logits_seq, dim=-1))
            current_probs = torch.stack(current_probs_list, dim=2)

            q_taken_list = []
            critic_hiddens = [torch.zeros(BatchSize, self.hidden_dim).to(self.device) for _ in range(self.n_agents)]
            critic_history_tensor = torch.zeros(BatchSize, MaxTime, self.n_agents, self.hidden_dim).to(self.device)

            for t in range(MaxTime):
                q_t_list = []
                for i in range(self.n_agents):
                    critic_history_tensor[:, t, i] = critic_hiddens[i]
                    q, h_new = self.critic(
                        local_obs=pad_obs[:, t, i], global_obs=pad_obs[:, t],
                        mask=pad_masks[:, t, i], key_idx=pad_keys[:, t],
                        hidden_state=critic_hiddens[i],
                        local_act=pad_acts_onehot[:, t, i], global_act=pad_acts_onehot[:, t]
                    )
                    critic_hiddens[i] = h_new
                    q_t_list.append(q)
                q_taken_list.append(torch.stack(q_t_list, dim=1))
            q_taken = torch.stack(q_taken_list, dim=1).squeeze(-1)

            baseline_v = torch.zeros_like(q_taken)
            all_actions_eye = torch.eye(self.act_dim).to(self.device)
            batch_all_actions = all_actions_eye.unsqueeze(0).repeat(BatchSize, 1, 1).view(-1, self.act_dim)

            for t in range(MaxTime):
                base_global_act = pad_acts_onehot[:, t]
                for i in range(self.n_agents):
                    obs_rep = pad_obs[:, t, i].repeat_interleave(self.act_dim, dim=0)
                    g_obs_rep = pad_obs[:, t].repeat_interleave(self.act_dim, dim=0)
                    mask_rep = pad_masks[:, t, i].repeat_interleave(self.act_dim, dim=0)
                    key_rep = pad_keys[:, t].repeat_interleave(self.act_dim, dim=0)
                    h_rep = critic_history_tensor[:, t, i].repeat_interleave(self.act_dim, dim=0)

                    global_act_rep = base_global_act.repeat_interleave(self.act_dim, dim=0).clone()
                    global_act_rep[:, i, :] = batch_all_actions

                    q_values_flat, _ = self.critic(
                        local_obs=obs_rep, global_obs=g_obs_rep,
                        mask=mask_rep, key_idx=key_rep,
                        hidden_state=h_rep,
                        local_act=batch_all_actions, global_act=global_act_rep
                    )
                    q_values = q_values_flat.view(BatchSize, self.act_dim)
                    baseline_v[:, t, i] = (current_probs[:, t, i] * q_values).sum(dim=-1)

            advantages = q_taken - baseline_v
            baseline_v_next = torch.cat([baseline_v[:, 1:], torch.zeros_like(baseline_v[:, :1])], dim=1)
            returns = pad_rews.squeeze(-1) + self.gamma * baseline_v_next * (1 - pad_dones.squeeze(-1))

        total_loss_log = 0
        indices = np.arange(BatchSize)

        for _ in range(self.ppo_epoch):
            np.random.shuffle(indices)
            for start_idx in range(0, BatchSize, self.mini_batch_size):
                mb_idx = indices[start_idx: start_idx + self.mini_batch_size]
                curr_mb_size = len(mb_idx)

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
                    logits_seq = self._forward_network_sequence(
                        self.actors[i],
                        mb_obs[:, :, i], mb_obs, mb_masks[:, :, i], mb_keys, h_init_mb
                    )

                    logits_flat = logits_seq.reshape(-1, self.act_dim)
                    acts_flat = mb_acts[:, :, i].reshape(-1)
                    avails_flat = mb_avails[:, :, i].reshape(-1, self.act_dim)
                    old_probs_flat = mb_old_probs[:, :, i].reshape(-1, self.act_dim)
                    adv_flat = mb_adv[:, :, i].reshape(-1)
                    valid_flat = mb_valid[:, :, i].reshape(-1)

                    logits_flat[avails_flat == 0] = -1e10
                    probs_flat = F.softmax(logits_flat, dim=-1)
                    dist = torch.distributions.Categorical(probs_flat)

                    new_log_prob = dist.log_prob(acts_flat)
                    old_dist_probs = torch.distributions.Categorical(old_probs_flat)
                    old_log_prob = old_dist_probs.log_prob(acts_flat)

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * adv_flat
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_flat

                    act_loss = -torch.min(surr1, surr2) * valid_flat
                    ent_loss = dist.entropy() * valid_flat

                    q_pred_seq = self._forward_network_sequence(
                        self.critic,
                        mb_obs[:, :, i], mb_obs, mb_masks[:, :, i], mb_keys, h_init_mb,
                        local_act=mb_acts_oh[:, :, i], global_act=mb_acts_oh
                    )
                    q_flat = q_pred_seq.reshape(-1)
                    ret_flat = mb_ret[:, :, i].reshape(-1)

                    crit_loss = F.mse_loss(q_flat, ret_flat, reduction='none') * valid_flat

                    loss_scalar += (act_loss.sum() + 0.5 * crit_loss.sum() - entropy_coef * ent_loss.sum())

                valid_sum = mb_valid.sum() + 1e-8
                final_loss = loss_scalar / valid_sum

                self.optimizer.zero_grad()
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                for actor in self.actors:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                self.optimizer.step()
                total_loss_log += final_loss.item()

        return total_loss_log / (self.ppo_epoch * (BatchSize / self.mini_batch_size))

    # === [关键修改] 支持传入 win_rate 参数 ===
    def save_model(self, path, episode, win_rate=None):
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir) and save_dir != '':
            os.makedirs(save_dir)
        actors_state = [actor.state_dict() for actor in self.actors]

        # 构建保存字典
        save_dict = {
            'episode': episode,
            'actors': actors_state,
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'obs_norm': {'mean': self.obs_norm.mean, 'var': self.obs_norm.var, 'count': self.obs_norm.count}
        }

        # 如果有胜率数据，也存进去
        if win_rate is not None:
            save_dict['win_rate'] = win_rate

        torch.save(save_dict, path)
        print(f"模型已保存到: {path} (胜率记录: {win_rate if win_rate else '无'})")

    def load_model(self, path):
        start_episode = 0
        if os.path.exists(path):
            try:
                # 🔧 修复：添加 weights_only=False
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