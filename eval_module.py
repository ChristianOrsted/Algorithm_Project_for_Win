import os
import warnings
import torch
import numpy as np
from weak_tie_env import WeakTieStarCraft2Env
from weak_tie_agent import WeakTieAgent
from weak_tie_module import WeakTieGraph

# 过滤警告并设置 SC2 路径
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["SC2PATH"] = "D:\Program Files (x86)\StarCraft II"


def evaluate_model(model_path, map_name="1c3s5z", n_episodes=50):
    """
    评估模型性能
    :param n_episodes: 评估局数，默认改为 50
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return 0.0, 0.0

    print(f"正在加载环境: {map_name} ...")
    try:
        env = WeakTieStarCraft2Env(map_name=map_name)
    except Exception as e:
        print(f"环境创建失败: {e}")
        return 0.0, 0.0

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    obs_dim = env_info["obs_shape"]
    n_actions = env_info["n_actions"]

    # =================================================================
    # 关键参数配置 (必须与 train_smac.py 保持一致)
    # =================================================================
    if map_name in ["1c3s5z", "50m", "10m_vs_11m"]:
        HIDDEN_DIM = 256
    else:
        HIDDEN_DIM = 128

    print(f"初始化 Agent (Hidden Dim: {HIDDEN_DIM}, Independent Actors)...")

    # 初始化 Agent (适配独立 Actor 结构)
    agent = WeakTieAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=n_actions,
        hidden_dim=HIDDEN_DIM,
        # 评估模式下 lr 等参数不影响，但为了初始化需要填入
        lr=0.0003,
        gamma=0.99
    )

    # 加载模型权重
    start_ep = agent.load_model(model_path)
    print(f"模型加载完成 (训练轮数: {start_ep})")

    # 初始化弱联系图
    # 注意：obs_range 必须与训练时一致
    wt_graph = WeakTieGraph(n_agents, obs_range=15.0, alpha_quantile=0.3)

    wins = 0
    total_reward = 0

    print(f"开始评估 {n_episodes} 局...")

    with torch.no_grad():  # 确保不计算梯度，节省显存
        for ep in range(n_episodes):
            obs, state = env.reset()
            terminated = False
            episode_reward = 0

            # 初始化 RNN 隐藏状态
            actor_hidden = agent.init_hidden(batch_size=1)

            step = 0
            while not terminated:
                avail_actions = env.get_avail_actions()
                positions = env.get_all_unit_positions()

                # 获取存活状态 (用于图计算)
                alive_mask = np.array([1 if env.agents[i].health > 0 else 0 for i in range(n_agents)])

                # 1. 计算图结构 (Weak Tie Graph)
                mask_beta, key_agent_idx = wt_graph.compute_graph_info(positions, alive_mask)

                # 2. 决策 (关键：deterministic=True)
                # 评估时必须使用确定性策略（选概率最大的动作），这是胜率高的关键
                actions, probs, actor_hidden = agent.select_action(
                    obs, avail_actions, mask_beta, key_agent_idx, actor_hidden,
                    deterministic=True
                )

                # 3. 环境步进
                reward, terminated, info = env.step(actions)
                obs = env.get_obs()
                episode_reward += reward
                step += 1

                # 防止死循环 (通常 SMAC 自己有步数限制，这里加一层保险)
                if step > 500:
                    terminated = True

            # 统计结果
            is_win = info.get('battle_won', False)
            if is_win:
                wins += 1
            total_reward += episode_reward

            res_str = "WIN" if is_win else "LOSS"
            print(f"   Episode {ep + 1:02d}/{n_episodes}: {res_str} | Reward: {episode_reward:.2f} | Steps: {step}")

    env.close()

    win_rate = wins / n_episodes
    avg_reward = total_reward / n_episodes

    return win_rate, avg_reward


if __name__ == "__main__":
    # 评估最佳模型 (50局)
    print("\n" + "=" * 50)
    print("正在评估 Best Model (50局) ...")
    wr, rew = evaluate_model("best_model.pt", n_episodes=50)
    print(f"\nBest Model 结果: 胜率 {wr * 100:.1f}% ({int(wr * 50)}/50) | 平均分 {rew:.2f}")

    # 评估最新存档 (50局)
    # 如果您只想评测其中一个，可以注释掉下面这段
    print("\n" + "=" * 50)
    print("正在评估 Latest Checkpoint (50局) ...")
    wr, rew = evaluate_model("checkpoints/ckpt_latest.pt", n_episodes=50)
    print(f"\nLatest Checkpoint 结果: 胜率 {wr * 100:.1f}% ({int(wr * 50)}/50) | 平均分 {rew:.2f}")
    print("=" * 50 + "\n")