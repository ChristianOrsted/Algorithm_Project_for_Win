import os
import warnings
import torch
import numpy as np
import time
from weak_tie_env import WeakTieStarCraft2Env
from weak_tie_agent import WeakTieAgent
from weak_tie_module import WeakTieGraph

# 过滤警告并设置 SC2 路径
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["SC2PATH"] = "C:\Program Files (x86)\StarCraft II"


def watch_agent_play(model_path, map_name="1c3s5z", n_episodes=3, step_delay=0.5):
    """
    可视化观看智能体对局
    :param model_path: 模型路径
    :param map_name: 地图名称
    :param n_episodes: 观看局数
    :param step_delay: 每步之间的延迟（秒），调整这个值来控制速度
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print(f"正在加载环境: {map_name} ...")
    print(f"每步延迟: {step_delay}秒 (可在代码中调整)")
    
    try:
        # 创建环境，注意要启用渲染
        env = WeakTieStarCraft2Env(map_name=map_name)
    except Exception as e:
        print(f"环境创建失败: {e}")
        return

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    obs_dim = env_info["obs_shape"]
    n_actions = env_info["n_actions"]

    # 参数配置 (与训练保持一致)
    if map_name in ["1c3s5z", "50m", "10m_vs_11m"]:
        HIDDEN_DIM = 256
    else:
        HIDDEN_DIM = 128

    print(f"初始化 Agent (Hidden Dim: {HIDDEN_DIM})...")

    # 初始化 Agent
    agent = WeakTieAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=n_actions,
        hidden_dim=HIDDEN_DIM,
        lr=0.0003,
        gamma=0.99
    )

    # 加载模型权重
    start_ep = agent.load_model(model_path)
    print(f"模型加载完成 (训练轮数: {start_ep})")

    # 初始化弱联系图
    wt_graph = WeakTieGraph(n_agents, obs_range=15.0, alpha_quantile=0.3)

    print(f"\n{'='*60}")
    print(f"开始观看 {n_episodes} 局对局...")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for ep in range(n_episodes):
            print(f"\n{'─'*60}")
            print(f"第 {ep + 1}/{n_episodes} 局")
            print(f"{'─'*60}")
            
            obs, state = env.reset()
            terminated = False
            episode_reward = 0
            
            # 初始化 RNN 隐藏状态
            actor_hidden = agent.init_hidden(batch_size=1)
            
            step = 0
            while not terminated:
                step += 1
                
                # 获取环境信息
                avail_actions = env.get_avail_actions()
                positions = env.get_all_unit_positions()
                
                # 获取存活状态
                alive_mask = np.array([1 if env.agents[i].health > 0 else 0 for i in range(n_agents)])
                alive_count = np.sum(alive_mask)
                
                # 计算图结构
                mask_beta, key_agent_idx = wt_graph.compute_graph_info(positions, alive_mask)
                
                # 决策（使用确定性策略）
                actions, probs, actor_hidden = agent.select_action(
                    obs, avail_actions, mask_beta, key_agent_idx, actor_hidden,
                    deterministic=True
                )
                
                # 打印详细信息
                print(f"\n📍 步数: {step}")
                print(f"   存活单位: {alive_count}/{n_agents}")
                print(f"   关键智能体: Agent {key_agent_idx}")
                
                # 打印每个智能体的动作和概率
                action_names = ["No-op", "停止", "向北", "向南", "向东", "向西", 
                               "攻击敌人..."]  # 根据实际动作集调整
                print(f"   动作决策:")
                for i in range(n_agents):
                    if alive_mask[i] == 0:
                        print(f"      Agent {i}: [已阵亡]")
                    else:
                        act = actions[i]
                        prob = probs[i][act] if probs is not None else 0.0
                        act_name = action_names[act] if act < len(action_names) else f"Action {act}"
                        print(f"      Agent {i}: {act_name} (置信度: {prob:.2%})")
                
                # 环境步进
                reward, terminated, info = env.step(actions)
                obs = env.get_obs()
                episode_reward += reward
                
                print(f"   本步奖励: {reward:.2f}")
                
                # 延迟，让你能看清楚
                time.sleep(step_delay)
                
                # 防止死循环
                if step > 500:
                    print("\n达到最大步数限制，强制结束")
                    terminated = True
            
            # 统计结果
            is_win = info.get('battle_won', False)
            result_emoji = "胜利" if is_win else "❌ 失败"
            
            print(f"\n{'='*60}")
            print(f"第 {ep + 1} 局结果: {result_emoji}")
            print(f"   总奖励: {episode_reward:.2f}")
            print(f"   总步数: {step}")
            print(f"{'='*60}\n")
            
            if ep < n_episodes - 1:
                print("⏳ 准备下一局...\n")
                time.sleep(2)  # 局间暂停2秒

    env.close()
    print(f"\n观看完成！")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("智能体对局可视化工具")
    print("="*60)
    
    # 配置参数
    MODEL_PATH = "best_model.pt"  # 模型路径
    MAP_NAME = "1c3s5z"           # 地图名称
    N_EPISODES = 3                # 观看局数
    STEP_DELAY = 0.5              # 每步延迟（秒）
    
    # 你可以调整这些参数：
    # STEP_DELAY = 0.3  # 更快
    # STEP_DELAY = 1.0  # 更慢，看得更清楚
    # STEP_DELAY = 0.1  # 快速浏览
    
    print(f"\n配置:")
    print(f"   模型: {MODEL_PATH}")
    print(f"   地图: {MAP_NAME}")
    print(f"   局数: {N_EPISODES}")
    print(f"   速度: 每步 {STEP_DELAY} 秒")
    print(f"\n提示: 可以在代码中调整 STEP_DELAY 来改变观看速度")
    print(f"   - 0.1-0.3: 快速浏览")
    print(f"   - 0.5-0.8: 正常观看")
    print(f"   - 1.0-2.0: 慢速分析")
    print()
    
    # 开始观看
    watch_agent_play(
        model_path=MODEL_PATH,
        map_name=MAP_NAME,
        n_episodes=N_EPISODES,
        step_delay=STEP_DELAY
    )
