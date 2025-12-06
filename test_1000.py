import os
import warnings
import torch
import numpy as np
from datetime import datetime
from weak_tie_env import WeakTieStarCraft2Env
from weak_tie_agent import WeakTieAgent
from weak_tie_module import WeakTieGraph

# 过滤警告并设置 SC2 路径
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["SC2PATH"] = "/root/autodl-tmp/StarCraftII"


def evaluate_model_1000(model_path, map_name="1c3s5z", n_episodes=1000, log_file="./log/test_log_1c3s5z.txt"):
    """
    评估模型性能 (1000局测试版本)
    :param model_path: 模型路径
    :param map_name: 地图名称
    :param n_episodes: 测试局数 (默认1000)
    :param log_file: 日志文件路径
    """
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 打开日志文件
    log = open(log_file, 'w', encoding='utf-8')
    
    def log_print(message):
        """同时打印到终端和日志文件"""
        print(message)
        log.write(message + '\n')
        log.flush()
    
    # 记录测试开始时间
    start_time = datetime.now()
    log_print("=" * 80)
    log_print(f"开始 1000 局测试")
    log_print(f"测试时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"地图名称: {map_name}")
    log_print(f"模型路径: {model_path}")
    log_print(f"测试局数: {n_episodes}")
    log_print("=" * 80 + "\n")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        log_print(f"模型文件不存在: {model_path}")
        log.close()
        return
    
    log_print(f"正在加载环境: {map_name} ...")
    try:
        env = WeakTieStarCraft2Env(map_name=map_name)
    except Exception as e:
        log_print(f"环境创建失败: {e}")
        log.close()
        return
    
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    obs_dim = env_info["obs_shape"]
    n_actions = env_info["n_actions"]
    
    # 关键参数配置
    if map_name in ["1c3s5z", "50m", "10m_vs_11m"]:
        HIDDEN_DIM = 256
    else:
        HIDDEN_DIM = 128
    
    log_print(f"智能体数量: {n_agents}")
    log_print(f"观测维度: {obs_dim}")
    log_print(f"动作数量: {n_actions}")
    log_print(f"隐藏层维度: {HIDDEN_DIM}\n")
    
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
    log_print(f"模型加载完成 (训练轮数: {start_ep})\n")
    
    # 初始化弱联系图
    wt_graph = WeakTieGraph(n_agents, obs_range=15.0, alpha_quantile=0.3)
    
    # 统计变量
    wins = 0
    total_reward = 0
    total_steps = 0
    episode_results = []  # 存储每局详细信息
    
    log_print("开始评估...\n")
    log_print("-" * 80)
    
    with torch.no_grad():
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
                
                # 获取存活状态
                alive_mask = np.array([1 if env.agents[i].health > 0 else 0 for i in range(n_agents)])
                
                # 计算图结构
                mask_beta, key_agent_idx = wt_graph.compute_graph_info(positions, alive_mask)
                
                # 决策 (确定性策略)
                actions, probs, actor_hidden = agent.select_action(
                    obs, avail_actions, mask_beta, key_agent_idx, actor_hidden,
                    deterministic=True
                )
                
                # 环境步进
                reward, terminated, info = env.step(actions)
                obs = env.get_obs()
                episode_reward += reward
                step += 1
                
                # 防止死循环
                if step > 500:
                    terminated = True
            
            # 统计结果
            is_win = info.get('battle_won', False)
            if is_win:
                wins += 1
            
            total_reward += episode_reward
            total_steps += step
            
            # 存储本局信息
            episode_info = {
                'episode': ep + 1,
                'win': is_win,
                'reward': episode_reward,
                'steps': step
            }
            episode_results.append(episode_info)
            
            # 输出本局结果
            res_str = "WIN" if is_win else "LOSS"
            log_print(f"Episode {ep + 1:04d}/{n_episodes}: {res_str} | "
                     f"Reward: {episode_reward:7.2f} | Steps: {step:3d} | "
                     f"当前胜率: {wins / (ep + 1) * 100:5.2f}%")
            
            # 每100局输出一次阶段性统计
            if (ep + 1) % 100 == 0:
                current_wr = wins / (ep + 1)
                current_avg_reward = total_reward / (ep + 1)
                current_avg_steps = total_steps / (ep + 1)
                log_print("-" * 80)
                log_print(f"阶段统计 [{ep + 1}/{n_episodes}]: "
                         f"胜率 {current_wr * 100:.2f}% | "
                         f"平均奖励 {current_avg_reward:.2f} | "
                         f"平均步数 {current_avg_steps:.2f}")
                log_print("-" * 80)
    
    env.close()
    
    # 计算最终统计数据
    win_rate = wins / n_episodes
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes
    
    # 计算胜利局的平均步数和失败局的平均步数
    win_episodes = [ep for ep in episode_results if ep['win']]
    loss_episodes = [ep for ep in episode_results if not ep['win']]
    
    avg_steps_win = np.mean([ep['steps'] for ep in win_episodes]) if win_episodes else 0
    avg_steps_loss = np.mean([ep['steps'] for ep in loss_episodes]) if loss_episodes else 0
    avg_reward_win = np.mean([ep['reward'] for ep in win_episodes]) if win_episodes else 0
    avg_reward_loss = np.mean([ep['reward'] for ep in loss_episodes]) if loss_episodes else 0
    
    # 测试结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 输出最终统计结果
    log_print("\n" + "=" * 80)
    log_print("最终测试结果")
    log_print("=" * 80)
    log_print(f"地图: {map_name}")
    log_print(f"模型: {model_path}")
    log_print(f"总局数: {n_episodes}")
    log_print(f"⏱耗时: {duration}")
    log_print("-" * 80)
    log_print(f"总胜率: {win_rate * 100:.2f}% ({wins}/{n_episodes})")
    log_print(f"平均奖励: {avg_reward:.2f}")
    log_print(f"平均步数: {avg_steps:.2f}")
    log_print("-" * 80)
    log_print(f"胜利局统计:")
    log_print(f"- 胜场数: {len(win_episodes)}")
    log_print(f"- 平均步数: {avg_steps_win:.2f}")
    log_print(f"- 平均奖励: {avg_reward_win:.2f}")
    log_print("-" * 80)
    log_print(f"失败局统计:")
    log_print(f"- 败场数: {len(loss_episodes)}")
    log_print(f"- 平均步数: {avg_steps_loss:.2f}")
    log_print(f"- 平均奖励: {avg_reward_loss:.2f}")
    log_print("=" * 80)
    log_print(f"\n详细日志已保存至: {log_file}")
    log_print(f"测试完成! {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    log.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("启动 1000 局测试程序")
    print("=" * 80 + "\n")
    
    # 执行1000局测试
    evaluate_model_1000(
        model_path="best_model.pt",
        map_name="1c3s5z",
        n_episodes=1000,
        log_file="./log/test_log_1c3s5z.txt"
    )
    
    print("\n" + "=" * 80)
    print("所有测试已完成!")
    print("=" * 80 + "\n")