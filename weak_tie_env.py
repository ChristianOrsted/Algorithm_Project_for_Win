from smac.env import StarCraft2Env
import numpy as np


class WeakTieStarCraft2Env(StarCraft2Env):

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
            if tag in self.agents:
                unit = self.agents[tag]
                # 直接访问 PySC2 Unit 对象的 pos 属性
                positions.append([unit.pos.x, unit.pos.y])
            else:
                # 死亡单位返回 [0.0, 0.0]
                positions.append([0.0, 0.0])

        # 强制转换为 float32，确保与 PyTorch tensor 转换时无警告
        return np.array(positions, dtype=np.float32)