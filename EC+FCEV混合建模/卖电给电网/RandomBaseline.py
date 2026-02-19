"""
Random Baseline - 随机策略基线

无学习、无参数，每步从 action_space 均匀随机采样。
用于证明 RL 算法优于随机策略（性能下界）。
"""

import numpy as np


class RandomBaseline:
    """
    随机基线 Agent
    每步输出随机动作，不进行任何学习
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state, evaluate=False):
        """随机采样动作，忽略 state 和 evaluate"""
        return self.action_space.sample().astype(np.float32)
