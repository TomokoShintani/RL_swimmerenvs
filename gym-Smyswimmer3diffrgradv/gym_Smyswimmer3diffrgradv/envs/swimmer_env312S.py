from typing import Optional, Union

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
import numpy as np
import math
from gym.error import DependencyNotInstalled

class MySwimmerDiffRGradV312S(gym.Env):
    def __init__(self):
        self.x_threshold = 300 # for the center of mass of the swimmer

        ### attributes for GUI
        self.screen_width = 1000
        self.screen_height = 400

        self.state = None
        self.screen = None

        ### attributes for calculations
        self.R = 1
        self.D = 10 * self.R
        self.eps = 4 * self.R
        self.r1 = (3/2) * self.R
        self.r2 = (1/2) * self.R
        self.r3 = self.R
        self.W = self.eps # 1秒で伸縮するように設定

        self.mu_0 = 1.0
        self.k = 0.1
        self.dist_from_zero = 10.0
        self.initL1 = None

        self.time = 0
        self.discretization = 10000

        high = np.array(
            [
                self.x_threshold,
                self.x_threshold + self.dist_from_zero,
                self.D,
                self.D
            ]
        )

        low = np.array(
            [
                - self.x_threshold,
                - self.x_threshold - self.dist_from_zero,
                self.D - self.eps,
                self.D - self.eps
            ]
        )

        ### action space
        self.action_space = spaces.Discrete(12)
        ### observation space
        self.observation_space = spaces.Box(low, high, dtype=np.float32) #これ直してみたけどどうかな？


    def reset(self):
        """
        state = (重心の位置x, 粘性が0のところから一番左の球までの距離p1, L1, L2)
        """
        x = 0
        p1 = self.dist_from_zero
        L1 = self.D  ##ゆくゆくはモンテカルロ法で決めていかないといけない
        L2 = self.D
        self.initL1 = L1
        self.state = (x, p1, L1, L2)
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        x, p1, L1, L2 = self.state

        reach = bool(x > self.x_threshold)

        if reach:
            print("#### GOAL ####")

        else:
            d = self.discretization
            if action==0:

                if (int(L1) != self.D - self.eps) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = (self.D - self.eps + self.W * _t)

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0


                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = (C / (B*C - A*D)) * self.W
                    f1 = (-D / (B*C - A*D)) * self.W
                    f2 = ((D - C) / (B*C - A*D)) * self.W

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled ## rewardのためにtraveledがあった方がやりやすい
                x += traveled
                L1 = self.D
                L2 = self.D - self.eps
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==1:

                if (int(L1) != self.D - self.eps) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L2 = self.D - self.eps + self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0


                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = (-A / (B*C - A*D)) * self.W
                    f1 = (B / (B*C - A*D)) * self.W
                    f2 = ((A - B) / (B*C - A*D)) * self.W

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==2:
                if (int(L1) != self.D) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L2 = self.D - self.eps + self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0


                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = (-A / (B*C - A*D)) * self.W
                    f1 = (B / (B*C - A*D)) * self.W
                    f2 = ((A - B) / (B*C - A*D)) * self.W

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==3:

                if (int(L1) != self.D) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0


                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = (-C / (B*C - A*D)) * self.W
                    f1 = (D / (B*C - A*D)) * self.W
                    f2 = ((C - D) / (B*C - A*D)) * self.W

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D - self.eps
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==4:

                if (int(L1) != self.D) or (int(L2) != self.D):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0

                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = (-C / (B*C - A*D)) * self.W
                    f1 = (D / (B*C - A*D)) * self.W
                    f2 = ((C - D) / (B*C - A*D)) * self.W

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==5:

                if (int(L1) != self.D) or (int(L2) != self.D):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L2 = self.D - self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0

                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = (A / (B*C - A*D)) * self.W
                    f1 = (-B / (B*C - A*D)) * self.W
                    f2 = ((B - A) / (B*C - A*D)) * self.W

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D - self.eps
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==6:

                if (int(L1) != self.D - self.eps) or (int(L2) != self.D):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L2 = self.D - self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0

                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = (A / (B*C - A*D)) * self.W
                    f1 = (-B / (B*C - A*D)) * self.W
                    f2 = ((B - A) / (B*C - A*D)) * self.W

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)


                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D - self.eps
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==7:

                if (int(L1) != self.D - self.eps) or (int(L2) != self.D):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.eps + self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0

                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = (C / (B*C - A*D)) * self.W
                    f1 = (-D / (B*C - A*D)) * self.W
                    f2 = ((D - C) / (B*C - A*D)) * self.W

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==8:
                if (int(L1) != self.D - self.eps) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.eps + self.W * _t
                    L2 = self.D - self.eps + self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0

                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f1 = ((D - B) / (D*A - B*C)) * self.W
                    f3 = ((A - C) / (D*A - B*C)) * self.W
                    f2 = 0 - f1 - f3

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==9:
                if (int(L1) != self.D) or (int(L2) != self.D):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.eps + self.W * _t
                    L2 = self.D - self.eps + self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0

                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f1 = ((B - D) / (D*A - B*C)) * self.W
                    f3 = ((C - A) / (D*A - B*C)) * self.W
                    f2 = 0 - f1 - f3

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D - self.eps
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)                


            elif action==10:
                if (int(L1) != self.D) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.W * _t
                    L2 = self.D - self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0

                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f3 = ((A + C) / (D*A - B*C)) * self.W
                    f1 = (-(B + D) / (D*A - B*C)) * self.W
                    f2 = 0 - f1 - f3

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

            elif action==11:
                if (int(L1) != self.D - self.eps) or (int(L2) != self.D):
                    print("Invalid action")
                    return

                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.W * _t
                    L2 = self.D - self.W * _t

                    mu_1 = self.k * p1 + self.mu_0
                    mu_2 = self.k * (p1 + L1) + self.mu_0
                    mu_3 = self.k * (p1 + L1 + L2) + self.mu_0

                    A = 1/(4 * math.pi * mu_1 * L1) \
                        - 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(6 * math.pi * mu_1 * self.r1) \
                        + 1/(4 * math.pi * mu_2 * L1)

                    B = - 1/(6 * math.pi * mu_2 * self.r2) \
                        + 1/(4 * math.pi * mu_3 * L2) \
                        + 1/(4 * math.pi * mu_2 * L1) \
                        - 1/(4 * math.pi * mu_3 * (L1 + L2))

                    C = 1/(4 * math.pi * mu_1 * (L1 + L2)) \
                        - 1/(4 * math.pi * mu_2 * L2) \
                        - 1/(4 * math.pi * mu_1 * L1) \
                        + 1/(6 * math.pi * mu_2 * self.r2)

                    D = - 1/(4 * math.pi * mu_2 * L2) \
                        + 1/(6 * math.pi * mu_3 * self.r3) \
                        + 1/(6 * math.pi * mu_2 * self.r2) \
                        - 1/(4 * math.pi * mu_3 * L2)

                    f1 = ((B + D) / (D*A - B*C)) * self.W
                    f3 = (-(C + A) / (D*A - B*C)) * self.W
                    f2 = 0 - f1 - f3

                    v1 = (f1 / (6 * math.pi * mu_1 * self.r1)) \
                        + (f2 / (4 * math.pi * mu_2 * L1)) \
                        + (f3 / (4 * math.pi * mu_3 * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * mu_1 * L1)) \
                        + (f2 / (6 * math.pi * mu_2 * self.r2)) \
                        + (f3 / (4 * math.pi * mu_3 * L2))

                    v3 = (f1 / (4 * math.pi * mu_1 * (L1 + L2))) \
                        + (f2 / (4 * math.pi * mu_2 * L2)) \
                        + (f3 / (6 * math.pi * mu_3 * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # print('v2-v1:', v2 - v1)
                    # print('v3-v2:', v3 - v2)

                    _t += T/d
                    traveled += T/d * V0
                    p1 += T/d * v1
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D - self.eps
                p1 = (3 * x - 2 * L1 - L2) / 3 + self.dist_from_zero + self.initL1
                self.state = (x, p1, L1, L2)

        return np.array(self.state, dtype=np.float32), reward, reach, {} ##ここで一応reach返しといたほうがいいのかね？

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((self.screen_width, self.screen_height))


        world_width = self.x_threshold # これ今回の場合は2倍にする必要ないかな？
        scale = (self.screen_width - self.screen_width*(1/40)*2) / world_width #スイマーを一番左から出発させたくないし、重心がx_threshold超えたかどうかを見るので、100引いとく
        rod_width = 0.1
        sphere_r1 = scale * self.r1
        sphere_r2 = scale * self.r2
        sphere_r3 = scale * self.r3

        if self.state is None:
            return None

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        x, p1, L1, L2 = self.state # xは重心の座標なので注意

        x1 = (3 * x - 2 * L1 - L2) / 3
        x2 = x1 + L1
        x3 = x2 + L2

        ## スケールを合わせたやつ
        scaled_x = x * scale + self.screen_width / 6.0
        scaled_x1 = x1 * scale + self.screen_width / 6.0
        scaled_x2 = x2 * scale + self.screen_width / 6.0
        scaled_x3 = x3 * scale + self.screen_width / 6.0
        scaled_L1 = L1 * scale
        scaled_L2 = L2 * scale

        ### 球の部分描く
        gfxdraw.aacircle(  ##引数の座標はint型じゃないといけん
            self.surf,
            int(scaled_x1),
            int(self.screen_height / 2),
            int(sphere_r1),
            (0, 0, 0)
        )

        gfxdraw.filled_circle(
            self.surf,
            int(scaled_x1),
            int(self.screen_height / 2),
            int(sphere_r1),
            (0, 0, 0)
        )

        gfxdraw.aacircle(  ##引数の座標はint型じゃないといけん
            self.surf,
            int(scaled_x2),
            int(self.screen_height / 2),
            int(sphere_r2),
            (0, 0, 0)
        )

        gfxdraw.filled_circle(
            self.surf,
            int(scaled_x2),
            int(self.screen_height / 2),
            int(sphere_r2),
            (0, 0, 0)
        )

        gfxdraw.aacircle(  ##引数の座標はint型じゃないといけん
            self.surf,
            int(scaled_x3),
            int(self.screen_height / 2),
            int(sphere_r3),
            (0, 0, 0)
        )

        gfxdraw.filled_circle(
            self.surf,
            int(scaled_x3),
            int(self.screen_height / 2),
            int(sphere_r3),
            (0, 0, 0)
        )


        top = self.screen_height / 2 + rod_width / 2
        bot = self.screen_height / 2 - rod_width / 2

        rod_left_coords = [(scaled_x1, bot), (scaled_x1, top), (scaled_x2, top), (scaled_x2, bot)]
        rod_right_coords = [(scaled_x2, bot), (scaled_x2, top), (scaled_x3, top), (scaled_x3, bot)]

        gfxdraw.aapolygon(self.surf, rod_left_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, rod_left_coords, (0, 0, 0))
        gfxdraw.aapolygon(self.surf, rod_right_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, rod_right_coords, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )