from typing import Optional, Union

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
import numpy as np
import math
from gym.error import DependencyNotInstalled

class MySwimmerDiffRadii231(gym.Env):
    def __init__(self):
        self.x_threshold = 300

        ### attributes for GUI
        self.screen_width = 1000
        self.screen_height = 400

        self.state = None          
        self.screen = None         

        ### attributes for calculations
        self.R = 1
        self.D = 10 * self.R
        self.eps = 4 * self.R
        self.r1 = self.R  
        self.r2 = (3/2) * self.R
        self.r3 = (1/2) * self.R
        self.W = self.eps 
        self.mu = 1.0
        self.time = 0
        self.discretization = 10000

        high = np.array(
            [
                self.x_threshold,
                self.D,
                self.D
            ]
        )

        low = np.array(
            [
                - self.x_threshold,
                self.D - self.eps,
                self.D - self.eps
            ]
        )

        ### action space 
        self.action_space = spaces.Discrete(8)
        ### observation space
        self.observation_space = spaces.Box(low, high, dtype=np.float32) 


    def reset(self):
        """
        state = (重心の位置, L1, L2)
        """
        x = 0
        L1 = self.D  ##ゆくゆくはモンテカルロ法で決めていかないといけない
        L2 = self.D
        self.state = (x, L1, L2)
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        x, L1, L2 = self.state

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
                    # V0 = (self.R / 6) * ((- self.W / (2 * (self.D - self.eps) + self.W * _t)) + 2 * (self.W / (self.D - self.eps)))
                    L1 = (self.D - self.eps + self.W * _t)

                    A = 1/(4 * math.pi * self.mu * L1) \
                        - 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(6 * math.pi * self.mu * self.r1) \
                        + 1/(4 * math.pi * self.mu * L1)

                    B = -1/(6 * math.pi * self.mu * self.r2) \
                        + 1/(4 * math.pi * self.mu * L2) \
                        + 1/(4 * math.pi * self.mu * L1) \
                        - 1/(4 * math.pi * self.mu * (L1 + L2))

                    C = - 1/(4 * math.pi * self.mu * L2) \
                        + 1/(6 * math.pi * self.mu * self.r3) \
                        + 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(4 * math.pi * self.mu * L2)

                    f3 = (B / (B**2 + A*C)) * self.W
                    f1 = (C / (B**2 + A*C)) * self.W
                    f2 = (-(B + C) / (B**2 + A*C)) * self.W

                    v1 = (f1 / (6 * math.pi * self.mu * self.r1)) \
                        + (f2 / (4 * math.pi * self.mu * L1)) \
                        + (f3 / (4 * math.pi * self.mu * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * self.mu * L1)) \
                        + (f2 / (6 * math.pi * self.mu * self.r2)) \
                        + (f3 / (4 * math.pi * self.mu * L2))

                    v3 = (f1 / (4 * math.pi * self.mu * (L1 + L2))) \
                        + (f2 / (4 * math.pi * self.mu * L2)) \
                        + (f3 / (6 * math.pi * self.mu * self.r3))

                    V0 = (v1 + v2 + v3) / 3

                    _t += T/d
                    traveled += T/d * V0
                    self.time += T/d

                ##多分Wtの部分を無視してるのがよくない。時間離散化して、0.2ずつとかこうしんしていけばいいのかな？
                """
                t = 0.0 ~ 0.1 V0 = ? → x += 0.1 * ?
                t = 0.1 ~ 0.2 V0 = ? → x += 0.1 * ?

                t = 0.9 ~ 1.0 V0 = ? → x += 0.1 * ? みたいな感じか

                """
                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D - self.eps
                self.state = (x, L1, L2)

            elif action==1:
                if (int(L1) != self.D - self.eps) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return
                
                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L2 = self.D - self.eps + self.W * _t

                    A = 1/(4 * math.pi * self.mu * L1) \
                        - 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(6 * math.pi * self.mu * self.r1) \
                        + 1/(4 * math.pi * self.mu * L1)

                    B = -1/(6 * math.pi * self.mu * self.r2) \
                        + 1/(4 * math.pi * self.mu * L2) \
                        + 1/(4 * math.pi * self.mu * L1) \
                        - 1/(4 * math.pi * self.mu * (L1 + L2))

                    C = - 1/(4 * math.pi * self.mu * L2) \
                        + 1/(6 * math.pi * self.mu * self.r3) \
                        + 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(4 * math.pi * self.mu * L2)

                    f1 = (-B / (B**2 + A*C)) * self.W
                    f3 = (A / (B**2 + A*C)) * self.W
                    f2 = ((B - A) / (B**2 + A*C)) * self.W

                    v1 = (f1 / (6 * math.pi * self.mu * self.r1)) \
                        + (f2 / (4 * math.pi * self.mu * L1)) \
                        + (f3 / (4 * math.pi * self.mu * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * self.mu * L1)) \
                        + (f2 / (6 * math.pi * self.mu * self.r2)) \
                        + (f3 / (4 * math.pi * self.mu * L2))

                    v3 = (f1 / (4 * math.pi * self.mu * (L1 + L2))) \
                        + (f2 / (4 * math.pi * self.mu * L2)) \
                        + (f3 / (6 * math.pi * self.mu * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # V0 = (self.R / 6) * (self.W / (2 * (self.D - self.eps) + self.W * _t) - 2 * (self.W / (self.D - self.eps)))
                    _t += T/d
                    traveled += T/d * V0
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D
                self.state = (x, L1, L2)

            elif action==2:
                if (int(L1) != self.D) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return 
                
                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L2 = self.D - self.eps + self.W * _t

                    A = 1/(4 * math.pi * self.mu * L1) \
                        - 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(6 * math.pi * self.mu * self.r1) \
                        + 1/(4 * math.pi * self.mu * L1)

                    B = -1/(6 * math.pi * self.mu * self.r2) \
                        + 1/(4 * math.pi * self.mu * L2) \
                        + 1/(4 * math.pi * self.mu * L1) \
                        - 1/(4 * math.pi * self.mu * (L1 + L2))

                    C = - 1/(4 * math.pi * self.mu * L2) \
                        + 1/(6 * math.pi * self.mu * self.r3) \
                        + 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(4 * math.pi * self.mu * L2)

                    f3 = (A / (B**2 + A*C)) * self.W
                    f1 = (-B / (B**2 + A*C)) * self.W
                    f2 = ((B - A) / (B**2 + A*C)) * self.W

                    v1 = (f1 / (6 * math.pi * self.mu * self.r1)) \
                        + (f2 / (4 * math.pi * self.mu * L1)) \
                        + (f3 / (4 * math.pi * self.mu * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * self.mu * L1)) \
                        + (f2 / (6 * math.pi * self.mu * self.r2)) \
                        + (f3 / (4 * math.pi * self.mu * L2))

                    v3 = (f1 / (4 * math.pi * self.mu * (L1 + L2))) \
                        + (f2 / (4 * math.pi * self.mu * L2)) \
                        + (f3 / (6 * math.pi * self.mu * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # V0 = (self.R / 6) * ((self.W / (2 * self.D - self.eps + self.W * _t)) - 2 * (self.W / self.D))
                    _t += T/d
                    traveled += T/d * V0
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D
                self.state = (x, L1, L2)

            elif action==3:
                if (int(L1) != self.D) or (int(L2) != self.D - self.eps):
                    print("Invalid action")
                    return
                
                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.W * _t

                    A = 1/(4 * math.pi * self.mu * L1) \
                        - 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(6 * math.pi * self.mu * self.r1) \
                        + 1/(4 * math.pi * self.mu * L1)

                    B = -1/(6 * math.pi * self.mu * self.r2) \
                        + 1/(4 * math.pi * self.mu * L2) \
                        + 1/(4 * math.pi * self.mu * L1) \
                        - 1/(4 * math.pi * self.mu * (L1 + L2))

                    C = - 1/(4 * math.pi * self.mu * L2) \
                        + 1/(6 * math.pi * self.mu * self.r3) \
                        + 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(4 * math.pi * self.mu * L2)

                    f3 = (-B / (B**2 + A*C)) * self.W
                    f1 = (-C / (B**2 + A*C)) * self.W
                    f2 = ((B + C) / (B**2 + A*C)) * self.W

                    v1 = (f1 / (6 * math.pi * self.mu * self.r1)) \
                        + (f2 / (4 * math.pi * self.mu * L1)) \
                        + (f3 / (4 * math.pi * self.mu * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * self.mu * L1)) \
                        + (f2 / (6 * math.pi * self.mu * self.r2)) \
                        + (f3 / (4 * math.pi * self.mu * L2))

                    v3 = (f1 / (4 * math.pi * self.mu * (L1 + L2))) \
                        + (f2 / (4 * math.pi * self.mu * L2)) \
                        + (f3 / (6 * math.pi * self.mu * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # V0 = (self.R / 6) * ((self.W / (2 * self.D - self.eps - self.W * _t)) - 2 * (self.W / (self.D - self.eps)))
                    _t += T/d
                    traveled += T/d * V0
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D - self.eps
                self.state = (x, L1, L2)

            elif action==4:
                if (int(L1) != self.D) or (int(L2) != self.D):
                    print("Invalid action")
                    return
                
                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.W * _t

                    A = 1/(4 * math.pi * self.mu * L1) \
                        - 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(6 * math.pi * self.mu * self.r1) \
                        + 1/(4 * math.pi * self.mu * L1)

                    B = -1/(6 * math.pi * self.mu * self.r2) \
                        + 1/(4 * math.pi * self.mu * L2) \
                        + 1/(4 * math.pi * self.mu * L1) \
                        - 1/(4 * math.pi * self.mu * (L1 + L2))

                    C = - 1/(4 * math.pi * self.mu * L2) \
                        + 1/(6 * math.pi * self.mu * self.r3) \
                        + 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(4 * math.pi * self.mu * L2)

                    f1 = - (C / (B**2 + A*C)) * self.W
                    f3 = - (B / (B**2 + A*C)) * self.W
                    f2 = ((B + C) / (B**2 + A*C)) * self.W

                    v1 = (f1 / (6 * math.pi * self.mu * self.r1)) \
                        + (f2 / (4 * math.pi * self.mu * L1)) \
                        + (f3 / (4 * math.pi * self.mu * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * self.mu * L1)) \
                        + (f2 / (6 * math.pi * self.mu * self.r2)) \
                        + (f3 / (4 * math.pi * self.mu * L2))

                    v3 = (f1 / (4 * math.pi * self.mu * (L1 + L2))) \
                        + (f2 / (4 * math.pi * self.mu * L2)) \
                        + (f3 / (6 * math.pi * self.mu * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # V0 = (self.R / 6) * (self.W / (2 * self.D - self.W * _t) - 2 * (self.W / self.D))
                    _t += T/d
                    traveled += T/d * V0
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D
                self.state = (x, L1, L2)

            elif action==5:
                if (int(L1) != self.D) or (int(L2) != self.D):
                    print("Invalid action")
                    return
                
                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L2 = self.D - self.W * _t

                    A = 1/(4 * math.pi * self.mu * L1) \
                        - 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(6 * math.pi * self.mu * self.r1) \
                        + 1/(4 * math.pi * self.mu * L1)

                    B = -1/(6 * math.pi * self.mu * self.r2) \
                        + 1/(4 * math.pi * self.mu * L2) \
                        + 1/(4 * math.pi * self.mu * L1) \
                        - 1/(4 * math.pi * self.mu * (L1 + L2))

                    C = - 1/(4 * math.pi * self.mu * L2) \
                        + 1/(6 * math.pi * self.mu * self.r3) \
                        + 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(4 * math.pi * self.mu * L2)

                    f3 = (-A / (B**2 + A*C)) * self.W
                    f1 = (B / (B**2 + A*C)) * self.W
                    f2 = ((A - B) / (B**2 + A*C)) * self.W

                    v1 = (f1 / (6 * math.pi * self.mu * self.r1)) \
                        + (f2 / (4 * math.pi * self.mu * L1)) \
                        + (f3 / (4 * math.pi * self.mu * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * self.mu * L1)) \
                        + (f2 / (6 * math.pi * self.mu * self.r2)) \
                        + (f3 / (4 * math.pi * self.mu * L2))

                    v3 = (f1 / (4 * math.pi * self.mu * (L1 + L2))) \
                        + (f2 / (4 * math.pi * self.mu * L2)) \
                        + (f3 / (6 * math.pi * self.mu * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # V0 = (self.R / 6) * ((- self.W / (2 * self.D - self.W * _t)) + 2 * (self.W / self.D))
                    _t += T/d
                    traveled += T/d * V0
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D - self.eps
                self.state = (x, L1, L2)

            elif action==6:
                if (int(L1) != self.D - self.eps) or (int(L2) != self.D):
                    print("Invalid action")
                    return
                
                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L2 = self.D - self.W * _t

                    A = 1/(4 * math.pi * self.mu * L1) \
                        - 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(6 * math.pi * self.mu * self.r1) \
                        + 1/(4 * math.pi * self.mu * L1)

                    B = -1/(6 * math.pi * self.mu * self.r2) \
                        + 1/(4 * math.pi * self.mu * L2) \
                        + 1/(4 * math.pi * self.mu * L1) \
                        - 1/(4 * math.pi * self.mu * (L1 + L2))

                    C = - 1/(4 * math.pi * self.mu * L2) \
                        + 1/(6 * math.pi * self.mu * self.r3) \
                        + 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(4 * math.pi * self.mu * L2)

                    f1 = (B / (B**2 + A*C)) * self.W
                    f3 = (-A / (B**2 + A*C)) * self.W
                    f2 = ((A - B) / (B**2 + A*C)) * self.W

                    v1 = (f1 / (6 * math.pi * self.mu * self.r1)) \
                        + (f2 / (4 * math.pi * self.mu * L1)) \
                        + (f3 / (4 * math.pi * self.mu * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * self.mu * L1)) \
                        + (f2 / (6 * math.pi * self.mu * self.r2)) \
                        + (f3 / (4 * math.pi * self.mu * L2))

                    v3 = (f1 / (4 * math.pi * self.mu * (L1 + L2))) \
                        + (f2 / (4 * math.pi * self.mu * L2)) \
                        + (f3 / (6 * math.pi * self.mu * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # V0 = (self.R / 6) * ((- self.W / (2 * self.D - self.eps - self.W * _t)) + 2 * (self.W / (self.D - self.eps)))
                    _t += T/d
                    traveled += T/d * V0
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D - self.eps
                L2 = self.D - self.eps
                self.state = (x, L1, L2)

            elif action==7:
                if (int(L1) != self.D - self.eps) or (int(L2) != self.D):
                    print("Invalid action")
                    return
                
                T = self.eps / self.W
                _t = 0
                traveled = 0

                for _ in range(d):
                    L1 = self.D - self.eps + self.W * _t

                    A = 1/(4 * math.pi * self.mu * L1) \
                        - 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(6 * math.pi * self.mu * self.r1) \
                        + 1/(4 * math.pi * self.mu * L1)

                    B = -1/(6 * math.pi * self.mu * self.r2) \
                        + 1/(4 * math.pi * self.mu * L2) \
                        + 1/(4 * math.pi * self.mu * L1) \
                        - 1/(4 * math.pi * self.mu * (L1 + L2))

                    C = - 1/(4 * math.pi * self.mu * L2) \
                        + 1/(6 * math.pi * self.mu * self.r3) \
                        + 1/(6 * math.pi * self.mu * self.r2) \
                        - 1/(4 * math.pi * self.mu * L2)

                    f1 = (C / (B**2 + A*C)) * self.W
                    f3 = (B / (B**2 + A*C)) * self.W
                    f2 = (-(C + B) / (B**2 + A*C)) * self.W

                    v1 = (f1 / (6 * math.pi * self.mu * self.r1)) \
                        + (f2 / (4 * math.pi * self.mu * L1)) \
                        + (f3 / (4 * math.pi * self.mu * (L1 + L2)))

                    v2 = (f1 / (4 * math.pi * self.mu * L1)) \
                        + (f2 / (6 * math.pi * self.mu * self.r2)) \
                        + (f3 / (4 * math.pi * self.mu * L2))

                    v3 = (f1 / (4 * math.pi * self.mu * (L1 + L2))) \
                        + (f2 / (4 * math.pi * self.mu * L2)) \
                        + (f3 / (6 * math.pi * self.mu * self.r3))

                    V0 = (v1 + v2 + v3) / 3
                    # V0 = (self.R / 6) * ((- self.W / (2 * self.D - self.eps + self.W * _t)) + 2 * (self.W / self.D))
                    _t += T/d
                    traveled += T/d * V0
                    self.time += T/d

                reward = traveled
                x += traveled
                L1 = self.D
                L2 = self.D
                self.state = (x, L1, L2)
                
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
        scale = (self.screen_width - self.screen_width*(1/6)*2) / world_width #スイマーを一番左から出発させたくないし、重心がx_threshold超えたかどうかを見るので、100引いとく
        rod_width = 2.0
        sphere_r1 = scale * self.r1
        sphere_r2 = scale * self.r2
        sphere_r3 = scale * self.r3

        if self.state is None:
            return None
        
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        
        x, L1, L2 = self.state # xは重心の座標なので注意

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