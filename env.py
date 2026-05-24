import numpy as np
from simulation.arm_sim import ArmSimulation


class ArmEnv:
    MAX_STEPS = 720
    REACH     = 240.0   # l1 + l2 — max possible distance from origin
    THRESHOLD = 20.0    # pixels — "close enough" to target

    ACTION_MAP = {
        0: ( 1,  1),
        1: ( 1, -1),
        2: (-1,  1),
        3: (-1, -1),
        4: ( 1,  0),
        5: (-1,  0),
        6: ( 0,  1),
        7: ( 0, -1),
    }

    N_ACTIONS = 8
    N_OBS     = 6   # [j1, j2, ee_x, ee_y, tx, ty] all normalised to [-1, 1]

    def __init__(self, render=False):
        self.sim    = ArmSimulation(render=render)
        self.steps  = 0
        self.target = np.zeros(2)

    # ── helpers ───────────────────────────────────────────────────────
    def _random_target(self):
        """Uniform random point inside the reachable workspace."""
        r     = np.random.uniform(30, self.REACH * 0.95)
        theta = np.random.uniform(0, 2 * np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)

    def _get_obs(self):
        _, ee = self.sim.calculate_positions()
        return np.array([
            self.sim.joint1_angle / 180.0,   # normalised angle in [-1, 1]
            self.sim.joint2_angle / 180.0,
            ee[0] / self.REACH,              # normalised position in [-1, 1]
            ee[1] / self.REACH,
            self.target[0] / self.REACH,
            self.target[1] / self.REACH,
        ], dtype=np.float32)

    # ── RL interface ──────────────────────────────────────────────────
    def reset(self):
        self.sim.reset()
        self.target      = self._random_target()
        self.sim.target  = self.target
        self.steps       = 0
        return self._get_obs()

    def step(self, action):
        delta = self.ACTION_MAP[int(action)]
        self.sim.step(delta)
        self.steps += 1

        _, ee = self.sim.calculate_positions()
        dist  = float(np.linalg.norm(ee - self.target))

        reached = dist < self.THRESHOLD
        timeout = self.steps >= self.MAX_STEPS
        done    = reached or timeout

        reward = -dist / self.REACH     # shaped reward: always in (-1, 0]
        
    
        if reached:
            reward += 5.0 + (self.MAX_STEPS - self.steps) * 0.05              # success bonus + speed bonus

        return self._get_obs(), reward, done

    def render(self):
        self.sim.render()

    def close(self):
        self.sim.close()
