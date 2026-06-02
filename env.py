# Responsible for the RL environment: observation building, action mapping, reward shaping, and episode logic.

import numpy as np
from simulation.arm_sim import ArmSimulation


class ArmEnv:
    MAX_STEPS = 720    # maximum number of steps allowed per episode before timeout
    REACH     = 240.0  # maximum reachable distance from origin in pixels (l1 + l2)
    THRESHOLD = 20.0   # distance in pixels considered "close enough" to count as reaching the target

    ACTION_MAP = {     # maps each discrete action index to a (delta_joint1, delta_joint2) angle pair in degrees
        0: ( 1,  1),   # both joints rotate forward
        1: ( 1, -1),   # joint 1 forward, joint 2 backward
        2: (-1,  1),   # joint 1 backward, joint 2 forward
        3: (-1, -1),   # both joints rotate backward
        4: ( 1,  0),   # only joint 1 forward
        5: (-1,  0),   # only joint 1 backward
        6: ( 0,  1),   # only joint 2 forward
        7: ( 0, -1),   # only joint 2 backward
    }

    # anti-jitter: maps each action to its direct opposite to detect immediate reversals
    OPPOSITES = {0: 3, 3: 0, 1: 2, 2: 1, 4: 5, 5: 4, 6: 7, 7: 6}

    N_ACTIONS = 8   # total number of discrete actions available to the agent
    N_OBS     = 8   # length of the observation vector: [sin(j1), cos(j1), sin(j2), cos(j2), ee_x, ee_y, tx, ty]

    def __init__(self, render=False):
        self.sim         = ArmSimulation(render=render)   # underlying physics and rendering simulation
        self.steps       = 0                              # step counter for the current episode
        self.target      = np.zeros(2)                   # current target position in sim-space (x, y)
        self.prev_action = -1                            # last action taken, used by anti-jitter penalty (-1 means none)

    # ── helpers ───────────────────────────────────────────────────────

    # Samples a random target point uniformly inside the reachable workspace annulus.
    def _random_target(self):
        r     = np.random.uniform(30, self.REACH * 0.95)          # random radius between inner dead-zone and 95% of max reach
        theta = np.random.uniform(0, 2 * np.pi)                   # random angle covering the full circle
        return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)   # (x, y) target in sim-space

    # Builds the 8-element observation vector from the current joint angles, end-effector position, and target.
    def _get_obs(self):
        _, ee = self.sim.calculate_positions()
        j1r = np.radians(self.sim.joint1_angle)   # joint 1 angle converted to radians for sin/cos encoding
        j2r = np.radians(self.sim.joint2_angle)   # joint 2 angle converted to radians for sin/cos encoding
        return np.array([
            np.sin(j1r), np.cos(j1r),             # sine and cosine of joint 1 angle (avoids angle discontinuity)
            np.sin(j2r), np.cos(j2r),             # sine and cosine of joint 2 angle
            ee[0] / self.REACH,                   # end-effector x position normalised to [-1, 1]
            ee[1] / self.REACH,                   # end-effector y position normalised to [-1, 1]
            self.target[0] / self.REACH,           # target x position normalised to [-1, 1]
            self.target[1] / self.REACH,           # target y position normalised to [-1, 1]
        ], dtype=np.float32)

    # ── RL interface ──────────────────────────────────────────────────

    # Resets the arm and picks a new random target, returning the initial observation.
    def reset(self):
        self.sim.reset()
        self.target      = self._random_target()   # new random target position for this episode
        self.sim.target  = self.target             # sync target into the sim so it gets rendered
        print(f'steps: {self.steps}')
        self.steps       = 0                       # reset step counter for the new episode
        self.prev_action = -1                      # clear previous action so no jitter penalty on the first step
        return self._get_obs()

    # Applies an action, advances the simulation one step, and returns (obs, reward, done).
    def step(self, action):
        delta = self.ACTION_MAP[int(action)]   # look up the (delta_j1, delta_j2) pair for this action index
        self.sim.step(delta)
        self.steps += 1

        _, ee = self.sim.calculate_positions()
        dist  = float(np.linalg.norm(ee - self.target))   # Euclidean distance from end-effector to target in pixels

        reached = dist < self.THRESHOLD          # True if the end-effector is within the success threshold
        timeout = self.steps >= self.MAX_STEPS   # True if the episode has used all allowed steps
        done    = reached or timeout             # episode ends on success or timeout

        reward = -dist / self.REACH   # base reward: negative normalised distance (always between -1 and 0)

        # anti-jitter: penalize immediately reversing the previous action
        if action == self.OPPOSITES.get(self.prev_action):
            reward -= 0.05   # small penalty to discourage back-and-forth oscillation
        self.prev_action = action   # store this action for the next step's jitter check

        if reached:
            reward += 5.0 + (self.MAX_STEPS - self.steps) * 0.05   # bonus for reaching target, larger if reached quickly

        return self._get_obs(), reward, done

    # Delegates rendering to the underlying simulation.
    def render(self):
        self.sim.render()

    # Shuts down the simulation and pygame window.
    def close(self):
        self.sim.close()
