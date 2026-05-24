"""
Evaluation mode — renders the arm and lets you click to set the target.

Controls:
  Left-click  →  move target to that position
  R           →  reset the arm to 0°/0°
  Q / Esc     →  quit
"""

import os
import sys
import numpy as np
import pygame

from env import ArmEnv
from agent import DQNAgent

MODELS_DIR = "models"
FPS        = 60


def latest_model():
    if not os.path.isdir(MODELS_DIR):
        return None
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    if not files:
        return None
    files.sort()
    return os.path.join(MODELS_DIR, files[-1])


def screen_to_sim(screen_pos, width, height):
    """Inverse of ArmSimulation._sim_to_screen."""
    cx, cy = width // 2, height // 2
    sim_x  =  (screen_pos[0] - cx)
    sim_y  = -(screen_pos[1] - cy)   # y-axis is flipped in pygame
    return np.array([sim_x, sim_y], dtype=np.float32)


def run_eval():
    model_path = latest_model()
    if model_path is None:
        print(f"No trained model found in '{MODELS_DIR}/'. Run train.py first.")
        sys.exit(1)
    print(f"Loading {model_path}")

    env   = ArmEnv(render=True)
    agent = DQNAgent.load(model_path, n_obs=ArmEnv.N_OBS, n_actions=ArmEnv.N_ACTIONS)

    obs  = env.reset()
    done = False

    print("Click anywhere in the window to place the target.")
    print("R = reset arm   |   Q / Esc = quit")

    while True:
        # ── event handling ─────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    env.close()
                    sys.exit()
                if event.key == pygame.K_r:
                    obs  = env.reset()
                    done = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                obs  = env.reset()
                sim_pos         = screen_to_sim(event.pos,
                                                env.sim.WIDTH, env.sim.HEIGHT)
                env.target      = sim_pos
                env.sim.target  = sim_pos
                obs             = env._get_obs()   # refresh obs with new target
                done            = False

        # ── agent step ────────────────────────────────────────────────
        if not done:
            action           = agent.select_action(obs)
            obs, reward, done = env.step(action)

        # ── when episode ends, pause on the last frame until user clicks ──
        env.render()
        env.sim.clock.tick(FPS)


if __name__ == "__main__":
    run_eval()
