# Responsible for interactive evaluation: loads the latest model and lets the user click to place targets in a live window.
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

MODELS_DIR = "models"   # directory to search for saved .pth model files
FPS        = 60         # frame rate cap for the pygame render loop


# Finds and returns the path to the most recently saved model file in MODELS_DIR, or None if none exist.
def latest_model():
    if not os.path.isdir(MODELS_DIR):
        return None
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]   # all model weight files in the directory
    if not files:
        return None
    files.sort()                                      # alphabetical sort; timestamp filenames sort chronologically
    return os.path.join(MODELS_DIR, files[-1])        # most recent model path


# Converts a pygame screen pixel coordinate to a sim-space (x, y) coordinate, inverting the y-flip.
def screen_to_sim(screen_pos, width, height):
    """Inverse of ArmSimulation._sim_to_screen."""
    cx, cy = width // 2, height // 2          # pixel coordinates of the screen centre
    sim_x  =  (screen_pos[0] - cx)            # x offset from centre, no flip needed
    sim_y  = -(screen_pos[1] - cy)            # y-axis is flipped in pygame so we negate
    return np.array([sim_x, sim_y], dtype=np.float32)   # sim-space (x, y) position


# Loads the latest model and runs the interactive eval loop until the user quits.
def run_eval():
    model_path = latest_model()
    if model_path is None:
        print(f"No trained model found in '{MODELS_DIR}/'. Run train.py first.")
        sys.exit(1)
    print(f"Loading {model_path}")

    env   = ArmEnv(render=True)                                                        # environment with pygame window enabled
    agent = DQNAgent.load(model_path, n_obs=ArmEnv.N_OBS, n_actions=ArmEnv.N_ACTIONS) # greedy agent loaded from disk
    font  = pygame.font.SysFont("monospace", 16)   # monospace font used to render the target coordinate label

    obs  = env.reset()   # initial observation after resetting arm and picking a random target
    done = False         # episode termination flag

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
                    obs  = env.reset()   # reset arm to 0°/0° and pick a new random target
                    done = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                obs     = env.reset()
                sim_pos         = screen_to_sim(event.pos,
                                                env.sim.WIDTH, env.sim.HEIGHT)   # convert click pixel to sim-space coords
                env.target      = sim_pos        # update the environment's target position
                env.sim.target  = sim_pos        # sync into the sim so the red dot moves
                obs             = env._get_obs() # rebuild observation with the new target
                done            = False

        # ── agent step ────────────────────────────────────────────────
        if not done:
            action            = agent.select_action(obs)           # greedy action from the loaded policy
            obs, reward, done = env.step(action)                   # advance the sim one step

        # ── when episode ends, pause on the last frame until user clicks ──
        env.render()

        cx, cy = env.sim.WIDTH // 2, env.sim.HEIGHT // 2                         # screen centre pixel coordinates
        pygame.draw.circle(env.sim.screen, (60, 60, 80),
                           (cx, cy), int(ArmEnv.REACH), 1)                       # reachable workspace boundary circle

        tx, ty = env.target                                                       # target x and y in sim-space
        label  = font.render(f"target  x={tx:6.1f}  y={ty:6.1f}", True, (255, 80, 80))   # text showing current target coords
        env.sim.screen.blit(label, (8, 8))
        pygame.display.flip()

        env.sim.clock.tick(FPS)


if __name__ == "__main__":
    run_eval()
