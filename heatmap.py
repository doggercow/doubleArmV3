# Responsible for generating a step-count heatmap that visualises agent performance across the full reachable workspace.
"""
heatmap.py — agent performance across the reachable workspace.

Samples a grid of target points, runs the agent silently on each,
then exports a step-count heatmap PNG to graphs/<model_name>/heatmap.png.
Green = reached quickly, red = slow / timed out at 720 steps.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from env import ArmEnv
from agent import DQNAgent

MODELS_DIR = "models"   # directory to search for saved .pth model files
GRID_RES   = 100        # number of sample points along each axis (100×100 = ~7850 inside the workspace circle)


# Finds and returns the path to the most recently saved model file in MODELS_DIR, or None if none exist.
def latest_model():
    if not os.path.isdir(MODELS_DIR):
        return None
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]   # all model weight files in the directory
    if not files:
        return None
    files.sort()                               # alphabetical sort; timestamp filenames sort chronologically
    return os.path.join(MODELS_DIR, files[-1])  # most recent model path


# Resets the arm to origin, places the given target, and runs the episode to completion. Returns the step count.
def run_episode(env, agent, target):
    """Silently reset arm to origin, set target, run to completion."""
    env.sim.reset()         # reset joint angles to 0°/0°
    env.steps       = 0     # reset the environment step counter
    env.prev_action = -1    # clear previous action so jitter penalty doesn't fire on the first step
    env.target      = target      # set the target position in the environment
    env.sim.target  = target      # sync the target into the sim for position calculations
    obs  = env._get_obs()   # build the initial observation for this target position
    done = False             # episode termination flag
    while not done:
        action = agent.select_action(obs)    # greedy action from the loaded policy
        obs, _, done = env.step(action)      # advance the sim (reward discarded; only step count matters)
    return env.steps   # number of steps taken to reach (or time out on) this target


# Loads the latest model, samples a grid of targets across the workspace, and saves a heatmap PNG.
def run_heatmap():
    model_path = latest_model()
    if model_path is None:
        print(f"No trained model found in '{MODELS_DIR}/'. Run train.py first.")
        sys.exit(1)
    print(f"Loading {model_path}")

    env   = ArmEnv(render=False)                                                          # headless environment (no pygame window)
    agent = DQNAgent.load(model_path, n_obs=ArmEnv.N_OBS, n_actions=ArmEnv.N_ACTIONS)    # greedy agent loaded from disk

    xs = np.linspace(-ArmEnv.REACH, ArmEnv.REACH, GRID_RES)   # evenly spaced x coordinates spanning the workspace
    ys = np.linspace(-ArmEnv.REACH, ArmEnv.REACH, GRID_RES)   # evenly spaced y coordinates spanning the workspace

    # grid[j, i] = steps for target at (xs[i], ys[j]), NaN outside workspace
    grid = np.full((GRID_RES, GRID_RES), np.nan)   # 2D array of step counts; NaN for points outside the workspace

    in_workspace = [
        (i, j, xs[i], ys[j])
        for i in range(GRID_RES)
        for j in range(GRID_RES)
        if 30 <= np.sqrt(xs[i]**2 + ys[j]**2) <= ArmEnv.REACH * 0.95   # only points inside the reachable annulus
    ]

    print(f"Sampling {len(in_workspace)} points...")
    for count, (i, j, x, y) in enumerate(in_workspace, 1):
        grid[j, i] = run_episode(env, agent, np.array([x, y], dtype=np.float32))   # fill in step count for this grid cell
        if count % 100 == 0:
            print(f"  {count}/{len(in_workspace)}")

    print(f"  {len(in_workspace)}/{len(in_workspace)} — done")
    env.close()

    # ── plot ──────────────────────────────────────────────────────────────
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "steps", ["green", "yellow", "red"]   # colour scale: green = few steps (fast), red = many steps (slow/timeout)
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    mesh = ax.pcolormesh(xs, ys, grid, cmap=cmap,
                         vmin=1, vmax=ArmEnv.MAX_STEPS)   # colour range from 1 step up to the timeout limit
    plt.colorbar(mesh, ax=ax, label="Steps to reach target")

    theta = np.linspace(0, 2 * np.pi, 360)   # angles for drawing the workspace boundary circle
    ax.plot(ArmEnv.REACH * np.cos(theta), ArmEnv.REACH * np.sin(theta),
            color="gray", linewidth=1, linestyle="--")   # dashed circle marking the outer edge of the workspace

    ax.set_aspect("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Agent performance heatmap  (green = fast, red = slow / timeout)")
    fig.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────
    model_name = os.path.splitext(os.path.basename(model_path))[0]   # model filename without extension, used as subfolder name
    graphs_dir = os.path.join("graphs", model_name)                   # output directory for this model's graphs
    os.makedirs(graphs_dir, exist_ok=True)
    out_path   = os.path.join(graphs_dir, "heatmap.png")              # full path of the output PNG file
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {out_path}")


if __name__ == "__main__":
    run_heatmap()
