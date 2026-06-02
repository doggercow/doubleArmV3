# Responsible for running the full training loop, saving the trained model, and plotting performance graphs.

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from env import ArmEnv
from agent import DQNAgent

EPISODES     = 1500   # total number of episodes to train for
RENDER_EVERY = 100    # open a live pygame window every N episodes to watch the arm move
MODELS_DIR   = "models"   # directory where trained model .pth files are saved


# Runs the full DQN training loop, saves the model, and exports reward/loss/steps graphs.
def train():
    env   = ArmEnv(render=False)                                    # environment starts in headless mode (no window)
    agent = DQNAgent(n_obs=ArmEnv.N_OBS, n_actions=ArmEnv.N_ACTIONS)   # fresh DQN agent with empty replay buffer

    history_rewards = []   # accumulated reward per episode, used for the reward graph
    history_losses  = []   # average TD loss per episode, used for the loss graph
    history_steps   = []   # number of steps taken per episode, used for the steps graph

    for ep in range(1, EPISODES + 1):

        # ── optional live render ───────────────────────────────────────
        render_this_ep = (ep % RENDER_EVERY == 0)   # True only on multiples of RENDER_EVERY
        if render_this_ep:
            env.sim.render_mode = True    # enable rendering for this episode
            env.sim._init_pygame()        # open the pygame window

        # ── episode loop ───────────────────────────────────────────────
        obs       = env.reset()   # reset arm and target, get initial observation
        done      = False         # episode termination flag
        ep_reward = 0.0           # running total reward accumulated across this episode's steps
        ep_losses = []            # TD losses collected during this episode for averaging

        while not done:
            # 1. agent picks an action
            action = agent.select_action(obs)   # epsilon-greedy action selection

            # 2. environment responds
            next_obs, reward, done = env.step(action)   # advance the sim by one step

            # 3. store transition in replay buffer
            agent.store(obs, action, reward, next_obs, done)

            # 4. one gradient update (returns None until buffer has ≥ BATCH_SIZE)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)   # collect loss for end-of-episode averaging

            obs        = next_obs   # advance observation to the next state
            ep_reward += reward     # accumulate reward for this episode

            if render_this_ep:
                env.render()

        # ── end of episode bookkeeping ─────────────────────────────────
        agent.decay_epsilon()   # reduce exploration rate after every episode

        if render_this_ep:
            env.sim.render_mode = False   # switch back to headless mode
            env.sim.close()              # close the pygame window

        avg_loss = np.mean(ep_losses) if ep_losses else 0.0   # mean TD loss this episode (0 if no updates yet)
        history_rewards.append(ep_reward)
        history_losses.append(avg_loss)
        history_steps.append(env.steps)
        print(
            f"ep {ep:4d} | "
            f"reward {ep_reward:8.3f} | "
            f"loss {avg_loss:.5f} | "
            f"epsilon {agent.epsilon:.3f} | "
            f"buffer {len(agent.buffer):5d}"
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")             # unique timestamp for this training run
    model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.pth")   # full path where the model weights will be saved
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()

    model_name  = os.path.splitext(os.path.basename(model_path))[0]   # filename without extension, used as the graphs subfolder name
    graphs_dir  = os.path.join("graphs", model_name)                   # folder where the three graph PNGs will be written
    os.makedirs(graphs_dir, exist_ok=True)

    eps = range(1, EPISODES + 1)   # x-axis values for all graphs (episode numbers)
    for data, label, color, fname in [
        (history_rewards, "Accumulated reward", "steelblue",      "rewards.png"),
        (history_losses,  "Avg loss",           "orange",         "loss.png"),
        (history_steps,   "Steps",              "mediumseagreen", "steps.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eps, data, linewidth=0.8, color=color)
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(os.path.join(graphs_dir, fname), dpi=150)
        plt.close(fig)

    print(f"Graphs saved to {graphs_dir}/")


if __name__ == "__main__":
    train()
