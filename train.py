import os
import numpy as np
from datetime import datetime
from env import ArmEnv
from agent import DQNAgent

EPISODES     = 1500
RENDER_EVERY = 250   # show a live window every N episodes to watch progress
MODELS_DIR   = "models"


def train():
    env   = ArmEnv(render=False)
    agent = DQNAgent(n_obs=ArmEnv.N_OBS, n_actions=ArmEnv.N_ACTIONS)

    for ep in range(1, EPISODES + 1):

        # ── optional live render ───────────────────────────────────────
        render_this_ep = (ep % RENDER_EVERY == 0)
        if render_this_ep:
            env.sim.render_mode = True
            env.sim._init_pygame()

        # ── episode loop ───────────────────────────────────────────────
        obs       = env.reset()
        done      = False
        ep_reward = 0.0
        ep_losses = []

        while not done:
            # 1. agent picks an action
            action = agent.select_action(obs)

            # 2. environment responds
            next_obs, reward, done = env.step(action)

            # 3. store transition in replay buffer
            agent.store(obs, action, reward, next_obs, done)

            # 4. one gradient update (returns None until buffer has ≥ BATCH_SIZE)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            obs        = next_obs
            ep_reward += reward

            if render_this_ep:
                env.render()

        # ── end of episode bookkeeping ─────────────────────────────────
        agent.decay_epsilon()

        if render_this_ep:
            env.sim.render_mode = False
            env.sim.close()

        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        print(
            f"ep {ep:4d} | "
            f"reward {ep_reward:8.3f} | "
            f"loss {avg_loss:.5f} | "
            f"epsilon {agent.epsilon:.3f} | "
            f"buffer {len(agent.buffer):5d}"
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.pth")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()


if __name__ == "__main__":
    train()
