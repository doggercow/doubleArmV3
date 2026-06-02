# 🦾 2-Joint Robotic Arm Control using Deep Q-Networks (DQN)

This project implements a Deep Reinforcement Learning agent that learns to control a 2-joint robotic arm in a 2D environment. The objective of the agent is to reach a dynamic target point in the workspace. 
The entire framework—including the physical simulation engine, the reinforcement learning environment, and the DQN algorithm—was built from scratch in Python using `PyTorch` and `Pygame`, without relying on pre-built environments like OpenAI Gym or Unity.

---

## 🛠️ Prerequisites and Installation

The project is built on Python (v3.8 or higher is recommended) and requires a few core libraries for mathematical computations, deep learning, and graphics rendering.

### Required Libraries:
1. **PyTorch (`torch`)**: Used to build, compute, and train the Deep Q-Network.
2. **Pygame (`pygame`)**: Used for rendering the 2D arm simulation and handling user interaction.
3. **NumPy (`numpy`)**: Used for vector mathematics, coordinate transformations, and forward kinematics.
4. **Matplotlib (`matplotlib`)**: Used to generate training performance graphs and the final spatial heatmap.

### Installation Command:
Open your Terminal or Command Prompt in the project's root directory and run the following command to install all dependencies at once:

```bash
    pip install torch pygame numpy matplotlib
```

Note: If you wish to utilize a GPU (CUDA) for training, please follow the specific installation instructions on the official PyTorch website. However, this codebase is fully optimized to run quickly and efficiently on standard CPUs.


📂 Project Structure
The codebase is highly modular, separating physics, environment rules, and learning logic:

arm_sim.py: The core physical simulation engine. Handles the arm's physical geometry, computes Forward Kinematics (angles to X,Y coordinates), and renders the elements using Pygame.

env.py: The Reinforcement Learning environment wrapper (MDP setup). Normalizes observations, handles step limits, and defines the reward function, including a built-in Anti-Jitter Penalty.

network.py: Defines the Deep Q-Network neural network architecture using PyTorch.

agent.py: The core AI logic. Contains the ReplayBuffer for experience replay, manages both the Policy and Target networks, and handles the Bellman optimization step.

train.py: The main execution script to train the agent from scratch, save trained weights, and log metrics.

eval.py: An interactive evaluation script that allows users to test the trained model by placing new targets anywhere using mouse clicks.

heatmap.py: An advanced evaluation tool that silently benchmarks the agent across thousands of coordinates to export a spatial performance heatmap.


🚀 How to Run
1. Training the Agent
To start the learning process from scratch, execute:
```bash
    python train.py
```
What happens? The agent will train for 1500 episodes. Every 100 episodes, a graphical window will pop up automatically to display the agent's current progress.

Modifying Parameters: You can adjust parameters at the top of train.py:

EPISODES: Change the total number of training episodes.

RENDER_EVERY: Change how frequently the visual progress window opens.

Outputs: Once finished, the trained weights are saved into models/, and performance plots (Reward, Loss, Steps) are saved under graphs/.


2. Interactive Evaluation
Once you have a trained model in the models/ directory, you can test it interactively by running:
```bash
    python eval.py
```
Controls:Left-Click: Instantly moves the target to the clicked position. The arm will calculate the optimal path and move there using pure exploitation ($\epsilon = 0$).R Key: Resets the arm back to its origin (0 degrees on both joints).Q or Esc Key: Exits and closes the window safely.


3. Generating a Spatial HeatmapTo analyze the agent's proficiency across the entire reachable workspace, run:
```bash
    python heatmap.py
```
What happens? The script runs a silent grid test across ~7,850 sample points within the arm's reach and measures how many steps it takes to reach each coordinate.

Output: A color-coded map will be exported to your graphs directory. Green regions indicate fast, optimal paths, while Red highlights areas that caused timeouts or slower adjustments.


Core Hyperparameters
Batch Size: 64
Replay Buffer Capacity: 600,000
experiencesGamma ($\gamma$): 0.99 (Discount factor)
Learning Rate: 3e-4 (Adam Optimizer)
Max Steps per Episode: 720 steps

these hyper parameters can be changed in the file called agent.py
