# 🐦 Flappy Bird AI with Deep Q-Learning

This repository contains two AI implementations of a simple **Flappy Bird-style game** built with **Pygame** and trained using **Deep Q-Learning** in **PyTorch**.

The goal of the agent is to learn how to control the bird, avoid obstacles, and survive as long as possible by choosing whether to jump or do nothing.

---

## 📌 Projects Included

### 1️⃣ Flappy Bird DQN with Epsilon-Greedy Exploration

This version trains a Deep Q-Network using the classic **epsilon-greedy strategy**.

The agent starts by exploring random actions and gradually shifts toward using the actions predicted by the neural network.

#### Features

- Deep Q-Network with fully connected layers
- Experience Replay Memory
- Target Network updates
- Epsilon-greedy action selection
- Reward-based learning
- Pygame visualization

---

### 2️⃣ Flappy Bird DQN with Softmax Action Selection

This version uses **Softmax action selection** instead of epsilon-greedy exploration.

The agent selects actions based on probabilities calculated from Q-values. A temperature parameter controls how much the agent explores.

#### Features

- Deep Q-Learning
- Softmax policy
- Temperature decay
- Experience Replay
- Target Network
- Pygame game loop

---

## 🧠 How the AI Works

The bird receives a state made of 4 values:

```python
[bird_y, pipe_x, pipe_height, bird_velocity]

The neural network predicts Q-values for 2 possible actions:
 0 = Do nothing
 1 = Jump
The agent learns by:

Playing the game
Collecting experiences
Storing them in replay memory
Sampling random batches
Updating the Q-network
Improving its decisions over time
🛠 Technologies Used
Python
PyGame
PyTorch
NumPy
Deep Q-Learning
Reinforcement Learning
📦 Installation

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install the required packages:

pip install pygame numpy torch
▶️ How to Run

Run the epsilon-greedy version:

python flappy_bird_dqn_epsilon.py

Run the softmax version:

python flappy_bird_dqn_softmax.py
Game Rules

The bird moves vertically while pipes move from right to left.

The AI receives:

+1 reward for staying alive
-100 reward for hitting the top or bottom boundary

The objective is to maximize survival time.

📊 Training Methods
Epsilon-Greedy Version
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

The agent begins with high exploration and slowly becomes more confident.

Softmax Version
temperature = 1.0
temp_decay = 0.99
temp_min = 0.1

Higher temperature means more exploration. Lower temperature means more exploitation.
