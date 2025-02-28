import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# -- Game Settings --
WIDTH, HEIGHT = 400, 600
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
PIPE_GAP = 150
PIPE_SPEED = 3
GRAVITY = 0.5
JUMP_STRENGTH = -10

# -- Pygame Initialization --
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))

# -- Neural Network (Q-Network) --
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 actions: 0 (Do nothing), 1 (Jump)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -- Experience Replay Memory --
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -- Game Environment --
class Bird:
    def __init__(self):
        self.y = HEIGHT // 2
        self.x = 50
        self.velocity = 0

    def update(self, action):
        if action == 1:
            self.velocity = JUMP_STRENGTH
        self.velocity += GRAVITY
        self.y += self.velocity
        if self.y > HEIGHT:
            self.y = HEIGHT

    def draw(self):
        pygame.draw.circle(win, BLUE, (self.x, int(self.y)), 10)

class Pipe:
    def __init__(self):
        self.x = WIDTH
        self.height = random.randint(100, 400)
        self.width = 50

    def update(self):
        self.x -= PIPE_SPEED
        if self.x < -self.width:
            self.x = WIDTH
            self.height = random.randint(100, 400)

    def draw(self):
        pygame.draw.rect(win, WHITE, (self.x, 0, self.width, self.height))
        pygame.draw.rect(win, WHITE, (self.x, self.height + PIPE_GAP, self.width, HEIGHT))

# -- AI Training with Deep Q-Learning and Softmax Action Selection --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)

def select_action(state, temperature=1.0):
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    
    # Apply softmax with temperature
    probabilities = F.softmax(q_values / temperature, dim=0).cpu().numpy()
    action = np.random.choice(len(probabilities), p=probabilities)  # Sample action
    return action

def train():
    if len(memory) < 64:
        return
    batch = memory.sample(64)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + (0.99 * next_q_values * (1 - dones))

    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -- Main Game Loop --
def main():
    bird = Bird()
    pipe = Pipe()
    temperature = 1.0  # Start with high exploration
    temp_decay = 0.99  # Decay factor for temperature
    temp_min = 0.1  # Minimum temperature
    episode = 0
    clock = pygame.time.Clock()

    while True:
        state = np.array([bird.y, pipe.x, pipe.height, bird.velocity])
        action = select_action(state, temperature)
        bird.update(action)
        pipe.update()

        reward = 1
        if bird.y <= 0 or bird.y >= HEIGHT:
            reward = -100

        next_state = np.array([bird.y, pipe.x, pipe.height, bird.velocity])
        done = reward == -100
        memory.push(state, action, reward, next_state, done)
        train()

        if done:
            bird = Bird()
            pipe = Pipe()
            episode += 1
            temperature = max(temperature * temp_decay, temp_min)

            if episode % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Pygame Display
        win.fill((0, 0, 0))
        bird.draw()
        pipe.draw()
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(30)

# -- Start Game --
main()
