import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer1,QTrainer2
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 512, 3)
        self.trainer1 = QTrainer1(self.model,self.model, lr=LR, gamma=self.gamma)
        self.trainer2 = QTrainer2(self.model,self.model, lr=LR, gamma=self.gamma)
        


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory1(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer1.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
    def train_long_memory2(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer2.train_step(states, actions, rewards, next_states, dones)
        
        
    def train_short_memory1(self, state, action, reward, next_state, done):
        self.trainer1.train_step(state, action, reward, next_state, done)
        
        
    def train_short_memory2(self, state, action, reward, next_state, done):
        self.trainer2.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        alpha=0.4
        self.epsilon =80*np.exp(-self.n_games/34)
        final_move = [0,0,0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def per(disti):
       n = len(disti)
       if n == 0:
        return False  # An empty list is not periodic

       for start in range(n):
        for period in range(1, (n - start) // 2 + 1):
            match = True
            for i in range(period):
                for j in range(1, (n - start) // period):
                    if start + i + j * period >= n or disti[start + i] != disti[start + i + j * period]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return True
        return False



def train(disti):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    agent = Agent()
    game = SnakeGameAI()
    it=0
    while True:
        it=it+1
        # get old state
        state_old = agent.get_state(game)
        #dist1=abs(game.head.x-game.food.x)+abs(game.head.y-game.food.y)
        

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        dist2=abs(game.head.x-game.food.x)+abs(game.head.y-game.food.y)
        disti.append(dist2)
        if len(disti)>2:
            prev=disti[-1]
            prev2=disti[-2]
            if prev>prev2:
                reward-=2
            else:
                reward+=2
        if Agent.per(disti)==True:
            reward-=8
        
        
        # train short memory
        if agent.n_games<200:
            if it%7==1:
               agent.train_short_memory2(state_old, final_move, reward, state_new, done)
            else:
               agent.train_short_memory1(state_old, final_move, reward, state_new, done)
        else:
            if it%7==1:
               agent.train_short_memory2(state_old, final_move, reward, state_new, done)
            else:
               agent.train_short_memory1(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            disti=[]
            agent.n_games += 1
            if agent.n_games<200:
              if it%7==1:
                agent.train_long_memory2()
              else:
                 agent.train_long_memory1()
            else:
                if it%7==1:
                 agent.train_long_memory2()
                else:
                 agent.train_long_memory1()
                           

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    disti=[]
    train(disti)