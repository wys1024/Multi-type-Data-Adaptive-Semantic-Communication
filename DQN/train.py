import os
import numpy as np
import argparse
from DQN import DQN
from DQN.utils import plot_learning_curve, create_directory
from DQN.env import MyEnvironment


parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=500)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DQN/')
args = parser.parse_args()



def DQNmain():
    env = MyEnvironment()
    agent = DQN(alpha=0.0003, state_dim=env.state_dim, action_dim=env.action_dim,
                fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=256)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, eps_history = [], [], []    #列表存储数据


    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation, isTrain=True)              #根据观测值observation选择要执行的动作
            observation_, reward, done, info = env.step(action)                  #返回执行后的结果
            agent.remember(observation, action, reward, observation_, done)      #将结果存入记忆体
            agent.learn()                                                        #学习更新网络
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{}'.
              format(episode + 1, total_reward, avg_reward, agent.epsilon))

        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)

    episodes = [i for i in range(args.max_episodes)]



