import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from DQN.buffer import ReplayBuffer

import tensorflow as tf


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim1, action_dim2, action_dim3, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)   #4*256
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q1 = nn.Linear(fc2_dim, action_dim1)
        self.q2 = nn.Linear(fc2_dim, action_dim2)
        self.q3 = nn.Linear(fc2_dim, action_dim3)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)


    def forward1(self, state):
        x = T.relu(self.fc1(state.to(device)))    #第一层全连接层的计算，并通过ReLU激活函数进行非线性变换
        x = T.relu(self.fc2(x))        #第一层的输出传给第二层全连接层，再进行ReLU激活
        q1 = self.q1(x)                  #传递给最终的全连接层
        return q1

    def forward2(self, state):
        x = T.relu(self.fc1(state.to(device)))    #第一层全连接层的计算，并通过ReLU激活函数进行非线性变换
        x = T.relu(self.fc2(x))        #第一层的输出传给第二层全连接层，再进行ReLU激活
        q2 = self.q2(x)                  #传递给最终的全连接层
        return q2

    def forward3(self, state):
        x = T.relu(self.fc1(state.to(device)))    #第一层全连接层的计算，并通过ReLU激活函数进行非线性变换
        x = T.relu(self.fc2(x))        #第一层的输出传给第二层全连接层，再进行ReLU激活
        q3 = self.q3(x)                  #传递给最终的全连接层
        return q3

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class DQN:
    def __init__(self, alpha, state_dim, action_dim1, action_dim2, action_dim3, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-4,
                 max_size=1000000, batch_size=256):
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.action_space1 = [i for i in range(action_dim1)]           
        self.action_space2 = [i for i in range(action_dim2)]  
        self.action_space3 = [i for i in range(action_dim3)] 
        self.checkpoint_dir = ckpt_dir

        self.q_eval = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim1=action_dim1, action_dim2=action_dim2, action_dim3=action_dim3,     #评估当前策略
                                   fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim1=action_dim1, action_dim2=action_dim2, action_dim3=action_dim3,   #计算目标 Q 值
                                     fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim1=action_dim1, action_dim2=action_dim2, action_dim3=action_dim3,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action1, action2, action3,reward, state_, done):
         self.memory.store_transition(state, action1, action2, action3, reward, state_, done)

    def observation_to_state(self, observation):

        test1, test2, test3, snr = observation

        batch_data1 = next(iter(test1))
        test1 = tf.convert_to_tensor(batch_data1)
        test1 = tf.reshape(test1, (-1, 4))
        test1 = tf.cast(test1, tf.int64)

        test2 = test2.take(1)
        test2 = tf.data.experimental.get_single_element(test2)
        test2 = test2[0]
        test2 = test2[0]
        test2 = tf.reshape(test2, [-1, 4])
        test2 = tf.cast(test2, tf.int64)

        test3 = test3.take(1)
        test3 = tf.data.experimental.get_single_element(test3)
        test3 = test3[0]
        test3 = tf.reshape(test3, [-1, 4])
        test3 = tf.cast(test3, tf.int64)

        snr = tf.convert_to_tensor(snr)
        snr = tf.reshape(snr, [-1, 1])
        snr = tf.broadcast_to(snr, [1, 4])
        snr = tf.cast(snr, tf.int64)

        merged_tensor = tf.concat([test1, test2, test3, snr], axis=0)
        state = tf.reshape(merged_tensor, (-1, 4))
        state = T.tensor(state.numpy(), dtype=T.float32)


        print('state:',state.shape)

        return state



    def choose_action1(self, observation, isTrain=True):
        actions = self.q_eval.forward1(observation)
        action1 = T.argmax(actions).item()
        if (np.random.random() < self.epsilon) and isTrain:
            action1 = np.random.randint(1,16)
        return action1

    def choose_action2(self, observation, isTrain=True):
        actions = self.q_eval.forward2(observation)
        action2 = T.argmax(actions).item()
        if (np.random.random() < self.epsilon) and isTrain:
            action2 = np.random.randint(1, 48)
        return action2

    def choose_action3(self, observation, isTrain=True):
       actions = self.q_eval.forward3(observation)
       action3 = T.argmax(actions).item()
       if (np.random.random() < self.epsilon) and isTrain:
           action3 = np.random.randint(1, 128)
       return action3



    def learn(self):
        if not self.memory.ready():
            return
        states, action1, action2, action3, rewards, next_states, terminals = self.memory.sample_buffer()   #随机采样出一批经验数据
        batch_idx = np.arange(self.batch_size)

        states_tensor = T.tensor(states, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            q_1 = self.q_target.forward1(next_states_tensor)
            q_2 = self.q_target.forward2(next_states_tensor)
            q_3 = self.q_target.forward3(next_states_tensor) #计算下一个状态的Q值
            q_1[terminals_tensor] = 0.0
            q_2[terminals_tensor] = 0.0
            q_3[terminals_tensor] = 0.0

            target1 = rewards_tensor + self.gamma * T.max(q_1, dim=-1)[0]
            target2 = rewards_tensor + self.gamma * T.max(q_2, dim=-1)[0]
            target3 = rewards_tensor + self.gamma * T.max(q_3, dim=-1)[0]
        q1 = self.q_eval.forward1(states_tensor)[batch_idx, action1]
        q2 = self.q_eval.forward2(states_tensor)[batch_idx, action2]
        q3 = self.q_eval.forward3(states_tensor)[batch_idx, action3] #获得当前动作的Q值

        loss1 = F.mse_loss(q1, target1.detach())
        loss2 = F.mse_loss(q2, target2.detach())
        loss3 = F.mse_loss(q3, target3.detach())

        self.q_eval.optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        loss3.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DQN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DQN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')
