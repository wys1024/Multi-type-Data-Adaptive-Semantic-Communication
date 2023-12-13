import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim1, action_dim2, action_dim3, max_size, batch_size):             #初始化了重放缓冲区的属性
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory1 = np.zeros((self.mem_size, ))
        self.action_memory2 = np.zeros((self.mem_size,))
        self.action_memory3 = np.zeros((self.mem_size,))
        self.reward_memory = np.zeros((self.mem_size, ))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size, ), dtype=bool)

    def store_transition(self, state, action1, action2, action3, reward, state_, done):             #存储一个经验样本      放入

        mem_idx = self.mem_cnt % self.mem_size
        self.state_memory = state
        self.action_memory1[mem_idx] = action1
        self.action_memory2[mem_idx] = action2
        self.action_memory3[mem_idx] = action3
        self.reward_memory[mem_idx] = reward
        self.next_state_memory = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    def sample_buffer(self):                                                    #从重放缓冲区中随机采样一批经验样本     取出
        mem_len = min(self.mem_size, self.mem_cnt)
        batch = np.random.choice(mem_len, self.batch_size, replace=True)
        states = self.state_memory[batch]
        action1 = self.action_memory1[batch]
        action2 = self.action_memory2[batch]
        action3 = self.action_memory3[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, action1, action2, action3, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt > self.batch_size

