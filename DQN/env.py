class MyEnvironment:
    def __init__(self, state_dim = 4, action_dim1 = 16, action_dim2 = 48, action_dim3 = 128):
        self.state_dim = state_dim   # 状态维度
        self.action_dim1 = action_dim1 # 动作维度
        self.action_dim2 = action_dim2
        self.action_dim3 = action_dim3

    def reset(self, test1, test2, test3, snr):
        # 重置环境并返回初始状态
        state = [test1, test2, test3, snr]
        return state










