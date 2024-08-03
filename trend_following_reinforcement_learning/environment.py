# environment.py
class Env:
    def __init__(self, df, feats):
        self.df = df
        self.n = len(df)
        self.current_idx = 0
        self.action_space = [0, 1, 2] # BUY, SELL, HOLD
        self.invested = 0
        self.states = self.df[feats].to_numpy()
        self.rewards = self.df['LogReturn'].to_numpy()
        self.total_buy_and_hold = 0

    def reset(self):
        self.current_idx = 0
        self.total_buy_and_hold = 0
        self.invested = 0
        return self.states[self.current_idx]

    def step(self, action):
        self.current_idx += 1
        if self.current_idx >= self.n:
            raise Exception("Episode already done")

        if action == 0: # BUY
            self.invested = 1
        elif action == 1: # SELL
            self.invested = 0
        
        if self.invested:
            reward = self.rewards[self.current_idx]
        else:
            reward = 0

        next_state = self.states[self.current_idx]
        self.total_buy_and_hold += self.rewards[self.current_idx]
        done = (self.current_idx == self.n - 1)
        return next_state, reward, done
