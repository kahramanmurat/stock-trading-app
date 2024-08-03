# agent.py
class Agent:
    def __init__(self):
        self.is_invested = False

    def act(self, state):
        if state[0] > state[1] and not self.is_invested:
            self.is_invested = True
            return 0 # Buy
        if state[0] < state[1] and self.is_invested:
            self.is_invested = False
            return 1 # Sell
        return 2 # Hold
