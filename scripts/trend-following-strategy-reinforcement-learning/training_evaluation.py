# training_evaluation.py
from agent import Agent
from environment import Env

def play_one_episode(agent, env):
    state = env.reset()
    done = False
    total_reward = 0
    agent.is_invested = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

def train_and_evaluate(train_data, test_data, feats):
    train_env = Env(train_data, feats)
    test_env = Env(test_data, feats)
    agent = Agent()

    train_reward = play_one_episode(agent, train_env)
    test_reward = play_one_episode(agent, test_env)

    return train_reward, test_reward, train_env.total_buy_and_hold, test_env.total_buy_and_hold, train_data, test_data
