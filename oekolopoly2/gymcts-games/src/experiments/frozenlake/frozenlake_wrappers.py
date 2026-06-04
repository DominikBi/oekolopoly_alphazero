import gymnasium as gym

class NegativeHoleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, negative_reward=-1.0):
        super().__init__(env)
        self.negative_reward = negative_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Hole detected, if terminated und reward == 0 (Standard in FrozenLake)
        if terminated and reward == 0:
            reward = self.negative_reward
        return obs, reward, terminated, truncated, info


class ShorterEpisodeBonusWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if terminated and reward > 0:
            # Bonus = max_bonus - steps (mindestens 0)
            bonus =  1 / self.steps
            reward += bonus
            info['shorter_episode_bonus'] = bonus
        return obs, reward, terminated, truncated, info
