import gymnasium as gym


class OekoRoundWeightedBalanceRewardWrapper(gym.Wrapper):
    EARLY_ROUND_TARGETS = {
        "SANITATION": 20,
        "PRODUCTION": 12,
        "EDUCATION": 21,
        "QUALITY_OF_LIFE": 20,
        "ENVIRONMENT": 15,
        "POPULATION": 34,
        "POLITICS": 0,
    }

    def __init__(
        self,
        env,
        start_round=10,
        max_round=30,
        weight_power=2.0,
        survival_reward=0.25,
        die_reward=-200,
        politics_low_reward=-100,
        politics_low_threshold=-5,
    ):
        super().__init__(env)
        self.start_round = start_round
        self.max_round = max_round
        self.weight_power = weight_power
        self.die_reward = die_reward
        self.survival_reward = survival_reward

    def mod_reward(self, terminated: bool, truncated: bool):
        oeko_env = self.env.unwrapped
        round_no = oeko_env.V[oeko_env.ROUND]
        scale_reward = (round_no / self.max_round) ** self.weight_power
        reward = 0

        if round_no < self.start_round:
            reward = self.survival_reward
            # reward = self.early_round_target_penalty()
            if terminated or truncated:
                reward += self.die_reward
        elif round_no >= self.start_round:
            reward = oeko_env.balance * scale_reward

        return reward

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.mod_reward(terminated, truncated)
        return obs, reward, terminated, truncated, info


class OekoPerRoundRewardWrapper(gym.Wrapper):
    def __init__(self, env, per_round_reward=1):
        super().__init__(env)
        self.per_round_reward = per_round_reward

    def mod_reward(self):
        if self.env.unwrapped.done and self.env.unwrapped.V[
            self.env.unwrapped.ROUND
        ] in range(10, 31):
            reward = self.env.unwrapped.balance
        else:
            reward = self.per_round_reward
        return reward

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.mod_reward()
        return obs, reward, terminated, truncated, info


class OekoAuxRewardWrapper(gym.Wrapper):
    def __init__(self, env, scaling=1):
        super().__init__(env)
        self.scaling = scaling

    def mod_reward(self):
        if self.env.unwrapped.done and self.env.unwrapped.V[
            self.env.unwrapped.ROUND
        ] in range(10, 31):
            return self.env.unwrapped.balance
        else:
            production_reward = 14 - abs(
                15 - self.env.unwrapped.V[self.env.unwrapped.PRODUCTION]
            )
            population_reward = 23 - abs(
                24 - self.env.unwrapped.V[self.env.unwrapped.POPULATION]
            )
            return self.scaling * (production_reward + population_reward)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.mod_reward()
        return obs, reward, terminated, truncated, info
