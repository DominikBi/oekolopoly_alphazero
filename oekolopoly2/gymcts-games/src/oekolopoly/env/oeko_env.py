import pprint
import sys
from typing import Any

from gymnasium.core import WrapperObsType
import oekolopoly.env.get_boxes as gb

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from oekolopoly.env.oeko_wrappers import (
    OekoPerRoundRewardWrapper,
    OekoAuxRewardWrapper,
    OekoRoundWeightedBalanceRewardWrapper,
)
from oekolopoly.env.render_functions import render_asni, render_round_transition_asni


class OekoActionBuilderWrapper(gym.ActionWrapper):
    """
    This wrapper builds up an action by spending one point at a time and performs a step
    in the underlying environment once action 0 (move to next round) is chosen.
    This allows to only choose valid actions for the underlying environment.
    However, this wrapper may have invalid actions in the action space, which need to be masked by an ActionMasker wrapper.

    A helper function is provided to get the valid actions for a given state, which can be used in the action_mask_fn of the ActionMasker wrapper.

    Valid actions:

    0: Move to Next Round (no changes to V, just move to next round)

    1: Increase Sanitation by 1

    2: Increase Production by 1
    3: Decrease Production by 1

    4: Increase Education by 1
    5: Increase Quality of Life by 1
    6: Increase Population Growth by 1

    7: Increase Population Growth by 1 (via additional points)
    8: Decrease Population Growth by 1 (via additional points)
    """

    def _calc_additional_population_points(self):
        education_level = self.env.unwrapped.V[self.env.unwrapped.EDUCATION]
        if education_level in range(21, 24):
            return 3
        elif education_level in range(24, 28):
            return 4
        elif education_level in range(28, 30):
            return 5
        else:
            return 0

    def _reset_wrapper_state(self):
        self._current_action_dict = {
            "Sanitation": 0,
            "Production": 0,
            "Education": 0,
            "Quality of Life": 0,
            "Population Growth": 0,
            "Population Growth extra": 0,
        }
        self._available_action_points = self.env.unwrapped.V[self.env.unwrapped.POINTS]
        self._available_extra_points = self._calc_additional_population_points()

        self._production_change_direction: None | str = None  # None, "up", "down".
        self._population_extra_change_direction: None | str = (
            None  # None, "up", "down".
        )
        self._cached_obs = None

    def __init__(
        self,
        env,
        auxilary_reward=False,
        expose_next_round_action=True,
        auto_next_round_when_no_points=False,
    ):
        super().__init__(env)

        self.auxilary_reward = auxilary_reward
        self.expose_next_round_action = expose_next_round_action
        self.auto_next_round_when_no_points = auto_next_round_when_no_points

        # for step caching
        self._cached_obs = None
        self._cached_reward = None
        self._cached_terminal = False
        self._cached_truncated = False
        self._cached_info = {}

        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete(
            [
                29,  # 0 Sanitation
                29,  # 1 Production
                29,  # 2 Education
                29,  # 3 Quality of Life
                29,  # 4 Population Growth
                29,  # 5 Environment
                48,  # 6 Population
                48,  # 7 Politics
                31,  # 8 Round
                37,  # 9 Actionpoints for next round
                # extended by buffered values for the currently build up action
                # which are not yet applied to the underlying environment,
                # but will be applied once action 0 (move to next round) is chosen
                29,  # 10 buffered Sanitation
                # 11 buffered Production, *2+1 since it can be increased and decreased,
                # so the range of possible buffered values is from -28 to +28
                29 * 2 + 1,
                29,  # 12 buffered Education
                29,  # 13 buffered Quality of Life
                29,  # 14 buffered Population Growth
                11,  # 15 buffered Population Growth extra (via additional points)
            ]
        )

    def step_next_round(self):
        return self.step(action=0)

    def step_increase_sanitation(self):
        return self.step(action=1)

    def step_increase_production(self):
        return self.step(action=2)

    def step_decrease_production(self):
        return self.step(action=3)

    def step_increase_education(self):
        return self.step(action=4)

    def step_increase_quality_of_life(self):
        return self.step(action=5)

    def step_increase_population_growth(self):
        return self.step(action=6)

    def step_increase_population_growth_extra(self):
        return self.step(action=7)

    def step_decrease_population_growth_extra(self):
        return self.step(action=8)

    def valid_action_mask(self):
        # action 0 commits the buffered point allocation to the next round. Some
        # agents hide it and let the wrapper commit automatically once no points
        # can be allocated anymore.
        next_round_valid = self.expose_next_round_action

        env_pop_growth = (
            self.env.unwrapped.V[self.env.unwrapped.POPULATION_GROWTH]
            + self._current_action_dict["Population Growth"]
            + self._current_action_dict["Population Growth extra"]
        )
        env_pop_growth_max_value = self.env.unwrapped.Vmax[
            self.env.unwrapped.POPULATION_GROWTH
        ]
        env_pop_growth_min_value = self.env.unwrapped.Vmin[
            self.env.unwrapped.POPULATION_GROWTH
        ]

        # actions 7, 8 are valid based on the education level and the available extra points
        valid_7_8 = self._available_extra_points > 0 and (
            self.env.unwrapped.V[self.env.unwrapped.EDUCATION] >= 21
        )

        increase_population_growth_extra = False
        decrease_population_growth_extra = False

        if self._available_extra_points > 0:
            if (
                self._population_extra_change_direction in ["up", None]
                and (env_pop_growth + 1) <= env_pop_growth_max_value
            ):
                increase_population_growth_extra = True
            if (
                self._population_extra_change_direction in ["down", None]
                and (env_pop_growth - 1) >= env_pop_growth_min_value
            ):
                decrease_population_growth_extra = True

        # case there are no available action points, actions 1-6 are not valid
        # in this case we can return early with only the next round and the extra population growth actions as valid
        if self._available_action_points <= 0:
            mask = np.array(
                [
                    next_round_valid,
                    False,  # increase sanitation
                    False,  # increase production
                    False,  # decrease production
                    False,  # increase education
                    False,  # increase quality of life
                    False,  # increase population growth
                    increase_population_growth_extra,
                    decrease_population_growth_extra,
                ]
            )
            return mask

        # for the remainin action it has to be check wether the current value of the corresponding variable
        # is already at the maximum or minimum, as defined in the underlying environment, to determine
        # if the action is valid or not

        # action 1
        increase_sanitation_valid = (
            self.env.unwrapped.V[self.env.unwrapped.SANITATION] + 1
        ) <= self.env.unwrapped.Vmax[
            self.env.unwrapped.SANITATION
        ] - self._current_action_dict[
            "Sanitation"
        ]
        # actions 2 and  3
        increase_production_valid = (
            self.env.unwrapped.V[self.env.unwrapped.PRODUCTION] + 1
        ) <= self.env.unwrapped.Vmax[
            self.env.unwrapped.PRODUCTION
        ] - self._current_action_dict[
            "Production"
        ]
        decrease_production_valid = (
            self.env.unwrapped.V[self.env.unwrapped.PRODUCTION] - 1
        ) >= self.env.unwrapped.Vmin[
            self.env.unwrapped.PRODUCTION
        ] - self._current_action_dict[
            "Production"
        ]
        # actions 4
        increase_education_valid = (
            self.env.unwrapped.V[self.env.unwrapped.EDUCATION] + 1
        ) <= self.env.unwrapped.Vmax[
            self.env.unwrapped.EDUCATION
        ] - self._current_action_dict[
            "Education"
        ]
        # actions 5
        increase_quality_of_life_valid = (
            self.env.unwrapped.V[self.env.unwrapped.QUALITY_OF_LIFE] + 1
        ) <= self.env.unwrapped.Vmax[
            self.env.unwrapped.QUALITY_OF_LIFE
        ] - self._current_action_dict[
            "Quality of Life"
        ]
        # actions 6
        # increase_population_growth_valid = (self.env.unwrapped.V[self.env.unwrapped.POPULATION_GROWTH] + 1) <= self.env.unwrapped.Vmax[self.env.unwrapped.POPULATION_GROWTH]

        mask = np.array(
            [
                next_round_valid,
                increase_sanitation_valid,
                increase_production_valid,
                decrease_production_valid,
                increase_education_valid,
                increase_quality_of_life_valid,
                # increase_population_growth_valid,
                # don't allow to increase population growth via normal points, only via extra points
                # that's the way it is descibed in the german manual
                # source: https://www.spiele4us.de/wp-content/uploads/2022/10/oekolopoly-grau-gebraucht-g003700960.pdf
                False,
                increase_population_growth_extra,
                decrease_population_growth_extra,
            ]
        )
        return mask

    def _step_next_round(self):
        action_to_pass = np.array(
            [
                self._current_action_dict["Sanitation"],
                self._current_action_dict["Production"],
                self._current_action_dict["Education"],
                self._current_action_dict["Quality of Life"],
                self._current_action_dict["Population Growth"],
                self._current_action_dict["Population Growth extra"],
            ]
        )

        # subtract Amin to get the actual action values to pass to the underlying environment
        # np.array([ 0,-28,  0,  0,  0, -5]) will be added in the __iner_step function so this plus minus zero overall
        action_to_pass -= np.array([0, -28, 0, 0, 0, -5])

        # print(f"performing action: \n {pprint.pformat(self._current_action_dict)} ", action_to_pass)

        obs, reward, terminated, truncated, info = self.env.step(action=action_to_pass)
        self._reset_wrapper_state()

        self._cached_obs = obs
        self._cached_reward = reward
        self._cached_terminal = terminated
        self._cached_truncated = truncated
        self._cached_info = info

        return (
            self._extend_obs_by_buffered_values(obs=obs),
            reward,
            terminated,
            truncated,
            info,
        )

    def _should_auto_step_next_round(self):
        return (
            self.auto_next_round_when_no_points
            and not self._cached_terminal
            and not np.any(self.valid_action_mask()[1:])
        )

    def step(self, action):
        action_mask = self.valid_action_mask()
        (
            a0_valid,
            a1_valid,
            a2_valid,
            a3_valid,
            a4_valid,
            a5_valid,
            a6_valid,
            a7_valid,
            a8_valid,
        ) = action_mask

        if action == 0 and a0_valid:
            # Move to next round
            return self._step_next_round()

        elif action == 1 and a1_valid:
            # Increase Sanitation by 1
            self._current_action_dict["Sanitation"] += 1
            self._available_action_points -= 1

        elif action == 2 and a2_valid:
            self._current_action_dict["Production"] += 1
            self._available_action_points -= 1
            self._production_change_direction = "up"

        elif action == 3 and a3_valid:
            # Decrease Production by 1
            self._current_action_dict["Production"] -= 1
            self._available_action_points -= 1
            self._production_change_direction = "down"

        elif action == 4 and a4_valid:
            # Increase Education by 1
            self._current_action_dict["Education"] += 1
            self._available_action_points -= 1

        elif action == 5 and a5_valid:
            self._current_action_dict["Quality of Life"] += 1
            self._available_action_points -= 1

        elif action == 6 and a6_valid:
            self._current_action_dict["Population Growth"] += 1
            self._available_action_points -= 1

        elif action == 7 and a7_valid:
            # Increase Population Growth by 1 (via additional points)
            self._current_action_dict["Population Growth extra"] += 1
            self._available_extra_points -= 1
            self._population_extra_change_direction = "up"

        elif action == 8 and a8_valid:
            self._current_action_dict["Population Growth extra"] -= 1
            self._available_extra_points -= 1
            self._population_extra_change_direction = "down"

        if self._should_auto_step_next_round():
            return self._step_next_round()

        if self.env.unwrapped.done and self.env.unwrapped.V[
            self.env.unwrapped.ROUND
        ] in range(0, 10):
            rew = 0.0
        elif self.auxilary_reward:
            rew = 0.1
        else:
            rew = 0.0

        return (
            self._extend_obs_by_buffered_values(obs=self._cached_obs),
            rew,
            self._cached_terminal,
            self._cached_truncated,
            {},
        )

    def _extend_obs_by_buffered_values(self, obs):
        # this function concatinates the values in the buffer to the observation, so that the a RL agent, has
        # access to currently buffered values, which are not yet applied to the underlying environment,
        # but will be applied once action 0 (move to next round) is chosen
        obs[9] = self._available_action_points
        buffered_values = np.array(
            [
                self._current_action_dict["Sanitation"],
                self._current_action_dict["Production"]
                + 28,  # add 28 to the buffered production change to get a positive value, since it can be increased and decreased, so the range of possible buffered values is from -28 to +28
                self._current_action_dict["Education"],
                self._current_action_dict["Quality of Life"],
                self._current_action_dict["Population Growth"],
                self._current_action_dict["Population Growth extra"]
                + 5,  # add 5 to the buffered population growth extra change to get a positive value, since it can be increased and decreased, so the range of possible buffered values is from -5 to +5
            ]
        )
        extended_obs = np.concatenate([obs, buffered_values])
        return extended_obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        self._reset_wrapper_state()
        self._cached_obs = obs.copy()
        return self._extend_obs_by_buffered_values(obs), info


class OekoEnv(gym.Env):

    PRINT_STEP_TRANSITIONS = False
    PRINT_DONE_REASONS = False

    SANITATION = 0
    PRODUCTION = 1
    EDUCATION = 2
    QUALITY_OF_LIFE = 3
    POPULATION_GROWTH = 4
    ENVIRONMENT = 5
    POPULATION = 6
    POLITICS = 7
    ROUND = 8
    POINTS = 9

    V_NAMES = [
        "Sanitation",
        "Production",
        "Education",
        "Quality of Life",
        "Population Growth",
        "Environment",
        "Population",
        "Politics",
        "Round",
        "Points",
    ]

    ACT_NAMES = [
        "SANITATION",
        "PRODUCTION",
        "EDUCATION",
        "QUALITY OF LIFE",
        "POPULATION GROWTH",
        "EXTRA",
    ]

    OBS_NAMES = V_NAMES[:-2]

    def __init__(self, render_mode=None):

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.last_v = None
        self.init_v = np.array(
            [
                1,  # 0 Sanitation
                12,  # 1 Production
                4,  # 2 Education
                10,  # 3 Quality of Life
                20,  # 4 Population Growth
                13,  # 5 Environment
                21,  # 6 Population
                0,  # 7 Politics
                0,  # 8 Round
                8,  # 9 Points
            ]
        )

        #                      0   1   2   3   4   5   6    7   8   9
        #                      S   Pr  Ed  Q   PG  En  Pop  Pol R   AP
        self.Vmin = np.array([1, 1, 1, 1, 1, 1, 1, -10, 0, 0])
        self.Vmax = np.array([29, 29, 29, 29, 29, 29, 48, 37, 30, 36])

        self.Amin = np.array([0, -28, 0, 0, 0, -5])
        self.Amax = np.array([28, 28, 28, 28, 28, 5])

        self.action_space = spaces.MultiDiscrete(
            [
                29,  # 0 Sanitation
                57,  # 1 Production
                29,  # 2 Education
                29,  # 3 Quality of Life
                29,  # 4 Population Growth
                11,  # 5 box9 Education > Population Growth
            ]
        )

        self.observation_space = spaces.MultiDiscrete(
            [
                29,  # 0 Sanitation
                29,  # 1 Production
                29,  # 2 Education
                29,  # 3 Quality of Life
                29,  # 4 Population Growth
                29,  # 5 Environment
                48,  # 6 Population
                48,  # 7 Politics
                31,  # 8 Round
                37,  # 9 Actionpoints for next round
            ]
        )

        self.done = False
        self.done_info = ""

    def seed(self, seed):
        """Dummy method needed by stable baselines3 when passing seed to model.
        The Oekolopoly environment is entirely deterministic"""
        pass

    def close(self):
        if self.clock is not None:
            self.clock = None

    def render(self):
        board = render_asni(env=self)
        print(board)

    def update_values(self, action):

        OOR = " out of allowed range"
        done = False
        done_info = None
        extra_points = action[5]

        # Update V and boxes
        box1 = gb.get_box1(self.V[self.SANITATION])
        if not done:
            self.V[self.ENVIRONMENT] += box1
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                if self.V[self.ENVIRONMENT] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = "Environment too " + l + str(self.V[self.ENVIRONMENT]) + OOR

        if not done:
            box2 = gb.get_box2(self.V[self.SANITATION])
            self.V[self.SANITATION] += box2
            if self.V[self.SANITATION] not in range(1, 30):
                done = True
                if self.V[self.SANITATION] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = "Sanitation too " + l + str(self.V[self.SANITATION]) + OOR

        if not done:
            box3 = gb.get_box3(self.V[self.PRODUCTION])
            self.V[self.PRODUCTION] += box3
            if self.V[self.PRODUCTION] not in range(1, 30):
                done = True
                if self.V[self.PRODUCTION] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = "Production too " + l + str(self.V[self.PRODUCTION]) + OOR

        if not done:
            box4 = gb.get_box4(self.V[self.PRODUCTION])
            self.V[self.ENVIRONMENT] += box4
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                if self.V[self.ENVIRONMENT] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = "Environment too " + l + str(self.V[self.ENVIRONMENT]) + OOR

        if not done:
            box5 = gb.get_box5(self.V[self.ENVIRONMENT])
            self.V[self.ENVIRONMENT] += box5
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                if self.V[self.ENVIRONMENT] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = "Environment too " + l + str(self.V[self.ENVIRONMENT]) + OOR

        if not done:
            box6 = gb.get_box6(self.V[self.ENVIRONMENT])
            self.V[self.QUALITY_OF_LIFE] += box6
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                if self.V[self.QUALITY_OF_LIFE] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = (
                    "Quality of Life too " + l + str(self.V[self.QUALITY_OF_LIFE]) + OOR
                )

        if not done:
            box7 = gb.get_box7(self.V[self.EDUCATION])
            self.V[self.EDUCATION] += box7
            if self.V[self.EDUCATION] not in range(1, 30):
                done = True
                if self.V[self.EDUCATION] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = "Education too " + l + str(self.V[self.EDUCATION]) + OOR

        if not done:
            box8 = gb.get_box8(self.V[self.EDUCATION])
            self.V[self.QUALITY_OF_LIFE] += box8
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                if self.V[self.QUALITY_OF_LIFE] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = (
                    "Quality of Life too " + l + str(self.V[self.QUALITY_OF_LIFE]) + OOR
                )

        if not done:
            if self.V[self.EDUCATION] in range(21, 24):
                extra_points = max(-3, min(3, extra_points))
            if self.V[self.EDUCATION] in range(24, 28):
                extra_points = max(-4, min(4, extra_points))
            if self.V[self.EDUCATION] in range(28, 30):
                extra_points = max(-5, min(5, extra_points))
            if self.V[self.EDUCATION] < 21:
                extra_points = 0
            box9 = gb.get_box9(self.V[self.EDUCATION], extra_points)
            self.V[self.POPULATION_GROWTH] += box9
            if self.V[self.POPULATION_GROWTH] not in range(1, 30):
                done = True
                if self.V[self.POPULATION_GROWTH] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = (
                    "Population Growth too "
                    + l
                    + str(self.V[self.POPULATION_GROWTH])
                    + OOR
                )

        if not done:
            box10 = gb.get_box10(self.V[self.QUALITY_OF_LIFE])
            self.V[self.QUALITY_OF_LIFE] += box10
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                if self.V[self.QUALITY_OF_LIFE] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = (
                    "Quality of Life too " + l + str(self.V[self.QUALITY_OF_LIFE]) + OOR
                )

        if not done:
            box11 = gb.get_box11(self.V[self.QUALITY_OF_LIFE])
            self.V[self.POPULATION_GROWTH] += box11
            if self.V[self.POPULATION_GROWTH] not in range(1, 30):
                done = True
                if self.V[self.POPULATION_GROWTH] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = (
                    "Population Growth too "
                    + l
                    + str(self.V[self.POPULATION_GROWTH])
                    + OOR
                )

        if not done:
            box12 = gb.get_box12(self.V[self.QUALITY_OF_LIFE])
            self.V[self.POLITICS] += box12
            if self.V[self.POLITICS] not in range(-10, 38):
                done = True
                if self.V[self.POLITICS] > 37:
                    l = "high. "
                else:
                    l = "low. "
                done_info = "Politics too " + l + str(self.V[self.POLITICS]) + OOR

        if not done:
            box13 = gb.get_box13(self.V[self.POPULATION_GROWTH])
            boxW = gb.get_boxW(self.V[self.POPULATION])
            self.V[self.POPULATION] += box13 * boxW
            if self.V[self.POPULATION] not in range(1, 49):
                done = True
                if self.V[self.POPULATION] > 48:
                    l = "high. "
                else:
                    l = "low. "
                done_info = "Population too " + l + str(self.V[self.POPULATION]) + OOR

        if not done:
            box14 = gb.get_box14(self.V[self.POPULATION])
            self.V[self.QUALITY_OF_LIFE] += box14
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                if self.V[self.QUALITY_OF_LIFE] > 29:
                    l = "high. "
                else:
                    l = "low. "
                done_info = (
                    "Quality of Life too " + l + str(self.V[self.QUALITY_OF_LIFE]) + OOR
                )

        if done and self.PRINT_DONE_REASONS:
            print(done_info)
        return self.V, done, done_info

    def step_kwarg(
        self,
        sanitation=0,
        production=0,
        education=0,
        quality_of_life=0,
        poulation_growth=0,
        extra_poulation_growth=0,
    ):
        self.step(
            np.array(
                [
                    sanitation,
                    production,
                    education,
                    quality_of_life,
                    poulation_growth,
                    extra_poulation_growth,
                ]
            )
        )

    def step(self, action):
        clipping = True
        # gym style step
        # print("performing action: ", action)
        obs, reward, terminated, info = self.__inner_step(action, clipping)
        # return gymnasium style step
        truncated = False
        return obs, reward, terminated, truncated, info

    def step_w_o_clip(self, action):
        clipping = False
        return self.__inner_step(action, clipping)

    def __inner_step(self, action, clipping=True):
        """realized as inner class because step(self,action) does not allow extra arguments"""
        # print("inner step action: ", action)
        assert self.action_space.contains(
            action
        ), f"Action not in action_space: {action}"

        # Transform action space
        action = action + self.Amin  # self.Amin = np.array([ 0,-28,  0,  0,  0, -5])
        self.prev_action = self.curr_action
        self.curr_action = action.copy()

        # print("transformed action: ", action)

        # Init
        self.done = False
        used_points = 0

        # Sum points from action
        used_points += action[self.SANITATION]
        used_points += abs(action[self.PRODUCTION])
        used_points += action[self.EDUCATION]
        used_points += action[self.QUALITY_OF_LIFE]
        used_points += action[self.POPULATION_GROWTH]

        # print("used points: ", used_points)
        # print("available points: ", self.V[self.POINTS])

        if self.PRINT_STEP_TRANSITIONS:
            # 0 Sanitation
            # 1 Production
            # 2 Education
            # 3 Quality of Life
            # 4 Population Growth
            # 5 box9 Education > Population Growth
            action_render_str = render_round_transition_asni(
                sanitation=action[0],
                production=action[1],
                education=action[2],
                quality_of_life=action[3],
                population_growth=action[4] + action[5],
                available_points=self.V[self.POINTS],
                used_points=used_points,
            )
            print(action_render_str)

        if used_points < 0 or used_points > self.V[self.POINTS]:
            self.done = True
            if used_points < 0:
                done_reason = "Tried to use negative amount of actionpoints. "
            elif used_points > self.V[self.POINTS]:
                done_reason = "Tried to exceed available amount of actionpoints. "
            done_reason += f"Tried to use {used_points} action points, but only between 0 and {self.V[self.POINTS]} are available"
            print(done_reason)
            return (
                self.obs,
                0,
                self.done,
                {
                    "balance (always)": self.balance_always,
                    "balance_numerator (always)": self.balance_numerator_always,
                    "balance": self.balance,
                    "balance_numerator": self.balance_numerator,
                    "round": self.V[self.ROUND],
                    "done_reason": done_reason,
                    "valid_move": False,
                    "invalid_move_info": "Unavailable number of actionpoints were used in this round.",
                },
            )
        assert (
            0 <= used_points <= self.V[self.POINTS]
        ), f"Action takes too many points: action={action} POINTS={self.V[self.POINTS]})"

        for i in range(5):
            if self.V[i] + action[i] not in range(self.Vmin[i], self.Vmax[i] + 1):
                self.done = True
                if self.V[i] + action[i] < self.Vmin[i]:
                    done_reason = f"Distribution of actionpoints pushes {self.V_NAMES[i]} below limit. "
                    print(done_reason)
                elif self.V[i] + action[i] > self.Vmax[i]:
                    done_reason = f"Distribution of actionpoints pushes {self.V_NAMES[i]} above limit. "
                    print(done_reason)
                done_reason += f"Tried to use {self.V[i] + action[i]} action points for {self.V_NAMES[i]}, but only between {self.Vmin[i]} and {self.Vmax[i]} are available"
                print(done_reason)
                return (
                    self.obs,
                    0,
                    self.done,
                    {
                        "balance (always)": self.balance_always,
                        "balance_numerator (always)": self.balance_numerator_always,
                        "balance": self.balance,
                        "balance_numerator": self.balance_numerator,
                        "round": self.V[self.ROUND],
                        "done_reason": done_reason,
                        "valid_move": False,
                        "invalid_move_info": f"Unavailable number of actionpoints assigned to {self.V_NAMES[i]}.",
                    },
                )
            assert (self.V[i] + action[i]) in range(
                self.Vmin[i], self.Vmax[i] + 1
            ), f"Action puts region out of action[{i}]: action={action} V={self.V}"

        # The turn is valid

        for i in range(5):
            self.V[i] += action[i]

        # Update boxes and V accordingly
        self.V, self.done, self.done_info = self.update_values(action)

        # Update points and round
        self.V[self.POINTS] -= used_points
        self.V[self.ROUND] += 1

        # Clip values if not in range
        if clipping:
            for i in range(8):
                if self.V[i] not in range(self.Vmin[i], self.Vmax[i] + 1):
                    self.V[i] = max(self.Vmin[i], min(self.Vmax[i], self.V[i]))
                    self.done = True

        if self.V[self.ROUND] == 30:
            self.done = True
            self.done_info = "Maximum number of rounds reached."

        # Points for next round
        if self.done:
            self.V[self.POINTS] = 0
        else:
            boxA = gb.get_boxA(self.V[self.POPULATION])
            boxB = gb.get_boxB(self.V[self.POLITICS])
            boxC = gb.get_boxC(self.V[self.PRODUCTION])
            boxV = gb.get_boxV(self.V[self.PRODUCTION])
            boxD = gb.get_boxD(self.V[self.QUALITY_OF_LIFE])

            self.V[self.POINTS] += boxA * boxV
            self.V[self.POINTS] += boxB
            self.V[self.POINTS] += boxC
            self.V[self.POINTS] += boxD

        if self.V[self.POINTS] < 0:
            self.V[self.POINTS] = 0
            self.done = True
            self.done_info = "Minimum amount of actionpoints reached."
        # define 36 as max of action points
        if self.V[self.POINTS] > 36:
            self.V[self.POINTS] = 36
            self.done = True
            self.done_info = "Maximum number of actionpoints reached."

        boxD = gb.get_boxD(self.V[self.QUALITY_OF_LIFE])
        a = float((boxD * 3 + self.V[self.POLITICS]) * 10)
        b = float(self.V[self.ROUND] + 3)
        self.balance_numerator_always = int(a)
        self.balance_always = a / b

        # Transform V in obs
        self.obs = self.V - self.Vmin
        if clipping:
            assert self.observation_space.contains(
                self.obs
            ), f"obs not in observation_space: obs={self.obs}"

        if self.V[self.ROUND] in range(10, 31):
            self.balance = self.balance_always
            self.balance_numerator = self.balance_numerator_always
        else:
            self.balance = 0
            self.balance_numerator = 0

        if self.done and self.V[self.ROUND] in range(10, 31):
            reward = self.balance
        else:
            reward = 0

        self.prev_result = self.curr_result
        self.curr_result = self.V.copy()

        self.last_v = self.V.copy()

        if self.render_mode == "human":
            self.render()

        return (
            self.obs,
            reward,
            self.done,
            {
                "balance (always)": self.balance_always,
                "balance_numerator (always)": self.balance_numerator_always,
                "balance": self.balance,
                "balance_numerator": self.balance_numerator,
                "round": self.V[self.ROUND],
                "done_reason": self.done_info,
                "valid_move": True,
                "invalid_move_info": "",
            },
        )

    def get_initial_v(self):
        return self.init_v.copy()

    def set_v(self, init_v):
        self.init_v = init_v

    def reset(self, options=None, seed=None):
        if options is not None and "v" in options:
            self.V = np.array(options["v"])  # non-default initial values v
        else:
            self.V = self.get_initial_v()

        self.curr_action = np.zeros(self.action_space.shape[0], "int64")
        self.curr_result = self.V.copy()

        self.done = False
        self.done_info = ""

        boxD = gb.get_boxD(self.V[self.QUALITY_OF_LIFE])
        a = float((boxD * 3 + self.V[self.POLITICS]) * 10)
        b = float(self.V[self.ROUND] + 3)
        self.balance_numerator_always = int(a)
        self.balance_always = a / b
        self.balance = 0
        self.balance_numerator = 0

        self.obs = self.V - self.Vmin
        assert self.observation_space.contains(self.obs), "obs not in observation_space"

        return self.obs, {}


if __name__ == "__main__":

    env = OekoEnv(render_mode="ansi")
    # env = OekoPerRoundRewardWrapper(env)
    # env = OekoAuxRewardWrapper(env)
    env = OekoRoundWeightedBalanceRewardWrapper(env)
    env = OekoActionBuilderWrapper(env)
    obs, _ = env.reset()

    # from stable_baselines3.common.env_checker import check_env
    # check_env(env=env)
    # sys.exit(1)

    env.render()

    for _ in range(1):
        env.step_increase_sanitation()
    for _ in range(4):
        env.step_increase_education()
    for _ in range(2):
        env.step_increase_quality_of_life()
    for _ in range(1):
        env.step_increase_production()
    for _ in range(0):
        env.step_decrease_production()
    for _ in range(0):
        env.step_decrease_population_growth_extra()

    _, rew, *_ = env.step_next_round()
    env.render()
    print(f"reward: {rew}")

    for _ in range(2):
        env.step_increase_sanitation()
    for _ in range(5):
        env.step_increase_education()
    for _ in range(2):
        env.step_increase_quality_of_life()
    for _ in range(0):
        env.step_increase_production()
    for _ in range(0):
        env.step_decrease_production()
    for _ in range(0):
        env.step_decrease_population_growth_extra()

    _, rew, *_ = env.step_next_round()
    env.render()
    print(f"reward: {rew}")

    for _ in range(0):
        env.step_increase_sanitation()
    for _ in range(7):
        env.step_increase_education()
    for _ in range(3):
        env.step_increase_quality_of_life()
    for _ in range(0):
        env.step_increase_production()
    for _ in range(0):
        env.step_decrease_production()
    for _ in range(0):
        env.step_decrease_population_growth_extra()

    _, rew, *_ = env.step_next_round()
    env.render()
    print(f"reward: {rew}")

    for _ in range(4):
        env.step_increase_sanitation()
    for _ in range(4):
        env.step_increase_education()
    for _ in range(3):
        env.step_increase_quality_of_life()
    for _ in range(0):
        env.step_increase_production()
    for _ in range(0):
        env.step_decrease_production()
    for _ in range(3):
        env.step_decrease_population_growth_extra()
    env.step_next_round()
    env.render()

    for _ in range(6):
        env.step_increase_sanitation()
    for _ in range(2):
        env.step_increase_education()
    for _ in range(4):
        env.step_increase_quality_of_life()
    for _ in range(0):
        env.step_increase_production()
    for _ in range(2):
        env.step_decrease_production()
    for _ in range(5):
        env.step_decrease_population_growth_extra()

    _, rew, *_ = env.step_next_round()
    env.render()
    print(f"reward: {rew}")

    for _ in range(7):
        env.step_increase_sanitation()
    for _ in range(0):
        env.step_increase_education()
    for _ in range(6):
        env.step_increase_quality_of_life()
    for _ in range(0):
        env.step_increase_production()
    for _ in range(1):
        env.step_decrease_production()
    for _ in range(5):
        env.step_decrease_population_growth_extra()
    _, rew, *_ = env.step_next_round()
    env.render()
    print(f"reward: {rew}")

    for _ in range(6):
        env.step_increase_sanitation()
    for _ in range(0):
        env.step_increase_education()
    for _ in range(0):
        env.step_increase_quality_of_life()
    for _ in range(0):
        env.step_increase_production()
    for _ in range(2):
        env.step_decrease_production()
    for _ in range(1):
        env.step_decrease_population_growth_extra()
    _, rew, *_ = env.step_next_round()
    env.render()
    print(f"reward: {rew}")

    for _ in range(6):
        env.step_increase_sanitation()
    for _ in range(0):
        env.step_increase_education()
    for _ in range(2):
        env.step_increase_quality_of_life()
    for _ in range(0):
        env.step_increase_production()
    for _ in range(4):
        env.step_decrease_production()
    for _ in range(1):
        env.step_decrease_population_growth_extra()

    _, rew, *_ = env.step_next_round()
    env.render()
    print(f"reward: {rew}")

    # env.step(action=np.array([5, 0, 0, 0, 0, 0]))
    # env.render()
    # env.step_increase_production()
    terminal, truncated = True, False
    while not terminal and not truncated:
        mask = np.array(env.valid_action_mask()).astype(np.int8)
        action = env.action_space.sample(mask=mask)
        obs, reward, terminal, truncated, info = env.step(action)

    # env.render()
    # print(reward, info)
