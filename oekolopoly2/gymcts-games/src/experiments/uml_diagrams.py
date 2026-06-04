import random
from abc import abstractmethod
from typing import Any, TypeVar

from gymcts.gymcts_neural_agent import GymctsNeuralAgent
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, Env
from manta.components.uml_utils import UmlUtils
from manta.slide_templates.minimal.minimal_slide_template import MinimalSlideTemplate

import numpy as np

TGymctsNode = TypeVar("TGymctsNode", bound="GymctsNode")



class Env():
    # Set these in ALL subclasses
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    def step(
        self, action: ActType
    ) -> tuple:
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: int = None,
        options: dict= None,
    ) -> tuple:  # type: ignore
        pass

    def render(self):
        raise NotImplementedError

    def close(self) -> None:
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise an error.
        """
        pass



class GymctsABC():

    @abstractmethod
    def get_state(self) -> Any:
        """
        Returns the current state of the environment. The state can be any datatype in principle, that allows to restore
        the environment to the same state. The state is used to restore the environment unsing the load_state method.

        It's recommended to use a numpy array if possible, as it is easy to serialize and deserialize.

        :return: the current state of the environment
        """
        pass

    @abstractmethod
    def load_state(self, state: Any) -> None:
        """
        Loads the state of the environment. The state can be any datatype in principle, that allows to restore the
        environment to the same state. The state is used to restore the environment unsing the load_state method.

        :param state: the state to load
        :return: None
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Returns True if the environment is in a terminal state, False otherwise.
        :return:
        """
        pass

    @abstractmethod
    def get_valid_actions(self) -> list[int]:
        """
        Returns a list of valid actions for the current state of the environment.
        This used to obtain potential actions/subsequent sates for the MCTS tree.
        :return: the list of valid actions
        """
        pass

    @abstractmethod
    def action_masks(self) -> np.ndarray | None:
        """
        Returns a numpy array of action masks for the environment. The array should have the same length as the number
        of actions in the action space. If an action is valid, the corresponding mask value should be 1, otherwise 0.
        If no action mask is available, it should return None.

        :return: a numpy array of action masks or None
        """
        pass

    @abstractmethod
    def rollout(self) -> float:
        """
        Performs a rollout from the current state of the environment and returns the return (sum of rewards) of the rollout.

        Please make sure the return value is in the interval [-1, 1].
        Otherwise, the MCTS algorithm will not work as expected (due to a male-fitted exploration coefficient;
        exploration and exploitation are not well-balanced then).

        :return: the return of the rollout
        """
        pass


class DeepCopyMCTSGymEnvWrapper(GymctsABC):
    def __init__(self):
        pass

class ActionHistoryMCTSGymEnvWrapper(GymctsABC):
    def __init__(self):
        pass




class GymctsNode:
    # static properties
    best_action_weight: float = 0.05 # weight for the best action
    ubc_c = 0.707 # exploration coefficient
    score_variate: str = "UCT_v0"

    visit_count: int = 0 # number of times the node has been visited
    mean_value: float = 0 # mean value of the node
    max_value: float = -float("inf") # maximum value of the node
    min_value: float = +float("inf") # minimum value of the node
    terminal: bool = False # whether the node is terminal or not
    state: Any = None # state of the node

    def get_root(self) -> TGymctsNode:
        if self.is_root():
            return self
        return self.parent.get_root()


    def reset(self) -> None:
        self.parent = None
        self.visit_count: int = 0

        self.mean_value: float = 0
        self.max_value: float = -float("inf")
        self.min_value: float = +float("inf")
        self.children: dict[int, GymctsNode] | None = None  # may be expanded later

        # just setting the children of the parent node to None should be enough to trigger garbage collection
        # however, we also set the parent to None to make sure that the parent is not referenced anymore
        if self.parent:
            self.parent.reset()

        if self.parent is not None:
            self.parent.remove_parent()

    def is_root(self) -> bool:
        """
        Returns true if the node is a root node. A root node is a node that has no parent.

        :return: true if the node is a root node, false otherwise.
        """
        return self.parent is None

    def is_leaf(self) -> bool:
        """
        Returns true if the node is a leaf node. A leaf node is a node that has no children. A leaf node is a node that has no children.

        :return: true if the node is a leaf node, false otherwise.
        """
        return self.children is None or len(self.children) == 0

    def get_random_child(self) -> TGymctsNode:
        """
        Returns a random child of the node. A random child is a child that is selected randomly from the list of children.
        :return:
        """
        if self.is_leaf():
            raise ValueError("cannot get random child of leaf node")  # todo: maybe return self instead?

        return list(self.children.values())[random.randint(0, len(self.children) - 1)]

    def get_best_action(self) -> int:
        """
        Returns the best action of the node. The best action is the action that has the highest score.
        The score is calculated using the get_score() method. The best action is the action that has the highest score.
        The best action is the action that has the highest score.

        :return: the best action of the node.
        """
        return max(self.children.values(), key=lambda child: child.get_score()).action

    def get_score(self) -> float:  # todo: make it an attribute?
        """
        Returns the score of the node. The score is calculated using the mean value and the maximum value of the node.
        The score is calculated using the formula: score = (1 - a) * mean_value + a * max_value
        where a is the best action weight.

        :return: the score of the node.
        """
        # return self.mean_value
        assert 0 <= GymctsNode.best_action_weight <= 1
        a = GymctsNode.best_action_weight
        return (1 - a) * self.mean_value + a * self.max_value

    def get_mean_value(self) -> float:
        return self.mean_value

    def get_max_value(self) -> float:
        return self.max_value

    def tree_policy_score(self) -> float:
        raise ValueError(f"unknown score variate: {GymctsNode.score_variate}. ")



class GymctsNeuralNode():
    _selection_score_prior: float

    def tree_policy_score(self) -> float:
        pass


class GymctsAgent:
    env: GymctsABC
    search_root_node: GymctsNode  # NOTE: this is not the same as the root of the tree!

    def __init__(self,
                 ):
        pass

    def navigate_to_leaf(self, from_node: GymctsNode) -> GymctsNode:
        pass

    def expand_node(self, node: GymctsNode) -> None:
        pass

    def solve(self, num_simulations_per_step: int = None, render_tree_after_step: bool = None) -> list[int]:
        pass

    def _load_state(self, node: GymctsNode) -> None:
        pass

    def perform_mcts_step(self, search_start_node: GymctsNode, num_simulations: int,
                          render_tree_after_step: bool)-> tuple[int, GymctsNode]:
        pass

    def vanilla_mcts_search(self, search_start_node: GymctsNode, num_simulations:int) -> int:
        pass

    def backpropagation(self, node: GymctsNode, episode_return: float) -> None:
       pass

import sb3_contrib

class GymctsNeuralAgent():

    _model: sb3_contrib.MaskablePPO

    def learn(self, total_timesteps: int, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass

    def expand_node(self, node: GymctsNeuralNode) -> None:
        pass

class MaskablePPO():

    def predict(self, total_timesteps: int, **kwargs) -> np.ndarray:
        pass

    def learn(self, total_timesteps:int) -> None:
        pass


class MyUmlClassDiagramScene(UmlUtils, MinimalSlideTemplate):
    def construct(self):

        gym_class_uml_diagram = self.uml_class_diagram(Env, class_name="gymnasium.Env").scale(0.5)

        self.uml_class_diagram(GymctsABC, class_name="GymctsABC").scale(0.5)
        self.uml_class_diagram(DeepCopyMCTSGymEnvWrapper, class_name="DeepCopyMCTSGymEnvWrapper").scale(0.5)
        self.uml_class_diagram(ActionHistoryMCTSGymEnvWrapper, class_name="ActionHistoryMCTSGymEnvWrapper").scale(0.5)

        self.uml_class_diagram(GymctsNode, class_name="GymctsNode").scale(0.5)
        self.uml_class_diagram(GymctsNeuralNode, class_name="GymctsNeuralNode").scale(0.5)

        self.uml_class_diagram(GymctsAgent, class_name="GymctsAgent").scale(0.5)
        self.uml_class_diagram(GymctsNeuralAgent, class_name="GymctsNeuralAgent").scale(0.5)

        self.uml_class_diagram(MaskablePPO, class_name="MaskablePPO").scale(0.5)


if __name__ == '__main__':
    MyUmlClassDiagramScene().construct()
