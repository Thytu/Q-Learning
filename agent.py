import logging
import numpy as np

from typing import Optional, Tuple, List

class Agent():
    """
    Q-Learning's agent abstracting the qtable
    """

    def __init__(self, obs_low: Tuple[float, ...], obs_high: Tuple[float, ...], action_space_n: int, goal_position: float, learning_rate=0.1, discount=0.95, epsilone=0, epsilone_decay=0.99, min_epsilone=0):
        logging.basicConfig(filename='agent.log', level=logging.INFO)
        self.__logger = logging.getLogger(__name__)

        discrete_observation_size: List[int] = [20] * len(obs_high)
        self._discrete_observation_window_size: Tuple[float, ...] = (obs_high - obs_low) / discrete_observation_size

        self._OBS_LOW = obs_low
        self._OBS_HIGH = obs_high
        self._GOAL_POSITION = goal_position

        self._q_table = np.random.uniform(low=2, high=0, size=(discrete_observation_size + [action_space_n])) # TODO: type hint
        self._epsilone = epsilone
        self._EPS_DECAY = epsilone_decay
        self._EPS_MIN = min_epsilone
        self._DISCOUNT = discount
        self._ACTION_SPACE_N = action_space_n
        self._LEARNING_RATE = learning_rate

    def get_discrete_state(self, state: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert continuous state to discrete state"""

        discrete_state: List[float] = (state - self._OBS_LOW) / self._discrete_observation_window_size

        return tuple(discrete_state.astype(np.int))

    def train(self, state: Tuple[float, ...], new_state: Tuple[float, ...], action: int, reward: float, done: bool, episode: Optional[int] = None, log=False) -> None:
        """Update th q-table accoirding to : state, new_state and reward"""

        discrete_state: Tuple[int, ...] = self.get_discrete_state(state)
        new_discrete_state: Tuple[int, ...] = self.get_discrete_state(new_state)

        if not done:
            max_future_q: float = np.max(self._q_table[new_discrete_state])
            current_q: float = self._q_table[discrete_state + (action, )]

            new_q: float = (1 - self._LEARNING_RATE) * current_q + self._LEARNING_RATE * (reward + self._DISCOUNT * max_future_q)
            self._q_table[discrete_state + (action, )] = new_q

        elif new_state[0] > self._GOAL_POSITION:
            self._q_table[discrete_state + (action, )] = 0
            if log:
                self.__logger.info(f"Reached the goal at episode {episode}")

    def select_action(self, state: Tuple[float, ...]) -> int:
        """Return the best possible action according the q-table"""

        discrete_state: Tuple[int, ...] = self.get_discrete_state(state)

        if np.random.random() > self._epsilone:
            return np.argmax(self._q_table[discrete_state])

        return np.random.randint(0, self._ACTION_SPACE_N)

    def update_epsilone(self, log=False) -> None:
        """Update epsilone according to it's decay and it's min value"""

        self._epsilone = max(self._epsilone * self._EPS_DECAY, self._EPS_MIN)
        if log:
            self.__logger.debug(f"Epsilone: {self._epsilone}")

    def display_epsilone(self) -> None:
        print("Epsilone: ", self._epsilone)
