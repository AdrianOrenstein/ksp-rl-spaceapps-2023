"""This file contains a basic agent class for linear function approximation."""

import math
import numpy as np
from utils.helpers import get_weights_from_npy
from utils.tilecoder import TileCoder


class LFAControlAgent:
    """
    A generic class that is re-used in the implementation of all sorts of
    control algorithms with LFA.
    In most cases, only agent_step needs to be implemented in the child classes.
    """

    def __init__(self, **agent_args):
        """
        Initializes the agent-class parameters.

        Args:
            **agent_args: a dictionary of agent parameters
            num_actions: number of discrete actions
            num_features: size of the observation vector
            rng_seed: seed for the random-number generator
            alpha: the step size for updating the weights
            eta: the step-size parameter for updating the reward-rate estimate
            alpha_sequence: sequence of step size: 'exponential' or '1byN'
            alpha_decay_rate: decay rate if sequence is exponential(ly decaying)
            value_init: optional initial values
            avg_reward_init: optional initial value of the reward-rate estimate
            weights: the vector of weights of size self.num_features
            avg_reward: the scalar reward-rate estimate
            pi: the target policy
            b: the behavior policy
        """
        # input the function approximator
        self.approximation_type = agent_args.get('approximation_type', 'linear')
        assert self.approximation_type in ['tabular', 'linear']

        # take the environment parameters as input
        assert 'num_actions' in agent_args
        self.num_actions = agent_args['num_actions']
        assert 'num_features' in agent_args
        self.num_features = agent_args['num_features']

        # initialize the random-number generator
        self.rng_seed = agent_args.get('rng_seed', 22)
        self.rng = np.random.default_rng(seed=self.rng_seed)

        # initialize the step size and optional step-size sequence
        self.alpha_init = agent_args.get('alpha', 0.001)
        self.alpha_sequence = agent_args.get('alpha_sequence', 'exponential')
        self.robust_to_initialization = agent_args.get('robust_to_initialization', False)
        # if self.robust_to_initialization:
        #     self.alpha_sequence = 'unbiased_trick'
        if self.alpha_sequence == 'exponential':
            self.alpha_decay_rate = agent_args.get('alpha_decay_rate', 1.0)

        # initialize the tile coder, if any
        self.tilecoder = agent_args.get("tilecoder", False)
        if self.tilecoder:
            self.num_tilings = agent_args.get("num_tilings", 8)
            assert "tiling_dims" in agent_args
            tiling_dims = agent_args["tiling_dims"]
            assert "limits_per_dim" in agent_args
            limits_per_dim = agent_args["limits_per_dim"]

            self.tilecoder = TileCoder(tiling_dims=tiling_dims,
                                       limits_per_dim=limits_per_dim,
                                       num_tilings=self.num_tilings,
                                       style='indices')

            self.num_features = self.tilecoder.n_tiles
            self.alpha_init /= self.num_tilings

        if self.approximation_type == 'linear':
            self.bias = True
            self.num_features += 1
        else:
            self.bias = False

        self.weights = np.zeros(self.num_features * self.num_actions)
        self.avg_reward = None

        # if the weights are supplied, they should be an ndarray in an npy file
        # as a dictionary element with key 'weights'
        if 'weights_file' in agent_args:
            weights = get_weights_from_npy(filename=agent_args['weights_file'], 
                                           seed_idx=agent_args['load_seed'])
            assert weights.size == self.num_features * self.num_actions
            self.weights = weights

        # initialize the e-greedy exploration parameters
        self.epsilon_start = agent_args.get('epsilon_start', 0.9)
        self.epsilon_end = agent_args.get('epsilon_end', 0.05)
        self.epsilon_decay_param = agent_args.get('epsilon_decay_param', 200000)
        self.epsilon = self.epsilon_start

        self.actions = list(range(self.num_actions))
        self.past_action = None
        self.past_obs = None
        self.timestep = 0
        self.max_value_per_step = None

    def _process_raw_observation(self, obs):
        """
        Processes raw observation into a encoding, e.g., a tile-coded encoding.
        """
        if self.tilecoder:
            observation = self.tilecoder.getitem(obs)
            # tilecoder observations are indices, so add the index of the bias
            observation = np.concatenate((observation, [self.num_features - 1]))
        else:
            observation = obs
            if self.bias:
                observation = np.concatenate((observation, [1]))

        return observation

    def _get_representation(self, observation, action):
        """Returns the agent state.

        This simple implementation of the agent-state-update function returns
        a one-hot version of the observation vector.
        For instance, ([1,2], 1) returns [0,0,1,2,0,0] for self.num_actions=3.

        Args:
            observation: the current observation
            action: the action index
        Returns:
            state: the agent-state vector
        """
        if self.tilecoder:
            offset = self.num_features * action
            state = np.array(observation + offset, dtype=int)
        else:
            state = np.zeros(self.num_features * self.num_actions)
            offset_start = self.num_features * action
            offset_end = self.num_features * (action + 1)
            state[offset_start:offset_end] = observation

        return state

    def _get_linear_value_approximation(self, representation):
        """Returns the linear estimation of the value for a given representation

        Args:
            representation: the representation vector
        Returns:
            value: w^T x
        """
        if self.tilecoder:      # assumes 'indices' style for tilecoder
            value = np.sum(self.weights[representation])
        else:
            value = np.dot(representation, self.weights)
        return value

    def _get_value(self, observation, action):
        """Returns the value corresponding to an observation and action.

        Args:
            observation: the observation vector
            action: the action index
        Returns:
            value: an approximation of the expected value
        """
        rep = self._get_representation(observation, action)
        return self._get_linear_value_approximation(rep)

    def _set_weights(self, given_weight_vector):
        """Sets the agent's weights to the given weight vector."""
        self.weights = given_weight_vector

    def _argmax(self, values):
        """Returns the argmax of a list. Breaks ties uniformly randomly."""
        self.max_value_per_step = np.max(values)
        return self.rng.choice(np.argwhere(values == self.max_value_per_step).flatten())

    def _choose_action_egreedy(self, state):
        """
        Returns an action using an epsilon-greedy policy w.r.t. the
        current action-value function.

        Args:
            state: the current agent state
        Returns:
            action: an action-index integer
        """
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
            self.max_value_per_step = None
        else:
            q_s = np.array([self._get_value(state, a) for a in self.actions])
            action = self._argmax(q_s)

        return action

    def _max_action_value(self, state):
        """
        Returns the action index corresponding to the maximum action value
        for a given agent-state vector. If the maximum action value is
        shared by more than one action, one of them is randomly chosen.
        """
        q_s = np.array([self._get_value(state, a) for a in self.actions])
        argmax_action = self._argmax(q_s)

        return q_s[argmax_action]

    def start(self, observation):
        """
        The first method called when the experiment starts,
        called after the environment starts.

        Args:
            observation: the first observation returned by the environment
        Returns:
            action: an action-index integer
        """
        obs = self._process_raw_observation(observation)
        action = self._choose_action_egreedy(obs)

        self.past_obs = obs
        self.past_action = action
        self.timestep += 1
        self._initialize_step_size()

        return action

    def step(self, reward, observation):
        """
        Returns a new action corresponding to the new observation
        and updates the agent parameters.

        Args:
            reward: the reward at the current time step
            observation: the new observation vector
        Returns:
            action: an action-index integer
        """
        obs = self._process_raw_observation(observation)
        self._update_weights(reward, obs)
        action = self._choose_action_egreedy(obs)

        self.timestep += 1
        self._update_epsilon()
        self._update_step_size()

        self.past_obs = obs
        self.past_action = action

        return action

    def _initialize_step_size(self):
        """Initializes the step size."""
        if self.alpha_sequence == 'unbiased_trick':
            self.o = self.alpha_init
            self.alpha = self.alpha_init / self.o
        elif self.alpha_sequence == '1byN':
            self.alpha = 1
        else:
            self.alpha = self.alpha_init

    def _update_step_size(self):
        """Updates the step size per step."""
        if self.alpha_sequence == 'exponential':
            self.alpha *= self.alpha_decay_rate
        elif self.alpha_sequence == '1byN':
            self.alpha = 1 / self.timestep
        elif self.alpha_sequence == 'unbiased_trick':
            self.o = self.o + self.alpha_init * (1 - self.o)
            self.alpha = self.alpha_init / self.o

    def _update_epsilon(self):
        """Decays the epsilon parameter per step."""
        self.epsilon = (self.epsilon_end +
                        (self.epsilon_start - self.epsilon_end) * math.e **
                        (-self.timestep / self.epsilon_decay_param))

    def _update_weights(self, reward, obs):
        raise NotImplementedError


class DifferentialDiscountedQlearningAgent(LFAControlAgent):
    """
    Implements Naik, Wan, Sutton's (2023) one-step 
    Centered Discounted Q-learning algorithm (CDiscQ).
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)

        # setting eta and gamma, the two main parameters of the algorithm
        self.eta = agent_args.get('eta', 0.1)
        self.gamma = agent_args.get('gamma', 0.95)

        # initializing the reward-rate estimate
        self.avg_reward_init = agent_args.get('avg_reward_init', 0)
        self.avg_reward = self.avg_reward_init

        # initializing the step size for the reward-rate estimate
        self.beta_sequence = agent_args.get('beta_sequence', 'exponential')
        assert self.beta_sequence in ['unbiased_trick', 'exponential']
        if self.robust_to_initialization:
            self.beta_sequence = 'unbiased_trick'
        self.beta_init = self.eta * self.alpha_init
        if self.tilecoder:
            self.beta_init *= self.num_tilings

        self.sarsa_update = agent_args.get("sarsa_update", False)

    def _update_weights(self, reward, obs):
        past_sa = self._get_representation(self.past_obs, self.past_action)
        prediction = self._get_linear_value_approximation(past_sa)
        next_action = None
        if self.sarsa_update:
            next_action = self._choose_action_egreedy(obs)
            q_next = self._get_value(obs, next_action)
        else:
            q_next = self._max_action_value(obs)
        target = reward - self.avg_reward + self.gamma * q_next
        delta = target - prediction
        avg_rew_delta = (target - prediction) if not self.sarsa_update else (reward - self.avg_reward)
        if self.robust_to_initialization:
            old_avg_reward = self.avg_reward
            self.avg_reward += self.beta * avg_rew_delta
            updated_target = target + (old_avg_reward - self.avg_reward)
            updated_delta = updated_target - prediction
        else:
            self.avg_reward += self.beta * avg_rew_delta
            updated_delta = delta

        if self.tilecoder:
            self.weights[past_sa] += self.alpha * updated_delta
        else:
            self.weights += self.alpha * updated_delta * past_sa

    def _initialize_step_size(self):
        """Initializes both the step sizes."""
        super()._initialize_step_size()
        if self.beta_sequence == 'unbiased_trick':
            self.o_b = self.beta_init if self.beta_init != 0 else 1
            self.beta = self.beta_init / self.o_b
        else:
            self.beta = self.beta_init

    def _update_step_size(self):
        """Updates both the step sizes per step."""
        super()._update_step_size()
        if self.beta_sequence == 'exponential':
            self.beta *= self.alpha_decay_rate
        elif self.beta_sequence == 'unbiased_trick':
            self.o_b = self.o_b + self.beta_init * (1 - self.o_b)
            self.beta = self.beta_init / self.o_b
