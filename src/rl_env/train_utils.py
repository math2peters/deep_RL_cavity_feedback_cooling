from stable_baselines3 import SAC
import numpy as np
from src.rl_env.environment import CavityCoolingEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import time
from torch.optim import AdamW
import sys
import select
import torch.nn as nn
import gymnasium
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy, Actor
from torch.distributions import Normal
from typing import Optional, Tuple


# Custom linear actor for SAC policy
class LinearActor(Actor):
    """
    Linear Actor network for SAC.
    
    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Features extractor to use
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_actions,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not, dividing by 255.0 (True by default)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add these two lines to define the min and max bounds for log_std
        self.log_std_min = -20
        self.log_std_max = 2
        
        # Replace the mlp_extractor with a linear layer directly to action mean and log_std
        input_dim = self.features_dim
        action_dim = self.action_space.shape[0]
        
        # Create a simple linear layer that outputs both mean and log_std
        # Parameters: 2N+2 = 2*input_dim (weights for mean and log_std) + 2 (biases for mean and log_std)
        self.mean_and_log_std = nn.Linear(input_dim, 2 * action_dim)
        
        # Initialize the weights and bias
        nn.init.zeros_(self.mean_and_log_std.weight)
        nn.init.zeros_(self.mean_and_log_std.bias)
        
        # Set the log_std parameters to the specified initialization value
        self.mean_and_log_std.bias.data[action_dim:] = torch.ones(action_dim) * self.log_std_init
        
        # We don't need the latent_pi network anymore
        del self.latent_pi
        del self.mu

    def get_action_dist_params(self, obs):
        """
        Get the parameters of the action distribution (mean and std)
        
        :param obs: Observation
        :return: Mean and log standard deviation
        """
        features = self.extract_features(obs, self.features_extractor)
        
        # Get mean and log_std directly from the linear layer
        mean_log_std = self.mean_and_log_std(features)
        action_dim = self.action_space.shape[0]
        
        mean_actions = mean_log_std[:, :action_dim]
        log_std = mean_log_std[:, action_dim:]
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean_actions, log_std, {}
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass in the actor network.
        
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: actions
        """
        mean_actions, log_std, _ = self.get_action_dist_params(obs)
        
        if deterministic:
            # When deterministic, use the mean as the action
            return torch.tanh(mean_actions)
        
        # Otherwise, sample from a normal distribution
        noise = torch.randn_like(mean_actions)
        actions = mean_actions + noise * torch.exp(log_std)
        
        # Apply tanh squashing to ensure actions are within [-1, 1]
        return torch.tanh(actions)

    def _predict(self, observation, deterministic=False):
        """
        Get the action according to the policy for a given observation.
        
        :param observation: the input observation
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action
        """
        with torch.no_grad():
            actions = self.forward(observation, deterministic)
            return actions


# Custom Linear SAC Policy class
class LinearSACPolicy(SACPolicy):
    """
    Policy class with a linear actor and critic for SAC.
    
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: Network architecture
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance above zero and
        prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not, dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic(s)
    """
    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return LinearActor(**actor_kwargs).to(self.device)
    
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation.
        
        :param observation: the input observation
        :param state: The last hidden states (not used for non-recurrent policies)
        :param episode_start: The last masks (not used for non-recurrent policies)
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: the model's action and the next state (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)
        
        # Convert to PyTorch tensor
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        
        with torch.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        
        if isinstance(self.action_space, gymnasium.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]
            
        return actions, state

class InterruptCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.readline().strip()
            if key == 'q':
                return False
        return True
    
# Define the custom callback to plot rewards during training and render episodes
class PlottingCallback(BaseCallback):
    def __init__(self, eval_env, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.all_rewards = []
        self.average_rewards = []
        self.all_total_trapped_steps = []
        self.average_trapped_steps = []
        self.all_final_temperature = []
        self.average_final_temperature= []
        self.all_trapped_end = []
        self.average_trapped_end = []
        self.eval_env = eval_env  # Evaluation environment for rendering
        

        # Initialize the plot
        plt.ion()
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_xlabel('Every 100 Episodes')
        self.ax.set_ylabel('Average Reward')
        
        # plot the average steps trapped
        self.figure2, self.ax2 = plt.subplots()
        self.line2, = self.ax2.plot([], [], 'r-')
        self.ax2.set_xlabel('Every 100 Episodes')
        self.ax2.set_ylabel('Average Steps Trapped')
        
        self.figure3, self.ax3 = plt.subplots()
        self.line3, = self.ax3.plot([], [], 'g-')
        self.ax3.set_xlabel('Every 100 Episodes')
        self.ax3.set_ylabel('End Temp')
        
        # plot the trapped end fraction
        self.figure4, self.ax4 = plt.subplots()
        self.line4, = self.ax4.plot([], [], 'y-')
        self.ax4.set_xlabel('Every 100 Episodes')
        self.ax4.set_ylabel('Trapped End Fraction')
        
    def _on_step(self) -> bool:
        # Check if new episodes have finished
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                # Append the episode reward
                self.all_rewards.append(info['episode']['r'])
                self.all_total_trapped_steps.append(info['trapped_steps'])
                self.all_final_temperature.append(info['final_temperature'])
                self.all_trapped_end.append(info['trapped_end'])
                
                num_episodes = len(self.all_rewards)
                # Every 100 episodes, compute average and update plot
                if num_episodes % 100 == 0:
                    # Compute average of the last 100 episodes
                    average_reward = sum(self.all_rewards[-100:]) / 100
                    self.average_rewards.append(average_reward)
                    average_trapped_steps = sum(self.all_total_trapped_steps[-100:]) / 100
                    self.average_trapped_steps.append(average_trapped_steps)
                    average_final_temperature = sum(self.all_final_temperature[-100:]) / 100
                    self.average_final_temperature.append(average_final_temperature)
                    average_trapped_end = sum(self.all_trapped_end[-100:]) / 100
                    self.average_trapped_end.append(average_trapped_end)

                    # Update the plot
                    self.line.set_xdata(range(len(self.average_rewards)))
                    self.line.set_ydata(self.average_rewards)
                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.figure.canvas.draw()
                    self.figure.canvas.flush_events()
                    
                    # update average trapped steps
                    self.line2.set_xdata(range(len(self.average_trapped_steps)))
                    self.line2.set_ydata(self.average_trapped_steps)
                    self.ax2.relim()
                    self.ax2.autoscale_view()
                    self.figure2.canvas.draw()
                    self.figure2.canvas.flush_events()
                    
                    # update average early photons
                    self.line3.set_xdata(range(len(self.average_final_temperature)))
                    self.line3.set_ydata(np.array(self.average_final_temperature)*1e6)
                    self.ax3.relim()
                    self.ax3.autoscale_view()
                    self.figure3.canvas.draw()
                    self.figure3.canvas.flush_events()
                    
                    # update all trapped end fraction
                    self.line4.set_xdata(range(len(self.average_trapped_end)))
                    self.line4.set_ydata(self.average_trapped_end)
                    self.ax4.relim()
                    self.ax4.autoscale_view()
                    self.figure4.canvas.draw()
                    self.figure4.canvas.flush_events()

        return True

class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, plot=True):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.trapped_steps = []
        self.trapped_end = []
        self.final_temp = []
        self.trapped_end_fraction = 0
        self.trapped_end_frame = 0
        self.final_temp_avg = 0
        self.plot = plot

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'trapped_steps' in info:
                self.trapped_steps.append(info['trapped_steps'])
            if 'trapped_end' in info:
                self.trapped_end.append(info['trapped_end'])
            if 'final_temperature' in info:
                self.final_temp.append(info['final_temperature'])

        return True

    def _on_rollout_end(self):
        # Aggregate metrics over the rollout
        self.mean_trapped_steps = np.mean(self.trapped_steps) if self.trapped_steps else 0
        self.end_trapped_fraction = np.mean(self.trapped_end) if self.trapped_end else 0
        self.final_temp_avg = np.mean(self.final_temp) if self.final_temp else 0

        # Log the metrics
        self.logger.record('rollout/trapped_steps_mean', self.mean_trapped_steps)
        self.logger.record('rollout/trapped_end_fraction', self.end_trapped_fraction)
        self.logger.record('rollout/final_temperature_mean', self.final_temp_avg)

        # Clear the lists for the next rollout
        self.trapped_steps = []
        self.trapped_end = []
        self.final_temp = []

    def _on_training_end(self):
        # Keep the plot open at the end of training
        if self.plot:   
            plt.ioff()
            plt.show()


def make_env(env_id, architecture, seed, render_mode=None, reward_scale=1.0, 
             reward_component_scale=[1, 1, 1], gym_env=False, full_observations=True, 
             frame_stacks=1, save_name=None):
    def _init():
        if gym_env:
            env = gymnasium.make("LunarLanderContinuous-v2")
        else:
            env = CavityCoolingEnv(render_mode=render_mode, architecture=architecture, seed=seed, 
                                   full_observations=full_observations, frame_stack_number=frame_stacks, 
                                   reward_scale=reward_scale, reward_component_scale=reward_component_scale)
            env.seed(seed + env_id)
        monitor_save_name = save_name if save_name is not None else str(env_id)  # Use provided save_name or default to env_id
        env = Monitor(env, filename=f"logs/mon{monitor_save_name}_{env_id}", info_keywords=("trapped_steps", "trapped_end"))
        return env
    return _init


def exponential_schedule(initial_value, final_value):
   
    decay_rate = np.log(final_value / initial_value)
    def func(progress_remaining):
        lr = initial_value * np.exp(decay_rate * (1 - progress_remaining))
        return lr
    return func 