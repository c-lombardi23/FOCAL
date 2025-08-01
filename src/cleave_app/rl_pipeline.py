"""Custom Gym environment that simulates cleaving fibers using a surrogate CNN model.

The agent adjusts tension over multiple steps to achieve optimal cleave quality,
which is evaluated via a CNN surrogate model. Observations include fiber context and
tension; rewards are based on CNN predictions and a guassian reward function when predicted
tensions is close to optimal value.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env


class CleaveEnv(gym.Env):
    """Creates the simulated cleave enviornment."""

    # use human readable mode
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: str,
        cnn_path: str,
        img_folder: str,
        feature_shape: List[int],
        threshold: float,
        max_steps: int,
        low_range: float,
        high_range: float,
        max_delta: float,
        max_tension_change: float,
        quality_weight=100.0,
        proximity_weight=50.0,
        scale=25.0,
    ) -> None:
        """
        Initialize the environment.

        Args:
            csv_path (str): Path to the CSV file with cleave data.
            cnn_path (str): Path to the trained CNN model used as a surrogate.
            img_folder (str): Directory containing cleave images.
            feature_shape (List): Shape of the numercal features
            threshold: (float): Classification threshold for good cleave.
            max_steps (int): Maximum number of steps to use per episode.
            low_range (float): Low percentage of maximum tension.
            high_range (float): High percentage of maximum tension.
            max_delta (float): Maximum change in tension per action
            max_tension_change (float): Absolute maximum tension change.
        """
        # call gym init method
        super().__init__()

        self.cnn_model = joblib.load(cnn_path)
        self.img_folder = img_folder
        self.df = pd.read_csv(csv_path)

        self.feature_shape = feature_shape
        self.threshold = threshold
        self.low_range = low_range
        self.high_range = high_range
        self.max_delta = max_delta
        self.max_tension_change = max_tension_change

        self.QUALITY_WEIGHT = quality_weight
        self.PROXIMITY_WEIGHT = proximity_weight
        self.SCALE = scale

        filtered_df = self.df[self.df["CNN_Predicition"] == 1]

        # calculate ideal tensions by fiber type
        self.ideal_tensions = dict(
            filtered_df.groupby("FiberType")["CleaveTension"]
            .mean()
            .astype(np.float32)
        )

        len_fibers = len(self.df["FiberType"].unique())

        # one hot encode fiber names
        self.df = pd.get_dummies(
            self.df, columns=["FiberType"], dtype=np.int32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.max_tension_change = max_tension_change

        fiber_types = self.df.iloc[:, -len_fibers:]

        self.model_features = self.cnn_model.feature_names_in_

        other_inputs = self.df["Diameter"]

        # combine fiber names and diameter
        combined_df = pd.concat([other_inputs, fiber_types], axis=1)

        self.context_df = combined_df

        # + 3 for current tension, current reward, and last tension
        observations_total = 3 + len(self.context_df.columns)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observations_total,),
            dtype=np.float32,
        )

        self.max_steps = max_steps
        self.current_step = 0
        self.current_context = None
        self.current_tension = 0
        self.render_mode = None

    # For use if you decide to include full CNN in enviornment
    def load_process_images(self, filename: str) -> "tf.Tensor":
        """Load and preprocess image from file path.

        Args:
            filename: Image filename or path

        Returns:
            tf.Tensor: Preprocessed image tensor
        """

        if tf is None:
            raise ImportError("TensorFlow is required for image processing")

        def load_image(file):
            """Load an image and process using same preprocessing as backbone.

            Args:
                file: path to image
                preprocess_input: processing from backbone model

            Returns:
                loaded and resized image
            """
            full_path = os.path.join(self.img_folder, file)

            try:
                img_raw = tf.io.read_file(full_path)
            except FileNotFoundError:
                print(f"Image file not found: {full_path}")
                return None
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                return None

            try:
                img = tf.image.decode_png(img_raw, channels=1)
                img = tf.image.resize(img, [224, 224])
                img = tf.image.grayscale_to_rgb(img)
                return img
            except Exception as e:
                print(f"Error processing image {full_path}: {e}")
                return None

        img = load_image(filename)
        img.set_shape([224, 224, 3])
        return img

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[dict]): Additional options for reset (unused).

        Returns:
            Tuple[np.ndarray, dict]: Initial observation and empty info dict.
        """

        super().reset(seed=seed)

        self.last_reward = 0.0

        self.current_context = self.context_df.sample(
            n=1, random_state=self.np_random
        )
        self.current_ideal_tension = self.ideal_tensions[
            self._get_current_fiber_type()
        ]
        self.current_tension = self.np_random.uniform(
            low=self.current_ideal_tension * (self.low_range),
            high=self.current_ideal_tension * (self.high_range),
        )
        self.current_step = 0

        observation = self._create_observation()

        if self.render_mode == "human":
            print("\n---------------EPISODE RESET----------------------")
            print(
                f"New Scenario: Fiber = {self._get_current_fiber_type()} Start Tension = {self.current_tension:.0f}"
            )
        fiber_type = self._get_current_fiber_type()

        # Log info
        info = {
            "fiber_type": fiber_type,
            "start_tension": self.current_tension,
        }
        return observation, info

    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment using the given action.

        Args:
            action (gym.ActType): A 1D array-like action representing tension adjustment.

        Returns:
            Tuple:
                - observation (np.ndarray): The next observation.
                - reward (float): The reward received after taking the action.
                - terminated (bool): True if the episode ends successfully.
                - truncated (bool): True if the episode is truncated (max steps reached).
                - info (dict): Additional info (empty by default).
        """
        delta_tension = float(action[0] * self.max_tension_change)
        self.current_tension = self.current_tension + delta_tension

        # Compute min and max tensions
        min_tension = self.current_ideal_tension * self.low_range
        max_tension = self.current_ideal_tension * self.high_range

        # Clip tensions if outside range
        self.current_tension = np.clip(
            self.current_tension, min_tension, max_tension
        )

        self.current_ideal_tension = self.ideal_tensions[
            self._get_current_fiber_type()
        ]

        # Increment steps in episode
        self.current_step += 1

        model_inputs = self.current_context.copy()
        model_inputs["CleaveTension"] = self.current_tension
        model_inputs = model_inputs[self.model_features]

        tension_error = self.current_tension - self.current_ideal_tension

        # Get CNN surrogate prediction in range (0, 1)
        cnn_pred = self.cnn_model.predict_proba(model_inputs)[0, 1]

        reward = 0.0
        terminated = False

        reward += self.QUALITY_WEIGHT * cnn_pred

        if cnn_pred >= self.threshold:
            reward += 50.0
            terminated = True

        scale = self.SCALE
        # Gaussian reward for proximity to current ideal tension
        proximity_reward = self.PROXIMITY_WEIGHT * np.exp(
            -(tension_error**2) / (2 * scale**2)
        )
        reward += proximity_reward

        # Decrease reward if close to max or min tension
        if np.isclose(self.current_tension, min_tension) or np.isclose(
            self.current_tension, max_tension
        ):
            reward -= 75.0

        # Decrease reward if tension change is opposite to direction of ideal tension
        if (tension_error > 0) and (action[0] > 0):
            reward -= 25.0
        elif (tension_error < 0) and (action[0] < 0):
            reward -= 25.0
        else:
            reward += 5.0

        reward -= 1.0

        reward -= (abs(action[0]) ** 2) * 2.0

        truncated = self.current_step >= self.max_steps
        if truncated and not terminated:
            reward -= self.PROXIMITY_WEIGHT * (1.0 - cnn_pred)

        if self.render_mode == "human":
            self.render(action, cnn_pred, reward)
        observation = self._create_observation()

        # Log info
        info = {
            "cnn_pred": float(cnn_pred),
            "current_tension": round(float(self.current_tension), 3),
            "current_ideal_tension": round(
                float(self.current_ideal_tension), 3
            ),
            "tension_error": round(float(tension_error), 3),
            "action": round(float(action) * self.max_tension_change, 3),
        }
        self.last_reward = reward
        return observation, float(reward), terminated, truncated, info

    def _get_current_fiber_type(self) -> str:
        """Get the name of the current fiber type from the one-hot encoded context.

        Returns:
            str: The name of the current fiber type, or 'Unknown' if not found.
        """
        for col_name in self.current_context.columns:
            if (
                "FiberType_" in col_name
                and self.current_context[col_name].iloc[0] == 1.0
            ):
                return col_name.replace("FiberType_", "")
        return "Unknown"

    def _create_observation(self) -> np.ndarray:
        """Create a numeric observation vector from the current state.

        Returns:
            np.ndarray: Concatenation of current tension and context values.
        """
        tension_error = self.current_ideal_tension - self.current_tension
        return np.concatenate(
            [
                [self.current_tension],
                [tension_error],
                self.current_context.values[0],
                np.array([self.last_reward]),
            ]
        ).astype(np.float32)

    def render(
        self, action: np.ndarray, cnn_pred: float, reward: float
    ) -> None:
        """Render the environment's current state in human-readable format.

        Args:
            action (np.ndarray): The action taken (as a 1D array).
            cnn_pred (float): The CNN's predicted cleave quality.
            reward (float): The reward received after the action.
        """
        action_str = f"{(action[0] *self.max_tension_change):+.2f}"
        cnn_str = "GOOD" if cnn_pred > self.threshold else "BAD"
        print(
            f"Step {self.current_step:2d} Tension: {self.current_tension:6.1f} (Action: {action_str:6s}) -> CNN: {cnn_str:4s}| Reward: {reward:6.1f}"
        )


class TrainAgent:
    """Class for training the RL agent"""

    def __init__(
        self,
        csv_path: str,
        cnn_path: str,
        img_folder: str,
        threshold: float,
        feature_shape: List[int],
        max_steps: int,
        low_range: float,
        high_range: float,
        max_delta: float,
        max_tension_change: float,
    ) -> None:
        """
        Initialize the training environment for the RL agent.

        Args:
            csv_path (str): Path to the CSV file with cleave data.
            cnn_path (str): Path to the trained CNN model used as a surrogate.
            img_folder (str): Directory containing cleave images.
            feature_shape (List): Shape of the numercal features
            threshold: (float): Classification threshold for good cleave.
            max_steps (int): Maximum number of steps to use per episode.
            low_range (float): Low percentage of maximum tension.
            high_range (float): High percentage of maximum tension.
            max_delta (float): Maximum change in tension per action
            max_tension_change (float): Absolute maximum tension change.
        """

        # Initialize Enviornment
        self.env = CleaveEnv(
            csv_path=csv_path,
            cnn_path=cnn_path,
            img_folder=img_folder,
            feature_shape=feature_shape,
            threshold=threshold,
            max_steps=max_steps,
            low_range=low_range,
            high_range=high_range,
            max_delta=max_delta,
            max_tension_change=max_tension_change,
        )
        check_env(self.env)

    def train(
        self,
        env: gym.Env,
        device: str,
        buffer_size: int,
        learning_rate: float,
        batch_size: int,
        tau: float,
        timesteps: int,
    ) -> None:
        """Train the agent using Soft Actor Critic algo.

        Args:
            env (gym.Env): simulated training enviornment
            device (str): cuda to use GPU
            buffer_size (int): replay buffer size
            learning_rate (float): typical learning rate for ml
            batch_size (int): number of episodes to batch together
            tau (float): Soft update coefficient
        """

        self.agent = SAC(
            "MlpPolicy",
            env=self.env,
            device=device,
            verbose=0,
            buffer_size=buffer_size,
            ent_coef=0.3,  # Coefficient for maximum entropy
            learning_rate=learning_rate,
            batch_size=batch_size,
            tau=tau,
        )
        self.agent.learn(total_timesteps=timesteps, progress_bar=True)

    def save_agent(self, save_path: str) -> None:
        self.agent.save(save_path)


class TestAgent:

    def __init__(
        self,
        csv_path: str,
        cnn_path: str,
        img_folder: str,
        agent_path: str,
        feature_shape: List[int],
        threshold: float,
        max_steps: int,
        low_range: float,
        high_range: float,
        max_delta: float,
        max_tension_change: float,
    ) -> None:
        """
        Initialize the environment and load a trained agent.

        Args:
            csv_path (str): Path to the CSV file with cleave data.
            cnn_path (str): Path to the trained CNN model used as a surrogate.
            img_folder (str): Directory containing cleave images.
            feature_shape (List): Shape of the numercal features
            threshold: (float): Classification threshold for good cleave.
            max_steps (int): Maximum number of steps to use per episode.
            low_range (float): Low percentage of maximum tension.
            high_range (float): High percentage of maximum tension.
            max_delta (float): Maximum change in tension per action
            max_tension_change (float): Absolute maximum tension change.
        """

        self.env = CleaveEnv(
            csv_path=csv_path,
            cnn_path=cnn_path,
            img_folder=img_folder,
            threshold=threshold,
            feature_shape=feature_shape,
            max_steps=max_steps,
            low_range=low_range,
            high_range=high_range,
            max_delta=max_delta,
            max_tension_change=max_tension_change,
        )
        self.env.render_mode = "human"
        self.agent = SAC.load(agent_path)

    def test_agent(self, episodes: int) -> Dict:
        """Test the trained RL agent on random episodes.

        Args:
            episodes (int): total number of episodes to test agent
        """
        all_episode_info = []

        for episode in range(episodes):
            obs, info_reset = self.env.reset()
            done = False
            episode_reward = 0

            episode_info = []
            rewards = []
            observations = []
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = self.env.step(
                    action
                )
                episode_info.append(info)
                rewards.append(round(reward, 3))
                observations.append(obs)

                episode_reward += reward
                done = terminated or truncated

            print(
                f"Episode {episode + 1} finished with a total reward of: {episode_reward:.2f}"
            )
            # Log metrics
            metrics = {
                "start tension": info_reset["start_tension"],
                "fiber type": info_reset["fiber_type"],
                "episode info": episode_info,
                "rewards": rewards,
                "episode reward": round(episode_reward, 3),
            }
            all_episode_info.append(metrics)

        self.env.close()
        return all_episode_info
