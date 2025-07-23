"""Custom Gym environment that simulates cleaving fibers using a surrogate CNN model.

The agent adjusts tension over multiple steps to achieve optimal cleave quality,
which is evaluated via a pre-trained CNN. Observations include fiber context and
tension; rewards are based on CNN predictions and tension dynamics.
"""
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from typing import Dict, Any, Tuple, Optional, List


class CleaveEnv(gym.Env):
    """Creates the simulated cleave enviornment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 csv_path: str, 
                 cnn_path: str,
                 img_folder: str,
                 feature_shape: List[int],
                 threshold: float) -> None:
        
        """
        Initialize the environment.

        Args:
            csv_path (str): Path to the CSV file with cleave data.
            cnn_path (str): Path to the trained CNN model used as a surrogate.
            img_folder (str): Directory containing cleave images.
        """

        super().__init__()

        self.cnn_model = tf.keras.models.load_model(cnn_path)
        self.img_folder = img_folder
        self.df = pd.read_csv(csv_path)

        self.feature_shape = feature_shape
        self.threshold = threshold

        filtered_df = self.df[self.df["CNN_Predicition"] == 1]
        self.ideal_tensions = dict(
            filtered_df.groupby("FiberType")["CleaveTension"]
            .mean()
            .astype(np.float32)
        )

        len_fibers = len(self.df["FiberType"].unique())

        self.df = pd.get_dummies(
            self.df, columns=["FiberType"], dtype=np.int32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.max_tension_change = 10.0

        fiber_types = self.df.iloc[:, -len_fibers:]
        other_inputs = self.df["Diameter"]

        combined_df = pd.concat([other_inputs, fiber_types], axis=1)

        self.context_df = combined_df
        observations_total = 1 + len(self.context_df.columns)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observations_total,),
            dtype=np.float32,
        )

        self.max_steps = 15
        self.current_step = 0
        self.current_context = None
        self.current_tension = 0
        self.render_mode = None

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

    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        """
        Reset the environment to an initial state.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[dict]): Additional options for reset (unused).

        Returns:
            Tuple[np.ndarray, dict]: Initial observation and empty info dict.
        """

        super().reset(seed=seed)

        self.current_context = self.context_df.sample(
            n=1, random_state=self.np_random
        )
        self.current_ideal_tension = self.ideal_tensions[
            self._get_current_fiber_type()
        ]
        self.current_tension = self.np_random.uniform(
            low=self.current_ideal_tension * (0.8),
            high=self.current_ideal_tension * (1.2),
        )
        self.current_step = 0

        observation = self._create_observation()

        if self.render_mode == "human":
            print("\n---------------EPISODE RESET----------------------")
            print(
                f"New Scenario: Fiber = {self._get_current_fiber_type()} Start Tension = {self.current_tension:.0f}"
            )

        return observation, {}

    def step(self, 
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
        self.current_tension = np.clip(self.current_tension, 50, 2000)
        self.current_ideal_tension = self.ideal_tensions[
            self._get_current_fiber_type()
        ]

        self.current_step = self.current_step + 1

        model_inputs = self.current_context.copy()
        model_inputs["CleaveTension"] = self.current_tension

        row_index = self.current_context.index[0]
        image_filename = self.df.iloc[row_index]["ImagePath"]
        image_tensor = self.load_process_images(image_filename)

        image_tensor = tf.expand_dims(image_tensor, axis=0)
        # feature_shape = (1, 5)
        dummy_features = np.zeros(self.feature_shape)
        cnn_raw = self.cnn_model.predict(
            [image_tensor, dummy_features], verbose=0
        )
        cnn_pred = cnn_raw[0][0]

        terminated = False
        if cnn_pred >= self.threshold:
            reward = 100.0
            terminated = True
        else:
            reward = 50.0 * cnn_pred - 3.0 * (1 - cnn_pred)

        SAFE_DELTA_THRESHOLD = 5.0

        if abs(delta_tension) <= SAFE_DELTA_THRESHOLD:
            reward += 1.5
        else:
            reward -= 0.25 * (abs(delta_tension) - SAFE_DELTA_THRESHOLD)

        tension_error = abs(self.current_tension - self.current_ideal_tension)
        reward += (
            max(0, 1 - (tension_error / self.current_ideal_tension)) * 20.0
        )

        action_cost = 0.1 * abs(delta_tension)
        reward = reward - action_cost

        truncated = self.current_step >= self.max_steps
        if truncated and not terminated:
            reward = reward - 25.0

        if self.render_mode == "human":
            self.render(action, cnn_pred, reward)
        observation = self._create_observation()
        return observation, float(reward), terminated, truncated, {}

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
        return np.concatenate(
            [[self.current_tension], self.current_context.values[0]]
        ).astype(np.float32)

    def render(self, 
               action: np.ndarray, 
               cnn_pred: float, 
               reward: float) -> None:
        """Render the environment's current state in human-readable format.

        Args:
            action (np.ndarray): The action taken (as a 1D array).
            cnn_pred (float): The CNN's predicted cleave quality.
            reward (float): The reward received after the action.
        """
        action_str = f"{(action[0] *10.0):+.2f}"
        cnn_str = "GOOD" if cnn_pred > self.threshold else "BAD"
        print(
            f"Step {self.current_step:2d} Tension: {self.current_tension:6.1f} (Action: {action_str:6s}) -> CNN: {cnn_str:4s}| Reward: {reward:6.1f}"
        )


class TrainAgent:
    """Class for training the RL agent"""

    def __init__(self, csv_path: str, 
                 cnn_path: str, 
                 img_folder: str,
                 threshold: float,
                 feature_shape: List[int]) -> None:
        """
        Initialize the training environment for the RL agent.

        Args:
            csv_path (str): Path to the CSV file with cleave data.
            cnn_path (str): Path to the trained CNN model used as a surrogate.
            img_folder (str): Directory containing the cleave images.
        """

        # Initialize Enviornment
        self.env = CleaveEnv(
            csv_path=csv_path, cnn_path=cnn_path, img_folder=img_folder,
            feature_shape=feature_shape, threshold=threshold
        )
        check_env(self.env)

    def train(self, 
              env: gym.Env,
              device: str,
              buffer_size: int, 
              learning_rate: float, 
              batch_size: int, 
              tau: float,
              timesteps: int) -> None:
        """Train the agent using Soft Actor Critic algo.

        Args:
            env (gym.Env): simulated training enviornment
            device (str): cuda to use GPU
            buffer_size (int): replay buffer size
            learning_rate (float): typicall lr for ml
            batch_size (int): number of episodes to batch together
            tau (float): 
        """

        self.agent = SAC(
            "MlpPolicy",
            env=self.env,
            device=device,
            verbose=0,
            buffer_size=buffer_size,
            ent_coef="auto",
            learning_rate=learning_rate,
            batch_size=batch_size,
            tau=tau,
        )
        self.agent.learn(total_timesteps=timesteps, progress_bar=True)

    def save_agent(self, save_path: str) -> None:
        self.agent.save(save_path)


class TestAgent:

    def __init__(self, 
                 csv_path: str, 
                 cnn_path: str, 
                 img_folder: str, 
                 agent_path: str,
                 feature_shape: List[int],
                 threshold: float) -> None:
        
        """
        Initialize the environment and load a trained agent.

        Args:
            csv_path (str): Path to the CSV file with cleave data.
            cnn_path (str): Path to the trained CNN model used as a surrogate.
            img_folder (str): Directory containing cleave images.
            agent_path (str): File path of the saved agent to load.
        """

        self.env = CleaveEnv(
            csv_path=csv_path, cnn_path=cnn_path, 
            img_folder=img_folder,
            threshold=threshold,
            feature_shape=feature_shape
        )
        self.env.render_mode = "human"
        self.agent = SAC.load(agent_path)

    def test_agent(self, episodes: int) -> None:
        """Test the trained RL agent on random episodes.

        Args:
            episodes (int): total number of episodes to test agent
        """

        for episode in range(episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = self.env.step(
                    action
                )

                episode_reward += reward
                done = terminated or truncated

            print(
                f"Episode {episode + 1} finished with a total reward of: {episode_reward:.2f}"
            )

        self.env.close()
