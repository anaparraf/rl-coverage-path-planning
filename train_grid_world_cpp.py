"""
APS08 — Coverage Path Planning with Partial Observation (SOLUTION B)

Usage examples:
  # Train on 5x5
  python train_grid_world_cpp_final.py train 5 3 200 1000000 --obs-window 5
  
  # Train on 10x10 (from scratch)
  python train_grid_world_cpp_final.py train 10 12 400 2000000 --obs-window 5
  
  # Train on 10x10 (curriculum from 5x5)
  python train_grid_world_cpp_final.py curriculum 10 12 400 2000000 \
    --model-path data/ppo_cpp_5x5.zip --obs-window 5
  
  # Test a model
  python train_grid_world_cpp_final.py test 10 12 --model-path data/model.zip --obs-window 5 --max-steps 400
  
  # Run one episode with visualization
  python train_grid_world_cpp_final.py run 10 12 --model-path data/model.zip --obs-window 5 --max-steps 400
"""

import argparse
import os
from datetime import datetime
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def print_action(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")


class CPPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dual-CNN Feature Extractor for CPP POMDP.
    
    Processes three independent streams:
    
    1. LOCAL CNN (neighbors window W×W):
       - Captures local obstacle patterns and visited cell layout
       - Input: 5×5 or 3×3 window
       - Output: 128 features
    
    2. GLOBAL CNN (visited_map S×S):
       - Processes global visited footprint to identify unexplored regions
       - Crucial for POMDP: tells agent where gaps in coverage are
       - Input: size×size binary map (5×5 or 10×10)
       - Output: 128 features
    
    3. AGENT MLP (scalar state):
       - Processes [x_norm, y_norm, coverage_ratio]
       - Output: 32 features
    
    All concatenated: 128 + 128 + 32 = 288 features → PPO policy/value heads
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_out_dim: int = 128):
        super().__init__(observation_space, features_dim=1)

        neighbors_shape = observation_space["neighbors"].shape
        visited_shape = observation_space["visited_map"].shape

        # --- LOCAL CNN (window)
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, neighbors_shape[0], neighbors_shape[1])
            local_flat_dim = self.local_cnn(dummy).shape[1]

        self.local_fc = nn.Sequential(
            nn.Linear(local_flat_dim, cnn_out_dim),
            nn.ReLU(),
        )

        # --- GLOBAL CNN (visited_map)
        # For larger grids, may need different architecture
        # For 5×5 and 10×10, this standard CNN works
        self.global_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) if visited_shape[0] >= 8 else nn.Identity(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, visited_shape[0], visited_shape[1])
            global_flat_dim = self.global_cnn(dummy).shape[1]

        self.global_fc = nn.Sequential(
            nn.Linear(global_flat_dim, cnn_out_dim),
            nn.ReLU(),
        )

        # --- AGENT MLP (position + coverage)
        self.agent_fc = nn.Sequential(
            nn.Linear(observation_space["agent"].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Total features: local + global + agent
        self._features_dim = cnn_out_dim + cnn_out_dim + 32

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass combining all three streams.
        
        Args:
            observations: dict with keys "neighbors", "visited_map", "agent"
        
        Returns:
            Concatenated feature vector of shape (batch_size, 288)
        """
        neighbors = observations["neighbors"].unsqueeze(1)  # (B, 1, W, W)
        visited = observations["visited_map"].unsqueeze(1)  # (B, 1, S, S)

        local_features = self.local_fc(self.local_cnn(neighbors))
        global_features = self.global_fc(self.global_cnn(visited))
        agent_features = self.agent_fc(observations["agent"])

        return torch.cat([local_features, global_features, agent_features], dim=1)


def register_env() -> None:
    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP-v0",
            entry_point=GridWorldCPPEnv,
        )
    except Exception:
        pass


def build_env(dim, obstacles, max_steps, obs_window_size, render_mode):
    return gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        obs_window_size=obs_window_size,
        render_mode=render_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test CPP agent (Solution B)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    curriculum_parser = subparsers.add_parser("curriculum")
    test_parser = subparsers.add_parser("test")
    run_parser = subparsers.add_parser("run")

    for p in (train_parser, curriculum_parser):
        p.add_argument("dim", type=int)
        p.add_argument("obstacles", type=int)
        p.add_argument("max_steps", type=int)
        p.add_argument("total_timesteps", type=int)

    for p in (test_parser, run_parser):
        p.add_argument("dim", type=int)
        p.add_argument("obstacles", type=int)
        p.add_argument("--max-steps", type=int, default=200)

    for p in (train_parser, curriculum_parser, test_parser, run_parser):
        p.add_argument("--obs-window", type=int, default=5)
        p.add_argument("--model-path", type=str, default=None)
        p.add_argument("--device", type=str, default="cpu")

    test_parser.add_argument("--metrics-path", type=str, default=None)

    for p in (train_parser, curriculum_parser):
        p.add_argument("--ent-coef", type=float, default=0.01)
        p.add_argument("--gamma", type=float, default=0.99)
        p.add_argument("--n-steps", type=int, default=2048)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--learning-rate", type=float, default=3e-4)
        p.add_argument("--clip-range", type=float, default=0.2)

    return parser.parse_args()


def ensure_model_path(model_path):
    if model_path:
        return model_path
    model_name = input("Enter model filename (without .zip): ")
    return f"data/{model_name}.zip"


def build_policy_kwargs():
    return {
        "features_extractor_class": CPPFeaturesExtractor,
        "features_extractor_kwargs": {"cnn_out_dim": 128},
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    }


def build_log_paths(dim, obstacles, max_steps, ent_coef, obs_window, suffix=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"ppo_cpp_{dim}_{obstacles}_{max_steps}_{ent_coef}_w{obs_window}_{timestamp}{suffix}"
    return f"log/{base}", f"data/{base}.zip"


def build_metrics_path(dim, obstacles, obs_window):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("results", f"metrics_cpp_{dim}_{obstacles}_w{obs_window}_{timestamp}.csv")


def main():
    args = parse_args()
    register_env()

    if args.mode == "train":
        print(f"--- Training CPP on {args.dim}×{args.dim} (Solution B) ---")
        env = build_env(args.dim, args.obstacles, args.max_steps, args.obs_window, "rgb_array")
        check_env(env)

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            policy_kwargs=build_policy_kwargs(),
            device=args.device,
        )

        log_dir, model_path = build_log_paths(
            args.dim, args.obstacles, args.max_steps, args.ent_coef, args.obs_window
        )
        os.makedirs(log_dir, exist_ok=True)
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        print(f"Starting learning with {args.total_timesteps} timesteps...")
        model.learn(total_timesteps=args.total_timesteps)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Logs saved to {log_dir}")

    elif args.mode == "curriculum":
        print(f"--- Curriculum Learning: {args.dim}×{args.dim} (Solution B) ---")
        model_path = ensure_model_path(args.model_path)
        env = build_env(args.dim, args.obstacles, args.max_steps, args.obs_window, "rgb_array")

        print(f"Loading pre-trained model from {model_path}...")
        model = PPO.load(model_path, env=env, device=args.device)

        log_dir, new_model_path = build_log_paths(
            args.dim, args.obstacles, args.max_steps, args.ent_coef,
            args.obs_window, "_curriculum"
        )
        os.makedirs(log_dir, exist_ok=True)
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        print(f"Continuing learning with {args.total_timesteps} timesteps...")
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False)

        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        model.save(new_model_path)
        print(f"✓ Model saved to {new_model_path}")
        print(f"✓ Logs saved to {log_dir}")

    elif args.mode == "run":
        model_path = ensure_model_path(args.model_path)
        print(f"--- Running one episode with {model_path} ---")

        model = PPO.load(model_path, device=args.device)
        env = build_env(args.dim, args.obstacles, args.max_steps, args.obs_window, "human")

        obs, info = env.reset()
        done = truncated = False
        steps = total_reward = 0
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action.item())
            total_reward += reward
            steps += 1
            print(
                f"Step {steps:3d} | Action: {print_action(action.item()):5s} | "
                f"Reward: {reward:6.2f} | Coverage: {info['coverage']:5.1%}"
            )
        
        print(f"\n--- Episode finished ---")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final coverage: {info['coverage']:.1%}")
        print(f"Total steps: {steps}")

    elif args.mode == "test":
        model_path = ensure_model_path(args.model_path)
        print(f"--- Testing {args.dim}×{args.dim} with {model_path} ---")

        model = PPO.load(model_path, device=args.device)
        env = build_env(args.dim, args.obstacles, args.max_steps, args.obs_window, "rgb_array")

        num_episodes = 100
        full_coverage_count = 0
        total_coverages = []
        total_steps_list = []

        for i in range(num_episodes):
            obs, info = env.reset()
            done = truncated = False
            steps = 0
            
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action.item())
                steps += 1

            total_coverages.append(info["coverage"])
            total_steps_list.append(steps)

            if done and not truncated:
                full_coverage_count += 1
                print(f"Episode {i+1:3d}: ✓ Full coverage in {steps:3d} steps")
            else:
                print(f"Episode {i+1:3d}: {info['coverage']:5.1%} coverage in {steps:3d} steps")

        # --- Statistics ---
        full_coverage_rate = (full_coverage_count / num_episodes) * 100
        avg_coverage = np.mean(total_coverages) * 100
        std_coverage = np.std(total_coverages) * 100
        avg_steps = np.mean(total_steps_list)
        std_steps = np.std(total_steps_list)

        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Full Coverage Rate:  {full_coverage_rate:6.2f}% ({full_coverage_count}/{num_episodes})")
        print(f"Avg Coverage:        {avg_coverage:6.2f}% ± {std_coverage:5.2f}%")
        print(f"  Min Coverage:      {np.min(total_coverages)*100:6.2f}%")
        print(f"  Max Coverage:      {np.max(total_coverages)*100:6.2f}%")
        print(f"Avg Steps:           {avg_steps:6.1f} ± {std_steps:5.1f}")
        print(f"  Min Steps:         {np.min(total_steps_list):6d}")
        print(f"  Max Steps:         {np.max(total_steps_list):6d}")
        print("="*70)

        # --- Save metrics
        metrics_path = args.metrics_path or build_metrics_path(args.dim, args.obstacles, args.obs_window)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("episode,coverage,steps,full_coverage\n")
            for idx, (cov, stp) in enumerate(zip(total_coverages, total_steps_list), 1):
                full = 1 if cov >= 0.999 else 0
                f.write(f"{idx},{cov:.6f},{stp},{full}\n")
        print(f"\n✓ Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
