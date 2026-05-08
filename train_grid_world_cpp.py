"""
Usage examples:
  python train_grid_world_cpp.py train 5 3 200 1000000 --obs-window 5
  python train_grid_world_cpp.py curriculum 10 12 400 1000000 --model-path data/ppo_cpp_5x5.zip --obs-window 5
  python train_grid_world_cpp.py test 5 3 --model-path data/ppo_cpp_5x5.zip --obs-window 5
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
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")


class CPPFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_out_dim: int = 128):
        super().__init__(observation_space, features_dim=1)

        neighbors_shape = observation_space["neighbors"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, neighbors_shape[0], neighbors_shape[1])
            cnn_flattened = self.cnn(dummy).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_flattened, cnn_out_dim),
            nn.ReLU(),
        )
        self.agent_fc = nn.Sequential(
            nn.Linear(observation_space["agent"].shape[0], 32),
            nn.ReLU(),
        )

        self._features_dim = cnn_out_dim + 32

    def forward(self, observations: dict) -> torch.Tensor:
        neighbors = observations["neighbors"].unsqueeze(1)
        cnn_out = self.cnn_fc(self.cnn(neighbors))
        agent_out = self.agent_fc(observations["agent"])
        return torch.cat([cnn_out, agent_out], dim=1)


def register_env() -> None:
    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP-v0",
            entry_point=GridWorldCPPEnv,
        )
    except Exception:
        pass


def build_env(
    dim: int,
    obstacles: int,
    max_steps: int,
    obs_window_size: int,
    render_mode: str,
) -> gym.Env:
    return gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        obs_window_size=obs_window_size,
        render_mode=render_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test CPP agent")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train a new model")
    curriculum_parser = subparsers.add_parser("curriculum", help="Continue training from a checkpoint")
    test_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    run_parser = subparsers.add_parser("run", help="Run a single episode with rendering")

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
        p.add_argument("--obs-window", type=int, default=5, help="Observation window size (3 or 5)")
        p.add_argument("--model-path", type=str, default=None, help="Path to a .zip model")
        p.add_argument("--device", type=str, default="cpu")

    test_parser.add_argument("--metrics-path", type=str, default=None, help="CSV path to save per-episode metrics")

    for p in (train_parser, curriculum_parser):
        p.add_argument("--ent-coef", type=float, default=0.01)
        p.add_argument("--gamma", type=float, default=0.995)
        p.add_argument("--n-steps", type=int, default=2048)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--learning-rate", type=float, default=3e-4)
        p.add_argument("--clip-range", type=float, default=0.2)

    return parser.parse_args()


def ensure_model_path(model_path: str | None) -> str:
    if model_path:
        return model_path
    model_name = input("Enter model filename (e.g., ppo_cpp_5_3_200_0.05_20260324_100000): ")
    return f"data/{model_name}.zip"


def build_policy_kwargs() -> dict:
    return {
        "features_extractor_class": CPPFeaturesExtractor,
        "features_extractor_kwargs": {"cnn_out_dim": 128},
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
    }


def build_log_paths(dim: int, obstacles: int, max_steps: int, ent_coef: float, obs_window: int, suffix: str = "") -> Tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"ppo_cpp_{dim}_{obstacles}_{max_steps}_{ent_coef}_w{obs_window}_{timestamp}{suffix}"
    log_dir = f"log/{base_name}"
    model_path = f"data/{base_name}.zip"
    return log_dir, model_path


def build_metrics_path(dim: int, obstacles: int, obs_window: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"metrics_cpp_{dim}_{obstacles}_w{obs_window}_{timestamp}.csv"
    return os.path.join("results", base_name)


def main() -> None:
    args = parse_args()
    register_env()

    if args.mode == "train":
        print("--- Starting CPP Training ---")
        env = build_env(args.dim, args.obstacles, args.max_steps, args.obs_window, "rgb_array")
        check_env(env)

        policy_kwargs = build_policy_kwargs()
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
            policy_kwargs=policy_kwargs,
            device=args.device,
        )

        log_dir, model_path = build_log_paths(
            args.dim, args.obstacles, args.max_steps, args.ent_coef, args.obs_window
        )
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        print(f"Starting learning with {args.total_timesteps} timesteps...")
        model.learn(total_timesteps=args.total_timesteps)
        model.save(model_path)
        print(f"Model trained and saved to {model_path}")
        print(f"Logs saved to {log_dir}")

    elif args.mode == "curriculum":
        print("--- Starting CPP Curriculum Learning Training ---")
        model_path = ensure_model_path(args.model_path)
        env = build_env(args.dim, args.obstacles, args.max_steps, args.obs_window, "rgb_array")

        model = PPO.load(model_path, env=env, device=args.device)
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False)

        log_dir, new_model_path = build_log_paths(
            args.dim,
            args.obstacles,
            args.max_steps,
            args.ent_coef,
            args.obs_window,
            suffix="_curriculum",
        )
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        print(f"Starting learning with {args.total_timesteps} timesteps...")
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False)
        model.save(new_model_path)
        print(f"Model trained and saved to {new_model_path}")
        print(f"Logs saved to {log_dir}")

    elif args.mode == "run":
        model_path = ensure_model_path(args.model_path)
        print(f"--- Loading model from {model_path} for a run ---")

        model = PPO.load(model_path, device=args.device)
        env = build_env(args.dim, args.obstacles, args.max_steps, args.obs_window, "human")

        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        total_reward = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action.item())
            total_reward += reward
            steps += 1
            print(
                f"Step: {steps}, Action: {print_action(action.item())}, "
                f"Reward: {reward:.2f}, Coverage: {info['coverage']:.1%}, "
                f"Done: {done}, Truncated: {truncated}"
            )
        print(f"--- Run Finished --- Total reward: {total_reward:.2f}, Coverage: {info['coverage']:.1%}")

    elif args.mode == "test":
        model_path = ensure_model_path(args.model_path)
        print(f"--- Loading model from {model_path} for testing ---")

        model = PPO.load(model_path, device=args.device)
        env = build_env(args.dim, args.obstacles, args.max_steps, args.obs_window, "rgb_array")

        num_episodes = 100
        full_coverage_count = 0
        total_coverages = []
        total_steps_list = []

        for i in range(num_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            steps = 0
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action.item())
                steps += 1

            total_coverages.append(info["coverage"])
            total_steps_list.append(steps)

            if done and not truncated:
                full_coverage_count += 1
                print(f"Episode {i+1}: Full coverage in {steps} steps.")
            else:
                print(f"Episode {i+1}: Coverage {info['coverage']:.1%} in {steps} steps.")

        full_coverage_rate = (full_coverage_count / num_episodes) * 100
        avg_coverage = np.mean(total_coverages) * 100
        standard_deviation = np.std(total_coverages) * 100
        avg_steps = np.mean(total_steps_list)
        standard_deviation_steps = np.std(total_steps_list)
        print("\n--- Test Finished ---")
        print(f"Full Coverage Rate: {full_coverage_rate:.2f}% ({full_coverage_count}/{num_episodes})")
        print(
            f"Average Coverage: {avg_coverage:.2f}% Standard Deviation: {standard_deviation:.2f}% "
            f"Min Coverage: {np.min(total_coverages)*100:.2f}% Max Coverage: {np.max(total_coverages)*100:.2f}%"
        )
        print(
            f"Average Steps: {avg_steps:.1f} Standard Deviation: {standard_deviation_steps:.1f} "
            f"Min Steps: {np.min(total_steps_list)} Max Steps: {np.max(total_steps_list)}"
        )

        metrics_path = args.metrics_path or build_metrics_path(args.dim, args.obstacles, args.obs_window)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            metrics_file.write("episode,coverage,steps,full_coverage\n")
            for idx, (coverage, steps) in enumerate(zip(total_coverages, total_steps_list), start=1):
                full = 1 if coverage >= 0.999 else 0
                metrics_file.write(f"{idx},{coverage:.6f},{steps},{full}\n")
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
