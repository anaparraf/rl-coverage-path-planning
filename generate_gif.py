"""utils/generate_gif.py — ADAPTADO PARA SOLUTION B (Robust)"""

import argparse
import os
import importlib
from dataclasses import dataclass
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
# Importação crucial: aponta para o ambiente correto e o extrator de features
from gymnasium_env.grid_world_cpp_final import GridWorldCPPEnv
from train_grid_world_cpp_final import CPPFeaturesExtractor 
from stable_baselines3 import PPO

def register_env() -> None:
    """Registra a versão final do ambiente no Gymnasium."""
    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP-v0",
            entry_point=GridWorldCPPEnv,
        )
    except Exception:
        # Já registrado ou erro de importação
        pass

@dataclass
class EpisodeRollout:
    coverage: float
    steps: int
    frames: List

def _make_env(dim: int, obstacles: int, max_steps: int, obs_window: int):
    """Cria o ambiente com as configurações de observação da Solução B."""
    return gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        obs_window_size=obs_window,
        render_mode="rgb_array",
    )

def rollout_episode(env: gym.Env, model: Optional[PPO], deterministic: bool, max_steps: int, random_policy: bool = False) -> EpisodeRollout:
    frames = []
    obs, info = env.reset()

    # Captura o frame inicial
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    done = False
    truncated = False
    steps = 0
    
    while not (done or truncated) and steps < max_steps:
        if random_policy:
            action = env.action_space.sample()
        else:
            # model.predict lida com o MultiInputPolicy (Dict observation) automaticamente
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    return EpisodeRollout(
        coverage=float(info.get("coverage", 0.0)), 
        steps=steps, 
        frames=frames
    )

def select_rollouts(rollouts: List[EpisodeRollout]) -> tuple[EpisodeRollout, EpisodeRollout, EpisodeRollout]:
    sorted_r = sorted(rollouts, key=lambda r: r.coverage)
    low = sorted_r[0]
    high = sorted_r[-1]
    # Mediana
    median_cov = sorted(r.coverage for r in rollouts)[len(rollouts) // 2]
    avg = min(rollouts, key=lambda r: abs(r.coverage - median_cov))
    return high, avg, low

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GIF from CPP Solution B policy")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--obstacles", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--obs-window", type=int, default=5)
    parser.add_argument("--out-path", type=str, default="results/cpp_run.gif")
    parser.add_argument("--out-prefix", type=str, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--save-summary-3", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    register_env()
    
    try:
        imageio = importlib.import_module("imageio.v2")
    except ImportError:
        raise ImportError("Instale imageio: pip install imageio")

    os.makedirs(os.path.dirname(args.out_path) or "results", exist_ok=True)

    # Carregamento do modelo com os objetos customizados necessários
    # O PPO precisa saber o que é o CPPFeaturesExtractor para reconstruir a rede
    custom_objects = {
        "features_extractor_class": CPPFeaturesExtractor
    }
    
    print(f"Carregando modelo de {args.model_path}...")
    model = PPO.load(args.model_path, device=args.device, custom_objects=custom_objects)

    if not args.save_summary_3:
        env = _make_env(args.dim, args.obstacles, args.max_steps, args.obs_window)
        rollout = rollout_episode(env, model, args.deterministic, args.max_steps)
        imageio.mimsave(args.out_path, rollout.frames, fps=args.fps)
        print(f"✓ GIF salvo em {args.out_path} | Cobertura: {rollout.coverage:.1%}")
        return

    # Modo Summary (High, Avg, Low)
    out_prefix = args.out_prefix or os.path.splitext(args.out_path)[0]
    env = _make_env(args.dim, args.obstacles, args.max_steps, args.obs_window)
    
    rollouts = []
    for ep in range(args.episodes):
        r = rollout_episode(env, model, args.deterministic, args.max_steps)
        rollouts.append(r)
        print(f"[Modelo] Ep {ep+1}/{args.episodes}: {r.coverage:.1%}")

    high, avg, low = select_rollouts(rollouts)
    
    for label, res in [("high", high), ("avg", avg), ("low", low)]:
        path = f"{out_prefix}_{label}.gif"
        imageio.mimsave(path, res.frames, fps=args.fps)
        print(f"✓ Salvo: {path} ({res.coverage:.1%})")

    # Baseline Aleatório
    rand_rollout = rollout_episode(env, None, False, args.max_steps, random_policy=True)
    rand_path = f"{out_prefix}_random.gif"
    imageio.mimsave(rand_path, rand_rollout.frames, fps=args.fps)
    print(f"✓ Salvo baseline aleatório: {rand_path} ({rand_rollout.coverage:.1%})")

if __name__ == "__main__":
    main()