import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_training_metrics(log_csv: str, out_dir: str):
    # Carrega os dados do progresso do treinamento
    df = pd.read_csv(log_csv)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Learning Curve (Reward Mean)
    plt.figure(figsize=(10, 5))
    plt.plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], 
             color='#2a6fdb', linewidth=2, label='Ep Reward Mean')
    plt.xlabel('Total Timesteps')
    plt.ylabel('Average Reward')
    plt.title('Learning Curve: Training Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'learning_curve.png'))
    plt.close()

    # 2. Entropy Loss (Métrica de Exploração)
    # Interessante para ver se o agente ainda está explorando ou se a política convergiu.
    if 'train/entropy_loss' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['time/total_timesteps'], df['train/entropy_loss'], 
                 color='#1b9e77', linewidth=2, label='Entropy Loss')
        plt.xlabel('Total Timesteps')
        plt.ylabel('Entropy Loss')
        plt.title('Policy Exploration (Entropy Loss)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'entropy_loss.png'))
        plt.close()

    # 3. Value Loss (Erro de Estimativa de Valor)
    if 'train/value_loss' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['time/total_timesteps'], df['train/value_loss'], 
                 color='#d95f02', linewidth=1, label='Value Loss')
        plt.yscale('log') # Escala logarítmica para melhor visualização de perdas variadas
        plt.xlabel('Total Timesteps')
        plt.ylabel('Value Loss (Log Scale)')
        plt.title('Critic Performance: Value Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'value_loss.png'))
        plt.close()

    print(f"Sucesso! Gráficos salvos em: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CPP training metrics from progress.csv")
    parser.add_argument("--log-csv", type=str, default="progress.csv", help="Caminho para o progress.csv")
    parser.add_argument("--out-dir", type=str, default="results/plots", help="Pasta de saída")
    args = parser.parse_args()

    plot_training_metrics(args.log_csv, args.out_dir)