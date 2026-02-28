import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(losses, save_path):
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_matrix_stats(conds, norms, save_path_prefix):
    plt.figure()
    plt.plot(conds, label='Condition Number')
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Condition Number')
    plt.title('A_t Condition Number Over Time')
    plt.legend()
    plt.savefig(f"{save_path_prefix}_cond.png")
    plt.close()

    plt.figure()
    plt.plot(norms, label='S_t Norm')
    plt.xlabel('Steps')
    plt.ylabel('Frobenius Norm')
    plt.title('S_t Norm Over Time')
    plt.legend()
    plt.savefig(f"{save_path_prefix}_norm.png")
    plt.close()

def plot_survival_heatmap(survival_matrix, save_path, title='Memory Survival Heatmap'):
    """
    survival_matrix: (T_write, T_read) numpy array or similar.
    Plots heatmap of which memory i was retrieved at time t.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(survival_matrix, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Survival Value')
    plt.xlabel('Read Time (t)')
    plt.ylabel('Write Time (i)')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def save_numpy_traces(data_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for k, v in data_dict.items():
        np.save(os.path.join(save_dir, f"{k}.npy"), v)
