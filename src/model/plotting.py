import matplotlib.pyplot as plt
import numpy as np
import re

def plot_entropy(file_path='/home/dan/RLPoker/logs/entropy.txt'):
    with open(file_path, 'r') as f:
        entropy_values = [float(line.strip()) for line in f if line.strip()]
    update_steps = np.arange(1, len(entropy_values) + 1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.plot(update_steps, entropy_values, color='#ff7f0e', linewidth=1.5)

    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_xlabel('Update Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy Value', fontsize=12, fontweight='bold')
    ax.set_title('Entropy per Update Cycle', fontsize=14, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    return fig, ax

def plot_actor_loss(file_path='/home/dan/RLPoker/logs/actor_loss.txt'):

    with open(file_path, 'r') as f:
        loss_values = [float(line.strip()) for line in f if line.strip()]
    update_steps = np.arange(1, len(loss_values) + 1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.plot(update_steps, loss_values, color='#1f77b4', linewidth=1.5)

    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_xlabel('Update Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax.set_title('Actor Loss per Update Cycle', fontsize=14, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    return fig, ax


def plot_critic_loss(file_path='/home/dan/RLPoker/logs/critic_loss.txt'):
    with open(file_path, 'r') as f:
        loss_values = [float(line.strip()) for line in f if line.strip()]

    update_steps = np.arange(1, len(loss_values) + 1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.plot(update_steps, loss_values, color='#2ca02c', linewidth=1.5)

    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_xlabel('Update Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax.set_title('Critic Loss per Update Cycle', fontsize=14, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    return fig, ax


def plot_action_distribution(file_path='/home/dan/RLPoker/logs/actions.txt'):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    numbers = []
    for line in lines:
        match = re.search(r'tensor\(\[(\d+)\]', line)
        if match:
            numbers.append(int(match.group(1)))

    fold_count = numbers.count(0)
    raise_count = numbers.count(1)
    call_count = numbers.count(2)

    total_relevant = fold_count + raise_count + call_count

    if total_relevant > 0:
        fold_percent = (fold_count / total_relevant) * 100
        raise_percent = (raise_count / total_relevant) * 100
        call_percent = (call_count / total_relevant) * 100
    else:
        fold_percent = raise_percent = call_percent = 0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    actions = ['Fold', 'Raise', 'Call']
    percentages = [fold_percent, raise_percent, call_percent]

    colors = ['#d62728', '#1f77b4', '#2ca02c']

    bars = ax.bar(actions, percentages, color=colors, width=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold')

    ax.set_ylim(0, max(percentages) * 1.15)  # Add some space for labels
    ax.set_ylabel('Percentage of Total Actions (%)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Poker Actions', fontsize=14, fontweight='bold')

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax.text(0.02, 0.95, f'Total Fold: {fold_count}', transform=ax.transAxes, fontsize=10)
    ax.text(0.02, 0.90, f'Total Raise: {raise_count}', transform=ax.transAxes, fontsize=10)
    ax.text(0.02, 0.85, f'Total Call: {call_count}', transform=ax.transAxes, fontsize=10)
    ax.text(0.02, 0.80, f'Total Actions: {total_relevant}', transform=ax.transAxes, fontsize=10, fontweight='bold')

    plt.tight_layout()

    return fig, ax
if __name__ == "__main__":
    fig_critic, ax_critic = plot_critic_loss()
    # plt.savefig('critic_loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot actor loss
    fig_actor, ax_actor = plot_actor_loss()
    # plt.savefig('actor_loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot entropy
    fig_entropy, ax_entropy = plot_entropy()
    # plt.savefig('entropy_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plot_action_distribution()
    # plt.savefig('action_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()