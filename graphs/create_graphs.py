import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Configuration and Styling
# Apply seaborn darkgrid style to match the image_da60e6.png background
sns.set_theme(style="darkgrid")
# Use a serif font to match the academic paper style
plt.rcParams["font.family"] = "serif" 

def load_and_smooth(csv_path, window_size=50):
    """
    Loads TensorBoard CSV data and calculates rolling mean and standard deviation.
    TensorBoard CSVs typically have columns: ['Wall time', 'Step', 'Value']
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please ensure the file exists.")
        return None, None, None

    # Sort by step just in case
    df = df.sort_values(by='Step')
    
    # Apply rolling window for smoothing (adjust window_size based on your data density)
    rolling_stats = df['Value'].rolling(window=window_size, min_periods=1)
    
    steps = df['Step'].values
    mean = rolling_stats.mean().values
    std = rolling_stats.std().values
    
    # Fill NaNs in std with 0 for the first few steps
    std = np.nan_to_num(std)
    
    return steps, mean, std

def plot_recreated_graph():
    # 2. Define file paths and labels
    # Update these filenames if yours are different
    datasets = {
        'PPO': {'file': 'PPO_Swimmer-v5_1_2026-07-25_22-26-17.csv', 'color': 'blue'},
        'Mb-Mf': {'file': 'MBMF_Swimmer-v5_1_2026-07-26_11-03-32.csv', 'color': 'red'},
        'Mb': {'file': 'MBMF_Swimmer-v5_1_2026-07-26_11-03-32_Phase1_MB.csv', 'color': 'green'},
    }
    title :str = "Environment: Swimmer"

    datasets = {
        'PPO': {'file': 'PPO_Hopper-v5_1_2026-07-25_22-37-30.csv', 'color': 'blue'},
        'Mb-Mf': {'file': 'MBMF_Hopper-v5_1_2026-07-25_23-43-04.csv', 'color': 'red'},
        'Mb': {'file': 'MBMF_Hopper-v5_1_2026-07-25_23-43-04_Phase1_MB.csv', 'color': 'green'},
    }
    title :str = "Environment: Hopper"

    datasets = {
        'PPO': {'file': 'PPO_HalfCheetah-v5_1_2026-07-25_22-48-07.csv', 'color': 'blue'},
        'Mb-Mf': {'file': 'MBMF_HalfCheetah-v5_1_2026-07-26_12-25-09.csv', 'color': 'red'},
        'Mb': {'file': 'MBMF_HalfCheetah-v5_1_2026-07-26_12-25-09_Phase1_MB.csv', 'color': 'green'},
    }
    title :str = "Environment: HalfCheetah"

    plt.figure(figsize=(10, 5))

    # 3. Process and Plot each dataset
    for label, config in datasets.items():
        steps, mean, std = load_and_smooth(config['file'], window_size=50)
        
        if steps is not None:
            # Plot the smoothed mean line
            plt.plot(steps, mean, label=label, color=config['color'], linewidth=1.5)
            
            # Plot the shaded region for standard deviation
            plt.fill_between(steps, mean - std, mean + std, color=config['color'], alpha=0.3)

    # 4. Axis Formatting
    # Set X-axis to logarithmic scale as requested and shown in the image
    plt.xscale('log')
    # Turn on minor ticks for the x-axis so matplotlib knows where to draw the grid
    plt.minorticks_on()
    # Draw major grid lines (solid, slightly more prominent white)
    plt.grid(True, which='major', axis='both', color='white', linestyle='-', linewidth=0.8, alpha=0.9)
    # Draw minor grid lines (thin white lines between the powers of 10)
    plt.grid(True, which='minor', axis='x', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    # ----------------------------------------------------------------
    
    # Set the exact limits based on your 1M step count (10^3 to 10^6)
    # Adjust the lower bound (e.g., 1000) based on where your data actually starts
    plt.xlim(left=3000, right=1000000) 

    # Labels and Title
    plt.xlabel('Steps', fontsize=22)
    plt.ylabel('Cumulative Reward', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.title(title, fontsize=24)

    # 5. Legend Formatting
    # Place the legend in the bottom right corner
    plt.legend(loc='upper left', frameon=False, fontsize=22)

    # Tight layout ensures no clipping of labels
    plt.tight_layout()
    
    # Save the output figure
    plt.savefig('recreated_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_recreated_graph()