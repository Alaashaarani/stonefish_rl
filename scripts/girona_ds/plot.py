import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def debug_npz_structure(npz_file_path):
    """
    Debug function to examine the structure of the .npz file
    """
    print("=== DEBUGGING NPZ FILE STRUCTURE ===")
    data = np.load(npz_file_path)
    
    for key in data.files:
        array = data[key]
        print(f"\n{key}:")
        print(f"  Shape: {array.shape}")
        print(f"  Dimensions: {array.ndim}")
        print(f"  Data type: {array.dtype}")
        print(f"  First few elements: {array[:5] if array.size > 5 else array}")
    
    data.close()

def plot_ppo_evaluation_safe(npz_file_path):
    """
    Safe plotting function that handles different array structures
    """
    try:
        # Load and examine the data
        data = np.load(npz_file_path)
        
        print("Loading arrays from .npz file...")
        timesteps = data['timesteps']
        results = data['results'] 
        ep_lengths = data['ep_lengths']
        
        # Handle different array dimensions
        def safe_squeeze(arr, name):
            original_shape = arr.shape
            arr = np.squeeze(arr)
            if arr.shape != original_shape:
                print(f"Note: {name} reshaped from {original_shape} to {arr.shape}")
            return arr
        
        timesteps = safe_squeeze(timesteps, "timesteps")
        results = safe_squeeze(results, "results")
        ep_lengths = safe_squeeze(ep_lengths, "ep_lengths")
        
        # Ensure all arrays are 1D
        if timesteps.ndim != 1 or results.ndim != 1 or ep_lengths.ndim != 1:
            print("Flattening arrays to 1D...")
            timesteps = timesteps.flatten()
            results = results.flatten()
            ep_lengths = ep_lengths.flatten()
        
        print(f"Final shapes - Timesteps: {timesteps.shape}, Results: {results.shape}, Ep_lengths: {ep_lengths.shape}")
        
        # Create plots
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PPO Training Evaluation (100,000 timesteps)', fontsize=16, fontweight='bold')
        
        # Plot 1: Rewards over time
        axes[0, 0].plot(timesteps, results, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Episode Rewards vs Training Timesteps')
        axes[0, 0].set_xlabel('Training Timesteps')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode lengths over time
        axes[0, 1].plot(timesteps, ep_lengths, 'g-', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('Episode Lengths vs Training Timesteps')
        axes[0, 1].set_xlabel('Training Timesteps')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Reward distribution
        axes[1, 0].hist(results, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(np.mean(results), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(results):.2f}')
        axes[1, 0].set_title('Distribution of Episode Rewards')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Episode length distribution
        axes[1, 1].hist(ep_lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(np.mean(ep_lengths), color='orange', linestyle='--', 
                          label=f'Mean: {np.mean(ep_lengths):.2f}')
        axes[1, 1].set_title('Distribution of Episode Lengths')
        axes[1, 1].set_xlabel('Episode Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\n" + "="*60)
        print("TRAINING SUMMARY STATISTICS")
        print("="*60)
        print(f"Total training timesteps: {timesteps[-1]:,}")
        print(f"Number of evaluation points: {len(timesteps)}")
        print(f"\nRewards:")
        print(f"  Average: {np.mean(results):.2f} ± {np.std(results):.2f}")
        print(f"  Range: [{np.min(results):.2f}, {np.max(results):.2f}]")
        print(f"  Final reward: {results[-1]:.2f}")
        print(f"\nEpisode Lengths:")
        print(f"  Average: {np.mean(ep_lengths):.2f} ± {np.std(ep_lengths):.2f}")
        print(f"  Range: [{np.min(ep_lengths):.2f}, {np.max(ep_lengths):.2f}]")
        
        data.close()
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        print("\nTrying alternative approach...")
        plot_alternative_approach(npz_file_path)

def plot_alternative_approach(npz_file_path):
    """
    Alternative plotting approach for problematic array structures
    """
    try:
        data = np.load(npz_file_path)
        
        # Try to load with different possible key names
        possible_keys = {
            'timesteps': ['timesteps', 'timesteps.npy', 'steps', 't'],
            'results': ['results', 'results.npy', 'rewards', 'r', 'returns'],
            'ep_lengths': ['ep_lengths', 'ep_lengths.npy', 'lengths', 'l', 'ep_lens']
        }
        
        def find_array(keys):
            for key in keys:
                if key in data.files:
                    return data[key]
            return None
        
        timesteps = find_array(possible_keys['timesteps'])
        results = find_array(possible_keys['results'])
        ep_lengths = find_array(possible_keys['ep_lengths'])
        
        if timesteps is None or results is None or ep_lengths is None:
            print("Available keys in .npz file:", data.files)
            raise ValueError("Could not find required arrays with expected names")
        
        print("Using alternative array loading approach...")
        print(f"Timesteps shape: {timesteps.shape}")
        print(f"Results shape: {results.shape}")
        print(f"Episode lengths shape: {ep_lengths.shape}")
        
        # Convert all to 1D arrays
        timesteps = timesteps.ravel()
        results = results.ravel()
        ep_lengths = ep_lengths.ravel()
        
        # Simple plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(timesteps, results, 'b-')
        plt.title('Rewards over Time')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(timesteps, ep_lengths, 'g-')
        plt.title('Episode Lengths over Time')
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Length')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        data.close()
        
    except Exception as e:
        print(f"Alternative approach also failed: {e}")

# Main execution
if __name__ == "__main__":
    file_path = "/home/cirs_alaa/repositories/stonefish_rl/scripts/girona_ds/logs/evaluations.npz"
    
    print("PPO Evaluation Data Plotter")
    print("=" * 40)
    
    # First, debug the file structure
    debug_npz_structure(file_path)
    
    print("\n" + "=" * 40)
    print("Generating plots...")
    print("=" * 40)
    
    # Then try to plot
    plot_ppo_evaluation_safe(file_path)