import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import argparse

def plot_speedup(output_dir: str):
    """
    Plot speedup comparison for all processed kernels.
    
    Args:
        output_dir: Directory containing the results
    """
    # Read the summary file
    summary_path = os.path.join(output_dir, "all_summary.json")
    if not os.path.exists(summary_path):
        print(f"Warning: No summary file found at {summary_path}")
        return
        
    with open(summary_path, 'r') as f:
        all_results = json.load(f)
    
    # Filter kernels with successful generation and execution
    valid_kernels = {
        name: data for name, data in all_results.items()
        if data.get('kernel_generation_success', False) and 
           data.get('execution_success', False) and
           data.get('speedup', 0) > 0
    }
    
    if not valid_kernels:
        print("No valid kernels found with speedup data")
        return
    
    # Create DataFrame for plotting
    df = pd.DataFrame([
        {
            'Kernel': name,
            'Speedup': data['speedup']
        }
        for name, data in valid_kernels.items()
    ])
    
    # Sort by speedup
    df = df.sort_values('Speedup', ascending=False)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create bar plot
    ax = sns.barplot(data=df, x='Kernel', y='Speedup')
    
    # Customize the plot
    plt.title('Speedup of LLM-Optimized vs Reference Kernels', pad=20)
    plt.xlabel('Kernel')
    plt.ylabel('Speedup (x)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(df['Speedup']):
        ax.text(i, v, f'{v:.2f}x', ha='center', va='bottom')
    
    # Add a horizontal line at y=1
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "speedup_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Speedup plot saved to: {plot_path}")
    
    # Print summary statistics
    print("\nSpeedup Summary:")
    print("=" * 50)
    print(f"Number of kernels: {len(valid_kernels)}")
    print(f"Average speedup: {df['Speedup'].mean():.2f}x")
    print(f"Maximum speedup: {df['Speedup'].max():.2f}x")
    print(f"Minimum speedup: {df['Speedup'].min():.2f}x")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    args = parser.parse_args()

    plot_speedup(args.work_dir)

if __name__ == "__main__":
    main()