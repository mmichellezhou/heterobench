import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os

def plot_speedup(work_dir):
    # Read the JSON file
    summary_file = os.path.join(work_dir, "all_summary.json")
    with open(summary_file, "r") as f:
        data = json.load(f)

    # Prepare data for plotting
    kernels = list(data.keys())
    successful_indices = []
    successful_speedups = []
    error_indices = []
    error_types = []

    for i, kernel in enumerate(kernels):
        kernel_data = data[kernel]

        # Check for different types of errors
        if not kernel_data["kernel_generation_success"]:
            error_indices.append(i)
            error_types.append("kernel_gen")
        elif not kernel_data["compilation_success"]:
            error_indices.append(i)
            error_types.append("compile")
        elif not kernel_data["execution_success"]:
            error_indices.append(i)
            error_types.append("runtime")
        elif not kernel_data["verification_success"]:
            error_indices.append(i)
            error_types.append("verification")
        else:
            # Only add to successful if speedup is not None
            if kernel_data["speedup"] is not None:
                successful_indices.append(i)
                successful_speedups.append(kernel_data["speedup"])
            else:
                # If speedup is None but everything else succeeded, mark as verification error
                error_indices.append(i)
                error_types.append("verification")

    # Set up the plot
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")

    # Set y-axis to log scale
    plt.yscale("log")

    # Add reference line at y=1
    plt.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Baseline (1x)")

    # Create bar plot for successful speedups
    if successful_indices:
        bars = plt.bar(
            successful_indices, successful_speedups, color="skyblue", alpha=0.7
        )

        # Add speedup values on top of bars
        for bar, speedup in zip(bars, successful_speedups):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{speedup:.1f}",
                ha="center",
                va="bottom",
                rotation=0,
            )

    # Add error markers
    error_colors = {
        "kernel_gen": "red",
        "compile": "orange",
        "runtime": "purple",
        "verification": "green",
    }

    for i, error_type in zip(error_indices, error_types):
        plt.plot(
            i,
            0.1,
            "x",
            color=error_colors[error_type],
            markersize=10,
            markeredgewidth=2,
        )

    # Customize the plot
    plt.xticks(range(len(kernels)), kernels, rotation=45, ha="right")
    plt.ylabel("Speedup (log scale)")
    plt.title("HeteroBench Speedup Results with Error Types")

    # Add legend for error types with proper colors
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color=color,
            markersize=10,
            markeredgewidth=2,
            label=error_type,
            linestyle="None",
        )
        for error_type, color in error_colors.items()
    ]

    plt.legend(handles=legend_elements, loc="upper right")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(work_dir, "speedup_plot.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    args = parser.parse_args()

    plot_speedup(args.work_dir)


if __name__ == "__main__":
    main()
