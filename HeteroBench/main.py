import os
import re
from typing import Dict, List, Optional, Tuple
import json
import argparse
from datetime import datetime
import logging
from agent import HeteroBenchCodeGenerator
from llm import LLM
from tqdm import tqdm
from plot import plot_speedup


# Configure logging
def setup_logging(output_dir: str):
    """Setup logging to both file and console"""
    log_file = os.path.join(output_dir, f"run.log")

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return log_file


# HeteroBench benchmarks
HETEROBENCH_BENCHMARKS = {
    "optical_flow": "benchmarks/optical_flow",
    "canny_edge_detection": "benchmarks/canny_edge_detection",
    "convolutional_neural_network": "benchmarks/convolutional_neural_network",
    "multilayer_perceptron": "benchmarks/multilayer_perceptron",
    "one_head_attention": "benchmarks/one_head_attention",
    "spam_filter": "benchmarks/spam_filter",
    "3_matrix_multiplication": "benchmarks/3_matrix_multiplication",
    "alternating_direction_implicit": "benchmarks/alternating_direction_implicit",
    "digit_recog": "benchmarks/digit_recog",
    "parallelize_particle": "benchmarks/parallelize_particle",
    "sobel_filter": "benchmarks/sobel_filter",
}


def process_benchmarks(
    generator: HeteroBenchCodeGenerator,
    kernel_names: List[str],
    output_dir: str,
    model: str,
    provider: str,
    max_iterations: int = 1,
) -> Dict:
    """
    Process a list of HeteroBench benchmarks and save the results.

    Args:
        generator: The HeteroBenchCodeGenerator instance
        kernel_names: List of benchmark names to process
        output_dir: Base output directory
        model: Model name being used
        provider: Provider name being used
        max_iterations: Number of feedback iterations to fix compilation/runtime errors

    Returns:
        Dictionary containing results for all processed benchmarks
    """
    all_results = {}

    for benchmark_name in kernel_names:
        if benchmark_name not in HETEROBENCH_BENCHMARKS:
            logging.error(
                f"Benchmark '{benchmark_name}' not found. Available benchmarks: {', '.join(HETEROBENCH_BENCHMARKS.keys())}"
            )
            continue

        benchmark_path = HETEROBENCH_BENCHMARKS[benchmark_name]

        logging.info(f"Processing benchmark '{benchmark_name}'...")
        logging.info("=" * 60)

        try:
            # Process all functions in the benchmark (includes compilation and running)
            benchmark_results = generator.process_benchmark(
                benchmark_name, benchmark_path, output_dir, max_iterations=max_iterations
            )

            # Extract summary information from the agent's results
            all_functions_successful = benchmark_results.get(
                "all_functions_successful", False
            )
            compilation_success = benchmark_results.get("compilation_success", False)
            execution_success = benchmark_results.get("execution_success", False)
            verification_success = benchmark_results.get("verification_success", False)

            # Get performance metrics from run analysis
            run_analysis = benchmark_results.get("run_analysis", {})
            speedup = run_analysis.get("speedup")
            original_time = run_analysis.get("original_time")
            optimized_time = run_analysis.get("optimized_time")

            # Count functions
            functions_processed = 0
            functions_successful = 0
            for key, value in benchmark_results.items():
                if isinstance(value, dict) and "function_generation_success" in value:
                    functions_processed += 1
                    if value.get("function_generation_success", False):
                        functions_successful += 1

            # Create flat benchmark summary
            benchmark_summary = {
                "kernel_generation_success": all_functions_successful,
                "compilation_success": compilation_success,
                "execution_success": execution_success,
                "verification_success": verification_success,
                "speedup": speedup,
                "original_time": original_time,
                "optimized_time": optimized_time,
                "functions_processed": functions_processed,
                "functions_successful": functions_successful,
            }

            all_results[benchmark_name] = benchmark_summary

        except Exception as e:
            logging.error(f"✗ Error processing benchmark '{benchmark_name}': {str(e)}")
            all_results[benchmark_name] = {
                "kernel_generation_success": False,
                "compilation_success": False,
                "execution_success": False,
                "verification_success": False,
                "speedup": None,
                "original_time": None,
                "optimized_time": None,
                "functions_processed": 0,
                "functions_successful": 0,
                "error": str(e),
            }

    # Save aggregated results
    all_summary_path = os.path.join(output_dir, "all_summary.json")
    with open(all_summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Summary saved to {all_summary_path}")

    return all_results


def main():
    """
    Main function to demonstrate the HeteroBench optimization framework.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate optimized code for HeteroBench using LLMs"
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["openai", "google"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (provider-specific)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        nargs="+",
        default=["optical_flow"],
        help="One or more benchmarks to process (default: optical_flow)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all available benchmarks"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=1,
        help="Number of feedback iterations to fix compilation/runtime errors."
    )
    args = parser.parse_args()

    # Create output directory first
    output_dir = f"llm_output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    # Setup logging
    log_file = setup_logging(output_dir)
    logging.info(f"Logging initialized. Log file: {log_file}")

    try:
        # Create LLM instance
        llm = LLM(provider=args.provider, model=args.model)
        logging.info(f"Initialized {llm}")

        # Initialize the generator
        generator = HeteroBenchCodeGenerator(llm)

        if args.all:
            # Process all benchmarks
            logging.info(
                f"Processing all HeteroBench examples using {args.model} by {args.provider.upper()}..."
            )
            process_benchmarks(
                generator,
                list(HETEROBENCH_BENCHMARKS.keys()),
                output_dir,
                args.model,
                args.provider,
                max_iterations=args.max_iterations,
            )
        else:
            # Process specified benchmarks
            process_benchmarks(
                generator, args.kernel, output_dir, args.model, args.provider, max_iterations=args.max_iterations
            )

    except ValueError as e:
        logging.error(f"Error: {e}")

    # Generate the speedup plot
    plot_speedup(output_dir)


if __name__ == "__main__":
    main()
