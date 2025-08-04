import os
import re
from typing import Dict, List, Optional, Tuple
import json
import argparse
from datetime import datetime
import logging
from agent import KernelCodeGenerator
from llm import LLM
from backend_factory import BackendFactory
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


def find_best_iteration_and_create_summary(
    results_all_iter: Dict, benchmark_name: str, backend_name: str
) -> Dict:
    """
    Find the best iteration from multiple iterations and create benchmark summary.

    Args:
        results_all_iter: Dictionary of results from all iterations
        benchmark_name: Name of the benchmark
        backend_name: Name of the backend

    Returns:
        Dictionary containing benchmark summary with best iteration results
    """
    # Handle case where only one result is returned (function generation failed)
    if (
        not isinstance(results_all_iter, dict)
        or "function_generation_success" in results_all_iter
    ):
        # Single result returned, use it directly
        best_results = results_all_iter
        best_iteration = 0
    else:
        # Multiple iterations, find the best one
        best_iteration = None
        best_results = None

        # Priority order: Better speedup > Verification pass > Execution pass > Compile pass
        best_speedup = -1

        for iteration, results in results_all_iter.items():
            # Priority 1: Best speedup
            if (
                results.get("compilation_success", False)
                and results.get("execution_success", False)
                and results.get("verification_success", False)
                and "performance_analysis" in results
                and results["performance_analysis"].get("speedup") is not None
            ):

                speedup = results["performance_analysis"]["speedup"]
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_iteration = iteration
                    best_results = results

        # If no speedup found, try verification pass
        if best_results is None:
            for iteration, results in results_all_iter.items():
                if (
                    results.get("compilation_success", False)
                    and results.get("execution_success", False)
                    and results.get("verification_success", False)
                ):
                    best_iteration = iteration
                    best_results = results
                    break

        # If no verification pass, try execution pass
        if best_results is None:
            for iteration, results in results_all_iter.items():
                if results.get("compilation_success", False) and results.get(
                    "execution_success", False
                ):
                    best_iteration = iteration
                    best_results = results
                    break

        # If no execution pass, try compilation pass
        if best_results is None:
            for iteration, results in results_all_iter.items():
                if results.get("compilation_success", False):
                    best_iteration = iteration
                    best_results = results
                    break

        # If no compilation pass, use the first iteration
        if best_results is None:
            best_iteration = list(results_all_iter.keys())[0]
            best_results = results_all_iter[best_iteration]

    # Create summary
    summary = {
        "benchmark_name": benchmark_name,
        "backend_name": backend_name,
        "best_iteration": best_iteration,
        "kernel_generation_success": best_results.get(
            "all_functions_successful", False
        ),
        "compilation_success": best_results.get("compilation_success", False),
        "execution_success": best_results.get("execution_success", False),
        "verification_success": best_results.get("verification_success", False),
        "speedup": None,
        "original_time": None,
        "optimized_time": None,
        "functions_processed": 0,
        "functions_successful": 0,
    }

    # Add performance metrics if available
    if "performance_analysis" in best_results:
        performance = best_results["performance_analysis"]
        summary["speedup"] = performance.get("speedup")
        summary["original_time"] = performance.get("original_time")
        summary["optimized_time"] = performance.get("optimized_time")

    # Add function statistics
    if "function_results" in best_results:
        function_results = best_results["function_results"]
        summary["functions_processed"] = len(function_results)
        summary["functions_successful"] = sum(
            1
            for f in function_results.values()
            if f.get("function_generation_success", False)
        )

    # Log the best iteration selection
    logging.info("=" * 60)
    if (
        isinstance(results_all_iter, dict)
        and "function_generation_success" not in results_all_iter
    ):
        if summary.get("speedup") is not None:
            logging.info(
                f"📈 Best iteration for {benchmark_name}: Iteration {best_iteration} (Speedup: {summary['speedup']:.2f}x)"
            )
        elif summary["verification_success"]:
            logging.info(
                f"📈 Best iteration for {benchmark_name}: Iteration {best_iteration} (Verification passed)"
            )
        elif summary["execution_success"]:
            logging.info(
                f"📈 Best iteration for {benchmark_name}: Iteration {best_iteration} (Execution passed)"
            )
        elif summary["compilation_success"]:
            logging.info(
                f"📈 Best iteration for {benchmark_name}: Iteration {best_iteration} (Compilation passed)"
            )
        else:
            logging.info(
                f"📈 Best iteration for {benchmark_name}: Iteration {best_iteration} (Last iteration)"
            )
    else:
        logging.info(f"📈 Single iteration for {benchmark_name}")

    return summary

def process_benchmarks(
    generator: KernelCodeGenerator,
    benchmark_names: List[str],
    available_benchmarks: Dict[str, str],
    output_dir: str,
    model: str,
    provider: str,
    backend_name: str,
    max_iterations: int = 1,
) -> Dict:
    """
    Process a list of HeteroBench benchmarks and save the results.

    Args:
        generator: The KernelCodeGenerator instance
        benchmark_names: List of benchmark names to process
        available_benchmarks: Dictionary of available benchmarks
        output_dir: Base output directory
        model: Model name being used
        provider: Provider name being used
        backend_name: Backend name being used
        max_iterations: Number of feedback iterations to fix compilation/runtime errors

    Returns:
        Dictionary containing results for all processed benchmarks
    """
    all_results = {}

    for benchmark_name in benchmark_names:
        if benchmark_name not in available_benchmarks:
            logging.error(
                f"Benchmark '{benchmark_name}' not found. Available benchmarks: {', '.join(available_benchmarks.keys())}"
            )
            continue

        benchmark_path = available_benchmarks[benchmark_name]

        try:
            # Process all functions in the benchmark (includes compilation and running)
            benchmark_results = generator.process_benchmark(
                benchmark_name, benchmark_path, output_dir, max_iterations
            )

            # Find best iteration and create benchmark summary
            benchmark_summary = find_best_iteration_and_create_summary(
                benchmark_results, benchmark_name, backend_name
            )
            all_results[benchmark_name] = benchmark_summary

        except Exception as e:
            logging.error(f"Error processing benchmark '{benchmark_name}': {str(e)}")
            all_results[benchmark_name] = {
                "benchmark_name": benchmark_name,
                "backend_name": backend_name,
                "best_iteration": None,
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

    # Show relative path starting from llm_output
    relative_path = os.path.relpath(
        all_summary_path, start=os.path.dirname(os.path.dirname(output_dir))
    )
    logging.info(f"Summary saved to {relative_path}")

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
        "--backend",
        type=str,
        required=True,
        choices=BackendFactory.get_available_backends(),
        help="Hardware backend to target",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        nargs="+",
        help="One or more benchmarks to process (backend-specific)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all available benchmarks"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1,
        help="Number of feedback iterations to fix compilation/runtime errors.",
    )
    args = parser.parse_args()

    # Create output directory first
    output_dir = f"llm_output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.backend}_{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    # Setup logging
    log_file = setup_logging(output_dir)
    relative_log_path = os.path.relpath(
        log_file, start=os.path.dirname(os.path.dirname(output_dir))
    )
    logging.info(f"Logging initialized. Log file: {relative_log_path}")

    try:
        # Create backend instance
        backend = BackendFactory.create_backend(args.backend)
        logging.info(f"Initialized {args.backend} backend")

        # Get available benchmarks for this backend
        available_benchmarks = backend.get_available_kernels()

        # Create LLM instance
        llm = LLM(provider=args.provider, model=args.model)
        logging.info(f"Initialized {llm}")

        # Initialize the generator with backend
        generator = KernelCodeGenerator(llm, backend)

        # Determine which benchmarks to process
        if args.all:
            # Process all available benchmarks for this backend
            benchmarks_to_process = list(available_benchmarks.keys())
            logging.info(
                f"Processing all {len(benchmarks_to_process)} benchmarks for {args.backend} backend using {args.model} by {args.provider.upper()}..."
            )
        else:
            # Process specified benchmarks or default
            if args.kernel:
                benchmarks_to_process = args.kernel
            else:
                # Use first available benchmark as default
                benchmarks_to_process = (
                    [list(available_benchmarks.keys())[0]]
                    if available_benchmarks
                    else []
                )
                logging.info(
                    f"No benchmarks specified, using default: {benchmarks_to_process}"
                )

            logging.info(
                f"Processing benchmarks {benchmarks_to_process} for {args.backend} backend using {args.model} by {args.provider.upper()}..."
            )

        if not benchmarks_to_process:
            logging.error(f"No benchmarks to process for {args.backend} backend")
            return

        # Process benchmarks
        process_benchmarks(
            generator,
            benchmarks_to_process,
            available_benchmarks,
            output_dir,
            args.model,
            args.provider,
            args.backend,
            max_iterations=args.max_iterations,
        )

    except ValueError as e:
        logging.error(f"Error: {e}")
        return

    # Generate the speedup plot
    try:
        plot_speedup(output_dir)
    except Exception as e:
        logging.warning(f"Could not generate plots: {e}")


if __name__ == "__main__":
    main()
