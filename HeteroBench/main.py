import os
import re
from typing import Dict, List, Optional, Tuple
import json
import argparse
from datetime import datetime
import logging
import subprocess
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
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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
    "sobel_filter": "benchmarks/sobel_filter"
}

def run_benchmark_and_analyze(generator: HeteroBenchCodeGenerator, benchmark_name: str, benchmark_path: str, output_dir: str) -> Dict:
    """
    Run the benchmark's main_simple.cpp and analyze the output for performance metrics.
    
    Args:
        generator: The HeteroBenchCodeGenerator instance
        benchmark_name: Name of the benchmark
        benchmark_path: Path to the benchmark directory
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing performance analysis results
    """
    cpp_path = os.path.join(benchmark_path, "homobackend_cpu", "Cpp")
    
    if not os.path.exists(cpp_path):
        raise FileNotFoundError(f"CPP directory not found: {cpp_path}")
    
    # Change to the CPP directory
    original_cwd = os.getcwd()
    os.chdir(cpp_path)
    
    try:
        # Build the benchmark
        logging.info(f"Building {benchmark_name}...")
        build_result = subprocess.run(
            ["make", "run_simple"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save build output
        with open(os.path.join(output_dir, f"{benchmark_name}_build_output.txt"), 'w') as f:
            f.write("BUILD COMMAND: make run_simple\n")
            f.write("STDOUT:\n")
            f.write(build_result.stdout)
            f.write("\nSTDERR:\n")
            f.write(build_result.stderr)
        
        # Analyze the output
        analysis = {
            "verification_success": False,
            "original_time": None,
            "optimized_time": None,
            "speedup": None,
            "build_success": build_result.returncode == 0
        }
        
        if build_result.returncode == 0:
            # Use the agent's verification logic
            analysis["verification_success"] = generator.determine_verification_success(build_result.stdout)

            # Parse the output for performance metrics
            lines = build_result.stdout.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for speedup
                if "Speedup:" in line:
                    # Look for "Total:" on the next few lines
                    for j in range(i+1, min(i+10, len(lines))):
                        next_line = lines[j].strip()
                        if "Total:" in next_line:
                            try:
                                # Extract speedup value (format: "Total: X.XXXXXXx")
                                speedup_str = next_line.split(":")[-1].replace("x", "").strip()
                                analysis["speedup"] = float(speedup_str)
                            except (ValueError, IndexError):
                                pass
                            break
                
                # Look for timing information
                elif "Original Implementation:" in line:
                    for j in range(i+1, min(i+10, len(lines))):
                        next_line = lines[j].strip()
                        if "Single iteration time:" in next_line:
                            try:
                                time_str = next_line.split(":")[-1].replace("seconds", "").strip()
                                analysis["original_time"] = float(time_str)
                            except (ValueError, IndexError):
                                pass
                            break
                
                elif "Optimized Implementation:" in line:
                    for j in range(i+1, min(i+10, len(lines))):
                        next_line = lines[j].strip()
                        if "Single iteration time:" in next_line:
                            try:
                                time_str = next_line.split(":")[-1].replace("seconds", "").strip()
                                analysis["optimized_time"] = float(time_str)
                            except (ValueError, IndexError):
                                pass
                            break
        
        # Calculate speedup if we have both timing values but speedup parsing failed
        if analysis["speedup"] is None and analysis["original_time"] is not None and analysis["optimized_time"] is not None:
            if analysis["optimized_time"] > 0:
                analysis["speedup"] = analysis["original_time"] / analysis["optimized_time"]
        
        return analysis
        
    finally:
        # Change back to original directory
        os.chdir(original_cwd)

def process_benchmarks(generator: HeteroBenchCodeGenerator, kernel_names: List[str], output_dir: str, model: str, provider: str) -> Dict:
    """
    Process a list of HeteroBench benchmarks and save the results.
    
    Args:
        generator: The HeteroBenchCodeGenerator instance
        kernel_names: List of benchmark names to process
        output_dir: Base output directory
        model: Model name being used
        provider: Provider name being used
        
    Returns:
        Dictionary containing results for all processed benchmarks
    """
    all_results = {}
    
    for benchmark_name in kernel_names:
        if benchmark_name not in HETEROBENCH_BENCHMARKS:
            logging.error(f"Benchmark '{benchmark_name}' not found. Available benchmarks: {', '.join(HETEROBENCH_BENCHMARKS.keys())}")
            continue
            
        benchmark_path = HETEROBENCH_BENCHMARKS[benchmark_name]
        
        logging.info(f"Processing benchmark '{benchmark_name}'...")
        
        try:
            # Process all functions in the benchmark
            function_results = generator.process_benchmark(benchmark_name, benchmark_path, output_dir)
            
            # Run the benchmark and analyze performance
            performance_analysis = run_benchmark_and_analyze(generator, benchmark_name, benchmark_path, output_dir)
            
            # Collect overall benchmark status
            benchmark_summary = {
                "function_results": function_results,
                "performance_analysis": performance_analysis,
                "overall_status": {
                    "functions_processed": len(function_results),
                    "functions_successful": sum(1 for r in function_results.values() if r.get("function_generation_success", False)),
                    "build_success": performance_analysis.get("build_success", False),
                    "verification_success": performance_analysis.get("verification_success", False)
                }
            }
            
            # Add speedup if available
            if performance_analysis.get("speedup") is not None:
                benchmark_summary["speedup"] = performance_analysis["speedup"]
            
            all_results[benchmark_name] = benchmark_summary
            
            # Log results
            logging.info(f"✓ Benchmark '{benchmark_name}' completed")
            if performance_analysis.get("speedup"):
                logging.info(f"  Speedup: {performance_analysis['speedup']:.2f}x")
            if performance_analysis.get("verification_success"):
                logging.info(f"  Verification: ✓ PASS")
            else:
                logging.warning(f"  Verification: ✗ FAIL")
                
        except Exception as e:
            logging.error(f"✗ Error processing benchmark '{benchmark_name}': {str(e)}")
            all_results[benchmark_name] = {
                "error": str(e),
                "overall_status": {
                    "functions_processed": 0,
                    "functions_successful": 0,
                    "build_success": False,
                    "verification_success": False
                }
            }
    
    # Save aggregated results
    all_summary_path = os.path.join(output_dir, "all_summary.json")
    with open(all_summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Summary saved to {all_summary_path}")
    
    return all_results


def main():
    """
    Main function to demonstrate the HeteroBench optimization framework.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate optimized code for HeteroBench using LLMs')
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'google'],
                      help='LLM provider to use')
    parser.add_argument('--model', type=str, required=True, help='Model name (provider-specific)')
    parser.add_argument('--kernel', type=str, nargs='+', default=['optical_flow'],
                      help='One or more benchmarks to process (default: optical_flow)')
    parser.add_argument('--all', action='store_true',
                      help='Process all available benchmarks')
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
            logging.info(f"Processing all HeteroBench examples using {args.model} by {args.provider.upper()}...")
            process_benchmarks(generator, list(HETEROBENCH_BENCHMARKS.keys()), output_dir, args.model, args.provider)
        else:
            # Process specified benchmarks
            process_benchmarks(generator, args.kernel, output_dir, args.model, args.provider)
            
    except ValueError as e:
        logging.error(f"Error: {e}")

    # Generate the speedup plot
    plot_speedup(output_dir)

if __name__ == "__main__":
    main()