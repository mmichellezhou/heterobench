import os
import re
from typing import Dict, List, Optional, Tuple
import json
import argparse
from datetime import datetime
import logging
from agent import HeteroBenchAgent
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

def process_kernels(agent: HeteroBenchAgent, kernel_names: List[str], output_dir: str, model: str, provider: str) -> Dict:
    """
    Process a list of kernels and save the results.
    
    Args:
        agent: The HeteroBenchAgent instance
        kernel_names: List of kernel names to process
        output_dir: Base output directory
        model: Model name being used
        provider: Provider name being used
        
    Returns:
        Dictionary containing results for all processed kernels
    """
    all_results = {}
    for kernel_name in kernel_names:
        if kernel_name not in agent.KERNEL_FILES:
            logging.error(f"Kernel '{kernel_name}' not found. Available kernels: {', '.join(agent.KERNEL_FILES.keys())}")
            continue
            
        kernel_output_dir = os.path.join(output_dir, kernel_name)
        os.makedirs(kernel_output_dir, exist_ok=True)
        
        # Process the kernel
        results = agent.process_kernel(kernel_name, kernel_output_dir)
        
        # Collect status and speedup for each kernel
        kernel_summary = {
            "kernel_generation_success": results["kernel_generation_success"],
            "compilation_success": results["compilation_success"],
            "execution_success": results["execution_success"],
        }
        # Add speedup if available
        if "compile_and_run" in results and results["compile_and_run"].get("run_output"):
            # Extract speedup from run output
            run_output = results["compile_and_run"]["run_output"]
            speedup_match = re.search(r"Speedup: ([\d.]+)x", run_output)
            if speedup_match:
                kernel_summary["speedup"] = float(speedup_match.group(1))
            # Also check if speedup is directly available in compile_and_run results
            elif results["compile_and_run"].get("speedup", 0) > 0:
                kernel_summary["speedup"] = results["compile_and_run"]["speedup"]
        all_results[kernel_name] = kernel_summary
    
    # Save aggregated results
    all_summary_path = os.path.join(output_dir, "all_summary.json")
    with open(all_summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Summary saved to {all_summary_path}")
    
    return all_results

def main():
    """
    Main function to demonstrate the framework.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate optimized kernel code using LLMs')
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'google'],
                      help='LLM provider to use (default: openai)')
    parser.add_argument('--model', type=str, required=True, help='Model name (provider-specific)')
    parser.add_argument('--kernel', type=str, nargs='+', default=['3_matrix_multiplication'],
                      help='One or more kernels to process (default: 3_matrix_multiplication)')
    parser.add_argument('--all', action='store_true',
                      help='Process all available kernels')
    args = parser.parse_args()

    # Create output directory first
    output_dir = f"llm_output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    logging.info(f"Logging initialized. Log file: {log_file}")

    try:
        # Create LLM instance
        llm = LLM(provider=args.provider, model=args.model)
        logging.info(f"Initialized {llm}")
        
        # Initialize the agent
        agent = HeteroBenchAgent(llm)

        if args.all:
            # Process all examples
            logging.info(f"Processing all HeteroBench examples using {args.model} by {args.provider.upper()}...")
            process_kernels(agent, list(agent.KERNEL_FILES.keys()), output_dir, args.model, args.provider)
        else:
            # Process specified kernels
            process_kernels(agent, args.kernel, output_dir, args.model, args.provider)
            
    except ValueError as e:
        logging.error(f"Error: {e}")

    # then run the plot_speedup.py
    plot_speedup(output_dir)

if __name__ == "__main__":
    main()