import os
import re
from typing import Dict, List, Optional, Tuple
import json
import argparse
from datetime import datetime
import logging
from agent import KernelCodeGenerator
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

# Example PolyBench kernels
POLYBENCH_CODE_PATH = {
    # datamining
    "correlation": "polybench/datamining/correlation/correlation_simple.c",
    "covariance": "polybench/datamining/covariance/covariance_simple.c",
    # linear-algebra/blas
    "gemm": "polybench/linear-algebra/blas/gemm/gemm_simple.c",
    "gemver": "polybench/linear-algebra/blas/gemver/gemver_simple.c",
    "gesummv": "polybench/linear-algebra/blas/gesummv/gesummv_simple.c",
    "symm": "polybench/linear-algebra/blas/symm/symm_simple.c",
    "syr2k": "polybench/linear-algebra/blas/syr2k/syr2k_simple.c",
    "syrk": "polybench/linear-algebra/blas/syrk/syrk_simple.c",
    "trmm": "polybench/linear-algebra/blas/trmm/trmm_simple.c",
    # linear-algebra/kernels
    "2mm": "polybench/linear-algebra/kernels/2mm/2mm_simple.c",
    "3mm": "polybench/linear-algebra/kernels/3mm/3mm_simple.c",
    "atax": "polybench/linear-algebra/kernels/atax/atax_simple.c",
    "bicg": "polybench/linear-algebra/kernels/bicg/bicg_simple.c",
    "doitgen": "polybench/linear-algebra/kernels/doitgen/doitgen_simple.c",
    "mvt": "polybench/linear-algebra/kernels/mvt/mvt_simple.c",
    # linear-algebra/solvers
    "cholesky": "polybench/linear-algebra/solvers/cholesky/cholesky_simple.c",
    "durbin": "polybench/linear-algebra/solvers/durbin/durbin_simple.c",
    "gramschmidt": "polybench/linear-algebra/solvers/gramschmidt/gramschmidt_simple.c",
    "lu": "polybench/linear-algebra/solvers/lu/lu_simple.c",
    "ludcmp": "polybench/linear-algebra/solvers/ludcmp/ludcmp_simple.c",
    "trisolv": "polybench/linear-algebra/solvers/trisolv/trisolv_simple.c",
    # medley
    "deriche": "polybench/medley/deriche/deriche_simple.c",
    "floyd-warshall": "polybench/medley/floyd-warshall/floyd-warshall_simple.c",
    "nussinov": "polybench/medley/nussinov/nussinov_simple.c",
    # stencils
    "adi": "polybench/stencils/adi/adi_simple.c",
    "fdtd-2d": "polybench/stencils/fdtd-2d/fdtd-2d_simple.c",
    "heat-3d": "polybench/stencils/heat-3d/heat-3d_simple.c",
    "jacobi-1d": "polybench/stencils/jacobi-1d/jacobi-1d_simple.c",
    "jacobi-2d": "polybench/stencils/jacobi-2d/jacobi-2d_simple.c",
    "seidel-2d": "polybench/stencils/seidel-2d/seidel-2d_simple.c" 
}

def process_kernels(generator: KernelCodeGenerator, kernel_names: List[str], output_dir: str, model: str, provider: str) -> Dict:
    """
    Process a list of kernels and save the results.
    
    Args:
        generator: The KernelCodeGenerator instance
        kernel_names: List of kernel names to process
        output_dir: Base output directory
        model: Model name being used
        provider: Provider name being used
        
    Returns:
        Dictionary containing results for all processed kernels
    """
    all_results = {}
    for kernel_name in kernel_names:
        if kernel_name not in POLYBENCH_CODE_PATH:
            logging.error(f"Kernel '{kernel_name}' not found. Available kernels: {', '.join(POLYBENCH_CODE_PATH.keys())}")
            continue
            
        kernel_code_path = POLYBENCH_CODE_PATH[kernel_name]
        kernel_output_dir = os.path.join(output_dir, kernel_name)
        os.makedirs(kernel_output_dir, exist_ok=True)
        
        # Process the kernel
        results = generator.process_kernel(kernel_code_path, kernel_name, kernel_output_dir)
        
        # Collect status and speedup for each kernel
        kernel_summary = {
            "kernel_generation_success": results["kernel_generation_success"],
            "compilation_success": results["compilation_success"],
            "execution_success": results["execution_success"],
            "verification_success": results["verification_success"],
        }
        # Add speedup if available
        if "run_analysis" in results and results["run_analysis"].get("speedup") is not None:
            kernel_summary["speedup"] = results["run_analysis"]["speedup"]
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
    parser.add_argument('--kernel', type=str, nargs='+', default=['gemm'],
                      help='One or more kernels to process (default: gemm)')
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
        
        # Initialize the generator
        generator = KernelCodeGenerator(llm)

        if args.all:
            # Process all examples
            logging.info(f"Processing all PolyBench examples using {args.model} by {args.provider.upper()}...")
            process_kernels(generator, list(POLYBENCH_CODE_PATH.keys()), output_dir, args.model, args.provider)
        else:
            # Process specified kernels
            process_kernels(generator, args.kernel, output_dir, args.model, args.provider)
            
    except ValueError as e:
        logging.error(f"Error: {e}")

    # then run the plot_speedup.py
    plot_speedup(output_dir)

if __name__ == "__main__":
    main()