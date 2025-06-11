import os
import re
from typing import Dict, List, Tuple, Optional
import json
import logging
from llm import LLM

class HeteroBenchAgent:
    """
    A framework for using LLMs to optimize HeteroBench kernels.
    """
    
    # Mapping of kernel names to their implementation files
    KERNEL_FILES = {
        "3_matrix_multiplication": {
            "main": "benchmarks/3_matrix_multiplication/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/3_matrix_multiplication/homobackend_cpu/Cpp/cpu_impl/kernel_3mm_0.cpp",
                "benchmarks/3_matrix_multiplication/homobackend_cpu/Cpp/cpu_impl/kernel_3mm_1.cpp",
                "benchmarks/3_matrix_multiplication/homobackend_cpu/Cpp/cpu_impl/kernel_3mm_2.cpp"
            ],
            "headers": [
                "benchmarks/3_matrix_multiplication/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "alternating_direction_implicit": {
            "main": "benchmarks/alternating_direction_implicit/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/alternating_direction_implicit/homobackend_cpu/Cpp/cpu_impl/kernel_adi.cpp"
            ],
            "headers": [
                "benchmarks/alternating_direction_implicit/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "canny_edge_detection": {
            "main": "benchmarks/canny_edge_detection/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/canny_edge_detection/homobackend_cpu/Cpp/cpu_impl/kernel_canny.cpp"
            ],
            "headers": [
                "benchmarks/canny_edge_detection/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "convolutional_neural_network": {
            "main": "benchmarks/convolutional_neural_network/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/convolutional_neural_network/homobackend_cpu/Cpp/cpu_impl/kernel_conv.cpp"
            ],
            "headers": [
                "benchmarks/convolutional_neural_network/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "digit_recog": {
            "main": "benchmarks/digit_recog/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/digit_recog/homobackend_cpu/Cpp/cpu_impl/kernel_digit.cpp"
            ],
            "headers": [
                "benchmarks/digit_recog/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "multilayer_perceptron": {
            "main": "benchmarks/multilayer_perceptron/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/multilayer_perceptron/homobackend_cpu/Cpp/cpu_impl/kernel_mlp.cpp"
            ],
            "headers": [
                "benchmarks/multilayer_perceptron/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "one_head_attention": {
            "main": "benchmarks/one_head_attention/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/one_head_attention/homobackend_cpu/Cpp/cpu_impl/kernel_attention.cpp"
            ],
            "headers": [
                "benchmarks/one_head_attention/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "optical_flow": {
            "main": "benchmarks/optical_flow/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/optical_flow/homobackend_cpu/Cpp/cpu_impl/kernel_optical_flow.cpp"
            ],
            "headers": [
                "benchmarks/optical_flow/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "parallelize_particle": {
            "main": "benchmarks/parallelize_particle/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/parallelize_particle/homobackend_cpu/Cpp/cpu_impl/kernel_particle.cpp"
            ],
            "headers": [
                "benchmarks/parallelize_particle/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "sobel_filter": {
            "main": "benchmarks/sobel_filter/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/sobel_filter/homobackend_cpu/Cpp/cpu_impl/kernel_sobel.cpp"
            ],
            "headers": [
                "benchmarks/sobel_filter/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        },
        "spam_filter": {
            "main": "benchmarks/spam_filter/homobackend_cpu/Cpp/main.cpp",
            "kernels": [
                "benchmarks/spam_filter/homobackend_cpu/Cpp/cpu_impl/kernel_spam.cpp"
            ],
            "headers": [
                "benchmarks/spam_filter/homobackend_cpu/Cpp/cpu_impl/include/cpu_impl.h"
            ]
        }
    }
    
    def __init__(self, llm: LLM):
        """
        Initialize the HeteroBench agent.
        
        Args:
            llm: LLM instance configured with provider and model
        """
        self.llm = llm
        
    def _read_file(self, file_path: str) -> str:
        """Read a file's contents."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return ""
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {str(e)}")
            return ""
    
    def _extract_kernel_code(self, kernel_files: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Extract kernel code from cpu_impl directory files.
        
        Args:
            kernel_files: Dictionary containing paths to main, kernel, and header files
            
        Returns:
            Dictionary containing the code from each file
        """
        code = {}
        
        # Read kernel files from cpu_impl
        code['kernels'] = []
        for kernel_file in kernel_files['kernels']:
            kernel_code = self._read_file(kernel_file)
            if kernel_code:
                code['kernels'].append(kernel_code)
        
        # Read header files from cpu_impl
        code['headers'] = []
        for header_file in kernel_files['headers']:
            header_code = self._read_file(header_file)
            if header_code:
                code['headers'].append(header_code)
        
        return code
    
    def create_prompt(self, kernel_name: str, code: Dict[str, str]) -> str:
        """
        Create a prompt for the LLM to optimize the kernel.
        
        Args:
            kernel_name: Name of the kernel
            code: Dictionary containing the code from all relevant files
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert in high-performance computing and kernel optimization. Your task is to optimize the kernel implementation in the cpu_impl directory while maintaining functional equivalence. Focus on single-threaded performance improvements.

The original test harness (main.cpp) will be used to verify correctness and measure performance.
Only optimize the kernel implementation in cpu_impl.

Given implementation to optimize:

Kernel implementation(s):
```cpp
{chr(10).join(code['kernels'])}
```

Header file(s):
```cpp
{chr(10).join(code['headers'])}
```

Optimization Requirements:
1. Maintain exact same function signatures and array access patterns as the original code
2. Do not modify any golden_* functions - these are reference implementations for correctness testing
3. The optimized code must be a drop-in replacement for the original code
4. Define any necessary macros or helper functions in the header file
5. Keep the same memory layout as the original code

Important Implementation Details:
1. Include all necessary constant definitions (e.g., N, TSTEPS) in the header file
2. Use the same array dimensions and types as the original code
3. Ensure all array declarations match the original function signatures exactly
4. Do not modify the array access patterns or memory layout
5. Only optimize the kernel implementation in cpu_impl
6. If you use any new variables (e.g., for tiling, unrolling, etc.), you MUST define them in the header file
7. Do not use any variables that aren't defined in the header file

Consider applying these optimizations:
- Vectorization (SIMD instructions)
- Memory access optimization (cache-friendly patterns)
- Loop tiling/blocking (define tile sizes in header)
- Loop unrolling (define unroll factors in header)
- Strength reduction
- Other relevant optimizations

Output format:
Provide the complete optimized implementation in code blocks, including:
1. Header files with all necessary constant definitions and the same interface as the original
2. Optimized kernel implementation(s) with identical function signatures
3. Any necessary helper functions or macros

Do not include any other text outside of code blocks.
"""
        return prompt
    
    def call_llm(self, prompt: str) -> Tuple[str, Optional[Dict]]:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Tuple of (The LLM's response text, entire response dictionary)
        """
        system_prompt = "You are an expert in high-performance computing and kernel optimization."
        return self.llm.generate_completion(system_prompt, prompt)
    
    def _extract_code_blocks(self, response: str) -> List[str]:
        """
        Extract code blocks from the LLM response.
        
        Args:
            response: The LLM's response
            
        Returns:
            List of code blocks found in the response
        """
        if not response:
            return []
        # Pattern to match code blocks with optional language specification
        pattern = r'```(?:[a-zA-Z]*\n)?(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        return [match.strip() for match in matches]
    
    def generate_optimized_code(self, kernel_name: str, output_dir: str) -> Dict:
        """
        Generate optimized implementations for all .cpp files in the cpu_impl directory.
        
        Args:
            kernel_name: Name of the kernel to optimize
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing the results
        """
        if kernel_name not in self.KERNEL_FILES:
            raise ValueError(f"Unknown kernel: {kernel_name}")
        
        # Get kernel files
        kernel_files = self.KERNEL_FILES[kernel_name]
        
        # Create cpu_impl directory structure
        cpu_impl_dir = os.path.join(output_dir, "cpu_impl")
        os.makedirs(cpu_impl_dir, exist_ok=True)
        
        # Create include directory
        include_dir = os.path.join(cpu_impl_dir, "include")
        os.makedirs(include_dir, exist_ok=True)
        
        # Copy all header files from original include directory
        original_include_dir = os.path.join(os.path.dirname(kernel_files['kernels'][0]), "include")
        if os.path.exists(original_include_dir):
            for header_file in os.listdir(original_include_dir):
                if header_file.endswith('.h'):
                    src_path = os.path.join(original_include_dir, header_file)
                    dest_path = os.path.join(include_dir, header_file)
                    try:
                        with open(src_path, 'r') as src, open(dest_path, 'w') as dst:
                            dst.write(src.read())
                    except Exception as e:
                        logging.error(f"Error copying header file {src_path}: {str(e)}")
                        return {"error": f"Failed to copy header file: {str(e)}"}
        
        # Copy any other header files from the original directory
        original_dir = os.path.dirname(kernel_files['main'])
        for header_file in os.listdir(original_dir):
            if header_file.endswith('.h'):
                src_path = os.path.join(original_dir, header_file)
                dest_path = os.path.join(include_dir, header_file)
                try:
                    with open(src_path, 'r') as src, open(dest_path, 'w') as dst:
                        dst.write(src.read())
                except Exception as e:
                    logging.error(f"Error copying header file {src_path}: {str(e)}")
                    return {"error": f"Failed to copy header file: {str(e)}"}
        
        # Get all .cpp files from the original cpu_impl directory
        kernel_dir = os.path.dirname(kernel_files['kernels'][0])
        cpp_files = [f for f in os.listdir(kernel_dir) if f.endswith('.cpp')]
        
        # Dictionary to store optimization results for each file
        optimization_results = {}
        
        # Optimize each .cpp file
        for cpp_file in cpp_files:
            # Read the original file
            src_path = os.path.join(kernel_dir, cpp_file)
            with open(src_path, 'r') as f:
                original_code = f.read()
            
            # Create prompt for this file
            prompt = f"""You are an expert in high-performance computing and kernel optimization. Your task is to optimize the following implementation while maintaining functional equivalence. Focus on single-threaded performance improvements.

The original test harness (main.cpp) will be used to verify correctness and measure performance.

Given implementation to optimize:

```cpp
{original_code}
```

Optimization Requirements:
1. Maintain exact same function signatures and array access patterns as the original code
2. Do not modify any golden_* functions - these are reference implementations for correctness testing
3. The optimized code must be a drop-in replacement for the original code
4. Keep the same memory layout as the original code

Important Implementation Details:
1. Use the same array dimensions and types as the original code
2. Ensure all array declarations match the original function signatures exactly
3. Do not modify the array access patterns or memory layout

Consider applying these optimizations:
- Vectorization (SIMD instructions)
- Memory access optimization (cache-friendly patterns)
- Loop tiling/blocking
- Loop unrolling
- Strength reduction
- Other relevant optimizations

Output format:
Provide the complete optimized implementation in a code block.
Do not include any other text outside of code blocks.
"""
            
            # Call LLM
            response, entire_response = self.call_llm(prompt)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(response)
            
            # Find the optimized implementation (usually the longest code block)
            optimized_code = None
            if response and code_blocks:
                optimized_code = max(code_blocks, key=len)
            
            # Save the optimized code
            if optimized_code:
                dest_path = os.path.join(cpu_impl_dir, cpp_file)
                with open(dest_path, 'w') as f:
                    f.write(optimized_code)
            
            # Store results for this file
            optimization_results[cpp_file] = {
                "original_code": original_code,
                "optimized_code": optimized_code,
                "prompt": prompt,
                "llm_response": response,
                "entire_llm_response": entire_response
            }
        
        return {
            "kernel_name": kernel_name,
            "original_files": kernel_files,
            "optimization_results": optimization_results,
            "kernel_generation_success": all(r["optimized_code"] is not None for r in optimization_results.values())
        }

    def compile_and_run(self, results: Dict, output_dir: str) -> Dict:
        """
        Compile and run both reference and optimized implementations.
        Reference implementation is run using heterobench.py.
        
        Args:
            results: The results dictionary containing optimized code
            output_dir: Directory where files are saved
            
        Returns:
            Dictionary with compilation and execution results
        """
        import subprocess
        
        if not results.get('optimization_results'):
            return {"error": "No optimized code to compile"}
        
        # Get the kernel name and files
        kernel_name = results['kernel_name']
        kernel_files = results['original_files']
        
        # Create cpu_impl/include directory
        include_dir = os.path.join(output_dir, "cpu_impl/include")
        os.makedirs(include_dir, exist_ok=True)
        
        # Copy header files to cpu_impl/include directory
        for header_file in kernel_files['headers']:
            header_name = os.path.basename(header_file)
            dest_path = os.path.join(include_dir, header_name)
            try:
                with open(header_file, 'r') as src, open(dest_path, 'w') as dst:
                    dst.write(src.read())
            except Exception as e:
                logging.error(f"Error copying header file {header_file}: {str(e)}")
                return {"error": f"Failed to copy header file: {str(e)}"}
        
        # Copy all .h and .cpp files from the original directory
        original_dir = os.path.dirname(kernel_files['main'])
        for file_name in os.listdir(original_dir):
            if file_name.endswith(('.h', '.cpp')) and file_name != 'main.cpp':
                src_path = os.path.join(original_dir, file_name)
                dest_path = os.path.join(output_dir, file_name)
                try:
                    with open(src_path, 'r') as src, open(dest_path, 'w') as dst:
                        content = src.read()
                        # If this is init_array.h, add include for cpu_impl.h
                        if file_name == 'init_array.h':
                            content = '#include "cpu_impl.h"\n' + content
                        dst.write(content)
                except Exception as e:
                    logging.error(f"Error copying file {file_name}: {str(e)}")
                    return {"error": f"Failed to copy file {file_name}: {str(e)}"}
        
        # Copy the original main.cpp to output directory
        main_cpp_path = os.path.join(output_dir, "main.cpp")
        try:
            with open(kernel_files['main'], 'r') as src, open(main_cpp_path, 'w') as dst:
                dst.write(src.read())
        except Exception as e:
            logging.error(f"Error copying main.cpp: {str(e)}")
            return {"error": f"Failed to copy main.cpp: {str(e)}"}
        
        # Copy optimized kernel files to cpu_impl directory
        cpu_impl_dir = os.path.join(output_dir, "cpu_impl")
        os.makedirs(cpu_impl_dir, exist_ok=True)
        
        # Copy optimized kernel files
        for file_name, file_results in results['optimization_results'].items():
            if file_results.get('optimized_code'):
                dest_path = os.path.join(cpu_impl_dir, file_name)
                try:
                    with open(dest_path, 'w') as f:
                        f.write(file_results['optimized_code'])
                except Exception as e:
                    logging.error(f"Error writing optimized kernel file {file_name}: {str(e)}")
                    return {"error": f"Failed to write optimized kernel file: {str(e)}"}
        
        # Compile and run
        compile_results = {
            "compilation_successful": False,
            "execution_successful": False,
            "compile_output": "",
            "compile_error": "",
            "run_output": "",
            "run_error": "",
            "reference_time": 0.0,
            "optimized_time": 0.0,
            "speedup": 0.0
        }
        
        try:
            # Get compilation flags from original Makefile
            makefile_path = os.path.join(os.path.dirname(kernel_files['main']), "Makefile")
            makefile_flags = []
            if os.path.exists(makefile_path):
                with open(makefile_path, 'r') as f:
                    makefile_content = f.read()
                    # Extract CXXFLAGS
                    cxxflags_match = re.search(r'CXXFLAGS\s*=\s*([^\n]+)', makefile_content)
                    if cxxflags_match:
                        makefile_flags = cxxflags_match.group(1).split()
            
            # Default flags if Makefile not found or no CXXFLAGS
            if not makefile_flags:
                makefile_flags = [
                    "-std=c++17", "-O3", "-march=native",
                    "-DITERATIONS=1"
                ]
            
            # Run reference implementation using heterobench.py
            ref_run_cmd = ["python", "heterobench.py", kernel_name, "run", "cpu", "serial"]
            ref_run_process = subprocess.run(
                ref_run_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if ref_run_process.returncode != 0:
                compile_results["run_error"] = f"Reference execution failed: {ref_run_process.stderr}"
                return compile_results
            
            # Extract reference time
            ref_output = ref_run_process.stdout
            ref_time_match = re.search(r'Single iteration time: ([\d.]+) ms', ref_output)
            if ref_time_match:
                compile_results["reference_time"] = float(ref_time_match.group(1))
            
            # Compile optimized implementation
            opt_cpp_files = [os.path.join(cpu_impl_dir, f) for f in os.listdir(cpu_impl_dir) if f.endswith('.cpp')]
            # Add all .cpp files from the output directory except main.cpp
            opt_cpp_files.extend([os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                                if f.endswith('.cpp') and f != 'main.cpp'])
            opt_compile_cmd = ["g++", "-o", os.path.join(output_dir, f"{kernel_name}_test"), 
                             main_cpp_path] + opt_cpp_files + makefile_flags + [
                             "-I", include_dir,  # Match Makefile include path
                             "-I", output_dir,   # Add output directory to include path
                             "-fopenmp", "-lgomp"]  # Add OpenMP flags
            
            opt_compile_process = subprocess.run(
                opt_compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            compile_results["compile_output"] = opt_compile_process.stdout
            compile_results["compile_error"] = opt_compile_process.stderr
            compile_results["compilation_successful"] = opt_compile_process.returncode == 0
            
            if compile_results["compilation_successful"]:
                # Run optimized implementation
                opt_run_process = subprocess.run(
                    [os.path.join(output_dir, f"{kernel_name}_test")],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                compile_results["run_output"] = opt_run_process.stdout
                compile_results["run_error"] = opt_run_process.stderr
                compile_results["execution_successful"] = opt_run_process.returncode == 0
                
                if compile_results["execution_successful"]:
                    # Extract optimized time
                    opt_output = opt_run_process.stdout
                    opt_time_match = re.search(r'Single iteration time: ([\d.]+) ms', opt_output)
                    if opt_time_match:
                        compile_results["optimized_time"] = float(opt_time_match.group(1))
                        if compile_results["reference_time"] > 0:
                            compile_results["speedup"] = compile_results["reference_time"] / compile_results["optimized_time"]
                
        except subprocess.TimeoutExpired:
            compile_results["compile_error"] = "Compilation or execution timeout"
        except Exception as e:
            compile_results["compile_error"] = f"Error during compilation/execution: {str(e)}"
        
        return compile_results

    def process_kernel(self, kernel_name: str, output_dir: str) -> Dict:
        """
        Complete workflow: process kernel, save results, compile and run optimized code.
        
        Args:
            kernel_name: Name of the kernel to process
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing all results including compilation and execution
        """
        logging.info(f"Processing kernel '{kernel_name}'")
        
        # Initialize status tracking
        compilation_success = False
        execution_success = False
        
        # Generate optimized code
        results = self.generate_optimized_code(kernel_name, output_dir)
        if not results.get('kernel_generation_success', False):
            logging.warning("✗ Kernel generation failed. Skipping compilation and execution.")
            results.update({
                "compilation_success": False,
                "execution_success": False,
            })
            self.save_results(results, output_dir)
            return results
        logging.info(f"✓ Kernel generation completed")
        
        # Compile and run if we have optimized code
        if results.get('optimization_results'):
            compile_results = self.compile_and_run(results, output_dir)
            results['compile_and_run'] = compile_results
            
            compilation_success = compile_results['compilation_successful']
            execution_success = compile_results['execution_successful']
            
            if compilation_success:
                logging.info(f"✓ Compilation successful")
                if execution_success:
                    logging.info(f"✓ Execution successful")

                    # Print performance analysis
                    logging.info(f"📊 Performance Analysis:")
                    if compile_results.get('run_output'):
                        logging.info(compile_results['run_output'])
                    
                    # Print reference and optimized times
                    if compile_results.get('reference_time') > 0 and compile_results.get('optimized_time') > 0:
                        logging.info(f"Reference Time: {compile_results['reference_time']:.2f} ms")
                        logging.info(f"Optimized Time: {compile_results['optimized_time']:.2f} ms")
                        logging.info(f"Speedup: {compile_results['speedup']:.2f}x")
                else:
                    logging.error(f"✗ Execution failed:")
                    logging.error(compile_results['run_error'])
            else:
                logging.error(f"✗ Compilation failed:")
                logging.error(compile_results['compile_error'])
        else:
            logging.warning("No optimized code generated to compile")
        
        # Add comprehensive status tracking
        results.update({
            "compilation_success": compilation_success,
            "execution_success": execution_success,
        })
        
        # Save results
        self.save_results(results, output_dir)
        
        # Print status summary
        logging.info("\n" + "="*60)
        logging.info("RESULTS SUMMARY")
        logging.info("="*60)
        logging.info(f"Kernel: {kernel_name}")
        logging.info(f"Files Optimized: {', '.join(results['optimization_results'].keys())}")
        logging.info(f"Optimization Success: {'✓' if results.get('kernel_generation_success', False) else '✗'}")
        logging.info(f"Compilation Success: {'✓' if compilation_success else '✗'}")
        logging.info(f"Execution Success: {'✓' if execution_success else '✗'}")
        logging.info(f"Files saved in: {output_dir}")
        logging.info("="*60)
        
        return results
    
    def save_results(self, results: Dict, output_dir: str):
        """
        Save results including all status information.
        
        Args:
            results: The complete results dictionary
            output_dir: Directory to save outputs
        """
        kernel_name = results['kernel_name']
        
        # Save prompts and responses for each file
        for file_name, file_results in results['optimization_results'].items():
            # Save the prompt
            with open(os.path.join(output_dir, f"{kernel_name}_{file_name}_prompt.txt"), 'w') as f:
                f.write(file_results['prompt'])
            
            # Save full LLM response
            with open(os.path.join(output_dir, f"{kernel_name}_{file_name}_llm_response.txt"), 'w') as f:
                f.write(file_results['llm_response'] if file_results['llm_response'] is not None else "")
        
        # Save compilation and execution logs
        if 'compile_and_run' in results:
            comp_results = results['compile_and_run']
            
            # Save compilation output
            with open(os.path.join(output_dir, f"{kernel_name}_compile_output.txt"), 'w') as f:
                f.write("COMPILATION COMMAND:\n")
                f.write(comp_results.get('compile_cmd', ''))
                f.write("\n\nSTDOUT:\n")
                f.write(comp_results.get('compile_output', ''))
                f.write("\n\nSTDERR:\n")
                f.write(comp_results.get('compile_error', ''))
            
            # Save execution output
            with open(os.path.join(output_dir, f"{kernel_name}_execution_output.txt"), 'w') as f:
                f.write("EXECUTION COMMAND:\n")
                f.write(comp_results.get('executable', ''))
                f.write("\n\nSTDOUT:\n")
                f.write(comp_results.get('run_output', ''))
                f.write("\n\nSTDERR:\n")
                f.write(comp_results.get('run_error', ''))
        
        # Save summary
        summary = {
            "kernel_name": kernel_name,
            "original_files": results.get('original_files', {}),
            "status": {
                "kernel_generation_success": results.get('kernel_generation_success', False),
                "compilation_success": results.get('compilation_success', False),
                "execution_success": results.get('execution_success', False)
            },
            "statistics": {
                "num_files_optimized": len(results.get('optimization_results', {})),
                "files_optimized": list(results.get('optimization_results', {}).keys()),
                "optimization_success": all(r["optimized_code"] is not None for r in results.get('optimization_results', {}).values())
            }
        }
        
        # Add compilation details if available
        if 'compile_and_run' in results:
            comp_results = results['compile_and_run']
            summary["compilation_details"] = {
                "compile_output_length": len(comp_results.get('compile_output', '')),
                "compile_error_length": len(comp_results.get('compile_error', '')),
                "run_output_length": len(comp_results.get('run_output', '')),
                "run_error_length": len(comp_results.get('run_error', ''))
            }
            # Add speedup data if available
            if comp_results.get('speedup', 0) > 0:
                summary["speedup"] = comp_results['speedup']
        
        with open(os.path.join(output_dir, f"{kernel_name}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)