import os
import re
from typing import Dict, List, Tuple, Optional
import json
import logging
from llm import LLM


class KernelCodeGenerator:
    """
    A framework for using LLMs to generate optimized kernel code using Intel MKL library.
    """
    
    def __init__(self, llm: LLM):
        """
        Initialize the kernel code generator.
        
        Args:
            llm: LLM instance configured with provider and model
        """
        self.llm = llm
        
    def create_prompt(self, complete_code: str, kernel_func_code: str, optimized_func_signature: str, kernel_name: str) -> str:
        """
        Create a prompt for the LLM to analyze and optimize kernel using MKL.
        
        Args:
            complete_code: The original complete code
            kernel_func_code: The original kernel code
            optimized_func_signature: The optimized signature
            kernel_name: Name of the kernel (e.g., gemm, syrk)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert in high-performance computing and kernel engineering on the CPU. You are familiar with different optimized libraries and their performance characteristics, including Intel MKL. You also know the performance improvement techniques, including vectorization, memory access optimization, tiling, unrolling, etc. Here we focus on the single-threaded performance improvement. Don't use multi-threading. 

Given the following code `{kernel_name}.c`:
```c
{complete_code}
```

with the following kernel implementation: 
```c
{kernel_func_code}
```

Task: Analyze this kernel and generate an optimized kernel implementation to get better performance while maintaining functional equivalence. You should first consider if the kernel can be implemented using a single corresponding MKL function. If not, consider if the kernel can be decomposed into multiple MKL calls. If there is no way to implement the kernel using MKL, or you think MKL cannot get good performance, then consider applying optimizations directly to the kernel, such as vectorization, memory access optimization, tiling, unrolling, etc. You should only use single thread for the optimized kernel implementation. 

Machine we are using: 
- Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
- L1d cache: 1.5MB
- L1i cache: 1.5MB
- L2 cache: 48MB
- L3 cache: 71.5MB
- Supports SSE, AVX2, AVX512

Requirements:
1. Identify which a single MKL function or a combination of MKL functions can replace this kernel to get better performance.
2. Provide the equivalent MKL implementation. 
3. Include necessary headers and initialization code. 
4. Ensure functional equivalence.
5. If there is no available MKL function that can get good performance, then consider applying optimizations directly to the kernel, such as vectorization, memory access optimization, tiling, unrolling, etc.
6. If neither MKL nor optimizations can get good performance, then just fallback to the default implementation.

Output format:
You should only output the optimized kernel implementation which follows the exact function signature as follows: 
```c
{optimized_func_signature}
```

Do not include any other text other than the optimized kernel implementation. ONLY output the optimized kernel implementation within the code block. 

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
        system_prompt = "You are an expert in high-performance computing, kernel optimization, and Intel MKL library."
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
    
    def _extract_function_signature_from_code(self, complete_code: str, function_name: str) -> str:
        """
        Extract just the function signature (declaration) from C code.
        
        Args:
            complete_code: The complete C source code
            function_name: Name of the function to extract signature for
            
        Returns:
            The function signature as a string (up to the opening brace)
        """
        # Split code into lines for easier processing
        lines = complete_code.split('\n')
        
        # Find the line containing the function name
        function_start_line = -1
        for i, line in enumerate(lines):
            if function_name in line and '(' in line:
                function_start_line = i
                break
        
        if function_start_line == -1:
            return ""
        
        # Look backwards to find the return type
        signature_start_line = function_start_line
        for i in range(function_start_line, -1, -1):
            line_stripped = lines[i].strip()
            if (line_stripped.startswith('void') or 
                line_stripped.startswith('int') or
                line_stripped.startswith('double') or
                line_stripped.startswith('float') or
                line_stripped.startswith('char') or
                line_stripped.startswith('static')):
                signature_start_line = i
                break
        
        # Extract lines from return type to the opening brace
        signature_lines = []
        for i in range(signature_start_line, len(lines)):
            line = lines[i]
            signature_lines.append(line)
            
            # Stop when we reach the opening brace
            if '{' in line:
                # Remove the opening brace and everything after it
                brace_pos = line.find('{')
                last_line = line[:brace_pos].rstrip()
                signature_lines[-1] = last_line
                break
        
        # Join the lines and clean up
        signature = '\n'.join(signature_lines).strip()
        return signature

    def _extract_function_from_code(self, complete_code: str, function_name: str) -> str:
        """
        Extract a specific function implementation from C code.
        
        Args:
            complete_code: The complete C source code
            function_name: Name of the function to extract (e.g., 'kernel_gemm')
            
        Returns:
            The function implementation as a string
        """
        # Pattern to match function definition and its body
        # This handles multi-line function signatures and nested braces
        pattern = rf'({function_name}\s*\([^{{]*\{{\s*.*?\n}})'
        
        # Search for the function with DOTALL flag to match across newlines
        match = re.search(pattern, complete_code, re.DOTALL)
        
        if match:
            function_code = match.group(1)
            
            # Count braces to find the complete function body
            open_braces = 0
            function_start = complete_code.find(match.group(0))
            
            # Find the actual start of the function (including return type and signature)
            lines = complete_code[:function_start + len(match.group(0))].split('\n')
            
            # Look backwards to find the function signature start
            for i in range(len(lines) - 1, -1, -1):
                if function_name in lines[i]:
                    # Found the line with function name, now find the return type
                    j = i
                    while j >= 0 and not (lines[j].strip().startswith('void') or 
                                         lines[j].strip().startswith('int') or
                                         lines[j].strip().startswith('double') or
                                         lines[j].strip().startswith('float') or
                                         lines[j].strip().startswith('char') or
                                         lines[j].strip().startswith('static')):
                        j -= 1
                    if j >= 0:
                        function_start_line = j
                        break
            
            # Extract from function start to end
            lines_from_start = complete_code.split('\n')[function_start_line:]
            result_lines = []
            open_braces = 0
            found_opening_brace = False
            
            for line in lines_from_start:
                result_lines.append(line)
                
                # Count braces
                for char in line:
                    if char == '{':
                        open_braces += 1
                        found_opening_brace = True
                    elif char == '}':
                        open_braces -= 1
                
                # If we found the opening brace and braces are balanced, we're done
                if found_opening_brace and open_braces == 0:
                    break
            
            return '\n'.join(result_lines)
        
        return ""

    def _replace_function_in_code(self, complete_code: str, function_name: str, new_implementation: str) -> str:
        """
        Replace a function implementation in the complete code with a new implementation.
        
        Args:
            complete_code: The complete C source code
            function_name: Name of the function to replace
            new_implementation: The new function implementation
            
        Returns:
            The updated complete code with replaced function
        """
        lines = complete_code.split('\n')
        
        # Find the function start and end
        function_start_line = -1
        function_end_line = -1
        
        # Find the line containing the function name
        for i, line in enumerate(lines):
            if function_name in line and '(' in line:
                # Look backwards to find the return type
                for j in range(i, -1, -1):
                    line_stripped = lines[j].strip()
                    if (line_stripped.startswith('void') or 
                        line_stripped.startswith('int') or
                        line_stripped.startswith('double') or
                        line_stripped.startswith('float') or
                        line_stripped.startswith('char') or
                        line_stripped.startswith('static')):
                        function_start_line = j
                        break
                break
        
        if function_start_line == -1:
            return complete_code  # Function not found, return original
        
        # Find the end of the function by counting braces
        open_braces = 0
        found_opening_brace = False
        
        for i in range(function_start_line, len(lines)):
            line = lines[i]
            
            # Count braces
            for char in line:
                if char == '{':
                    open_braces += 1
                    found_opening_brace = True
                elif char == '}':
                    open_braces -= 1
            
            # If we found the opening brace and braces are balanced, we found the end
            if found_opening_brace and open_braces == 0:
                function_end_line = i
                break
        
        if function_end_line == -1:
            return complete_code  # Could not find function end, return original
        
        # Replace the function
        new_lines = (lines[:function_start_line] + 
                    [new_implementation] + 
                    lines[function_end_line + 1:])
        
        return '\n'.join(new_lines)



    def generate_kernel(self, file_path: str, kernel_name: str, output_dir: str) -> Dict:
        """
        Generate an optimized kernel implementation through the LLM pipeline by reading from a file.
        
        Args:
            file_path: Path to the C source file
            kernel_name: Name of the kernel. The function name in the code is `kernel_{kernel_name}`
            output_dir: Directory to save outputs
        Returns:
            Dictionary containing the results
        """
        
        # Read the complete code from file
        try:
            with open(file_path, 'r') as f:
                complete_code = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
        
        # kernel name: `-` -> `_`
        kernel_name = kernel_name.replace('-', '_')

        # Extract the kernel function implementation
        kernel_func_name = f"kernel_{kernel_name}"
        kernel_func_code = self._extract_function_from_code(complete_code, kernel_func_name)
        
        if not kernel_func_code:
            raise ValueError(f"Could not find function '{kernel_func_name}' in {file_path}")
        
        # Extract the optimized function signature
        optimized_func_name = f"kernel_{kernel_name}_optimized"
        optimized_func_signature = self._extract_function_signature_from_code(complete_code, optimized_func_name)

        if not optimized_func_signature:
            raise ValueError(f"Could not find function '{optimized_func_name}' in {file_path}")
        
        # Create prompt
        prompt = self.create_prompt(complete_code, kernel_func_code, optimized_func_signature, kernel_name)

        # Call LLM
        response, entire_response = self.call_llm(prompt)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(response)
        
        # Find the main MKL implementation (usually the longest code block)
        optimized_code_generated = None
        kernel_generation_success = False
        if response and code_blocks:
            optimized_code_generated = max(code_blocks, key=len)
            kernel_generation_success = True
        
        # Generate optimized complete code by replacing the optimized function
        optimized_complete_code = complete_code
        if optimized_code_generated:
            optimized_complete_code = self._replace_function_in_code(
                complete_code, optimized_func_name, optimized_code_generated
            )
        
        # save the optimized complete code
        optimized_file_path = os.path.join(output_dir, f"{kernel_name}_optimized.c")
        with open(optimized_file_path, 'w') as f:
            f.write(optimized_complete_code)
        
        return {
            "kernel_name": kernel_name,
            "original_file_path": file_path,
            "prompt": prompt,
            "llm_response": response,
            "entire_llm_response": entire_response,
            "optimized_code_generated": optimized_code_generated,
            "num_code_blocks": len(code_blocks),
            "optimized_complete_code": optimized_complete_code,
            "optimized_file_path": optimized_file_path,
            "kernel_generation_success": kernel_generation_success
        }
    
    
    def save_results(self, results: Dict, output_dir: str):
        """
        Save results including all status information.
        
        Args:
            results: The complete results dictionary
            output_dir: Directory to save outputs
        """
        kernel_name = results['kernel_name']
        
        # Save the prompt
        with open(os.path.join(output_dir, f"{kernel_name}_prompt.txt"), 'w') as f:
            f.write(results['prompt'])
        
        # Save full LLM response
        with open(os.path.join(output_dir, f"{kernel_name}_llm_response.txt"), 'w') as f:
            f.write(results['llm_response'] if results['llm_response'] is not None else "")
        
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
            "original_file_path": results.get('original_file_path', ''),
            "optimized_file_path": results.get('optimized_file_path', ''),
            "status": {
                "kernel_generation_success": results.get('kernel_generation_success', False),
                "compilation_success": results.get('compilation_success', False),
                "execution_success": results.get('execution_success', False),
                "verification_success": results.get('verification_success', False)
            },
            "statistics": {
                "num_code_blocks_generated": results.get('num_code_blocks', 0),
                "optimized_code_length": len(results.get('optimized_code_generated') or ''),
                "prompt_length": len(results.get('prompt', '')),
                "llm_response_length": len(results.get('llm_response') or '')
            }, 
            "details": {
                "entire_llm_response": results.get('entire_llm_response', '')
            }
        }
        
        # Add run analysis if available
        if 'run_analysis' in results:
            run_analysis = results['run_analysis']
            
            summary["performance_analysis"] = {
                "verification_success": run_analysis.get('verification_success', False),
                "original_time_seconds": run_analysis.get('original_time'),
                "optimized_time_seconds": run_analysis.get('optimized_time'),
                "speedup": run_analysis.get('speedup'),
            }
        
        # Add compilation details if available
        if 'compile_and_run' in results:
            comp_results = results['compile_and_run']
            summary["compilation_details"] = {
                "source_file": comp_results.get('source_file', ''),
                "executable": comp_results.get('executable', ''),
                "compile_output_length": len(comp_results.get('compile_output', '')),
                "compile_error_length": len(comp_results.get('compile_error', '')),
                "run_output_length": len(comp_results.get('run_output', '')),
                "run_error_length": len(comp_results.get('run_error', ''))
            }
        
        with open(os.path.join(output_dir, f"{kernel_name}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print status summary
        logging.info("\n" + "="*60)
        logging.info("RESULTS SUMMARY")
        logging.info("="*60)
        logging.info(f"Kernel: {kernel_name}")
        logging.info(f"Kernel Generation Success: {'✓' if summary['status']['kernel_generation_success'] else '✗'}")
        logging.info(f"Compilation Success: {'✓' if summary['status']['compilation_success'] else '✗'}")
        logging.info(f"Execution Success: {'✓' if summary['status']['execution_success'] else '✗'}")
        logging.info(f"Verification Success: {'✓' if summary['status']['verification_success'] else '✗'}")
        
        # Print performance summary if available
        if 'performance_analysis' in summary:
            perf = summary['performance_analysis']
            if perf['speedup'] is not None:
                logging.info(f"Speedup: {perf['speedup']:.2f}x")
        
        logging.info(f"Files saved in: {output_dir}")
        logging.info("="*60)

    def compile_and_run(self, results: Dict, output_dir: str) -> Dict:
        """
        Compile and run the optimized code.
        
        Args:
            results: The results dictionary containing optimized code
            output_dir: Directory where files are saved
            
        Returns:
            Dictionary with compilation and execution results
        """
        import subprocess
        
        if not results.get('optimized_complete_code'):
            return {"error": "No optimized code to compile"}
        
        # Get the optimized file path
        optimized_file_path = results.get('optimized_file_path')
        if not optimized_file_path:
            optimized_file_path = os.path.join(output_dir, f"{results['kernel_name']}_optimized.c")
        
        # Executable path
        executable_path = os.path.join(output_dir, f"{results['kernel_name']}_optimized")
        
        compile_results = {
            "source_file": optimized_file_path,
            "executable": executable_path,
            "compilation_successful": False,
            "execution_successful": False,
            "compile_output": "",
            "compile_error": "",
            "run_output": "",
            "run_error": ""
        }
        
        try:
            # Compile the code
            compile_cmd = ["gcc", "-o", executable_path, optimized_file_path, "-I", "${MKL_ROOT}/include", "-L", "${MKL_ROOT}/lib/intel64", "-lmkl_intel_lp64", "-lmkl_sequential", "-lmkl_core", "-lpthread", "-lm", "-march=native"]
            
            compile_process = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            compile_results["compile_cmd"] = " ".join(compile_cmd)
            compile_results["compile_output"] = compile_process.stdout
            compile_results["compile_error"] = compile_process.stderr
            compile_results["compilation_successful"] = compile_process.returncode == 0
            
            if compile_results["compilation_successful"]:
                # Run the executable
                run_process = subprocess.run(
                    [executable_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                compile_results["run_output"] = run_process.stdout
                compile_results["run_error"] = run_process.stderr
                compile_results["execution_successful"] = run_process.returncode == 0
                
        except subprocess.TimeoutExpired:
            compile_results["compile_error"] = "Compilation or execution timeout"
        except Exception as e:
            compile_results["compile_error"] = f"Error during compilation/execution: {str(e)}"
        
        return compile_results

    def _analyze_run_output(self, run_output: str) -> Dict:
        """
        Analyze the run output to extract verification results, speedup, and timing information.
        
        Args:
            run_output: The stdout from running the program
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "verification_success": False,
            "original_time": None,
            "optimized_time": None, 
            "speedup": None,
        }
        
        lines = run_output.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for verification results
            if "Verification Results:" in line:
                # Check next few lines for PASS/FAIL
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    if "PASS:" in next_line:
                        analysis["verification_success"] = True
                        break
                    elif "FAIL:" in next_line:
                        analysis["verification_success"] = False
                        break
            
            # Look for performance results
            elif "Performance Results:" in line:
                # Look for timing information in next few lines
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    
                    # Extract original kernel time
                    if "Original kernel average time:" in next_line:
                        try:
                            # Extract time value (format: "Original kernel average time: X.XXXXXX seconds")
                            time_str = next_line.split(":")[-1].replace("seconds", "").strip()
                            analysis["original_time"] = float(time_str)
                        except (ValueError, IndexError):
                            pass
                    
                    # Extract optimized kernel time
                    elif "Optimized kernel average time:" in next_line:
                        try:
                            # Extract time value (format: "Optimized kernel average time: X.XXXXXX seconds")
                            time_str = next_line.split(":")[-1].replace("seconds", "").strip()
                            analysis["optimized_time"] = float(time_str)
                        except (ValueError, IndexError):
                            pass
                    
                    # Extract speedup
                    elif "Speedup:" in next_line:
                        try:
                            # Extract speedup value (format: "Speedup: X.XXXXXX")
                            speedup_str = next_line.split(":")[-1].strip()
                            analysis["speedup"] = float(speedup_str)
                        except (ValueError, IndexError):
                            pass
        
        return analysis

    def process_kernel(self, file_path: str, kernel_name: str, output_dir: str) -> Dict:
        """
        Complete workflow: process kernel, save results, compile and run optimized code.
        
        Args:
            file_path: Path to the C source file
            kernel_name: Name of the kernel
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing all results including compilation and execution
        """
        logging.info(f"Processing kernel '{kernel_name}' from file: {file_path}")
        
        # Initialize status tracking
        compilation_success = False
        execution_success = False
        verification_success = False
        
        # Process the kernel
        results = self.generate_kernel(file_path, kernel_name, output_dir)
        if not results.get('kernel_generation_success', False):
            logging.warning("✗ Kernel generation failed. Skipping compilation and execution.")
            results.update({
                "compilation_success": False,
                "execution_success": False,
                "verification_success": False,
            })
            self.save_results(results, output_dir)
            return results
        logging.info(f"✓ Kernel generation completed")
        
        # Compile and run if we have optimized code
        if results.get('optimized_complete_code'):
            compile_results = self.compile_and_run(results, output_dir)
            results['compile_and_run'] = compile_results
            
            compilation_success = compile_results['compilation_successful']
            execution_success = compile_results['execution_successful']
            
            # Analyze run output if execution was successful
            run_analysis = {}
            if execution_success and compile_results.get('run_output'):
                run_analysis = self._analyze_run_output(compile_results['run_output'])
                results['run_analysis'] = run_analysis
            
            if compilation_success:
                logging.info(f"✓ Compilation successful")
                if execution_success:
                    logging.info(f"✓ Execution successful")
                    
                    # print verification results
                    verification_success = run_analysis.get('verification_success')
                    if verification_success:
                        logging.info(f"🔍 Verification: ✓ PASS")
                    else:
                        logging.warning(f"🔍 Verification: ✗ FAIL")

                    # Print performance analysis
                    logging.info(f"📊 Performance Analysis:")
                    if run_analysis.get('original_time') is not None:
                        logging.info(f"  Original time: {run_analysis['original_time']:.6f} seconds")
                    if run_analysis.get('optimized_time') is not None:
                        logging.info(f"  Optimized time: {run_analysis['optimized_time']:.6f} seconds")
                    if run_analysis.get('speedup') is not None:
                        logging.info(f"  Speedup: {run_analysis['speedup']:.2f}x")
                    
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
            "verification_success": verification_success,
        })
        
        # Save results
        self.save_results(results, output_dir)
        
        return results