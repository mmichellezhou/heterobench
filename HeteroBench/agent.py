import os
import re
from typing import Dict, List, Tuple, Optional
import json
import logging
from llm import LLM


class HeteroBenchCodeGenerator:
    """
    A framework for using LLMs to generate optimized kernel code for HeteroBench.
    Processes multiple files in cpu_impl and optimizes individual functions.
    """
    
    def __init__(self, llm: LLM):
        """
        Initialize the HeteroBench code generator.
        
        Args:
            llm: LLM instance configured with provider and model
        """
        self.llm = llm
        
    def create_prompt(self, complete_code: str, function_code: str, optimized_func_signature: str, function_name: str) -> str:
        """
        Create a prompt for the LLM to analyze and optimize a specific function.
        
        Args:
            complete_code: The original complete code from the file
            function_code: The original function code to optimize
            optimized_func_signature: The optimized function signature
            function_name: Name of the function to optimize
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert in high-performance computing and kernel engineering on the CPU. You are familiar with different optimization techniques including vectorization, memory access optimization, tiling, unrolling, loop transformations, and SIMD instructions. Focus on single-threaded performance improvement. Don't use multi-threading nor vectorization.

Given the following code:
```cpp
{complete_code}
```

with the following function to optimize: 
```cpp
{function_code}
```

Task: Analyze this function and generate an optimized implementation to get better performance while maintaining functional equivalence. Apply optimizations such as memory access optimization, tiling, unrolling, loop transformations, strength reduction, register optimization, and instruction-level parallelism.

Machine we are using: 
- Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
- L1d cache: 1.5MB
- L1i cache: 1.5MB
- L2 cache: 48MB
- L3 cache: 71.5MB
- Supports SSE, AVX2, AVX512

Requirements:
1. Optimize the function for better single-threaded performance
2. Maintain exact functional equivalence
3. Use appropriate compiler intrinsics and optimizations
4. Focus on the most impactful optimizations for this specific function
5. Only use variables, constants, and types that are already available in the function scope
6. Do not define any constants, macros, or types that would need to be defined outside the function body

Output format:
You should only output the optimized function implementation which follows the exact function signature as follows: 
```cpp
{optimized_func_signature}
```

Do not include any other text other than the optimized function implementation. ONLY output the optimized function implementation within the code block.
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
        system_prompt = "You are an expert in high-performance computing, kernel optimization, and CPU performance tuning."
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
    
    def _extract_function_from_code(self, complete_code: str, function_name: str) -> str:
        """
        Extract a specific function implementation from C++ code.
        
        Args:
            complete_code: The complete C++ source code
            function_name: Name of the function to extract
            
        Returns:
            The function implementation as a string
        """
        # Split code into lines for easier processing
        lines = complete_code.split('\n')
        
        # Find the function by looking for the function name followed by parentheses
        function_start_line = -1
        function_end_line = -1
        
        for i, line in enumerate(lines):
            # Look for function name followed by parentheses (with possible whitespace)
            if re.search(rf'\b{re.escape(function_name)}\s*\(', line):
                function_start_line = i
                break
        
        if function_start_line == -1:
            return ""
        
        # Look backwards to find the return type (start of function signature)
        signature_start_line = function_start_line
        for i in range(function_start_line, -1, -1):
            line_stripped = lines[i].strip()
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                continue
            # If this line contains the function name, we've found the signature start
            if function_name in line_stripped:
                signature_start_line = i
                break
            # If this line looks like it could be part of a return type (contains alphanumeric chars)
            if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', line_stripped):
                signature_start_line = i
            else:
                # If we hit a line that doesn't look like part of the signature, stop
                break
        
        # Find the end of the function by counting braces
        open_braces = 0
        found_opening_brace = False
        
        for i in range(signature_start_line, len(lines)):
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
            return ""
        
        # Extract the function
        function_lines = lines[signature_start_line:function_end_line + 1]
        return '\n'.join(function_lines)

    def _extract_function_body(self, function_code: str) -> str:
        """
        Extract just the function body (between braces) from a function.
        
        Args:
            function_code: The complete function code
            
        Returns:
            The function body as a string
        """
        # Find the first opening brace
        start = function_code.find('{')
        if start == -1:
            return ""
        
        # Count braces to find the matching closing brace
        open_braces = 0
        for i, char in enumerate(function_code[start:], start):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
                if open_braces == 0:
                    # Return everything between the braces (excluding the braces themselves)
                    return function_code[start + 1:i].strip()
        
        return ""

    def _extract_function_body_from_response(self, response_code: str) -> str:
        """
        Extract just the function body from LLM response, handling cases where
        the LLM returns a complete function instead of just the body.
        
        Args:
            response_code: The code returned by the LLM
            
        Returns:
            The function body as a string (code between braces)
        """
        # First try to extract function body using the same method as for functions
        body = self._extract_function_body(response_code)
        if body:
            return body
        
        # If that fails, the response might already be just the function body
        # Check if it starts with a brace or looks like function body code
        lines = response_code.strip().split('\n')
        
        # If the first line doesn't contain function signature patterns, 
        # assume it's already just the function body
        first_line = lines[0].strip()
        if not any(keyword in first_line for keyword in ['void', 'int', 'double', 'float', 'char', 'static', 'template', 'inline']):
            return response_code.strip()
        
        # If we still can't extract, return the original response
        return response_code.strip()

    def _replace_function_body_in_complete_code(self, complete_code: str, function_name: str, new_body: str) -> str:
        """
        Replace the function body of a specific function in the complete code.
        
        Args:
            complete_code: The complete source code
            function_name: Name of the function to replace body for
            new_body: The new function body (code between braces)
            
        Returns:
            The complete code with the function body replaced
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
                        line_stripped.startswith('static') or
                        line_stripped.startswith('template') or
                        line_stripped.startswith('inline')):
                        function_start_line = j
                        break
                break
        
        if function_start_line == -1:
            return complete_code  # Function not found, return original
        
        # Find the end of the function by counting braces
        open_braces = 0
        found_opening_brace = False
        body_start_line = -1
        body_end_line = -1
        
        for i in range(function_start_line, len(lines)):
            line = lines[i]
            
            # Count braces
            for char in line:
                if char == '{':
                    open_braces += 1
                    found_opening_brace = True
                    if body_start_line == -1:
                        body_start_line = i
                elif char == '}':
                    open_braces -= 1
                    if found_opening_brace and open_braces == 0:
                        body_end_line = i
                        break
            
            # If we found the opening brace and braces are balanced, we found the end
            if found_opening_brace and open_braces == 0:
                function_end_line = i
                break
        
        if function_end_line == -1 or body_start_line == -1:
            return complete_code  # Could not find function end, return original
        
        # Replace only the function body (between the braces)
        # Keep the opening brace line but replace its content
        opening_brace_line = lines[body_start_line]
        closing_brace_line = lines[body_end_line]
        
        # Find the position of the opening brace in the line
        brace_pos = opening_brace_line.find('{')
        if brace_pos != -1:
            # Keep everything before the brace, add the new body, then the closing brace
            new_lines = (lines[:body_start_line] + 
                        [opening_brace_line[:brace_pos + 1]] +
                        [new_body] +
                        [closing_brace_line] +
                        lines[body_end_line + 1:])
        else:
            # Fallback: replace the entire body lines
            new_lines = (lines[:body_start_line] + 
                        [new_body] + 
                        lines[body_end_line + 1:])
        
        return '\n'.join(new_lines)

    def _replace_function_body(self, original_function: str, new_body: str) -> str:
        """
        Replace the function body in a function with a new body.
        
        Args:
            original_function: The original function code
            new_body: The new function body
            
        Returns:
            The function with replaced body
        """
        # Find the first opening brace
        start = original_function.find('{')
        if start == -1:
            return original_function
        
        # Find the matching closing brace
        open_braces = 0
        for i, char in enumerate(original_function[start:], start):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
                if open_braces == 0:
                    # Replace the body
                    return original_function[:start + 1] + '\n' + new_body + '\n' + original_function[i:]
        
        return original_function

    def generate_optimized_function(self, file_path: str, function_name: str, benchmark_name: str) -> Dict:
        """
        Generate an optimized function implementation through the LLM pipeline.
        
        Args:
            file_path: Path to the C++ source file in cpu_impl
            function_name: Name of the function to optimize (same as filename without extension)
            benchmark_name: Name of the benchmark
            
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
        
        # Extract the function implementation
        function_code = self._extract_function_from_code(complete_code, function_name)
        
        if not function_code:
            raise ValueError(f"Could not find function '{function_name}' in {file_path}")
        
        # Get the optimized file path
        original_dir = os.path.dirname(file_path)
        optimized_dir = os.path.join(os.path.dirname(original_dir), "cpu_impl_optimized")
        optimized_file_path = os.path.join(optimized_dir, f"{function_name}_optimized.cpp")
        
        # Read the existing optimized file
        try:
            with open(optimized_file_path, 'r') as f:
                optimized_file_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find optimized file: {optimized_file_path}")
        except Exception as e:
            raise Exception(f"Error reading optimized file {optimized_file_path}: {str(e)}")
        
        # Extract the optimized function signature
        optimized_function_name = f"{function_name}_optimized"
        optimized_func_signature = self._extract_function_signature_from_code(optimized_file_content, optimized_function_name)

        if not optimized_func_signature:
            raise ValueError(f"Could not find function '{optimized_function_name}' in {optimized_file_path}")
        
        # Create prompt
        prompt = self.create_prompt(complete_code, function_code, optimized_func_signature, function_name)

        # Call LLM
        response, entire_response = self.call_llm(prompt)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(response)
        
        # Find the main implementation (usually the longest code block)
        optimized_code_generated = None
        function_generation_success = False
        if response and code_blocks:
            optimized_code_generated = max(code_blocks, key=len)
            function_generation_success = True
        
        # Generate optimized complete code by replacing the optimized function
        optimized_complete_code = optimized_file_content
        if optimized_code_generated:
            optimized_complete_code = self._replace_function_in_code(
                optimized_file_content, optimized_function_name, optimized_code_generated
            )
        
        # Save the updated optimized file
        os.makedirs(optimized_dir, exist_ok=True)
        with open(optimized_file_path, 'w') as f:
            f.write(optimized_complete_code)
        
        return {
            "function_name": function_name,
            "original_file_path": file_path,
            "prompt": prompt,
            "llm_response": response,
            "entire_llm_response": entire_response,
            "optimized_code_generated": optimized_code_generated,
            "num_code_blocks": len(code_blocks),
            "optimized_complete_code": optimized_complete_code,
            "optimized_file_path": optimized_file_path,
            "function_generation_success": function_generation_success
        }
    
    def save_results(self, results: Dict, output_dir: str, benchmark_name: str):
        """
        Save results including all status information.
        
        Args:
            results: The complete results dictionary
            output_dir: Directory to save outputs
            benchmark_name: Name of the benchmark
        """
        function_name = results['function_name']
        
        # Save the prompt
        with open(os.path.join(output_dir, f"{benchmark_name}_{function_name}_prompt.txt"), 'w') as f:
            f.write(results['prompt'])
        
        # Save full LLM response
        with open(os.path.join(output_dir, f"{benchmark_name}_{function_name}_llm_response.txt"), 'w') as f:
            f.write(results['llm_response'] if results['llm_response'] is not None else "")
        
        # Save compilation and execution logs
        if 'compile_and_run' in results:
            comp_results = results['compile_and_run']
            
            # Save compilation output
            with open(os.path.join(output_dir, f"{benchmark_name}_{function_name}_compile_output.txt"), 'w') as f:
                f.write("COMPILATION COMMAND:\n")
                f.write(comp_results.get('compile_cmd', ''))
                f.write("\n\nSTDOUT:\n")
                f.write(comp_results.get('compile_output', ''))
                f.write("\n\nSTDERR:\n")
                f.write(comp_results.get('compile_error', ''))
            
            # Save execution output
            with open(os.path.join(output_dir, f"{benchmark_name}_{function_name}_execution_output.txt"), 'w') as f:
                f.write("EXECUTION COMMAND:\n")
                f.write(comp_results.get('executable', ''))
                f.write("\n\nSTDOUT:\n")
                f.write(comp_results.get('run_output', ''))
                f.write("\n\nSTDERR:\n")
                f.write(comp_results.get('run_error', ''))
        
        # Save summary
        summary = {
            "function_name": function_name,
            "original_file_path": results.get('original_file_path', ''),
            "optimized_file_path": results.get('optimized_file_path', ''),
            "status": {
                "function_generation_success": results.get('function_generation_success', False),
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
        
        with open(os.path.join(output_dir, f"{benchmark_name}_{function_name}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print status summary
        logging.info("\n" + "="*60)
        logging.info("RESULTS SUMMARY")
        logging.info("="*60)
        logging.info(f"Function: {function_name}")
        logging.info(f"Function Generation Success: {'✓' if summary['status']['function_generation_success'] else '✗'}")
        
        # Only show compilation/execution status if they were actually set
        if 'compilation_success' in results:
            logging.info(f"Compilation Success: {'✓' if summary['status']['compilation_success'] else '✗'}")
        if 'execution_success' in results:
            logging.info(f"Execution Success: {'✓' if summary['status']['execution_success'] else '✗'}")
        if 'verification_success' in results:
            logging.info(f"Verification Success: {'✓' if summary['status']['verification_success'] else '✗'}")
        
        # Print performance summary if available
        if 'performance_analysis' in summary:
            perf = summary['performance_analysis']
            if perf['speedup'] is not None:
                logging.info(f"Speedup: {perf['speedup']:.2f}x")
        
        logging.info(f"Files saved in: {output_dir}")
        logging.info("="*60)

    def determine_verification_success(self, run_output: str) -> bool:
        """
        Determine if verification was successful based on benchmark output.
        If failure keywords are found, verification fails; otherwise assumes success.
        
        Args:
            run_output: The output from running the benchmark
            
        Returns:
            True if verification passed, False otherwise
        """
        output_lower = run_output.lower()
        
        # Check for failure keywords
        fail_keywords = ["incorrect", "fail", "wrong"]
        if any(keyword in output_lower for keyword in fail_keywords):
            return False
        
        # If no failure indicators found, assume success
        return True

    def _analyze_run_output(self, run_output: str) -> Dict:
        """
        Analyze the run output to extract verification results, speedup, and timing information.
        
        Args:
            run_output: The stdout from running the program
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "verification_success": self.determine_verification_success(run_output),
            "original_time": None,
            "optimized_time": None, 
            "speedup": None,
        }
        
        lines = run_output.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for performance results
            if "Speedup:" in line:
                try:
                    # Extract speedup value (format: "Speedup: X.XXXXXX")
                    speedup_str = line.split(":")[-1].strip()
                    analysis["speedup"] = float(speedup_str)
                except (ValueError, IndexError):
                    pass
            
            # Look for timing information
            elif "Original Implementation:" in line:
                # Look for timing in next few lines
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
                # Look for timing in next few lines
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    if "Single iteration time:" in next_line:
                        try:
                            time_str = next_line.split(":")[-1].replace("seconds", "").strip()
                            analysis["optimized_time"] = float(time_str)
                        except (ValueError, IndexError):
                            pass
                        break
        
        return analysis

    def process_benchmark(self, benchmark_name: str, benchmark_path: str, output_dir: str) -> Dict:
        """
        Process all functions in a benchmark's cpu_impl directory.
        
        Args:
            benchmark_name: Name of the benchmark
            benchmark_path: Path to the benchmark directory
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing results for all processed functions
        """
        cpu_impl_path = os.path.join(benchmark_path, "homobackend_cpu", "Cpp", "cpu_impl")
        
        if not os.path.exists(cpu_impl_path):
            raise FileNotFoundError(f"CPU implementation directory not found: {cpu_impl_path}")
        
        # Load skip files configuration
        config_path = os.path.join(os.path.dirname(__file__), "config_json", "opt_config.json")
        skip_files = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    skip_files = config.get("skip_files", {})
            except Exception as e:
                logging.warning(f"Could not load optimization config: {e}")
        
        # Get all .cpp files in cpu_impl
        cpp_files = [f for f in os.listdir(cpu_impl_path) if f.endswith('.cpp')]
        
        # Filter out files that should be skipped for this benchmark
        files_to_skip = skip_files.get(benchmark_name, [])
        cpp_files = [f for f in cpp_files if f not in files_to_skip]
                
        all_results = {}
        
        for cpp_file in cpp_files:
            function_name = os.path.splitext(cpp_file)[0]  # Remove .cpp extension
            file_path = os.path.join(cpu_impl_path, cpp_file)
            
            logging.info(f"Processing function '{function_name}' from file: {file_path}")
            
            # Process the function
            results = self.generate_optimized_function(file_path, function_name, benchmark_name)
            
            # Save results for this function with benchmark prefix
            self.save_results(results, output_dir, benchmark_name)
            
            # Collect status for each function
            function_summary = {
                "function_generation_success": results["function_generation_success"],
            }
            
            # Add speedup if available
            if "run_analysis" in results and results["run_analysis"].get("speedup") is not None:
                function_summary["speedup"] = results["run_analysis"]["speedup"]
            
            all_results[function_name] = function_summary
        
        # Save aggregated results
        all_summary_path = os.path.join(output_dir, f"{benchmark_name}_all_summary.json")
        with open(all_summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Summary saved to {all_summary_path}")
        
        return all_results

    def _extract_function_signature_from_code(self, complete_code: str, function_name: str) -> str:
        """
        Extract just the function signature (declaration) from C++ code.
        
        Args:
            complete_code: The complete C++ source code
            function_name: Name of the function to extract signature for
            
        Returns:
            The function signature as a string (up to the opening brace)
        """
        # Split code into lines for easier processing
        lines = complete_code.split('\n')
        
        # Find the function by looking for the function name followed by parentheses
        function_start_line = -1
        
        for i, line in enumerate(lines):
            # Look for function name followed by parentheses (with possible whitespace)
            if re.search(rf'\b{re.escape(function_name)}\s*\(', line):
                function_start_line = i
                break
        
        if function_start_line == -1:
            return ""
        
        # Look backwards to find the return type (start of function signature)
        signature_start_line = function_start_line
        for i in range(function_start_line, -1, -1):
            line_stripped = lines[i].strip()
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                continue
            # If this line contains the function name, we've found the signature start
            if function_name in line_stripped:
                signature_start_line = i
                break
            # If this line looks like it could be part of a return type (contains alphanumeric chars)
            if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', line_stripped):
                signature_start_line = i
            else:
                # If we hit a line that doesn't look like part of the signature, stop
                break
        
        # Extract lines from signature start to the opening brace
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

    def _replace_function_in_code(self, complete_code: str, function_name: str, new_implementation: str) -> str:
        """
        Replace a function implementation in the complete code with a new implementation.
        
        Args:
            complete_code: The complete C++ source code
            function_name: Name of the function to replace
            new_implementation: The new function implementation
            
        Returns:
            The updated complete code with replaced function
        """
        lines = complete_code.split('\n')
        
        # Find the function by looking for the function name followed by parentheses
        function_start_line = -1
        function_end_line = -1
        
        for i, line in enumerate(lines):
            # Look for function name followed by parentheses (with possible whitespace)
            if re.search(rf'\b{re.escape(function_name)}\s*\(', line):
                function_start_line = i
                break
        
        if function_start_line == -1:
            return complete_code  # Function not found, return original
        
        # Look backwards to find the return type (start of function signature)
        signature_start_line = function_start_line
        for i in range(function_start_line, -1, -1):
            line_stripped = lines[i].strip()
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                continue
            # If this line contains the function name, we've found the signature start
            if function_name in line_stripped:
                signature_start_line = i
                break
            # If this line looks like it could be part of a return type (contains alphanumeric chars)
            if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', line_stripped):
                signature_start_line = i
            else:
                # If we hit a line that doesn't look like part of the signature, stop
                break
        
        # Find the end of the function by counting braces
        open_braces = 0
        found_opening_brace = False
        
        for i in range(signature_start_line, len(lines)):
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
        new_lines = (lines[:signature_start_line] + 
                    [new_implementation] + 
                    lines[function_end_line + 1:])
        
        return '\n'.join(new_lines)