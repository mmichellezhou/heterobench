import os
import re
from typing import Dict, List, Tuple, Optional
import json
import logging
from llm import LLM


class HeteroBenchCodeGenerator:
    """
    A framework for using LLMs to generate optimized kernel code for HeteroBench.
    """

    def __init__(self, llm: LLM):
        """
        Initialize the HeteroBench code generator.

        Args:
            llm: LLM instance configured with provider and model
        """
        self.llm = llm

    def create_prompt(
        self,
        complete_code: str,
        function_code: str,
        optimized_func_signature: str,
        function_name: str,
    ) -> str:
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
        prompt = f"""You are an expert in high-performance computing and kernel engineering on the CPU. You are familiar with different optimization techniques including MKL, memory access optimization, tiling, unrolling, loop transformations, and SIMD instructions. Focus on single-threaded performance improvement. Don't use multi-threading nor vectorization.

        Given the following code:
        ```cpp
        {complete_code}
        ```

        with the following function to optimize: 
        ```cpp
        {function_code}
        ```

        Task: Analyze this kernel and generate an optimized kernel implementation to get better performance while maintaining functional equivalence. You should first consider if the kernel can be implemented using a single corresponding MKL function. If not, consider if the kernel can be decomposed into multiple MKL calls. If there is no way to implement the kernel using MKL, or you think MKL cannot get good performance, then consider applying optimizations directly to the kernel, such as memory access optimization, tiling, unrolling, etc. You should only use single thread for the optimized kernel implementation.

        Machine we are using: 
        - Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
        - L1d cache: 1.5MB
        - L1i cache: 1.5MB
        - L2 cache: 48MB
        - L3 cache: 71.5MB
        - Supports SSE, AVX2, AVX512

        Requirements:
        1. Optimize the function for better single-threaded performance.
        2. Ensure functional equivalence.
        3. Include all original and any new headers/dependencies.
        4. Assume all variables and helper functions used inside the target function are already defined.
        5. If no optimizations can get good performance, then just fallback to the default implementation.

        Output format:
        You should output the optimized function implementation with the exact function signature as follows:
        ```cpp
        {optimized_func_signature}
        ```

        Do not include any other text other than the optimized function implementation and necessary headers/dependencies.
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
        pattern = r"```(?:[a-zA-Z]*\n)?(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        return [match.strip() for match in matches]

    def _extract_function_from_code(
        self, complete_code: str, function_name: str
    ) -> str:
        """
        Extract a specific function implementation from C++ code.

        Args:
            complete_code: The complete C++ source code
            function_name: Name of the function to extract

        Returns:
            The function implementation as a string
        """
        # Split code into lines for easier processing
        lines = complete_code.split("\n")

        # Find the function by looking for the function name followed by parentheses
        function_start_line = -1
        function_end_line = -1

        for i, line in enumerate(lines):
            # Look for function name followed by parentheses (with possible whitespace)
            if re.search(rf"\b{re.escape(function_name)}\s*\(", line):
                function_start_line = i
                break

        if function_start_line == -1:
            return ""

        # Look backwards to find the return type (start of function signature)
        signature_start_line = function_start_line
        for i in range(function_start_line, -1, -1):
            line_stripped = lines[i].strip()
            # Skip empty lines and comments
            if (
                not line_stripped
                or line_stripped.startswith("//")
                or line_stripped.startswith("/*")
            ):
                continue
            # If this line contains the function name, we've found the signature start
            if function_name in line_stripped:
                signature_start_line = i
                break
            # If this line looks like it could be part of a return type (contains alphanumeric chars)
            if re.search(r"[a-zA-Z_][a-zA-Z0-9_]*", line_stripped):
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
                if char == "{":
                    open_braces += 1
                    found_opening_brace = True
                elif char == "}":
                    open_braces -= 1

            # If we found the opening brace and braces are balanced, we found the end
            if found_opening_brace and open_braces == 0:
                function_end_line = i
                break

        if function_end_line == -1:
            return ""

        # Extract the function
        function_lines = lines[signature_start_line : function_end_line + 1]
        return "\n".join(function_lines)

    def generate_optimized_function(
        self, file_path: str, function_name: str, benchmark_name: str
    ) -> Dict:
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
            with open(file_path, "r") as f:
                complete_code = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")

        # Extract the function implementation
        function_code = self._extract_function_from_code(complete_code, function_name)

        if not function_code:
            raise ValueError(
                f"Could not find function '{function_name}' in {file_path}"
            )

        # Get the optimized file path
        original_dir = os.path.dirname(file_path)
        optimized_dir = os.path.join(
            os.path.dirname(original_dir), "cpu_impl_optimized"
        )
        optimized_file_path = os.path.join(
            optimized_dir, f"{function_name}_optimized.cpp"
        )

        # Read the existing optimized file
        try:
            with open(optimized_file_path, "r") as f:
                optimized_file_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find optimized file: {optimized_file_path}"
            )
        except Exception as e:
            raise Exception(
                f"Error reading optimized file {optimized_file_path}: {str(e)}"
            )

        # Extract the optimized function signature
        optimized_function_name = f"{function_name}_optimized"
        optimized_func_signature = self._extract_function_signature_from_code(
            optimized_file_content, optimized_function_name
        )

        if not optimized_func_signature:
            raise ValueError(
                f"Could not find function '{optimized_function_name}' in {optimized_file_path}"
            )

        # Create prompt
        prompt = self.create_prompt(
            complete_code, function_code, optimized_func_signature, function_name
        )

        # Call LLM
        response, entire_response = self.call_llm(prompt)

        # Use the entire LLM response as the optimized code, but strip markdown code block markers
        def strip_code_block_markers(text):
            import re

            # Remove triple backtick code blocks (with or without language)
            return re.sub(
                r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE
            ).strip()

        optimized_code_generated = (
            strip_code_block_markers(response) if response else ""
        )
        function_generation_success = bool(optimized_code_generated)

        # The optimized file is just the code (no markdown)
        optimized_complete_code = optimized_code_generated

        # Save the updated optimized file
        os.makedirs(optimized_dir, exist_ok=True)
        with open(optimized_file_path, "w") as f:
            f.write(optimized_complete_code)

        return {
            "function_name": function_name,
            "original_file_path": file_path,
            "prompt": prompt,
            "llm_response": response,
            "entire_llm_response": entire_response,
            "optimized_code_generated": optimized_code_generated,
            "num_code_blocks": 1 if response else 0,
            "optimized_complete_code": optimized_complete_code,
            "optimized_file_path": optimized_file_path,
            "function_generation_success": function_generation_success,
        }

    def save_results(self, results: Dict, output_dir: str):
        """
        Save results including all status information.

        Args:
            results: The complete results dictionary
            output_dir: Directory to save outputs
        """
        function_name = results["function_name"]

        # Save the prompt
        with open(os.path.join(output_dir, f"{function_name}_prompt.txt"), "w") as f:
            f.write(results["prompt"])

        # Save full LLM response
        with open(
            os.path.join(output_dir, f"{function_name}_llm_response.txt"), "w"
        ) as f:
            f.write(
                results["llm_response"] if results["llm_response"] is not None else ""
            )

        # Save the final optimized C++ file in llm_output
        if results.get("optimized_complete_code"):
            optimized_file_name = f"{function_name}_optimized.cpp"
            with open(os.path.join(output_dir, optimized_file_name), "w") as f:
                f.write(results["optimized_complete_code"])

        # Save summary
        summary = {
            "function_name": function_name,
            "original_file_path": results.get("original_file_path", ""),
            "optimized_file_path": results.get("optimized_file_path", ""),
            "function_generation_success": results.get(
                "function_generation_success", False
            ),
            "num_code_blocks_generated": results.get("num_code_blocks", 0),
            "optimized_code_length": len(results.get("optimized_code_generated") or ""),
            "prompt_length": len(results.get("prompt", "")),
            "llm_response_length": len(results.get("llm_response") or ""),
            "entire_llm_response": results.get("entire_llm_response", ""),
        }

        # Save summary
        with open(os.path.join(output_dir, f"{function_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Print status summary
        logging.info(
            f"{'✓' if summary['function_generation_success'] else '✗'} Function generation completed"
        )
        logging.info("=" * 60)

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

        lines = run_output.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for performance results
            if "Total:" in line:
                try:
                    # Extract speedup value (format: "Total: X.XXXXXXx")
                    speedup_str = line.split(":")[-1].replace("x", "").strip()
                    analysis["speedup"] = float(speedup_str)
                except (ValueError, IndexError):
                    pass

            # Look for timing information
            elif "Original Implementation:" in line:
                # Look for timing in next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    if "Single iteration time:" in next_line:
                        try:
                            time_str = (
                                next_line.split(":")[-1].replace("seconds", "").strip()
                            )
                            analysis["original_time"] = float(time_str)
                        except (ValueError, IndexError):
                            pass
                        break

            elif "Optimized Implementation:" in line:
                # Look for timing in next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    if "Single iteration time:" in next_line:
                        try:
                            time_str = (
                                next_line.split(":")[-1].replace("seconds", "").strip()
                            )
                            analysis["optimized_time"] = float(time_str)
                        except (ValueError, IndexError):
                            pass
                        break

        # Calculate speedup if we have both timing values but speedup parsing failed
        if (
            analysis["speedup"] is None
            and analysis["original_time"] is not None
            and analysis["optimized_time"] is not None
        ):
            if analysis["optimized_time"] > 0:
                analysis["speedup"] = (
                    analysis["original_time"] / analysis["optimized_time"]
                )

        return analysis

    def compile_and_run(
        self, benchmark_name: str, benchmark_path: str, output_dir: str
    ) -> Dict:
        """
        Compile and run the benchmark using make run_simple.

        Args:
            benchmark_name: Name of the benchmark
            benchmark_path: Path to the benchmark directory
            output_dir: Directory to save outputs

        Returns:
            Dictionary with compilation and execution results
        """
        import subprocess
        import glob

        cpp_path = os.path.join(benchmark_path, "homobackend_cpu", "Cpp")
        benchmark_output_dir = os.path.join(output_dir, benchmark_name)
        os.makedirs(benchmark_output_dir, exist_ok=True)

        compile_results = {
            "benchmark_name": benchmark_name,
            "cpp_path": cpp_path,
            "compilation_successful": False,
            "execution_successful": False,
            "compile_output": "",
            "compile_error": "",
            "run_output": "",
            "run_error": "",
        }

        # Change to the CPP directory
        original_cwd = os.getcwd()
        os.chdir(cpp_path)

        try:
            env = os.environ.copy()
            use_mkl = False

            # Scan optimized files for MKL usage
            opt_files = glob.glob("cpu_impl_optimized/*.cpp")
            for fname in opt_files:
                try:
                    with open(fname, "r") as f:
                        content = f.read()
                        if "#include <mkl.h>" in content:
                            use_mkl = True
                            break
                except Exception:
                    continue
            if use_mkl:
                env["USE_MKL"] = "1"
                if "MKLROOT" not in env:
                    compile_results["compile_error"] = (
                        "MKLROOT is not set. Please source the Intel oneAPI environment (e.g., source /opt/intel/oneapi/setvars.sh))"
                    )
                    return compile_results
            # Build and run the benchmark
            logging.info(f"Building and running {benchmark_name}...")
            build_result = subprocess.run(
                ["make", "run_simple"],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            with open(
                os.path.join(
                    benchmark_output_dir, f"{benchmark_name}_build_output.txt"
                ),
                "w",
            ) as f:
                f.write("BUILD COMMAND: make run_simple\n")
                f.write("STDOUT:\n")
                f.write(build_result.stdout)
                f.write("\nSTDERR:\n")
                f.write(build_result.stderr)
            compile_results["compile_cmd"] = "make run_simple"
            compile_results["run_output"] = build_result.stdout
            compile_results["run_error"] = build_result.stderr
            compile_results["compilation_successful"] = build_result.returncode == 0
            compile_results["execution_successful"] = build_result.returncode == 0
        except subprocess.TimeoutExpired:
            compile_results["compile_error"] = "Compilation or execution timeout"
        except Exception as e:
            compile_results["compile_error"] = (
                f"Error during compilation/execution: {str(e)}"
            )
        finally:
            os.chdir(original_cwd)
        return compile_results

    def process_benchmark(
        self, benchmark_name: str, benchmark_path: str, output_dir: str
    ) -> Dict:
        """
        Process all functions in a benchmark's cpu_impl directory, then compile and run.

        Args:
            benchmark_name: Name of the benchmark
            benchmark_path: Path to the benchmark directory
            output_dir: Directory to save outputs

        Returns:
            Dictionary containing results for all processed functions and compilation/execution
        """
        cpu_impl_path = os.path.join(
            benchmark_path, "homobackend_cpu", "Cpp", "cpu_impl"
        )

        if not os.path.exists(cpu_impl_path):
            raise FileNotFoundError(
                f"CPU implementation directory not found: {cpu_impl_path}"
            )

        # Create benchmark-specific output directory
        benchmark_output_dir = os.path.join(output_dir, benchmark_name)
        os.makedirs(benchmark_output_dir, exist_ok=True)

        # Load skip files configuration
        config_path = os.path.join(
            os.path.dirname(__file__), "config_json", "opt_config.json"
        )
        skip_files = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    skip_files = config.get("skip_files", {})
            except Exception as e:
                logging.warning(f"Could not load optimization config: {e}")

        # Get all .cpp files in cpu_impl
        cpp_files = [f for f in os.listdir(cpu_impl_path) if f.endswith(".cpp")]

        # Filter out files that should be skipped for this benchmark
        files_to_skip = skip_files.get(benchmark_name, [])
        cpp_files = [f for f in cpp_files if f not in files_to_skip]

        all_results = {}
        all_functions_successful = True

        for cpp_file in cpp_files:
            function_name = os.path.splitext(cpp_file)[0]  # Remove .cpp extension
            file_path = os.path.join(cpu_impl_path, cpp_file)

            logging.info(
                f"Processing function '{function_name}' from file: {file_path}"
            )

            # Process the function
            results = self.generate_optimized_function(
                file_path, function_name, benchmark_name
            )

            # Save results for this function in benchmark-specific directory
            self.save_results(results, benchmark_output_dir)

            # Collect status for each function
            function_summary = {
                "function_generation_success": results["function_generation_success"],
            }

            # Add speedup if available
            if (
                "run_analysis" in results
                and results["run_analysis"].get("speedup") is not None
            ):
                function_summary["speedup"] = results["run_analysis"]["speedup"]

            all_results[function_name] = function_summary

            # Track if all functions were successful
            if not results["function_generation_success"]:
                all_functions_successful = False

        # Compile and run if all functions were generated successfully
        compilation_success = False
        execution_success = False
        verification_success = False
        performance_analysis = {}

        if all_functions_successful:
            compile_results = self.compile_and_run(
                benchmark_name, benchmark_path, output_dir
            )
            all_results["compile_and_run"] = compile_results

            compilation_success = compile_results["compilation_successful"]
            execution_success = compile_results["execution_successful"]

            # Analyze run output if execution was successful
            if execution_success and compile_results.get("run_output"):
                performance_analysis = self._analyze_run_output(
                    compile_results["run_output"]
                )
                all_results["run_analysis"] = performance_analysis
                verification_success = self.determine_verification_success(
                    compile_results["run_output"]
                )

            if compilation_success:
                logging.info(f"✓ Compilation successful")
                if execution_success:
                    logging.info(f"✓ Execution successful")

                    # Print verification results
                    if verification_success:
                        logging.info(f"🔍 Verification: ✓ PASS")
                    else:
                        logging.warning(f"🔍 Verification: ✗ FAIL")

                    # Print performance analysis
                    logging.info(f"📊 Performance Analysis:")
                    if performance_analysis.get("original_time") is not None:
                        logging.info(
                            f"  Original time: {performance_analysis['original_time']:.6f} seconds"
                        )
                    if performance_analysis.get("optimized_time") is not None:
                        logging.info(
                            f"  Optimized time: {performance_analysis['optimized_time']:.6f} seconds"
                        )
                    if performance_analysis.get("speedup") is not None:
                        logging.info(
                            f"  Speedup: {performance_analysis['speedup']:.2f}x"
                        )

                else:
                    logging.error(f"✗ Execution failed:")
                    logging.error(compile_results["run_error"])
            else:
                logging.error(f"✗ Compilation failed:")
                logging.error(compile_results["compile_error"])
        else:
            logging.warning(
                "✗ Some functions failed to generate. Skipping compilation and execution."
            )

        # Add comprehensive status tracking
        all_results.update(
            {
                "all_functions_successful": all_functions_successful,
                "compilation_success": compilation_success,
                "execution_success": execution_success,
                "verification_success": verification_success,
            }
        )

        # Save aggregated results in benchmark-specific directory
        summary_path = os.path.join(
            benchmark_output_dir, f"{benchmark_name}_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Print benchmark-level summary like PolyBench
        logging.info(f"Files saved in: {benchmark_output_dir}")
        logging.info("=" * 60)
        # (No speedup print here; only in performance analysis block)

        return all_results

    def _extract_function_signature_from_code(
        self, complete_code: str, function_name: str
    ) -> str:
        """
        Extract just the function signature (declaration) from C++ code.

        Args:
            complete_code: The complete C++ source code
            function_name: Name of the function to extract signature for

        Returns:
            The function signature as a string (up to the opening brace)
        """
        # Split code into lines for easier processing
        lines = complete_code.split("\n")

        # Find the function by looking for the function name followed by parentheses
        function_start_line = -1

        for i, line in enumerate(lines):
            # Look for function name followed by parentheses (with possible whitespace)
            if re.search(rf"\b{re.escape(function_name)}\s*\(", line):
                function_start_line = i
                break

        if function_start_line == -1:
            return ""

        # Look backwards to find the return type (start of function signature)
        signature_start_line = function_start_line
        for i in range(function_start_line, -1, -1):
            line_stripped = lines[i].strip()
            # Skip empty lines and comments
            if (
                not line_stripped
                or line_stripped.startswith("//")
                or line_stripped.startswith("/*")
            ):
                continue
            # If this line contains the function name, we've found the signature start
            if function_name in line_stripped:
                signature_start_line = i
                break
            # If this line looks like it could be part of a return type (contains alphanumeric chars)
            if re.search(r"[a-zA-Z_][a-zA-Z0-9_]*", line_stripped):
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
            if "{" in line:
                # Remove the opening brace and everything after it
                brace_pos = line.find("{")
                last_line = line[:brace_pos].rstrip()
                signature_lines[-1] = last_line
                break

        # Join the lines and clean up
        signature = "\n".join(signature_lines).strip()
        return signature
