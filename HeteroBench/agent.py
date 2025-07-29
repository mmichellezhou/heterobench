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

    def create_prompt(
        self, complete_code: str, function_code: str, function_name: str
    ) -> str:
        """
        Create a prompt for the LLM to analyze and optimize a specific function.

        Args:
            complete_code: The original complete code from the file
            function_code: The original function code to optimize
            function_name: Name of the function to optimize

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert in high-performance computing and kernel engineering on the CPU. You are familiar with different optimization techniques including vectorization, memory access optimization, tiling, unrolling, loop transformations, SIMD instructions, and MKL. Focus on single-threaded performance improvement. Don't use multi-threading.

        Given the following code:
        ```cpp
        {complete_code}
        ```

        with the following function to optimize: 
        ```cpp
        {function_code}
        ```

        Task: Analyze this kernel and generate an optimized kernel implementation to get better performance while maintaining functional equivalence. Consider applying optimizations directly to the kernel, such as vectorization, memory access optimization, tiling, unrolling, MKL, etc. You should only use single thread for the optimized kernel implementation.

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
        {function_name}_optimized(...)
        ```

        Do not include any other text other than the optimized function implementation and necessary headers/dependencies.
        """
        return prompt

    def _generate_feedback(
        self,
        previous_results: Dict,
        iteration: int,
        complete_code: str,
        function_code: str,
        function_name: str,
    ) -> str:
        """
        Generate feedback based on previous iteration results.

        Args:
            previous_results: Results from previous iteration
            iteration: Current iteration number
            complete_code: The original complete code from the file
            function_code: The original function code to optimize
            function_name: Name of the function to optimize

        Returns:
            Complete prompt string with feedback
        """
        feedback_parts = []

        feedback_parts.append(
            f"FEEDBACK FROM PREVIOUS ITERATION (ITERATION {iteration - 1}):"
        )
        feedback_parts.append("=" * 50)

        # Check for function generation failure
        if not previous_results.get("all_functions_successful", False):
            feedback_parts.append("❌ FUNCTION GENERATION FAILED")
            feedback_parts.append(
                "The LLM failed to generate valid function implementations for some functions."
            )
            feedback_parts.append("Try generating the functions again.")

        # Check for compilation errors
        elif not previous_results.get("compilation_success", False):
            feedback_parts.append("❌ COMPILATION FAILED")
            if "compile_and_run" in previous_results:
                compile_error = previous_results["compile_and_run"].get(
                    "compile_error", ""
                )
                if compile_error:
                    feedback_parts.append("Compilation Error Details:")
                    feedback_parts.append(f"```\n{compile_error}\n```")
            feedback_parts.append("Fix the compilation error in your implementation.")

        # Check for runtime errors
        elif not previous_results.get("execution_success", False):
            feedback_parts.append("❌ EXECUTION FAILED")
            if "compile_and_run" in previous_results:
                run_error = previous_results["compile_and_run"].get("run_error", "")
                if run_error:
                    feedback_parts.append("Runtime Error Details:")
                    feedback_parts.append(f"```\n{run_error}\n```")
            feedback_parts.append("Fix the runtime error in your implementation.")

        # Check for verification errors
        elif not previous_results.get("verification_success", False):
            feedback_parts.append("❌ VERIFICATION FAILED")
            feedback_parts.append(
                "The optimized implementation produces incorrect results compared to the original implementation."
            )
            if "compile_and_run" in previous_results:
                compile_and_run = previous_results["compile_and_run"]
                if "run_output" in compile_and_run:
                    feedback_parts.append("Run Output:")
                    feedback_parts.append(f"```\n{compile_and_run['run_output']}\n```")
            feedback_parts.append(
                "Fix the verification error by ensuring functional equivalence with the original functions."
            )

        # If everything passed, provide performance feedback
        else:
            feedback_parts.append("✅ PREVIOUS ITERATION SUCCESSFUL")
            feedback_parts.append(
                "Compilation, execution, and verification all passed."
            )

            # Add performance information if available
            if "run_analysis" in previous_results:
                run_analysis = previous_results["run_analysis"]
                if run_analysis.get("original_time") is not None:
                    feedback_parts.append(
                        f"Original benchmark time: {run_analysis['original_time']:.6f} seconds"
                    )
                if run_analysis.get("optimized_time") is not None:
                    feedback_parts.append(
                        f"Optimized benchmark time: {run_analysis['optimized_time']:.6f} seconds"
                    )
                if run_analysis.get("speedup") is not None:
                    speedup = run_analysis["speedup"]
                    feedback_parts.append(f"Current speedup: {speedup:.2f}x")

            feedback_parts.append(
                "Try to improve the performance further if possible, while maintaining correctness."
            )

        feedback_parts.append("=" * 50)
        feedback_parts.append(
            "Based on the above feedback, provide an improved implementation."
        )

        feedback = "\n".join(feedback_parts)

        # Create complete prompt with original task and feedback
        complete_prompt = f"""You are an expert in high-performance computing and kernel engineering on the CPU. You are familiar with different optimization techniques including vectorization, memory access optimization, tiling, unrolling, loop transformations, SIMD instructions, and MKL. Focus on single-threaded performance improvement. Don't use multi-threading.

        Given the following code:
        ```cpp
        {complete_code}
        ```

        with the following function to optimize: 
        ```cpp
        {function_code}
        ```

        Task: Analyze this kernel and generate an optimized kernel implementation to get better performance while maintaining functional equivalence. Consider applying optimizations directly to the kernel, such as vectorization, memory access optimization, tiling, unrolling, MKL, etc. You should only use single thread for the optimized kernel implementation.

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
        {function_name}_optimized(...)
        ```

        Do not include any other text other than the optimized function implementation and necessary headers/dependencies.

        {feedback}
        """

        return complete_prompt

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
        self,
        file_path: str,
        function_name: str,
        benchmark_name: str,
        iteration: int = 0,
        previous_results: Dict = None,
    ) -> Dict:
        """
        Generate an optimized function implementation through the LLM pipeline.

        Args:
            file_path: Path to the C++ source file in cpu_impl
            function_name: Name of the function to optimize (same as filename without extension)
            benchmark_name: Name of the benchmark
            iteration: Current iteration number (0-based)
            previous_results: Results from previous iteration for feedback

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

        # Create prompt based on iteration
        if iteration == 0:
            prompt = self.create_prompt(complete_code, function_code, function_name)
        elif previous_results:
            prompt = self._generate_feedback(
                previous_results, iteration, complete_code, function_code, function_name
            )
        else:
            raise ValueError(
                f"This is iteration {iteration}, but no previous results are given"
            )

        # Call LLM
        response, entire_response = self.call_llm(prompt)

        # Extract code blocks and find the main optimized implementation
        code_blocks = self._extract_code_blocks(response)
        llm_gen_code = None
        function_generation_success = False

        if response and code_blocks:
            llm_gen_code = max(code_blocks, key=len)
            # Only set success to True if we have meaningful generated code
            function_generation_success = (
                llm_gen_code is not None and len(llm_gen_code.strip()) > 0
            )

        # Strip markdown code block markers
        def strip_code_block_markers(text):
            return re.sub(
                r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE
            ).strip()

        optimized_code_generated = (
            strip_code_block_markers(llm_gen_code) if llm_gen_code else ""
        )

        # Get the optimized file path
        original_dir = os.path.dirname(file_path)
        optimized_dir = os.path.join(
            os.path.dirname(original_dir), "cpu_impl_optimized"
        )
        optimized_file_path = os.path.join(
            optimized_dir, f"{function_name}_optimized.cpp"
        )

        # Save the optimized file
        os.makedirs(optimized_dir, exist_ok=True)
        with open(optimized_file_path, "w") as f:
            f.write(optimized_code_generated)

        return {
            "function_name": function_name,
            "original_file_path": file_path,
            "prompt": prompt,
            "llm_response": response,
            "entire_llm_response": entire_response,
            "llm_gen_code": llm_gen_code,
            "optimized_code_generated": optimized_code_generated,
            "num_code_blocks": len(code_blocks),
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

        # Save the final optimized C++ file
        if results.get("optimized_code_generated"):
            optimized_file_name = f"{function_name}_optimized.cpp"
            with open(os.path.join(output_dir, optimized_file_name), "w") as f:
                f.write(results["optimized_code_generated"])

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
        fail_keywords = ["incorrect", "fail", "wrong"]
        return not any(keyword in output_lower for keyword in fail_keywords)

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
                    speedup_str = line.split(":")[-1].replace("x", "").strip()
                    analysis["speedup"] = float(speedup_str)
                except (ValueError, IndexError):
                    pass

            # Look for timing information
            elif "Original Implementation:" in line:
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
                os.path.join(output_dir, f"{benchmark_name}_build_output.txt"), "w"
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
        self,
        benchmark_name: str,
        benchmark_path: str,
        output_dir: str,
        max_iterations: int = 1,
    ) -> Dict:
        """
        Process all functions in a benchmark's cpu_impl directory, then compile and run.

        Args:
            benchmark_name: Name of the benchmark
            benchmark_path: Path to the benchmark directory
            output_dir: Directory to save outputs
            max_iterations: Maximum number of iterations to try

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
        files_to_skip = skip_files.get(benchmark_name, [])
        cpp_files = [f for f in cpp_files if f not in files_to_skip]

        # Process benchmark with multiple iterations
        results_all_iter = self._process_benchmark_with_iterations(
            benchmark_name,
            benchmark_path,
            cpu_impl_path,
            cpp_files,
            benchmark_output_dir,
            max_iterations,
        )

        # Find best iteration and use its results
        best_results = self._find_best_iteration(results_all_iter)
        return best_results

    def _process_benchmark_with_iterations(
        self,
        benchmark_name: str,
        benchmark_path: str,
        cpu_impl_path: str,
        cpp_files: List[str],
        output_dir: str,
        max_iterations: int,
    ) -> Dict:
        """
        Process benchmark with multiple iterations and feedback.

        Args:
            benchmark_name: Name of the benchmark
            benchmark_path: Path to the benchmark directory
            cpu_impl_path: Path to cpu_impl directory
            cpp_files: List of .cpp files to process
            output_dir: Directory to save outputs
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary containing results from all iterations
        """
        results_all_iter = {}  # [iteration_num] -> results
        previous_results = None

        for iteration in range(max_iterations):
            # Initialize status tracking
            compilation_success = False
            execution_success = False
            verification_success = False
            all_functions_successful = True

            logging.info(f"=" * 10 + f" Iteration {iteration} " + "=" * 10)

            output_iter_dir = os.path.join(output_dir, f"iteration_{iteration}")
            os.makedirs(output_iter_dir, exist_ok=True)

            # Process all functions for this iteration
            all_results = {}
            for cpp_file in cpp_files:
                function_name = os.path.splitext(cpp_file)[0]  # Remove .cpp extension
                file_path = os.path.join(cpu_impl_path, cpp_file)

                logging.info(
                    f"Processing function '{function_name}' from file: {file_path}"
                )

                # Generate optimized function (pass previous results for feedback)
                results = self.generate_optimized_function(
                    file_path,
                    function_name,
                    benchmark_name,
                    iteration,
                    previous_results,
                )

                # Save results for this function
                self.save_results(results, output_iter_dir)

                # Collect status for each function
                function_summary = {
                    "function_generation_success": results[
                        "function_generation_success"
                    ],
                }

                all_results[function_name] = function_summary

                # Track if all functions were successful
                if not results["function_generation_success"]:
                    all_functions_successful = False

            # Compile and run if all functions were generated successfully
            performance_analysis = {}
            if all_functions_successful:
                compile_results = self.compile_and_run(
                    benchmark_name, benchmark_path, output_iter_dir
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
                output_iter_dir, f"{benchmark_name}_summary.json"
            )
            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2)

            # Store current results for next iteration feedback
            previous_results = all_results.copy()
            results_all_iter[iteration] = all_results.copy()

        return results_all_iter

    def _find_best_iteration(self, results_all_iter: Dict) -> Dict:
        """
        Find the best iteration from multiple iterations.

        Args:
            results_all_iter: Dictionary of results from all iterations

        Returns:
            Best iteration results
        """
        # Handle case where only one result is returned
        if (
            not isinstance(results_all_iter, dict)
            or "all_functions_successful" in results_all_iter
        ):
            # Single result returned, use it directly
            return results_all_iter

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
                and "run_analysis" in results
                and results["run_analysis"].get("speedup") is not None
            ):

                speedup = results["run_analysis"]["speedup"]
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

        # If nothing passes, use the last iteration
        if best_results is None:
            last_iteration = max(results_all_iter.keys())
            best_results = results_all_iter[last_iteration]
            best_iteration = last_iteration

        # Log the best iteration selection
        if (
            isinstance(results_all_iter, dict)
            and "all_functions_successful" not in results_all_iter
        ):
            if best_results.get("run_analysis", {}).get("speedup") is not None:
                logging.info(
                    f"📈 Best iteration: Iteration {best_iteration} (Speedup: {best_results['run_analysis']['speedup']:.2f}x)"
                )
            elif best_results.get("verification_success"):
                logging.info(
                    f"📈 Best iteration: Iteration {best_iteration} (Verification passed)"
                )
            elif best_results.get("execution_success"):
                logging.info(
                    f"📈 Best iteration: Iteration {best_iteration} (Execution passed)"
                )
            elif best_results.get("compilation_success"):
                logging.info(
                    f"📈 Best iteration: Iteration {best_iteration} (Compilation passed)"
                )
            else:
                logging.info(
                    f"📈 Best iteration: Iteration {best_iteration} (Last iteration)"
                )

        return best_results
