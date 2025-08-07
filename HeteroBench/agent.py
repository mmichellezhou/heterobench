import os
import re
from typing import Dict, List, Tuple, Optional
import json
import logging
from llm import LLM
from backend import Backend


class KernelCodeGenerator:
    """
    A framework for using LLMs to generate optimized kernel code for HeteroBench.
    """

    def __init__(self, llm: LLM, backend: Backend):
        """
        Initialize the kernel code generator.

        Args:
            llm: LLM instance configured with provider and model
            backend: Backend instance for specific hardware target
        """
        self.llm = llm
        self.backend = backend
        # Store conversation histories for each file
        # Each file maintains its own conversation history to ensure
        # that feedback from previous iterations is preserved per file
        self.conversation_histories: Dict[str, List[Dict[str, str]]] = {}

        # Load opt_config.json for skip_files
        opt_config_path = os.path.join(
            os.path.dirname(__file__), "config_json", "opt_config.json"
        )
        if os.path.exists(opt_config_path):
            with open(opt_config_path, "r") as f:
                self.opt_config = json.load(f)
        else:
            self.opt_config = {"skip_files": {}}

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

    def call_llm(self, prompt: str, file_path: str) -> Tuple[str, Optional[Dict]]:
        """
        Call the LLM API with the given prompt, maintaining conversation history per file.

        Args:
            prompt: The prompt to send to the LLM
            file_path: Path to the file being processed (used as key for conversation history)

        Returns:
            Tuple of (The LLM's response text, entire response dictionary)
        """
        system_prompt = f"You are an expert in high-performance computing, kernel optimization, and hardware acceleration for {self.backend.name}."

        # Get or create conversation history for this file
        if file_path not in self.conversation_histories:
            self.conversation_histories[file_path] = []
            logging.debug(f"Created new conversation history for file: {file_path}")
        else:
            history_length = len(self.conversation_histories[file_path])
            logging.debug(
                f"Using existing conversation history for file: {file_path} ({history_length} messages)"
            )

        # Create a temporary LLM instance with the file-specific conversation history
        temp_llm = LLM(
            provider=self.llm.provider,
            model=self.llm.model,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        )
        temp_llm.conversation_history = self.conversation_histories[file_path].copy()

        try:
            response, entire_response = temp_llm.generate_completion(
                system_prompt, prompt
            )
            # Update the file-specific conversation history
            self.conversation_histories[file_path] = (
                temp_llm.conversation_history.copy()
            )
            return response, entire_response
        finally:
            # Clean up the temporary LLM instance
            del temp_llm

    def clear_conversation_history(self, file_path: str = None):
        """
        Clear conversation history for a specific file or all files.

        Args:
            file_path: Path to the file to clear history for. If None, clears all histories.
        """
        if file_path is None:
            self.conversation_histories.clear()
        else:
            self.conversation_histories.pop(file_path, None)

    def get_conversation_history(self, file_path: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of conversation messages for the file
        """
        return self.conversation_histories.get(file_path, []).copy()

    def get_conversation_history_length(self, file_path: str) -> int:
        """
        Get the number of messages in the conversation history for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Number of messages in the conversation history
        """
        return len(self.conversation_histories.get(file_path, []))

    def generate_optimized_function(
        self,
        file_path: str,
        function_name: str,
        benchmark_name: str,
        output_dir: str = None,
        iteration: int = 0,
        previous_results: Dict = None
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

        # Create prompt based on iteration
        if iteration == 0:
            prompt = self.backend.create_prompt(complete_code, function_name)
        elif previous_results:
            prompt = self._generate_feedback(previous_results, iteration, function_name)
        else:
            raise ValueError(
                f"This is iteration {iteration}, but no previous results are given"
            )

        # Call LLM with file-specific conversation history
        response, entire_response = self.call_llm(prompt, file_path)

        # Extract code blocks and find the main optimized implementation
        code_blocks = self._extract_code_blocks(response)
        llm_gen_code = None
        function_generation_success = False

        # Handle None/empty response
        if response and code_blocks:
            # Find the longest code block as the main implementation
            llm_gen_code = max(code_blocks, key=len)
            # Only set success to True if we have meaningful generated code
            function_generation_success = (
                llm_gen_code is not None and len(llm_gen_code.strip()) > 0
            )

        # Create the optimized output code
        output_complete_code = None
        output_code_path = None
        if function_generation_success and llm_gen_code:
            output_complete_code = self.backend.create_output(
                complete_code,
                llm_gen_code,
                function_name,  # Use extracted code, not entire response
            )

            # Save the optimized code to cpu_impl_optimized directory
            cpu_impl_dir = os.path.dirname(file_path)
            cpu_impl_optimized_dir = cpu_impl_dir.replace(
                "cpu_impl", "cpu_impl_optimized"
            )
            os.makedirs(cpu_impl_optimized_dir, exist_ok=True)
            output_code_path = os.path.join(
                cpu_impl_optimized_dir, f"{function_name}_optimized.cpp"
            )
            with open(output_code_path, "w") as f:
                f.write(output_complete_code)

            # Also save to output_dir if provided (PolyBench style)
            if output_dir is not None:
                optimized_file = os.path.join(
                    output_dir, f"{function_name}_optimized.cpp"
                )
                with open(optimized_file, "w") as f:
                    f.write(output_complete_code)

        return {
            "file_path": file_path,
            "function_name": function_name,
            "benchmark_name": benchmark_name,
            "iteration": iteration,
            "prompt": prompt,
            "response": response,
            "entire_response": entire_response,
            "code_blocks": code_blocks,
            "llm_gen_code": llm_gen_code,
            "function_generation_success": function_generation_success,
            "output_complete_code": output_complete_code,
            "output_code_path": output_code_path,
        }

    def _generate_feedback(
        self,
        previous_results: Dict,
        iteration: int,
        function_name: str,
    ) -> str:
        """
        Generate feedback prompt based on previous iteration results.
        Since conversation history is enabled, only include results/status,
        not the previous implementation code.

        Args:
            previous_results: Results from previous iteration
            iteration: Current iteration number
            complete_code: The original complete code
            function_name: Name of the function to optimize

        Returns:
            Formatted feedback prompt string
        """
        feedback_parts = []

        # Add iteration context
        feedback_parts.append(
            f"FEEDBACK FROM PREVIOUS ITERATION (ITERATION {iteration - 1}):"
        )
        feedback_parts.append("=" * 50)

        # Add compilation feedback
        if not previous_results.get("compilation_success", False):
            feedback_parts.append("❌ COMPILATION FAILED")
            if (
                "compile_and_run" in previous_results
                and previous_results["compile_and_run"]
            ):
                compile_error = previous_results["compile_and_run"].get(
                    "compile_error", ""
                )
                if compile_error:
                    feedback_parts.append("Compilation Error Details:")
                    feedback_parts.append(f"```\n{compile_error}\n```")
            feedback_parts.append("Fix the compilation error in your implementation.")

        # Add execution feedback
        elif not previous_results.get("execution_success", False):
            feedback_parts.append("❌ EXECUTION FAILED")
            if (
                "compile_and_run" in previous_results
                and previous_results["compile_and_run"]
            ):
                run_error = previous_results["compile_and_run"].get("run_error", "")
                if run_error:
                    feedback_parts.append("Runtime Error Details:")
                    feedback_parts.append(f"```\n{run_error}\n```")
            feedback_parts.append("Fix the runtime error in your implementation.")

        # Add verification feedback
        elif not previous_results.get("verification_success", False):
            feedback_parts.append("❌ VERIFICATION FAILED")
            feedback_parts.append(
                "The optimized implementation produces incorrect results."
            )
            feedback_parts.append(
                "Fix the verification error by ensuring functional equivalence."
            )

        # If everything passed, provide performance feedback and improvement instructions
        else:
            feedback_parts.append("✅ PREVIOUS ITERATION SUCCESSFUL")
            feedback_parts.append(
                "Compilation, execution, and verification all passed."
            )

            # Add performance information if available
            if "performance_analysis" in previous_results:
                performance_analysis = previous_results["performance_analysis"]

                # Look for function-specific performance in function_times
                function_times = performance_analysis.get("function_times", {})
                if function_name in function_times:
                    func_data = function_times[function_name]
                    if self.backend.show_speedup_in_feedback:
                        if func_data.get("original_time_seconds") is not None:
                            feedback_parts.append(
                                f"Original time: {func_data['original_time_seconds']:.6f} seconds"
                            )
                        if func_data.get("optimized_time_seconds") is not None:
                            feedback_parts.append(
                                f"Optimized time: {func_data['optimized_time_seconds']:.6f} seconds"
                            )
                        if func_data.get("speedup") is not None:
                            speedup = func_data["speedup"]
                            feedback_parts.append(f"Current speedup: {speedup:.2f}x")
                    else:
                        if func_data.get("optimized_time_seconds") is not None:
                            feedback_parts.append(
                                f"Current time: {func_data['optimized_time_seconds']:.6f} seconds"
                            )
                else:
                    # Fallback to overall performance if function-specific data not available
                    if self.backend.show_speedup_in_feedback:
                        if performance_analysis.get("original_time") is not None:
                            feedback_parts.append(
                                f"Original time: {performance_analysis['original_time']:.6f} seconds"
                            )
                        if performance_analysis.get("optimized_time") is not None:
                            feedback_parts.append(
                                f"Optimized time: {performance_analysis['optimized_time']:.6f} seconds"
                            )
                        if performance_analysis.get("speedup") is not None:
                            speedup = performance_analysis["speedup"]
                            feedback_parts.append(f"Current speedup: {speedup:.2f}x")
                    else:
                        if performance_analysis.get("optimized_time") is not None:
                            feedback_parts.append(
                                f"Current time: {performance_analysis['optimized_time']:.6f} seconds"
                            )

            feedback_parts.append(
                "Try to improve the performance further if possible, while maintaining correctness."
            )

        feedback_parts.append("=" * 50)
        feedback_parts.append(
            "Based on the above feedback, provide an improved implementation."
        )

        return "\n".join(feedback_parts)

    def save_results(self, results: Dict, output_dir: str):
        """
        Save results to output directory.

        Args:
            results: Results dictionary to save
            output_dir: Directory to save results
        """
        # Save the prompt
        prompt_file = os.path.join(output_dir, f"{results['function_name']}_prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(results["prompt"])

        # Save the LLM response
        response_file = os.path.join(
            output_dir, f"{results['function_name']}_response.txt"
        )
        with open(response_file, "w") as f:
            f.write(results["response"] if results["response"] is not None else "")

    def process_benchmark(
        self,
        benchmark_name: str,
        benchmark_path: str,
        output_dir: str,
        max_iterations: int = 1,
    ) -> Dict:
        """
        Complete workflow: process benchmark, save results, compile and run optimized code.

        Args:
            benchmark_name: Name of the benchmark
            benchmark_path: Path to the benchmark directory
            output_dir: Directory to save outputs
            max_iterations: Maximum number of iterations to try

        Returns:
            Dictionary containing all results including compilation and execution
        """
        cpu_impl_path = os.path.join(
            benchmark_path, "homobackend_cpu", "Cpp", "cpu_impl"
        )

        if not os.path.exists(cpu_impl_path):
            raise FileNotFoundError(
                f"CPU implementation directory not found: {cpu_impl_path}"
            )

        # Get all .cpp files in cpu_impl
        cpp_files = [f for f in os.listdir(cpu_impl_path) if f.endswith(".cpp")]

        # Get skip list for this kernel/benchmark
        skip_files = set(self.opt_config.get("skip_files", {}).get(benchmark_name, []))

        results_all_iter = {}  # [iteration_num] -> results

        logging.info(
            f"Processing benchmark '{benchmark_name}' from file: {cpu_impl_path} using {self.backend.name} backend"
        )

        previous_results = None

        for iteration in range(max_iterations):
            # Initialize status tracking
            compilation_success = False
            execution_success = False
            verification_success = False
            all_functions_successful = True

            logging.info(f"=" * 10 + f" Iteration {iteration} " + "=" * 10)

            output_iter_dir = os.path.join(
                output_dir, benchmark_name, f"iteration_{iteration}"
            )
            os.makedirs(output_iter_dir, exist_ok=True)

            # Process all functions for this iteration
            all_results = {}
            all_function_results = {}  # Store detailed results for each function
            for cpp_file in cpp_files:
                if cpp_file in skip_files:
                    # logging.info(
                    #     f"Skipping file '{cpp_file}' for kernel '{benchmark_name}' as specified in opt_config.json."
                    # )
                    continue
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
                    output_iter_dir,
                    iteration,
                    previous_results,
                )

                # Save results for this function (prompts, responses, optimized code)
                self.save_results(results, output_iter_dir)

                # Collect status for each function
                function_summary = {
                    "function_generation_success": results[
                        "function_generation_success"
                    ],
                }

                # Handle kernel generation failure for individual function
                if not results.get("function_generation_success", False):
                    results.update(
                        {
                            "compilation_success": False,
                            "execution_success": False,
                            "verification_success": False,
                        }
                    )
                    all_functions_successful = False

                all_results[function_name] = results
                all_function_results[function_name] = function_summary

                # Store detailed results for summary file
                all_function_results[function_name] = {
                    "statistics": {
                        "num_code_blocks_generated": len(
                            results.get("code_blocks", [])
                        ),
                        "llm_gen_code_length": (
                            len(results.get("llm_gen_code", ""))
                            if results.get("llm_gen_code")
                            else 0
                        ),
                        "prompt_length": len(results.get("prompt", "")),
                        "llm_response_length": (
                            len(results.get("response", ""))
                            if results.get("response")
                            else 0
                        ),
                    }
                }

                # Store full function results for feedback
                all_results[function_name] = results

            # Check overall kernel generation success
            if not all_functions_successful:
                logging.warning(
                    "✗ Kernel generation failed. Skipping compilation and execution."
                )

                # Store current results for next iteration feedback and continue to next iteration
                iteration_results = {
                    "iteration": iteration,
                    "all_functions_successful": all_functions_successful,
                    "compilation_success": False,
                    "execution_success": False,
                    "verification_success": False,
                    "performance_analysis": {},
                    "compile_and_run": None,
                    "function_results": all_results,
                }
                results_all_iter[iteration] = iteration_results
                previous_results = iteration_results.copy()
                continue

            logging.info(f"✓ Kernel generation completed")

            # Compile and run if all functions were generated successfully
            performance_analysis = {}
            compilation_success = False
            execution_success = False
            verification_success = False

            # Use backend to compile and run
            compile_results = self.backend.compile_and_run(
                benchmark_name, benchmark_path, output_iter_dir
            )

            compilation_success = compile_results["compilation_successful"]
            execution_success = compile_results["execution_successful"]

            # Analyze run output if execution was successful
            if execution_success and compile_results.get("run_output"):
                performance_analysis = self.backend.analyze_run_output(
                    compile_results["run_output"], benchmark_name
                )
                verification_success = performance_analysis.get(
                    "verification_success", False
                )

                # Add performance analysis to each function's results
                function_times = performance_analysis.get("function_times", {})
                for function_name in all_function_results:
                    # Get individual function performance data if available
                    if function_name in function_times:
                        func_data = function_times[function_name]
                        all_function_results[function_name]["performance_analysis"] = {
                            "original_time_seconds": func_data.get(
                                "original_time_seconds"
                            ),
                            "optimized_time_seconds": func_data.get(
                                "optimized_time_seconds"
                            ),
                            "speedup": func_data.get("speedup"),
                        }
                    else:
                        # Fallback to overall performance data
                        all_function_results[function_name]["performance_analysis"] = {
                            "original_time_seconds": performance_analysis.get(
                                "original_time"
                            ),
                            "optimized_time_seconds": performance_analysis.get(
                                "optimized_time"
                            ),
                            "speedup": performance_analysis.get("speedup"),
                        }

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

            # Create summary file for this iteration
            summary_data = {
                "kernel": benchmark_name,
                "backend": self.backend.name,
                "input_code_path": cpu_impl_path,
                "output_code_path": output_iter_dir,
                "status": {
                    "kernel_generation_success": all_functions_successful,
                    "compilation_success": compilation_success,
                    "execution_success": execution_success,
                    "verification_success": verification_success,
                },
                "performance_analysis": {
                    "original_time_seconds": performance_analysis.get("original_time"),
                    "optimized_time_seconds": performance_analysis.get(
                        "optimized_time"
                    ),
                    "speedup": performance_analysis.get("speedup"),
                },
                "files": all_function_results,
            }

            # Save the summary file
            summary_file = os.path.join(
                output_iter_dir, f"{benchmark_name}_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

            # Show relative path starting from llm_output
            relative_path = os.path.relpath(
                output_iter_dir,
                start=os.path.dirname(
                    os.path.dirname(os.path.dirname(output_iter_dir))
                ),
            )
            logging.info(f"Files saved in: {relative_path}")

            # Write compile_output.txt and execution_output.txt
            compile_output_file = os.path.join(
                output_iter_dir, f"{benchmark_name}_compile_output.txt"
            )
            with open(compile_output_file, "w") as f:
                f.write("COMPILATION COMMAND:\n")
                f.write(compile_results.get("compile_cmd", "") + "\n")
                f.write("\nSTDOUT:\n")
                f.write(compile_results.get("compile_output", ""))
                f.write("\nSTDERR:\n")
                f.write(compile_results.get("compile_error", ""))

            if compile_results.get("compilation_successful"):
                execution_output_file = os.path.join(
                    output_iter_dir, f"{benchmark_name}_execution_output.txt"
                )
                with open(execution_output_file, "w") as f:
                    f.write("EXECUTION COMMAND:\n")
                    f.write(compile_results.get("executable", "") + "\n")
                    f.write("\nSTDOUT:\n")
                    f.write(compile_results.get("run_output", ""))
                    f.write("\nSTDERR:\n")
                    f.write(compile_results.get("run_error", ""))

            # Store results for this iteration
            iteration_results = {
                "iteration": iteration,
                "all_functions_successful": all_functions_successful,
                "compilation_success": compilation_success,
                "execution_success": execution_success,
                "verification_success": verification_success,
                "performance_analysis": performance_analysis,
                "compile_and_run": compile_results,
                "function_results": all_results,
                "individual_function_results": all_function_results,
            }

            # Store current results for next iteration feedback
            results_all_iter[iteration] = iteration_results
            previous_results = iteration_results.copy()

        return results_all_iter
