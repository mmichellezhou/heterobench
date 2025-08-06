import os
import re
import subprocess
import glob
from typing import Dict, List
from backend import Backend


class CPUBackend(Backend):
    """CPU backend implementation for HeteroBench"""

    def __init__(self):
        super().__init__("cpu")

        # HeteroBench benchmarks for CPU
        self.heterobench_benchmarks = {
            "3mm": "benchmarks/3_matrix_multiplication",
            "adi": "benchmarks/alternating_direction_implicit",
            "ced": "benchmarks/canny_edge_detection",
            "cnn": "benchmarks/convolutional_neural_network",
            "dgr": "benchmarks/digit_recog",
            "mlp": "benchmarks/multilayer_perceptron",
            "oha": "benchmarks/one_head_attention",
            "opf": "benchmarks/optical_flow",
            "ppc": "benchmarks/parallelize_particle",
            "sbf": "benchmarks/sobel_filter",
            "spf": "benchmarks/spam_filter",
        }

    def get_available_kernels(self) -> Dict[str, str]:
        """Get available HeteroBench benchmarks."""
        return self.heterobench_benchmarks.copy()

    def create_prompt(self, complete_code: str, function_name: str) -> str:
        """Create CPU-specific optimization prompt for HeteroBench functions."""
        # Extract the function implementation
        function_code = self._extract_function_from_code(complete_code, function_name)

        if not function_code:
            raise ValueError(f"Could not find function '{function_name}' in code")

        return f"""You are an expert in high-performance computing and kernel engineering on the CPU. You are familiar with different optimization techniques including memory access optimization, tiling, unrolling, loop transformations, and SIMD instructions. Focus on single-threaded performance improvement. Don't use multi-threading nor vectorization.

Given the following code:
```cpp
{complete_code}
```

with the following function to optimize: 
```cpp
{function_code}
```

Task: Analyze this kernel and generate an optimized kernel implementation to get better performance while maintaining functional equivalence. Consider applying optimizations directly to the kernel, such as memory access optimization, tiling, unrolling, etc. You should only use single thread for the optimized kernel implementation.

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

Do not include any other text other than the optimized function implementation and necessary headers/dependencies."""

    def create_output(
        self, complete_code: str, optimized_code_generated: str, function_name: str
    ) -> str:
        """Create the optimized code file content - return the entire LLM response."""
        # Remove markdown code block markers from the LLM response
        cleaned_code = optimized_code_generated

        # Remove opening ```cpp or ``` markers
        if cleaned_code.startswith("```cpp"):
            cleaned_code = cleaned_code[6:]  # Remove ```cpp
        elif cleaned_code.startswith("```"):
            cleaned_code = cleaned_code[3:]  # Remove ```

        # Remove closing ``` markers
        if cleaned_code.endswith("```"):
            cleaned_code = cleaned_code[:-3]  # Remove ```

        # Strip any leading/trailing whitespace
        cleaned_code = cleaned_code.strip()

        return cleaned_code

    def _extract_function_from_code(
        self, complete_code: str, function_name: str
    ) -> str:
        """Extract C++ function implementation."""
        # Pattern to match function definition and its body
        pattern = rf"({function_name}\s*\([^{{]*\{{\s*.*?\n}})"

        # Search for the function with DOTALL flag to match across newlines
        match = re.search(pattern, complete_code, re.DOTALL)

        if match:
            function_code = match.group(1)

            # Count braces to find the complete function body
            open_braces = 0
            function_start = complete_code.find(match.group(0))

            # Find the actual start of the function (including return type and signature)
            lines = complete_code[: function_start + len(match.group(0))].split("\n")

            # Look backwards to find the function signature start
            function_start_line = -1  # Initialize with default value
            for i in range(len(lines) - 1, -1, -1):
                if function_name in lines[i]:
                    # Found the line with function name, now find the return type
                    j = i
                    while j >= 0 and not (
                        lines[j].strip().startswith("void")
                        or lines[j].strip().startswith("int")
                        or lines[j].strip().startswith("double")
                        or lines[j].strip().startswith("float")
                        or lines[j].strip().startswith("char")
                        or lines[j].strip().startswith("static")
                        or lines[j].strip().startswith("template")
                    ):
                        j -= 1
                    if j >= 0:
                        function_start_line = j
                        break

            # Check if we found a valid function start line
            if function_start_line == -1:
                # Fallback: use the line where function name was found
                for i in range(len(lines) - 1, -1, -1):
                    if function_name in lines[i]:
                        function_start_line = i
                        break

            # Extract from function start to end
            lines_from_start = complete_code.split("\n")[function_start_line:]
            result_lines = []
            open_braces = 0
            found_opening_brace = False

            for line in lines_from_start:
                result_lines.append(line)

                # Count braces
                for char in line:
                    if char == "{":
                        open_braces += 1
                        found_opening_brace = True
                    elif char == "}":
                        open_braces -= 1

                # If we found the opening brace and braces are balanced, we're done
                if found_opening_brace and open_braces == 0:
                    break

            return "\n".join(result_lines)

        return ""

    def _replace_function_in_code(
        self, complete_code: str, function_name: str, new_implementation: str
    ) -> str:
        """Replace a function in the code with new implementation."""
        # Find the function to replace
        old_function = self._extract_function_from_code(complete_code, function_name)

        if not old_function:
            # Function doesn't exist, append the new implementation
            return complete_code + "\n\n" + new_implementation

        # Replace the old function with the new one
        return complete_code.replace(old_function, new_implementation)

    def get_output_code_name(self, function_name: str) -> str:
        """Get the output code name."""
        return f"{function_name}_optimized.cpp"

    def compile_and_run(
        self, benchmark_name: str, benchmark_path: str, output_dir: str
    ) -> Dict:
        """Compile and run HeteroBench benchmark using make run_simple."""
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

    def analyze_run_output(self, run_output: str, benchmark_name: str) -> Dict:
        """Analyze HeteroBench benchmark output."""
        analysis = {
            "verification_success": True,  # Default to True, will be set to False if we find error words
            "original_time": None,
            "optimized_time": None,
            "speedup": None,
            "function_times": {},  # Store individual function times and speedups
        }

        lines = run_output.split("\n")
        in_performance_section = False
        in_original_section = False
        in_optimized_section = False
        in_speedup_section = False

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for verification results
            if "Verification Results:" in line:
                # Check next few lines for error words
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip().lower()
                    if any(
                        error_word in next_line
                        for error_word in ["incorrect", "fail", "wrong"]
                    ):
                        analysis["verification_success"] = False
                        break

            # Look for performance results
            elif "Performance Results:" in line:
                in_performance_section = True
                continue

            # Track which section we're in
            if in_performance_section:
                if "Original Implementation:" in line:
                    in_original_section = True
                    in_optimized_section = False
                    in_speedup_section = False
                    continue
                elif "Optimized Implementation:" in line:
                    in_original_section = False
                    in_optimized_section = True
                    in_speedup_section = False
                    continue
                elif "Speedup:" in line:
                    in_original_section = False
                    in_optimized_section = False
                    in_speedup_section = True
                    continue

                # Parse individual function times
                if in_original_section or in_optimized_section:
                    # Look for lines like "kernel_3mm_0 time: 6.65018 seconds"
                    if (
                        "time:" in line
                        and "seconds" in line
                        and "Single iteration time:" not in line
                    ):
                        try:
                            parts = line.split("time:")
                            if len(parts) == 2:
                                function_name = parts[0].strip()
                                time_str = parts[1].replace("seconds", "").strip()
                                time_value = float(time_str)

                                if function_name not in analysis["function_times"]:
                                    analysis["function_times"][function_name] = {}

                                if in_original_section:
                                    analysis["function_times"][function_name][
                                        "original_time_seconds"
                                    ] = time_value
                                elif in_optimized_section:
                                    analysis["function_times"][function_name][
                                        "optimized_time_seconds"
                                    ] = time_value
                        except (ValueError, IndexError):
                            pass

                # Parse individual function speedups
                elif in_speedup_section:
                    # Look for lines like "kernel_3mm_0: 3.47727x" (but not "Total:")
                    if ":" in line and "x" in line and "Total:" not in line:
                        try:
                            parts = line.split(":")
                            if len(parts) == 2:
                                function_name = parts[0].strip()
                                speedup_str = parts[1].replace("x", "").strip()
                                speedup_value = float(speedup_str)

                                if function_name not in analysis["function_times"]:
                                    analysis["function_times"][function_name] = {}

                                analysis["function_times"][function_name][
                                    "speedup"
                                ] = speedup_value
                        except (ValueError, IndexError):
                            pass

                # Extract total times and speedup
                if "Single iteration time:" in line:
                    try:
                        time_str = line.split(":")[-1].replace("seconds", "").strip()
                        time_value = float(time_str)

                        if in_original_section:
                            analysis["original_time"] = time_value
                        elif in_optimized_section:
                            analysis["optimized_time"] = time_value
                    except (ValueError, IndexError):
                        pass
                elif "Total:" in line and in_speedup_section:
                    try:
                        speedup_str = line.split(":")[-1].replace("x", "").strip()
                        analysis["speedup"] = float(speedup_str)
                    except (ValueError, IndexError):
                        pass

        return analysis

    def get_system_info(self) -> str:
        """Get CPU system information."""
        return """Machine we are using: 
- Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
- L1d cache: 1.5MB
- L1i cache: 1.5MB
- L2 cache: 48MB
- L3 cache: 71.5MB
- Supports SSE, AVX2, AVX512"""
