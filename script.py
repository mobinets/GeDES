import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import subprocess
import re
import threading
import time


# invoke executable file GeDES and capture output
def run_gedes(
    ft_k=32,
    packet_pool_size=60000000,
    output_file="data/flow_statics.csv",
    average_flow_size=100,
    flow_time_range=10000000,
):
    max_gpu_memory = 0  # Track maximum GPU memory usage

    def monitor_gpu_memory():
        """Monitor GPU memory usage in a separate thread"""
        nonlocal max_gpu_memory
        while monitor_thread.is_alive():
            try:
                # Get GPU memory usage using nvidia-smi
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    memory_used = int(result.stdout.strip())
                    if memory_used > max_gpu_memory:
                        max_gpu_memory = memory_used
            except:
                pass
            time.sleep(0.1)  # Check every 100ms

    try:
        # Start GPU memory monitoring thread
        monitor_thread = threading.Thread(target=monitor_gpu_memory, daemon=True)
        monitor_thread.start()

        # Build command with parameters
        command = [
            "./GeDES_app/GeDES",
            f"--ft_k={ft_k}",
            f"--packet_pool_size={packet_pool_size}",
            f"--output={output_file}",
            f"--average_flow_size={average_flow_size}",
            f"--flow_time_range={flow_time_range}",
        ]

        # Execute GeDES and capture output
        result = subprocess.run(command, capture_output=True, text=True, cwd=".")

        # Stop monitoring thread
        monitor_thread.join(timeout=1)

        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += "\nError: " + result.stderr

        return output, result.returncode, max_gpu_memory

    except FileNotFoundError:
        return "Error: GeDES executable not found", 1, 0
    except Exception as e:
        return f"Error: {str(e)}", 1, 0


def extract_simulation_duration(output):
    """Extract simulation duration from the output string"""
    pattern = r"Simulation Duration: (\d+\.\d+)s"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    return None


def get_available_gpu_memory():
    """Get available GPU memory in GB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / 1024  # Convert MB to GB
    except:
        pass
    return 0


params = [
    (8, 1000000, "data/flow_statics_8.csv", 10000, 1000000),
    (16, 5000000, "data/flow_statics_16.csv", 10000, 5000000),
    (32, 10000000, "data/flow_statics_32.csv", 10000, 10000000),
    (48, 20000000, "data/flow_statics_48.csv", 10000, 10000000),
    (64, 50000000, "data/flow_statics_64.csv", 1000, 20000000),
]

# left 0.5 GB for safety
required_memory = {
    8: 0.89 + 0.5,
    16: 2.09 + 0.5,
    32: 4.28 + 0.5,
    48: 9.23 + 0.5,
    64: 22.32 + 0.5,
}


def run_gedes_simulations(params_list, memory_requirements):
    """
    Run GeDES simulations for all parameter combinations with memory requirement check.

    Args:
        params_list (list): List of tuples containing simulation parameters in the format:
            (ft_k, packet_pool_size, output_file, average_flow_size, flow_time_range)
        memory_requirements (dict): Dictionary mapping ft_k values to required memory in GB

    Returns:
        tuple: (gpu_memory_usage, simulation_duration) where:
            gpu_memory_usage: List of dictionaries containing GPU memory usage data
            simulation_duration: List of simulation durations in seconds
    """
    gpu_memory_usage = []
    simulation_duration = []

    for (
        ft_k,
        packet_pool_size,
        output_file,
        average_flow_size,
        flow_time_range,
    ) in params_list:
        # Check memory requirement before running
        available_memory = get_available_gpu_memory()
        required_memory_gb = memory_requirements.get(ft_k, 0)

        print(
            f"\nChecking memory for ft_k={ft_k} ({ft_k**3/4}servers, and {ft_k**2/4*5} switches): available={available_memory:.2f} GB, required={required_memory_gb:.2f} GB"
        )

        if available_memory < required_memory_gb:
            print(
                f"⚠️  Skipping ft_k={ft_k}: Not enough GPU memory (available: {available_memory:.2f} GB, required: {required_memory_gb:.2f} GB)"
            )
            continue

        print(f"✅ Running GeDES with FatTree{ft_k}")

        output_str, return_code, max_gpu_memory = run_gedes(
            ft_k=ft_k,
            packet_pool_size=packet_pool_size,
            output_file=output_file,
            average_flow_size=average_flow_size,
            flow_time_range=flow_time_range,
        )

        # Store GPU memory usage
        gpu_memory_usage.append(
            {
                "ft_k": ft_k,
                "packet_pool_size": packet_pool_size,
                "output_file": output_file,
                "average_flow_size": average_flow_size,
                "flow_time_range": flow_time_range,
                "max_gpu_memory_mb": max_gpu_memory,
                "max_gpu_memory_gb": max_gpu_memory / 1024,
            }
        )

        # Extract and print simulation duration
        duration = extract_simulation_duration(output_str)
        if duration is not None:
            print(f"Simulation Duration: {duration:.5f} seconds")
        else:
            print("Simulation Duration not found in output")

        # Print GPU memory usage
        print(
            f"Maximum GPU Memory Usage: {max_gpu_memory / 1024:.2f} GB ({max_gpu_memory} MB)"
        )
        print("Simulation Duration: ", duration, "s")
        simulation_duration.append(duration)

    return gpu_memory_usage, simulation_duration


def create_results_table(gpu_memory_usage, simulation_duration):
    """Create a table from GPU memory usage and simulation duration data and save to PDF"""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import pandas as pd
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        import os

        # Create output directory if it doesn't exist
        output_dir = "img"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create DataFrame from GPU memory usage data
        df = pd.DataFrame(gpu_memory_usage)

        # Add simulation duration to DataFrame
        df["simulation_duration_seconds"] = simulation_duration

        # Select and reorder columns for the table
        table_data = df[["ft_k", "max_gpu_memory_gb", "simulation_duration_seconds"]]
        table_data.columns = [
            "FatTree-k",
            "Max GPU Memory (GB)",
            "Simulation Duration (s)",
        ]

        # Format floating point numbers to 2 decimal places
        table_data = table_data.copy()  # Avoid SettingWithCopyWarning
        table_data.loc[:, "Max GPU Memory (GB)"] = table_data[
            "Max GPU Memory (GB)"
        ].round(2)
        table_data.loc[:, "Simulation Duration (s)"] = table_data[
            "Simulation Duration (s)"
        ].round(2)

        # Create PDF with table
        output_path = os.path.join(output_dir, "simulation_results_table.pdf")

        with PdfPages(output_path) as pdf:
            # Create a figure for the table
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis("tight")
            ax.axis("off")

            # Create table
            table = ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # Add title
            plt.title("GeDES Simulation Results Summary", fontsize=16, pad=20)

            # Save to PDF
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"Results table successfully created and saved to {output_path}")
        return output_path

    except ImportError as e:
        print(f"Error: Required libraries not available - {e}")
        return None
    except Exception as e:
        print(f"Error creating results table: {e}")
        return None


def run_profilling_executables(executables=None):
    """
    Run executables and save output to separate TXT files

    Args:
        executables (list): List of tuples containing (executable_path, output_file)
                          Default: [("./GeDES_profilling_cache", "data/raw_active_rate_cache.txt"),
                                   ("./GeDES_profilling_disable_cache", "data/raw_active_rate_disable_cache.txt")]
    """
    if executables is None:
        executables = [
            ("GeDES_app/GeDES_profilling_cache", "data/raw_active_rate_cache.txt"),
            (
                "GeDES_app/GeDES_profilling_disable_cache",
                "data/raw_active_rate_disable_cache.txt",
            ),
        ]

    try:
        for executable, output_file in executables:
            with open(output_file, "w") as f:
                f.write(f"=== Output from {executable} ===\n\n")

                try:
                    # Execute the executable
                    result = subprocess.run(
                        [executable], capture_output=True, text=True, cwd="."
                    )

                    # Write stdout and stderr to file
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    f.write("\n")

                    if result.stderr:
                        f.write("STDERR:\n")
                        f.write(result.stderr)
                        f.write("\n")

                    f.write(f"Return code: {result.returncode}\n")

                except FileNotFoundError:
                    f.write(f"Error: Executable {executable} not found\n")
                except Exception as e:
                    f.write(f"Error running {executable}: {str(e)}\n")

                f.write("\n" + "=" * 50 + "\n\n")

            print(f"Profilling results saved to: {output_file}")

        return True

    except Exception as e:
        print(f"Error writing to files: {str(e)}")
        return False


def run_alignment_executables(executables=None):
    """
    Run alignment executables and save output to separate TXT files

    Args:
        executables (list): List of tuples containing (executable_path, output_file)
                          Default: [("./GeDES_alignment", "data/alignment_output.txt"),
                                   ("./GeDES_unalignment", "data/unalignment_output.txt")]

    Returns:
        bool: True if execution successful, False otherwise
    """
    if executables is None:
        executables = [
            ("GeDES_app/GeDES_alignment", "data/alignment_output.txt"),
            ("GeDES_app/GeDES_unalignment", "data/unalignment_output.txt"),
        ]

    try:
        for executable, output_file in executables:
            with open(output_file, "w") as f:
                f.write(f"=== Output from {executable} ===\n\n")

                try:
                    # Execute the executable
                    result = subprocess.run(
                        [executable], capture_output=True, text=True, cwd="."
                    )

                    # Write stdout and stderr to file
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    f.write("\n")

                    if result.stderr:
                        f.write("STDERR:\n")
                        f.write(result.stderr)
                        f.write("\n")

                    f.write(f"Return code: {result.returncode}\n")

                except FileNotFoundError:
                    f.write(f"Error: Executable {executable} not found\n")
                except Exception as e:
                    f.write(f"Error running {executable}: {str(e)}\n")

                f.write("\n" + "=" * 50 + "\n\n")

            print(f"Alignment results saved to: {output_file}")

        return True

    except Exception as e:
        print(f"Error writing to files: {str(e)}")
        return False


def process_alignment_data(
    alignment_file="data/alignment_output.txt",
    unalignment_file="data/unalignment_output.txt",
    output_filename="img/GPU_utilization_plot.pdf",
    alignment_label="Enable Workload Alignment",
    unalignment_label="Disable Workload Alignment",
):
    """
    Process alignment data files and create visualization comparison

    Args:
        alignment_file (str): Path to alignment data file
        unalignment_file (str): Path to unalignment data file
        output_filename (str): Output PDF filename for visualization
        alignment_label (str): Label for alignment data in visualization
        unalignment_label (str): Label for unalignment data in visualization

    Returns:
        bool: True if processing successful, False otherwise
    """
    print("\nProcessing alignment data...")

    try:
        from process_alignment_raw_data import process_and_plot_utilization

        # Process the alignment data files
        process_and_plot_utilization(
            [alignment_file, unalignment_file],
            output_filename,
            {alignment_file: alignment_label, unalignment_file: unalignment_label},
        )

        print(f"Alignment data processing completed! Generated {output_filename}")
        return True

    except ImportError as e:
        print(f"Error importing process_alignment_raw_data module: {e}")
        return False
    except Exception as e:
        print(f"Error processing alignment data: {e}")
        return False


def process_fct_data(
    csv_files=[param[2] for param in params],
    output_filename="img/flow_completion_time_cdf.pdf",
    legend_labels={param[2]: f"FatTree-{param[0]}" for param in params},
):
    """
    Process flow completion time data files and create CDF visualization

    Args:
        csv_files (list): List of CSV file paths containing flow statistics
        output_filename (str): Output PDF filename for CDF visualization
        legend_labels (dict): Dictionary mapping file paths to legend labels

    Returns:
        bool: True if processing successful, False otherwise
    """
    print("\nProcessing flow completion time data...")

    try:
        from process_fct_raw_data import process_fct_data as process_fct

        # Process the FCT data files
        process_fct(csv_files, output_filename, legend_labels)

        print(f"FCT data processing completed! Generated {output_filename}")
        return True

    except ImportError as e:
        print(f"Error importing process_fct_raw_data module: {e}")
        return False
    except Exception as e:
        print(f"Error processing FCT data: {e}")
        return False


def process_profiling_data(
    cache_file="data/raw_active_rate_cache.txt",
    disable_cache_file="data/raw_active_rate_disable_cache.txt",
    output_filename="img/active_rate_comparison.pdf",
    cache_label="With Cache",
    disable_cache_label="Without Cache",
):
    """
    Process profiling data files and create visualization comparison

    Args:
        cache_file (str): Path to profiling data file with cache enabled
        disable_cache_file (str): Path to profiling data file with cache disabled
        output_filename (str): Output PDF filename for visualization
        cache_label (str): Label for cache-enabled data in visualization
        disable_cache_label (str): Label for cache-disabled data in visualization

    Returns:
        bool: True if processing successful, False otherwise
    """
    print("\nProcessing profiling data...")

    try:
        from process_cache_raw_data import (
            process_and_smooth_active_rates,
            visualize_active_rate_comparison,
        )

        # Process the profiling data files
        smoothed_rates = process_and_smooth_active_rates(
            [cache_file, disable_cache_file]
        )

        # Create visualization comparison
        visualize_active_rate_comparison(
            smoothed_rates, [cache_label, disable_cache_label], output_filename
        )

        print(f"Profiling data processing completed! Generated {output_filename}")
        return True

    except ImportError as e:
        print(f"Error importing process_cache_raw_data module: {e}")
        return False
    except Exception as e:
        print(f"Error processing profiling data: {e}")
        return False


# Run GeDES for all parameter combinations with memory requirement check
gpu_memory_usage, simulation_duration = run_gedes_simulations(params, required_memory)

# After the main loop, create the results table
if gpu_memory_usage and simulation_duration:
    table_path = create_results_table(gpu_memory_usage, simulation_duration)
    print("\n")
    if table_path:
        print(f"Results table saved to: {table_path}")
    else:
        print("Failed to create results table")
else:
    print("No simulation data available to create results table")

# Run the profilling executables
cache_profiling = [
    ("GeDES_app/GeDES_profilling_cache", "data/raw_active_rate_cache.txt"),
    (
        "GeDES_app/GeDES_profilling_disable_cache",
        "data/raw_active_rate_disable_cache.txt",
    ),
]
run_profilling_executables(cache_profiling)

# # Process profiling data
process_profiling_data()

# Run alignment executables
alignment_executables = [
    ("GeDES_app/GeDES_alignment", "data/alignment_output.txt"),
    ("GeDES_app/GeDES_unalignment", "data/unalignment_output.txt"),
]
run_alignment_executables(alignment_executables)

# Process alignment data
process_alignment_data(
    alignment_file="data/alignment_output.txt",
    unalignment_file="data/unalignment_output.txt",
    output_filename="img/GPU_utilization_plot.pdf",
)

# Process flow completion time data
process_fct_data(
    csv_files=[param[2] for param in params],
    output_filename="img/flow_completion_time_cdf.pdf",
    legend_labels={param[2]: f"FatTree-{param[0]}" for param in params},
)
