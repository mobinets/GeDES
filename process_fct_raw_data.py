import os
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def read_flow_statics_files():
    """
    Read all files starting with 'flow_statics' in the data directory and store them in dictionary format
    
    Returns:
        dict: Dictionary with filenames as keys and file contents (DataFrame) as values
    """
    data_dir = Path("data")
    flow_statics_files = {}
    
    # Find all files starting with 'flow_statics'
    for file_path in data_dir.glob("flow_statics*"):
        if file_path.is_file():
            # Read CSV file
            df = pd.read_csv(file_path)
            # Use filename (without extension) as key
            file_name = file_path.stem
            flow_statics_files[file_name] = df
    
    return flow_statics_files

def extract_flow_completion_times(flow_statics_data, convert_to_ms=True):
    """
    Extract flow_completion_time data from all flow_statics files
    
    Args:
        flow_statics_data (dict): Dictionary containing flow_statics DataFrames
        convert_to_ms (bool): Whether to convert from nanoseconds to milliseconds
    
    Returns:
        dict: Dictionary with filenames as keys and flow_completion_time arrays as values
    """
    completion_times = {}
    
    for file_name, df in flow_statics_data.items():
        if 'flow_completion_time' in df.columns:
            # Convert from nanoseconds to milliseconds if requested
            if convert_to_ms:
                completion_times[file_name] = df['flow_completion_time'].values / 1_000_000.0
            else:
                completion_times[file_name] = df['flow_completion_time'].values
    
    return completion_times

def calculate_cdf(data, normalize=True):
    """
    Calculate Cumulative Distribution Function (CDF) for given data
    
    Args:
        data (array-like): Input data
        normalize (bool): Whether to normalize CDF to [0,1]
    
    Returns:
        tuple: (sorted_data, cdf_values)
    """
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data) if normalize else np.arange(1, len(sorted_data) + 1)
    
    return sorted_data, cdf_values

def plot_cdf(completion_times_dict, output_file="flow_completion_time_cdf.pdf", legend_labels=None):
    """
    Plot CDF for flow completion times from multiple files
    
    Args:
        completion_times_dict (dict): Dictionary with flow completion time data
        output_file (str): Output plot filename
        legend_labels (dict): Dictionary mapping file names to legend labels
    """
    plt.figure(figsize=(10, 6))
    
    # Find global min and max for x-axis alignment
    all_data = []
    for data in completion_times_dict.values():
        if len(data) > 0:
            all_data.extend(data)
    
    if all_data:
        x_min = np.min(all_data)
        x_max = np.max(all_data)
    else:
        x_min, x_max = 0, 1
    
    for file_name, data in completion_times_dict.items():
        if len(data) > 0:
            sorted_data, cdf_values = calculate_cdf(data)
            label = legend_labels[file_name] if legend_labels and file_name in legend_labels else file_name
            plt.plot(sorted_data, cdf_values, label=label, linewidth=2)
    
    # Set x-axis limits for precise alignment
    plt.xlim(x_min, x_max)
    
    plt.xlabel('Flow Completion Time (ms)')
    plt.ylabel('CDF')
    plt.title('CDF of Flow Completion Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save as PDF
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    plt.close()

def process_fct_data(csv_file_paths, output_image_path, legend_labels=None):
    """
    Process flow completion time data from CSV files and generate CDF plot
    
    Args:
        csv_file_paths (list): List of paths to CSV files
        output_image_path (str): Path for output image file
        legend_labels (dict): Dictionary mapping file names to legend labels
    """
    # Read specified CSV files
    flow_statics_files = {}
    
    for file_path in csv_file_paths:
        file_path = Path(file_path)
        if file_path.is_file():
            # Read CSV file
            df = pd.read_csv(file_path)
            # Use filename (without extension) as key
            file_name = file_path.stem
            flow_statics_files[file_name] = df
    
    # Extract flow completion time data and convert to milliseconds
    completion_times = extract_flow_completion_times(flow_statics_files, convert_to_ms=True)
    
    # Plot CDF
    plot_cdf(completion_times, output_file=output_image_path, legend_labels=legend_labels)
