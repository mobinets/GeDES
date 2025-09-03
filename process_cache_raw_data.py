import csv
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# Regular expression pattern to match data lines
pattern = r"tid: (\d+), transmit_num: (\d+), total queue size: (\d+)"


def convert_txt_to_csv(txt_file_path, csv_file_path):
    """
    Convert txt format data to CSV format

    Args:
        txt_file_path (str): Input txt file path
        csv_file_path (str): Output csv file path
    """
    # Prepare CSV file
    with open(txt_file_path, "r") as infile, open(
        csv_file_path, "w", newline=""
    ) as outfile:
        csv_writer = csv.writer(outfile)

        # Write CSV header
        csv_writer.writerow(["tid", "transmit_num", "total_queue_size"])

        # Process each line
        for line in infile:
            # Skip non-data lines (e.g., "Simulation Start")
            if line.strip() == "Simulation Start":
                continue

            # Use regular expression to match data
            match = re.match(pattern, line.strip())
            if match:
                tid = match.group(1)
                transmit_num = match.group(2)
                total_queue_size = match.group(3)

                # Write CSV row
                csv_writer.writerow([tid, transmit_num, total_queue_size])


def calculate_positional_sums(csv_file_path):
    """
    Calculate sum values by position index for different tids

    Args:
        csv_file_path (str): Input CSV file path containing tid, transmit_num, total_queue_size columns

    Returns:
        pandas.DataFrame: Result DataFrame containing position_index, transmit_num_sum, total_queue_size_sum
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Add position index for each tid's data
    df["position_index"] = df.groupby("tid").cumcount()

    # Group by position index, calculate sum of transmit_num and total_queue_size for each position
    result_df = (
        df.groupby("position_index")
        .agg({"transmit_num": "sum", "total_queue_size": "sum"})
        .reset_index()
    )

    # Rename columns
    result_df.columns = ["position_index", "transmit_num_sum", "total_queue_size_sum"]

    return result_df


def process_and_smooth_active_rates(txt_files):
    """
    Process txt files and calculate smoothed active rate

    Args:
        txt_files (list): List of txt file paths

    Returns:
        list: List of smoothed active rate data
    """
    output_files = [f"{os.path.splitext(txt_file)[0]}.csv" for txt_file in txt_files]
    results = []

    for input_file, output_file in zip(txt_files, output_files):
        convert_txt_to_csv(input_file, output_file)
        # Use function to process csv file
        result = calculate_positional_sums(output_file)
        active_rate = result["transmit_num_sum"] / result["total_queue_size_sum"]
        results.append(active_rate)

    min_element_num = 0
    for result in results:
        if min_element_num == 0:
            min_element_num = len(result)
        else:
            min_element_num = min(min_element_num, len(result))
            
    # Align results
    for i, result in enumerate(results):
        # Drop middle elements to align all results to the same length
        if len(result) > min_element_num:
            # Calculate the number of elements to drop from the middle
            drop_count = len(result) - min_element_num
            start_index = (len(result) - drop_count) // 2
            end_index = start_index + drop_count
            
            # Drop the middle elements
            results[i] = result.drop(result.index[start_index:end_index])
            # Reset index to consecutive integers starting from 0
            results[i] = results[i].reset_index(drop=True)

    # Calculate smoothed active rate (window size 10)
    smoothed_active_rates = []
    for active_rate in results:
        smoothed = active_rate.rolling(window=10, min_periods=1).mean()
        smoothed_active_rates.append(smoothed)

    return smoothed_active_rates


def visualize_active_rate_comparison(smoothed_rates, labels, output_filename='active_rate_comparison.pdf'):
    """
    Create PDF visualization of active rate comparison

    Args:
        smoothed_rates (list): List of smoothed active rate data
        labels (list): List of labels for each dataset
        output_filename (str): Output PDF filename
    """
    # Create PDF file
    with PdfPages(output_filename) as pdf:
        # Create single subplot layout
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Set colors
        colors = ['blue', 'red']
        
        # Smoothed active rate comparison plot
        for i, smoothed in enumerate(smoothed_rates):
            ax.plot(smoothed, color=colors[i], label=labels[i], linewidth=2)
        ax.set_title('Active Rate Comparison')
        ax.set_xlabel('Simulation Duration (us)')
        ax.set_ylabel('Active Rate (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        # Save to PDF
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)

