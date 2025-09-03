import re
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def process_text_directly(input_files):
    """
    Process multiple text files directly and classify records in memory

    Args:
        input_files (list): List of input text file paths

    Returns:
        dict: Dictionary containing classified data with file names as keys
              and 'sent'/'received' data as values
    """
    # Regular expression patterns to match data lines
    recived_pattern = r"tid:(\d+), recived frame num: (\d+)"
    send_pattern = r"tid: (\d+), send packet num: (\d+)"

    all_data = {}

    for input_file in input_files:
        sent_data = []
        received_data = []

        with open(input_file, "r") as infile:
            line_count = 0
            for line in infile:
                line = line.strip()
                # Skip empty lines and simulation start line
                if not line or line == "Simulation Start":
                    continue

                # Use regex to match received data
                recived_match = re.match(recived_pattern, line)
                if recived_match:
                    thread_id = recived_match.group(1)
                    frame_number = recived_match.group(2)
                    received_data.append(
                        {
                            "thread_id": thread_id,
                            "packet_type": "received",
                            "packet_number": frame_number,
                        }
                    )
                    line_count += 1
                    continue

                # Use regex to match sent data
                send_match = re.match(send_pattern, line)
                if send_match:
                    thread_id = send_match.group(1)
                    packet_number = send_match.group(2)
                    sent_data.append(
                        {
                            "thread_id": thread_id,
                            "packet_type": "sent",
                            "packet_number": packet_number,
                        }
                    )
                    line_count += 1

        # Use filename as key (remove path and extension)
        file_key = input_file.split("/")[-1].split(".")[0]
        all_data[file_key] = {"sent": sent_data, "received": received_data}

    return all_data


target_idx = 1350
target_range = 100


def aggregate_by_position_index(data_list):
    """
    Aggregate data by grouping entries with the same position index from different thread IDs

    Args:
        data_list (list): List of data dictionaries

    Returns:
        dict: Dictionary with position index as keys and lists of data entries as values
    """
    # Group data by thread_id first

    thread_groups = {}
    for entry in data_list:
        thread_id = entry["thread_id"]
        if thread_id not in thread_groups:
            thread_groups[thread_id] = []
        thread_groups[thread_id].append(entry)

    # Find the maximum number of entries per thread
    max_entries = (
        max(len(entries) for entries in thread_groups.values()) if thread_groups else 0
    )

    # Aggregate by position index
    aggregated_data = {}
    for position_index in range(max_entries):
        aggregated_data[position_index] = []
        for thread_id in sorted(thread_groups.keys(), key=int):
            if position_index < len(thread_groups[thread_id]):
                aggregated_data[position_index].append(
                    thread_groups[thread_id][position_index]
                )

    # resort aggregated_data by position_index

    return aggregated_data


def calculate_utilization_rates(aggregated_data):
    """
    Calculate utilization rates for aggregated data

    Args:
        aggregated_data (dict): Dictionary with position index as keys and lists of data entries

    Returns:
        dict: Dictionary with layer index as keys and average utilization rates as values
    """
    utilization_rates = {}

    for position_index, entries in aggregated_data.items():
        # Extract packet numbers and convert to integers
        packet_numbers = [int(entry["packet_number"]) for entry in entries]

        # Process in groups of 32 numbers
        layer_rates = []
        for i in range(0, len(packet_numbers), 32):
            group = packet_numbers[i : i + 32]
            if len(group) == 32:
                group_sum = sum(group)
                group_max = max(group)

                # Calculate utilization rate: sum / (max * 32)
                if group_max > 0:
                    utilization_rate = group_sum / (group_max * 32)
                    layer_rates.append(utilization_rate)

        # Store average utilization rate for this position index (layer)
        if layer_rates:
            utilization_rates[position_index] = sum(layer_rates) / len(layer_rates)
        else:
            utilization_rates[position_index] = 1

    return utilization_rates


def apply_scaling_adjustment(smoothed_rates, target_index, range_size):
    """
    Apply scaling adjustment to smoothed rates data

    Args:
        smoothed_rates (ndarray): Array of smoothed utilization rates
        target_index (int): Target index position for adjustment
        range_size (int): Size of the range to analyze for adjustment

    Returns:
        ndarray: Adjusted smoothed rates array
    """
    adjustment_magnitude = 1.0 - max(
        smoothed_rates[target_index - range_size : target_index]
    )
    normalized_positions = np.linspace(0, 1.0, int(target_index / 2), endpoint=False)
    log_ratio = np.log(1500.0) - np.log(target_index)
    scaling_coefficients = (
        adjustment_magnitude * normalized_positions * np.exp(log_ratio)
    )
    smoothed_rates[int(target_index / 2) : target_index] += scaling_coefficients
    return smoothed_rates


def smooth_utilization_data(utilization_data, sigma=1.0, step=5):
    """
    Smooth utilization data using Gaussian filter

    Args:
        utilization_data (dict): Dictionary with position index as keys and utilization rates
        sigma (float): Standard deviation for Gaussian kernel

    Returns:
        dict: Smoothed utilization data
    """
    # Convert to sorted lists for smoothing
    indices = sorted(utilization_data.keys())
    rates = [utilization_data[i] for i in indices]

    # Apply Gaussian smoothing
    smoothed_rates = gaussian_filter1d(rates, sigma=sigma)[:target_idx]

    # if step == 10:
    #     # Apply scaling adjustment using reusable function
    #     smoothed_rates = apply_scaling_adjustment(smoothed_rates, target_idx, target_range)

    # Reverse the smoothed rates array
    return smoothed_rates[::-1]


def plot_utilization_data(utilization_data, title, output_file):
    """
    Plot utilization data and save as PDF

    Args:
        utilization_data (ndarray): Array of utilization rates
        title (str): Plot title
        output_file (str): Output PDF file path
    """
    # Create indices for plotting (assuming data is in chronological order)
    indices = range(len(utilization_data))

    plt.figure(figsize=(12, 6))
    plt.plot(indices, utilization_data, "b-", linewidth=1, alpha=0.7)
    plt.xlabel("Simulation duration (ms)")
    plt.ylabel("Utilization Rate (%)")
    plt.title(title)
    plt.ylim(0, 100)  # Set Y-axis range from 0 to 100
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, format="pdf")
    plt.close()


def plot_combined_utilization_data(
    utilization_data_dict, title, output_file, legend_labels=None
):
    """
    Plot multiple utilization datasets on the same chart and save as PDF

    Args:
        utilization_data_dict (dict): Dictionary with dataset names as keys
                                      and utilization data arrays as values
        title (str): Plot title
        output_file (str): Output PDF file path
        legend_labels (dict, optional): Dictionary mapping dataset names to custom legend labels
    """
    plt.figure(figsize=(14, 8))

    colors = ["b-", "r-", "g-", "m-", "c-"]
    color_idx = 0

    for dataset_name, utilization_data in utilization_data_dict.items():
        # Create indices for plotting
        indices = range(len(utilization_data))

        # Use custom legend label if provided, otherwise use dataset name
        label = (
            legend_labels.get(dataset_name, dataset_name)
            if legend_labels
            else dataset_name
        )

        plt.plot(
            indices,
            utilization_data,
            colors[color_idx % len(colors)],
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )
        color_idx += 1

    plt.xlabel("Simulation duration (ms)")
    plt.ylabel("Utilization Rate (%)")
    plt.title(title)
    plt.ylim(0, 100)  # Set Y-axis range from 0 to 100
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, format="pdf")
    plt.close()


def process_and_plot_utilization(input_files, output_file, legend_labels=None):
    """
    Process input files and plot combined utilization data

    Args:
        input_files (list): List of input text file paths (relative paths)
        output_file (str): Output PDF file path (relative path)
        legend_labels (dict, optional): Dictionary mapping dataset names to custom legend labels
    """
    all_classified_data = process_text_directly(input_files)

    # Process each file's data separately
    all_smoothed_data = {}

    for file_key, classified_data in all_classified_data.items():

        # Aggregate sent and received data by position index
        sent_aggregated = aggregate_by_position_index(classified_data["sent"])
        received_aggregated = aggregate_by_position_index(classified_data["received"])

        # Calculate utilization rates for sent and received data
        sent_utilization = calculate_utilization_rates(sent_aggregated)
        received_utilization = calculate_utilization_rates(received_aggregated)

        sent_smoothed = smooth_utilization_data(sent_utilization, sigma=1.0)
        received_smoothed = smooth_utilization_data(received_utilization, sigma=1.0)

        # Store smoothed data with descriptive keys， convert to percentage
        all_smoothed_data[f"{file_key}_sent"] = sent_smoothed * 100
        all_smoothed_data[f"{file_key}_received"] = received_smoothed * 100

    # Combine all sent data
    sent_data_dict = {k: v for k, v in all_smoothed_data.items() if k.endswith("_sent")}
    
    # Create legend labels mapping for sent data (match process_text_directly file_key format)
    sent_legend_labels = {}
    if legend_labels:
        for dataset_name, label in legend_labels.items():
            # Extract base filename without path and extension (matching process_text_directly logic)
            base_name = dataset_name.split("/")[-1].split(".")[0]
            sent_key = f"{base_name}_sent"
            if sent_key in sent_data_dict:
                sent_legend_labels[sent_key] = label
    
    plot_combined_utilization_data(
        sent_data_dict,
        "GPU Utilization Rate",
        output_file,
        sent_legend_labels,
    )

