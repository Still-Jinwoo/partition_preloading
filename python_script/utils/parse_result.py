# I'll update the file content with better error handling and logging.
from pathlib import Path
import pandas as pd
import re
from typing import Optional, Type

def get_number_from_line(line, number_idx, mode: Optional[Type] = int):
    numbers = re.findall(r"[-e.0-9]*", line)
    numbers = [x for x in numbers if (len(x) > 0 and x != 'e')]
    if mode == int:
        return int(numbers[number_idx])
    elif mode == float:
        return float(numbers[number_idx])
    else:
        raise Exception("Invalid number type requested.")
    
# def round_dict_values(dictionary, decimal_places=5):
#     """Round the values in a dictionary that contain lists of floats."""
#     for key, value in dictionary.items():
#         if isinstance(value[0], list):  # List of lists
#             dictionary[key][0] = [round(v, decimal_places) for v in value[0]]
#         elif isinstance(value[0], float):  # Single float value
#             dictionary[key][0] = round(value[0], decimal_places)
            
def parse_celeritas_output(input_file: Path) -> pd.DataFrame:

    assert(input_file.exists())

    dict_results = {
        "init_time": [-1],
        "epoch_time": [[]],
        "valid_acc": [[]],
        "valid_mr": [[]],
        "valid_mrr": [[]],
        "valid_hits1": [[]],
        "valid_hits3": [[]],
        "valid_hits5": [[]],
        "valid_hits10": [[]],
        "valid_hits50": [[]],
        "valid_hits100": [[]],
        "test_acc": [[]],
        "test_mr": [[]],
        "test_mrr": [[]],
        "test_hits1": [[]],
        "test_hits3": [[]],
        "test_hits5": [[]],
        "test_hits10": [[]],
        "test_hits50": [[]],
        "test_hits100": [[]],
    }

    valid = True

    with open(input_file, "r") as f:
        for line in f.readlines():
            if "Initialization" in line:
                dict_results["init_time"][0] = float(line.split()[-1][:-1])

            if "Epoch Runtime" in line:
                dict_results["epoch_time"][0].append(float(line.split()[-1][:-2]) / 1000.0)

            if "Evaluating validation set" in line:
                valid = True

            if "Evaluating test set" in line:
                valid = False

            if "Accuracy" in line:
                if valid:
                    dict_results["valid_acc"][0].append(float(line.split()[-1][:-1]))
                else:
                    dict_results["test_acc"][0].append(float(line.split()[-1][:-1]))

            if "Mean Rank" in line:
                if valid:
                    dict_results["valid_mr"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_mr"][0].append(float(line.split()[-1]))

            if "MRR" in line:
                if valid:
                    dict_results["valid_mrr"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_mrr"][0].append(float(line.split()[-1]))

            if "Hits@1:" in line:
                if valid:
                    dict_results["valid_hits1"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits1"][0].append(float(line.split()[-1]))

            if "Hits@3:" in line:
                if valid:
                    dict_results["valid_hits3"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits3"][0].append(float(line.split()[-1]))

            if "Hits@5:" in line:
                if valid:
                    dict_results["valid_hits5"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits5"][0].append(float(line.split()[-1]))

            if "Hits@10:" in line:
                if valid:
                    dict_results["valid_hits10"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits10"][0].append(float(line.split()[-1]))

            if "Hits@50:" in line:
                if valid:
                    dict_results["valid_hits50"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits50"][0].append(float(line.split()[-1]))

            if "Hits@100:" in line:
                if valid:
                    dict_results["valid_hits100"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits100"][0].append(float(line.split()[-1]))

    return pd.DataFrame(dict_results)

# def parse_dstat(input_file: Path):
#     map_columns = {
#         "time": "Timestamp",
#         "usr": "CPU User Utilization",
#         "sys": "CPU Sys Utilization",
#         "read.1": "Bytes Read",
#         "writ.1": "Bytes Written",
#         "used": "Memory Used"
#     }

#     try:
#         # Read the file without specifying a separator
#         with open(input_file, 'r') as file:
#             lines = file.readlines()

#         # Skip the first few lines if necessary (header lines)
#         data = [line.strip().split('|') for line in lines[5:]]

#         # Convert to DataFrame
#         df = pd.DataFrame(data)

#         # Attempt to rename the columns based on known structure
#         df.columns = ["Timestamp", "CPU", "Disk", "Memory", "Network"]

#         # Clean up column data if necessary
#         df['Timestamp'] = df['Timestamp'].str.strip()

#         # If 'Timestamp' column is missing, raise an error
#         if 'Timestamp' not in df.columns:
#             raise KeyError("Timestamp column not found in dstat output.")

#         # Convert the 'Timestamp' column to datetime
#         df["Timestamp"] = pd.to_datetime(df['Timestamp'], format="%d-%m %H:%M:%S") + pd.offsets.DateOffset(years=120)

#         df = df[:-1]  # Drop the last row if it is incomplete

#     except KeyError as e:
#         print(f"Unexpected error while parsing {input_file}: {e}")
#         print(f"Columns found in the file: {df.columns}")
#         return pd.DataFrame()  # Return an empty DataFrame
#     except pd.errors.EmptyDataError:
#         print(f"Error: {input_file} contains no valid data.")
#         return pd.DataFrame()  # Return an empty DataFrame
#     except Exception as e:
#         print(f"Unexpected error while parsing {input_file}: {e}")
#         return pd.DataFrame()  # Return an empty DataFrame

#     return df

def parse_dstat(input_file: Path):
    map_columns = {"time": "Timestamp",
                   "usr": "CPU User Utilization",
                   "sys": "CPU Sys Utilization",
                   "read.1": "Bytes Read",
                   "writ.1": "Bytes Written",
                   "used": "Memory Used"}

    df = pd.read_csv(input_file, header=5).rename(columns=map_columns)
    df = df[:-1] # last line might not have been completely written, ignore it
    #df["Timestamp"] = pd.to_datetime(df['Timestamp'], format="%d-%m %H:%M:%S") + pd.offsets.DateOffset(years=120)
    return df

def parse_nvidia_smi(input_file: Path):
    map_columns = {" utilization.memory [%]": "GPU Memory Utilization",
                   " utilization.gpu [%]": "GPU Compute Utilization",
                   " memory.used [MiB]": "GPU Memory Used",
                   " timestamp": "Timestamp"}

    if not input_file.exists() or input_file.stat().st_size == 0:
        print(f"Warning: {input_file} is empty or does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame

    try:
        df = pd.read_csv(input_file).rename(columns=map_columns)
        df = df[:-1]  # Last line might not have been completely written, ignore it
        df['Timestamp'] = df['Timestamp'].str.strip()  # Remove leading/trailing spaces
        try:
            df["Timestamp"] = pd.to_datetime(df['Timestamp'], format="%Y/%m/%d %H:%M:%S.%f")
        except ValueError as e:
            print(f"ValueError: {e}")
            print(f"First few timestamps: {df['Timestamp'].head()}")
            df["Timestamp"] = pd.to_datetime(df['Timestamp'], format="%Y/%m/%d %H:%M:%S")  # Adjust the format
        df['GPU Memory Utilization'] = df['GPU Memory Utilization'].str.rstrip('%').astype('float') / 100.0
        df['GPU Compute Utilization'] = df['GPU Compute Utilization'].str.rstrip('%').astype('float') / 100.0
        df['GPU Memory Used'] = df['GPU Memory Used'].str.rstrip(" MiB").astype('float')
    except pd.errors.EmptyDataError:
        print(f"Error: {input_file} contains no valid data.")
        return pd.DataFrame()  # Return an empty DataFrame
    except Exception as e:
        print(f"Unexpected error while parsing {input_file}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

    return df
