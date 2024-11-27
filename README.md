## Installation

To install dependencies, follow these steps:

1. **Initial Setup**:
   - Open your terminal and navigate to the root directory of this repository.
   - Run the following command:
     ```bash
     pip3 install .
     ```
   - This command checks your system environment, installs necessary software dependencies, compiles C/C++ code, and sets up Python bindings with pybind11.

## Execution

After installing Celeritas, you can run it using the following steps:

1. **Navigate to Script Directory**:
   - Change your current directory to `python_script` within the repository.

2. **Running the Model**:
   - To execute the script, use:
     ```bash
     python3 run.py
     ```
   - You can specify the dataset for the experiment using one of the following commands:
     - For `ogbn-arxiv` dataset:
       ```bash
       python3 run.py --experiment instance_arxiv
       ```
     - For `ogbn-paper100M` dataset:
       ```bash
       python3 run.py --experiment instance_paper100M
       ```

3. **Output Options**:
   - To display the results in the console, add the `--show_output_console` flag:
     ```bash
     python3 run.py --show_output_console
     ```
   - Without this flag, results are saved in the `python_script/results` directory.

