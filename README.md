# Temporal Cherry Picking with MCTS and Neural Networks

This repository contains a suite of tools to solve the Temporal Cherry Picking Sequence (CPS) problem for phylogenetic networks. It leverages Monte Carlo Tree Search (MCTS) algorithms, enhanced with scale-invariant feature extraction and Deep Learning (Neural Networks), to efficiently find valid pruning sequences in forests of phylogenetic trees.

## Repository Structure

The project is divided into four main Python scripts:

* **`FullBenchmark.py`**: The primary entry point for running automated experiments. It orchestrates the data generation, runs multiple search algorithms, and compiles the results into comprehensive logs and Excel/CSV summaries.
* **`MainAlgorithmDefinition.py`**: The core logic hub. It contains the implementations of the search algorithms (Classic MCTS, MCTS+NN, Random Search), backtracking logic, and the pipeline for training the Neural Network.
* **`SyntheticDataGenerator.py`**: Generates synthetic temporal phylogenetic networks and extracts binary (or non-binary) tree sets from them. Used by the benchmark but fully functional on its own.
* **`HierarchicalFeatureExtracture.py`**: Extracts scale-invariant topological features from phylogenetic trees using Biopython. These features are fed into the Neural Network to guide the MCTS algorithm.

---

## Requirements

Ensure you have Python 3.8+ installed along with the following dependencies:


pip install biopython networkx pandas numpy scikit-learn joblib openpyxl

*(Note: If you plan to compare against the optimal C++ solver, you will also need WSL/Bash configured as called in the benchmark script. The optimal C++ solver can be found in https://github.com/mathcals/temporal_hybridization_number)*

---

## How to Use

### 1. Running the Benchmark (Default Workflow)

The easiest way to use this repository is through the automated benchmark.

1.  Open **`FullBenchmark.py`**.
2.  Locate the `# Configuration` and `EXPERIMENTS` section at the top of the file.
3.  Adjust the test parameters to your liking (e.g., number of leaves, number of trees, target reticulations, binary flag, and repetitions).
4.  Run the script:

python FullBenchmark.py

**Outputs:**
The benchmark will create an output folder (e.g., `limit_iterations_2/`) containing:
* `results.xlsx` / `results.csv`: A complete summary of the performance of all algorithms (hybrids found, execution time, etc.).
* `datasets/`: The generated `.txt` files containing the Newick trees for each experiment.
* `logs/`: Detailed console outputs and execution logs for every individual run.

### 2. Tweak Algorithm Hyperparameters

If you want to modify how the AI or the search algorithms behave, open **`MainAlgorithmDefinition.py`**. Inside, you can:
* Change the number of simulations per step or MCTS iterations.
* Adjust the exploration parameter (`c_param`).
* Swap out the active algorithm version (e.g., enable/disable backtracking or "panic resets").
* Modify the `MAX_BRANCH` limit for state expansion.

### 3. Retraining the Neural Network

To train the Neural Network on a new dataset or with different hyperparameters:

1.  Open **`MainAlgorithmDefinition.py`**.
2.  Scroll down to the `# TRAINING OF THE NEURAL NETWORK` section.
3.  Uncomment the training block at the end of that section.
4.  Point the script to your folder of training `.txt` tree files.
5.  Run the script directly. It will generate and save a new `.joblib` model file that the benchmark can then load.

### 4. Standalone Synthetic Data Generation

You can generate specific temporal networks and trees without running a full benchmark. 

1.  Open **`SyntheticDataGenerator.py`**.
2.  Modify the `USER INPUTS` section at the top (`N_LEAVES`, `N_TREES`, `BINARY_FLAG`, `TARGET_RETICULATIONS`).
3.  Run the script:

```bash
python SyntheticDataGenerator.py
```
This will output a `.txt` file with the generated Newick trees that you can manually inspect or pass to other tools.
