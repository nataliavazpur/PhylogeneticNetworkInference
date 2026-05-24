import os
import re
import time
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd  # pip install pandas openpyxl

from treegeneratorbinaryornot import generate_dataset

# =========================
# Configuration
# =========================

THESIS_SCRIPT = "ThesisCleanWithNN.py"
TREESDOCUMENT_FILE = "trees.txt"
OUTPUT_DIR = "limit_iterations_2"


# We decide the complexity of each test. These parameters can be changed. 
#Binary_flag=1 to enforce binarity
EXPERIMENTS = [
    {"n_leaves": 30,  "n_trees": 10, "binary_flag": 1, "target_retics": 4, "reps": 5},
    {"n_leaves": 50,  "n_trees": 10, "binary_flag": 1, "target_retics": 4, "reps": 5},
    {"n_leaves": 60,  "n_trees": 10, "binary_flag": 1, "target_retics": 4, "reps": 5},
    {"n_leaves": 90,  "n_trees": 10, "binary_flag": 1, "target_retics": 4, "reps": 5},
    {"n_leaves": 100, "n_trees": 10, "binary_flag": 1, "target_retics": 4, "reps": 5},
    {"n_leaves": 120, "n_trees": 10, "binary_flag": 1, "target_retics": 4, "reps": 5},
]

# =========================
# Data structure
# =========================

@dataclass
class RunResult:
    sample_id: int
    n_trees: int
    n_leaves: int
    binary_flag: int
    seed_used: int
    upper_bound: int
    trees_file: str

    # MCTS + NN
    mcts_nn_hybrids: Optional[float]
    mcts_nn_time_s: Optional[float]
    
    # Classic MCTS
    mcts_hybrids: Optional[float]
    mcts_time_s: Optional[float]
    
    # Trivial Random
    trivial_rand_hybrids: Optional[float]
    trivial_rand_time_s: Optional[float]
    
    # Pure Random
    pure_rand_hybrids: Optional[float]
    pure_rand_time_s: Optional[float]

    # Optimal C++
    optimal_hybrids: Optional[float]
    optimal_time_s: Optional[float]

    status: str
    raw_log_file: str


# =========================
# Helpers
# =========================

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)


def write_treesdocument(newick_trees: List[str]) -> None:
    with open(TREESDOCUMENT_FILE, "w", encoding="utf-8") as f:
        for t in newick_trees:
            f.write(t.strip() + "\n")


def save_dataset_file(sample_id: int, n_leaves: int, n_trees: int, binary_flag: int,
                      seed_used: int, reticulations_used: int, newick_trees: List[str]) -> str:
    binary_label = "binary" if binary_flag == 1 else "non-binary"
    filename = f"dataset_{sample_id:04d}_L{n_leaves}_T{n_trees}_{binary_label}.txt"
    path = os.path.join(OUTPUT_DIR, "datasets", filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"n_leaves={n_leaves}\n")
        f.write(f"n_trees={n_trees}\n")
        f.write(f"binary_flag={binary_flag}\n")
        f.write(f"reticulations_used={reticulations_used}\n")
        f.write(f"seed_used={seed_used}\n\n")
        for t in newick_trees:
            f.write(t.strip() + "\n")

    return path


def parse_thesis_output(stdout: str) -> Dict[str, Optional[float]]:
    out = {
        "mcts_nn_hybrids": None, "mcts_nn_time_s": None,
        "mcts_hybrids": None, "mcts_time_s": None,
        "trivial_rand_hybrids": None, "trivial_rand_time_s": None,
        "pure_rand_hybrids": None, "pure_rand_time_s": None,
    }

    # Internal function to search for the number of hybridizations in a text block
    def extract_hybrid(text: str) -> Optional[float]:
        m = re.search(r"(?:Hybridizations|Minimum hybrids|Hibridaciones mínimas):\s*([0-9]+|inf|None)", text, re.IGNORECASE)
        if m:
            val = m.group(1).lower()
            if val in ["none", "inf"]: return float("inf")
            return float(val)
        return None

    # Internal function to search for the time in a text block
    def extract_time(text: str) -> Optional[float]:
        m = re.search(r"(?:time \(in seconds\)|Tiempo total|time.*?:)\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None

    # Separate the console text by algorithms to avoid mixing data
    headers = {
        "mcts_nn": ["MCTS with NN", "INICIANDO MCTS+NN"],
        "mcts": ["MCTS ALGORITHM", "MCTS Clásico"],
        "trivial_rand": ["Random Search Algorithm"],
        "pure_rand": ["PURE Random Search", "PURE Random"]
    }

    positions = []
    for key, alt_list in headers.items():
        pos = -1
        for alt in alt_list:
            idx = stdout.find(alt)
            if idx != -1:
                pos = idx
                break
        if pos != -1:
            positions.append((pos, key))

    positions.sort() # Sort by order of appearance

    blocks = {}
    for i in range(len(positions)):
        pos, key = positions[i]
        end_pos = positions[i+1][0] if i + 1 < len(positions) else len(stdout)
        blocks[key] = stdout[pos:end_pos]

    # Extract variables from each block found
    if "mcts_nn" in blocks:
        out["mcts_nn_hybrids"] = extract_hybrid(blocks["mcts_nn"])
        out["mcts_nn_time_s"] = extract_time(blocks["mcts_nn"])

    if "mcts" in blocks:
        out["mcts_hybrids"] = extract_hybrid(blocks["mcts"])
        out["mcts_time_s"] = extract_time(blocks["mcts"])

    if "trivial_rand" in blocks:
        out["trivial_rand_hybrids"] = extract_hybrid(blocks["trivial_rand"])
        out["trivial_rand_time_s"] = extract_time(blocks["trivial_rand"])

    if "pure_rand" in blocks:
        out["pure_rand_hybrids"] = extract_hybrid(blocks["pure_rand"])
        out["pure_rand_time_s"] = extract_time(blocks["pure_rand"])

    return out


def run_thesis_script() -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", THESIS_SCRIPT],
        capture_output=True,
        text=True
    )


def run_optimal_cpp(trees_file: str) -> subprocess.CompletedProcess:
    cmd = [
        "wsl",
        "bash",
        "-lc",
        f"cd ~/temporal_hybridization_number && "
        f"./cherrypick_cpp --input {trees_file}"
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_optimal_output(stdout: str) -> Dict[str, Optional[float]]:
    out = {
        "optimal_value": None,
        "optimal_time_s": None,
    }
    m = re.search(r"value\s*=\s*([0-9]+)", stdout)
    if m:
        out["optimal_value"] = float(m.group(1))

    m = re.search(r"running_time\s*=\s*([0-9]*\.?[0-9]+)", stdout)
    if m:
        out["optimal_time_s"] = float(m.group(1))

    return out


def n_trees_label(n_trees: int, binary_flag: int) -> str:
    return f"{n_trees} ({'binary' if binary_flag == 1 else 'non binary'})"


# =========================
# Main
# =========================

def main() -> None:
    ensure_dirs()
    results: List[RunResult] = []
    sample_id = 0

    for exp in EXPERIMENTS:
        for _ in range(exp["reps"]):
            sample_id += 1

            
            seed_used, reticulations_used, trees_newick = generate_dataset(
                exp["n_leaves"], 
                exp["n_trees"], 
                exp["binary_flag"], 
                exp["target_retics"]  # <-- Pass the exact value from the dictionary
            )
            dataset_path = save_dataset_file(
                sample_id,
                exp["n_leaves"],
                exp["n_trees"],
                exp["binary_flag"],
                seed_used,
                reticulations_used,
                trees_newick
            )

            write_treesdocument(trees_newick)

            start_wall = time.time()
            proc = run_thesis_script()
            wall_s = time.time() - start_wall

            opt_proc = run_optimal_cpp(TREESDOCUMENT_FILE)
            opt_parsed = parse_optimal_output(opt_proc.stdout or "")

            log_filename = f"log_{sample_id:04d}.txt"
            log_path = os.path.join(OUTPUT_DIR, "logs", log_filename)

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("==== PYTHON STDOUT ====\n")
                f.write(proc.stdout or "")
                f.write("\n\n==== PYTHON STDERR ====\n")
                f.write(proc.stderr or "")
                f.write(f"\n\n==== PYTHON WALL TIME (s) ====\n{wall_s}\n")
                f.write("\n\n==== OPTIMAL (C++) ====\n")
                f.write(opt_proc.stdout or "")
                f.write("\n")

            if proc.returncode != 0:
                results.append(RunResult(
                    sample_id=sample_id, n_trees=exp["n_trees"], n_leaves=exp["n_leaves"],
                    binary_flag=exp["binary_flag"], seed_used=seed_used,
                    upper_bound=reticulations_used, trees_file=dataset_path,
                    mcts_nn_hybrids=None, mcts_nn_time_s=None,
                    mcts_hybrids=None, mcts_time_s=None,
                    trivial_rand_hybrids=None, trivial_rand_time_s=None,
                    pure_rand_hybrids=None, pure_rand_time_s=None,
                    optimal_hybrids=None, optimal_time_s=None,
                    status=f"ERROR (returncode={proc.returncode})",
                    raw_log_file=log_path
                ))
                continue

            parsed = parse_thesis_output(proc.stdout or "")

            results.append(RunResult(
                sample_id=sample_id, n_trees=exp["n_trees"], n_leaves=exp["n_leaves"],
                binary_flag=exp["binary_flag"], seed_used=seed_used,
                upper_bound=reticulations_used, trees_file=dataset_path,
                mcts_nn_hybrids=parsed["mcts_nn_hybrids"], mcts_nn_time_s=parsed["mcts_nn_time_s"],
                mcts_hybrids=parsed["mcts_hybrids"], mcts_time_s=parsed["mcts_time_s"],
                trivial_rand_hybrids=parsed["trivial_rand_hybrids"], trivial_rand_time_s=parsed["trivial_rand_time_s"],
                pure_rand_hybrids=parsed["pure_rand_hybrids"], pure_rand_time_s=parsed["pure_rand_time_s"],
                optimal_hybrids=opt_parsed["optimal_value"], optimal_time_s=opt_parsed["optimal_time_s"],
                status="OK", raw_log_file=log_path
            ))

    table_rows: List[Dict[str, Any]] = []
    for r in results:
        table_rows.append({
            "Sample Set": r.sample_id,
            "n° trees": n_trees_label(r.n_trees, r.binary_flag),
            "n° leaves": r.n_leaves,
            "MCTS+NN (n°)": r.mcts_nn_hybrids,
            "MCTS+NN (s)": r.mcts_nn_time_s,
            "MCTS (n°)": r.mcts_hybrids,
            "MCTS (s)": r.mcts_time_s,
            "Trivial Rand (n°)": r.trivial_rand_hybrids,
            "Trivial Rand (s)": r.trivial_rand_time_s,
            "Pure Rand (n°)": r.pure_rand_hybrids,
            "Pure Rand (s)": r.pure_rand_time_s,
            "Optimal (n°)": r.optimal_hybrids,
            "Optimal (s)": r.optimal_time_s,
            "Upper Bound": r.upper_bound,
            "seed_used": r.seed_used,
            "status": r.status,
            "log_file": r.raw_log_file,
        })

    df = pd.DataFrame(table_rows)

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)

    xlsx_path = os.path.join(OUTPUT_DIR, "results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)

    print("\n DONE")
    print("Excel:", xlsx_path)
    print("Logs:", os.path.join(OUTPUT_DIR, "logs"))


if __name__ == "__main__":
    main()