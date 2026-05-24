# ------------------------------------------------------------
# Feature extraction for CPS (Temporal Cherry Picking) using Biopython trees.
# Works with Bio.Phylo.Newick.Tree objects.
# Scale-invariant features for Transfer Learning
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd

# ==========================================
# 1. OPTIMIZED HELPER FUNCTIONS (Caches)
# ==========================================

def compute_depths(tree) -> Tuple[Dict[Any, float], float]:
    """Calculates depths of all clades and returns the maximum depth."""
    depths = {}
    max_depth = [0.0]  # Using a list to allow mutation inside the recursion
    
    def _dfs(clade, d):
        depths[clade] = d
        if d > max_depth[0]:
            max_depth[0] = d
            
        for child in clade.clades:
            bl = child.branch_length
            depths[child] = d + (1.0 if bl is None else float(bl))
            _dfs(child, depths[child])
            
    if tree.root:
        _dfs(tree.root, 0.0)
    return depths, max_depth[0]

def build_name_to_clade_map(tree) -> Dict[str, Any]:
    return {terminal.name: terminal for terminal in tree.get_terminals()}

def build_parent_map(tree) -> Dict[Any, Any]:
    parent_map = {}
    for clade in tree.find_clades():
        for child in clade.clades:
            parent_map[child] = clade
    return parent_map

# ==========================================
# 2. CONFIGURATION & MAIN CLASS
# ==========================================

@dataclass
class FeatureConfig:
    eps: float = 1e-12
    leaf_aggs: Tuple[str, ...] = ("mean", "max", "std")
    state_aggs: Tuple[str, ...] = ("mean", "max", "std")

class CPSFeatureExtractor:
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.cfg = config if config is not None else FeatureConfig()

    def compute_cherry_feature_table(
        self,
        trees: List[Any],
        reducible_pairs: Dict[Tuple[str, str], Set[int]],
        all_depths: List[Dict[Any, float]],
        all_max_depths: List[float],
        all_name_maps: List[Dict[str, Any]],
        all_parent_maps: List[Dict[Any, Any]],
        num_trees: int
    ) -> pd.DataFrame:
        pairs = list(reducible_pairs.keys())
        rows = []

        for (x, y) in pairs:
            tset = reducible_pairs[(x, y)]
            count = len(tset)
            freq = count / max(num_trees, 1)

            leaf_depth_vals = []
            dist_vals = []
            parent_depth_vals = []

            for t_idx in tset:
                tree = trees[t_idx]
                depths = all_depths[t_idx]
                max_d = all_max_depths[t_idx] if all_max_depths[t_idx] > 0 else 1.0
                name_map = all_name_maps[t_idx]
                parent_map = all_parent_maps[t_idx]

                ca = name_map.get(x)
                cb = name_map.get(y)

                if ca and cb:
                    # NORMALIZATION: Divide by the maximum depth of the tree
                    dx_norm, dy_norm = depths[ca] / max_d, depths[cb] / max_d
                    leaf_depth_vals.append(0.5 * (dx_norm + dy_norm))
                    
                    pa = parent_map.get(ca)
                    if pa:
                        dp_norm = depths[pa] / max_d
                        parent_depth_vals.append(dp_norm)
                        dist_vals.append(dx_norm + dy_norm - 2.0 * dp_norm)

            rows.append({
                "pair": (x, y),
                "count": float(count),
                "freq": float(freq),
                "tree_coverage": float(freq),
                "mean_leaf_depth": float(np.mean(leaf_depth_vals)) if leaf_depth_vals else 0.0,
                "mean_pair_distance": float(np.mean(dist_vals)) if dist_vals else 0.0,
                "mean_cherry_parent_depth": float(np.mean(parent_depth_vals)) if parent_depth_vals else 0.0,
            })

        if not rows:
            return pd.DataFrame(columns=["count", "freq", "tree_coverage", "mean_leaf_depth", "mean_pair_distance", "mean_cherry_parent_depth"])
        
        return pd.DataFrame(rows).set_index("pair")

    def compute_leaf_feature_table(self, cherry_features: pd.DataFrame, remaining_leaves: Set[str]) -> pd.DataFrame:
        leaves = sorted(list(remaining_leaves))
        max_deg = max(len(remaining_leaves) - 1, 1)
        pair_index = list(cherry_features.index)
        leaf_to_pairs: Dict[str, List[Tuple[str, str]]] = {l: [] for l in leaves}
        
        for (a, b) in pair_index:
            if a in leaf_to_pairs: leaf_to_pairs[a].append((a, b))
            if b in leaf_to_pairs: leaf_to_pairs[b].append((a, b))

        feature_cols = [c for c in cherry_features.columns if c != "pair"]
        rows = []

        for x in leaves:
            pairs_x = leaf_to_pairs.get(x, [])
            deg = float(len(pairs_x))
            row = {"leaf": x, "deg": deg, "deg_norm": deg / float(max_deg)}

            if deg == 0:
                for col in feature_cols:
                    for agg in self.cfg.leaf_aggs: row[f"{col}_{agg}"] = 0.0
            else:
                sub = cherry_features.loc[pairs_x, feature_cols]
                for col in feature_cols:
                    vals = sub[col].values
                    if "mean" in self.cfg.leaf_aggs: row[f"{col}_mean"] = float(np.mean(vals))
                    if "max" in self.cfg.leaf_aggs: row[f"{col}_max"] = float(np.max(vals))
                    if "std" in self.cfg.leaf_aggs: row[f"{col}_std"] = float(np.std(vals))
            rows.append(row)

        return pd.DataFrame(rows).set_index("leaf")

    def compute_state_feature_vector_from_leaf_table(
        self, leaf_features: pd.DataFrame, remaining_leaves: Set[str], 
        initial_leaf_count: int, hybrids_so_far: float = 0.0
    ) -> Tuple[np.ndarray, List[str]]:
        cols = list(leaf_features.columns)
        vals = leaf_features[cols].values

        feat_vec = []
        feat_names = []

        for j, col in enumerate(cols):
            col_vals = vals[:, j]
            if "mean" in self.cfg.state_aggs:
                feat_vec.append(float(np.mean(col_vals)))
                feat_names.append(f"{col}_state_mean")
            if "max" in self.cfg.state_aggs:
                feat_vec.append(float(np.max(col_vals)))
                feat_names.append(f"{col}_state_max")
            if "std" in self.cfg.state_aggs:
                feat_vec.append(float(np.std(col_vals)))
                feat_names.append(f"{col}_state_std")

        # NORMALIZATION: Percentages instead of raw values
        safe_N = float(max(initial_leaf_count, 1))
        rem_ratio = float(len(remaining_leaves)) / safe_N
        hyb_ratio = float(hybrids_so_far) / safe_N

        feat_vec.extend([rem_ratio, hyb_ratio])
        feat_names.extend(["remaining_leaves_ratio", "hybrids_ratio"])

        return np.array(feat_vec, dtype=float), feat_names

    def extract_all(
        self, trees: List[Any], remaining_leaves: Set[str], 
        initial_leaf_count: int = None, hybrids_so_far: float = 0.0
    ) -> Dict[str, Any]:
        
        all_depths_info = [compute_depths(t) for t in trees]
        all_depths = [info[0] for info in all_depths_info]
        all_max_depths = [info[1] for info in all_depths_info]
        
        all_name_maps = [build_name_to_clade_map(t) for t in trees]
        all_parent_maps = [build_parent_map(t) for t in trees]
        
        # If the initial count is not provided, we assume the total number of current leaves in tree 0
        if initial_leaf_count is None:
            initial_leaf_count = max(len(all_name_maps[0]), 1)
            
        reducible_pairs = {}
        for t_idx, tree in enumerate(trees):
            term_set = set(all_name_maps[t_idx].values())
            for clade in tree.find_clades():
                childs = clade.clades
                if len(childs) == 2 and childs[0] in term_set and childs[1] in term_set:
                    x, y = childs[0].name, childs[1].name
                    if x in remaining_leaves and y in remaining_leaves:
                        pair = tuple(sorted((x, y)))
                        if pair not in reducible_pairs: 
                            reducible_pairs[pair] = set()
                        reducible_pairs[pair].add(t_idx)

        cherry_df = self.compute_cherry_feature_table(
            trees, reducible_pairs, all_depths, all_max_depths, all_name_maps, all_parent_maps, len(trees)
        )
        leaf_df = self.compute_leaf_feature_table(cherry_df, remaining_leaves)
        state_vec, state_names = self.compute_state_feature_vector_from_leaf_table(
            leaf_df, remaining_leaves, initial_leaf_count, hybrids_so_far
        )
    
        return {
            "reducible_pairs": reducible_pairs,
            "cherry_features": cherry_df,
            "leaf_features": leaf_df,
            "state_vector": state_vec,
            "state_feature_names": state_names,
        }