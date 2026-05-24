from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass
import networkx as nx

# ============================================================
# USER INPUTS (only change these)
# ============================================================

N_LEAVES = 16
N_TREES = 10
BINARY_FLAG = 1     # 1 = binary output trees, 0 = non-binary output trees
TARGET_RETICULATIONS = 8 # Force the desired complexity

OUTPUT_FILE = "temporal_dataset.txt"

# ============================================================
# Predicates & Temporal network check
# ============================================================

def is_leaf(G: nx.DiGraph, v) -> bool:
    return G.out_degree(v) == 0

def is_reticulation(G: nx.DiGraph, v) -> bool:
    return G.in_degree(v) >= 2 and G.out_degree(v) == 1

def assign_depth_times(G: nx.DiGraph, root) -> Dict:
    times: Dict = {root: 0}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in G.successors(u):
            if v not in times:
                times[v] = times[u] + 1
                queue.append(v)
    nx.set_node_attributes(G, times, "time")
    return times

def is_temporal_network(G: nx.DiGraph, root) -> bool:
    if not nx.is_directed_acyclic_graph(G):
        return False
    if G.in_degree(root) != 0:
        return False
    for u, v in G.edges():
        tu = G.nodes[u].get("time", None)
        tv = G.nodes[v].get("time", None)
        if tu is None or tv is None:
            return False
        if is_reticulation(G, v):
            if tu != tv:
                return False
        else:
            if tu >= tv:
                return False
    return True

# ============================================================
# Random rooted binary base tree 
# ============================================================

def generate_random_binary_tree(leaf_labels: Sequence[str], rng: random.Random) -> Tuple[nx.DiGraph, str]:
    G = nx.DiGraph()
    for x in leaf_labels:
        G.add_node(x)

    pool: List[str] = list(leaf_labels)
    rng.shuffle(pool) # Shuffle so that pairs are random
    internal_id = 0

    # Build the tree level by level to ensure branches are on the same time layer
    while len(pool) > 1:
        next_pool = []
        for i in range(0, len(pool) - 1, 2):
            a = pool[i]
            b = pool[i+1]
            parent = f"I{internal_id}"
            internal_id += 1

            G.add_node(parent)
            G.add_edge(parent, a)
            G.add_edge(parent, b)
            next_pool.append(parent)
        
        if len(pool) % 2 != 0:
            next_pool.append(pool[-1])
            
        pool = next_pool

    root = pool[0]
    assign_depth_times(G, root)
    return G, root

# ============================================================
# Add temporal reticulations
# ============================================================

@dataclass
class ReticulationInfo:
    node: str
    time: int
    child: str
    base_parent: str
    extra_parent: str

def _descendants_set(G: nx.DiGraph, v) -> Set:
    return set(nx.descendants(G, v))

def add_temporal_reticulation(
    G: nx.DiGraph, root: str, rng: random.Random, reticulation_id: int, max_attempts: int = 4000
) -> Optional[ReticulationInfo]:
    edges = list(G.edges())

    for _ in range(max_attempts):
        x, y = edges[rng.randrange(len(edges))]

        if is_reticulation(G, y): continue
        if is_leaf(G, x) or is_reticulation(G, x): continue

        tx = G.nodes[x]["time"]
        ty = G.nodes[y]["time"]
        if tx >= ty: continue

        tree_children = [c for c in G.successors(x) if not is_reticulation(G, c)]
        if y in tree_children and len(tree_children) < 2: continue

        desc_y = _descendants_set(G, y)
        candidates_u2 = [
            u for u in G.nodes()
            if u != x and not is_leaf(G, u) and not is_reticulation(G, u)
            and G.nodes[u].get("time", None) == tx and u not in desc_y
        ]
        if not candidates_u2: continue

        u2 = candidates_u2[rng.randrange(len(candidates_u2))]
        rnode = f"H{reticulation_id}"
        if rnode in G: continue
        if not G.has_edge(x, y): continue

        G.add_node(rnode, time=tx)
        G.remove_edge(x, y)
        G.add_edge(x, rnode)
        G.add_edge(u2, rnode)
        G.add_edge(rnode, y)

        if G.out_degree(rnode) != 1 or G.in_degree(rnode) < 2 or not is_temporal_network(G, root):
            if G.has_edge(x, rnode): G.remove_edge(x, rnode)
            if G.has_edge(u2, rnode): G.remove_edge(u2, rnode)
            if G.has_edge(rnode, y): G.remove_edge(rnode, y)
            if rnode in G: G.remove_node(rnode)
            G.add_edge(x, y)
            continue

        return ReticulationInfo(node=rnode, time=tx, child=y, base_parent=x, extra_parent=u2)
    return None

def build_temporal_network(
    n_leaves: int, n_reticulations: int, seed: int, max_restarts: int = 1000
) -> Tuple[nx.DiGraph, str, List[str], int]:
    rng = random.Random(seed)
    leaves = [f"x{i+1}" for i in range(n_leaves)]

    for _ in range(max_restarts):
        G, root = generate_random_binary_tree(leaves, rng)
        added = 0
        ok = True
        for rid in range(n_reticulations):
            info = add_temporal_reticulation(G, root, rng, reticulation_id=rid)
            if info is None:
                ok = False
                break
            added += 1

        if ok and is_temporal_network(G, root):
            return G, root, leaves, added

    raise RuntimeError("Failed to build a temporal network for the chosen parameters.")

# ============================================================
# Tree extraction + Newick
# ============================================================

def _reachable_from_root(G: nx.DiGraph, root) -> Set:
    return {root} | set(nx.descendants(G, root))

def suppress_degree_two_nodes(G: nx.DiGraph, root, leaf_set: Set[str]) -> str:
    while root in G and G.in_degree(root) == 0 and G.out_degree(root) == 1 and root not in leaf_set:
        child = next(iter(G.successors(root)))
        G.remove_node(root)
        root = child

    changed = True
    while changed:
        changed = False
        for v in list(G.nodes()):
            if v == root or v in leaf_set: continue
            if G.in_degree(v) == 1 and G.out_degree(v) == 1:
                p = next(iter(G.predecessors(v)))
                c = next(iter(G.successors(v)))
                if G.has_edge(p, v): G.remove_edge(p, v)
                if G.has_edge(v, c): G.remove_edge(v, c)
                if p != c: G.add_edge(p, c)
                if v in G: G.remove_node(v)
                changed = True
                break
    return root

def is_rooted_tree(G: nx.DiGraph, root, leaf_set: Set[str]) -> bool:
    if not nx.is_directed_acyclic_graph(G): return False
    if G.in_degree(root) != 0: return False
    for v in G.nodes():
        if v == root: continue
        if G.in_degree(v) != 1: return False
    leaves = {v for v in G.nodes() if G.out_degree(v) == 0}
    return leaves == leaf_set

def is_binary_tree(G: nx.DiGraph, root: str, leaf_set: Set[str]) -> bool:
    for v in G.nodes():
        if v in leaf_set:
            if G.out_degree(v) != 0: return False
        else:
            if G.out_degree(v) != 2: return False
    return True

def has_polytomy(G: nx.DiGraph, leaf_set: Set[str]) -> bool:
    return any((v not in leaf_set) and (G.out_degree(v) > 2) for v in G.nodes())

def to_newick(G: nx.DiGraph, root, leaf_set: Set[str]) -> str:
    def rec(v) -> str:
        if v in leaf_set: return v
        children = list(G.successors(v))
        children.sort(key=str)
        return "(" + ",".join(rec(c) for c in children) + ")"
    return rec(root) + ";"

def sample_displayed_tree_graph(network: nx.DiGraph, root: str, leaf_labels: Sequence[str], rng: random.Random) -> Tuple[nx.DiGraph, str]:
    leaf_set = set(leaf_labels)
    H = network.copy()

    retic_nodes = [v for v in H.nodes() if is_reticulation(H, v)]
    for r in retic_nodes:
        parents = list(H.predecessors(r))
        keep = parents[rng.randrange(len(parents))]
        for p in parents:
            if p != keep and H.has_edge(p, r):
                H.remove_edge(p, r)

    reachable = _reachable_from_root(H, root)
    for v in list(H.nodes()):
        if v not in reachable:
            H.remove_node(v)

    new_root = suppress_degree_two_nodes(H, root, leaf_set)
    if not is_rooted_tree(H, new_root, leaf_set):
        raise RuntimeError("Sampled structure is not a rooted tree.")
    return H, new_root

# ============================================================
# DETERMINISTIC BINARIZATION (Prevents Fake Conflicts)
# ============================================================

def binarize_tree_fast(G: nx.DiGraph, root: str, leaf_set: Set[str], rng: random.Random) -> Tuple[nx.DiGraph, str]:
    H = G.copy()
    internal_counter = 0

    for v in list(H.nodes()):
        if v in leaf_set: continue
        children = list(H.successors(v))
        if len(children) <= 2: continue

        # SORT ALPHABETICALLY: Ensures all trees binarize the same way
        children.sort(key=str)

        for c in children:
            if H.has_edge(v, c):
                H.remove_edge(v, c)

        H.add_edge(v, children[0])
        prev = v
        for i in range(1, len(children) - 1):
            bnode = f"B_{v}_{internal_counter}"
            internal_counter += 1
            H.add_node(bnode)
            H.add_edge(prev, bnode)
            H.add_edge(bnode, children[i])
            prev = bnode

        H.add_edge(prev, children[-1])

    new_root = suppress_degree_two_nodes(H, root, leaf_set)
    if not is_binary_tree(H, new_root, leaf_set):
        raise RuntimeError("Fast binarization failed (tree not binary after resolution).")
    return H, new_root

def force_nonbinary_newick(G: nx.DiGraph, root: str, leaf_set: Set[str], rng: random.Random, p_contract: float = 0.25, max_tries: int = 30) -> str:
    for _ in range(max_tries):
        H = G.copy()
        candidates = [v for v in H.nodes() if v != root and v not in leaf_set and H.in_degree(v) == 1 and H.out_degree(v) == 2]
        rng.shuffle(candidates)
        for v in candidates:
            if rng.random() > p_contract: continue
            p = next(iter(H.predecessors(v)))
            children = list(H.successors(v))
            if H.has_edge(p, v): H.remove_edge(p, v)
            for c in children:
                if H.has_edge(v, c): H.remove_edge(v, c)
                if not H.has_edge(p, c): H.add_edge(p, c)
            if v in H: H.remove_node(v)
        new_root = suppress_degree_two_nodes(H, root, leaf_set)
        if is_rooted_tree(H, new_root, leaf_set) and has_polytomy(H, leaf_set):
            return to_newick(H, new_root, leaf_set)

    H = G.copy()
    candidates = [v for v in H.nodes() if v != root and v not in leaf_set and H.in_degree(v) == 1 and H.out_degree(v) == 2]
    if candidates:
        v = rng.choice(candidates)
        p = next(iter(H.predecessors(v)))
        children = list(H.successors(v))
        if H.has_edge(p, v): H.remove_edge(p, v)
        for c in children:
            if H.has_edge(v, c): H.remove_edge(v, c)
            if not H.has_edge(p, c): H.add_edge(p, c)
        if v in H: H.remove_node(v)
        new_root = suppress_degree_two_nodes(H, root, leaf_set)
        if is_rooted_tree(H, new_root, leaf_set):
            return to_newick(H, new_root, leaf_set)
    return to_newick(G, root, leaf_set)

# ============================================================
# Dataset generator 
# ============================================================

def generate_dataset(n_leaves: int, n_trees: int, binary_flag: int, target_retics: int) -> Tuple[int, int, List[str]]:
    seed_used = random.SystemRandom().randrange(1, 10**9)
    h_try = target_retics
    
    while h_try >= 1:
        try:
            net, root, leaves, retics_used = build_temporal_network(n_leaves, h_try, seed_used)
            break
        except RuntimeError:
            print(f"Failed with {h_try} reticulations. Stepping down to {h_try - 1}...")
            h_try -= 1
            seed_used += 1000
    else:
        raise RuntimeError("Could not build any temporal network.")

    rng = random.Random(seed_used + 1)
    leaf_set = set(leaves)
    trees: List[str] = []
    max_attempts_per_tree = 2000

    for _ in range(n_trees):
        for _attempt in range(max_attempts_per_tree):
            T, rT = sample_displayed_tree_graph(net, root, leaves, rng)
            if binary_flag == 1:
                T_bin, rT_bin = binarize_tree_fast(T, rT, leaf_set, rng)
                trees.append(to_newick(T_bin, rT_bin, leaf_set))
                break
            else:
                nw = force_nonbinary_newick(T, rT, leaf_set, rng, p_contract=0.25)
                trees.append(nw)
                break
        else:
            raise RuntimeError("Could not sample a tree satisfying the requested binarity.")

    if not is_temporal_network(net, root):
        raise RuntimeError("Internal error: generated network is not temporal.")

    return seed_used, retics_used, trees

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    if BINARY_FLAG not in (0, 1):
        raise ValueError("BINARY_FLAG must be 0 or 1.")

    print(f"Attempting to generate temporal network with {N_LEAVES} leaves and {TARGET_RETICULATIONS} target reticulations...")
    seed_used, reticulations_used, trees_newick = generate_dataset(N_LEAVES, N_TREES, BINARY_FLAG, TARGET_RETICULATIONS)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"n_leaves={N_LEAVES}\n")
        f.write(f"n_trees={N_TREES}\n")
        f.write(f"binary_flag={BINARY_FLAG}  # 1=binary, 0=non-binary\n")
        f.write(f"reticulations_used={reticulations_used}\n")
        f.write(f"seed_used={seed_used}\n")
        f.write("\n")
        for t in trees_newick:
            f.write(t + "\n")

    print("\n SUCCESS! Saved to:", OUTPUT_FILE)
    print("Final generated reticulations:", reticulations_used)
    print("Unique trees:", len(set(trees_newick)), "out of", len(trees_newick))
