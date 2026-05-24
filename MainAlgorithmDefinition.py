from Bio import Phylo
from io import StringIO
from FeaturesMyThesis import CPSFeatureExtractor
import math
import numpy as np
import random  
import copy 
import joblib
import time 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

############################################################

# Initialize the global cache. It resets on each execution.
evaluation_cache = {}

###################################################################
##################   FUNCTION  DEFINITIONS  #######################
###################################################################


def is_cherry(tree, leaf):  
    """
    Uses Biopython methods to decide whether a leaf belongs to a cherry 
    """
    # 1. Find the target leaf clade
    target = None
    for clade in tree.find_clades():
        if clade.name == leaf:
            target = clade
            break

    if target is None:
        return False

    # 2. Obtain the path to get the parent
    path = tree.get_path(target)
    if len(path) < 2:
        return False   # root or isolated → cannot be cherry

    parent = path[-2]

    # 3. Parent must have exactly two children
    if len(parent.clades) != 2:
        return False

    # 4. Identify sibling
    siblings = [c for c in parent.clades if c is not target]
    sibling = siblings[0]

    # 5. Both must be leaves (terminal nodes)
    return target.is_terminal() and sibling.is_terminal()

def load_newick_trees_as_list(filename):
    """
    Reads a text file where each line contains a Newick tree.
    Returns a list of trees: [T1, T2, T3, ...]
    """
    trees = []
    
    with open(filename, "r") as file:
        lines = file.readlines()
    
    for line in lines:
        newick = line.strip()
        if newick == "":
            continue
        
        tree = Phylo.read(StringIO(newick), "newick")
        trees.append(tree)
    
    return trees

def get_leaves_from_trees(tree_list):
    """
    Defines the set of leaves from the trees
    """
    leaves = set()
    
    for t in tree_list:
        for leaf in t.get_terminals():
            leaves.add(leaf.name)
    
    return leaves

def prune_leaf(tree, leaf_name):
    """
   Prunes a leaf from a tree
    """
    if tree.root is None:
        return tree

    # 1. Try to prune the node, catching errors if it's missing or the last leaf.
    try:
        tree.prune(leaf_name)
    except ValueError:
        # If we get an error, we check whether the tree is composed of only that leaf
        terminals = tree.get_terminals()
        if len(terminals) == 1 and terminals[0].name == leaf_name:
            tree.root = None
        return tree

    # 2. Cleanup: collapse single-child nodes.
    def clean_up(clade):
        if clade.is_terminal():
            return clade
        
        clade.clades = [clean_up(c) for c in clade.clades if c is not None]
        
        if len(clade.clades) == 1:
            child = clade.clades[0]
            clade.name = child.name
            clade.clades = child.clades
            clade.branch_length = (clade.branch_length or 0) + (child.branch_length or 0)
            
        while len(clade.clades) > 2:
            from Bio.Phylo.Newick import Clade
            a = clade.clades.pop(0)
            b = clade.clades.pop(0)
            clade.clades.insert(0, Clade(clades=[a, b]))
            
        return clade

    if tree.root:
        tree.root = clean_up(tree.root)
        if not tree.root.is_terminal() and len(tree.root.clades) == 1:
            tree.root = tree.root.clades[0]
            
    return tree

def evaluate_state_with_value_nn(state, value_model, feature_extractor):
    # 1. Logic shortcut: Divide by 10.0 to maintain the same scale as the AI
    if len(state.remaining) <= 1:
        return -float(state.hybrids) / 10.0

    # 2. Cache: Add initial_leaf_count to the cache key to ensure uniqueness
    state_key = (frozenset(state.remaining), state.hybrids, state.initial_leaf_count)
    
    if state_key in evaluation_cache:
        return evaluation_cache[state_key]

    # # 3. Extractor: Provide the static anchor so it can compute the percentage
    out = feature_extractor.extract_all(
        state.trees,
        state.remaining,
        initial_leaf_count=state.initial_leaf_count, # <-- we send the anchor (the initial number of leaves)
        hybrids_so_far=state.hybrids
    )
    
    phi_s = out["state_vector"].reshape(1, -1)
    v_s = value_model.predict(phi_s)[0]
    val = float(v_s)
    
    evaluation_cache[state_key] = val
    return val

def get_sibling(tree, leaf):
    # Find the corresponding leaf clade
    target = None
    for clade in tree.find_clades():
        if clade.name == leaf:
            target = clade
            break

    if target is None:
        return None

    path = tree.get_path(target)
    if len(path) < 2:
        return None

    parent = path[-2]

    if len(parent.clades) != 2:
        return None

    # The other child is the sibling
    for c in parent.clades:
        if c is not target:
            return c.name

    return None

def pure_random_cherry_picking(trees, remaining_leaves):
    """
    Finds a completely random (but legal) pruning sequence,
    summing the actual hybridizations when choosing nontrivial options.
    """
    # Deep copy to avoid altering the original trees during this iteration
    trees = [copy.deepcopy(t) for t in trees]
    remaining = set(remaining_leaves)
    
    sequence = []
    total_hybrids = 0

    while len(remaining) > 0:
        # SPECIAL CASE: If only 1 leaf remains, prune it and finish
        if len(remaining) == 1:
            last_leaf = next(iter(remaining))
            sequence.append(last_leaf)
            remaining.remove(last_leaf)
            break

        
        consistent, inconsistent, weights = get_global_cherries(trees, remaining)
        
        # Combine all legal options (perfect cherries + conflicting cherries)
        valid_options = consistent + inconsistent
        
        if len(valid_options) == 0:
            # Deadlock: no valid cherries available, this path is a dead end
            return None, None
            
        # 1. Choose a COMPLETELY random option from the legal ones
        chosen = random.choice(valid_options)
        
        # 2. Add the actual hybridization cost
        total_hybrids += weights.get(chosen, 0)
        
        # 3. Prune the leaf across the entire forest
        for i in range(len(trees)):
            trees[i] = prune_leaf(trees[i], chosen)
            
        # 4. Update state
        sequence.append(chosen)
        remaining.remove(chosen)

    return sequence, total_hybrids

def get_global_cherries(trees, remaining_leaves):
    """
    Original/strict version: Only considers a leaf as a candidate 
    if it is a cherry in ALL trees across the forest.
    consistent corresponds to trivial cherries, as they share the same sibling
    consistently across all trees, while non-trivial cherries correspond to 
    inconsistent
    """
    remaining_leaves = list(remaining_leaves)
    consistent = []
    inconsistent = []
    weights = {}

    # SPECIAL CASE: 1 leaf remaining
    if len(remaining_leaves) == 1:
        leaf = remaining_leaves[0]
        return [leaf], [], {leaf: 0}

    # SPECIAL CASE: 2 leaves remaining → cherry
    if len(remaining_leaves) == 2:
        leaf1, leaf2 = remaining_leaves
        return [leaf1, leaf2], [], {leaf1: 0, leaf2: 0}

    # Other cases
    for leaf in remaining_leaves:
        siblings = []
        is_global = True

        for t in trees:
            if not is_cherry(t, leaf):
                # If it is not a cherry in this tree, discard the leaf entirely
                is_global = False
                break
            sib = get_sibling(t, leaf)
            siblings.append(sib)

        if not is_global:
            continue

        # If execution reaches here, the leaf is a cherry in all trees.
        # Now, calculate the hybridization cost (based on distinct siblings).
        k = len(set(siblings))
        weights[leaf] = k - 1

        # Classify based on whether the sibling is the same across all trees (consistent/trivial) or not (inconsistent/non-trivial)
        if k == 1:
            consistent.append(leaf)
        else:
            inconsistent.append(leaf)

    return consistent, inconsistent, weights

def full_cherry_picking_zero_prints(trees, remaining_leaves):
    """
    Executes a complete cherry-picking sequence across the set of trees, prioritizing 
    trivial cherries over nontrivial ones, and returns the pruning sequence 
    along with the total hybridization cost.
    """
    # Make deep copies so original trees are untouched
    trees = [copy.deepcopy(t) for t in trees]
    remaining = set(remaining_leaves)

    sequence = []
    hybrids = 0
    step = 1  # debugging step counter

    while len(remaining) > 0:

    # SPECIAL CASE: only one leaf left → finish directly
        if len(remaining) == 1:
            last_leaf = next(iter(remaining))
            sequence.append(last_leaf)
            remaining.remove(last_leaf)
            break

        # Get cherries
        consistent, inconsistent, weights = get_global_cherries(trees, remaining)

   
        # CASE 3: No cherries at all → ERROR
        if len(consistent) == 0 and len(inconsistent) == 0:
            return None, None

            
        # CASE 1: If we have consistent cherries → pick one
        if len(consistent) > 0:
            chosen = random.choice(consistent)
            is_global = True

        # CASE 2: No consistent but inconsistent exist → pick one
        else:
            chosen = random.choice(inconsistent)
            is_global = False

        # Apply pruning to all trees
        for i in range(len(trees)):
            trees[i] = prune_leaf(trees[i], chosen)

        # Update hybrids count
        if not is_global:
            hybrids += weights[chosen]

        # Update remaining leaves
        remaining.remove(chosen)

        # Add to sequence
        sequence.append(chosen)

 
        step += 1


    return sequence, hybrids



###################################################################
############ CLASS STATE AND CLASS NODE ###########################
###################################################################



class State:
    def __init__(self, trees, remaining_leaves, sequence=None, hybrids=0, initial_leaf_count=None):
        self.trees = [copy.deepcopy(t) for t in trees]
        self.remaining = set(remaining_leaves)
        self.sequence = list(sequence) if sequence else []
        self.hybrids = hybrids
        self.initial_leaf_count = initial_leaf_count
        self._cached_actions = None
        self._cached_weights = None

    def is_terminal(self):
        return len(self.remaining) == 0  #there are 0 remaining leaves on the terminal sets 

    
    def get_actions(self):
        # 0) if cached, return immediately
        if self._cached_actions is not None:
            return self._cached_actions
    
        rem = list(self.remaining)
        
        # special case: 0 → no actions
        if len(rem) == 0:
            self._cached_actions = []
            self._cached_weights = {}
            return self._cached_actions
    
        # special case: 1 → only action is that leaf
        if len(rem) == 1:
            self._cached_actions = rem
            self._cached_weights = {}
            return self._cached_actions
    
        # special case: 2 → both valid
        if len(rem) == 2:
            self._cached_actions = rem
            self._cached_weights = {}
            return self._cached_actions
    
        # otherwise normal cherry computation
        consistent, inconsistent, weights = get_global_cherries(self.trees, self.remaining)
        
        # ---- BRANCH LIMITING ----
        # keep all consistent first, then sample inconsistent to fill up to MAX_BRANCH
        actions = list(consistent)
    
        if len(actions) < MAX_BRANCH:
            inc = list(inconsistent)
            random.shuffle(inc)
            actions += inc[: (MAX_BRANCH - len(actions))]
        else:
            random.shuffle(actions)
            actions = actions[:MAX_BRANCH]

        self._cached_actions = actions
        self._cached_weights = weights
        return self._cached_actions

    def apply_action(self, leaf):
        # Instead of deepcopying the entire list, we copy the trees individually
        import copy
        
        # Create new tree copies for the new MCTS branch
        # Using a list comprehension for better performance
        new_trees = [prune_leaf(copy.deepcopy(t), leaf) for t in self.trees]
        
        new_remaining = self.remaining.copy()
        new_remaining.remove(leaf)
        
        new_sequence = self.sequence + [leaf]
        
        # Retrieve the cost (hybrids) from the cache, which should be populated in get_actions
        if self._cached_weights is None:
            _, _, self._cached_weights = get_global_cherries(self.trees, self.remaining)
        
        new_hybrids = self.hybrids + self._cached_weights.get(leaf, 0)
        
        #Inherit initial_leaf_count by passing it to the new constructor
        return State(
            new_trees, 
            new_remaining, 
            new_sequence, 
            new_hybrids, 
            initial_leaf_count=self.initial_leaf_count  
        )
    
    def rollout(self):
        from copy import deepcopy
        trees_copy = [deepcopy(t) for t in self.trees]
        remaining_copy = set(self.remaining)
        seq, hyb = full_cherry_picking_zero_prints(trees_copy, remaining_copy)
    
        if seq is None or hyb is None:
            return -1e9  # invalid rollout 
    
        return -hyb

class Node:
    def __init__(self, state, parent=None):
        self.state = state            # A State object
        self.parent = parent          # Parent Node
        self.children = {}            # action → child Node
        self.untried_actions = list(state.get_actions())  # actions not explored yet

        self.visits = 0               # N(s)
        self.value = 0.0              # Q(s), cumulative reward

    def is_fully_expanded(self):
    # A node can only be "fully expanded" if:
    # - there are no untried actions left, AND
    # - it already has at least one child
        return (len(self.untried_actions) == 0) and (len(self.children) > 0)

    def best_child(self, c_param=1.41):
        #Select child with maximum UCB1 value.
        if not self.children:
            raise RuntimeError("best_child() called but this node has no children.")

        choices = []
        for action, child in self.children.items():
            exploitation = child.value / child.visits
            exploration = math.sqrt(2 * math.log(self.visits) / child.visits)
            ucb1 = exploitation + c_param * exploration
            choices.append((ucb1, action, child))

        _, action, best = max(choices, key=lambda x: x[0])
        return best

    def expand(self):
        #Try to expand using remaining untried actions.
 
        while self.untried_actions:
            action = self.untried_actions.pop()
            new_state = self.state.apply_action(action)
    
            if new_state is None:
                continue  # invalid action → try another
    
            child = Node(new_state, parent=self)
            self.children[action] = child
            return child

        # No valid actions left
        return None


    def backpropagate(self, reward):
        #Propagate reward up the tree.
        self.visits += 1
        self.value += reward #we add the neural network's prediction

        if self.parent is not None:
            self.parent.backpropagate(reward)

    def is_terminal_node(self):
        return self.state.is_terminal()

###################################################################
#################   PARAMETER DEFINITION   ########################
###################################################################

MAX_BRANCH = 15 
ia_brain = joblib.load("value_nn_dataset_60_redes.joblib")
extractor = CPSFeatureExtractor()
trees = load_newick_trees_as_list("trees.txt")
remaining = get_leaves_from_trees(trees)
initial_count = len(remaining)
initial_state = State(trees, remaining, initial_leaf_count=initial_count)


###################################################################
############ CORE OF MCTS, WITH AND WITHOUT NN ####################
###################################################################     
    
def mcts(root_state, iterations=1000, c_param=1.41):
    root = Node(root_state)
    best_terminal_state = None
    best_terminal_score = -float("inf")

    for _ in range(iterations):
        node = root
        # 1. SELECTION
        while node.is_fully_expanded() and (not node.is_terminal_node()) and node.children:
            node = node.best_child(c_param)

        # 2. EXPANSION
        if not node.is_terminal_node():
            child = node.expand()
            if child is None:
                # If there's a deadlock, we don't kill the branch, we just heavily penalize it
                node.backpropagate(-100) 
                continue
            node = child

            # 3. SIMULATION (The key to success)
            # We use the Greedy version to ensure a "high-quality" simulation
            seq_rest, hyb_rest = full_cherry_picking_zero_prints(
            [copy.deepcopy(t) for t in node.state.trees],
            set(node.state.remaining)
        )
        
        if seq_rest is None or hyb_rest is None:
            # # OPTIMIZATION: We penalize based on the "distance to success"
            leaves_left = len(node.state.remaining)
            reward = -50 - (leaves_left * 10) 
        else:
            full_seq = node.state.sequence + seq_rest
            full_hyb = node.state.hybrids + hyb_rest
            reward = -full_hyb
            
            if reward > best_terminal_score:
                best_terminal_score = reward
                best_terminal_state = (full_seq, full_hyb)
        
        # 4. BACKPROPAGATION
        node.backpropagate(reward)

    return best_terminal_state if best_terminal_state else (None, None)
    
def mcts_with_nn(initial_state, value_model, feature_extractor, 
                 simulations_per_step=250, c_param=0.5):
    """
    Executes the AI-guided MCTS algorithm sequentially.
    At each step, it performs 'N' simulations to choose the best leaf to prune,
    advances to that state, and repeats until the forest is resolved.
    """
    current_state = initial_state
    
    print(f"\n--- STARTING MCTS+NN (Sequential Execution) ---")
    print(f"Initial leaves to prune: {len(current_state.remaining)}")

    while len(current_state.remaining) > 1:
        # Create a new root for the subtree of this step
        root = Node(current_state)

        # 1. SEARCH PHASE (The actual MCTS)
        for _ in range(simulations_per_step):
            node = root
            
            # Selection
            while node.is_fully_expanded() and not node.is_terminal_node() and node.children:
                node = node.best_child(c_param)

            # Expansion
            if not node.is_terminal_node():
                if len(node.state.remaining) <= 2:
                    child = node.expand()
                    if child is not None:
                        node = child
                else:
                    child = node.expand()
                    if child is None:
                        node.backpropagate(-300) 
                        continue
                    node = child

            # Evaluation
            if node.is_terminal_node():
                eval_reward = -float(node.state.hybrids)
            else:
                eval_reward = evaluate_state_with_value_nn(node.state, value_model, feature_extractor)

            # Backpropagate
            node.backpropagate(eval_reward)

        # 2. ACTION PHASE (Choose the best actual move)
        if not root.children:
            print(" DEADLOCK REACHED: The MCTS is cornered and cannot find a way out.")
            break

        # Choose the most visited child (the most robust path according to MCTS)
        best_child = max(root.children.values(), key=lambda c: c.visits)
        pruned_leaf = best_child.state.sequence[-1]
        
        # 3. ADVANCE TO THE NEXT STATE
        current_state = best_child.state
        print(f"Step {len(current_state.sequence)}: Pruned leaf '{pruned_leaf}'.  {len(current_state.remaining)} leaves remaining...")

    # Return the final sequence and the number of hybrids
    return current_state.sequence, current_state.hybrids

def mcts_with_nn_with_backtracking_without_panicreset(initial_state, value_model, feature_extractor, 
                 simulations_per_step=1000, c_param=1.41):
    """
    Executes the AI-guided MCTS algorithm with a BACKTRACKING system.
    If a deadlock is detected, it backtracks to the previous state and bans the action
    that led to that dead end.
    """
    print(f"\n--- STARTING MCTS+NN (Execution with Backtracking) ---")
    print(f"Initial leaves to prune: {len(initial_state.remaining)}")

    # NAVIGATION STACK: We store tuples of (Current_State, Banned_Actions)
    path_stack = [(initial_state, set())]

    while path_stack:
        current_state, banned_actions = path_stack[-1]

        # 1. SUCCESS CONDITION: 1 or 0 leaves remaining
        if len(current_state.remaining) <= 1:
            print("\n SOLUTION SUCCESSFULLY FOUND!")
            return current_state.sequence, current_state.hybrids

        # 2. Create the root to explore from here
        root = Node(current_state)

        # BACKTRACKING MAGIC: Filter out actions that we already know are lethal
        root.untried_actions = [a for a in root.untried_actions if a not in banned_actions]

        # If we already tried all branches and they all die (or there were no actions from the start)
        if len(root.untried_actions) == 0 and len(root.children) == 0:
            print(f" Absolute deadlock at step {len(current_state.sequence)}. Backtracking!")
            path_stack.pop() # Remove this state from history
            
            if path_stack:
                # Tell the parent state: "The action you used to bring me here is poison"
                bad_action = current_state.sequence[-1]
                path_stack[-1][1].add(bad_action)
            continue # Restart the loop from the parent state

        # 3. SEARCH PHASE (The MCTS itself)
        for _ in range(simulations_per_step):
            node = root
            
            # Selection
            while node.is_fully_expanded() and not node.is_terminal_node() and node.children:
                node = node.best_child(c_param)

            # Expansion
            if not node.is_terminal_node():
                child = node.expand()
                if child is None:
                    node.backpropagate(-500) # Strong penalty for entering dead branches
                    continue
                node = child

            # Evaluation
            if node.is_terminal_node():
                eval_reward = -float(node.state.hybrids)
            else:
                eval_reward = evaluate_state_with_value_nn(node.state, value_model, feature_extractor)

            # Backpropagate
            node.backpropagate(eval_reward)

        # 4. ACTION PHASE (Choose the best actual move)
        # For safety, we make sure not to choose something banned
        valid_children = {a: c for a, c in root.children.items() if a not in banned_actions}

        if not valid_children:
            print(f" After simulating, all paths are deadlocks. Backtracking!")
            path_stack.pop()
            if path_stack:
                bad_action = current_state.sequence[-1]
                path_stack[-1][1].add(bad_action)
            continue

        # Choose the most visited child (the most robust path)
        best_action = max(valid_children.keys(), key=lambda a: valid_children[a].visits)
        best_child = valid_children[best_action]
        
        # 5. ADVANCE TO THE NEXT STATE
        print(f"Step {len(current_state.sequence) + 1}: Pruned leaf '{best_action}'. {len(best_child.state.remaining)} leaves remaining...")
        
        # Add the new state to the stack, with a clean list of banned actions
        path_stack.append((best_child.state, set()))

    # If the loop ends because the stack completely emptied, the forest is unsolvable
    print("CRITICAL ERROR: The entire tree was explored and the algorithm could not find a solution.")
    return None, None

def mcts_time_budget(root_state, time_budget, c_param=1.41):
    root = Node(root_state)
    best_terminal_state = None
    best_terminal_score = -float("inf")
    
    start_time = time.time()
    n_iter = 0

    while time.time() - start_time < time_budget:
        node = root
        # 1. SELECTION
        while node.is_fully_expanded() and (not node.is_terminal_node()) and node.children:
            node = node.best_child(c_param)

        # 2. EXPANSION
        if not node.is_terminal_node():
            child = node.expand()
            if child is None:
                # If there is a deadlock, we don't kill the branch, we just heavily penalize it
                node.backpropagate(-500) 
                continue
            node = child

        # 3. SIMULATION (The key to success)
        # We use the Greedy version to ensure a "high-quality" simulation
        seq_rest, hyb_rest = full_cherry_picking_zero_prints(
            [copy.deepcopy(t) for t in node.state.trees],
            set(node.state.remaining)
        )
        
        if seq_rest is None or hyb_rest is None:
            # We penalize by "distance to success"
            # If it missed 10 leaves to prune, it's worse than if it missed 1.
            leaves_left = len(node.state.remaining)
            reward = -200 - (leaves_left * 10) 
        else:
            full_seq = node.state.sequence + seq_rest
            full_hyb = node.state.hybrids + hyb_rest
            reward = -full_hyb
            
            if reward > best_terminal_score:
                best_terminal_score = reward
                best_terminal_state = (full_seq, full_hyb)
        
        # 4. BACKPROPAGATION
        node.backpropagate(reward)
        n_iter += 1

    # Unpack the state to return (sequence, hybridizations, iterations)
    if best_terminal_state:
        return best_terminal_state[0], best_terminal_state[1], n_iter
    else:
        return None, None, n_iter

def mcts_with_nn_with_backtracking(initial_state, value_model, feature_extractor, 
                 simulations_per_step=250, c_param=1.41, max_patience=3):
    """
    Executes the AI-guided MCTS algorithm with SMART BACKTRACKING and PANIC RESET.
    If it gets stuck multiple times (Thrashing), it resets to Step 0 and bans the root branch.
    """
    print(f"\n--- STARTING MCTS+NN (Execution with Panic Reset) ---")
    print(f"Initial leaves to prune: {len(initial_state.remaining)}")

    path_stack = [(initial_state, set())]
    consecutive_backtracks = 0  # Our patience meter

    while path_stack:
        current_state, banned_actions = path_stack[-1]

        # 1. SUCCESS CONDITION: 1 or 0 leaves remaining
        if len(current_state.remaining) <= 1:
            print("\n SOLUTION SUCCESSFULLY FOUND!")
            return current_state.sequence, current_state.hybrids

        # 2. Prepare the root node for this step
        root = Node(current_state)
        root.untried_actions = [a for a in root.untried_actions if a not in banned_actions]

        # --- DEADLOCK MANAGEMENT AND PANIC RESET ---
        if len(root.untried_actions) == 0 and len(root.children) == 0:
            consecutive_backtracks += 1
            
            # Check if we have run out of patience and are not already at step 0
            if consecutive_backtracks >= max_patience and len(current_state.sequence) > 0:
                bad_root_action = current_state.sequence[0]
                print(f"\n PANIC RESET! Thrashing detected. Resetting everything and banning '{bad_root_action}' from the root.")
                
                # Return to the initial tuple (Step 0)
                root_tuple = path_stack[0]
                root_tuple[1].add(bad_root_action) # Ban the toxic root
                
                path_stack = [root_tuple] # Clear all progress
                consecutive_backtracks = 0 # Reset patience
                continue

            # If we still have patience, do a Normal (Chronological) Backtrack
            bad_action = current_state.sequence[-1] if current_state.sequence else None
            print(f" Deadlock at step {len(current_state.sequence)}. Backtracking to the previous step and banning '{bad_action}'.")
            
            path_stack.pop()
            if path_stack and bad_action:
                path_stack[-1][1].add(bad_action)
            continue

        # 3. SEARCH PHASE (The MCTS itself)
        for _ in range(simulations_per_step):
            node = root
            
            # Selection
            while node.is_fully_expanded() and not node.is_terminal_node() and node.children:
                node = node.best_child(c_param)

            # Expansion
            if not node.is_terminal_node():
                child = node.expand()
                if child is None:
                    node.backpropagate(-200) # Penalty for instant death
                    continue
                node = child

            # Evaluation
            if node.is_terminal_node():
                eval_reward = -float(node.state.hybrids)
            else:
                eval_reward = evaluate_state_with_value_nn(node.state, value_model, feature_extractor)

            # Backpropagate
            node.backpropagate(eval_reward)

        # 4. ACTION PHASE (Choose the best actual move)
        valid_children = {a: c for a, c in root.children.items() if a not in banned_actions}

        if not valid_children:
            consecutive_backtracks += 1
            
            # Same Panic Reset logic if it fails after simulating
            if consecutive_backtracks >= max_patience and len(current_state.sequence) > 0:
                bad_root_action = current_state.sequence[0]
                print(f"\n PANIC RESET! All simulated paths are toxic. Banning '{bad_root_action}'.")
                root_tuple = path_stack[0]
                root_tuple[1].add(bad_root_action)
                path_stack = [root_tuple]
                consecutive_backtracks = 0
                continue

            bad_action = current_state.sequence[-1] if current_state.sequence else None
            print(f" No safe options at step {len(current_state.sequence)}. Backtracking...")
            path_stack.pop()
            if path_stack and bad_action:
                path_stack[-1][1].add(bad_action)
            continue

        # If we find a good path, we reset patience
        consecutive_backtracks = 0
        best_action = max(valid_children.keys(), key=lambda a: valid_children[a].visits)
        best_child = valid_children[best_action]
        
        print(f"Step {len(current_state.sequence) + 1}: Pruned leaf '{best_action}'. {len(best_child.state.remaining)} leaves remaining...")
        path_stack.append((best_child.state, set()))

    print(" CRITICAL ERROR: The tree was explored and the algorithm couldn't find a solution.")
    return None, None
###################################################################
############### GREEDY RANDOM + PURE RANDOM #######################
################################################################### 

def find_best_cherry_picking(trees, remaining_leaves, time_budget):

    start_time = time.time()

    best_hybrids = float("inf")
    best_sequence = None
    n_iter = 0

    while time.time() - start_time < time_budget:
        trees_copy = [copy.deepcopy(t) for t in trees]
        remaining_copy = set(remaining_leaves)

        seq, hyb = full_cherry_picking_zero_prints(trees_copy, remaining_copy)

        n_iter += 1

        # Skip failed attempts
        if seq is None:
            continue

        # Keep best solution
        if hyb < best_hybrids:
            best_hybrids = hyb
            best_sequence = seq

    print("\n================ BEST RESULT ================")
    print("Minimum hybrids:", best_hybrids)
    print("Sequence:", best_sequence)
    print("Random iterations:", n_iter)
    print("=============================================\n")

    return best_sequence, best_hybrids, n_iter

def find_best_PURE_random(trees, remaining_leaves, time_budget):
    """
    Executes purely random simulations for a time limit (time_budget)
    and keeps the sequence that resulted in the minimum number of hybridizations.
    """
    start_time = time.time()
    best_seq = None
    best_hybrids = float('inf')
    n_iter = 0

    while time.time() - start_time < time_budget:
        # IMPORTANT: We copy the trees and leaves before sending them to the slaughterhouse
        # This way the function above doesn't destroy the original data for the next round
        trees_copy = [copy.deepcopy(t) for t in trees]
        rem_copy = set(remaining_leaves)
        
        # Pass the fresh copies to the random simulation
        seq, hyb = pure_random_cherry_picking(trees_copy, rem_copy)
        
        n_iter += 1

        # If the path didn't end in a deadlock (None) and is better than the previous one, save it
        if hyb is not None and hyb < best_hybrids:
            best_hybrids = hyb
            best_seq = seq

    return best_seq, best_hybrids, n_iter
###################################################################
######### TRAINING OF THE NEURAL NETWORK###########################
###################################################################

def estimate_value_target_from_expert(state):
    trees_copy = [copy.deepcopy(t) for t in state.trees]
    rem_copy = set(state.remaining)
    
    _, hyb_remaining = full_cherry_picking_zero_prints(trees_copy, rem_copy)
    
    if hyb_remaining is None:
        return None

    total_hybs_est = float(state.hybrids) + float(hyb_remaining)
    
    # Divide by 10.0 to maintain smooth linearity
    return -total_hybs_est / 10.0

def sample_states_from_instance(initial_state, n_trajectories=50, max_steps=50, rng_seed=0):
    """ Collects states along random paths to diversify the dataset """
    rng = np.random.default_rng(rng_seed)
    sampled_states = []

    for _ in range(n_trajectories):
        s = initial_state
        steps = 0
        while (s is not None) and (not s.is_terminal()) and (steps < max_steps):
            sampled_states.append(s)
            actions = s.get_actions()
            if not actions: break

            a = rng.choice(actions)
            s_next = s.apply_action(a)
            
            # Retry if we fall into a non-temporal state (None)
            tries = 0
            while (s_next is None) and (tries < 5) and actions:
                a = rng.choice(actions)
                s_next = s.apply_action(a)
                tries += 1

            if s_next is None: break
            s = s_next
            steps += 1
    return sampled_states

def build_value_dataset(initial_states, n_trajectories=50, max_steps=50, rng_seed=0):
    fx = CPSFeatureExtractor()
    X_list, y_list = [], []

    for idx, init_state in enumerate(initial_states):
        states = sample_states_from_instance(init_state, n_trajectories, max_steps, rng_seed + idx)
        for s in states:
            if len(s.remaining) <= 1: continue
            
            # Pass s.initial_leaf_count so the percentage works
            out = fx.extract_all(
                s.trees, 
                s.remaining, 
                initial_leaf_count=s.initial_leaf_count, 
                hybrids_so_far=s.hybrids
            )
            X_list.append(out["state_vector"])

            y = estimate_value_target_from_expert(s)
            if y is not None:
                y_list.append(float(y))
            else:
                X_list.pop()

    return np.vstack(X_list), np.array(y_list)

def train_value_nn(initial_states, model_path="value_nn_expert.joblib", 
                   n_trajectories=100, max_iter=300, rng_seed=0):
    
    X, y = build_value_dataset(initial_states, n_trajectories, rng_seed=rng_seed)

    if X.size == 0:
        raise RuntimeError("Could not collect data. Check the Greedy function.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            # NEW: Reduce neurons to avoid overfitting
            hidden_layer_sizes=(128, 64), 
            activation="relu",
            solver="adam",
            # NEW: Increase alpha to 0.01 to penalize memorization
            alpha=0.01,              
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,     
            validation_fraction=0.1, 
            random_state=rng_seed
        ))
    ])

    print(f"Training with {X.shape[0]} states...")
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    print(f"Model saved in {model_path}. R^2: {model.score(X, y):.4f}")
    return model

"""
# THIS IS IF WE WANT TO RETRAIN THE NEURAL NETWORK
list_of_initial_states = []
# HERE IS THE KEY: Indicate the name of your subfolder
carpeta_redes = "./dataset_arboles" 

# The loop enters the subfolder and reads all the .txt files inside
for archivo in os.listdir(carpeta_redes):
    if archivo.endswith(".txt"):
        # This safely joins the folder with the file name
        ruta = os.path.join(carpeta_redes, archivo) 
        
        trees = load_newick_trees_as_list(ruta)
        remaining = get_leaves_from_trees(trees)
        
        init_count = len(remaining) 
        state = State(trees, remaining, initial_leaf_count=init_count)
        lista_de_estados_iniciales.append(state)

print("\n--- TRAINING THE NEURAL NETWORK WITH THE EXPERT ---")
# We use 100 trajectories so it sees enough examples of the 50-leaf tree
ia_brain = train_value_nn(
    initial_states=list_of_initial_states, 
    n_trajectories=15,
    model_path="value_nn_dataset_60_redes.joblib"  
)
"""
###################################################################
################# ALGORITHMS INITIALIZATION #######################
###################################################################

"""
Along the various callings for each type of algorithms, the parameters 
can be changed

"""

###################################################################
################### MCTS + NN + BACKTRACKING ######################
###################################################################

"""
print("MCTS with NN (Optimized AlphaZero Style): Results and Timing")

evaluation_cache.clear()  # necessary to clear memory so RAM doesn't fill up and mix up with old data
start_time = time.time()


best_seq, best_hyb = mcts_with_nn_with_backtracking(
    initial_state=initial_state,   # The initial 20-leaf state
    value_model=ia_brain,             # expert neural network
    feature_extractor=extractor,      # The feature extractor
    simulations_per_step=500,         # test several values
    c_param=1.41,                       # Low value: we trust the AI more than random exploration
    max_patience=3
)

elapsed_mcts_nn_backtracking = time.time() - start_time

if best_seq is None:
    print("MCTS with NN did not build a valid solution.")
else:
    print("Best CPS sequence found:", best_seq)
    print("Hybridizations:", best_hyb)

print("MCTS with NN with backtracking time (in seconds):", elapsed_mcts_nn_backtracking)

"""

###################################################################
####################### MCTS + NN #################################
###################################################################


print("MCTS with NN (Optimized AlphaZero Style): Results and Timing")

evaluation_cache.clear()  # necessary to clear memory so RAM doesn't fill up and mix up with old data
start_time = time.time()

best_seq, best_hyb = mcts_with_nn(
    initial_state=initial_state,   # The initial 20-leaf state
    value_model=ia_brain,             # expert neural network
    feature_extractor=extractor,      # The feature extractor
    simulations_per_step=500,         # We do 300 simulations just to decide 1 step
    c_param=1.41                       # Low value: we trust the AI more than random exploration
)

elapsed_mcts_nn = time.time() - start_time

if best_seq is None:
    print("MCTS with NN did not build a valid solution.")
else:
    print("Best CPS sequence found:", best_seq)
    print("Hybridizations:", best_hyb)

print("MCTS with NN time (in seconds):", elapsed_mcts_nn)

###################################################################
####################### SIMPLE MCTS ###############################
###################################################################

trees = load_newick_trees_as_list("trees.txt")
remaining = get_leaves_from_trees(trees)
initial_state = State(trees, remaining)

print("MCTS ALGORITHM: Results and Timing")
start_time = time.time()
best_seq, best_hyb = mcts(initial_state, iterations=1000)
elapsed_mcts = time.time() - start_time
if best_seq is None:
    print("MCTS did not find any valid full CPS within the iteration budget.")
else:
    print("Optimal CPS sequence found:", best_seq)
    print("Hybridizations:", best_hyb)
print("MCTS time (in seconds):", elapsed_mcts)

"""
###################################################################
##################### MCTS TIME BUDGET ############################
###################################################################

print("Classic MCTS (Time Budget): Results and Timing")
start_time = time.time()

best_seq_mcts, best_hyb_mcts, iters_mcts = mcts_time_budget(
    root_state=initial_state, 
    time_budget=elapsed_mcts_nn_backtracking
)

print(f"Optimal CPS sequence found: {best_seq_mcts}")
print(f"Hybridizations: {best_hyb_mcts}")
print(f"Completed iterations: {iters_mcts}")
print(f"MCTS time (in seconds): {elapsed_mcts_nn_backtracking}")
"""
###################################################################
################### TRIVIAL-RAND HEURISTIC ########################
###################################################################
elapsed_random = min(elapsed_mcts, elapsed_mcts_nn)

print("Random Search Algorithm: Results and Timing")

seq_random, hyb_random, random_iters = find_best_cherry_picking(
    trees, remaining, time_budget = elapsed_random)

elapsed_random = min(elapsed_mcts, elapsed_mcts_nn)

print("Random Algorithm time (in seconds):", elapsed_random)
print("Random iterations:", random_iters)

###################################################################
##################### FULLY RANDOM LOOP  ##########################
###################################################################


print("\n--- PURE Random Search Algorithm (No Bias) ---")


seq_pure, hyb_pure, iters_pure = find_best_PURE_random(
    trees, remaining, time_budget=elapsed_random)

print(f"Best sequence: {seq_pure}")
print(f"Minimum hybridizations: {hyb_pure}")
print(f"Iterations performed: {iters_pure}")
print(f"Total time: {elapsed_mcts} s")