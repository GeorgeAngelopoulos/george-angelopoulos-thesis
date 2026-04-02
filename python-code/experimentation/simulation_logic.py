import random
import math
import csv, os
from enum import Enum, auto
from collections import defaultdict

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONSTANTS & HYPERPARAMETERS
# =============================================================================
# We define these globally so they match Unity exactly.
C_MOVE = 3.0
BETA = 1.0
STARTING_FOOD = 100.0
LEADER_SHARE = 0.2
PLACEHOLDER = 0

PREDATOR_PROFILES = {
    "Baseline": {"agent_type": "Baseline", "M_i": 1.5, "Dbase_i": 0.1, "C_hunt_i": 5.0, "F_i": STARTING_FOOD},
    "Specialist": {"agent_type": "Specialist", "M_i": 3.0, "Dbase_i": 0.266, "C_hunt_i": 15.0, "F_i": STARTING_FOOD},
    "Newbie": {"agent_type": "Newbie", "M_i": 0.17, "Dbase_i": 0.1, "C_hunt_i": 3.0, "F_i": STARTING_FOOD},
    "Freeloader": {"agent_type": "Freeloader", "M_i": 0.8, "Dbase_i": 0.4, "C_hunt_i": 3.33, "F_i": STARTING_FOOD}
}

PREY_PROFILES = {
    "Hare": {"prey_type": 0, "P_solo": 0.7, "R_a": 20.0, "b_a": BETA},
    "Stag": {"prey_type": 1, "P_solo": 0.4, "R_a": 55.0, "b_a": 1.5*BETA},
    "Mammoth": {"prey_type": 2, "P_solo": 0.1, "R_a": 160.0, "b_a": 2.5*BETA}
}

# Axial directions for neighbors in a hex grid
HEX_DIRECTIONS = [
    (1, 0), (1, -1), (0, -1), 
    (-1, 0), (-1, 1), (0, 1)
]

def axial_distance(node_a, node_b):
    """ Calculates the number of hex steps between two (q, r) tuples. """
    return (abs(node_a[0] - node_b[0])
            + abs(node_a[0] + node_a[1] - node_b[0] - node_b[1])
            + abs(node_a[1] - node_b[1])) // 2
# =============================================================================
# 2. HEXGRID CLASSES
# =============================================================================
class HexNode:
    def __init__(self, q, r):
        self.q = q
        self.r = r
        self.coords = (q, r)
        # Tracking what is currently occupying this specific tile
        self.resident_predators = []       # List of predator IDs
        self.resident_prey = []            # List of prey IDS  

class HexGrid:
    def __init__(self):
        """ The Registry: {(q, r): HexNode} """
        self.nodes = {}
    
    def add_node(self, q, r):
        """Adds a walkable tile to the simulation."""
        node = HexNode(q, r)
        self.nodes[(q, r)] = node
    
    def is_walkable(self, q, r):
        """Checks if a coordinate exists in the map."""
        return (q, r) in self.nodes
    
    def get_neighbor_hexes(self, q, r):
        """Returns only the neighbors that actually exist in the map."""
        neighbors = []
        for dq, dr in HEX_DIRECTIONS:
            target = (q + dq, r + dr)
            if self.is_walkable(*target):
                neighbors.append(target)
        return neighbors

    def get_random_node(self):
        """Returns a random valid coordinate from the registry."""
        return random.choice(list(self.nodes.keys()))       

# =============================================================================
# 3. AGENT CLASSES
# =============================================================================
class Predator:
    """ Defines the Predator Agent Constants and Variables. """
    def __init__(self, agent_id, agent_type, q, r, M_i, Dbase_i, C_hunt_i, F_i):
        # --- Basic Variables ---
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.q = q
        self.r = r
        self.M_i = M_i
        self.Dbase_i = Dbase_i
        self.C_hunt_i = C_hunt_i    
        self.F_i = F_i
        # --- Useful Variables ---
        self.is_alive = True
        self.has_acted = False
        self.decision = {
            'action_type': None,
            'utility': 0.0, 
            'target_q': 0,
            'target_r': 0,
            'partners': None,       
            'final_share': 0.0,                    
            'target_prey_id': 0,                             
            'hunt_success': 0.0
        }
        # --- Tracking Stats ---
        self.solo_hunts_performed = 0
        self.coalition_hunts_performed = 0
        self.utility_gained_solo = 0.0
        self.utility_gained_initiator = 0.0
        self.utility_gained_partner = 0.0

class Prey:
    """ Defines the Prey Agent Constants and Variables. """
    def __init__(self, prey_id, q, r, prey_type, P_solo, R_a, b_a):
        # --- Basic Variables ---
        self.prey_id = prey_id
        self.q = q
        self.r = r
        self.prey_type = prey_type
        self.P_solo = P_solo
        self.R_a = R_a
        self.b_a = b_a    
        # --- Useful Variables ---
        self.is_alive = True
        self.targeted = False

class Action(Enum):
    """ Defines the discrete, mutually exclusive choices an agent can make. """
    REST = auto()                     # Default action: Metabolically cheap, small guaranteed recovery.
    MOVE = auto()                     # Search action: Change hex to find better opportunities.
    SOLO_HUNT = auto()                # High Risk/Gain action: Attack the prey alone.
    COALITION_HUNT = auto()           # Negotiation action: Attempt to coordinate with neighbors.
    ...

# =============================================================================
# 4. SIMULATION LOGGER CLASS
# =============================================================================
class SimulationLogger:
    def __init__(self, base_dir="results"):
        """
        Initializes the logger to organize experimental data.
        
        Args:
            base_dir (str): The directory where tactic-specific CSVs will be saved.
        """
        self.base_dir = base_dir
        
        # Create the results directory if it doesn't exist
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        # Headers strictly following your experiment tracking variables
        self.headers = [
            "episode_id",
            "predator_id",
            "solo_hunts_performed",
            "coalition_hunts_performed",
            "utility_gained_solo",
            "utility_gained_initiator",
            "utility_gained_partner"
        ]

        # Structure: global_stats[tactic][predator_id] -> dict of totals
        self.global_stats = defaultdict(lambda: defaultdict(lambda: {
            "solo_hunts_performed": 0,
            "coalition_hunts_performed": 0,
            "utility_gained_solo": 0.0,
            "utility_gained_initiator": 0.0,
            "utility_gained_partner": 0.0
        }))        

    def _get_filepath(self, tactic):
        """Helper to create a filename based on the current negotiation tactic."""
        return os.path.join(self.base_dir, f"results_{tactic}.csv")

    def compute_episode_statistics(self, predators):
        """
        Computes per-predator-type averages for a single episode.

        Returns:
            dict: {
                predator_type: {
                    "avg_solo_hunts": float,
                    "avg_coalition_hunts": float,
                    "avg_utility_solo": float,
                    "avg_utility_initiator": float,
                    "avg_utility_partner": float
                }
            }
        """
        # --- Group predators by type ---
        type_groups = defaultdict(list)
        for p in predators:
            type_groups[p.agent_type].append(p)

        episode_stats = {}

        # --- Compute averages per type ---
        for predator_type, group in type_groups.items():

            n = len(group)
            if n == 0:
                continue

            total_solo = sum(p.solo_hunts_performed for p in group)
            total_coal = sum(p.coalition_hunts_performed for p in group)
            total_u_solo = sum(p.utility_gained_solo for p in group)
            total_u_init = sum(p.utility_gained_initiator for p in group)
            total_u_part = sum(p.utility_gained_partner for p in group)

            episode_stats[predator_type] = {
                "avg_solo_hunts": total_solo / n,
                "avg_coalition_hunts": total_coal / n,
                "avg_utility_solo": total_u_solo / n,
                "avg_utility_initiator": total_u_init / n,
                "avg_utility_partner": total_u_part / n
            }

        return episode_stats

    def batch_update(self, tactic, episode_stats, episode_id, batch_size=1000):
        """
        Batching layer that aggregates episode-level statistics into fixed-size batches.

        Args:
            tactic (str): current tactic
            episode_stats (dict): output of compute_episode_statistics()
            episode_id (int): current episode index
            batch_size (int): number of episodes per batch

        Returns:
            dict or None:
                - dict → batch completed (ready to log)
                - None → batch still accumulating
        """

        # -----------------------------
        # 1. Initialize buffer if needed
        # -----------------------------
        if not hasattr(self, "batch_buffer"):
            self.batch_buffer = defaultdict(lambda: defaultdict(lambda: {
                "count": 0,
                "sum_solo_hunts": 0.0,
                "sum_coalition_hunts": 0.0,
                "sum_utility_solo": 0.0,
                "sum_utility_initiator": 0.0,
                "sum_utility_partner": 0.0
            }))

        # -----------------------------
        # 2. Accumulate episode stats
        # -----------------------------
        for predator_type, stats in episode_stats.items():

            buf = self.batch_buffer[tactic][predator_type]

            buf["count"] += 1

            buf["sum_solo_hunts"] += stats["avg_solo_hunts"]
            buf["sum_coalition_hunts"] += stats["avg_coalition_hunts"]
            buf["sum_utility_solo"] += stats["avg_utility_solo"]
            buf["sum_utility_initiator"] += stats["avg_utility_initiator"]
            buf["sum_utility_partner"] += stats["avg_utility_partner"]

        # -----------------------------
        # 3. Check if batch is complete
        # -----------------------------
        if (episode_id + 1) % batch_size != 0:
            return None

        # -----------------------------
        # 4. Finalize batch results
        # -----------------------------
        batch_result = {}

        for predator_type, buf in self.batch_buffer[tactic].items():

            n = buf["count"]
            if n == 0:
                continue

            batch_result[predator_type] = {
                "avg_solo_hunts": buf["sum_solo_hunts"] / n,
                "avg_coalition_hunts": buf["sum_coalition_hunts"] / n,
                "avg_utility_solo": buf["sum_utility_solo"] / n,
                "avg_utility_initiator": buf["sum_utility_initiator"] / n,
                "avg_utility_partner": buf["sum_utility_partner"] / n,
                "batch_size": n
            }

        # -----------------------------
        # 5. Reset buffer for next batch
        # -----------------------------
        self.batch_buffer[tactic] = defaultdict(lambda: {
            "count": 0,
            "sum_solo_hunts": 0.0,
            "sum_coalition_hunts": 0.0,
            "sum_utility_solo": 0.0,
            "sum_utility_initiator": 0.0,
            "sum_utility_partner": 0.0
        })

        return batch_result
    
    def aggregate_batch(self, batch_list):
        """
        Aggregates a list of episode-level AVERAGE stats into batch averages.
        """

        totals = defaultdict(lambda: {
            "avg_solo_hunts": 0.0,
            "avg_coalition_hunts": 0.0,
            "avg_utility_solo": 0.0,
            "avg_utility_initiator": 0.0,
            "avg_utility_partner": 0.0,
            "count": 0
        })

        # -----------------------------
        # Sum averages across episodes
        # -----------------------------
        for episode_stats in batch_list:

            for predator_type, stats in episode_stats.items():

                totals[predator_type]["avg_solo_hunts"] += stats["avg_solo_hunts"]
                totals[predator_type]["avg_coalition_hunts"] += stats["avg_coalition_hunts"]
                totals[predator_type]["avg_utility_solo"] += stats["avg_utility_solo"]
                totals[predator_type]["avg_utility_initiator"] += stats["avg_utility_initiator"]
                totals[predator_type]["avg_utility_partner"] += stats["avg_utility_partner"]
                totals[predator_type]["count"] += 1

        # -----------------------------
        # Convert to batch averages
        # -----------------------------
        batch_results = {}

        for predator_type, stats in totals.items():

            n = stats["count"]

            if n == 0:
                continue

            batch_results[predator_type] = {
                "batch_size": n,
                "avg_solo_hunts": stats["avg_solo_hunts"] / n,
                "avg_coalition_hunts": stats["avg_coalition_hunts"] / n,
                "avg_utility_solo": stats["avg_utility_solo"] / n,
                "avg_utility_initiator": stats["avg_utility_initiator"] / n,
                "avg_utility_partner": stats["avg_utility_partner"] / n,
            }

        return batch_results
    
    def log_results(self, episode_id, tactic, episode_stats, filepath="results"):
        """
        Logs episode-level aggregated results to CSV.

        Args:
            episode_id (int): current episode index
            tactic (str): current tactic
            episode_stats (dict): output of compute_episode_statistics()
            filepath (str): output directory
        """
        # 1. Prepare file path
        os.makedirs(filepath, exist_ok=True)
        file_path = os.path.join(filepath, f"results_{tactic}.csv")

        file_exists = os.path.exists(file_path)

        # 2. Define headers
        headers = [
            "episode_id",
            "tactic",
            "predator_type",
            "avg_solo_hunts",
            "avg_coalition_hunts",
            "avg_utility_solo",
            "avg_utility_initiator",
            "avg_utility_partner"
        ]

        # 3. Write episode results
        with open(file_path, mode="a", newline="") as f:
            writer = csv.writer(f)

            # write header only once
            if not file_exists:
                writer.writerow(headers)

            for predator_type, stats in episode_stats.items():

                writer.writerow([
                    episode_id,
                    tactic,
                    predator_type,
                    round(stats["avg_solo_hunts"], 2),
                    round(stats["avg_coalition_hunts"], 2),
                    round(stats["avg_utility_solo"], 2),
                    round(stats["avg_utility_initiator"], 2),
                    round(stats["avg_utility_partner"], 2),
                ])

    def log_batch_results(self, batch_id, tactic, batch_stats, filepath="results"):
        """
        Logs batch-aggregated results to CSV.

        Args:
            batch_id (int): index of the batch
            tactic (str): current tactic
            batch_stats (dict): output of aggregate_batch()
            filepath (str): output directory
        """

        # -----------------------------
        # 1. Prepare file path
        # -----------------------------
        os.makedirs(filepath, exist_ok=True)
        file_path = os.path.join(filepath, f"results_{tactic}.csv")

        file_exists = os.path.exists(file_path)

        # -----------------------------
        # 2. Define headers
        # -----------------------------
        headers = [
            "batch_id",
            "tactic",
            "predator_type",
            "batch_size",
            "avg_solo_hunts",
            "avg_coalition_hunts",
            "avg_utility_solo",
            "avg_utility_initiator",
            "avg_utility_partner"
        ]

        # -----------------------------
        # 3. Write batch results
        # -----------------------------
        with open(file_path, mode="a", newline="") as f:
            writer = csv.writer(f)

            # write header only once
            if not file_exists:
                writer.writerow(headers)

            for predator_type, stats in batch_stats.items():

                writer.writerow([
                    batch_id,
                    tactic,
                    predator_type,
                    stats["batch_size"],
                    round(stats["avg_solo_hunts"], 2),
                    round(stats["avg_coalition_hunts"], 2),
                    round(stats["avg_utility_solo"], 2),
                    round(stats["avg_utility_initiator"], 2),
                    round(stats["avg_utility_partner"], 2),
                ])

    def plot_tactic_comparison_means(self, results_dir="results"):
        """
        Episode-based plotting.

        Each row:
            (episode_id, tactic, predator_type, metrics...)

        We compute:
            mean across episodes
            95% CI across episodes
        """

        pattern = os.path.join(results_dir, "results_*.csv")
        files = glob.glob(pattern)

        if not files:
            print("No result files found.")
            return

        all_data = {}
        predator_types = set()

        # -----------------------------
        # Load all CSVs
        # -----------------------------
        for filepath in files:
            filename = os.path.basename(filepath)
            tactic = filename.replace("results_", "").replace(".csv", "")

            df = pd.read_csv(filepath)

            all_data[tactic] = df
            predator_types.update(df["predator_type"].unique())

        tactics = sorted(all_data.keys())
        predator_types = sorted(predator_types)

        # -----------------------------
        # Metrics
        # -----------------------------
        hunt_stats = [
            "avg_solo_hunts",
            "avg_coalition_hunts"
        ]

        utility_stats = [
            "avg_utility_solo",
            "avg_utility_initiator",
            "avg_utility_partner"
        ]

        stat_names = {
            "avg_solo_hunts": "Solo Hunts Performed",
            "avg_coalition_hunts": "Coalition Hunts Performed",
            "avg_utility_solo": "Utility Gained (Solo)",
            "avg_utility_initiator": "Utility Gained (Initiator)",
            "avg_utility_partner": "Utility Gained (Partner)"
        }

        color_map = {
            "EGALITARIAN": "green",
            "ASYMMETRIC": "orange",
            "ALTRUISTIC": "red",
            "MERITOCRATIC": "blue"
        }

        bar_width = 0.5

        # -----------------------------
        # Core stats computation
        # -----------------------------
        def compute_stats(all_data, predator_type, stat):

            means = []
            ci_values = []
            colors = []

            for tactic in tactics:

                df = all_data[tactic]
                type_df = df[df["predator_type"] == predator_type]

                if type_df.empty:
                    means.append(0)
                    ci_values.append(0)
                    colors.append(color_map.get(tactic, "black"))
                    continue

                # -----------------------------
                # EPISODE VALUES (direct!)
                # -----------------------------
                values = type_df[stat]

                mean_val = values.mean()
                std_val = values.std()
                n = len(values)

                ci = 1.96 * (std_val / np.sqrt(n)) if n > 0 else 0

                means.append(mean_val)
                ci_values.append(ci)
                colors.append(color_map.get(tactic, "black"))

            return means, ci_values, colors

        # -----------------------------
        # Label helper
        # -----------------------------
        def add_value_labels(ax, bars, means, cis):
            for bar, mean, ci in zip(bars, means, cis):
                height = bar.get_height()

                label = f"{mean:.2f}\n±{ci:.2f}"

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + ci * 1.1,
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        # =============================
        # PLOTTING LOOP
        # =============================
        for predator_type in predator_types:

            # -------------------------
            # FIGURE 1 — Hunts
            # -------------------------
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for idx, stat in enumerate(hunt_stats):

                ax = axes[idx]

                means, ci_values, colors = compute_stats(all_data, predator_type, stat)

                x = np.arange(len(tactics))

                bars = ax.bar(
                    x,
                    means,
                    width=bar_width,
                    yerr=ci_values,
                    capsize=6,
                    color=colors
                )

                add_value_labels(ax, bars, means, ci_values)

                ax.set_xticks(x)
                ax.set_xticklabels(tactics)
                ax.set_title(stat_names[stat])
                ax.set_ylabel("Mean Value (95% CI)")
                ax.grid(True, linestyle="--", alpha=0.3)

                max_height = max([m + c for m, c in zip(means, ci_values)])
                ax.set_ylim(0, max_height * 1.3)

            fig.suptitle(f"Hunting Performance — Predator Type: {predator_type}", fontsize=16)
            plt.subplots_adjust(wspace=0.35, top=0.85)
            plt.show()

            # -------------------------
            # FIGURE 2 — Utilities
            # -------------------------
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            for idx, stat in enumerate(utility_stats):

                ax = axes[idx]

                means, ci_values, colors = compute_stats(all_data, predator_type, stat)

                x = np.arange(len(tactics))

                bars = ax.bar(
                    x,
                    means,
                    width=bar_width,
                    yerr=ci_values,
                    capsize=6,
                    color=colors
                )

                add_value_labels(ax, bars, means, ci_values)

                ax.set_xticks(x)
                ax.set_xticklabels(tactics)
                ax.set_title(stat_names[stat])
                ax.set_ylabel("Mean Value (95% CI)")
                ax.grid(True, linestyle="--", alpha=0.3)

                max_height = max([m + c for m, c in zip(means, ci_values)])
                ax.set_ylim(0, max_height * 1.3)

            fig.suptitle(f"Utility Distribution — Predator Type: {predator_type}", fontsize=16)
            plt.subplots_adjust(wspace=0.35, top=0.85)
            plt.show()

# =============================================================================
# 5. CORE SIMULATION ENGINE
# =============================================================================
class SimulationEngine:
    def __init__(self, predator_count, prey_count, radius, tactic, seed=None):

        if seed is not None:                                        # This seed initializes the internal state of the random number generator.
            random.seed(seed)                                       # Every subsequent call to random.randint or random.uniform will follow
            self.seed = seed                                        # the same deterministic sequence for this seed.

        self.current_tick = 0
        self.tactic = tactic

        # Creating the HexGrid Map
        self.grid = HexGrid()
        self.radius = radius                                        # radius of hex grid
        self.setup_circular_map(radius)

        self.predators = []
        self.prey = []
        self.setup_agents(predator_count, prey_count)

# ------------------------------------------------------------------------------------------------------------------------------------
    def setup_circular_map(self, radius):
        """
        Populates the grid with a circular shape. 
        In the future, this function could instead load a 
        list of (q,r) from a JSON file exported from Unity.
        """
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                self.grid.add_node(q, r)

# ------------------------------------------------------------------------------------------------------------------------------------
    def setup_agents(self, predator_count, prey_count):
        """Deterministic setup of agents on the grid."""
        grid = self.grid
        predator_list = self.predators
        prey_list = self.prey

        # 1. Instantiate Predators
        type_cycle = list(PREDATOR_PROFILES.keys())
        
        for i in range(predator_count):
            # Seeded selection of predator type
            q, r = grid.get_random_node()
            p_type = type_cycle[i % len(type_cycle)]
            profile = PREDATOR_PROFILES[p_type]
            pr = Predator(
                agent_id = i,
                agent_type = profile["agent_type"],
                q = q,
                r = r,
                M_i = profile["M_i"],
                Dbase_i = profile["Dbase_i"],
                C_hunt_i = profile["C_hunt_i"],
                F_i = profile["F_i"]
                )
            predator_list.append(pr)

        # 2. Instantiate Prey
        type_choices = list(PREY_PROFILES.keys())

        for a in range(prey_count):
            # Seeded selection of prey type
            q, r = grid.get_random_node()
            p_type = random.choice(type_choices)
            profile = PREY_PROFILES[p_type]
            pr = Prey(
                prey_id = a + 100,
                q = q,
                r = r,
                prey_type = profile["prey_type"],
                P_solo = profile["P_solo"],
                R_a = profile["R_a"],
                b_a = profile["b_a"]
                )
            prey_list.append(pr)       

# ------------------------------------------------------------------------------------------------------------------------------------
# --- THE "BRAINS" (Decision-Making Function) ---
# ------------------------------------------------------------------------------------------------------------------------------------
    def form_best_decision(self, predator):
        """ Calculates the expected utility for REST, MOVE, SOLO_HUNT and COALITION_HUNT.
        Returns the Best Action to execute based on that utility. """
        # --- INITIAL OBSERVATION ---
        predators_present, prey_present = self.get_nearby_agents(predator, predator.q, predator.r)
        
        best_action = None
        best_utility = float("-inf")
        best_data = {}

        # --- REST --- 
        utility = self.Action_REST_utility()
        if utility > best_utility:
            best_action = Action.REST
            best_utility = utility
            best_data = {}
        
        # --- MOVE ---
        utility, q, r = self.Action_MOVE_utility(predator)
        if utility > best_utility:
            best_action = Action.MOVE
            best_utility = utility
            best_data = {"q": q, "r": r}
        
        # --- SOLO HUNT ---
        if prey_present:

            utility, target_prey = self.Action_SOLO_HUNT_utility(predator, prey_present)
            if utility > best_utility:
                best_action = Action.SOLO_HUNT
                best_utility = utility
                best_data = {"prey": target_prey}
        
        # --- COALITION HUNT ---
        if prey_present and predators_present:

            utility, target_prey, partners, share = self.Action_COALITION_HUNT_utility(predator, predators_present, prey_present)
            if utility > best_utility:
                best_action = Action.COALITION_HUNT
                best_utility = utility
                best_data = {"prey": target_prey, "partners": partners, "share": share}

        # --- BUILD DECISION ---
        predator.decision = {
            'action_type': best_action,
            'utility': best_utility,
            'target_q': predator.q,
            'target_r': predator.r,
            'partners': None,
            'final_share': 0.0,
            'target_prey_id': 0,
            'hunt_success': 0.0
        }

        if best_action == Action.MOVE:

            self.update_decision(predator, 
                target_q = best_data["q"],
                target_r = best_data["r"]
                )
            
        elif best_action == Action.SOLO_HUNT:

            prey = best_data["prey"]
            prey.targeted = True
            self.update_decision(predator,
                target_prey_id = prey.prey_id,
                final_share = 1.0,
                hunt_success = self.get_hunt_success(predator, None, prey))
        
        elif best_action == Action.COALITION_HUNT:

            prey = best_data["prey"]
            prey.targeted = True
            self.update_decision(predator,
                partners = best_data["partners"],
                final_share = best_data["share"],
                target_prey_id = prey.prey_id,
                hunt_success = self.get_hunt_success(predator, best_data["partners"], prey)
                )

        predator.has_acted = True

# ------------------------------------------------------------------------------------------------------------------------------------
# --- THE UTILITY FUNCTIONS ---
# ------------------------------------------------------------------------------------------------------------------------------------     
    def Action_REST_utility(self):
        """ Calculates the Expected Utility of resting, E[U_Rest] = 0.0 (Idle, No Gain). """
        return 0.0

    def Action_MOVE_utility(self, predator):
        """
        Assigns a forward-looking (myopic) utility to moving.
        Encourages exploration towards valuable prey using discounted rewards.
        """
        gamma = 0.9
        neighbor_hexes = self.grid.get_neighbor_hexes(predator.q, predator.r)

        # --- Precompute utility per prey type ---
        type_utilities = {}
        for profile in PREY_PROFILES.values():
            p_type = profile['prey_type']

            P_capture = COMPUTE.P_capture_value(profile["P_solo"], predator.M_i, profile["b_a"], 1)
            U_solo = COMPUTE.Exp_Utility_value(P_capture, 1.0, profile["R_a"], predator.C_hunt_i)

            type_utilities[p_type] = U_solo

        # --- Evaluate each neighboring Hex ---
        best_move_utility = float('-inf')
        best_hex = (predator.q, predator.r)

        for new_q, new_r in neighbor_hexes:

            V_best_hex = float('-inf')
            for prey in self.prey:

                U_solo = type_utilities[prey.prey_type]
                distance = axial_distance((new_q, new_r), (prey.q, prey.r))

                if distance == 0:
                    V_adj_hex = U_solo
                else:
                    gamma_d = gamma ** distance
                    # discounted future reward
                    discounted_reward = gamma_d * U_solo
                    # cumulative movement cost (geometric series)
                    discounted_cost = C_MOVE * (1 - gamma_d) / (1 - gamma)
                    V_adj_hex = discounted_reward - discounted_cost
                   
                if V_adj_hex > V_best_hex:
                    V_best_hex = V_adj_hex
                                
            # final move utility
            move_utility = -C_MOVE + gamma * V_best_hex

            if move_utility > best_move_utility:
                best_move_utility = move_utility
                best_hex = (new_q, new_r)

        return best_move_utility, best_hex[0], best_hex[1]

    def Action_SOLO_HUNT_utility(self, predator, prey_present):
        """ Calculates the best Expected Utility of Solo Hunting in the specific Hex.
            That utility comes from first finding the best Prey Target for hunting. """
        best_utility = float("-inf")
        target_prey = None

        for prey in prey_present:
            P_capture = COMPUTE.P_capture_value(prey.P_solo, predator.M_i, prey.b_a, 1)
            utility = COMPUTE.Exp_Utility_value(P_capture, 1.0, prey.R_a, predator.C_hunt_i)

            if utility > best_utility:
                best_utility = utility
                target_prey = prey
        
        return best_utility, target_prey

    def Action_COALITION_HUNT_utility(self, predator, predators_present, prey_present):
        """ Calculates best expected utility and target prey for coalition hunting. """
        best_utility = float("-inf")
        target_prey = None

        coalition = [predator] + predators_present
        coalition_size = len(coalition)
        total_E_C = COMPUTE.E_C_value(coalition)

        initiator_share = self.get_split_shares(predator, coalition)[predator]

        for prey in prey_present:
            P_capture = COMPUTE.P_capture_value(prey.P_solo, total_E_C, prey.b_a, coalition_size)
            utility = COMPUTE.Exp_Utility_value(P_capture, initiator_share, prey.R_a, predator.C_hunt_i)
            
            if utility > best_utility:
                best_utility = utility
                target_prey = prey

        return best_utility, target_prey, predators_present, initiator_share
    
# ------------------------------------------------------------------------------------------------------------------------------------
# --- THE "PHYSICS" (State Updates) ---
# ------------------------------------------------------------------------------------------------------------------------------------
    def execute_best_decision(self, predator):
        """ Take the Predator's decision dict and executes the action. """
        action = predator.decision['action_type']

        if action == Action.MOVE:
            self.execute_movement(predator)
            return

        elif action in (Action.SOLO_HUNT, Action.COALITION_HUNT):
            target_prey = self.get_prey_by_id(predator.decision['target_prey_id'])
            if action == Action.COALITION_HUNT:
                self.execute_coalition_negotiation(predator, target_prey)
            self.execute_hunt(predator, target_prey)
            self.execute_prey_movement(target_prey)
            return

    def execute_movement(self, predator):
        """ Predator moves to target Hex coords. """
        predator.q = predator.decision['target_q']
        predator.r = predator.decision['target_r']
        predator.F_i -= C_MOVE

    def execute_prey_movement(self, prey):
        """ Executes random movement for Prey. """
        """ Prey either moves after a successful hunt or periodically.  """
        neighbor_hexes = self.grid.get_neighbor_hexes(prey.q, prey.r)
        new_hex = random.choice(list(neighbor_hexes))
        prey.q = new_hex[0]
        prey.r = new_hex[1]

    def execute_hunt(self, predator, target_prey):
        """ Resolves the Predator's or Coalition's Hunt.
            Updates Food Stockpile based on the success. """
        
        decision = predator.decision
        success = random.random() < decision['hunt_success']

        partners = decision['partners'] or []
        participants = [predator, *partners]

        for p in participants:

            p_decision = p.decision
            action_type = p_decision['action_type']

            if success:
                reward = p_decision['final_share'] * target_prey.R_a
                utility = reward - p.C_hunt_i
            else:
                utility = - p.C_hunt_i
                
            p.F_i += utility

            # --- Stats Update ---
            if action_type == Action.COALITION_HUNT:
                p.coalition_hunts_performed += 1
                if p is predator:
                    p.utility_gained_initiator += utility
                else:
                    p.utility_gained_partner += utility
            
            elif action_type == Action.SOLO_HUNT:
                p.utility_gained_solo += utility
                p.solo_hunts_performed += 1

    def execute_coalition_negotiation(self, initiator, target_prey):
        """Iterative coalition negotiation until stable acceptance."""
        partners = initiator.decision['partners']
        coalition_predators = partners

        while True:

            coalition = [initiator, *coalition_predators]
            coalition_size = len(coalition)

            total_E_C = COMPUTE.E_C_value(coalition)
            coalition_success = COMPUTE.P_capture_value(target_prey.P_solo, total_E_C, target_prey.b_a, coalition_size)
            coalition_shares = self.get_split_shares(initiator, coalition)

            # --- Acceptance Step ---
            accepted_predators = [p for p in coalition_predators
                                  if self.get_coalition_checks(p, coalition_success, coalition_shares[p], target_prey)]
            # --- STOP CONDITION ---
            if len(accepted_predators) == len(coalition_predators):
                break # stable coalition

            # --- If everyone rejected -> Revert to Solo Hunt ---
            if not accepted_predators:
                self.update_decision(initiator,
                    action_type = Action.SOLO_HUNT,
                    hunt_success = COMPUTE.P_capture_value(target_prey.P_solo, initiator.M_i, target_prey.b_a, 1),
                    final_share = 1.0,
                    partners = None
                )
                return
            # --- Otherwise shrink coalition and repeat ---
            coalition_predators = accepted_predators

        # --- FINAL COALITION ---
        coalition = [initiator, *coalition_predators]
        total_E_C = COMPUTE.E_C_value(coalition)
        coalition_success = COMPUTE.P_capture_value(target_prey.P_solo, total_E_C, target_prey.b_a, len(coalition))
        coalition_shares = self.get_split_shares(initiator, coalition)

        # --- Update Initiator ---
        self.update_decision(initiator,
            hunt_success = coalition_success,
            final_share = coalition_shares[initiator],
            partners = coalition_predators
        )

        # --- Update Partners ---
        for p in coalition_predators:
            self.update_decision(p,
                action_type = Action.COALITION_HUNT,
                hunt_success = coalition_success,
                final_share = coalition_shares[p]
            )
            p.has_acted = True

# ------------------------------------------------------------------------------------------------------------------------------------
# --- THE GETTER FUNCTIONS ---
# ------------------------------------------------------------------------------------------------------------------------------------
    def get_hunt_success(self, initiator, partners, target_prey):
        """ Returns the hunt's success probability. """
        """ When No Partners, it is a Solo Hunt. """
        preds = partners or []
        E_C = COMPUTE.E_C_value([initiator]) + COMPUTE.E_C_value(preds)
        coalition_size = 1 + len(preds)
            
        return COMPUTE.P_capture_value(target_prey.P_solo, E_C, target_prey.b_a, coalition_size)

    def get_nearby_agents(self, predator, q, r):
        """ Gets all the Agents in the same Hex(q, r) as the Target. """
        """ Returns Alive Predators that haven't acted and alive Prey whose not been targeted. """
        predator_list = self.predators
        prey_list = self.prey

        predators_present = [
            pred for pred in predator_list
            if pred.q == q and pred.r == r and pred.is_alive == True and pred.has_acted == False and pred != predator
        ]

        prey_present = [
            prey for prey in prey_list
            if prey.q == q and prey.r == r and prey.is_alive == True and prey.targeted == False
        ]

        return predators_present, prey_present
    
    def get_prey_by_id(self, id):
        """ Returns Prey Agent by ID. """
        for prey in self.prey:
            if prey.prey_id == id:
                return prey
        return None
    
    def get_split_shares(self, initiator, coalition):
        """
        Calculates the predator shares for a given coalition and split tactic.
        Returns a dict of Predator object -> share.
        """
        shares = {}

        if self.tactic not in ["EGALITARIAN", "MERITOCRATIC", "ALTRUISTIC", "ASYMMETRIC"]:
            raise ValueError(f"Unknown tactic: {self.tactic}")

        for predator in coalition:

            if self.tactic == "EGALITARIAN":
                share = COMPUTE.Egalitarian_split_value(coalition)

            elif self.tactic == "MERITOCRATIC":
                share = COMPUTE.Meritocratic_split_value(predator, coalition)

            elif self.tactic == "ALTRUISTIC":
                share = COMPUTE.Altruistic_split_value(predator, coalition)

            elif self.tactic == "ASYMMETRIC":
                share = COMPUTE.Asymmetric_split_value(predator, initiator, coalition)

            shares[predator] = share

        return shares
        
    def get_coalition_checks(self, predator, coalition_success, predator_share, target_prey):
        """ Calculates the Utility Check and Barganing Check """
        if not COMPUTE.Bargaining_check(predator, predator_share):
            return False
        
        if not COMPUTE.Utility_check(predator, coalition_success, predator_share, target_prey):
            return False
        
        return True

    def update_decision(self, predator, **kwargs):
        if not hasattr(predator, "decision") or predator.decision is None:
            predator.decision = {}

        predator.decision.update(kwargs)

    def update_is_alive(self, predator):
        """ Checks if the Predator is alive. """
        predator.is_alive = predator.F_i > 0

# ------------------------------------------------------------------------------------------------------------------------------------       
    def reset_cycle(self):
        """ Resets the useful variables for every step cycle. """
        for predator in self.predators:
            predator.has_acted = False
            self.update_decision(predator, 
                action_type=None, 
                utility=0.0, 
                target_q=0, 
                target_r=0, 
                partners=None, 
                final_share=0.0, 
                target_prey_id=0, 
                hunt_success=0.0)

        for prey in self.prey:
            prey.targeted = False

# ------------------------------------------------------------------------------------------------------------------------------------
    def step(self):
        """
        Runs one full cycle of the simulation:
        1. Finds and Shuffles every *Alive* Predator that hasn't acted yet
        2. For Every Predator , calculate the best expected utility for each Action and make their decision Dict
        3. Execution for every Predator's Decision
        4. Updating Agents and Stats
        """
        predator_list = self.predators
        prey_list = self.prey
        tick = self.current_tick

        avail_preds = [pr for pr in predator_list if pr.is_alive and not pr.has_acted]
        random.shuffle(avail_preds)

        for predator in avail_preds:                                       # For each *alive* Predator that hasn't *acted* yet
            self.form_best_decision(predator)                                       # Decide what is best for him to do
            self.execute_best_decision(predator)                                    # Execute based on that decision
            self.update_is_alive(predator)

        if tick % 10 == 0:
            for prey in prey_list:
                self.execute_prey_movement(prey)

        self.reset_cycle()

# =============================================================================
# 6. HELPFUL COMPUTATION CLASS
# =============================================================================
class COMPUTE:
    def D_eff_value(F_i, Dbase_i, C_hunt_i):
        """ Calculating Function for the Effective Bargaining Demand ( Deff,i ). """
        if C_hunt_i != 0: # Ensure F_i does not drop to zero (predator dies before F_i=0)            
            if F_i <= 0.001: 
                F_i = 0.001 # Treat near-death state as maximum demand

        return Dbase_i * (1 + math.exp(-F_i / C_hunt_i))

    def P_capture_value(P_solo, E_C, b_a, pred_count):
        """ Calculating Function for the propability of a successful capture of prey. """
        if pred_count > 1:
            beta = b_a
        else:
            beta = 1.0

        return 1.0 - math.pow(1.0 - P_solo, E_C * beta)
    
    def E_C_value(predators):
        """ Calculating Function for the Collective Effectiveness of a coalitional/solo capture chance. """
        return sum(p.M_i for p in predators)
    
    def Exp_Utility_value(P_capture, x_coal, R_a, C_hunt_i):
        """ Calculating Function for the expected Utility when hunting Prey(a). """
        """ Either use x_coal, or 1.0 for solo Hunt. """
        return P_capture * (x_coal * R_a) - C_hunt_i
    
    def Utility_check(pred, coal_success, share, prey):
        """ Returns True/False whether the expected utility of the group hunting exceeds their individual solo hunting utility. """
        solo_success = COMPUTE.P_capture_value(prey.P_solo, pred.M_i, prey.b_a, 1)

        solo_utility = COMPUTE.Exp_Utility_value(solo_success, 1.0, prey.R_a, pred.C_hunt_i)
        group_utility = COMPUTE.Exp_Utility_value(coal_success, share, prey.R_a, pred.C_hunt_i)
        return group_utility >= solo_utility
    
    def Bargaining_check(pred, share):
        """Return True/False whether the predator's share of the reward is higher than its effective Demand Factor. """
        return share >= COMPUTE.D_eff_value(pred.F_i, pred.Dbase_i, pred.C_hunt_i)
    
    def Egalitarian_split_value(coalition):
        """ Fair Split. Shares are equally divided among all Predators."""
        coalition_size = len(coalition)
        return 1/coalition_size

    def Meritocratic_split_value(predator, coalition):
        """ Contribution Split... Share is proportional to Predator's skill versus the sum skill of the coalition. """
        coalition_M_i = sum(pr.M_i for pr in coalition)

        return predator.M_i / coalition_M_i

    def Altruistic_split_value(predator, coalition):
        """ Need-Based Split... Share is proportional to Predator's Effective demand versus the sum of effective demands of the coalition. """
        demand = COMPUTE.D_eff_value(predator.F_i, predator.Dbase_i, predator.C_hunt_i)
        coalition_demand = sum(COMPUTE.D_eff_value(pr.F_i, pr.Dbase_i, pr.C_hunt_i) for pr in coalition)

        return demand / coalition_demand

    def Asymmetric_split_value(predator, initiator, coalition):
        """ Lion's share Split. Initiator gets a flat share cut and the rest is equally distributed among all Predators. """
        coalition_size = len(coalition)
        partners_share = (1 - LEADER_SHARE) / coalition_size

        if predator == initiator:
            return LEADER_SHARE + partners_share
        else:
            return partners_share

# =============================================================================
# 6. HEADLESS CONTROLLER (The Episode Loop)
# =============================================================================
def run_normal_simulation(episodes = 50):
    """ Main loop to run multiple episodes and log results to CSV. """
    logger = SimulationLogger("results")
    tactics = ["EGALITARIAN", "MERITOCRATIC", "ALTRUISTIC", "ASYMMETRIC"]
    max_ticks = 500

    for tactic in tactics:
        for ep in range(episodes):

            engine = SimulationEngine(
                predator_count=10, 
                prey_count=25, 
                radius=5, 
                tactic=tactic, 
                seed = ep
            )

            predators = engine.predators

            while  engine.current_tick < max_ticks and any(p.is_alive for p in predators):
                engine.step()
                engine.current_tick += 1

            episode_stats = logger.compute_episode_statistics(predators)
            logger.log_results(ep, tactic, episode_stats)
            # # if (ep + 1) % 5 == 0 or ep == episodes - 1:
            # logger.log_results(ep, predators, tactic)

    logger.plot_tactic_comparison_means("results")

def run_batch_simulation(episodes = 50, batch_size = 10):
    """ Main loop to run multiple episodes and log results to CSV. """
    logger = SimulationLogger("results")
    tactics = ["EGALITARIAN", "MERITOCRATIC", "ALTRUISTIC", "ASYMMETRIC"]
    max_ticks = 500

    batch_storage = {tactic: [] for tactic in tactics}

    for tactic in tactics:

        batch_id = 0
        for ep in range(episodes):

            engine = SimulationEngine(predator_count=10, prey_count=25, radius=5, tactic=tactic, seed = ep)
            predators = engine.predators

            # 1. --- Run the Episode ---
            while  engine.current_tick < max_ticks and any(p.is_alive for p in predators):
                engine.step()
                engine.current_tick += 1
            
            # 2. --- Compute the Episode Stats ---
            episode_stats = logger.compute_episode_statistics(predators)

            # 3. --- Store in batch ---
            batch_storage[tactic].append(episode_stats)

            # 4. --- If batch full -> aggregate + log ---
            if len(batch_storage[tactic]) == batch_size:
                batch_stats = logger.aggregate_batch(batch_storage[tactic])
                logger.log_batch_results(batch_id, tactic, batch_stats)

                batch_storage[tactic] = []
                batch_id += 1

        # 5. --- Handle Leftover Episodes --- 
        if batch_storage[tactic]:
            batch_stats = logger.aggregate_batch(batch_storage[tactic])

            logger.log_batch_results(batch_id, tactic, batch_stats)

            batch_storage[tactic] = []

    # 6. --- Plot the results ---
    logger.plot_tactic_comparison_means("results")

if __name__ == "__main__":
    run_batch_simulation()

# =============================================================================
# 7. SEED VERIFICATION SCRIPT
# =============================================================================
# if __name__ == "__main__":
#     # To prove reproducibility, we run two separate engines with the same seed.
#     print("RUNNING TEST 1...")
#     sim1 = SimulationEngine(predator_count=3, prey_count=2, radius=5, tactic="EGALITARIAN", seed=12345)
    
#     print("\nRUNNING TEST 2 (Should match Test 1 exactly)...")
#     sim2 = SimulationEngine(predator_count=3, prey_count=2, radius=5, tactic="EGALITARIAN", seed=12345)
    
#     # Check first predator's position to verify
#     if sim1.predators[0].q == sim2.predators[0].q and sim1.predators[0].r == sim2.predators[0].r:
#         print(f"1) Predator_{sim1.predators[0].agent_id} with ({sim1.predators[0].q},{sim1.predators[0].r})")
#         print(f"2) Predator_{sim2.predators[0].agent_id} with ({sim2.predators[0].q},{sim2.predators[0].r})")
#         print("\nSUCCESS: Seeds matched. Setup is deterministic.")
#     else:
#         print(f"1) Predator_{sim1.predators[0].agent_id} with ({sim1.predators[0].q},{sim1.predators[0].r})")
#         print(f"2) Predator_{sim2.predators[0].agent_id} with ({sim2.predators[0].q},{sim2.predators[0].r})")
#         print("\nFAILURE: Setup is inconsistent.")