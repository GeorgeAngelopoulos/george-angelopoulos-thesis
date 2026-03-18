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
STARTING_FOOD = 150.0
LEADER_SHARE = 0.2
PLACEHOLDER = 0

PREDATOR_PROFILES = {
    "Baseline": {"agent_type": "Baseline", "M_i": 1.5, "Dbase_i": 0.1, "C_hunt_i": 5.0, "F_i": STARTING_FOOD},
    "Expert": {"agent_type": "Expert", "M_i": 3.0, "Dbase_i": 0.15, "C_hunt_i": 3.0, "F_i": STARTING_FOOD},
    "Freeloader": {"agent_type": "Freeloader", "M_i": 1.0, "Dbase_i": 0.3, "C_hunt_i": 2.0, "F_i": STARTING_FOOD},
    "Greedy": {"agent_type": "Greedy", "M_i": 1.5, "Dbase_i": 0.4, "C_hunt_i": 5.0, "F_i": STARTING_FOOD},
    "Newbie": {"agent_type": "Newbie", "M_i": 0.5, "Dbase_i": 0.1, "C_hunt_i": 10.0, "F_i": STARTING_FOOD}
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

    def update_global_stats(self, predators, tactic):
        for p in predators:
            # Add this episode's stats to cumulative totals for this tactic
            self.global_stats[tactic][p.agent_id]["solo_hunts_performed"] += p.solo_hunts_performed
            self.global_stats[tactic][p.agent_id]["coalition_hunts_performed"] += p.coalition_hunts_performed
            self.global_stats[tactic][p.agent_id]["utility_gained_solo"] += p.utility_gained_solo
            self.global_stats[tactic][p.agent_id]["utility_gained_initiator"] += p.utility_gained_initiator
            self.global_stats[tactic][p.agent_id]["utility_gained_partner"] += p.utility_gained_partner

    def log_results(self, episode_id, predators, tactic):
        """
        Logs the final state of all predators for a single episode.
        Creates one row per predator (Tidy Data format) to allow for 
        detailed role and inequality analysis.
        
        Args:
            episode_id (int): The current episode number (0 to 1,000,000).
            predators (list): The list of Predator objects from your engine.
            tactic (str): The active tactic name (e.g., "EGALITARIAN", "ASYMMETRIC").
        """
        filepath = self._get_filepath(tactic)
        file_exists = os.path.exists(filepath)

        # Open in append mode to keep results from all episodes
        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header only if this is a new file for this tactic
            if not file_exists:
                # Add predator_type column
                headers = self.headers.copy()
                headers.insert(2, "predator_type")
                writer.writerow(headers)

            for p in predators:

                cumulative = self.global_stats[tactic][p.agent_id]

                row = [
                    episode_id,
                    p.agent_id,
                    p.agent_type,  # <-- log the predator type
                    cumulative["solo_hunts_performed"],
                    cumulative["coalition_hunts_performed"],
                    round(cumulative["utility_gained_solo"], 2),
                    round(cumulative["utility_gained_initiator"], 2),
                    round(cumulative["utility_gained_partner"], 2)
                ]
                writer.writerow(row)

    def plot_predator_dynamics(self, results_dir="results"):
        """
        For each predator:
            - One figure
            - 3x2 subplot grid
            - X-axis = episode_id
            - Y-axis = per-interval change (not cumulative)
            - Lines = different tactics
            - Predator type included in the figure title
        """

        pattern = os.path.join(results_dir, "results_*.csv")
        files = glob.glob(pattern)

        if not files:
            print("No result files found.")
            return

        all_data = {}
        predators_set = set()

        # Load CSVs
        for filepath in files:
            filename = os.path.basename(filepath)
            tactic = filename.replace("results_", "").replace(".csv", "")

            df = pd.read_csv(filepath)
            predators_set.update(df["predator_id"].unique())
            all_data[tactic] = df

        predators = sorted(list(predators_set))
        tactics = sorted(list(all_data.keys()))

        stats = [
            "solo_hunts_performed",
            "coalition_hunts_performed",
            "utility_gained_solo",
            "utility_gained_initiator",
            "utility_gained_partner"
        ]

        # Nice display names for subplots
        stat_names = {
            "solo_hunts_performed": "Solo Hunts Performed",
            "coalition_hunts_performed": "Coalition Hunts Performed",
            "utility_gained_solo": "Utility Gained (Solo)",
            "utility_gained_initiator": "Utility Gained (Initiator)",
            "utility_gained_partner": "Utility Gained (Partner)"
        }

        # Fixed colors for tactics
        color_map = {
            "EGALITARIAN": "green",
            "ASYMMETRIC": "orange",
            "ALTRUISTIC": "red",
            "MERITOCRATIC": "blue"
        }

        for predator in predators:

            fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
            axes = axes.flatten()

            # 🔹 Get predator type from the first tactic where it exists
            pred_type = None
            for tactic in tactics:
                df = all_data[tactic]
                predator_df = df[df["predator_id"] == predator]
                if not predator_df.empty:
                    pred_type = predator_df["predator_type"].iloc[0]
                    break

            fig.suptitle(
                f"Predator {predator} ({pred_type}) - Per-Interval Dynamics Across Tactics",
                fontsize=14
            )

            for i, stat in enumerate(stats):

                ax = axes[i]

                for tactic in tactics:
                    df = all_data[tactic]
                    predator_df = df[df["predator_id"] == predator]

                    if predator_df.empty:
                        continue

                    predator_df = predator_df.sort_values("episode_id")

                    # 🔹 Compute per-interval change (diff)
                    values = predator_df[stat].diff().fillna(0)

                    ax.plot(
                        predator_df["episode_id"],
                        values,
                        label=tactic,
                        color=color_map.get(tactic, "black")
                    )

                ax.set_title(stat_names[stat])
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.axhline(0, linewidth=1, color='black')

            # Hide unused subplot (6th slot)
            axes[-1].axis("off")

            # X labels only for bottom row
            axes[4].set_xlabel("Episode")
            axes[5].set_xlabel("Episode")

            # Legend once
            axes[1].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    def plot_tactic_comparison_means(self, results_dir="results"):
        """
        For each predator type:

        Figure 1 (1x2):
            - Solo Hunts
            - Coalition Hunts

        Figure 2 (1x3):
            - Utility Solo
            - Utility Initiator
            - Utility Partner

        Each plot:
            - Bar chart across tactics
            - Mean across episodes
            - 95% confidence interval
            - Numeric value labels above bars
        """

        pattern = os.path.join(results_dir, "results_*.csv")
        files = glob.glob(pattern)

        if not files:
            print("No result files found.")
            return

        all_data = {}
        predator_types = set()

        for filepath in files:
            filename = os.path.basename(filepath)
            tactic = filename.replace("results_", "").replace(".csv", "")

            df = pd.read_csv(filepath)
            predator_types.update(df["predator_type"].unique())
            all_data[tactic] = df

        tactics = sorted(all_data.keys())
        predator_types = sorted(predator_types)

        hunt_stats = [
            "solo_hunts_performed",
            "coalition_hunts_performed"
        ]

        utility_stats = [
            "utility_gained_solo",
            "utility_gained_initiator",
            "utility_gained_partner"
        ]

        stat_names = {
            "solo_hunts_performed": "Solo Hunts Performed",
            "coalition_hunts_performed": "Coalition Hunts Performed",
            "utility_gained_solo": "Utility Gained (Solo)",
            "utility_gained_initiator": "Utility Gained (Initiator)",
            "utility_gained_partner": "Utility Gained (Partner)"
        }

        color_map = {
            "EGALITARIAN": "green",
            "ASYMMETRIC": "orange",
            "ALTRUISTIC": "red",
            "MERITOCRATIC": "blue"
        }

        bar_width = 0.5

        def compute_stats(df, predator_type, stat):
            means = []
            ci_values = []
            colors = []

            for tactic in tactics:

                tactic_df = df[tactic]
                type_df = tactic_df[tactic_df["predator_type"] == predator_type]

                if type_df.empty:
                    means.append(0)
                    ci_values.append(0)
                    colors.append(color_map.get(tactic, "black"))
                    continue

                type_df = type_df.sort_values(["predator_id", "episode_id"])

                type_df["diff"] = (
                    type_df.groupby("predator_id")[stat]
                    .diff()
                    .fillna(0)
                )

                episode_values = (
                    type_df.groupby("episode_id")["diff"]
                    .mean()
                )

                mean_val = episode_values.mean()
                std_val = episode_values.std()
                n = len(episode_values)

                ci = 1.96 * (std_val / np.sqrt(n)) if n > 0 else 0

                means.append(mean_val)
                ci_values.append(ci)
                colors.append(color_map.get(tactic, "black"))

            return means, ci_values, colors

        def add_value_labels(ax, bars, means, cis):
            """Add mean ± CI above each bar."""
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

        for predator_type in predator_types:

            # =========================
            # FIGURE 1 — Hunts
            # =========================

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
                    color=colors,
                    capsize=6
                )

                add_value_labels(ax, bars, means, ci_values)

                # ADD SPACE ABOVE BARS
                max_height = max([m + c for m, c in zip(means, ci_values)])
                ax.set_ylim(0, max_height * 1.3)

                ax.set_xticks(x)
                ax.set_xticklabels(tactics)
                ax.set_title(stat_names[stat])
                ax.set_ylabel("Mean Value (95% CI)")
                ax.grid(True, linestyle="--", alpha=0.3)

            fig.suptitle(
                f"Hunting Performance — Predator Type: {predator_type}",
                fontsize=16
            )

            plt.subplots_adjust(wspace=0.35, top=0.85)

            plt.show()

            # =========================
            # FIGURE 2 — Utilities
            # =========================

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
                    color=colors,
                    capsize=6
                )

                add_value_labels(ax, bars, means, ci_values)

                # ADD SPACE ABOVE BARS
                max_height = max([m + c for m, c in zip(means, ci_values)])
                ax.set_ylim(0, max_height * 1.3)

                ax.set_xticks(x)
                ax.set_xticklabels(tactics)
                ax.set_title(stat_names[stat])
                ax.set_ylabel("Mean Value (95% CI)")
                ax.grid(True, linestyle="--", alpha=0.3)

            fig.suptitle(
                f"Utility Distribution — Predator Type: {predator_type}",
                fontsize=16
            )

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
        """ Calculates the expected utility for REST, MOVE, SOLO_HUNT and COALITION_HUNT. """
        """ Returns the Best Action to execute based on that utility. """
        avail_actions = []
        utility_options = {}
        
        # --- INITIAL OBSERVATION ---
        predators_present, prey_present = self.get_nearby_agents(predator, predator.q, predator.r)

        # --- BASE AVAILABLE ACTIONS ---
        avail_actions.append(Action.REST)
        avail_actions.append(Action.MOVE)
        
        # --- CONDITIONAL ACTIONS ---
        if prey_present:
            avail_actions.append(Action.SOLO_HUNT)

        if prey_present and predators_present:
            avail_actions.append(Action.COALITION_HUNT)

        
        # --- FINDING EXPECTED UTILITIES ---
        for action in avail_actions:

            if action == Action.REST:
                utility_options[action] = self.Action_REST_utility()

            elif action == Action.MOVE:
                utility_options[action], best_q, best_r = self.Action_MOVE_utility(predator)
            
            elif action == Action.SOLO_HUNT:
                utility_options[action], target_prey = self.Action_SOLO_HUNT_utility(predator, prey_present)

            elif action == Action.COALITION_HUNT:
                utility_options[action], target_prey, potential_partners, share = self.Action_COALITION_HUNT_utility(predator, predators_present, prey_present)
        
        # --- DECISION BASED ON MAX EXPECTED UTILITY --- 
        best_action = max(utility_options, key=utility_options.get)

        if best_action == Action.MOVE:
            predator.decision = {
                'action_type': best_action,
                'utility': utility_options[best_action],
                'target_q': best_q,
                'target_r': best_r,
                'partners': None,       
                'final_share': 0.0,                    
                'target_prey_id': 0,                             
                'hunt_success': 0.0
            }

        elif best_action == Action.SOLO_HUNT:
            target_prey.targeted = True
            predator.decision = {
                'action_type': best_action,
                'utility': utility_options[best_action],
                'target_q': predator.q,
                'target_r': predator.r,
                'partners': None,       
                'final_share': 1.0,                    
                'target_prey_id': target_prey.prey_id,                             
                'hunt_success': self.get_hunt_success(predator, None, target_prey)                
            }

        elif best_action == Action.COALITION_HUNT:
            target_prey.targeted = True
            predator.decision = {
                'action_type': best_action,
                'utility': utility_options[best_action],
                'target_q': predator.q,
                'target_r': predator.r,
                'partners': potential_partners,       
                'final_share': share,                    
                'target_prey_id': target_prey.prey_id,                             
                'hunt_success': self.get_hunt_success(predator, potential_partners, target_prey)                
            }

        else: # Action.Rest
            predator.decision = {
                'action_type': best_action,
                'utility': utility_options[best_action],
                'target_q': predator.q,
                'target_r': predator.r,
                'partners': None,       
                'final_share': 0.0,                    
                'target_prey_id': 0,                             
                'hunt_success': 0.0
            }

        predator.has_acted = True

# ------------------------------------------------------------------------------------------------------------------------------------
# --- THE UTILITY FUNCTIONS ---
# ------------------------------------------------------------------------------------------------------------------------------------     
    def Action_REST_utility(self):
        """ Calculates the Expected Utility of resting, E[U_Rest] = 0.0 (Idle, No Gain). """
        return 0.0

    def Action_MOVE_utility(self, predator):
        """ Calculates the best Expected Utility of Moving to a specific Hex. """
        """ Using the Bellman's Equation, Hexes with worthwhile prey are prioritized. """
        gamma = 0.9
        neighbor_hexes = self.grid.get_neighbor_hexes(predator.q, predator.r)

        best_move_utility = float('-inf')
        best_hex = None

        for (new_q, new_r) in neighbor_hexes:
            
            V_best_hex = 0
            for prey in self.prey:

                P_capture = COMPUTE.P_capture_value(prey.P_solo, predator.M_i, prey.b_a, 1)
                U_solo = COMPUTE.Exp_Utility_value(P_capture, 1.0, prey.R_a, predator.C_hunt_i)

                distance = axial_distance((new_q, new_r), (prey.q, prey.r))

                if distance == 0:
                    V_adj_hex = U_solo
                else:
                    discounted_reward = (gamma ** distance) * U_solo
                    discounted_move_cost = C_MOVE * (1 - gamma ** distance)  / (1 - gamma)
                    V_adj_hex = discounted_reward - discounted_move_cost

                V_best_hex = max(V_best_hex, V_adj_hex)

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
        
        success = random.random() < predator.decision['hunt_success']
        partners = predator.decision['partners'] or []
        
        participants = [predator] + partners

        for p in participants:

            if success:
                reward = p.decision['final_share'] * target_prey.R_a
                utility = reward - p.C_hunt_i
            else:
                utility = (-1)*p.C_hunt_i
            
            p.F_i += utility

            if p == predator and p.decision['action_type'] == Action.COALITION_HUNT:
                p.utility_gained_initiator += utility
                p.coalition_hunts_performed += 1

            elif p != predator and p.decision['action_type'] == Action.COALITION_HUNT:
                p.utility_gained_partner += utility
                p.coalition_hunts_performed += 1

            elif p.decision['action_type'] == Action.SOLO_HUNT:
                p.utility_gained_solo += utility
                p.solo_hunts_performed += 1            

    def execute_coalition_negotiation(self, initiator, target_prey):
        """ 
        1. Initiator proposes a coalition hunt for the chosen prey using the same split rule to all predator partners.

        2. Each predator either accepts or declines the Utility Check by:
            - Evaluating its computed share vs their computed Effective Demand Factor (Deff_i)
            
        3. Coalition forms with accepted predators only.
            - Real shares are recomputed based on the new coalition size

        4. Hunt occurs
            - Success depends on realized coalition size
            - Rewards are distributed

        5. Turn Resolution:
            - Initiator + acceptors -> have acted, ending their turn
            - Rejectors -> have not acted, will decide and act normally later 
        """
        coalition_predators = initiator.decision['partners']

        while True:

            total_E_C = COMPUTE.E_C_value([initiator])
            total_E_C += COMPUTE.E_C_value(coalition_predators)
            coalition_success = COMPUTE.P_capture_value(target_prey.P_solo, total_E_C, target_prey.b_a, len([initiator]+coalition_predators))

            coalition_shares = self.get_split_shares(initiator, [initiator]+coalition_predators)
            
            accepted = []
            rejected = []

            for pred in coalition_predators:
                if self.get_coalition_checks(pred, coalition_success, coalition_shares[pred], target_prey):
                    accepted.append(pred)
                else:
                    rejected.append(pred)
            
            coalition_predators = accepted

            if len(coalition_predators) == 0:

                self.update_decision(initiator, 
                    action_type = Action.SOLO_HUNT,
                    hunt_success = COMPUTE.P_capture_value(target_prey.P_solo, initiator.M_i, target_prey.b_a, 1),
                    final_share = 1.0,
                    partners = None
                )
                break

            else:
                self.update_decision(initiator, 
                    hunt_success = coalition_success,
                    final_share = coalition_shares[initiator],
                    partners = coalition_predators
                )

                for pred in coalition_predators:

                    self.update_decision(pred,
                        action_type = Action.COALITION_HUNT, 
                        hunt_success = coalition_success,
                        final_share = coalition_shares[pred],
                    )                    
                    pred.has_acted = True
                break
                

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
def run_batch_simulation(episodes = 50):
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

            logger.update_global_stats(predators, tactic)
            # if (ep + 1) % 5 == 0 or ep == episodes - 1:
            logger.log_results(ep, predators, tactic)

    logger.plot_tactic_comparison_means("results")

if __name__ == "__main__":
    run_batch_simulation()
    #print("Structure ready. Waiting for step-by-step logic implementation.")

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