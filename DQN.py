import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import random
from collections import deque
from Project import *
# --- THE BRAIN ---
# Input: [Price, Inventory, Time] -> Output: 11 possible sell-rates
model_structure = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 11)  # Actions: 0%, 10%, 20%... 100% of current inventory
)

net_policy = copy.deepcopy(model_structure)
net_target = copy.deepcopy(model_structure)
net_target.load_state_dict(net_policy.state_dict())
optimizer = optim.Adam(net_policy.parameters(), lr=0.001)

# --- THE MEMORY (REPLAY BUFFER) ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity=5000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = random, 1 = full)
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        # New transitions get maximum priority to ensure they are trained on at least once
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # Calculate sampling probabilities: P(i) = p_i^alpha / sum(p_k^alpha)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance Sampling (IS) weights to correct the bias of non-random sampling
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (states, actions, rewards, next_states, dones, 
                indices, torch.FloatTensor(weights))

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5 # Add small constant to avoid zero priority

    def __len__(self):
        return len(self.buffer)

memory = PrioritizedReplayBuffer()

# --- UTILITIES ---
def get_state_norm(state, env):
    """ Scales state variables to [0, 1] range for the Neural Network """
    return np.array([
        state[0] / env["S0"],        # Price / Initial Price
        state[1] / env["q0"],        # Current / Total Inventory
        state[2] / env["N"]          # Current Step / Total Steps
    ], dtype=np.float32)

def select_action(state_norm, epsilon):
    """ Epsilon-Greedy selection """
    if random.random() < epsilon:
        return random.randint(0, 10)
    state_t = torch.FloatTensor(state_norm).unsqueeze(0)
    with torch.no_grad():
        return torch.argmax(net_policy(state_t)).item()
    
def train_step(batch_size=32, gamma=0.99, beta=0.4):
    if len(memory) < batch_size: return
    
    (states, actions, rewards, next_states, dones, 
     indices, weights) = memory.sample(batch_size, beta)
    
    # Convert lists to tensors
    states_t = torch.FloatTensor(np.array(states))
    next_states_t = torch.FloatTensor(np.array(next_states))
    actions_t = torch.LongTensor(actions).view(-1, 1)
    rewards_t = torch.FloatTensor(rewards).view(-1, 1)
    dones_t = torch.FloatTensor(dones).view(-1, 1)
    weights_t = weights.view(-1, 1) # Weights for IS

    # 2. Compute TD-error
    current_q = net_policy(states_t).gather(1, actions_t)
    with torch.no_grad():
        max_next_q = net_target(next_states_t).max(1)[0].view(-1, 1)
        expected_q = rewards_t + (gamma * max_next_q * (1 - dones_t))
    
    # TD error is the absolute difference
    td_errors = torch.abs(current_q - expected_q).detach().numpy()

    # 3. Update priorities in buffer
    memory.update_priorities(indices, td_errors.flatten())

    # 4. Apply weighted loss (using Huber/SmoothL1 for stability)
    loss = (weights_t * nn.functional.mse_loss(current_q, expected_q, reduction='none')).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_dqn(env, episodes=300, eps_decay=0.995):
    epsilon = 1.0
    for ep in range(episodes):
        state = reset_env(env)
        done = False
        while not done:
            s_norm = get_state_norm(state, env)
            
            # Check if last step to force liquidation
            if env["t"] >= (env["N"] - 1):
                action_idx = 10
            else:
                action_idx = select_action(s_norm, epsilon)
                
            v_t = (action_idx / 10.0 * env["q"]) / env["dt"]
            next_state, reward, done, info = step_env(env, v_t)
            
            # Push to PER buffer
            ns_norm = get_state_norm(next_state, env)
            memory.push(s_norm, action_idx, reward, ns_norm, done)
            
            # Step-level update with beta annealing
            fraction = min(ep / episodes, 1.0)
            beta = 0.4 + fraction * (1.0 - 0.4)
            train_step(batch_size=b_size, beta=beta)
            
            state = next_state
            
        epsilon = max(0.01, epsilon * eps_decay)
        
        # Periodic target sync
        if ep % 10 == 0:
            net_target.load_state_dict(net_policy.state_dict())
            
def run_dqn_eval(env):
    """ Runs the trained agent without randomness """
    state = reset_env(env)
    done = False
    history = []
    while not done:
        s_norm = get_state_norm(state, env)
        action_idx = select_action(s_norm, epsilon=0) # 0 = no randomness
        v_t = (action_idx / 10.0 * env["q"]) / env["dt"]
        
        curr_q, curr_S = env["q"], env["S"]
        state, reward, done, info = step_env(env, v_t)
        history.append({"step": env["t"], "inventory": curr_q, "cash": info["cash"]})
    return pd.DataFrame(history)

def calculate_tca(results, env):
    # The value if we could sell everything instantly at t=0
    benchmark_value = env["q0"] * env["S0"] 
    actual_value = results['cash'].iloc[-1]
    
    # Implementation Shortfall in Basis Points (bps)
    slippage_usd = benchmark_value - actual_value
    slippage_bps = (slippage_usd / benchmark_value) * 10000
    
    return {
        "Slippage ($)": slippage_usd,
        "Slippage (bps)": slippage_bps,
        "Avg Price": actual_value / env["q0"]
    }

def plot_final_comparison(env, best_dqn_df):
    plt.figure(figsize=(12, 7))
    steps = np.arange(env["N"] + 1)
    
    # DQN
    plt.plot(best_dqn_df['step'], best_dqn_df['inventory'], label="Best DQN", linewidth=3, color='blue')

    # TWAP
    twap_inv = [env['q0'] * (1 - t/env['N']) for t in steps]
    plt.plot(steps, twap_inv, label="TWAP", linestyle='--', color='gray')

    # VWAP Proxy (Commonly front-loaded)
    vwap_inv = [env['q0'] * (1 - (t/env['N'])**0.7) for t in steps]
    plt.plot(steps, vwap_inv, label="VWAP Proxy", linestyle=':', color='green')

    # Almgren-Chriss
    kappa = np.sqrt(abs(env['gamma'] / env['eta']))
    T = env['N'] * env['dt']
    ac_inv = [env['q0'] * (np.sinh(kappa * (T - t * env['dt'])) / np.sinh(kappa * T)) for t in steps]
    plt.plot(steps, ac_inv, label="Almgren-Chriss", color='red', linewidth=2)

    plt.title("Inventory Decay: DQN vs Benchmarks")
    plt.xlabel("Step"); plt.ylabel("Shares Remaining"); plt.legend(); plt.grid(True)
    plt.show()
     
if __name__ == "__main__":
    # --- Setup the Grid ---
    S0_val = df_reg['Close'].iloc[-1].item() 
    q0_val = 100000.0
    scaled_eta = eta_estimate / q0_val 
    
    env = init_env(S0=S0_val, q0=q0_val, gamma=gamma_estimate, 
                   eta=scaled_eta, sigma_daily=sigma_daily, seed=42)
    
    lr = 0.001
    alphas = [0.8, 0.7, 0.6]                # 0.6=Aggressive PER, 0.2=More random + , 0.4, 0.3
    decays = [0.99, 0.98, 0.975]            # Exploration speed + 0.999, 0.998, 0.995, 
    batch_sizes = [32, 64]  #, 128

    all_results = []

    print(f"{'='*60}")
    print(f"STARTING HYPERPARAMETER GRID SEARCH")
    print(f"{'='*60}")

    best_overall_price = -np.inf
    best_df = None

    for alpha in alphas:
        for decay in decays:
            for b_size in batch_sizes:
                print(f"\n[RUNNING] LR: {lr} | Alpha: {alpha} | Decay: {decay} | Batch: {b_size}")
                
                # --- 3. RESET INFRASTRUCTURE ---
                net_policy = nn.Sequential(
                    nn.Linear(3, 64), 
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 11)
                )
                net_target = copy.deepcopy(net_policy)
                optimizer = optim.Adam(net_policy.parameters(), lr=lr)
                memory = PrioritizedReplayBuffer(capacity=5000, alpha=alpha)
                
                # --- 4. TRAIN ---
                # Note: pass b_size to a modified train_dqn/train_step if needed
                train_dqn(env, episodes=400, eps_decay=decay) 
                
                # --- 5. EVALUATE ---
                dqn_eval_df = run_dqn_eval(env)
                metrics = calculate_tca(dqn_eval_df, env)
                
                # Add hyperparams to results for the table
                metrics.update({
                    "LR": lr, "Alpha": alpha, "Decay": decay, "Batch": b_size
                })
                all_results.append(metrics)
                
                print(f"Success: Avg Price ${metrics['Avg Price']:.2f}")

                if metrics['Avg Price'] > best_overall_price:
                    best_overall_price = metrics['Avg Price']
                    best_df = dqn_eval_df

                    all_results.append(metrics)
    # --- 6. FINAL LEADERBOARD ---
    leaderboard = pd.DataFrame(all_results).sort_values("Slippage ($)")
    print("\n" + "="*90)
    print("HYPERPARAMETER RANKING (TOP 10)")
    print("="*90)
    print(leaderboard[['LR', 'Alpha', 'Decay', 'Batch', 'Avg Price', 'Slippage ($)']].head(10).to_string(index=False))
    
    plot_final_comparison(env, best_df)