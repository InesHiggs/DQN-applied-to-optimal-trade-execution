import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt

# ------------- 1. Download data -------------

stockData = yf.download('MSFT', '2019-01-10', '2024-05-15', auto_adjust=False)

# Quick sanity plot
stockData['Close'].plot(title='MSFT Close')
plt.show()
print(stockData.columns)

# ------------- 2. Build features -------------

#(close_t - close_{t-1})/close_{t-1} = return at time t
#With .shift(-1) -> row t contains tomorrow's return
stockData['perm_return'] = stockData['Close'].pct_change().shift(-1)

# For the Temporary impact
# Intraday slippage: Close - Open -> recursion is the Y axis
stockData['temp_cost'] = stockData['Close'] - stockData['Open']

# For the permanent impact
# If Close > Open => net buying ( +1 ), else net selling ( -1 ) -> recursion is the X axis
stockData['trade_sign'] = np.sign(stockData['Close'] - stockData['Open'])

# Removes any row that contains a NaN anywhere
df_reg = stockData.dropna()

# ------------- 3. Estimate permanent impact gamma (γ) -------------

Y_perm = df_reg['perm_return']                   # dependent variable -> this should be tomorrow's return
X_perm = sm.add_constant(df_reg['trade_sign'])   # regress on trade sign 

perm_model = sm.OLS(Y_perm, X_perm).fit()
gamma_estimate = perm_model.params['trade_sign']

print("\n--- Permanent Impact (gamma) Regression ---")
print(perm_model.summary())
print(f"\nEstimated Gamma (γ): {gamma_estimate:.6f} (return per unit trade sign)")

# ------------- 4. Estimate temporary impact eta (η) -------------

Y_temp = df_reg['temp_cost']                     # in dollars
X_temp = sm.add_constant(df_reg['trade_sign'])

temp_model = sm.OLS(Y_temp, X_temp).fit()
eta_estimate = temp_model.params['trade_sign']

print("\n--- Temporary Impact (eta) Regression ---")
print(temp_model.summary())
print(f"\nEstimated Eta (η): {eta_estimate:.4f} USD per unit trade sign")

# ------------- 5. Volatility sigma (σ) -------------

#log(S_t) - log(S_{t-1}) ~=(S_t - S_{t-1})/S_{t-1} = return 
log_returns = np.log(df_reg['Close']).diff().dropna()
sigma_daily = log_returns.std().iloc[0]

print("\n--- Final Parameters for ExecutionEnv ---")
print(f"gamma (γ): {gamma_estimate:.6f}")
print(f"eta (η):   {eta_estimate:.4f} USD")
print(f"Daily volatility σ: {sigma_daily:.4f}")

def init_env(S0,
             q0,
             gamma,
             eta,
             sigma_daily,
             T=1.0,        # trading horizon in "days"
             N=20,         # number of time steps
             side='sell',  # 'sell' or 'buy'
             seed=None):
    """
    Create an AC execution environment as a dict.
    """
    dt = T / N
    sigma_step = sigma_daily * np.sqrt(dt)
    rng = np.random.default_rng(seed)

    env = {
        "S0": float(S0), #Initial Price
        "q0": float(q0), #initial stock number
        "gamma": float(gamma),
        "eta": float(eta),
        "sigma_daily": float(sigma_daily),
        "sigma_step": float(sigma_step),
        "T": float(T),
        "N": int(N),
        "dt": float(dt),
        "side": side, #Sell or buy
        "rng": rng,
        # mutable state (filled by reset_env)
        "t": None,
        "S": None, #Current Price
        "q": None, #current inventory
        "cash": None, #accumulated money
        "done": None, #whether episode is over
    }
    return env


def reset_env(env):
    """
    Reset the environment to the start of a new episode.
    Returns the initial state (S, q, t).
    """
    env["t"] = 0
    env["S"] = env["S0"]
    env["q"] = env["q0"]
    env["cash"] = 0.0
    env["done"] = False

    state = np.array([env["S"], env["q"], env["t"]], dtype=float)
    return state


def step_env(env, action):
    """
    One AC step:
      - action = trading rate v_t (shares per unit time)
      - updates env in-place
      - returns (state, reward, done, info)
    """
    if env["done"]:
        raise RuntimeError("Episode is done, call reset_env(env) to start again.")

    v_t = float(action)  # trading rate

    dt = env["dt"]
    side = env["side"]
    rng = env["rng"]

    # Enforce basic constraints on v_t
    if side == "sell":
        v_t = max(0.0, v_t)  # cannot buy in a sell program
        max_v = env["q"] / dt  # don't trade more than remaining inventory
        v_t = min(v_t, max_v)
        n_t = v_t * dt         # shares sold this step
    else:  # 'buy'
        v_t = max(0.0, v_t)
        max_v = env["q0"] - env["q"]  # crude cap, can refine later
        max_v = max(max_v, 0.0)
        n_t = v_t * dt

    # Execution price (temporary impact)
    S_exec = env["S"] - env["eta"] * v_t 

    # Cash & inventory update
    if side == "sell":
        env["cash"] += n_t * S_exec
        env["q"] -= n_t
    else:
        env["cash"] -= n_t * S_exec
        env["q"] += n_t

    # Mid-price update (permanent impact + noise)
    noise = rng.normal()
    env["S"] = env["S"] + env["sigma_step"] * noise - env["gamma"] * v_t * dt

    # Time update
    env["t"] += 1

    # Termination condition -> end of trading horizonor no more shares
    env["done"] = (env["t"] >= env["N"]) or (env["q"] <= 1e-8)

    # 1. SCALE THE REWARD: Divide by S0 to bring it into "return" space -> avoids selling too fast
    step_revenue = (n_t * S_exec) / env["S0"]
    
    # Scale penalty to be a percentage of the trade value, not a massive raw number
    #Discourages leaving any shares behind
    impact_penalty = 50.0 * (env["eta"] * (v_t**2) * env["dt"]) / env["S0"]
    reward = step_revenue - impact_penalty
    
    # Optional terminal penalty for leftover inventory
    if env["done"] and env["q"] > 1e-8:
        #End with leftover inventory -> forced to dump it at the final mid-price.
        penalty = (env["q"] * env["S"]) / env["S0"]
        reward -= penalty
        env["cash"] -= (env["q"] * env["S"])
        env["q"] = 0.0

    # New state
    state = np.array([env["S"], env["q"], env["t"]], dtype=float)
    info = {
        "S_exec": S_exec, # execution price per share this step
        "n_t": n_t, # number of shares traded this step
        "cash": env["cash"], # total accumulated money from selling
    }

    return state, reward, env["done"], info

def run_twap(env):
    '''
    Executes a standard TWAP strategy in the provided environment
    '''
    state = reset_env(env)
    done = False
    twap_v_t = env["q0"] / env["T"] #amount of shares for each time interval
    
    history = []
    
    while not done:
        # Save current state before the step
        current_q = env["q"] #current inventory
        current_S = env["S"] #Current price
        
        # Take the step
        state, reward, done, info = step_env(env, twap_v_t)
        
        history.append({
            "step": env["t"],
            "price": current_S,
            "inventory": current_q,
            "exec_price": info["S_exec"],
            "cash": info["cash"]
        })
        
    return pd.DataFrame(history)
    
def get_volume_profile(N):
    #The data is daily, so I'll create a 'U-shaped' volume curve
    #(High volume at open/close, low at lunch)
    x = np.linspace(-2, 2, N)
    profile = np.exp(x**2) #Creates a U-shape
    return profile / profile.sum()

def run_vwap(env, volume_profile):
    state = reset_env(env)
    done = False
    history = []
    
    while not done:
        #Sell a % of the shares based on yhe volume profile
        # current_step_volume_percentage * Total_Shares / time_delta
        target_shares = volume_profile[env['t']] * env['q0']
        v_t = target_shares / env['dt']
        
        curr_q, curr_S = env['q'], env['S']
        state, reward, done, info = step_env(env, v_t)
        
        history.append({
            "step": env["t"], "inventory": curr_q,
            "exec_price": info["S_exec"], "cash": info["cash"]
        })
    return pd.DataFrame(history)

def run_almgren_chriss(env, risk_aversion=1e-6):
    '''
    Executes the Almgren Chriss optimal strategy
    '''   
    state = reset_env(env)
    done= False
    history = []
    
    #kappa = balance between the impact and the volatility risk
    kappa = np.sqrt((risk_aversion * env["sigma_daily"]**2) / env["eta"])
    
    while not done:
        t_remaining = env["N"] - env["t"]
        
        if t_remaining <= 1: #If we are at the last time interval
            v_t = env["q"] / env["dt"]  #Sell everything
        else:
            # AC formula for the optimal number of shares to sell in this step
            n_t = (2 * np.sinh(kappa * env["dt"] / 2) / np.sinh(kappa * env["T"])) * \
                  np.cosh(kappa * (env["T"] - (env["t"] * env["dt"] + env["dt"]/2))) * env["q0"]
            v_t = n_t / env["dt"]
        
        curr_q, curr_S = env["q"], env["S"]
        state, reward, done, info = step_env(env, v_t)
        
        history.append({
            "step": env["t"],
            "inventory": curr_q,
            "price": curr_S,
            "exec_price": info["S_exec"],
            "cash": info["cash"]
        })
    return pd.DataFrame(history)

if __name__ == "__main__":
    # 1. Setup Environment with Scaled Eta
    S0_val = df_reg['Close'].iloc[-1].item() 
    q0_val = 100000.0
    scaled_eta = eta_estimate / q0_val 
    
    env = init_env(S0=S0_val, q0=q0_val, gamma=gamma_estimate, 
                   eta=scaled_eta, sigma_daily=sigma_daily, seed=42)

    # 2. Run TWAP
    twap_results = run_twap(env)
    
    # 3. Run VWAP (using U-shaped profile)
    x = np.linspace(-2, 2, 20)
    v_profile = np.exp(x**2)
    v_profile /= v_profile.sum()
    vwap_results = run_vwap(env, v_profile)
    
    #4. Run AC
    ac_results = run_almgren_chriss(env, risk_aversion=1e-1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot 1: Inventory (The "How") ---
    ax1.plot(twap_results['step'], twap_results['inventory'], label='TWAP (Linear)', linestyle='--')
    ax1.plot(vwap_results['step'], vwap_results['inventory'], label='VWAP (U-Shape)')
    ax1.plot(ac_results['step'], ac_results['inventory'], label='Almgren-Chriss', linewidth=2)
    ax1.set_title("Inventory Decay Comparison")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Shares Held")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Cumulative Cash (The "Result") ---
    ax2.plot(twap_results['step'], twap_results['cash'], label='TWAP', linestyle='--')
    ax2.plot(vwap_results['step'], vwap_results['cash'], label='VWAP')
    ax2.plot(ac_results['step'], ac_results['cash'], label='AC', linewidth=2)
    ax2.set_title("Total Cash Captured")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Cash ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()