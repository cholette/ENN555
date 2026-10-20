# # %% Condition-based cleaning of heliostat mirrors (ENN555, week 13)
#
# This script uses the MDP solver in maintkit, which is currently on the
# `mdp` branch (not yet on main). Install it into your ENN555 environment with:
#
#     pip install "git+https://github.com/cholette/maintkit.git@mdp"
#
# This also installs maintkit's dependencies (numpy, scipy, matplotlib,
# numdifftools). If you already have maintkit installed from main, add
# --force-reinstall to replace it.
#
# The mdp branch is still changing. If an import below fails after you
# reinstall, check the branch for renamed classes, or install the exact
# version this script was written against:
#
#     pip install "git+https://github.com/cholette/maintkit.git@27f06d8"
#
# Check the install with:  python -c "from maintkit.mdp import MDP"

# %% 
from maintkit import mdp
from maintkit.time_indexed import TimeIndexed
import matplotlib.pyplot as plt
from enn555.paths import data_dir
import pandas as pd
import numpy as np
import scipy.stats as sps
from datetime import datetime, timedelta
from tqdm import tqdm

dd = data_dir()/'cbm_tutorial'
df = pd.read_csv(dd/'POWER_Point_Hourly_20150101_20241231_027d47S_153d01E_UTC.csv',header=19)

df['Timestamp'] = pd.to_datetime(dict(
                                year=df['YEAR'],
                                month=df['MO'],
                                day=df['DY'],
                                hour=df['HR'],
                                minute=df.get('minute', 0),  # use 0 if not present
                                second=df.get('second', 0),  # use 0 if not present
                                ))

# Adjust timezone
utc_str = "+10:00"
df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC')
df['Timestamp'] = df['Timestamp'].dt.tz_convert(utc_str)

# NASA POWER marks missing values as -999; treat them as no sun
df['ALLSKY_SFC_SW_DNI'] = df['ALLSKY_SFC_SW_DNI'].clip(lower=0)


# take only complete years
mask = (df['Timestamp'].dt.year > 2015) & (df['Timestamp'].dt.year < 2025)
df = df.iloc[mask,:]

# %% Set up state grid, compute losses
Δρ = 0.002
N_years = 2024-2015
price = 100
pr = lambda d: 0.075 + 0.04*np.cos(2*np.pi*(d-15)/365)
loss_dist = sps.gamma(4,scale=0.001)
rain_recovery = sps.weibull_min(0.471,scale=0.0045)
P_rated, A, ρ0 = 100, 600e3, 0.95
η_th = pd.read_csv(dd/'brisbane_tower_eta.csv')
S = [i*Δρ for i in range(351)]


loss = np.zeros((365,len(S)))
loss_by_year = np.zeros((N_years, 365, len(S)))   # realised loss in each year, for Monte Carlo
for doy in tqdm(range(365)):

    date = datetime(2026, 1, 1) + timedelta(days=doy)

    mask = (    (df['Timestamp'].dt.day == date.day) & 
                (df['Timestamp'].dt.month == date.month))

    dni = df.loc[mask, 'ALLSKY_SFC_SW_DNI'].values            # year 1 h0..h23, year 2 h0..h23, ...
    eff = np.tile(η_th.loc[η_th['day_of_year'] == doy+1, 'eta'].values, N_years)   # repeat 24 hour pattern
    assert len(dni) == len(eff) == 24 * N_years

    P_perfect = eff * A * dni / 1e6                           # MW per unit reflectance
    P_clean = np.minimum(P_perfect * ρ0, P_rated)

    for jj,s in enumerate(S):
        P = np.minimum(P_perfect*(ρ0-s), P_rated)
        lost_energy = (P_clean - P).reshape(N_years, 24).sum(axis=1)   # MWh lost on this day, each year
        loss_by_year[:, doy, jj] = price*lost_energy
        loss[doy, jj] = loss_by_year[:, doy, jj].mean()


# %% Transition probabilities
N = len(S)

# P(loss increases by k states in a day): same for every day and every starting state
inc = np.array([loss_dist.cdf((k+0.5)*Δρ) - loss_dist.cdf((k-0.5)*Δρ) for k in range(N)])

# P(post-rain state = j): same for every day and every starting state
rain = np.array([rain_recovery.cdf(sj+Δρ/2) - rain_recovery.cdf(sj-Δρ/2) for sj in S])
rain[-1] += 1 - rain.sum()                      # lump tail into last state

P_soil = np.zeros((N, N))                       # [from i, to j]
for i in range(N):
    for j in range(i, N):
        P_soil[i, j] = inc[j-i]
    P_soil[i, -1] += 1 - P_soil[i].sum()        # lump out-of-range into last state

P_rain = np.tile(rain, (N, 1))                  # every row identical
P_wash = np.zeros((N, N)); P_wash[:, 0] = 1     # cleaning: everything goes to state 0

transition = []
for d in range(365):
    P_none = pr(d+1)*P_rain + (1-pr(d+1))*P_soil
    transition.append(np.stack([P_none, P_wash]))   # transition[d][a] is (N, N)

# %% Backward induction
C_cl,K = 30_000, 2*365

# stage costs: one (S, A) table per day of year
costs = []
for d in range(365):
    c = np.zeros((N, 2))
    c[:, 0] = loss[d]              # do nothing: lose revenue to soiling
    c[:, 1] = loss[d] + C_cl       # clean tonight: same day's loss plus the wash
    costs.append(c)

# use the MDP code available in maintkit. the periodic function repeats the transition outside of the
# horizon. Necessary because K = 365*2.
model = mdp.MDP(TimeIndexed.periodic(transition), TimeIndexed.periodic(costs),
                states=S, actions=["nothing", "clean"])
sol = model.solve(horizon=K)       # undiscounted, terminal cost 0, starting 1 Jan

policy = sol.policy.window(0, K)   # (K, N) array of actions
threshold = np.array([S[np.argmax(policy[k])] if policy[k].any() else np.nan
                      for k in range(K)])

from matplotlib.colors import ListedColormap

white_gray_cmap = ListedColormap(['white', '#7F7F7F'])
fig, ax = plt.subplots()
im = ax.imshow(policy.T, origin='lower', cmap=white_gray_cmap, aspect='auto',
               vmin=-0.5, vmax=1.5,                                # centre the two colours on 0 and 1
               extent=[0.5, K + 0.5,                               # days 1..K at pixel centres
                       -Δρ/2*100, (S[-1] + Δρ/2)*100])             # reflectance loss in %, pixel edges
cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
cbar.set_ticklabels(['Do nothing', 'Clean'])
ax.set_xlabel('Days into simulation')
ax.set_ylabel('Reflectance loss (%)')
ax.axvline(365.5, ls='--', color='red')


# %% Monte Carlo simulation
def monte_carlo(rules, n_runs=300, horizon=365, seed=0):
    """Simulate a policy from clean mirrors on 1 Jan.

    rules[k] is an (N,) array of actions on day k. Each day's DNI profile is
    drawn from a random year. Returns the cost and number of cleans of each run.
    """
    rng = np.random.default_rng(seed)
    rules = TimeIndexed.finite(rules)
    paths = model.induced_chain(rules).simulate(S[0], horizon, n_paths=n_runs, rng=rng)
    years = rng.integers(N_years, size=(n_runs, horizon))

    cost = np.zeros(n_runs)
    cleans = np.zeros(n_runs)
    for r in range(n_runs):
        for k in range(horizon):
            i = paths[r, k]                       # state on day k
            a = rules[k][i]                       # action the policy takes
            cost[r] += loss_by_year[years[r, k], k % 365, i] + C_cl*a
            cleans[r] += a
    return cost, cleans


def fixed_interval(T):
    """Clean every T days regardless of condition."""
    rules = np.zeros((K, N), dtype=int)
    for k in range(K):
        if (k + 1) % T == 0: # periodic clean for when k is a multiple of the period T
            rules[k, :] = 1
    return rules


# %% Optimise the fixed cleaning interval
# The same seed for every T gives each interval the same random weather, so the
# comparison between intervals is not swamped by Monte Carlo noise.
intervals = list(range(5, 81))
fixed_mean = []
for T in tqdm(intervals,desc="Grid search on fixed cleaning period parameter T"):
    cost, _ = monte_carlo(fixed_interval(T))
    fixed_mean.append(cost.mean())
T_best = intervals[int(np.argmin(fixed_mean))]


# %% Compare condition-based and fixed-interval cleaning
cbm_cost, cbm_cleans = monte_carlo(policy)
fix_cost, fix_cleans = monte_carlo(fixed_interval(T_best))

for name, cost, cleans in [("CBM", cbm_cost, cbm_cleans),
                            (f"Fixed, T={T_best} d", fix_cost, fix_cleans)]:
    half_width = 1.96*cost.std(ddof=1)/np.sqrt(len(cost))
    print(f"{name:>16}: ${cost.mean()/1e6:.3f}M/yr +/- {half_width/1e6:.3f}M (95% CI), "
          f"sd ${cost.std(ddof=1)/1e6:.3f}M, {cleans.mean():.1f} cleans/yr")
print(f"Fixed-interval cleaning costs {100*(fix_cost.mean()/cbm_cost.mean() - 1):.1f}% more")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(intervals, np.array(fixed_mean)/1e6, label='Fixed interval')
ax1.axhline(cbm_cost.mean()/1e6, color='k', ls='--', label='CBM')
ax1.set_xlabel('Cleaning interval T (days)')
ax1.set_ylabel('Mean annual cost (M$)')
ax1.legend()
ax2.hist(cbm_cost/1e6, bins=30, alpha=0.6, label='CBM')
ax2.hist(fix_cost/1e6, bins=30, alpha=0.6, label=f'Fixed, T={T_best} d')
ax2.set_xlabel('Annual cost (M$)')
ax2.set_ylabel('Number of runs')
ax2.legend()
fig.tight_layout()
