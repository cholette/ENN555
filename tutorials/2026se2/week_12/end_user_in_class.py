# %%
import pandas as pd
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import scipy.stats as sps
from enn555.paths import data_dir
data_path = data_dir()/'dispatching_tutorials'

# %% Define parameters (later)

# %% Demand model loading and sampling
T = 3*48
df = pd.read_csv(data_path/'par_demand_model.csv')
times = pd.date_range(start='2026-10-13 00:00:00',
                          periods=T,
                          freq='0.5h')
times_storage = list(times)
times_storage += [times_storage[-1]+pd.Timedelta(0.5,'h')]

dist_D0 = sps.norm(loc=40,scale=np.sqrt(25))
def sample_par(M=1):
    T = len(times)
    D = np.zeros((M,T))
    for m in range(M):
        D_current = dist_D0.rvs()
        for ii,t in enumerate(times):
            hod = t.time().hour + t.time().minute/60.0 # fractional hours
            row_mask = (df['Hours']==hod)
            params = df.loc[row_mask,:].values[0]
            α,d,σ2 = params[1],params[2],params[3]

            ɛ = np.random.normal(loc=0,scale=np.sqrt(σ2)) # samples noise
            D_current = α*D_current + d + ɛ
            D[m,ii] = D_current
    
    return D

demand_sample = sample_par(M=1)

fig,ax = plt.subplots()
ax.plot(times,demand_sample.T)
ax.set_xlabel('Date/time')
ax.set_ylabel('Demand (kW)')

# %% Prices
price = 0.25*np.ones(T)
mask = (times.hour<22) & (times.hour>=17)
price[mask] = 1.0

fig,ax = plt.subplots()
ax.plot(times,price,'.')
ax.set_xlabel('Date/time')
ax.set_ylabel('Price ($/kWh)')

# %% Gurobi time!!!!!!!
D = demand_sample[0,:]
cont = gp.GRB.CONTINUOUS
P_chg_max,P_dis_max = 50,50 # kW
s_max,s0 = 200, 100 # kWh
η_c,η_d = 1.0,1.0
Δt = 0.5 # hr
cd = 0.1


def create_model(demand):

    model = gp.Model()
    P_chg = model.addVars(range(T),vtype=cont,lb=0,ub=P_chg_max,name='P_chg')
    P_dis = model.addVars(range(T),vtype=cont,lb=0,ub=P_dis_max,name='P_dis')
    S =model.addVars(range(T+1),vtype=cont,lb=0,ub=s_max)
    P_grid = model.addVars(range(T),vtype=cont,lb=0,name='P_grid')
    P_curt = model.addVars(range(T),vtype=cont,lb=0,name='P_curt')

    model.addConstr(S[0]==s0)
    model.addConstr(S[T]>=s0)
    model.addConstrs( (S[ii+1] == S[ii] + (η_c*P_chg[ii] - 1/η_d * P_dis[ii])*Δt for ii in range(T)) )
    model.addConstrs( ( (demand[ii]== P_dis[ii] - P_chg[ii] + P_grid[ii] - P_curt[ii]) for ii in range(T) ) )

    model.setObjective(Δt*gp.quicksum( (price[ii]*P_grid[ii] + cd*(P_chg[ii]+P_dis[ii]) for ii in range(T) ) ),sense=gp.GRB.MINIMIZE )

    return model,P_chg, P_dis, P_grid, P_curt, S

model,P_chg, P_dis, P_grid, P_curt, S = create_model(D)
model.optimize()

# %% Extract variables
p_chg = np.array([P_chg[ii].X for ii in range(T)])
p_dis = np.array([P_dis[ii].X for ii in range(T)])
p_grid = np.array([P_grid[ii].X for ii in range(T)])
p_curt = np.array([P_curt[ii].X for ii in range(T)])
soc = np.array([S[ii].X for ii in range(T+1)])

# %% Plots
fig2,ax2 = plt.subplots(nrows=4,sharex=True,figsize=(10,10))
ax2[0].plot(times,p_dis - p_chg)
ax2[0].set_ylabel('P_dis - P_chg')

ax2[1].plot(times,p_grid)
ax2[1].set_ylabel('P_grid')

ax2[2].plot(times_storage,soc)
ax2[2].set_ylabel('SOC ')

ax2[3].plot(times,price)
ax2[3].set_ylabel('Prices')


# %% Simulate on a new demand trajectory
D2 = sample_par()[0]
p_grid2 = np.zeros(T)
p_curt2 = np.zeros(T)
soc2 = np.zeros(T+1)
soc2[0] = s0
for ii in range(T):
    pg = D2[ii]-p_dis[ii]+p_chg[ii]
    if pg >= 0:
        p_grid2[ii] = pg
    else:
        p_grid2[ii]=0
        p_curt2[ii] = -pg

    soc2[ii+1] = soc[ii] + (η_c*p_chg[ii] - 1/η_d * p_dis[ii])*Δt

# %%
fig3,ax3 = plt.subplots(nrows=5,sharex=True,figsize=(10,10))
ax3[0].plot(times,p_dis - p_chg)
ax3[0].set_ylabel('P_dis - P_chg')

ax3[1].plot(times,p_grid2)
ax3[1].set_ylabel('P_grid2')

ax3[2].plot(times_storage,soc2)
ax3[2].set_ylabel('SOC2 ')

ax3[3].plot(times,price)
ax3[3].set_ylabel('Prices')

ax3[4].plot(times,p_curt2)
ax3[4].set_ylabel('P_curt')
# %%
fig4,ax4 = plt.subplots()
ax4.plot(times,D,times,D2)