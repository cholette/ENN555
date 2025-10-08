# %% End user with on-site renewables and a battery under Time-of-use charges
import pandas as pd
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import scipy.stats as sps

# Optimization parameters
Δt = 0.5 # hours
P_dis_max, P_chg_max = 50,50
s_max = 200
s0 = 0.5*s_max
cd = 0.01
cr = 0 # start at zero and note that there are spikes. Increase to smooth charging. 

# %% Demand model loading and sampling
T = 1488 # This can go up to the end of the month (31*2*4=1488), if so, possibly remove final storage constraint. 
df = pd.read_csv('../../data/par_demand_model.csv')
times = pd.date_range(start='2025-10-01 00:00:00',
                      periods=T,
                      freq='0.5h',
                      )

dist0 = sps.norm(loc=40,scale=np.sqrt(25))
def sample_par(N=1):
    T = len(times)
    D = np.zeros((N,T))
    for n in range(N):
        D_current = dist0.rvs()
        for ii,t in enumerate(times):
            hod = t.time().hour + t.time().minute/60.0
            row = df.loc[df['Hours']==hod,:].values[0]
            α,d,σ2 = row[1],row[2],row[3]

            ɛ = np.random.normal(loc=0,scale=np.sqrt(σ2))
            D_current = α*D_current + d + ɛ
            D[n,ii] = D_current
    
    return D

demand_sample = sample_par(N=1)

fig,ax = plt.subplots()
ax.plot(times,demand_sample.T)
ax.tick_params(axis='x', labelrotation=30)
ax.set_xlabel('Date/time')
ax.set_ylabel('Demand (kW)')


# %% prices
price = 0.25*np.ones(T)
for ii,_ in enumerate(price):
    if times[ii].hour > 17 and times[ii].hour<22:
        price[ii] = 1.0

# %% Gurobi
D = demand_sample[0,:]

def create_model(demand):
    model = gp.Model()
    P_batt = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,
                           lb=-P_chg_max,ub=P_dis_max,name='P_batt')
    S = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,lb=0,ub=s_max,name='S')
    P_grid = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,lb=0,name='P_grid')
    z_deg = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,lb=0,name='z_deg')
    z_change = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,lb=0,name='z_change')

    # initial/final values
    model.addConstr(S[0]==s0)
    # model.addConstr(S[T-1]>=s0)         # ADD AFTER DISCUSSION with students about "cheating" when storage discharges fully
    model.addConstr(P_batt[0]==0)       # don't count profit/loss from the zeroth epoch
    model.addConstr(z_deg[0]==0)

    # storage & demand constraints
    model.addConstrs( (S[ii] == S[ii-1] - P_batt[ii]*Δt for ii in range(1,T)) )
    model.addConstrs( (demand[ii] == P_batt[ii] + P_grid[ii]) for ii in range(1,T))

    # Absolute value linearizations
    model.addConstrs((z_deg[ii] >= P_batt[ii]  for ii in range (1,T)))
    model.addConstrs((z_deg[ii] >= -P_batt[ii] for ii in range (1,T)))
    model.addConstrs((z_change[ii] >=  P_batt[ii]-P_batt[ii-1]  for ii in range (1,T)))
    model.addConstrs((z_change[ii] >= -P_batt[ii]+P_batt[ii-1] for ii in range (1,T)))

    # objective
    model.setObjective(Δt*gp.quicksum((price[ii]*P_grid[ii] + cd*z_deg[ii] + cr*z_change[ii] for ii in range(1,T))),sense=gp.GRB.MINIMIZE)
    
    return model,P_batt,P_grid,S,z_deg

# %% optimize & extract
model,P_batt,P_grid,S,z_deg = create_model(D)
model.optimize()

p_batt = [P_batt[ii].X for ii in range(T)]
p_grid = [P_grid[ii].X for ii in range(T)]
soc = [S[ii].X/s_max*100 for ii in range(T)]


# %% plot optimal battery charge/discharge, state of charge, and grid demand
fig2,ax2 = plt.subplots(nrows=4,sharex=True)
ax2[0].plot(times,p_batt)
ax2[0].set_ylabel('P_batt')

ax2[1].plot(times,p_grid)
ax2[1].set_ylabel('P_grid')

ax2[2].plot(times,soc)
ax2[2].set_xticklabels(ax2[2].get_xticklabels(), rotation=30, ha='right')
ax2[2].set_ylabel('SOC (%)')

ax2[3].plot(times,price)
ax2[3].set_xlabel('Date/time')
ax2[3].set_ylabel('Price ($)')

# %% sample another trajectory & simulate SAME P_batt
D2 = sample_par()[0]

p_grid2 = np.zeros(T)
soc2 = np.zeros(T)
soc2[0] = s0
for ii in range(1,len(D2)):
    p_grid2[ii] = max([0,D2[ii] - p_batt[ii]]) # needed to ensure there is no feed-in
    soc2[ii] = soc2[ii-1] - p_batt[ii]*Δt

soc2 = soc2/s_max*100
    

model2,P_batt2,P_grid2,S2,z_deg2 = create_model(D2)
model2.optimize()

p_batt2_opt = [P_batt2[ii].X for ii in range(T)]
p_grid2_opt = [P_grid2[ii].X for ii in range(T)]
soc2_opt = [S2[ii].X/s_max*100 for ii in range(T)]


# %% plot optimal battery charge/discharge, state of charge, and grid demand
fig2,ax2 = plt.subplots(nrows=4,sharex=True)
ax2[0].plot(times,p_batt,times,p_batt2_opt)
ax2[0].set_ylabel('P_batt')

ax2[1].plot(times,p_grid2,times,p_grid2_opt)
ax2[1].set_ylabel('P_grid')

ax2[2].plot(times,soc2,times,soc2_opt)
ax2[2].set_xticklabels(ax2[2].get_xticklabels(), rotation=30, ha='right')
ax2[2].set_ylabel('SOC (%)')
ax2[2].axhline(y=100,ls='--',color='red')
ax2[2].axhline(y=0,ls='--',color='red')

ax2[3].plot(times,price)
ax2[3].set_xlabel('Date/time')
ax2[3].set_ylabel('Price ($)')

print(f'Total (novel trajectory, previous p_batt): {Δt*sum(price*p_grid2 + cd*np.abs(p_batt)):.2f}')
if np.any(soc2 < -1e-6) or np.any(soc2 > 100+1e-6):
    print('SOC NOT FEASIBLE')

print(f'Total (perfect knowledge, optimal p_batt): {Δt*sum(price*p_grid2_opt+cd*np.abs(p_batt2_opt)):.2f}')

# %%
