# %% End user with on-site renewables and a battery under Time-of-use charges
import pandas as pd
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import scipy.stats as sps
from enn555.paths import data_dir
data_path = data_dir()/'dispatching_tutorials' 

# Optimization parameters
Δt = 0.5 # hours
P_dis_max, P_chg_max = 50,50
s_max = 200
s0 = 0.5*s_max
cd = 0.01
cr = 0 # start at zero and note that there are spikes. Increase to smooth charging. 
η_c, η_d = 0.95, 0.95 # charge/discharge efficiencies (set both to 1 for the ideal device model)

# %% Demand model loading and sampling
T = 3*48 # This can go up to the end of the month (31*2*4=1488), if so, possibly remove final storage constraint. 
df = pd.read_csv(data_path/'par_demand_model.csv')
times = pd.date_range(start='2025-10-01 00:00:00',
                      periods=T,
                      freq='0.5h',
                      )
times_storage = list(times) # storage time is one longer
times_storage += [times_storage[-1]+pd.Timedelta(Δt,'h')]


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
    # Battery power is split into non-negative charge and discharge powers (see lecture notes)
    P_chg = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,lb=0,ub=P_chg_max,name='P_chg')
    P_dis = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,lb=0,ub=P_dis_max,name='P_dis')
    S = model.addVars(range(T+1),vtype=gp.GRB.CONTINUOUS,lb=0,ub=s_max,name='S')
    P_grid = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,lb=0,name='P_grid')
    z_change = model.addVars(range(T),vtype=gp.GRB.CONTINUOUS,lb=0,name='z_change')

    # initial/final values
    model.addConstr(S[0]==s0)
    # model.addConstr(S[T]>=s0)         # ADD AFTER DISCUSSION with students about "cheating" when storage discharges fully

    # storage (charge/discharge efficiency model) & demand constraints
    model.addConstrs( (S[ii+1] == S[ii] + (η_c*P_chg[ii] - P_dis[ii]/η_d)*Δt for ii in range(T)) )
    model.addConstrs( (demand[ii] == P_dis[ii] - P_chg[ii] + P_grid[ii]) for ii in range(T))

    # Absolute value linearization (ramp penalty on the net battery power P_dis - P_chg)
    # Note: no linearization is needed for the degradation cost cd*(P_chg + P_dis)
    model.addConstrs((z_change[ii] >=  (P_dis[ii]-P_chg[ii]) - (P_dis[ii-1]-P_chg[ii-1]) for ii in range (1,T)))
    model.addConstrs((z_change[ii] >= -(P_dis[ii]-P_chg[ii]) + (P_dis[ii-1]-P_chg[ii-1]) for ii in range (1,T)))

    # objective
    model.setObjective(Δt*gp.quicksum((price[ii]*P_grid[ii] + cd*(P_chg[ii] + P_dis[ii]) for ii in range(T)))
                       + Δt*gp.quicksum(cr*z_change[ii] for ii in range(1,T)),sense=gp.GRB.MINIMIZE)
    
    return model,P_chg,P_dis,P_grid,S

def check_simultaneous(p_chg,p_dis,tol=1e-6):
    # Simultaneous charging & discharging should not occur when prices are >= 0 and cd > 0 (see notes)
    both = np.minimum(p_chg,p_dis) > tol
    if np.any(both):
        print(f'WARNING: simultaneous charge/discharge in {both.sum()} intervals')
    else:
        print('No simultaneous charge/discharge')

# %% optimize & extract
model,P_chg,P_dis,P_grid,S = create_model(D)
model.optimize()

p_chg = np.array([P_chg[ii].X for ii in range(T)])
p_dis = np.array([P_dis[ii].X for ii in range(T)])
p_batt = p_dis - p_chg   # net battery power (positive = discharging)
p_grid = [P_grid[ii].X for ii in range(T)]
soc = [S[ii].X/s_max*100 for ii in range(T+1)]
check_simultaneous(p_chg,p_dis)


# %% plot optimal battery charge/discharge, state of charge, and grid demand
fig2,ax2 = plt.subplots(nrows=4,sharex=True)
ax2[0].plot(times,p_batt)
ax2[0].set_ylabel('P_dis - P_chg')

ax2[1].plot(times,p_grid)
ax2[1].set_ylabel('P_grid')

ax2[2].plot(times_storage,soc)
ax2[2].set_xticklabels(ax2[2].get_xticklabels(), rotation=30, ha='right')
ax2[2].set_ylabel('SOC (%)')

ax2[3].plot(times,price)
ax2[3].set_xlabel('Date/time')
ax2[3].set_ylabel('Price ($)')

# %% sample another trajectory & simulate SAME battery schedule (p_chg, p_dis)
D2 = sample_par()[0]

p_grid2 = np.zeros(T)
soc2 = np.zeros(T+1)
soc2[0] = s0
for ii in range(T):
    p_grid2[ii] = max([0,D2[ii] - (p_dis[ii] - p_chg[ii])]) # needed to ensure there is no feed-in
    soc2[ii+1] = soc2[ii] + (η_c*p_chg[ii] - p_dis[ii]/η_d)*Δt

soc2 = soc2/s_max*100

model2,P_chg2,P_dis2,P_grid2,S2 = create_model(D2)
model2.optimize()

p_chg2_opt = np.array([P_chg2[ii].X for ii in range(T)])
p_dis2_opt = np.array([P_dis2[ii].X for ii in range(T)])
p_batt2_opt = p_dis2_opt - p_chg2_opt
p_grid2_opt = [P_grid2[ii].X for ii in range(T)]
soc2_opt = [S2[ii].X/s_max*100 for ii in range(T+1)]
check_simultaneous(p_chg2_opt,p_dis2_opt)


# %% plot optimal battery charge/discharge, state of charge, and grid demand
fig2,ax2 = plt.subplots(nrows=4,sharex=True)
ax2[0].plot(times,p_batt,label='optimal')
ax2[0].plot(times,p_batt2_opt,label='different traj.')
ax2[0].set_ylabel('P_dis - P_chg')

ax2[1].plot(times,p_grid2,times,p_grid2_opt)
ax2[1].set_ylabel('P_grid')

ax2[2].plot(times_storage,soc2,times_storage,soc2_opt)
ax2[2].set_xticklabels(ax2[2].get_xticklabels(), rotation=30, ha='right')
ax2[2].set_ylabel('SOC (%)')
ax2[2].axhline(y=100,ls='--',color='red')
ax2[2].axhline(y=0,ls='--',color='red')

ax2[3].plot(times,price)
ax2[3].set_xlabel('Date/time')
ax2[3].set_ylabel('Price ($)')

ax2[0].legend()

print(f'Total (novel trajectory, previous battery schedule): {Δt*sum(price*p_grid2 + cd*(p_chg + p_dis)):.2f}')
if np.any(soc2 < -1e-6) or np.any(soc2 > 100+1e-6):
    print('SOC NOT FEASIBLE')

print(f'Total (perfect knowledge, optimal battery schedule): {Δt*sum(price*np.array(p_grid2_opt) + cd*(p_chg2_opt + p_dis2_opt)):.2f}')

# %%
