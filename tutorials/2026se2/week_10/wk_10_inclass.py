# %%
# Note: I had to change the file location from the class demo to work in the repo.
# They now live in the subfolder data/disptaching_tutorials/ which lives in the 
# main repo folder. You'll need to create this folder yourself and place the data
# in it for this scrip to work.


# I'll use this data_dir utility to locate the data/ folder
# in your scripts. 
from enn555.paths import data_dir
data_path = data_dir()/'dispatching_tutorials' 


# %%
import pandas as pd
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt

lat,lon = -38.4, 147.4
now = datetime(2024,10,5,0,0,tzinfo=timezone(timedelta(hours=10)))
end = now + timedelta(hours=5*24)

df_pmax = pd.read_csv(data_path/'sam_wind_farm_production.csv') # Updated data path after class
df_pmax['Time stamp'] = pd.to_datetime("2024 "+df_pmax["Time stamp"],format="%Y %b %d, %I:%M %p")
df_pmax['Time stamp'] = df_pmax["Time stamp"].dt.tz_localize("+1000")
df_pmax.set_index('Time stamp',inplace=True)
df_pmax = df_pmax[(df_pmax.index<=end) & (df_pmax.index>=now)]
df_pmax_5min = df_pmax.resample('5min').interpolate('linear')


# %% import prices
df_prices = pd.read_csv(data_path/'price_data_2026se2.csv')
df_prices.timestamp = pd.to_datetime(df_prices.timestamp)
df_prices.set_index('timestamp',inplace=True)

# %% plot prices and available generation
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(df_pmax_5min.index,df_pmax_5min['System power generated | (kW)'],
         color='red')
ax1.set_ylabel('Power generated (kW)',color='red')
ax1.tick_params('y',labelcolor='red')

ax2.plot(df_prices.index,df_prices['RRP (settlement price, $/MWh)'],
         color='green')
ax2.set_ylabel('Prices ($/MWh)',color='green')
ax2.tick_params('y',labelcolor='green')

# %% Set up optimisation
import gurobipy as gp

s0, s_max = 100e3,150e3
Cs, Cg = 0,0
P_dis_max, P_chg_max,P_buy_max = 100e3,100e3,100e3
Δt = 5.0/60.0 # hours
P_ramp_max = 0.05*P_dis_max/Δt
P_max = df_pmax_5min['System power generated | (kW)'].values
prices = df_prices['RRP (settlement price, $/MWh)'].values/1e3 # $/kWh

N = prices.shape[0]

# model in Gurobi
model = gp.Model("Wind with battery dispatching")
Ps = model.addVars(range(N),vtype=gp.GRB.CONTINUOUS,lb=0,
                   ub=P_dis_max,name='Power sold')
Pg = model.addVars(range(N),vtype=gp.GRB.CONTINUOUS,lb=0,
                   ub=P_chg_max,name='Power generated')
Pb = model.addVars(range(N),vtype=gp.GRB.CONTINUOUS,lb=0,
                   ub=P_buy_max,name='Power generated')

S = model.addVars(range(N+1),vtype=gp.GRB.CONTINUOUS,lb=0,ub=s_max,name='Stored energy')

# setting the initials to zero since they are in the past
model.addConstr(S[0]==s0)

# storage constraints
model.addConstrs((S[ii+1] == S[ii] - Ps[ii]*Δt + Pg[ii]*Δt + Pb[ii]*Δt for ii in range(N)))

# power capacity constraints
model.addConstrs((Pg[ii]<=P_max[ii] for ii in range(1,N)))
model.addConstrs((Pg[ii]+Pb[ii]<=P_chg_max for ii in range(1,N)))

# ramping constraint (linearized absolute value)
model.addConstrs(((Ps[ii]-Pb[ii]) - (Ps[ii-1] - Pb[ii-1]) <= P_ramp_max*Δt 
                  for ii in range(1,N))) 
model.addConstrs(((Ps[ii-1]-Pb[ii-1]) - (Ps[ii] - Pb[ii]) <= P_ramp_max*Δt for ii in range(1,N)))

# objective (revenue for now)
model.setObjective(Δt*gp.quicksum((  prices[ii]*(Ps[ii]-Pb[ii]) - Cg*Pg[ii] - Cs*Ps[ii] 
                                   for ii in range(1,N))),sense=gp.GRB.MAXIMIZE)


# %%
model.optimize()

# %% Post processing
import numpy as np
power_sold = []
power_generated = []
power_bought = []
storage = []
revenue = 0.0
for ii in range(N):
    power_sold += [Ps[ii].X]
    power_generated += [Pg[ii].X]
    power_bought += [Pb[ii].X]
    storage += [S[ii].X]
    revenue += prices[ii]*(Ps[ii].X-Pb[ii].X)*Δt

storage += [S[N].X]

power_generated = np.array(power_generated)
power_bought = np.array(power_bought)
power_sold = np.array(power_sold)
storage = np.array(storage)
ramp_rate = np.diff(power_sold-power_bought,prepend=np.nan)/Δt

# %% Plotting
fig,ax = plt.subplots(nrows=4,figsize=(7,8))


time = df_prices.index
time_storage = list(time) # storage time is one longer
time_storage += [time_storage[-1]+pd.Timedelta(Δt,'h')]

ax[0].plot(time,(power_sold-power_bought)/1e3,label='Net export')
ax[0].plot(time,power_generated/1e3,label='Generated')
# ax.plot(time,P_max-power_generated/1e3,label='Curtailed')
ax[0].set_ylabel('Power (MW)')
ax[0].set_xticklabels([]) 
ax[0].legend(loc='best')

ax[1].plot(time_storage,storage/1e3,label='Storage',color='green')
ax[1].axhline(s_max/1000,ls='--',color='red',label='Limits')
ax[1].axhline(0,ls='--',color='red')
ax[1].set_xticklabels([]) 
ax[1].set_ylabel("Stored energy (MWh)")
ax[1].legend(loc='best')

ax[2].plot(time,prices*1e3,label='Prices')
ax[2].set_xticklabels([]) 
ax[2].set_ylabel("Prices ($/MWh)")

ax[3].plot(time,ramp_rate/1000.0/60.0,label='Ramp rate')
ax[3].set_ylabel("Ramp rate (Net MW sold/min)")
ax[3].axhline(P_ramp_max/1000/60.0,ls='--',color='red',label='Limits')
ax[3].axhline(-P_ramp_max/1000/60.0,ls='--',color='red')
ax[3].legend(loc='best')

fig.tight_layout()
ax[0].set_title(f'Objective: {model.objVal:.3e} AUD, Revenue: {revenue:.3e} AUD')

# %%
