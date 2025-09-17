# %% Dispatching in class 2025
import pandas as pd
import numpy as np
import gurobipy as gp
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt

now = datetime(2024,10,5,0,0,tzinfo=ZoneInfo('Australia/Melbourne'))
end = now + timedelta(hours=5*24)

# %% Load in Wind data 
dfw = pd.read_csv('data/sam_export_2024.csv',header=1)
dfw['utc'] = pd.to_datetime(dfw[['Year','Month','Day','Hour']]).dt.tz_localize('UTC')
dfw['local'] = dfw['utc'].dt.tz_convert('Australia/Melbourne')
dfw.set_index('local',inplace=True)

# %% Load in price data
df_prices = pd.read_csv('data/price_data.csv',parse_dates=['timestamp'])
df_prices.set_index('timestamp',inplace=True)

# %% apply time period filter
dfw = dfw[(dfw.index<=end) & (dfw.index>=now)]
now = max([dfw.index[0],now])
end = min([dfw.index[-1],end])
df_prices = df_prices[(df_prices.index>=now) & (df_prices.index<=end)]


# %% plot windspeed / prices
fig1,ax1 = plt.subplots()
ln1 = ax1.plot(dfw.index,dfw['Wind speed at 100m'],label='Wind speed (m/s)')
ax1.set_xlabel('Date/time')
ax1.set_ylabel('Wind speed')

ax2 = ax1.twinx()
ax2.plot(df_prices.index,df_prices['Settlement Price'],color='red')
ax2.set_xlabel('Date/time')
ax2.set_ylabel('Price (AUD/MWh)',color='red')
ax2.tick_params(axis='y',labelcolor='red')

# %% plot farm max power / prices
df_pmax = pd.read_csv('data/wind_farm_production.csv')
df_pmax['Time stamp'] = pd.to_datetime('2024 '+df_pmax['Time stamp'],format="%Y %b %d, %I:%M %p")
df_pmax['Time stamp'] = df_pmax['Time stamp'].dt.tz_localize('UTC')
df_pmax['Time stamp'] = df_pmax['Time stamp'].dt.tz_convert('Australia/Melbourne')
df_pmax.set_index('Time stamp',inplace=True)
df_pmax = df_pmax[(df_pmax.index<=end) & (df_pmax.index>=now)]
df_pmax_5min = df_pmax.resample("5T").interpolate("linear")


fig1,ax1 = plt.subplots()
ln1 = ax1.plot(df_pmax_5min.index,df_pmax_5min['System power generated | (kW)'],label='Power (kW)')
ax1.set_xlabel('Date/time')
ax1.set_ylabel('Wind farm power (kW)')

ax2 = ax1.twinx()
ax2.plot(df_prices.index,df_prices['Settlement Price'],color='red')
ax2.set_xlabel('Date/time')
ax2.set_ylabel('Price (AUD/MWh)',color='red')
ax2.tick_params(axis='y',labelcolor='red')


# %% Optimization in Gurobi
s0,s_max = 100e3,150e3 # kWh
Cs,Cg = 0,0 # no degradation costs for now
P_dis_max,P_chg_max,P_buy_max = 100e3,100e3,100e3
Δt = 5.0/60.0
P_ramp_max = 0.2*P_dis_max/Δt # kW/hr
P_max = df_pmax_5min['System power generated | (kW)'].values
prices = df_prices['Settlement Price'].values/1e3 # AUD/kWh
N = df_prices.shape[0]


# gurobi setup
model = gp.Model('Wind with battery dispatching')
Ps = model.addVars(range(N),vtype=gp.GRB.CONTINUOUS,lb=0,ub=P_dis_max,name='Power to grid from battery')
Pg = model.addVars(range(N),vtype=gp.GRB.CONTINUOUS,lb=0,ub=P_chg_max,name='Power to battery from wind farm')
Pb = model.addVars(range(N),vtype=gp.GRB.CONTINUOUS,lb=0,ub=P_buy_max,name='Power from grid to battery')
S = model.addVars(range(N),vtype=gp.GRB.CONTINUOUS,lb=0,ub=s_max,name='Energy in battery (kWh)')

# initial values
model.addConstr(S[0]==s0)
model.addConstr(Ps[0]==0) # don't count profit/loss from the zeroth epoch
model.addConstr(Pg[0]==0)
model.addConstr(Pb[0]==0)

# storage constraints
model.addConstrs( (S[ii]==S[ii-1] + Δt*(Pg[ii]+Pb[ii]-Ps[ii]) for ii in range(1,N)) )

# power capacity
model.addConstrs((Pg[ii]<=P_max[ii] for ii in range (1,N)))
model.addConstrs((Pg[ii]+Pb[ii]<=P_chg_max for ii in range(1,N)))

# ramping constraint (linearized)
model.addConstrs(( (Ps[ii]-Pb[ii])-(Ps[ii-1]-Pb[ii-1])<=P_ramp_max*Δt  for ii in range(1,N)))
model.addConstrs(( (Ps[ii-1]-Pb[ii-1])-(Ps[ii]-Pb[ii])<=P_ramp_max*Δt  for ii in range(1,N)))

# objective
model.setObjective(Δt*gp.quicksum((  prices[ii]*(Ps[ii]-Pb[ii]) - Cg*Pg[ii] - Cs*Ps[ii] for ii in range(1,N))),sense=gp.GRB.MAXIMIZE)

model.optimize()

# %% look at solution

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

power_generated = np.array(power_generated)
power_bought = np.array(power_bought)
power_sold = np.array(power_sold)
storage = np.array(storage)
ramp_rate = np.diff(power_sold-power_bought,prepend=np.nan)/Δt

# plotting
fig,ax = plt.subplots(nrows=4,figsize=(7,8))
time = df_prices.index
ax[0].plot(time,(power_sold-power_bought)/1e3,label='Net export')
ax[0].plot(time,power_generated/1e3,label='Generated')
# ax.plot(time,P_max-power_generated/1e3,label='Curtailed')
ax[0].set_ylabel('Power (MW)')
ax[0].set_xticklabels([]) 
ax[0].legend(loc='best')

ax[1].plot(time,storage/1e3,label='Storage',color='green')
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
