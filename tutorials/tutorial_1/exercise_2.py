# %% Exercise 2: Monte carlo simulation of wind power
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np

def load_data(file):
    from datetime import date 

    # create pandas dataframe
    df = pd.read_csv(file,header=17)

    # convert to datetime objects using pandas
    df = df.rename(columns={'MO':'MONTH','DY':'DAY','HR':'HOUR'})
    df['utc'] = pd.to_datetime(df[['YEAR','MONTH','DAY','HOUR']]).dt.tz_localize('UTC') # NASA POWER | DAV provides UTC time
    df['local'] = df['utc'].dt.tz_convert('Australia/Brisbane') # localize to UTC+10

    return df

file_name = "data/POWER_Point_Hourly_20150101_20241231_026d74S_151d45E_UTC.csv"
df = load_data(file_name)

# %% a and b: add wind speed at 100m and fit a weibull
α = 1.0/7.0 # change to assess robustness
df['WS100M'] = df['WS50M'] * (100.0/50.0)**α

fig,ax = plt.subplots() # I always create the axis first and add plots
ws = df[['WS10M','WS50M','WS100M']].values
ax.hist(ws[:,0],bins=100,density=True,label='10M')
ax.hist(ws[:,1],bins=100,density=True,label='50M')
ax.hist(ws[:,2],bins=100,density=True,label='100M')
ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('density')

# scipy stats weibull fitting is strange
k,_,c = sps.weibull_min.fit(ws[:,2],floc=False) # MLE is the default
ws_dist = sps.weibull_min(k,scale=c)

# plot fit
x = np.linspace(ws[:,2].min(),ws[:,2].max(),1000)
ax.plot(x,ws_dist.pdf(x),linewidth=3,color='red',label='Weibull Fit')
ax.legend()

# %% d-f
v_in,v_rated,v_out = 1.0,10.0,25
P_rated = 1600 # kW
def power(v):
    if v<=v_in:
        return 0
    elif v<=v_rated:
        return P_rated * ((v-v_in)/(v_rated-v_in))**3
    elif v<=v_out:
        return P_rated
    else:
        return 0

fig,ax = plt.subplots()
ax.plot(x,[power(w) for w in x])
ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Power (kW)')
ax.set_title('Simplified Power Curve')

N = 1000 # number of MC samples
hourly_ws = ws_dist.rvs(size=(N,24*365))
hourly_power = np.zeros_like(hourly_ws)
for ii in range(N):
    for jj in range(24*365):
        hourly_power[ii,jj] = power(hourly_ws[ii,jj])

aep = hourly_power.sum(axis=1) # I can just sum because timesteps are hours
fig,ax = plt.subplots()
ax.hist(aep,density=True,bins=25)
ax.set_title('Annual Energy Production')
ax.set_xlabel('AEP (kWh)')
ax.set_ylabel('Density')

print(f'P50: {np.percentile(aep,50)/1e3:.3e} MWh')
print(f'P90: {np.percentile(aep,10)/1e3:.3e} MWh')

