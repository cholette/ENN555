# %% Exercise 2
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np

def load_data(file):
    df = pd.read_csv(file,header=17)
    df = df.rename(columns={'MO':'MONTH','DY':'DAY','HR':'HOUR'})
    time_df = df[['YEAR','MONTH','DAY','HOUR']]
    df['utc'] = pd.to_datetime(time_df).dt.tz_localize('UTC')
    df['local'] = df['utc'].dt.tz_convert('Australia/Brisbane')

    return df

file = 'data/POWER_Point_Hourly_20150101_20241231_026d74S_151d45E_UTC.csv'
df = load_data(file)

# 
α = 1/7.0
df['WS100M'] = df['WS50M'] * (100/50)**α # power law
ax = df.plot(kind='hist',y=['WS10M','WS50M','WS100M'],bins=100,density=True)

k,_,c = sps.weibull_min.fit(df['WS100M'].values,floc=False)
ws_dist = sps.weibull_min(k,scale=c)
w = np.linspace(df['WS100M'].values.min(),df['WS100M'].values.max(),1000)
ax.plot(w,ws_dist.pdf(w),linewidth=3,label='Fit')
ax.legend()

# %% d-f
v_in,v_rated,v_out = 2.0,9.0,25
P_rated = 1600 # kW
def power(v):
    if v <= v_in:
        return 0
    elif v<= v_rated:
        return P_rated *((v-v_in)/(v_out-v_in))**3
    elif v<=v_out:
        return P_rated
    else:
        return 0
    
N = 5000
hourly_ws = ws_dist.rvs((N,24*365))
hourly_power = np.zeros_like(hourly_ws)
for ii in range(N):
    for jj in range(24*365):
        hourly_power[ii,jj] = power(hourly_ws[ii,jj])


aep = hourly_power.sum(axis=1) # timesteps in hours
fig,ax = plt.subplots()
ax.hist(aep/1000.0,density=True,bins=100)
ax.set_xlabel('AEP (MWh)')

print(f'P50:{np.percentile(aep,50)/1000:.2f}')
print(f'P90:{np.percentile(aep,10)/1000:.2f}')















