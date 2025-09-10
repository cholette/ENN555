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

# %% 
λ = 5.0/8760.0 # failures per hour
MTTR = 24.0 # hours
var = 20.0 # play with this to see how it effects results
N_inv, N_sims = 3, 5000
A = 3000
η_sys = 0.18
GHI = df[df['local'].dt.year == 2023]['CLRSKY_SFC_SW_DWN'].values

# reliability distribution 
failure_dist = sps.expon(scale=1.0/λ)

# maintainability distributions (distribution of DTs)
s = np.sqrt(np.log(1+var/MTTR**2))
μ = np.log(MTTR**2/np.sqrt(MTTR**2+var)) # see Definitions section of https://en.wikipedia.org/wiki/Log-normal_distribution
repair_dist = sps.norm(loc=μ,scale=s) # draws from this need to be exponetially transformed to produce simulated repair times

# simluate failure and repair times
times = np.arange(0,8760)
inverter_down = np.zeros((N_inv,len(times),N_sims))
for n in range(N_sims):
    t = 0
    while t < times[-1]:
        for ii in range(N_inv):
            dt_fail = failure_dist.rvs()
            t += dt_fail

            if t >= times[-1]:
                break

            # downtime
            dt_repair = np.exp(repair_dist.rvs())
            dn_start = np.where(t<=times)[0][0]
            t_repair = np.min([t+dt_repair,times[-1]])
            dn_end = np.where(t_repair<=times)[0][0]
            
            inverter_down[ii,dn_start:dn_end,n] = 1

number_online = N_inv - inverter_down.sum(axis=0)

P_base = η_sys * GHI * A
P = P_base[:,np.newaxis] * number_online/N_inv          # tricky broadcasting!
E_lost = (P_base.sum() - P.sum(axis=0))/1e6             # MWh lost

fig,ax = plt.subplots()
ax.hist(E_lost,density=True)
ax.set_title(f'Lost energy (Max={P_base.sum()/1e6:.1f} MWh)')
ax.set_xlabel('Lost energy (MWh)')
ax.set_ylabel('Density')



# Of course, the above is a little silly because this should be in operating 
# hours to avoid having failures at night (that have no energy consequence).
