# %% Exercise 1 (in class)
import pandas as pd
import matplotlib.pyplot as plt

file = 'data/POWER_Point_Hourly_20150101_20241231_026d74S_151d45E_UTC.csv'
df = pd.read_csv(file,header=17)

# fig,ax = plt.subplots()
# ws = df[['WS10M','WS50M']].values
# ax.hist(ws[:,0],label='WS10M',bins=100,density=True)
# ax.hist(ws[:,1],label='WS50M',bins=100,density=True)
# ax.legend()
# ax.set_xlabel('Wind speed (m/s)')
# ax.set_ylabel('density')

ax = df.plot(kind='hist',y=['WS10M','WS50M'],bins=100)
ax.legend()
ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('density')


# %% Irradiance plotting

# Datetime fun :(
df = df.rename(columns={'MO':'MONTH','DY':'DAY','HR':'HOUR'})
time_df = df[['YEAR','MONTH','DAY','HOUR']]
df['utc'] = pd.to_datetime(time_df).dt.tz_localize('UTC')
df['local'] = df['utc'].dt.tz_convert('Australia/Brisbane')

# plot GHI and clear sky GHI
ax = df.plot(x='local',y=['CLRSKY_SFC_SW_DWN','ALLSKY_SFC_SW_DWN','ALLSKY_SFC_SW_DNI'])
ax.set_ylabel('Irradiance')
ax.set_xlabel('Local time')

# now for a day of interest
from datetime import date
doi = date(2024,12,25)
day_df = df[df['local'].dt.date == doi]
axd = day_df.plot(x='local',y=['CLRSKY_SFC_SW_DWN','ALLSKY_SFC_SW_DWN','ALLSKY_SFC_SW_DNI'])
axd.set_ylabel(f'Irradiance [$W/m^2$]')
axd.set_xlabel('Local time')















