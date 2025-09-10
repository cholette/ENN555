# %% Exercise 1: Loading and plotting
import pandas as pd
from datetime import date 

# create pandas dataframe
file_name = "data/POWER_Point_Hourly_20150101_20241231_026d74S_151d45E_UTC.csv"
df = pd.read_csv(file_name,header=17)

# %% Histogram of the wind speed
# I will directly use Matplotlib for illustrative purposes, but this can 
# also be done using the Pandas interface to Matplotlib as above. 
import matplotlib.pyplot as plt

fig,ax = plt.subplots() # I always create the axis first and add plots
ws = df[['WS10M','WS50M']].values
ax.hist(ws[:,0],bins=100,density=True,label='10M')
ax.hist(ws[:,1],bins=100,density=True,label='50M')
ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('density')
ax.legend()

# %% Irradiance plotting

# convert to datetime objects using pandas
df = df.rename(columns={'MO':'MONTH','DY':'DAY','HR':'HOUR'})
df['utc'] = pd.to_datetime(df[['YEAR','MONTH','DAY','HOUR']]).dt.tz_localize('UTC') # NASA POWER | DAV provides UTC time
df['local'] = df['utc'].dt.tz_convert('Australia/Brisbane') # localize to UTC+10

# plot GHI & Clear sky GHI
ax = df.plot(x='local',y=['CLRSKY_SFC_SW_DWN']) # axis output is a matplotlib axis
df.plot(x='local',y='ALLSKY_SFC_SW_DWN',ax=ax)
ax.set_xlabel('Local time')
ax.set_ylabel(r'Irradiance [$W/m^2]$')

# GHI & Clear sky GHI for a day
day_of_interest = date(2024,12,25) # play with this to see different types of days
day_df = df[df['local'].dt.date==day_of_interest]
axd = day_df.plot(x='local',y='CLRSKY_SFC_SW_DWN')
day_df.plot(x='local',y='ALLSKY_SFC_SW_DWN',ax=axd) # axis output is a matplotlib axis
axd.set_xlabel('Local time')
axd.set_ylabel(r'Irradiance [$W/m^2]$')

# Plotting DNI
df.plot(x='local',y='ALLSKY_SFC_SW_DNI')
ax.set_xlabel('Local time')
ax.set_ylabel(r'Irradiance [$W/m^2]$')

# DNI for a day
day_df.plot(x='local',y='ALLSKY_SFC_SW_DNI',ax=axd)

# %% Get clear sky DNI from PVLib
# The clear sky DNI is not included in the NASA POWER | DAV.  The below
# code shows how to get the clear sky values from PVLib. I will skip this
# in class, but this is here for the interested student.

# pvlib (briefly discuss what PVLib is)
from pvlib.location import Location
site = Location(latitude=-26.743,longitude=151.45,tz='Australia/Brisbane') # lon + east of prime meridian
times = pd.DatetimeIndex(df['local'])
cs = site.get_clearsky(times, model='ineichen') # columns: ghi, dni, dhi

# compare Clearsky GHI (Do at home)
axc = df.plot(x='local',y='CLRSKY_SFC_SW_DWN')
cs.plot(y='ghi',label='pvlib clearsky GHI',ax=axc)
axc.set_xlabel('Local time')
axc.set_ylabel(r'Irradiance [$W/m^2]$')

# plot DNI and clear sky DNI
axd = df.plot(x='local',y=['ALLSKY_SFC_SW_DNI']) # axis output is a matplotlib axis
cs.plot(y='dni',label='pvlib clearsky DNI',ax=axd)
ax.set_xlabel('Local time')
ax.set_ylabel('DNI')

