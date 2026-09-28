import scipy.stats as stats
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import re
from typing import Any

class merra_wind_speed_data:
    """Container for MERRA-2 reanalysis wind data downloaded from NASA POWER.

    Imports single-point CSV exports from the NASA POWER Data Access Viewer and
    provides methods to extrapolate wind speed to arbitrary hub heights, fit
    wind profile parameters, and export data in SAM-compatible format.

    Typical MERRA-2 columns after import:
        ``WS10M``, ``WS50M`` – wind speed at 10 m and 50 m (m/s)
        ``WD10M``, ``WD50M`` – wind direction at 10 m and 50 m (degrees)
        ``PS``               – surface pressure (kPa in raw data)
        ``T2M``              – air temperature at 2 m (°C)

    Attributes
    ----------
    data_source_url : str
        URL of the NASA POWER data access viewer.
    latitude : float or None
        Site latitude in degrees (positive north).
    longitude : float or None
        Site longitude in degrees (positive east of Greenwich).
    date_range : str or None
        Date range string parsed from the CSV header.
    average_elevation : str or None
        Site elevation string parsed from the CSV header.
    time_reference : {'LST', 'UTC'} or None
        Time reference reported in the NASA POWER header. ``'LST'`` means the
        timestamps are in Local Standard Time (a fixed UTC offset — no DST);
        ``'UTC'`` means timestamps are in Coordinated Universal Time.
    data : pandas.DataFrame or None
        Imported time-series data with a ``Timestamp`` column.
    """

    def __init__(self):
        self.data_source_url = "https://power.larc.nasa.gov/data-access-viewer/"
        self.latitude = None
        self.longitude = None
        self.date_range = None
        self.average_elevation = None
        self.time_reference = None  # 'LST' or 'UTC' as reported in the NASA POWER header
        self.data = None

    def import_data(self,merra_single_point_csv: str):
        """Import a MERRA-2 single-point CSV file exported from NASA POWER.

        Parses the custom NASA POWER header (lines up to and including
        ``-END HEADER-``) to extract site metadata, then reads the remaining
        rows as a time-series DataFrame with a ``Timestamp`` column.

        Parameters
        ----------
        merra_single_point_csv : str or path-like
            Path to the NASA POWER CSV export file.

        Returns
        -------
        None
            Populates ``self.latitude``, ``self.longitude``,
            ``self.average_elevation``, ``self.time_reference``, and
            ``self.data`` in place.
        """
        file = merra_single_point_csv

        # read header
        with open(file,'r') as f:
            reader = csv.reader(f)
            header_count = 0
            for row in reader:
                header_count += 1
                if row and row[0] == '-END HEADER-':
                    break

                if "location" in row[0].lower():
                    r_split = row[0].split()
                    islat = [('latitude' in r.lower()) for r in r_split]
                    if any(islat):
                        self.latitude = float(r_split[islat.index(True)+1])

                    islon = [('longitude' in r.lower()) for r in r_split]
                    if any(islon):
                        self.longitude = float(r_split[islon.index(True)+1])

                if "elevation" in row[0].lower():
                    self.average_elevation = row[0].split('=')[-1]

                # The Dates line ends with "in LST" or "in UTC"
                if "dates" in row[0].lower():
                    tokens = row[0].split()
                    if tokens[-1].upper() in ('LST', 'UTC'):
                        self.time_reference = tokens[-1].upper()

        # read the rest
        df = pd.read_csv(file,skiprows=header_count)
        df['Timestamp'] = pd.to_datetime(dict(
                                year=df['YEAR'],
                                month=df['MO'],
                                day=df['DY'],
                                hour=df['HR'],
                                minute=df.get('minute', 0),  # use 0 if not present
                                second=df.get('second', 0),  # use 0 if not present
                                ))
        
        if self.time_reference == 'UTC':
            print(f"Time reference is UTC.")
            df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC')
        else:
            print(f"""Time reference is {self.time_reference}. Assuming timestamps are in local 
                      time with a fixed UTC offset (no DST). If this is from NASA POWER, this is 
                      likley incorrect since LST in the DAV is Local Solar Time.""")

        df = df.drop(['YEAR','MO','DY','HR'],axis=1)
        # NASA POWER names all shortwave irradiance parameters with 'SW'
        # (e.g. ALLSKY_SFC_SW_DWN, ALLSKY_SFC_SW_DNI, CLRSKY_SFC_SW_DWN)
        solar_cols = [c for c in df.columns if 'SW' in c]
        df = df.drop(solar_cols, axis=1)
        df = df[['Timestamp',*df.columns[:-1]]]
        self.data = df

    @staticmethod
    def _meta_get(meta, *keys):
        """Case-insensitive lookup of the first matching key in a (possibly nested) dict."""
        if not isinstance(meta, dict):
            return None
        lower = {str(k).lower(): v for k, v in meta.items()}
        for k in keys:
            if k in lower and lower[k] is not None:
                return lower[k]
        for v in meta.values():
            if isinstance(v, dict):
                found = merra_wind_speed_data._meta_get(v, *keys)
                if found is not None:
                    return found
        return None

    def import_dataframe(self, df: pd.DataFrame, meta: dict | None = None,
                         time_reference: str | None = None,
                         latitude: float | None = None,
                         longitude: float | None = None,
                         elevation: float | str | None = None):
        """Import NASA POWER single-point data that is already in a DataFrame.

        Assumes the output of ``pvlib.iotools.get_nasa_power`` called with
        ``map_variables=False`` (checked via the presence of a ``WS<height>M`` column), so columns keep NASA POWER names (``WS10M``,
        ``WD50M``, ``PS``, ``T2M`` ...) and units (``PS`` in kPa), which the
        rest of this class assumes. Produces ``self.data`` in the same layout
        as :meth:`import_data`.

        Parameters
        ----------
        df : pandas.DataFrame
            Either indexed by a ``DatetimeIndex`` or containing the raw
            ``YEAR, MO, DY, HR`` columns.
        meta : dict, optional
            Metadata returned alongside the data (e.g. pvlib's ``meta``).
            Searched for latitude / longitude / elevation, including a GeoJSON
            ``coordinates`` list ``[lon, lat, elev]``.
        time_reference : {'UTC', 'LST'}, optional
            Required only when the timestamps are timezone-naive. A tz-aware
            index is converted to UTC and ``time_reference`` is set to 'UTC'.
        latitude, longitude, elevation : optional
            Override anything found in ``meta``.

        Returns
        -------
        None
            Populates ``self.latitude``, ``self.longitude``,
            ``self.average_elevation``, ``self.time_reference`` and
            ``self.data`` in place.
        """
        # NASA POWER wind speed columns are named WS<height>M (e.g. WS10M, WS50M).
        # pvlib with map_variables=True renames these (e.g. WS10M -> wind_speed).
        if not any(re.fullmatch(r'WS\d+M', str(c)) for c in df.columns):
            raise ValueError(
                "No NASA POWER wind speed column (WS<height>M, e.g. WS10M) found. "
                f"Columns: {list(df.columns)}. If this came from pvlib, call "
                "get_nasa_power(..., map_variables=False) and request WS10M/WS50M.")

        df = df.copy()

        # --- timestamps -----------------------------------------------------
        date_cols = ['YEAR', 'MO', 'DY', 'HR']
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
            df = df.drop(columns=[c for c in date_cols if c in df.columns])
        elif set(date_cols) <= set(df.columns):
            ts = pd.DatetimeIndex(pd.to_datetime(dict(
                year=df['YEAR'], month=df['MO'], day=df['DY'], hour=df['HR'])))
            df = df.drop(columns=date_cols)
        else:
            raise ValueError("df needs a DatetimeIndex or YEAR, MO, DY, HR columns.")

        if ts.tz is not None:
            ts = ts.tz_convert('UTC')
            self.time_reference = 'UTC'
        else:
            if time_reference is None or time_reference.upper() not in ('UTC', 'LST'):
                raise ValueError("Timestamps are timezone-naive: pass time_reference='UTC' or 'LST'.")
            self.time_reference = time_reference.upper()
            if self.time_reference == 'UTC':
                ts = ts.tz_localize('UTC')
            else:
                print("Time reference is LST. NASA POWER 'LST' is Local Solar Time, "
                      "not a standard time zone; timestamps left timezone-naive.")

        # --- values ---------------------------------------------------------
        df = df.replace(-999.0, np.nan)       # NASA POWER missing-value flag
        solar_cols = [c for c in df.columns if 'SW' in c]
        df = df.drop(columns=solar_cols)
        df = df.reset_index(drop=True)
        df.insert(0, 'Timestamp', pd.Series(ts))

        # --- site metadata --------------------------------------------------
        coords = self._meta_get(meta, 'coordinates')
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon_m, lat_m = coords[0], coords[1]
            elev_m = coords[2] if len(coords) > 2 else None
        else:
            lat_m = self._meta_get(meta, 'latitude', 'lat')
            lon_m = self._meta_get(meta, 'longitude', 'lon')
            elev_m = self._meta_get(meta, 'elevation', 'altitude')
        lat = latitude if latitude is not None else lat_m
        lon = longitude if longitude is not None else lon_m
        self.latitude = float(lat) if lat is not None else None
        self.longitude = float(lon) if lon is not None else None
        self.average_elevation = elevation if elevation is not None else elev_m
        if self.latitude is None or self.longitude is None:
            print("Warning: latitude/longitude not found; pass them explicitly "
                  "(needed for export_to_sam_csv).")

        self.data = df

    def utc_to_lst(self, lst_offset: int):
        utc_str = f"{lst_offset:+d}:00"
        print(utc_str)
        self.time_reference = 'LST'
        self.data['Timestamp'] = self.data['Timestamp'].dt.tz_convert(utc_str)

    def add_speed_at_height(self,
                        heights: list[int],
                        z0: float | None = None,
                        alpha: float | None = None,
                        model: str = "power_law",
                        plot: bool = False):
        """Extrapolate wind speed from 50 m to one or more target heights.

        Uses either the power law or the logarithmic wind profile model.
        If the required profile parameter (``alpha`` or ``z0``) is not
        supplied it is fitted automatically from the 10 m and 50 m data
        already present in ``self.data``.

        New columns are added to ``self.data`` using the naming convention
        ``WS{height}M`` and ``WD{height}M``. Wind direction at the new
        height is approximated as the circular mean of the 10 m and 50 m
        directions.

        Parameters
        ----------
        heights : list of int or int
            Target height(s) in metres.
        z0 : float or None, optional
            Surface roughness length in metres (logarithmic model only).
            If None, ``fit_logarithmic`` is called automatically.
        alpha : float or None, optional
            Power law exponent (power law model only).
            If None, ``fit_power_law`` is called automatically.
        model : {'power_law', 'logarithmic'}, optional
            Wind profile model to use. Default is ``'power_law'``.
        plot : bool, optional
            If True, plot the fitted profile against the 10 m vs 50 m data.
            Default is False.

        Returns
        -------
        None
            Modifies ``self.data`` in place.
        """

        assert model.lower() in ['power_law','logarithmic'], 'Invalid model. Must be "power_law" or "logarithmic"'
        
        # ensure it is a list
        if isinstance(heights,int):
            heights = [heights]

        ws10,ws50= self.data['WS10M'], self.data['WS50M']
        wd10,wd50= self.data['WD10M'], self.data['WD50M']
        if model.lower() == 'logarithmic':
            if z0 is None:
                print('z0 not provided. Will try to fit it based on data available from 10m and 50m')
                p = self.fit_logarithmic()
            else:
                p = z0
                            
            for height in heights:
                new_col = f'WS{height}M'
                scale = log(height/p)/log(50/p)
                self.data[new_col] = scale * ws50
                self.data[f'WD{height}M'] = stats.circmean(np.c_[wd10,wd50],low=0,high=360,axis=1)

        else: # model.lower() == 'power_law':
            if alpha is None:
                print('alpha not provided. Will try to fit it based on data available from 10m and 50m')
                p = self.fit_power_law()
            else:
                p = alpha

            for height in heights:
                new_col = f'WS{height}M'                
                scale = (height/50.0)**p
                self.data[new_col] = scale * ws50
                self.data[f'WD{height}M'] = stats.circmean(np.c_[wd10,wd50],low=0,high=360,axis=1)

        if plot:
            fig,ax = plt.subplots()
            ax.plot(ws50,ws10,'o')
            xl = ax.get_xlim()
            x = np.linspace(xl[0],xl[1],1000)
            if model.lower() == 'logarithmic':
                ax.plot(x,log(10.0/p)/log(50.0/p) * x,linewidth=3,label="Logarithmic fit")
            else:
                ax.plot(x,(10.0/50.0)**p *x,linewidth=3,label='Power law fit')
            ax.set_xlabel('Wind speed at 50m [m/s]')
            ax.set_ylabel('Wind speed at 10m [m/s]')
            ax.legend()
    
    def fit_power_law(self):
        """Fit the power law exponent α to the 10 m and 50 m wind speed data.

        Minimises the residual sum of squares between the measured 10 m wind
        speeds and those predicted from 50 m using the power law. Optimisation
        is performed in log-space (over ``log_alpha``) to guarantee α > 0.

        Returns
        -------
        float
            Fitted power law exponent α (dimensionless).
        """
        ws10,ws50= self.data['WS10M'], self.data['WS50M']
        rss = lambda log_alpha: np.sum((ws10 - power_law(ws50,50,10,np.exp(log_alpha)) )**2)
        alpha = np.exp(minimize_scalar(rss)['x'])
        print(f'alpha = {alpha}')
        return alpha
    
    def fit_logarithmic(self):
        """Fit the surface roughness length z₀ to the 10 m and 50 m wind speed data.

        Minimises the residual sum of squares between the measured 10 m wind
        speeds and those predicted from 50 m using the logarithmic profile.
        Optimisation is performed in log-space (over ``log_z0``) to guarantee
        z₀ > 0.

        Returns
        -------
        float
            Fitted surface roughness length z₀ in metres.
        """
        ws10,ws50= self.data['WS10M'], self.data['WS50M']
        rss = lambda log_z0: np.sum((ws10 - logarithmic(ws50,50,10,np.exp(log_z0)) )**2)
        z0 = np.exp(minimize_scalar(rss)['x'])
        print(f'z0 = {z0:.3e}m')
        return z0
    
    def export_to_sam_csv(self,utc_offset: int, file_name: str,
                          date_range: list | None = None):
        """Export the wind data to a SAM-compatible CSV file.

        Writes a single-row header followed by the time-series data in the
        format expected by NREL's System Advisor Model (SAM) wind resource
        CSV format. Column names are converted to SAM conventions, e.g.
        ``WS80M`` → ``wind speed at 80m (m/s)``.

        Two assumptions are made and printed as runtime warnings:

        * Surface pressure (``PS``) is labelled as measured at 10 m even
          though MERRA-2 does not provide a height-resolved pressure field.
        * Air temperature uses the 2 m value (``T2M``) as a proxy for 10 m.

        Unit conversions applied:

        * Pressure: kPa → Pa.

        Parameters
        ----------
        utc_offset : int
            UTC offset in hours for the site (e.g. ``10`` for AEST, ``-7`` for
            MST). NASA POWER LST data uses a fixed offset with no DST, so an
            integer offset is the appropriate representation. Written to both
            ``Site Timezone`` and ``Data Timezone`` header fields.
        file_name : str or path-like
            Output file path for the SAM CSV.
        date_range : two-element array-like of datetime-like, optional
            ``[start, end]`` bounds (inclusive) used to slice ``self.data``
            before export. Compared against the ``Timestamp`` column.
            If ``None`` (default), all rows are exported.

        Returns
        -------
        None
            Writes the file to disk.
        """
        # See SAM CSV Format for Wind for more details on format.

        # header
        row1 = ['Site Timezone',utc_offset,
                'Data Timezone',utc_offset,
                'Latitude',self.latitude,
                'Longitude',self.longitude,
                'Elevation',self.average_elevation]
        row1 = [str(r) for r in row1]
        row1 = ','.join(row1)

        df2 = self.data.copy()

        # optional date range filter
        if date_range is not None:
            mask = (df2['Timestamp'] >= date_range[0]) & (df2['Timestamp'] <= date_range[1])
            df2 = df2.loc[mask]

        # rename wind speed and direction columns
        heights = [name[2:-1] for name in df2.columns if 'WS' in name]
        df2 = df2.rename(columns={f'WS{h}M':f'wind speed at {h}m (m/s)' for h in heights})
        df2 = df2.rename(columns={f'WD{h}M':f'wind direction at {h}m (degrees)' for h in heights})
        df2['PS'] = df2['PS'] * 1000  # convert from kPa to Pa

        print("Warning: Arbitrarily setting pressure data to be at 10m.")
        df2 = df2.rename(columns={'PS':'air pressure at 10m (Pa)'})

        print("Warning: Temperature at 10m is not available in MERRA-2 data. Using temperature at 2m instead.")
        df2 = df2.rename(columns={'T2M':'air temperature at 10m (C)'})

        # add back columns for year, month, day, hour, minute if not there
        def prepend(c,v):
            if c not in df2.columns:
                df2.insert(0,c,v)
        prepend('Minute',df2.Timestamp.dt.minute)
        prepend('Hour',df2.Timestamp.dt.hour)
        prepend('Day',df2.Timestamp.dt.day)
        prepend('Month',df2.Timestamp.dt.month)
        prepend('Year',df2.Timestamp.dt.year)

        df2 = df2.drop(columns='Timestamp')

        with open(file_name,'w',newline='') as f:
            f.write(row1+'\n')
            df2.to_csv(f,header=True,index=False)
