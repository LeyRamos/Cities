# Databricks notebook source
# MAGIC %md
# MAGIC # general import

# COMMAND ----------

import os 
# Disable warnings, set Matplotlib inline plotting and load Pandas package
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone
from dateutil import tz
from datetime import datetime, timedelta
import geojson
import geopandas as gpd  
from fiona.crs import from_epsg
import os, json
from shapely.geometry import shape, Point, Polygon, MultiPoint

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
%matplotlib inline
import matplotlib.pyplot as plt
import osmnx as ox


import os 
# Disable warnings, set Matplotlib inline plotting and load Pandas package
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
#pd.options.display.mpl_style = 'default'
from datetime import datetime
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
from dateutil import tz
import geojson
import geopandas as gpd
from fiona.crs import from_epsg
import os, json
from shapely.geometry import shape, Point, Polygon, MultiPoint
%matplotlib inline
import matplotlib.pyplot as plt
from geopandas.tools import sjoin
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm

import plotly.express as px

import folium

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # load datasets

# COMMAND ----------

DF_and = pd.read_csv('../input/belgium-obu/Anderlecht_15.csv', header=None)
DF_and.columns = ['datetime','street_id','count','vel']
nRow_and, nCol_and = DF_and.shape

DF_bxl = pd.read_csv('../input/belgium-obu/Bxl_15.csv', header=None)
DF_bxl.columns = ['datetime','street_id','count','vel']
nRow_bxl, nCol_bxl = DF_bxl.shape

DF_bel = pd.read_csv('../input/belgium-obu/Bel_15.csv', header=None)
DF_bel.columns = ['datetime','street_id','count','vel']
nRow_bel, nCol_bel = DF_bel.shape

print(f'in Anderlecht 15 min there are {nRow_and} rows and {nCol_and} columns')
print(f'in Bruxelles 15 min there are {nRow_bxl} rows and {nCol_bxl} columns')
print(f'in Belgium 15 min there are {nRow_bel} rows and {nCol_bel} columns')

# COMMAND ----------

DF_bel.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Belgium

# COMMAND ----------

# MAGIC %md
# MAGIC ## visualise pattern

# COMMAND ----------

DF_bel.sort_values(by=['datetime']).groupby(['datetime']).agg({'count':'sum'}).plot(figsize=(20,5), color = 'red', rot=45, title='Belgium')
plt.show()

# COMMAND ----------

DF_bel.sort_values(by=['datetime']).groupby(['datetime']).agg({'count':'mean'}).plot(figsize=(20,5), color = 'blue', rot=45, title='Belgium')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## count trucks: split working days, saturdays and sundays

# COMMAND ----------

DF_bel_sum = DF_bel.sort_values(by=['datetime']).groupby(['datetime']).agg({'count':'sum'}).reset_index()
DF_bel_sum['time'] = pd.to_datetime(DF_bel_sum['datetime']).dt.time
DF_bel_sum['DayOfWeek'] = pd.to_datetime(DF_bel_sum['datetime']).dt.dayofweek

DF_bel_working_ = DF_bel_sum[DF_bel_sum['DayOfWeek'] < 5]
DF_bel_saturday_ = DF_bel_sum[DF_bel_sum['DayOfWeek'] == 5]
DF_bel_sunday_ = DF_bel_sum[DF_bel_sum['DayOfWeek'] == 6]


# COMMAND ----------

# MAGIC %md
# MAGIC ## total distribution

# COMMAND ----------

import seaborn as sns

sns.distplot(DF_bel_sum['count'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})


# COMMAND ----------

# MAGIC %md
# MAGIC ## working days distribution

# COMMAND ----------

sns.distplot(DF_bel_working_['count'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})



# COMMAND ----------

# MAGIC %md
# MAGIC ## working days: day/night

# COMMAND ----------

start = datetime.strptime('03:00:00', '%H:%M:%S').time()
end = datetime.strptime('15:00:00', '%H:%M:%S').time()

DF_bel_working_day = DF_bel_working_[DF_bel_working_['time'].between(start, end)]


start = datetime.strptime('15:00:00', '%H:%M:%S').time()
middle_1 = datetime.strptime('23:59:00', '%H:%M:%S').time()
middle_2 = datetime.strptime('00:00:00', '%H:%M:%S').time()
end = datetime.strptime('02:59:00', '%H:%M:%S').time()

DF_bel_working_night_1 = DF_bel_working_[DF_bel_working_['time'].between(start, middle_1)]
DF_bel_working_night_2 = DF_bel_working_[DF_bel_working_['time'].between(middle_2, end)]

DF_bel_working_night = pd.concat([DF_bel_working_night_1, DF_bel_working_night_2], axis=0)


# COMMAND ----------

sns.distplot(DF_bel_working_day['count'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})


# COMMAND ----------

sns.distplot(DF_bel_working_night['count'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})


# COMMAND ----------

# MAGIC %md
# MAGIC ## saturdays distribution

# COMMAND ----------

sns.distplot(DF_bel_saturday_['count'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})



# COMMAND ----------

# MAGIC %md
# MAGIC ## sundays distribution

# COMMAND ----------

sns.distplot(DF_bel_sunday_['count'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})

# COMMAND ----------

# MAGIC %md
# MAGIC ## avg vel trcuks on streets: split working days, saturdays and sundays

# COMMAND ----------

DF_bel_sum_vel = DF_bel.sort_values(by=['datetime']).groupby(['datetime']).agg({'vel':'mean'}).reset_index()
DF_bel_sum_vel['time'] = pd.to_datetime(DF_bel_sum_vel['datetime']).dt.time
DF_bel_sum_vel['DayOfWeek'] = pd.to_datetime(DF_bel_sum_vel['datetime']).dt.dayofweek

DF_bel_working_vel = DF_bel_sum_vel[DF_bel_sum_vel['DayOfWeek'] < 5]
DF_bel_saturday_vel = DF_bel_sum_vel[DF_bel_sum_vel['DayOfWeek'] == 5]
DF_bel_sunday_vel = DF_bel_sum_vel[DF_bel_sum_vel['DayOfWeek'] == 6]

# COMMAND ----------

## total distribution

# COMMAND ----------

sns.distplot(DF_bel_sum_vel['vel'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})

# COMMAND ----------

## workind days distribution

# COMMAND ----------

sns.distplot(DF_bel_working_vel['vel'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})

# COMMAND ----------

## saturdays distribution

# COMMAND ----------

sns.distplot(DF_bel_saturday_vel['vel'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})

# COMMAND ----------

## sundays distribution

# COMMAND ----------

sns.distplot(DF_bel_sunday_vel['vel'], hist=False, kde=True, 
             bins= 200, color = 'blue',
             hist_kws={'edgecolor':'black'})

# COMMAND ----------

# MAGIC %md
# MAGIC ## visualise average daily pattern

# COMMAND ----------


# ---------- plot working days

DF_bel_working = DF_bel_working_.groupby('time').agg({'count':['mean','std']})
DF_bel_working.columns = ['mean','std']
DF_bel_working['mean + std'] = DF_bel_working['mean'] + DF_bel_working['std']
DF_bel_working['mean - std'] = DF_bel_working['mean'] - DF_bel_working['std']

ax = DF_bel_working[['mean']].plot(color="orange", title = 'avg working days daily pattern')
DF_bel_working[['mean + std']].plot.area(ax=ax, color="gray", alpha=0.2)
DF_bel_working[['mean - std']].plot.area(ax=ax, color="white")
plt.show()


# ---------- plot saturdays

DF_bel_saturday = DF_bel_saturday_.groupby('time').agg({'count':['mean','std']})
DF_bel_saturday.columns = ['mean','std']
DF_bel_saturday['mean + std'] = DF_bel_saturday['mean'] + DF_bel_saturday['std']
DF_bel_saturday['mean - std'] = DF_bel_saturday['mean'] - DF_bel_saturday['std']

ax = DF_bel_saturday[['mean']].plot(color="orange", title = 'avg saturdays daily pattern')
DF_bel_saturday[['mean + std']].plot.area(ax=ax, color="gray", alpha=0.2)
DF_bel_saturday[['mean - std']].plot.area(ax=ax, color="white")
plt.show()


# ---------- plot sundays

DF_bel_sunday = DF_bel_sunday_.groupby('time').agg({'count':['mean','std']})
DF_bel_sunday.columns = ['mean','std']
DF_bel_sunday['mean + std'] = DF_bel_sunday['mean'] + DF_bel_sunday['std']
DF_bel_sunday['mean - std'] = DF_bel_sunday['mean'] - DF_bel_sunday['std']

ax = DF_bel_sunday[['mean']].plot(color="orange", title = 'avg sundays daily pattern')
DF_bel_sunday[['mean + std']].plot.area(ax=ax, color="gray", alpha=0.2)
DF_bel_sunday[['mean - std']].plot.area(ax=ax, color="white")
plt.show()



# ---------- plot all together

DF_bel_working['avg working days'] = DF_bel_working[['mean']]
DF_bel_saturday['avg saturdays'] = DF_bel_saturday[['mean']]
DF_bel_sunday['avg sunday'] = DF_bel_sunday[['mean']]

ax = DF_bel_working[['avg working days']].plot(color="red", title = 'avg  day pattern')
DF_bel_saturday[['avg saturdays']].plot(ax=ax, color="green")
DF_bel_sunday[['avg sunday']].plot(ax=ax, color="blue")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## spot outlier days

# COMMAND ----------

DF_bel['datetime'] = pd.to_datetime(DF_bel['datetime'])
DF_bel['date'] = DF_bel['datetime'].dt.date

# COMMAND ----------

plt_date = DF_bel.groupby('date').agg({'count':'sum'})

ax = plt_date.plot.bar(figsize=(20,5), alpha=0.5)
plt_date.plot(alpha=0.5, color ='red', ax=ax )
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC outliers
# MAGIC
# MAGIC * first days of january
# MAGIC * 13-02-2020

# COMMAND ----------

# MAGIC %md
# MAGIC ## consider working days 

# COMMAND ----------

DF_bel['DayOfWeek'] = DF_bel['datetime'].dt.dayofweek

# working days
DF_bel_working_ = DF_bel[DF_bel['DayOfWeek'] < 5]

DF_bel_working_.groupby('date').agg({'count':'sum'}).plot.bar(figsize=(20,5), alpha=0.5)
plt.show()

# COMMAND ----------

DF_bel_working = DF_bel_working_.groupby('date').agg({'count':'sum'}).reset_index()
DF_bel_working['Month'] = pd.to_datetime(DF_bel_working['date']).dt.month


fig = px.box(DF_bel_working, x='Month', y="count", hover_data=["date"])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # working in progress...

# COMMAND ----------

# MAGIC %md
# MAGIC ## once upon time

# COMMAND ----------

DF_and.sort_values(by=['datetime']).groupby(['datetime']).agg({'count':'sum'}).plot(figsize=(20,5), color = 'red', rot=45, title='Anderlecht-Normal')
plt.show()

# COMMAND ----------

DF_and_sum = DF_and.sort_values(by=['datetime']).groupby(['datetime']).agg({'count':'mean'}).reset_index()
DF_and_sum['time'] = pd.to_datetime(DF_and_sum['datetime']).dt.time
DF_and_sum['DayOfWeek'] = pd.to_datetime(DF_and_sum['datetime']).dt.dayofweek

DF_and_working_ = DF_and_sum[DF_and_sum['DayOfWeek'] < 5]

# COMMAND ----------

# ---------- plot working days

DF_and_working = DF_and_working_.groupby('time').agg({'count':['mean','std']})
DF_and_working.columns = ['mean','std']
DF_and_working['mean + std'] = DF_and_working['mean'] + DF_and_working['std']
DF_and_working['mean - std'] = DF_and_working['mean'] - DF_and_working['std']

ax = DF_and_working[['mean']].plot(color="orange", title = 'avg working days daily pattern')
DF_and_working[['mean + std']].plot.area(ax=ax, color="gray", alpha=0.2)
DF_and_working[['mean - std']].plot.area(ax=ax, color="white")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## covid time

# COMMAND ----------

DF_and_21 = pd.read_csv('../input/belgium-obu/Anderlecht_15_2021.csv', header=None)
DF_and_21.columns = ['datetime','street_id','count','vel']
nRow_and, nCol_and = DF_and.shape

# COMMAND ----------

DF_and_21.sort_values(by=['datetime']).groupby(['datetime']).agg({'count':'sum'}).plot(figsize=(20,5), color = 'red', rot=45, title='Anderlecht-Covid')
plt.show()

# COMMAND ----------

DF_and_21_sum = DF_and_21.sort_values(by=['datetime']).groupby(['datetime']).agg({'count':'mean'}).reset_index()
DF_and_21_sum['time'] = pd.to_datetime(DF_and_21_sum['datetime']).dt.time
DF_and_21_sum['DayOfWeek'] = pd.to_datetime(DF_and_21_sum['datetime']).dt.dayofweek

DF_and_21_working_ = DF_and_21_sum[DF_and_21_sum['DayOfWeek'] < 5]

# COMMAND ----------

# ---------- plot working days

DF_and_21_working = DF_and_21_working_.groupby('time').agg({'count':['mean','std']})
DF_and_21_working.columns = ['mean','std']
DF_and_21_working['mean + std'] = DF_and_21_working['mean'] + DF_and_21_working['std']
DF_and_21_working['mean - std'] = DF_and_21_working['mean'] - DF_and_21_working['std']

ax = DF_and_21_working[['mean']].plot(color="orange", title = 'avg working days daily pattern')
DF_and_21_working[['mean + std']].plot.area(ax=ax, color="gray", alpha=0.2)
DF_and_21_working[['mean - std']].plot.area(ax=ax, color="white")
plt.show()

# COMMAND ----------



# COMMAND ----------

