# Databricks notebook source
# MAGIC %md ## **Data transformation and EDA** 
# MAGIC
# MAGIC Using output table from [**HA_import_data** notebook](https://community.cloud.databricks.com/?o=2860523018733622#notebook/1117193569927625/command/2169505508737844)
# MAGIC
# MAGIC Data Pipeline:
# MAGIC 1. **HA_import_data** notebook --> import all data and create a parquet table as output 
# MAGIC 2. **HA_data_transformation** notebook --> handle missing data and perform EDA (THIS!)
# MAGIC 3. **HA_feature_engineering** notebook --> create features, target and prepare dataset for modeling
# MAGIC 4. **HA_modeling** notebook --> perform ML modeling and evaluations
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import Libraries
import pyspark
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyspark.sql.functions as f
from pyspark.sql.functions import *
from  pyspark.sql.functions import input_file_name
from pyspark.sql import Window
from pyspark.sql.functions import sum,avg,max,min,mean,count

from pyspark.sql.types import *;
from scipy.stats import *
from scipy import stats

import gzip
# import StringIO --> ModuleNotFoundError: No module named 'StringIO'


from functools import reduce
from pyspark.sql import DataFrame

# COMMAND ----------

# DBTITLE 1,Functions
# Save distionary to csv file
def dict_to_csv(dict_input):
  import csv
  with open(dict_input + '.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in dict_input.items():
       writer.writerow([key, value])
        
# Read back csv file to distionary
def csv_to_dict(dict_input):
  import csv
  with open(dict_input + '.csv') as csv_file:
    reader = csv.reader(csv_file)
    dict_input = dict(reader)

# COMMAND ----------

schema= (StructField("start", TimestampType(), True),
StructField("end", TimestampType(), True),
StructField("user_id", StringType(), True),
StructField("acc_magnitude_mean_mean", DoubleType(), True),
StructField("acc_magnitude_std_mean", DoubleType(), True),
StructField("acc_magnitude_moment3_mean", DoubleType(), True),
StructField("acc_magnitude_moment4_mean", DoubleType(), True),
StructField("acc_magnitude_perc25_mean", DoubleType(), True),
StructField("acc_magnitude_perc50_mean", DoubleType(), True),
StructField("acc_magnitude_perc75_mean", DoubleType(), True),
StructField("acc_magnitude_value_entropy_mean", DoubleType(), True),
StructField("acc_magnitude_time_entropy_mean", DoubleType(), True),
StructField("acc_magnitude_spect_energy_band0_mean", DoubleType(), True),
StructField("acc_magnitude_spect_energy_band1_mean", DoubleType(), True),
StructField("acc_magnitude_spect_energy_band2_mean", DoubleType(), True),
StructField("acc_magnitude_spect_energy_band3_mean", DoubleType(), True),
StructField("acc_magnitude_spect_energy_band4_mean", DoubleType(), True),
StructField("acc_magnitude_spec_spectral_entropy_mean", DoubleType(), True),
StructField("acc_magnitude_autoc_period_mean", DoubleType(), True),
StructField("acc_magnitude_autoc_normalized_ac_mean", DoubleType(), True),
StructField("acc_3d_mean_x_mean", DoubleType(), True),
StructField("acc_3d_mean_y_mean", DoubleType(), True),
StructField("acc_3d_mean_z_mean", DoubleType(), True),
StructField("acc_3d_std_x_mean", DoubleType(), True),
StructField("acc_3d_std_y_mean", DoubleType(), True),
StructField("acc_3d_std_z_mean", DoubleType(), True),
StructField("acc_3d_ro_x_mean", DoubleType(), True),
StructField("acc_3d_ro_y_mean", DoubleType(), True),
StructField("acc_3d_ro_z_mean", DoubleType(), True),
StructField("gyro_magnitude_mean_mean", DoubleType(), True),
StructField("gyro_magnitude_std_mean", DoubleType(), True),
StructField("gyro_magnitude_moment3_mean", DoubleType(), True),
StructField("gyro_magnitude_moment4_mean", DoubleType(), True),
StructField("gyro_magnitude_perc25_mean", DoubleType(), True),
StructField("gyro_magnitude_perc50_mean", DoubleType(), True),
StructField("gyro_magnitude_perc75_mean", DoubleType(), True),
StructField("gyro_magnitude_value_entropy_mean", DoubleType(), True),
StructField("gyro_magnitude_time_entropy_mean", DoubleType(), True),
StructField("gyro_magnitude_spect_energy_band0_mean", DoubleType(), True),
StructField("gyro_magnitude_spect_energy_band1_mean", DoubleType(), True),
StructField("gyro_magnitude_spect_energy_band2_mean", DoubleType(), True),
StructField("gyro_magnitude_spect_energy_band3_mean", DoubleType(), True),
StructField("gyro_magnitude_spect_energy_band4_mean", DoubleType(), True),
StructField("gyro_magnitude_spec_spectral_entropy_mean", DoubleType(), True),
StructField("gyro_magnitude_autoc_period_mean", DoubleType(), True),
StructField("gyro_magnitude_autoc_normalized_ac_mean", DoubleType(), True),
StructField("gyro_3d_mean_x_mean", DoubleType(), True),
StructField("gyro_3d_mean_y_mean", DoubleType(), True),
StructField("gyro_3d_mean_z_mean", DoubleType(), True),
StructField("gyro_3d_std_x_mean", DoubleType(), True),
StructField("gyro_3d_std_y_mean", DoubleType(), True),
StructField("gyro_3d_std_z_mean", DoubleType(), True),
StructField("gyro_3d_ro_xy_mean", DoubleType(), True),
StructField("gyro_3d_ro_xz_mean", DoubleType(), True),
StructField("gyro_3d_ro_yz_mean", DoubleType(), True),
StructField("magnet_magnitude_mean_mean", DoubleType(), True),
StructField("magnet_magnitude_std_mean", DoubleType(), True),
StructField("magnet_magnitude_moment3_mean", DoubleType(), True),
StructField("magnet_magnitude_moment4_mean", DoubleType(), True),
StructField("magnet_magnitude_perc25_mean", DoubleType(), True),
StructField("magnet_magnitude_perc50_mean", DoubleType(), True),
StructField("magnet_magnitude_perc75_mean", DoubleType(), True),
StructField("magnet_magnitude_value_entropy_mean", DoubleType(), True),
StructField("magnet_magnitude_time_entropy_mean", DoubleType(), True),
StructField("magnet_magnitude_spect_energy_band0_mean", DoubleType(), True),
StructField("magnet_magnitude_spect_energy_band1_mean", DoubleType(), True),
StructField("magnet_magnitude_spect_energy_band2_mean", DoubleType(), True),
StructField("magnet_magnitude_spect_energy_band3_mean", DoubleType(), True),
StructField("magnet_magnitude_spect_energy_band4_mean", DoubleType(), True),
StructField("magnet_magnitude_spec_spectral_entropy_mean", DoubleType(), True),
StructField("magnet_3d_mean_x_mean", DoubleType(), True),
StructField("magnet_3d_mean_y_mean", DoubleType(), True),
StructField("magnet_3d_mean_z_mean", DoubleType(), True),
StructField("magnet_3d_std_x_mean", DoubleType(), True),
StructField("magnet_3d_std_y_mean", DoubleType(), True),
StructField("magnet_3d_std_z_mean", DoubleType(), True),
StructField("magnet_3d_ro_xy_mean", DoubleType(), True),
StructField("magnet_3d_ro_xz_mean", DoubleType(), True),
StructField("magnet_3d_ro_yz_mean", DoubleType(), True),
StructField("magnet_avr_cosine_similarity_lag0_mean", DoubleType(), True),
StructField("magnet_avr_cosine_similarity_lag1_mean", DoubleType(), True),
StructField("magnet_avr_cosine_similarity_lag2_mean", DoubleType(), True),
StructField("magnet_avr_cosine_similarity_lag3_mean", DoubleType(), True),
StructField("magnet_avr_cosine_similarity_lag4_mean", DoubleType(), True),
StructField("acc_watch_magnitude_mean_mean", DoubleType(), True),
StructField("acc_watch_magnitude_std_mean", DoubleType(), True),
StructField("acc_watch_magnitude_moment3_mean", DoubleType(), True),
StructField("acc_watch_magnitude_moment4_mean", DoubleType(), True),
StructField("acc_watch_magnitude_perc25_mean", DoubleType(), True),
StructField("acc_watch_magnitude_perc50_mean", DoubleType(), True),
StructField("acc_watch_magnitude_perc75_mean", DoubleType(), True),
StructField("acc_watch_magnitude_value_entropy_mean", DoubleType(), True),
StructField("acc_watch_magnitude_time_entropy_mean", DoubleType(), True),
StructField("acc_watch_magnitude_spect_energy_band0_mean", DoubleType(), True),
StructField("acc_watch_magnitude_spect_energy_band1_mean", DoubleType(), True),
StructField("acc_watch_magnitude_spect_energy_band2_mean", DoubleType(), True),
StructField("acc_watch_magnitude_spect_energy_band3_mean", DoubleType(), True),
StructField("acc_watch_magnitude_spec_spectral_entropy_mean", DoubleType(), True),
StructField("acc_watch_magnitude_autoc_period_mean", DoubleType(), True),
StructField("acc_watch_magnitude_autoc_normalized_ac_mean", DoubleType(), True),
StructField("acc_watch_3d_mean_x_mean", DoubleType(), True),
StructField("acc_watch_3d_mean_y_mean", DoubleType(), True),
StructField("acc_watch_3d_mean_z_mean", DoubleType(), True),
StructField("acc_watch_3d_std_x_mean", DoubleType(), True),
StructField("acc_watch_3d_std_y_mean", DoubleType(), True),
StructField("acc_watch_3d_std_z_mean", DoubleType(), True),
StructField("acc_watch_3d_ro_xy_mean", DoubleType(), True),
StructField("acc_watch_3d_ro_xz_mean", DoubleType(), True),
StructField("acc_watch_3d_ro_yz_mean", DoubleType(), True),
StructField("acc_watch_spec_x_energy_band0_mean", DoubleType(), True),
StructField("acc_watch_spec_x_energy_band1_mean", DoubleType(), True),
StructField("acc_watch_spec_x_energy_band2_mean", DoubleType(), True),
StructField("acc_watch_spec_x_energy_band3_mean", DoubleType(), True),
StructField("acc_watch_spec_y_energy_band0_mean", DoubleType(), True),
StructField("acc_watch_spec_y_energy_band1_mean", DoubleType(), True),
StructField("acc_watch_spec_y_energy_band2_mean", DoubleType(), True),
StructField("acc_watch_spec_y_energy_band3_mean", DoubleType(), True),
StructField("acc_watch_spec_z_energy_band0_mean", DoubleType(), True),
StructField("acc_watch_spec_z_energy_band1_mean", DoubleType(), True),
StructField("acc_watch_spec_z_energy_band2_mean", DoubleType(), True),
StructField("acc_watch_spec_z_energy_band3_mean", DoubleType(), True),
StructField("acc_watch__avr_cosine_similarity_lag0_mean", DoubleType(), True),
StructField("acc_watch__avr_cosine_similarity_lag1_mean", DoubleType(), True),
StructField("acc_watch__avr_cosine_similarity_lag2_mean", DoubleType(), True),
StructField("acc_watch__avr_cosine_similarity_lag3_mean", DoubleType(), True),
StructField("acc_watch__avr_cosine_similarity_lag4_mean", DoubleType(), True),
StructField("acc_watch_head_men_cos_mean", DoubleType(), True),
StructField("acc_watch_head_std_cos_mean", DoubleType(), True),
StructField("acc_watch_head_mom3_cos_mean", DoubleType(), True),
StructField("acc_watch_head_mom4_cos_mean", DoubleType(), True),
StructField("acc_watch_head_men_sin_mean", DoubleType(), True),
StructField("acc_watch_head_std_sin_mean", DoubleType(), True),
StructField("acc_watch_head_mom3_sin_mean", DoubleType(), True),
StructField("acc_watch_head_mom4_sin_mean", DoubleType(), True),
StructField("loc_valid_updates_sum", IntegerType(), True),
StructField("loc_log_latitude_range_mean", DoubleType(), True),
StructField("loc_log_longitude_range_mean", DoubleType(), True),
StructField("loc_min_altitude_min", DoubleType(), True),
StructField("loc_max_altitude_max", DoubleType(), True),
StructField("loc_min_speed_min", DoubleType(), True),
StructField("loc_max_speed_max", DoubleType(), True),
StructField("loc_best_horizontal_accuracy_mean", DoubleType(), True),
StructField("loc_best_vertical_accuracy_mean", DoubleType(), True),
StructField("loc_diameter_mean", DoubleType(), True),
StructField("loc_log_diameter_mean", DoubleType(), True),
StructField("loc_features_std_lat_mean", DoubleType(), True),
StructField("loc_features_std_long_mean", DoubleType(), True),
StructField("loc_features_lat_change_mean", DoubleType(), True),
StructField("loc_features_log_change_mean", DoubleType(), True),
StructField("loc_features_mean_abs_lat_deriv_mean", DoubleType(), True),
StructField("loc_features_mean_abs_long_deriv_mean", DoubleType(), True),
StructField("lab_vehicle", IntegerType(), True),
StructField("lab_bicycling", IntegerType(), True),
StructField("lab_walking", IntegerType(), True),
StructField("lab_sitting", IntegerType(), True),
StructField("lab_standing", IntegerType(), True),
StructField("lab_no_traveling", IntegerType(), True),
StructField("lab_no_traveling_definition", IntegerType(), True)
)


df = (
spark.read.option("delimiter", ',').
csv("/datasets/features_labels/output/df_all_windows.csv.gz",
schema=schema,
header=True,
ignoreLeadingWhiteSpace=True,
ignoreTrailingWhiteSpace=True,
nullValue='NA'))


# COMMAND ----------

df_ha =spark.read.parquet("/datasets/features_labels/output/df_all.parquet").schema(df_ha_schema)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE HA_DB

# COMMAND ----------

# MAGIC %sql DESCRIBE HA_DATA

# COMMAND ----------

# Use table as df
df_ha = spark.sql("SELECT * FROM HA_DATA")

display(df_ha)

# COMMAND ----------

dbutils.data.summarize(df_all_windows_test, precise= True)

# COMMAND ----------

# DBTITLE 1,Define Numerical variables
# Use table as df
#df_all_windows_test = spark.sql("SELECT * FROM HA_DATA")

# Numerical columns
columns_sensor_startswith = ['acc', 'acc_watch', 'loc']
columns_sensor = [i for i in df_all_windows_test.columns if i.startswith(tuple(columns_sensor_startswith))]

# Label columns
columns_label = [i for i in df_all_windows_test.columns if i.startswith('lab')]

# COMMAND ----------

# DBTITLE 1,Correlation Matrix
from pyspark.mllib.stat import Statistics

# create RDD table for correlation calculation
rdd_table = df_all_windows_test.select(columns_sensor).rdd.map(lambda row: row[0:])

# get the correlation matrix
corr_matrix = Statistics.corr(rdd_table, method="pearson")



# COMMAND ----------

def plot_corr_matrix(correlations,attr,fig_no):
    fig=plt.figure(fig_no)
    ax=fig.add_subplot(111)
    ax.set_title("Correlation Matrix for Sensor columns")
    ax.set_xticklabels(['']+attr)
    ax.set_yticklabels(['']+attr)
    cax=ax.matshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    plt.figure(figsize=(40,40))
    plt.show()

plot_corr_matrix(corr_matrix, columns_sensor, 10)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(40,40))
sns.heatmap(corr_matrix, annot=False, fmt="g", cmap='viridis')
plt.show()

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols= columns_sensor, outputCol=vector_col)
df_vector = assembler.transform(df_all_windows_test).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)
cor_np = matrix.collect()[0][matrix.columns[0]].toArray().tolist()

print(cor_np)

# convert to dataframe
df_sensor_corr = spark.createDataFrame(cor_np,columns_sensor)
df_sensor_corr.show()




# COMMAND ----------

# plot

def plot_corr_matrix(correlations,attr,fig_no):
    fig=plt.figure(fig_no)
    ax=fig.add_subplot(111)
    ax.set_title("Correlation Matrix for Specified Attributes")
    ax.set_xticklabels(['']+attr)
    ax.set_yticklabels(['']+attr)
    cax=ax.matshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    plt.show()

plot_corr_matrix(corrmatrix, columns, 234)

# COMMAND ----------

dfpd = df_all_windows_test.toPandas()

# COMMAND ----------

dfCorr = dfpd.corr()

# COMMAND ----------

filteredDf = dfCorr[((dfCorr >= .5) | (dfCorr <= -.5)) & (dfCorr !=1.000)]
plt.figure(figsize=(30,30))
sn.heatmap(filteredDf, annot=True, cmap="Reds")
plt.show()

# COMMAND ----------

dfCorr = df.corr()
filteredDf = dfCorr[((dfCorr >= .5) | (dfCorr <= -.5)) & (dfCorr !=1.000)]
plt.figure(figsize=(30,10))
sn.heatmap(filteredDf, annot=True, cmap="Reds")
plt.show()


corr = df.corr()

kot = corr[corr>=.9]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Greens")


import pandas as pd
d = {'x1': [1, 4, 4, 5, 6], 
     'x2': [0, 0, 8, 2, 4], 
     'x3': [2, 8, 8, 10, 12], 
     'x4': [-1, -4, -4, -4, -5]}
df = pd.DataFrame(data = d)
print("Data Frame")
print(df)
print()

print("Correlation Matrix")
print(df.corr())
print()

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 3))

# COMMAND ----------

columns_sensor

# COMMAND ----------

# DBTITLE 1,Missing Data - Replace null values on label columns
# Replace null values with 0 for all label columns
df_ha = df_ha.na.fill(value=0,subset= columns_label)

# COMMAND ----------

df_ha = df_ha.na.drop("all")

# COMMAND ----------

# DBTITLE 1,Missing Data - Count nulls
# for col in df.columns:
#   print(col, "\t", "Nulls: ", df.filter(df[col].isNull()).count())

# First, remove rows with null values on all columns, if any
df_ha = df_ha.na.drop("all")

# Count null values for each numerical (sensor) column and create a dictionary
dict_null_sensor = {col:df_ha.filter(df_ha[col].isNull()).count() for col in columns_sensor}
dict_null_sensor

# Count null values for each numerical (sensor) column and create a dictionary
#dict_null_label = {col:df_ha.filter(df_ha[col].isNull()).count() for col in columns_label}
#dict_null_label

# COMMAND ----------

# Save null dictionariy as csv file
dict_input = dict_null_sensor
dict_to_csv(dict_input)

# Read back csv file to distionary
#dict_input = dict_null_sensor
#dict_null_sensor_from_csv = csv_to_dict(dict_input)

# COMMAND ----------

# Save dictionaries as csv files
import csv

with open('dict_null_sensor.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file) # writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #writer.writeheader()
    for key, value in dict_null_sensor.items():
       writer.writerow([key, value])
        
# Read back the csv files as dictionaries
with open('dict_null_sensor.csv') as csv_file:
    reader = csv.reader(csv_file)
    dict_null_sensor = dict(reader)

# COMMAND ----------

# DBTITLE 1,Drop columns with null values
# Drop numerical/sensor columns with > 50% of null values to drop
nan_percent_sensor= 0.5
rows_original = 22640760
rows_threshold_sensor = rows_original*nan_percent_sensor

columns_to_drop_2 = []

for col in dict_null_sensor.keys():
  if dict_null_sensor.values() >= rows_threshold_sensor:
    columns_to_drop_2.append(col) 

#list(dict_null.keys())
columns_to_drop_2

# df.drop(*columns_to_drop_2)

# COMMAND ----------

# Drop label columns with > 80% of null values to drop
nan_percent_label = 0.8
rows_original = 22640760
rows_threshold_label = rows_original*nan_percent_label

labels_to_drop_2 = []

for col in dict_null_label.keys():
  if dict_null_label.values() >= rows_threshold_label:
    labels_to_drop_2.append(col) 
    
labels_to_drop_2

# df.drop(*labels_to_drop_2)

# COMMAND ----------

# Drop label columns with > 80% of null values to drop


def drop_null_columns (df, dict_null, nan_percent, rows_original):
  rows_threshold = rows_original*nan_percent
  to_drop = []
  
  for col in dict_null.keys():
  if dict_null.values() >= rows_threshold:
    to_drop.append(col) 
  df.drop(*to_drop)
  
  print("Columns dropped:")
  to_drop


# COMMAND ----------

df =
nan_percent_label = 0.8
rows_original = 22640760



# COMMAND ----------

# Redefine columns after transformations
# Numerical columns
columns_sensor_startswith = ['acc', 'gyro', 'magnet', 'acc_watch', 'loc', 'measure']
columns_sensor = [i for i in df_ha.columns if i.startswith(tuple(columns_sensor_startswith))]

# Label columns
columns_label = [i for i in df_ha.columns if i.startswith('lab')]

# COMMAND ----------

# df.select('var_0','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13','var_14').describe().toPandas()
df_ha.select(columns_sensor).describe()

# COMMAND ----------

quantile = df.approxQuantile(['var_0'], [0.25, 0.5, 0.75], 0)
quantile_25 = quantile[0][0]
quantile_50 = quantile[0][1]
quantile_75 = quantile[0][2]
print('quantile_25: '+str(quantile_25))
print('quantile_50: '+str(quantile_50))
print('quantile_75: '+str(quantile_75))
'''
quantile_25: 8.4537 
quantile_50: 10.5247 
quantile_75: 12.7582
'''

# COMMAND ----------

from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import udf
import numpy as np

var = "var_0"
# create the split list ranging from 0 to  21, interval of 0.5
split_list = [float(i) for i in np.arange(0,21,0.5)]
# initialize buketizer
bucketizer = Bucketizer(splits=split_list,inputCol=var, outputCol="buckets")
# transform
df_buck = bucketizer.setHandleInvalid("keep").transform(df.select(var).dropna())

# the "buckets" column gives the bucket rank, not the acctual bucket value(range), 
# use dictionary to match bucket rank and bucket value
bucket_names = dict(zip([float(i) for i in range(len(split_list[1:]))],split_list[1:]))
# user defined function to update the data frame with the bucket value
udf_foo = udf(lambda x: bucket_names[x], DoubleType())
bins = df_buck.withColumn("bins", udf_foo("buckets")).groupBy("bins").count().sort("bins").toPandas()

# COMMAND ----------

from pyspark.ml.feature import QuantileDiscretizer

# Utilizar el QuantileDiscretizer para crear una nueva columna llamada time_bin que genere 10 bins para el valor de la columna time https://spark.apache.org/docs/latest/ml-features.html#quantilediscretizer
discretizer = QuantileDiscretizer(numBuckets=10, inputCol="time", outputCol="time_bin")

joinedDf = discretizer.fit(joinedDf).transform(joinedDf)

display(joinedDf)

# COMMAND ----------

df_ha_corr = spark.createDataFrame(corr_matrix,columns_sensor)
df_ha_corr.show()

# COMMAND ----------

from pyspark.ml.stat import Correlation

# COMMAND ----------

# DBTITLE 1,EDA - Categorical variables
freq_table = df.select(col("target").cast("string")).groupBy("target").count().toPandas()

# COMMAND ----------

#to identify whether a DataFrame/Dataset has streaming data or not by using
df_ha.isStreaming()

# COMMAND ----------

words = ...  # streaming DataFrame of schema { timestamp: Timestamp, word: String }

# Group the data by window and word and compute the count of each group
windowedCounts = words.groupBy(
    window(words.timestamp, "10 minutes", "5 minutes"),
    words.word).count()

# COMMAND ----------



# COMMAND ----------

df_ha_w = df_ha.groupBy(
    window(col("timestamp"), "10 minutes", "5 minutes"), col("user_id"))
.agg(avg("acc_magnitude_mean").as("acc_magnitude_mean_avg"))
.orderBy(col("window.start"))