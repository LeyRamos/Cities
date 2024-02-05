# Databricks notebook source
# MAGIC %md ## Import all **feature label data** 
# MAGIC from [GitHub repository](https://github.com/LeyRamos/HA/raw/main/datasets/features_labels/)
# MAGIC
# MAGIC Data Pipeline:
# MAGIC 1. **HA_import_data** notebook --> import all data and create a parquet table as output (THIS!)
# MAGIC 2. **HA_data_transformation** notebook --> handle missing data and perform EDA
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
from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql.functions import sum,avg,max,min,mean,count

from pyspark.sql.types import *;
from scipy.stats import *
from scipy import stats

import gzip
# import StringIO --> ModuleNotFoundError: No module named 'StringIO'


from functools import reduce


# COMMAND ----------

# DBTITLE 1,Organize data storage
# Create folders to store the input data

dbutils.fs.mkdirs("/datasets/features_labels")
dbutils.fs.mkdirs("/datasets/original_labels")
dbutils.fs.mkdirs("/datasets/absolute_loaction")
dbutils.fs.mkdirs("/datasets/features_labels/input")
dbutils.fs.mkdirs("/datasets/features_labels/output")
dbutils.fs.mkdirs("/datasets/original_labels/input")
dbutils.fs.mkdirs("/datasets/absolute_loaction/input")

# COMMAND ----------

# DBTITLE 1,Functions
# Get a list of csv files in the Github repository main

def get_original_files(user, repo):
  import requests
  url = "https://api.github.com/repos/{}/{}/git/trees/main?recursive=1".format(user, repo)
  r = requests.get(url)
  res = r.json()

  orig_files = []

  for file in res["tree"]:
    if '.csv.gz' in file["path"]:
      orig_files.append(file["path"]) ## l.replace(subfolder, "") for l in file["path"]
  return orig_files

# =======================================
# Download files from a Githhub repository

def download_file(github_repo, dest_folder, file):
  import urllib
  import tempfile
  with urllib.request.urlopen(github_repo + file) as response:
    gzipcontent = response.read()
  
  # Persiste archivo en tmp
  with open('/tmp/'+ file, 'wb') as f:
    f.write(gzipcontent)
    
  # Copia archivo al path solicitado
  dbutils.fs.cp('file:/tmp/' + file, dest_folder + file)
 

# =======================================
# Read csv files and create a DataFrame

def dataframe_from_csvs(input_folder, input_files_list):
  input_files_list = [input_folder + "*.csv.gz" for l in input_files_list]
  li = []
  for filename in input_files_list:
    df_import = (
    spark.read.option("delimiter", ',').
    csv(filename,
        #schema=schema,
        header=True,
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
        nullValue='NA')
     )
    li.append(df_import)
    
    # Append all files imported in a single dataframe
  df_union = reduce(DataFrame.unionAll, li)
  return df_union


# =======================================
# PySpark Get Size and Shape of DataFrame

def sparkShape(dataFrame):
    from pyspark.sql import DataFrame
    return (dataFrame.count(), len(dataFrame.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape
#print(sparkDF.shape())


#import pandas as pd    
#spark.conf.set("spark.sql.execution.arrow.enabled", "true")
#pandasDF=sparkDF.toPandas()
#print(pandasDF.shape)

# COMMAND ----------

# DBTITLE 1,Import data
# List of input files in the repository
user = "LeyRamos"
repo = "HA"

#Load Feature label files
subfolder = "datasets/features_labels/"

original_files = []

for f in get_original_files(user, repo):
  if f.startswith(subfolder):
    temp= f.replace(subfolder, "")
    original_files.append(temp)

original_files

#Load Absolute location files
subfolder = "datasets/absolute_location/"

original_files_loc = []

for f in get_original_files(user, repo):
  if f.startswith(subfolder):
    temp= f.replace(subfolder, "")
    original_files_loc.append(temp)

original_files_loc

# COMMAND ----------

# Download input feature labels files from the repository and copy to cloud destination folder
github_repo = 'https://github.com/LeyRamos/HA/raw/main/datasets/features_labels/'
dest_folder = '/datasets/features_labels/input/'
file = ''

for f in original_files:
  download_file(github_repo, dest_folder, f)
  
# Download input absolute location files from the repository and copy to cloud destination folder
github_repo = 'https://github.com/LeyRamos/HA/raw/main/datasets/absolute_location/'
dest_folder = '/datasets/absolute_loaction/input/'
file = ''

for f in original_files_loc:
  download_file(github_repo, dest_folder, f)

# COMMAND ----------

# Check the files properly loaded
dbutils.fs.ls('/datasets/features_labels/input/')
dbutils.fs.ls('/datasets/absolute_loaction/input/')

# COMMAND ----------

# DBTITLE 1,Create dataframes
#Load all feature label files into a df
input_folder = '/datasets/features_labels/input/'
input_files_list = original_files
df = dataframe_from_csvs(input_folder, input_files_list)

input_folder = '/datasets/absolute_loaction/input/'
input_files_list = original_files_loc
df_loc = dataframe_from_csvs(input_folder, input_files_list)

# COMMAND ----------

#Load all csv files loaded into a single dataframe
#dest_folder = '/datasets/features_labels/input/'
#path = dest_folder # path='dbfs:/datasets/features_labels/input/
#original_files = [dest_folder + "*.csv.gz" for l in original_files]

#li = []

#for filename in original_files:
#    df_import = (
#    spark.read.option("delimiter", ',').
#    csv(filename,
#        #schema=schema,
#        header=True,
#        ignoreLeadingWhiteSpace=True,
#        ignoreTrailingWhiteSpace=True,
#        nullValue='NA')
#     )
#    li.append(df_import)
    
    # Append all files imported in a single dataframe
#    df = reduce(DataFrame.unionAll, li)

# COMMAND ----------

#Original dataframe dimensions
#print(df_loc.shape())

# COMMAND ----------

# DBTITLE 1,Initial dataframe conditioning - Sensor dataframe
# Get user ID in a new column from each file name 

# Get file name in a new column
df = df.withColumn("filename", input_file_name())
df_loc = df_loc.withColumn("filename", input_file_name())

# Get User ID from the filename column
df = df.withColumn("user_id", substring("filename", 38,36))

# Changing unix time format to datetime 
df = df.withColumn("timestamp", from_unixtime('timestamp', "yyyy-MM-dd HH:mm:ss"))

# Setting the correct time_zone = PST ('US/Pacific'), according to http://extrasensory.ucsd.edu/papers/vaizman2017a_pervasiveAcceptedVersion.pdf
df = df.withColumn("timestamp", from_utc_timestamp(df.timestamp, 'PST')) 

# Show values
df.select("user_id", "timestamp").show(n=3, vertical=True,truncate=100)

# COMMAND ----------

# Drop original columns not required (54)
columns_to_drop_startswith = ['audio_', 
                              'discrete:app_', 
                              'discrete:battery_', 
                              'discrete:on', 
                              'discrete:ringer', 
                              'discrete:wifi_',
                              'discrete:time_of_day', 
                              'lf_measurements', 
                              'label_source', 
                              'filename']

columns_to_drop = [i for i in df.columns if i.startswith(tuple(columns_to_drop_startswith))]
df= df.drop(*columns_to_drop)
#df.drop(*columns_to_drop).show(n=1, vertical=True,truncate=100)

# COMMAND ----------

# Create new label columns
df = df.withColumn("lab_vehicle", 
                   when((
                     (col("label:DRIVE_-_I_M_THE_DRIVER") == 1) | 
                     (col("label:IN_A_CAR") == 1) | 
                     (col("label:DRIVE_-_I_M_A_PASSENGER") == 1) | 
                     (col("label:ON_A_BUS") == 1)), 1)
                   .otherwise(0))   \
.withColumn("lab_bicycling", 
                   when((col("label:BICYCLING") == 1), 1)
                   .otherwise(0))   \
.withColumn("lab_walking", 
                   when((
                     (col("label:FIX_walking") == 1) |
                     (col("label:FIX_running") == 1) |
                     (col("label:STROLLING") == 1) |
                     (col("label:STAIRS_-_GOING_UP") == 1) |
                     (col("label:STAIRS_-_GOING_DOWN") == 1)), 1)
                   .otherwise(0))   \
.withColumn("lab_sitting", 
                   when((
                     (col("label:SITTING") == 1) |
                     (col("label:COMPUTER_WORK") == 1) |
                     (col("label:SURFING_THE_INTERNET") == 1) |
                     (col("label:EATING") == 1) |
                     (col("label:TOILET") == 1)), 1)
                   .otherwise(0))   \
.withColumn("lab_standing", 
                   when((
                     (col("label:PHONE_ON_TABLE") == 1) |
                     (col("label:LYING_DOWN") == 1) |
                     (col("label:SLEEPING") == 1) |
                     (col("label:OR_standing") == 1) |
                     (col("label:WATCHING_TV") == 1) |
                     (col("label:BATHING_-_SHOWER") == 1) |
                     (col("label:ELEVATOR") == 1)), 1)
                   .otherwise(0))   \
.withColumn("lab_no_traveling", 
                   when((
                     (col("label:OR_indoors") == 1) |
                     (col("label:LOC_home") == 1) |
                     (col("label:AT_SCHOOL") == 1) |
                     (col("label:LOC_main_workplace") == 1) |
                     (col("label:IN_CLASS") == 1) |
                     (col("label:IN_A_MEETING") == 1) |
                     (col("label:COOKING") == 1) |
                     (col("label:LAB_WORK") == 1) |
                     (col("label:CLEANING") == 1) |
                     (col("label:GROOMING") == 1) |
                     (col("label:DRESSING") == 1) |
                     (col("label:FIX_restaurant") == 1) |
                     (col("label:AT_A_PARTY") == 1) |
                     (col("label:WASHING_DISHES") == 1) |
                     (col("label:AT_THE_GYM") == 1) |
                     (col("label:LOC_beach") == 1) |
                     (col("label:DOING_LAUNDRY") == 1) |
                     (col("label:AT_A_BAR") == 1)), 1)
                   .otherwise(0))   \
.withColumn("lab_no_traveling_definition", 
                   when((
                     (col("label:TALKING") == 1) |
                     (col("label:WITH_FRIENDS") == 1) |
                     (col("label:PHONE_IN_POCKET") == 1) |
                     (col("label:PHONE_IN_HAND") == 1) |
                     (col("label:OR_outside") == 1) |
                     (col("label:PHONE_IN_BAG") == 1) |
                     (col("label:OR_exercise") == 1) |
                     (col("label:WITH_CO-WORKERS") == 1) |
                     (col("label:SHOPPING") == 1) |
                     (col("label:DRINKING__ALCOHOL_") == 1) |
                     (col("label:SINGING") == 1)), 1)
                   .otherwise(0)) 

# Drop original labels not required
labels_to_drop = [i for i in df.columns if i.startswith('label:')]
df = df.drop(*labels_to_drop)

# COMMAND ----------

# Rename column names

df = df.withColumnRenamed("timestamp","timestamp")  \
.withColumnRenamed("user_id","user_id")  \
.withColumnRenamed("raw_acc:magnitude_stats:mean","acc_magnitude_mean")  \
.withColumnRenamed("raw_acc:magnitude_stats:std","acc_magnitude_std")  \
.withColumnRenamed("raw_acc:magnitude_stats:moment3","acc_magnitude_moment3")  \
.withColumnRenamed("raw_acc:magnitude_stats:moment4","acc_magnitude_moment4")  \
.withColumnRenamed("raw_acc:magnitude_stats:percentile25","acc_magnitude_perc25")  \
.withColumnRenamed("raw_acc:magnitude_stats:percentile50","acc_magnitude_perc50")  \
.withColumnRenamed("raw_acc:magnitude_stats:percentile75","acc_magnitude_perc75")  \
.withColumnRenamed("raw_acc:magnitude_stats:value_entropy","acc_magnitude_value_entropy")  \
.withColumnRenamed("raw_acc:magnitude_stats:time_entropy","acc_magnitude_time_entropy")  \
.withColumnRenamed("raw_acc:magnitude_spectrum:log_energy_band0","acc_magnitude_spect_energy_band0")  \
.withColumnRenamed("raw_acc:magnitude_spectrum:log_energy_band1","acc_magnitude_spect_energy_band1")  \
.withColumnRenamed("raw_acc:magnitude_spectrum:log_energy_band2","acc_magnitude_spect_energy_band2")  \
.withColumnRenamed("raw_acc:magnitude_spectrum:log_energy_band3","acc_magnitude_spect_energy_band3")  \
.withColumnRenamed("raw_acc:magnitude_spectrum:log_energy_band4","acc_magnitude_spect_energy_band4")  \
.withColumnRenamed("raw_acc:magnitude_spectrum:spectral_entropy","acc_magnitude_spec_spectral_entropy")  \
.withColumnRenamed("raw_acc:magnitude_autocorrelation:period","acc_magnitude_autoc_period")  \
.withColumnRenamed("raw_acc:magnitude_autocorrelation:normalized_ac","acc_magnitude_autoc_normalized_ac")  \
.withColumnRenamed("raw_acc:3d:mean_x","acc_3d_mean_x")  \
.withColumnRenamed("raw_acc:3d:mean_y","acc_3d_mean_y")  \
.withColumnRenamed("raw_acc:3d:mean_z","acc_3d_mean_z")  \
.withColumnRenamed("raw_acc:3d:std_x","acc_3d_std_x")  \
.withColumnRenamed("raw_acc:3d:std_y","acc_3d_std_y")  \
.withColumnRenamed("raw_acc:3d:std_z","acc_3d_std_z")  \
.withColumnRenamed("raw_acc:3d:ro_xy","acc_3d_ro_x")  \
.withColumnRenamed("raw_acc:3d:ro_xz","acc_3d_ro_y")  \
.withColumnRenamed("raw_acc:3d:ro_yz","acc_3d_ro_z")  \
.withColumnRenamed("proc_gyro:magnitude_stats:mean","gyro_magnitude_mean")  \
.withColumnRenamed("proc_gyro:magnitude_stats:std","gyro_magnitude_std")  \
.withColumnRenamed("proc_gyro:magnitude_stats:moment3","gyro_magnitude_moment3")  \
.withColumnRenamed("proc_gyro:magnitude_stats:moment4","gyro_magnitude_moment4")  \
.withColumnRenamed("proc_gyro:magnitude_stats:percentile25","gyro_magnitude_perc25")  \
.withColumnRenamed("proc_gyro:magnitude_stats:percentile50","gyro_magnitude_perc50")  \
.withColumnRenamed("proc_gyro:magnitude_stats:percentile75","gyro_magnitude_perc75")  \
.withColumnRenamed("proc_gyro:magnitude_stats:value_entropy","gyro_magnitude_value_entropy")  \
.withColumnRenamed("proc_gyro:magnitude_stats:time_entropy","gyro_magnitude_time_entropy")  \
.withColumnRenamed("proc_gyro:magnitude_spectrum:log_energy_band0","gyro_magnitude_spect_energy_band0")  \
.withColumnRenamed("proc_gyro:magnitude_spectrum:log_energy_band1","gyro_magnitude_spect_energy_band1")  \
.withColumnRenamed("proc_gyro:magnitude_spectrum:log_energy_band2","gyro_magnitude_spect_energy_band2")  \
.withColumnRenamed("proc_gyro:magnitude_spectrum:log_energy_band3","gyro_magnitude_spect_energy_band3")  \
.withColumnRenamed("proc_gyro:magnitude_spectrum:log_energy_band4","gyro_magnitude_spect_energy_band4")  \
.withColumnRenamed("proc_gyro:magnitude_spectrum:spectral_entropy","gyro_magnitude_spec_spectral_entropy")  \
.withColumnRenamed("proc_gyro:magnitude_autocorrelation:period","gyro_magnitude_autoc_period")  \
.withColumnRenamed("proc_gyro:magnitude_autocorrelation:normalized_ac","gyro_magnitude_autoc_normalized_ac")  \
.withColumnRenamed("proc_gyro:3d:mean_x","gyro_3d_mean_x")  \
.withColumnRenamed("proc_gyro:3d:mean_y","gyro_3d_mean_y")  \
.withColumnRenamed("proc_gyro:3d:mean_z","gyro_3d_mean_z")  \
.withColumnRenamed("proc_gyro:3d:std_x","gyro_3d_std_x")  \
.withColumnRenamed("proc_gyro:3d:std_y","gyro_3d_std_y")  \
.withColumnRenamed("proc_gyro:3d:std_z","gyro_3d_std_z")  \
.withColumnRenamed("proc_gyro:3d:ro_xy","gyro_3d_ro_xy")  \
.withColumnRenamed("proc_gyro:3d:ro_xz","gyro_3d_ro_xz")  \
.withColumnRenamed("proc_gyro:3d:ro_yz","gyro_3d_ro_yz")  \
.withColumnRenamed("raw_magnet:magnitude_stats:mean","magnet_magnitude_mean")  \
.withColumnRenamed("raw_magnet:magnitude_stats:std","magnet_magnitude_std")  \
.withColumnRenamed("raw_magnet:magnitude_stats:moment3","magnet_magnitude_moment3")  \
.withColumnRenamed("raw_magnet:magnitude_stats:moment4","magnet_magnitude_moment4")  \
.withColumnRenamed("raw_magnet:magnitude_stats:percentile25","magnet_magnitude_perc25")  \
.withColumnRenamed("raw_magnet:magnitude_stats:percentile50","magnet_magnitude_perc50")  \
.withColumnRenamed("raw_magnet:magnitude_stats:percentile75","magnet_magnitude_perc75")  \
.withColumnRenamed("raw_magnet:magnitude_stats:value_entropy","magnet_magnitude_value_entropy")  \
.withColumnRenamed("raw_magnet:magnitude_stats:time_entropy","magnet_magnitude_time_entropy")  \
.withColumnRenamed("raw_magnet:magnitude_spectrum:log_energy_band0","magnet_magnitude_spect_energy_band0")  \
.withColumnRenamed("raw_magnet:magnitude_spectrum:log_energy_band1","magnet_magnitude_spect_energy_band1")  \
.withColumnRenamed("raw_magnet:magnitude_spectrum:log_energy_band2","magnet_magnitude_spect_energy_band2")  \
.withColumnRenamed("raw_magnet:magnitude_spectrum:log_energy_band3","magnet_magnitude_spect_energy_band3")  \
.withColumnRenamed("raw_magnet:magnitude_spectrum:log_energy_band4","magnet_magnitude_spect_energy_band4")  \
.withColumnRenamed("raw_magnet:magnitude_spectrum:spectral_entropy","magnet_magnitude_spec_spectral_entropy")  \
.withColumnRenamed("raw_magnet:magnitude_autocorrelation:period","magnet_magnitude_autoc_period")  \
.withColumnRenamed("raw_magnet:magnitude_autocorrelation:normalized_ac","magnet_magnitude_autoc_normalized_ac")  \
.withColumnRenamed("raw_magnet:3d:mean_x","magnet_3d_mean_x")  \
.withColumnRenamed("raw_magnet:3d:mean_y","magnet_3d_mean_y")  \
.withColumnRenamed("raw_magnet:3d:mean_z","magnet_3d_mean_z")  \
.withColumnRenamed("raw_magnet:3d:std_x","magnet_3d_std_x")  \
.withColumnRenamed("raw_magnet:3d:std_y","magnet_3d_std_y")  \
.withColumnRenamed("raw_magnet:3d:std_z","magnet_3d_std_z")  \
.withColumnRenamed("raw_magnet:3d:ro_xy","magnet_3d_ro_xy")  \
.withColumnRenamed("raw_magnet:3d:ro_xz","magnet_3d_ro_xz")  \
.withColumnRenamed("raw_magnet:3d:ro_yz","magnet_3d_ro_yz")  \
.withColumnRenamed("raw_magnet:avr_cosine_similarity_lag_range0","magnet_avr_cosine_similarity_lag0")  \
.withColumnRenamed("raw_magnet:avr_cosine_similarity_lag_range1","magnet_avr_cosine_similarity_lag1")  \
.withColumnRenamed("raw_magnet:avr_cosine_similarity_lag_range2","magnet_avr_cosine_similarity_lag2")  \
.withColumnRenamed("raw_magnet:avr_cosine_similarity_lag_range3","magnet_avr_cosine_similarity_lag3")  \
.withColumnRenamed("raw_magnet:avr_cosine_similarity_lag_range4","magnet_avr_cosine_similarity_lag4")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:mean","acc_watch_magnitude_mean")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:std","acc_watch_magnitude_std")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:moment3","acc_watch_magnitude_moment3")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:moment4","acc_watch_magnitude_moment4")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:percentile25","acc_watch_magnitude_perc25")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:percentile50","acc_watch_magnitude_perc50")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:percentile75","acc_watch_magnitude_perc75")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:value_entropy","acc_watch_magnitude_value_entropy")  \
.withColumnRenamed("watch_acceleration:magnitude_stats:time_entropy","acc_watch_magnitude_time_entropy")  \
.withColumnRenamed("watch_acceleration:magnitude_spectrum:log_energy_band0","acc_watch_magnitude_spect_energy_band0")  \
.withColumnRenamed("watch_acceleration:magnitude_spectrum:log_energy_band1","acc_watch_magnitude_spect_energy_band1")  \
.withColumnRenamed("watch_acceleration:magnitude_spectrum:log_energy_band2","acc_watch_magnitude_spect_energy_band2")  \
.withColumnRenamed("watch_acceleration:magnitude_spectrum:log_energy_band3","acc_watch_magnitude_spect_energy_band3")  \
.withColumnRenamed("watch_acceleration:magnitude_spectrum:log_energy_band4","acc_watch_magnitude_spect_energy_band4")  \
.withColumnRenamed("watch_acceleration:magnitude_spectrum:spectral_entropy","acc_watch_magnitude_spec_spectral_entropy")  \
.withColumnRenamed("watch_acceleration:magnitude_autocorrelation:period","acc_watch_magnitude_autoc_period")  \
.withColumnRenamed("watch_acceleration:magnitude_autocorrelation:normalized_ac","acc_watch_magnitude_autoc_normalized_ac")  \
.withColumnRenamed("watch_acceleration:3d:mean_x","acc_watch_3d_mean_x")  \
.withColumnRenamed("watch_acceleration:3d:mean_y","acc_watch_3d_mean_y")  \
.withColumnRenamed("watch_acceleration:3d:mean_z","acc_watch_3d_mean_z")  \
.withColumnRenamed("watch_acceleration:3d:std_x","acc_watch_3d_std_x")  \
.withColumnRenamed("watch_acceleration:3d:std_y","acc_watch_3d_std_y")  \
.withColumnRenamed("watch_acceleration:3d:std_z","acc_watch_3d_std_z")  \
.withColumnRenamed("watch_acceleration:3d:ro_xy","acc_watch_3d_ro_xy")  \
.withColumnRenamed("watch_acceleration:3d:ro_xz","acc_watch_3d_ro_xz")  \
.withColumnRenamed("watch_acceleration:3d:ro_yz","acc_watch_3d_ro_yz")  \
.withColumnRenamed("watch_acceleration:spectrum:x_log_energy_band0","acc_watch_spec_x_energy_band0")  \
.withColumnRenamed("watch_acceleration:spectrum:x_log_energy_band1","acc_watch_spec_x_energy_band1")  \
.withColumnRenamed("watch_acceleration:spectrum:x_log_energy_band2","acc_watch_spec_x_energy_band2")  \
.withColumnRenamed("watch_acceleration:spectrum:x_log_energy_band3","acc_watch_spec_x_energy_band3")  \
.withColumnRenamed("watch_acceleration:spectrum:x_log_energy_band4","acc_watch_spec_x_energy_band4")  \
.withColumnRenamed("watch_acceleration:spectrum:y_log_energy_band0","acc_watch_spec_y_energy_band0")  \
.withColumnRenamed("watch_acceleration:spectrum:y_log_energy_band1","acc_watch_spec_y_energy_band1")  \
.withColumnRenamed("watch_acceleration:spectrum:y_log_energy_band2","acc_watch_spec_y_energy_band2")  \
.withColumnRenamed("watch_acceleration:spectrum:y_log_energy_band3","acc_watch_spec_y_energy_band3")  \
.withColumnRenamed("watch_acceleration:spectrum:y_log_energy_band4","acc_watch_spec_y_energy_band4")  \
.withColumnRenamed("watch_acceleration:spectrum:z_log_energy_band0","acc_watch_spec_z_energy_band0")  \
.withColumnRenamed("watch_acceleration:spectrum:z_log_energy_band1","acc_watch_spec_z_energy_band1")  \
.withColumnRenamed("watch_acceleration:spectrum:z_log_energy_band2","acc_watch_spec_z_energy_band2")  \
.withColumnRenamed("watch_acceleration:spectrum:z_log_energy_band3","acc_watch_spec_z_energy_band3")  \
.withColumnRenamed("watch_acceleration:spectrum:z_log_energy_band4","acc_watch_spec_z_energy_band4")  \
.withColumnRenamed("watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0","acc_watch__avr_cosine_similarity_lag0")  \
.withColumnRenamed("watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1","acc_watch__avr_cosine_similarity_lag1")  \
.withColumnRenamed("watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2","acc_watch__avr_cosine_similarity_lag2")  \
.withColumnRenamed("watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3","acc_watch__avr_cosine_similarity_lag3")  \
.withColumnRenamed("watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4","acc_watch__avr_cosine_similarity_lag4")  \
.withColumnRenamed("watch_heading:mean_cos","acc_watch_head_men_cos")  \
.withColumnRenamed("watch_heading:std_cos","acc_watch_head_std_cos")  \
.withColumnRenamed("watch_heading:mom3_cos","acc_watch_head_mom3_cos")  \
.withColumnRenamed("watch_heading:mom4_cos","acc_watch_head_mom4_cos")  \
.withColumnRenamed("watch_heading:mean_sin","acc_watch_head_men_sin")  \
.withColumnRenamed("watch_heading:std_sin","acc_watch_head_std_sin")  \
.withColumnRenamed("watch_heading:mom3_sin","acc_watch_head_mom3_sin")  \
.withColumnRenamed("watch_heading:mom4_sin","acc_watch_head_mom4_sin")  \
.withColumnRenamed("watch_heading:entropy_8bins","acc_watch_head_entropy_8bins")  \
.withColumnRenamed("location:num_valid_updates","loc_valid_updates")  \
.withColumnRenamed("location:log_latitude_range","loc_log_latitude_range")  \
.withColumnRenamed("location:log_longitude_range","loc_log_longitude_range")  \
.withColumnRenamed("location:min_altitude","loc_min_altitude")  \
.withColumnRenamed("location:max_altitude","loc_max_altitude")  \
.withColumnRenamed("location:min_speed","loc_min_speed")  \
.withColumnRenamed("location:max_speed","loc_max_speed")  \
.withColumnRenamed("location:best_horizontal_accuracy","loc_best_horizontal_accuracy")  \
.withColumnRenamed("location:best_vertical_accuracy","loc_best_vertical_accuracy")  \
.withColumnRenamed("location:diameter","loc_diameter")  \
.withColumnRenamed("location:log_diameter","loc_log_diameter")  \
.withColumnRenamed("location_quick_features:std_lat","loc_features_std_lat")  \
.withColumnRenamed("location_quick_features:std_long","loc_features_std_long")  \
.withColumnRenamed("location_quick_features:lat_change","loc_features_lat_change")  \
.withColumnRenamed("location_quick_features:long_change","loc_features_log_change")  \
.withColumnRenamed("location_quick_features:mean_abs_lat_deriv","loc_features_mean_abs_lat_deriv")  \
.withColumnRenamed("location_quick_features:mean_abs_long_deriv","loc_features_mean_abs_long_deriv")  

# COMMAND ----------

# Change numeric data types

df = df.withColumn("acc_magnitude_mean", df.acc_magnitude_mean.cast(DoubleType()))   \
.withColumn("acc_magnitude_std", df.acc_magnitude_std.cast(DoubleType()))   \
.withColumn("acc_magnitude_moment3", df.acc_magnitude_moment3.cast(DoubleType()))   \
.withColumn("acc_magnitude_moment4", df.acc_magnitude_moment4.cast(DoubleType()))   \
.withColumn("acc_magnitude_perc25", df.acc_magnitude_perc25.cast(DoubleType()))   \
.withColumn("acc_magnitude_perc50", df.acc_magnitude_perc50.cast(DoubleType()))   \
.withColumn("acc_magnitude_perc75", df.acc_magnitude_perc75.cast(DoubleType()))   \
.withColumn("acc_magnitude_value_entropy", df.acc_magnitude_value_entropy.cast(DoubleType()))   \
.withColumn("acc_magnitude_time_entropy", df.acc_magnitude_time_entropy.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band0", df.acc_magnitude_spect_energy_band0.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band1", df.acc_magnitude_spect_energy_band1.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band2", df.acc_magnitude_spect_energy_band2.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band3", df.acc_magnitude_spect_energy_band3.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band4", df.acc_magnitude_spect_energy_band4.cast(DoubleType()))   \
.withColumn("acc_magnitude_spec_spectral_entropy", df.acc_magnitude_spec_spectral_entropy.cast(DoubleType()))   \
.withColumn("acc_magnitude_autoc_period", df.acc_magnitude_autoc_period.cast(DoubleType()))   \
.withColumn("acc_magnitude_autoc_normalized_ac", df.acc_magnitude_autoc_normalized_ac.cast(DoubleType()))   \
.withColumn("acc_3d_mean_x", df.acc_3d_mean_x.cast(DoubleType()))   \
.withColumn("acc_3d_mean_y", df.acc_3d_mean_y.cast(DoubleType()))   \
.withColumn("acc_3d_mean_z", df.acc_3d_mean_z.cast(DoubleType()))   \
.withColumn("acc_3d_std_x", df.acc_3d_std_x.cast(DoubleType()))   \
.withColumn("acc_3d_std_y", df.acc_3d_std_y.cast(DoubleType()))   \
.withColumn("acc_3d_std_z", df.acc_3d_std_z.cast(DoubleType()))   \
.withColumn("acc_3d_ro_x", df.acc_3d_ro_x.cast(DoubleType()))   \
.withColumn("acc_3d_ro_y", df.acc_3d_ro_y.cast(DoubleType()))   \
.withColumn("acc_3d_ro_z", df.acc_3d_ro_z.cast(DoubleType()))   \
.withColumn("gyro_magnitude_mean", df.gyro_magnitude_mean.cast(DoubleType()))   \
.withColumn("gyro_magnitude_std", df.gyro_magnitude_std.cast(DoubleType()))   \
.withColumn("gyro_magnitude_moment3", df.gyro_magnitude_moment3.cast(DoubleType()))   \
.withColumn("gyro_magnitude_moment4", df.gyro_magnitude_moment4.cast(DoubleType()))   \
.withColumn("gyro_magnitude_perc25", df.gyro_magnitude_perc25.cast(DoubleType()))   \
.withColumn("gyro_magnitude_perc50", df.gyro_magnitude_perc50.cast(DoubleType()))   \
.withColumn("gyro_magnitude_perc75", df.gyro_magnitude_perc75.cast(DoubleType()))   \
.withColumn("gyro_magnitude_value_entropy", df.gyro_magnitude_value_entropy.cast(DoubleType()))   \
.withColumn("gyro_magnitude_time_entropy", df.gyro_magnitude_time_entropy.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band0", df.gyro_magnitude_spect_energy_band0.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band1", df.gyro_magnitude_spect_energy_band1.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band2", df.gyro_magnitude_spect_energy_band2.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band3", df.gyro_magnitude_spect_energy_band3.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band4", df.gyro_magnitude_spect_energy_band4.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spec_spectral_entropy", df.gyro_magnitude_spec_spectral_entropy.cast(DoubleType()))   \
.withColumn("gyro_magnitude_autoc_period", df.gyro_magnitude_autoc_period.cast(DoubleType()))   \
.withColumn("gyro_magnitude_autoc_normalized_ac", df.gyro_magnitude_autoc_normalized_ac.cast(DoubleType()))   \
.withColumn("gyro_3d_mean_x", df.gyro_3d_mean_x.cast(DoubleType()))   \
.withColumn("gyro_3d_mean_y", df.gyro_3d_mean_y.cast(DoubleType()))   \
.withColumn("gyro_3d_mean_z", df.gyro_3d_mean_z.cast(DoubleType()))   \
.withColumn("gyro_3d_std_x", df.gyro_3d_std_x.cast(DoubleType()))   \
.withColumn("gyro_3d_std_y", df.gyro_3d_std_y.cast(DoubleType()))   \
.withColumn("gyro_3d_std_z", df.gyro_3d_std_z.cast(DoubleType()))   \
.withColumn("gyro_3d_ro_xy", df.gyro_3d_ro_xy.cast(DoubleType()))   \
.withColumn("gyro_3d_ro_xz", df.gyro_3d_ro_xz.cast(DoubleType()))   \
.withColumn("gyro_3d_ro_yz", df.gyro_3d_ro_yz.cast(DoubleType()))   \
.withColumn("magnet_magnitude_mean", df.magnet_magnitude_mean.cast(DoubleType()))   \
.withColumn("magnet_magnitude_std", df.magnet_magnitude_std.cast(DoubleType()))   \
.withColumn("magnet_magnitude_moment3", df.magnet_magnitude_moment3.cast(DoubleType()))   \
.withColumn("magnet_magnitude_moment4", df.magnet_magnitude_moment4.cast(DoubleType()))   \
.withColumn("magnet_magnitude_perc25", df.magnet_magnitude_perc25.cast(DoubleType()))   \
.withColumn("magnet_magnitude_perc50", df.magnet_magnitude_perc50.cast(DoubleType()))   \
.withColumn("magnet_magnitude_perc75", df.magnet_magnitude_perc75.cast(DoubleType()))   \
.withColumn("magnet_magnitude_value_entropy", df.magnet_magnitude_value_entropy.cast(DoubleType()))   \
.withColumn("magnet_magnitude_time_entropy", df.magnet_magnitude_time_entropy.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band0", df.magnet_magnitude_spect_energy_band0.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band1", df.magnet_magnitude_spect_energy_band1.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band2", df.magnet_magnitude_spect_energy_band2.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band3", df.magnet_magnitude_spect_energy_band3.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band4", df.magnet_magnitude_spect_energy_band4.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spec_spectral_entropy", df.magnet_magnitude_spec_spectral_entropy.cast(DoubleType()))   \
.withColumn("magnet_magnitude_autoc_period", df.magnet_magnitude_autoc_period.cast(DoubleType()))   \
.withColumn("magnet_magnitude_autoc_normalized_ac", df.magnet_magnitude_autoc_normalized_ac.cast(DoubleType()))   \
.withColumn("magnet_3d_mean_x", df.magnet_3d_mean_x.cast(DoubleType()))   \
.withColumn("magnet_3d_mean_y", df.magnet_3d_mean_y.cast(DoubleType()))   \
.withColumn("magnet_3d_mean_z", df.magnet_3d_mean_z.cast(DoubleType()))   \
.withColumn("magnet_3d_std_x", df.magnet_3d_std_x.cast(DoubleType()))   \
.withColumn("magnet_3d_std_y", df.magnet_3d_std_y.cast(DoubleType()))   \
.withColumn("magnet_3d_std_z", df.magnet_3d_std_z.cast(DoubleType()))   \
.withColumn("magnet_3d_ro_xy", df.magnet_3d_ro_xy.cast(DoubleType()))   \
.withColumn("magnet_3d_ro_xz", df.magnet_3d_ro_xz.cast(DoubleType()))   \
.withColumn("magnet_3d_ro_yz", df.magnet_3d_ro_yz.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag0", df.magnet_avr_cosine_similarity_lag0.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag1", df.magnet_avr_cosine_similarity_lag1.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag2", df.magnet_avr_cosine_similarity_lag2.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag3", df.magnet_avr_cosine_similarity_lag3.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag4", df.magnet_avr_cosine_similarity_lag4.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_mean", df.acc_watch_magnitude_mean.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_std", df.acc_watch_magnitude_std.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_moment3", df.acc_watch_magnitude_moment3.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_moment4", df.acc_watch_magnitude_moment4.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_perc25", df.acc_watch_magnitude_perc25.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_perc50", df.acc_watch_magnitude_perc50.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_perc75", df.acc_watch_magnitude_perc75.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_value_entropy", df.acc_watch_magnitude_value_entropy.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_time_entropy", df.acc_watch_magnitude_time_entropy.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band0", df.acc_watch_magnitude_spect_energy_band0.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band1", df.acc_watch_magnitude_spect_energy_band1.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band2", df.acc_watch_magnitude_spect_energy_band2.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band3", df.acc_watch_magnitude_spect_energy_band3.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band4", df.acc_watch_magnitude_spect_energy_band4.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spec_spectral_entropy", df.acc_watch_magnitude_spec_spectral_entropy.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_autoc_period", df.acc_watch_magnitude_autoc_period.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_autoc_normalized_ac", df.acc_watch_magnitude_autoc_normalized_ac.cast(DoubleType()))   \
.withColumn("acc_watch_3d_mean_x", df.acc_watch_3d_mean_x.cast(DoubleType()))   \
.withColumn("acc_watch_3d_mean_y", df.acc_watch_3d_mean_y.cast(DoubleType()))   \
.withColumn("acc_watch_3d_mean_z", df.acc_watch_3d_mean_z.cast(DoubleType()))   \
.withColumn("acc_watch_3d_std_x", df.acc_watch_3d_std_x.cast(DoubleType()))   \
.withColumn("acc_watch_3d_std_y", df.acc_watch_3d_std_y.cast(DoubleType()))   \
.withColumn("acc_watch_3d_std_z", df.acc_watch_3d_std_z.cast(DoubleType()))   \
.withColumn("acc_watch_3d_ro_xy", df.acc_watch_3d_ro_xy.cast(DoubleType()))   \
.withColumn("acc_watch_3d_ro_xz", df.acc_watch_3d_ro_xz.cast(DoubleType()))   \
.withColumn("acc_watch_3d_ro_yz", df.acc_watch_3d_ro_yz.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band0", df.acc_watch_spec_x_energy_band0.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band1", df.acc_watch_spec_x_energy_band1.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band2", df.acc_watch_spec_x_energy_band2.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band3", df.acc_watch_spec_x_energy_band3.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band4", df.acc_watch_spec_x_energy_band4.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band0", df.acc_watch_spec_y_energy_band0.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band1", df.acc_watch_spec_y_energy_band1.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band2", df.acc_watch_spec_y_energy_band2.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band3", df.acc_watch_spec_y_energy_band3.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band4", df.acc_watch_spec_y_energy_band4.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band0", df.acc_watch_spec_z_energy_band0.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band1", df.acc_watch_spec_z_energy_band1.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band2", df.acc_watch_spec_z_energy_band2.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band3", df.acc_watch_spec_z_energy_band3.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band4", df.acc_watch_spec_z_energy_band4.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag0", df.acc_watch__avr_cosine_similarity_lag0.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag1", df.acc_watch__avr_cosine_similarity_lag1.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag2", df.acc_watch__avr_cosine_similarity_lag2.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag3", df.acc_watch__avr_cosine_similarity_lag3.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag4", df.acc_watch__avr_cosine_similarity_lag4.cast(DoubleType()))   \
.withColumn("acc_watch_head_men_cos", df.acc_watch_head_men_cos.cast(DoubleType()))   \
.withColumn("acc_watch_head_std_cos", df.acc_watch_head_std_cos.cast(DoubleType()))   \
.withColumn("acc_watch_head_mom3_cos", df.acc_watch_head_mom3_cos.cast(DoubleType()))   \
.withColumn("acc_watch_head_mom4_cos", df.acc_watch_head_mom4_cos.cast(DoubleType()))   \
.withColumn("acc_watch_head_men_sin", df.acc_watch_head_men_sin.cast(DoubleType()))   \
.withColumn("acc_watch_head_std_sin", df.acc_watch_head_std_sin.cast(DoubleType()))   \
.withColumn("acc_watch_head_mom3_sin", df.acc_watch_head_mom3_sin.cast(DoubleType()))   \
.withColumn("acc_watch_head_mom4_sin", df.acc_watch_head_mom4_sin.cast(DoubleType()))   \
.withColumn("acc_watch_head_entropy_8bins", df.acc_watch_head_entropy_8bins.cast(DoubleType()))   \
.withColumn("loc_valid_updates", df.loc_valid_updates.cast(IntegerType()))   \
.withColumn("loc_log_latitude_range", df.loc_log_latitude_range.cast(DoubleType()))   \
.withColumn("loc_log_longitude_range", df.loc_log_longitude_range.cast(DoubleType()))   \
.withColumn("loc_min_altitude", df.loc_min_altitude.cast(DoubleType()))   \
.withColumn("loc_max_altitude", df.loc_max_altitude.cast(DoubleType()))   \
.withColumn("loc_min_speed", df.loc_min_speed.cast(DoubleType()))   \
.withColumn("loc_max_speed", df.loc_max_speed.cast(DoubleType()))   \
.withColumn("loc_best_horizontal_accuracy", df.loc_best_horizontal_accuracy.cast(DoubleType()))   \
.withColumn("loc_best_vertical_accuracy", df.loc_best_vertical_accuracy.cast(DoubleType()))   \
.withColumn("loc_diameter", df.loc_diameter.cast(DoubleType()))   \
.withColumn("loc_log_diameter", df.loc_log_diameter.cast(DoubleType()))   \
.withColumn("loc_features_std_lat", df.loc_features_std_lat.cast(DoubleType()))   \
.withColumn("loc_features_std_long", df.loc_features_std_long.cast(DoubleType()))   \
.withColumn("loc_features_lat_change", df.loc_features_lat_change.cast(DoubleType()))   \
.withColumn("loc_features_log_change", df.loc_features_log_change.cast(DoubleType()))   \
.withColumn("loc_features_mean_abs_lat_deriv", df.loc_features_mean_abs_lat_deriv.cast(DoubleType()))   \
.withColumn("loc_features_mean_abs_long_deriv", df.loc_features_mean_abs_long_deriv.cast(DoubleType())) 
.withColumn("lab_vehicle", df.lab_vehicle.cast(IntegerType()))   \
.withColumn("lab_bicycling", df.lab_bicycling.cast(IntegerType()))   \
.withColumn("lab_walking", df.lab_walking.cast(IntegerType()))   \
.withColumn("lab_sitting", df.lab_sitting.cast(IntegerType()))   \
.withColumn("lab_standing", df.lab_standing.cast(IntegerType()))   \
.withColumn("lab_no_traveling", df.lab_no_traveling.cast(IntegerType()))   \
.withColumn("lab_no_traveling_definition", df.lab_no_traveling_definition.cast(IntegerType())) 

# COMMAND ----------

# df.show(n=1, vertical=True,truncate=100)
display(df)

# COMMAND ----------

# DBTITLE 1,Initial dataframe conditioning - Location dataframe
# Get user ID in a new column from each file name 

# Get file name in a new column
df_loc = df_loc.withColumn("filename", input_file_name())

# Get User ID from the filename column
df_loc = df_loc.withColumn("user_id", substring("filename", 40,36))
df_loc = df_loc.drop("filename")

# Rename columns
df_loc = df_loc.withColumnRenamed("timestamp","loc_timestamp")  \
.withColumnRenamed("user_id","loc_user_id")  \
.withColumnRenamed("latitude","loc_latitude")  \
.withColumnRenamed("longitude","loc_longitude")

# Change data types
df_loc = df_loc.withColumn("loc_latitude", df_loc.loc_latitude.cast(DoubleType()))   \
.withColumn("loc_longitude", df_loc.loc_longitude.cast(DoubleType()))

# Changing unix time format to datetime 
df_loc = df_loc.withColumn("loc_timestamp", from_unixtime('loc_timestamp', "yyyy-MM-dd HH:mm:ss"))

# Setting the correct time_zone = PST ('US/Pacific'), according to http://extrasensory.ucsd.edu/papers/vaizman2017a_pervasiveAcceptedVersion.pdf
df_loc = df_loc.withColumn("loc_timestamp", from_utc_timestamp(df_loc.loc_timestamp, 'PST')) 

# Show values
df_loc.select("loc_user_id", "loc_timestamp").show(n=3, vertical=True)

# COMMAND ----------

display(df_loc)

# COMMAND ----------

# DBTITLE 1,Join Sensor and location data
# Join sensor data (df) and location data (df_loc)
df_all = df.join(df_loc, (df.user_id == df_loc.loc_user_id) & (df.timestamp == df_loc.loc_timestamp), 'inner')
df_all = df_all.drop("loc_user_id", "loc_timestamp")

#display(df_all)

# Columns to drop after missing data analysis
columns_to_drop_2 = [
  "acc_watch_magnitude_spect_energy_band4",
  "acc_watch_spec_x_energy_band4",
  "acc_watch_spec_y_energy_band4",
  "acc_watch_spec_z_energy_band4",
  "acc_watch_head_entropy_8bins",
  "magnet_magnitude_autoc_period",
  "magnet_magnitude_autoc_normalized_ac"
]
df_all = df_all.drop(*columns_to_drop_2)

# COMMAND ----------

df_all = df_all.withColumn("acc_magnitude_mean", df_all.acc_magnitude_mean.cast(DoubleType()))   \
.withColumn("acc_magnitude_std", df_all.acc_magnitude_std.cast(DoubleType()))   \
.withColumn("acc_magnitude_moment3", df_all.acc_magnitude_moment3.cast(DoubleType()))   \
.withColumn("acc_magnitude_moment4", df_all.acc_magnitude_moment4.cast(DoubleType()))   \
.withColumn("acc_magnitude_perc25", df_all.acc_magnitude_perc25.cast(DoubleType()))   \
.withColumn("acc_magnitude_perc50", df_all.acc_magnitude_perc50.cast(DoubleType()))   \
.withColumn("acc_magnitude_perc75", df_all.acc_magnitude_perc75.cast(DoubleType()))   \
.withColumn("acc_magnitude_value_entropy", df_all.acc_magnitude_value_entropy.cast(DoubleType()))   \
.withColumn("acc_magnitude_time_entropy", df_all.acc_magnitude_time_entropy.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band0", df_all.acc_magnitude_spect_energy_band0.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band1", df_all.acc_magnitude_spect_energy_band1.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band2", df_all.acc_magnitude_spect_energy_band2.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band3", df_all.acc_magnitude_spect_energy_band3.cast(DoubleType()))   \
.withColumn("acc_magnitude_spect_energy_band4", df_all.acc_magnitude_spect_energy_band4.cast(DoubleType()))   \
.withColumn("acc_magnitude_spec_spectral_entropy", df_all.acc_magnitude_spec_spectral_entropy.cast(DoubleType()))   \
.withColumn("acc_magnitude_autoc_period", df_all.acc_magnitude_autoc_period.cast(DoubleType()))   \
.withColumn("acc_magnitude_autoc_normalized_ac", df_all.acc_magnitude_autoc_normalized_ac.cast(DoubleType()))   \
.withColumn("acc_3d_mean_x", df_all.acc_3d_mean_x.cast(DoubleType()))   \
.withColumn("acc_3d_mean_y", df_all.acc_3d_mean_y.cast(DoubleType()))   \
.withColumn("acc_3d_mean_z", df_all.acc_3d_mean_z.cast(DoubleType()))   \
.withColumn("acc_3d_std_x", df_all.acc_3d_std_x.cast(DoubleType()))   \
.withColumn("acc_3d_std_y", df_all.acc_3d_std_y.cast(DoubleType()))   \
.withColumn("acc_3d_std_z", df_all.acc_3d_std_z.cast(DoubleType()))   \
.withColumn("acc_3d_ro_x", df_all.acc_3d_ro_x.cast(DoubleType()))   \
.withColumn("acc_3d_ro_y", df_all.acc_3d_ro_y.cast(DoubleType()))   \
.withColumn("acc_3d_ro_z", df_all.acc_3d_ro_z.cast(DoubleType()))   \
.withColumn("gyro_magnitude_mean", df_all.gyro_magnitude_mean.cast(DoubleType()))   \
.withColumn("gyro_magnitude_std", df_all.gyro_magnitude_std.cast(DoubleType()))   \
.withColumn("gyro_magnitude_moment3", df_all.gyro_magnitude_moment3.cast(DoubleType()))   \
.withColumn("gyro_magnitude_moment4", df_all.gyro_magnitude_moment4.cast(DoubleType()))   \
.withColumn("gyro_magnitude_perc25", df_all.gyro_magnitude_perc25.cast(DoubleType()))   \
.withColumn("gyro_magnitude_perc50", df_all.gyro_magnitude_perc50.cast(DoubleType()))   \
.withColumn("gyro_magnitude_perc75", df_all.gyro_magnitude_perc75.cast(DoubleType()))   \
.withColumn("gyro_magnitude_value_entropy", df_all.gyro_magnitude_value_entropy.cast(DoubleType()))   \
.withColumn("gyro_magnitude_time_entropy", df_all.gyro_magnitude_time_entropy.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band0", df_all.gyro_magnitude_spect_energy_band0.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band1", df_all.gyro_magnitude_spect_energy_band1.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band2", df_all.gyro_magnitude_spect_energy_band2.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band3", df_all.gyro_magnitude_spect_energy_band3.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spect_energy_band4", df_all.gyro_magnitude_spect_energy_band4.cast(DoubleType()))   \
.withColumn("gyro_magnitude_spec_spectral_entropy", df_all.gyro_magnitude_spec_spectral_entropy.cast(DoubleType()))   \
.withColumn("gyro_magnitude_autoc_period", df_all.gyro_magnitude_autoc_period.cast(DoubleType()))   \
.withColumn("gyro_magnitude_autoc_normalized_ac", df_all.gyro_magnitude_autoc_normalized_ac.cast(DoubleType()))   \
.withColumn("gyro_3d_mean_x", df_all.gyro_3d_mean_x.cast(DoubleType()))   \
.withColumn("gyro_3d_mean_y", df_all.gyro_3d_mean_y.cast(DoubleType()))   \
.withColumn("gyro_3d_mean_z", df_all.gyro_3d_mean_z.cast(DoubleType()))   \
.withColumn("gyro_3d_std_x", df_all.gyro_3d_std_x.cast(DoubleType()))   \
.withColumn("gyro_3d_std_y", df_all.gyro_3d_std_y.cast(DoubleType()))   \
.withColumn("gyro_3d_std_z", df_all.gyro_3d_std_z.cast(DoubleType()))   \
.withColumn("gyro_3d_ro_xy", df_all.gyro_3d_ro_xy.cast(DoubleType()))   \
.withColumn("gyro_3d_ro_xz", df_all.gyro_3d_ro_xz.cast(DoubleType()))   \
.withColumn("gyro_3d_ro_yz", df_all.gyro_3d_ro_yz.cast(DoubleType()))   \
.withColumn("magnet_magnitude_mean", df_all.magnet_magnitude_mean.cast(DoubleType()))   \
.withColumn("magnet_magnitude_std", df_all.magnet_magnitude_std.cast(DoubleType()))   \
.withColumn("magnet_magnitude_moment3", df_all.magnet_magnitude_moment3.cast(DoubleType()))   \
.withColumn("magnet_magnitude_moment4", df_all.magnet_magnitude_moment4.cast(DoubleType()))   \
.withColumn("magnet_magnitude_perc25", df_all.magnet_magnitude_perc25.cast(DoubleType()))   \
.withColumn("magnet_magnitude_perc50", df_all.magnet_magnitude_perc50.cast(DoubleType()))   \
.withColumn("magnet_magnitude_perc75", df_all.magnet_magnitude_perc75.cast(DoubleType()))   \
.withColumn("magnet_magnitude_value_entropy", df_all.magnet_magnitude_value_entropy.cast(DoubleType()))   \
.withColumn("magnet_magnitude_time_entropy", df_all.magnet_magnitude_time_entropy.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band0", df_all.magnet_magnitude_spect_energy_band0.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band1", df_all.magnet_magnitude_spect_energy_band1.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band2", df_all.magnet_magnitude_spect_energy_band2.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band3", df_all.magnet_magnitude_spect_energy_band3.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spect_energy_band4", df_all.magnet_magnitude_spect_energy_band4.cast(DoubleType()))   \
.withColumn("magnet_magnitude_spec_spectral_entropy", df_all.magnet_magnitude_spec_spectral_entropy.cast(DoubleType()))   \
.withColumn("magnet_3d_mean_x", df_all.magnet_3d_mean_x.cast(DoubleType()))   \
.withColumn("magnet_3d_mean_y", df_all.magnet_3d_mean_y.cast(DoubleType()))   \
.withColumn("magnet_3d_mean_z", df_all.magnet_3d_mean_z.cast(DoubleType()))   \
.withColumn("magnet_3d_std_x", df_all.magnet_3d_std_x.cast(DoubleType()))   \
.withColumn("magnet_3d_std_y", df_all.magnet_3d_std_y.cast(DoubleType()))   \
.withColumn("magnet_3d_std_z", df_all.magnet_3d_std_z.cast(DoubleType()))   \
.withColumn("magnet_3d_ro_xy", df_all.magnet_3d_ro_xy.cast(DoubleType()))   \
.withColumn("magnet_3d_ro_xz", df_all.magnet_3d_ro_xz.cast(DoubleType()))   \
.withColumn("magnet_3d_ro_yz", df_all.magnet_3d_ro_yz.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag0", df_all.magnet_avr_cosine_similarity_lag0.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag1", df_all.magnet_avr_cosine_similarity_lag1.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag2", df_all.magnet_avr_cosine_similarity_lag2.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag3", df_all.magnet_avr_cosine_similarity_lag3.cast(DoubleType()))   \
.withColumn("magnet_avr_cosine_similarity_lag4", df_all.magnet_avr_cosine_similarity_lag4.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_mean", df_all.acc_watch_magnitude_mean.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_std", df_all.acc_watch_magnitude_std.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_moment3", df_all.acc_watch_magnitude_moment3.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_moment4", df_all.acc_watch_magnitude_moment4.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_perc25", df_all.acc_watch_magnitude_perc25.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_perc50", df_all.acc_watch_magnitude_perc50.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_perc75", df_all.acc_watch_magnitude_perc75.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_value_entropy", df_all.acc_watch_magnitude_value_entropy.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_time_entropy", df_all.acc_watch_magnitude_time_entropy.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band0", df_all.acc_watch_magnitude_spect_energy_band0.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band1", df_all.acc_watch_magnitude_spect_energy_band1.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band2", df_all.acc_watch_magnitude_spect_energy_band2.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spect_energy_band3", df_all.acc_watch_magnitude_spect_energy_band3.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_spec_spectral_entropy", df_all.acc_watch_magnitude_spec_spectral_entropy.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_autoc_period", df_all.acc_watch_magnitude_autoc_period.cast(DoubleType()))   \
.withColumn("acc_watch_magnitude_autoc_normalized_ac", df_all.acc_watch_magnitude_autoc_normalized_ac.cast(DoubleType()))   \
.withColumn("acc_watch_3d_mean_x", df_all.acc_watch_3d_mean_x.cast(DoubleType()))   \
.withColumn("acc_watch_3d_mean_y", df_all.acc_watch_3d_mean_y.cast(DoubleType()))   \
.withColumn("acc_watch_3d_mean_z", df_all.acc_watch_3d_mean_z.cast(DoubleType()))   \
.withColumn("acc_watch_3d_std_x", df_all.acc_watch_3d_std_x.cast(DoubleType()))   \
.withColumn("acc_watch_3d_std_y", df_all.acc_watch_3d_std_y.cast(DoubleType()))   \
.withColumn("acc_watch_3d_std_z", df_all.acc_watch_3d_std_z.cast(DoubleType()))   \
.withColumn("acc_watch_3d_ro_xy", df_all.acc_watch_3d_ro_xy.cast(DoubleType()))   \
.withColumn("acc_watch_3d_ro_xz", df_all.acc_watch_3d_ro_xz.cast(DoubleType()))   \
.withColumn("acc_watch_3d_ro_yz", df_all.acc_watch_3d_ro_yz.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band0", df_all.acc_watch_spec_x_energy_band0.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band1", df_all.acc_watch_spec_x_energy_band1.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band2", df_all.acc_watch_spec_x_energy_band2.cast(DoubleType()))   \
.withColumn("acc_watch_spec_x_energy_band3", df_all.acc_watch_spec_x_energy_band3.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band0", df_all.acc_watch_spec_y_energy_band0.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band1", df_all.acc_watch_spec_y_energy_band1.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band2", df_all.acc_watch_spec_y_energy_band2.cast(DoubleType()))   \
.withColumn("acc_watch_spec_y_energy_band3", df_all.acc_watch_spec_y_energy_band3.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band0", df_all.acc_watch_spec_z_energy_band0.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band1", df_all.acc_watch_spec_z_energy_band1.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band2", df_all.acc_watch_spec_z_energy_band2.cast(DoubleType()))   \
.withColumn("acc_watch_spec_z_energy_band3", df_all.acc_watch_spec_z_energy_band3.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag0", df_all.acc_watch__avr_cosine_similarity_lag0.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag1", df_all.acc_watch__avr_cosine_similarity_lag1.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag2", df_all.acc_watch__avr_cosine_similarity_lag2.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag3", df_all.acc_watch__avr_cosine_similarity_lag3.cast(DoubleType()))   \
.withColumn("acc_watch__avr_cosine_similarity_lag4", df_all.acc_watch__avr_cosine_similarity_lag4.cast(DoubleType()))   \
.withColumn("acc_watch_head_men_cos", df_all.acc_watch_head_men_cos.cast(DoubleType()))   \
.withColumn("acc_watch_head_std_cos", df_all.acc_watch_head_std_cos.cast(DoubleType()))   \
.withColumn("acc_watch_head_mom3_cos", df_all.acc_watch_head_mom3_cos.cast(DoubleType()))   \
.withColumn("acc_watch_head_mom4_cos", df_all.acc_watch_head_mom4_cos.cast(DoubleType()))   \
.withColumn("acc_watch_head_men_sin", df_all.acc_watch_head_men_sin.cast(DoubleType()))   \
.withColumn("acc_watch_head_std_sin", df_all.acc_watch_head_std_sin.cast(DoubleType()))   \
.withColumn("acc_watch_head_mom3_sin", df_all.acc_watch_head_mom3_sin.cast(DoubleType()))   \
.withColumn("acc_watch_head_mom4_sin", df_all.acc_watch_head_mom4_sin.cast(DoubleType()))   \
.withColumn("loc_valid_updates", df_all.loc_valid_updates.cast(IntegerType()))   \
.withColumn("loc_latitude", df_all.loc_latitude.cast(DoubleType()))   \
.withColumn("loc_longitude", df_all.loc_longitude.cast(DoubleType()))   \
.withColumn("loc_log_latitude_range", df_all.loc_log_latitude_range.cast(DoubleType()))   \
.withColumn("loc_log_longitude_range", df_all.loc_log_longitude_range.cast(DoubleType()))   \
.withColumn("loc_min_altitude", df_all.loc_min_altitude.cast(DoubleType()))   \
.withColumn("loc_max_altitude", df_all.loc_max_altitude.cast(DoubleType()))   \
.withColumn("loc_min_speed", df_all.loc_min_speed.cast(DoubleType()))   \
.withColumn("loc_max_speed", df_all.loc_max_speed.cast(DoubleType()))   \
.withColumn("loc_best_horizontal_accuracy", df_all.loc_best_horizontal_accuracy.cast(DoubleType()))   \
.withColumn("loc_best_vertical_accuracy", df_all.loc_best_vertical_accuracy.cast(DoubleType()))   \
.withColumn("loc_diameter", df_all.loc_diameter.cast(DoubleType()))   \
.withColumn("loc_log_diameter", df_all.loc_log_diameter.cast(DoubleType()))   \
.withColumn("loc_features_std_lat", df_all.loc_features_std_lat.cast(DoubleType()))   \
.withColumn("loc_features_std_long", df_all.loc_features_std_long.cast(DoubleType()))   \
.withColumn("loc_features_lat_change", df_all.loc_features_lat_change.cast(DoubleType()))   \
.withColumn("loc_features_log_change", df_all.loc_features_log_change.cast(DoubleType()))   \
.withColumn("loc_features_mean_abs_lat_deriv", df_all.loc_features_mean_abs_lat_deriv.cast(DoubleType()))   \
.withColumn("loc_features_mean_abs_long_deriv", df_all.loc_features_mean_abs_long_deriv.cast(DoubleType()))    \
.withColumn("lab_vehicle", df_all.lab_vehicle.cast(IntegerType()))   \
.withColumn("lab_bicycling", df_all.lab_bicycling.cast(IntegerType()))   \
.withColumn("lab_walking", df_all.lab_walking.cast(IntegerType()))   \
.withColumn("lab_sitting", df_all.lab_sitting.cast(IntegerType()))   \
.withColumn("lab_standing", df_all.lab_standing.cast(IntegerType()))   \
.withColumn("lab_no_traveling", df_all.lab_no_traveling.cast(IntegerType()))   \
.withColumn("lab_no_traveling_definition", df_all.lab_no_traveling_definition.cast(IntegerType())) 


# COMMAND ----------

df_all_columns = [c for c in df_all.columns if c not in {'timestamp', 'user_id'}]
for column in df_all_columns:
    df_all = df_all.withColumn(column, when(isnan(col(column)), None).otherwise(col(column)))

# COMMAND ----------

# DBTITLE 1,Time windows aggregation
# Analysis of time ranges for avalilable data

df_all_timestamp = df_all.groupBy(col('user_id')).agg(
  min('timestamp').alias('timestamp_min'), 
  max('timestamp').alias('timestamp_max'),
  count(lit(1)).alias('no_records'))

df_all_timestamp = df_all_timestamp.withColumn('timestamp_dif_sec', col("timestamp_max").cast(DoubleType()) - col('timestamp_min').cast(DoubleType()))  \
.withColumn('timestamp_dif_min', round(col('timestamp_dif_sec')/60))  \
.withColumn('timestamp_dif_hour', round(col('timestamp_dif_sec')/3600))  \
.withColumn('timestamp_dif_day', round(col('timestamp_dif_sec')/86400))

display(df_all_timestamp)

# COMMAND ----------

# Analysis of time ranges for avalilable data

df_all_windows_labels = df_all_windows.groupBy().agg(
count(lit(1)).alias('no_windows'),
sum("no_records_sum").alias("no_records"),
sum("lab_vehicle_sum").alias("lab_vehicle_sum"),
sum("lab_bicycling_sum").alias("lab_bicycling_sum"),
sum("lab_walking_sum").alias("lab_walking_sum"),
sum("lab_sitting_sum").alias("lab_sitting_sum"),
sum("lab_standing_sum").alias("lab_standing_sum"),
sum("lab_no_traveling_sum").alias("lab_no_traveling_sum"),
sum("lab_no_traveling_definition_sum").alias("lab_no_traveling_definition_sum"))

df_all_windows_labels = df_all_windows_labels.withColumn("lab_vehicle_perc", col("lab_vehicle_sum")/col("no_records"))  \
.withColumn("lab_bicycling_perc", col("lab_bicycling_sum")/col("no_records"))  \
.withColumn("lab_walking_perc", col("lab_walking_sum")/col("no_records"))  \
.withColumn("lab_sitting_perc", col("lab_sitting_sum")/col("no_records"))  \
.withColumn("lab_standing_perc", col("lab_standing_sum")/col("no_records"))  \
.withColumn("lab_no_traveling_perc", col("lab_no_traveling_sum")/col("no_records"))  \
.withColumn("lab_no_traveling_definition_perc", col("lab_no_traveling_definition_sum")/col("no_records"))  

display(df_all_windows_labels)

# COMMAND ----------

df_all = df_all.filter(df_all.user_id == "0BFC35E2-4817-4865-BFA7-764742302A2D")

# COMMAND ----------

# Group data by time tumbling window of 1 minute
# Tumbling window : where the 2 consecutive windows are non-overlapping.
df_all_windows = df_all.groupBy(window(col("timestamp"), "1 minutes"), col("user_id")).agg(
count(lit(1)).alias('no_records'),
mean("acc_magnitude_mean").alias("acc_magnitude_mean_mean"),
mean("acc_magnitude_std").alias("acc_magnitude_std_mean"),
mean("acc_magnitude_moment3").alias("acc_magnitude_moment3_mean"),
mean("acc_magnitude_moment4").alias("acc_magnitude_moment4_mean"),
mean("acc_magnitude_perc25").alias("acc_magnitude_perc25_mean"),
mean("acc_magnitude_perc50").alias("acc_magnitude_perc50_mean"),
mean("acc_magnitude_perc75").alias("acc_magnitude_perc75_mean"),
mean("acc_magnitude_value_entropy").alias("acc_magnitude_value_entropy_mean"),
mean("acc_magnitude_time_entropy").alias("acc_magnitude_time_entropy_mean"),
mean("acc_magnitude_spect_energy_band0").alias("acc_magnitude_spect_energy_band0_mean"),
mean("acc_magnitude_spect_energy_band1").alias("acc_magnitude_spect_energy_band1_mean"),
mean("acc_magnitude_spect_energy_band2").alias("acc_magnitude_spect_energy_band2_mean"),
mean("acc_magnitude_spect_energy_band3").alias("acc_magnitude_spect_energy_band3_mean"),
mean("acc_magnitude_spect_energy_band4").alias("acc_magnitude_spect_energy_band4_mean"),
mean("acc_magnitude_spec_spectral_entropy").alias("acc_magnitude_spec_spectral_entropy_mean"),
mean("acc_magnitude_autoc_period").alias("acc_magnitude_autoc_period_mean"),
mean("acc_magnitude_autoc_normalized_ac").alias("acc_magnitude_autoc_normalized_ac_mean"),
mean("acc_3d_mean_x").alias("acc_3d_mean_x_mean"),
mean("acc_3d_mean_y").alias("acc_3d_mean_y_mean"),
mean("acc_3d_mean_z").alias("acc_3d_mean_z_mean"),
mean("acc_3d_std_x").alias("acc_3d_std_x_mean"),
mean("acc_3d_std_y").alias("acc_3d_std_y_mean"),
mean("acc_3d_std_z").alias("acc_3d_std_z_mean"),
mean("acc_3d_ro_x").alias("acc_3d_ro_x_mean"),
mean("acc_3d_ro_y").alias("acc_3d_ro_y_mean"),
mean("acc_3d_ro_z").alias("acc_3d_ro_z_mean"),
mean("acc_watch_magnitude_mean").alias("acc_watch_magnitude_mean_mean"),
mean("acc_watch_magnitude_std").alias("acc_watch_magnitude_std_mean"),
mean("acc_watch_magnitude_moment3").alias("acc_watch_magnitude_moment3_mean"),
mean("acc_watch_magnitude_moment4").alias("acc_watch_magnitude_moment4_mean"),
mean("acc_watch_magnitude_perc25").alias("acc_watch_magnitude_perc25_mean"),
mean("acc_watch_magnitude_perc50").alias("acc_watch_magnitude_perc50_mean"),
mean("acc_watch_magnitude_perc75").alias("acc_watch_magnitude_perc75_mean"),
mean("acc_watch_magnitude_value_entropy").alias("acc_watch_magnitude_value_entropy_mean"),
mean("acc_watch_magnitude_time_entropy").alias("acc_watch_magnitude_time_entropy_mean"),
mean("acc_watch_magnitude_spect_energy_band0").alias("acc_watch_magnitude_spect_energy_band0_mean"),
mean("acc_watch_magnitude_spect_energy_band1").alias("acc_watch_magnitude_spect_energy_band1_mean"),
mean("acc_watch_magnitude_spect_energy_band2").alias("acc_watch_magnitude_spect_energy_band2_mean"),
mean("acc_watch_magnitude_spect_energy_band3").alias("acc_watch_magnitude_spect_energy_band3_mean"),
mean("acc_watch_magnitude_spec_spectral_entropy").alias("acc_watch_magnitude_spec_spectral_entropy_mean"),
mean("acc_watch_magnitude_autoc_period").alias("acc_watch_magnitude_autoc_period_mean"),
mean("acc_watch_magnitude_autoc_normalized_ac").alias("acc_watch_magnitude_autoc_normalized_ac_mean"),
mean("acc_watch_3d_mean_x").alias("acc_watch_3d_mean_x_mean"),
mean("acc_watch_3d_mean_y").alias("acc_watch_3d_mean_y_mean"),
mean("acc_watch_3d_mean_z").alias("acc_watch_3d_mean_z_mean"),
mean("acc_watch_3d_std_x").alias("acc_watch_3d_std_x_mean"),
mean("acc_watch_3d_std_y").alias("acc_watch_3d_std_y_mean"),
mean("acc_watch_3d_std_z").alias("acc_watch_3d_std_z_mean"),
mean("acc_watch_3d_ro_xy").alias("acc_watch_3d_ro_xy_mean"),
mean("acc_watch_3d_ro_xz").alias("acc_watch_3d_ro_xz_mean"),
mean("acc_watch_3d_ro_yz").alias("acc_watch_3d_ro_yz_mean"),
mean("acc_watch_spec_x_energy_band0").alias("acc_watch_spec_x_energy_band0_mean"),
mean("acc_watch_spec_x_energy_band1").alias("acc_watch_spec_x_energy_band1_mean"),
mean("acc_watch_spec_x_energy_band2").alias("acc_watch_spec_x_energy_band2_mean"),
mean("acc_watch_spec_x_energy_band3").alias("acc_watch_spec_x_energy_band3_mean"),
mean("acc_watch_spec_y_energy_band0").alias("acc_watch_spec_y_energy_band0_mean"),
mean("acc_watch_spec_y_energy_band1").alias("acc_watch_spec_y_energy_band1_mean"),
mean("acc_watch_spec_y_energy_band2").alias("acc_watch_spec_y_energy_band2_mean"),
mean("acc_watch_spec_y_energy_band3").alias("acc_watch_spec_y_energy_band3_mean"),
mean("acc_watch_spec_z_energy_band0").alias("acc_watch_spec_z_energy_band0_mean"),
mean("acc_watch_spec_z_energy_band1").alias("acc_watch_spec_z_energy_band1_mean"),
mean("acc_watch_spec_z_energy_band2").alias("acc_watch_spec_z_energy_band2_mean"),
mean("acc_watch_spec_z_energy_band3").alias("acc_watch_spec_z_energy_band3_mean"),
mean("acc_watch__avr_cosine_similarity_lag0").alias("acc_watch__avr_cosine_similarity_lag0_mean"),
mean("acc_watch__avr_cosine_similarity_lag1").alias("acc_watch__avr_cosine_similarity_lag1_mean"),
mean("acc_watch__avr_cosine_similarity_lag2").alias("acc_watch__avr_cosine_similarity_lag2_mean"),
mean("acc_watch__avr_cosine_similarity_lag3").alias("acc_watch__avr_cosine_similarity_lag3_mean"),
mean("acc_watch__avr_cosine_similarity_lag4").alias("acc_watch__avr_cosine_similarity_lag4_mean"),
mean("acc_watch_head_men_cos").alias("acc_watch_head_men_cos_mean"),
mean("acc_watch_head_std_cos").alias("acc_watch_head_std_cos_mean"),
mean("acc_watch_head_mom3_cos").alias("acc_watch_head_mom3_cos_mean"),
mean("acc_watch_head_mom4_cos").alias("acc_watch_head_mom4_cos_mean"),
mean("acc_watch_head_men_sin").alias("acc_watch_head_men_sin_mean"),
mean("acc_watch_head_std_sin").alias("acc_watch_head_std_sin_mean"),
mean("acc_watch_head_mom3_sin").alias("acc_watch_head_mom3_sin_mean"),
mean("acc_watch_head_mom4_sin").alias("acc_watch_head_mom4_sin_mean"),
mean("gyro_magnitude_mean").alias("gyro_magnitude_mean_mean"),
mean("gyro_magnitude_std").alias("gyro_magnitude_std_mean"),
mean("gyro_magnitude_moment3").alias("gyro_magnitude_moment3_mean"),
mean("gyro_magnitude_moment4").alias("gyro_magnitude_moment4_mean"),
mean("gyro_magnitude_perc25").alias("gyro_magnitude_perc25_mean"),
mean("gyro_magnitude_perc50").alias("gyro_magnitude_perc50_mean"),
mean("gyro_magnitude_perc75").alias("gyro_magnitude_perc75_mean"),
mean("gyro_magnitude_value_entropy").alias("gyro_magnitude_value_entropy_mean"),
mean("gyro_magnitude_time_entropy").alias("gyro_magnitude_time_entropy_mean"),
mean("gyro_magnitude_spect_energy_band0").alias("gyro_magnitude_spect_energy_band0_mean"),
mean("gyro_magnitude_spect_energy_band1").alias("gyro_magnitude_spect_energy_band1_mean"),
mean("gyro_magnitude_spect_energy_band2").alias("gyro_magnitude_spect_energy_band2_mean"),
mean("gyro_magnitude_spect_energy_band3").alias("gyro_magnitude_spect_energy_band3_mean"),
mean("gyro_magnitude_spect_energy_band4").alias("gyro_magnitude_spect_energy_band4_mean"),
mean("gyro_magnitude_spec_spectral_entropy").alias("gyro_magnitude_spec_spectral_entropy_mean"),
mean("gyro_magnitude_autoc_period").alias("gyro_magnitude_autoc_period_mean"),
mean("gyro_magnitude_autoc_normalized_ac").alias("gyro_magnitude_autoc_normalized_ac_mean"),
mean("gyro_3d_mean_x").alias("gyro_3d_mean_x_mean"),
mean("gyro_3d_mean_y").alias("gyro_3d_mean_y_mean"),
mean("gyro_3d_mean_z").alias("gyro_3d_mean_z_mean"),
mean("gyro_3d_std_x").alias("gyro_3d_std_x_mean"),
mean("gyro_3d_std_y").alias("gyro_3d_std_y_mean"),
mean("gyro_3d_std_z").alias("gyro_3d_std_z_mean"),
mean("gyro_3d_ro_xy").alias("gyro_3d_ro_xy_mean"),
mean("gyro_3d_ro_xz").alias("gyro_3d_ro_xz_mean"),
mean("gyro_3d_ro_yz").alias("gyro_3d_ro_yz_mean"),
mean("magnet_magnitude_mean").alias("magnet_magnitude_mean_mean"),
mean("magnet_magnitude_std").alias("magnet_magnitude_std_mean"),
mean("magnet_magnitude_moment3").alias("magnet_magnitude_moment3_mean"),
mean("magnet_magnitude_moment4").alias("magnet_magnitude_moment4_mean"),
mean("magnet_magnitude_perc25").alias("magnet_magnitude_perc25_mean"),
mean("magnet_magnitude_perc50").alias("magnet_magnitude_perc50_mean"),
mean("magnet_magnitude_perc75").alias("magnet_magnitude_perc75_mean"),
mean("magnet_magnitude_value_entropy").alias("magnet_magnitude_value_entropy_mean"),
mean("magnet_magnitude_time_entropy").alias("magnet_magnitude_time_entropy_mean"),
mean("magnet_magnitude_spect_energy_band0").alias("magnet_magnitude_spect_energy_band0_mean"),
mean("magnet_magnitude_spect_energy_band1").alias("magnet_magnitude_spect_energy_band1_mean"),
mean("magnet_magnitude_spect_energy_band2").alias("magnet_magnitude_spect_energy_band2_mean"),
mean("magnet_magnitude_spect_energy_band3").alias("magnet_magnitude_spect_energy_band3_mean"),
mean("magnet_magnitude_spect_energy_band4").alias("magnet_magnitude_spect_energy_band4_mean"),
mean("magnet_magnitude_spec_spectral_entropy").alias("magnet_magnitude_spec_spectral_entropy_mean"),
mean("magnet_3d_mean_x").alias("magnet_3d_mean_x_mean"),
mean("magnet_3d_mean_y").alias("magnet_3d_mean_y_mean"),
mean("magnet_3d_mean_z").alias("magnet_3d_mean_z_mean"),
mean("magnet_3d_std_x").alias("magnet_3d_std_x_mean"),
mean("magnet_3d_std_y").alias("magnet_3d_std_y_mean"),
mean("magnet_3d_std_z").alias("magnet_3d_std_z_mean"),
mean("magnet_3d_ro_xy").alias("magnet_3d_ro_xy_mean"),
mean("magnet_3d_ro_xz").alias("magnet_3d_ro_xz_mean"),
mean("magnet_3d_ro_yz").alias("magnet_3d_ro_yz_mean"),
mean("magnet_avr_cosine_similarity_lag0").alias("magnet_avr_cosine_similarity_lag0_mean"),
mean("magnet_avr_cosine_similarity_lag1").alias("magnet_avr_cosine_similarity_lag1_mean"),
mean("magnet_avr_cosine_similarity_lag2").alias("magnet_avr_cosine_similarity_lag2_mean"),
mean("magnet_avr_cosine_similarity_lag3").alias("magnet_avr_cosine_similarity_lag3_mean"),
mean("magnet_avr_cosine_similarity_lag4").alias("magnet_avr_cosine_similarity_lag4_mean"),  
sum("loc_valid_updates").alias("loc_valid_updates_sum"),
min("loc_latitude").alias("loc_latitude_min"),
max("loc_latitude").alias("loc_latitude_max"),
mean("loc_latitude").alias("loc_latitude_mean"),
min("loc_longitude").alias("loc_longitude_min"),
max("loc_longitude").alias("loc_longitude_max"),
mean("loc_longitude").alias("loc_longitude_mean"),
mean("loc_log_latitude_range").alias("loc_log_latitude_range_mean"),
mean("loc_log_longitude_range").alias("loc_log_longitude_range_mean"),
min("loc_min_altitude").alias("loc_min_altitude_min"),
max("loc_max_altitude").alias("loc_max_altitude_max"),
min("loc_min_speed").alias("loc_min_speed_min"),
max("loc_max_speed").alias("loc_max_speed_max"),
mean("loc_best_horizontal_accuracy").alias("loc_best_horizontal_accuracy_mean"),
mean("loc_best_vertical_accuracy").alias("loc_best_vertical_accuracy_mean"),
mean("loc_diameter").alias("loc_diameter_mean"),
mean("loc_log_diameter").alias("loc_log_diameter_mean"),
mean("loc_features_std_lat").alias("loc_features_std_lat_mean"),
mean("loc_features_std_long").alias("loc_features_std_long_mean"),
mean("loc_features_lat_change").alias("loc_features_lat_change_mean"),
mean("loc_features_log_change").alias("loc_features_log_change_mean"),
mean("loc_features_mean_abs_lat_deriv").alias("loc_features_mean_abs_lat_deriv_mean"),
mean("loc_features_mean_abs_long_deriv").alias("loc_features_mean_abs_long_deriv_mean"),
sum("lab_vehicle").alias("lab_vehicle_sum"),
sum("lab_bicycling").alias("lab_bicycling_sum"),
sum("lab_walking").alias("lab_walking_sum"),
sum("lab_sitting").alias("lab_sitting_sum"),
sum("lab_standing").alias("lab_standing_sum"),
sum("lab_no_traveling").alias("lab_no_traveling_sum"),
sum("lab_no_traveling_definition").alias("lab_no_traveling_definition_sum")).orderBy(col("window.start"))

# COMMAND ----------

df_all_windows = df_all_windows.select(
  col("user_id"),
  col("window.*"),
  col('no_records'),
  col("acc_magnitude_mean_mean"),
  col("acc_magnitude_std_mean"),
  col("acc_magnitude_moment3_mean"),
  col("acc_magnitude_moment4_mean"),
  col("acc_magnitude_perc25_mean"),
  col("acc_magnitude_perc50_mean"),
  col("acc_magnitude_perc75_mean"),
  col("acc_magnitude_value_entropy_mean"),
  col("acc_magnitude_time_entropy_mean"),
  col("acc_magnitude_spect_energy_band0_mean"),
  col("acc_magnitude_spect_energy_band1_mean"),
  col("acc_magnitude_spect_energy_band2_mean"),
  col("acc_magnitude_spect_energy_band3_mean"),
  col("acc_magnitude_spect_energy_band4_mean"),
  col("acc_magnitude_spec_spectral_entropy_mean"),
  col("acc_magnitude_autoc_period_mean"),
  col("acc_magnitude_autoc_normalized_ac_mean"),
  col("acc_3d_mean_x_mean"),
  col("acc_3d_mean_y_mean"),
  col("acc_3d_mean_z_mean"),
  col("acc_3d_std_x_mean"),
  col("acc_3d_std_y_mean"),
  col("acc_3d_std_z_mean"),
  col("acc_3d_ro_x_mean"),
  col("acc_3d_ro_y_mean"),
  col("acc_3d_ro_z_mean"),
  col("gyro_magnitude_mean_mean"),
  col("gyro_magnitude_std_mean"),
  col("gyro_magnitude_moment3_mean"),
  col("gyro_magnitude_moment4_mean"),
  col("gyro_magnitude_perc25_mean"),
  col("gyro_magnitude_perc50_mean"),
  col("gyro_magnitude_perc75_mean"),
  col("gyro_magnitude_value_entropy_mean"),
  col("gyro_magnitude_time_entropy_mean"),
  col("gyro_magnitude_spect_energy_band0_mean"),
  col("gyro_magnitude_spect_energy_band1_mean"),
  col("gyro_magnitude_spect_energy_band2_mean"),
  col("gyro_magnitude_spect_energy_band3_mean"),
  col("gyro_magnitude_spect_energy_band4_mean"),
  col("gyro_magnitude_spec_spectral_entropy_mean"),
  col("gyro_magnitude_autoc_period_mean"),
  col("gyro_magnitude_autoc_normalized_ac_mean"),
  col("gyro_3d_mean_x_mean"),
  col("gyro_3d_mean_y_mean"),
  col("gyro_3d_mean_z_mean"),
  col("gyro_3d_std_x_mean"),
  col("gyro_3d_std_y_mean"),
  col("gyro_3d_std_z_mean"),
  col("gyro_3d_ro_xy_mean"),
  col("gyro_3d_ro_xz_mean"),
  col("gyro_3d_ro_yz_mean"),
  col("magnet_magnitude_mean_mean"),
  col("magnet_magnitude_std_mean"),
  col("magnet_magnitude_moment3_mean"),
  col("magnet_magnitude_moment4_mean"),
  col("magnet_magnitude_perc25_mean"),
  col("magnet_magnitude_perc50_mean"),
  col("magnet_magnitude_perc75_mean"),
  col("magnet_magnitude_value_entropy_mean"),
  col("magnet_magnitude_time_entropy_mean"),
  col("magnet_magnitude_spect_energy_band0_mean"),
  col("magnet_magnitude_spect_energy_band1_mean"),
  col("magnet_magnitude_spect_energy_band2_mean"),
  col("magnet_magnitude_spect_energy_band3_mean"),
  col("magnet_magnitude_spect_energy_band4_mean"),
  col("magnet_magnitude_spec_spectral_entropy_mean"),
  col("magnet_3d_mean_x_mean"),
  col("magnet_3d_mean_y_mean"),
  col("magnet_3d_mean_z_mean"),
  col("magnet_3d_std_x_mean"),
  col("magnet_3d_std_y_mean"),
  col("magnet_3d_std_z_mean"),
  col("magnet_3d_ro_xy_mean"),
  col("magnet_3d_ro_xz_mean"),
  col("magnet_3d_ro_yz_mean"),
  col("magnet_avr_cosine_similarity_lag0_mean"),
  col("magnet_avr_cosine_similarity_lag1_mean"),
  col("magnet_avr_cosine_similarity_lag2_mean"),
  col("magnet_avr_cosine_similarity_lag3_mean"),
  col("magnet_avr_cosine_similarity_lag4_mean"),
  col("acc_watch_magnitude_mean_mean"),
  col("acc_watch_magnitude_std_mean"),
  col("acc_watch_magnitude_moment3_mean"),
  col("acc_watch_magnitude_moment4_mean"),
  col("acc_watch_magnitude_perc25_mean"),
  col("acc_watch_magnitude_perc50_mean"),
  col("acc_watch_magnitude_perc75_mean"),
  col("acc_watch_magnitude_value_entropy_mean"),
  col("acc_watch_magnitude_time_entropy_mean"),
  col("acc_watch_magnitude_spect_energy_band0_mean"),
  col("acc_watch_magnitude_spect_energy_band1_mean"),
  col("acc_watch_magnitude_spect_energy_band2_mean"),
  col("acc_watch_magnitude_spect_energy_band3_mean"),
  col("acc_watch_magnitude_spec_spectral_entropy_mean"),
  col("acc_watch_magnitude_autoc_period_mean"),
  col("acc_watch_magnitude_autoc_normalized_ac_mean"),
  col("acc_watch_3d_mean_x_mean"),
  col("acc_watch_3d_mean_y_mean"),
  col("acc_watch_3d_mean_z_mean"),
  col("acc_watch_3d_std_x_mean"),
  col("acc_watch_3d_std_y_mean"),
  col("acc_watch_3d_std_z_mean"),
  col("acc_watch_3d_ro_xy_mean"),
  col("acc_watch_3d_ro_xz_mean"),
  col("acc_watch_3d_ro_yz_mean"),
  col("acc_watch_spec_x_energy_band0_mean"),
  col("acc_watch_spec_x_energy_band1_mean"),
  col("acc_watch_spec_x_energy_band2_mean"),
  col("acc_watch_spec_x_energy_band3_mean"),
  col("acc_watch_spec_y_energy_band0_mean"),
  col("acc_watch_spec_y_energy_band1_mean"),
  col("acc_watch_spec_y_energy_band2_mean"),
  col("acc_watch_spec_y_energy_band3_mean"),
  col("acc_watch_spec_z_energy_band0_mean"),
  col("acc_watch_spec_z_energy_band1_mean"),
  col("acc_watch_spec_z_energy_band2_mean"),
  col("acc_watch_spec_z_energy_band3_mean"),
  col("acc_watch__avr_cosine_similarity_lag0_mean"),
  col("acc_watch__avr_cosine_similarity_lag1_mean"),
  col("acc_watch__avr_cosine_similarity_lag2_mean"),
  col("acc_watch__avr_cosine_similarity_lag3_mean"),
  col("acc_watch__avr_cosine_similarity_lag4_mean"),
  col("acc_watch_head_men_cos_mean"),
  col("acc_watch_head_std_cos_mean"),
  col("acc_watch_head_mom3_cos_mean"),
  col("acc_watch_head_mom4_cos_mean"),
  col("acc_watch_head_men_sin_mean"),
  col("acc_watch_head_std_sin_mean"),
  col("acc_watch_head_mom3_sin_mean"),
  col("acc_watch_head_mom4_sin_mean"),
  col("loc_valid_updates_sum"),
  col("loc_latitude_min"),
  col("loc_latitude_max"),
  col("loc_latitude_mean"),
  col("loc_longitude_min"),
  col("loc_longitude_max"),
  col("loc_longitude_mean"),
  col("loc_log_latitude_range_mean"),
  col("loc_log_longitude_range_mean"),
  col("loc_min_altitude_min"),
  col("loc_max_altitude_max"),
  col("loc_min_speed_min"),
  col("loc_max_speed_max"),
  col("loc_best_horizontal_accuracy_mean"),
  col("loc_best_vertical_accuracy_mean"),
  col("loc_diameter_mean"),
  col("loc_log_diameter_mean"),
  col("loc_features_std_lat_mean"),
  col("loc_features_std_long_mean"),
  col("loc_features_lat_change_mean"),
  col("loc_features_log_change_mean"),
  col("loc_features_mean_abs_lat_deriv_mean"),
  col("loc_features_mean_abs_long_deriv_mean"),
  col("lab_vehicle_sum"),
  col("lab_bicycling_sum"),
  col("lab_walking_sum"),
  col("lab_sitting_sum"),
  col("lab_standing_sum"),
  col("lab_no_traveling_sum"),
  col("lab_no_traveling_definition_sum")
  )

# COMMAND ----------

#Original dataframe dimensions
#print(df_all_windows.shape())

# COMMAND ----------

#display(df_all_windows)

# COMMAND ----------

dbutils.fs.mkdirs('/FileStore/df_all_windows/')

# COMMAND ----------

#3
df_all_windows.write.csv("/FileStore/df_all_windows/0BFC35E2-4817-4865-BFA7-764742302A2D.csv", 
                     mode = "overwrite",
                     compression = "gzip",
                     header = True)

# COMMAND ----------

#4
df_all_windows.filter(df_all_windows.user_id == "0BFC35E2-4817-4865-BFA7-764742302A2D").write.csv("/FileStore/df_all_windows/0BFC35E2-4817-4865-BFA7-764742302A2D.csv", 
                     mode = "overwrite",
                     compression = "gzip",
                     header = True)

# COMMAND ----------

#5
df_all_windows.filter(df_all_windows.user_id == "0E6184E1-90C0-48EE-B25A-F1ECB7B9714E").write.csv("/FileStore/df_all_windows/0E6184E1-90C0-48EE-B25A-F1ECB7B9714E.csv", 
                     mode = "overwrite",
                     compression = "gzip",
                     header = True)

# COMMAND ----------

#6
df_all_windows.filter(df_all_windows.user_id == "1155FF54-63D3-4AB2-9863-8385D0BD0A13").write.csv("/FileStore/df_all_windows/1155FF54-63D3-4AB2-9863-8385D0BD0A13.csv", 
                     mode = "overwrite",
                     compression = "gzip",
                     header = True)

# COMMAND ----------

#7
df_all_windows.filter(df_all_windows.user_id == "11B5EC4D-4133-4289-B475-4E737182A406").write.csv("/FileStore/df_all_windows/11B5EC4D-4133-4289-B475-4E737182A406.csv", 
                     mode = "overwrite",
                     compression = "gzip",
                     header = True)

# COMMAND ----------

#8
df_all_windows.filter(df_all_windows.user_id == "136562B6-95B2-483D-88DC-065F28409FD2").write.csv("/FileStore/df_all_windows/136562B6-95B2-483D-88DC-065F28409FD2.csv", 
                     mode = "overwrite",
                     compression = "gzip",
                     header = True)

# COMMAND ----------

#9
df_all_windows.filter(df_all_windows.user_id == "1538C99F-BA1E-4EFB-A949-6C7C47701B20").write.csv("/FileStore/df_all_windows/1538C99F-BA1E-4EFB-A949-6C7C47701B20.csv", 
                     mode = "overwrite",
                     compression = "gzip",
                     header = True)

# COMMAND ----------

#10
df_all_windows.filter(df_all_windows.user_id == "1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842").write.csv("/FileStore/df_all_windows/1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842.csv", 
                     mode = "overwrite",
                     compression = "gzip",
                     header = True)

# COMMAND ----------

dbutils.fs.rm('/FileStore/df_all_windows/0A986513-7828-4D53-AA1F-E02D6DF9561B.csv', True)

# COMMAND ----------

# DBTITLE 1,Clean storage
# Delete files from storage
dbutils.fs.rm('/datasets/features_labels/input/', True) # Files used to create df
dbutils.fs.rm('/datasets/absolute_location/input/', True)
#dbutils.fs.rm('/datasets/features_labels/output', True) # If parquet file exists, remove it

# Recreate folders
#dbutils.fs.mkdirs('/datasets/features_labels/input/')
#dbutils.fs.mkdirs('/datasets/absolute_location/input/')

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

# Create parquet table as output
dbutils.fs.mkdirs('/datasets/features_labels/output/HA_DATA')

df.write.saveAsTable("HA_DATA" , 
                     format = "parquet", 
                     mode = "overwrite",
                     partitionBy = "user_id", 
                     path = "/datasets/features_labels/output/HA_DATA")

#Metadata refreshing. Spark SQL caches Parquet metadata for better performance.
#spark.catalog.refreshTable("HA_DATA_ACC")

# COMMAND ----------

# DBTITLE 1,Create DB and Output table
# MAGIC %sql 
# MAGIC CREATE DATABASE IF NOT EXISTS HA_DB
# MAGIC LOCATION "/datasets/features_labels/output"

# COMMAND ----------

# Check if the DB creation is correct
dbutils.fs.mkdirs("/datasets/features_labels/output")

# COMMAND ----------

# MAGIC %sql
# MAGIC USE HA_DB

# COMMAND ----------

ids_columns = ['user_id', 'timestamp']
sensor_columns_acc = ids_columns + [i for i in df_all.columns if i.startswith('acc')]
sensor_columns_acc_watch = ids_columns + [i for i in df_all.columns if i.startswith('acc_watch')]
sensor_columns_gyro = ids_columns + [i for i in df_all.columns if i.startswith('gyro')]
sensor_columns_magnet = ids_columns + [i for i in df_all.columns if i.startswith('magnet')]
sensor_columns_loc = ids_columns + [i for i in df_all.columns if i.startswith('loc')]

# COMMAND ----------

# MAGIC %sql DESCRIBE HA_DATA

# COMMAND ----------

dbutils.fs.rm('/datasets/features_labels/output/HA_DATA', True)

# COMMAND ----------

df_all.write.partitionBy("user_id").mode('overwrite').parquet("/datasets/features_labels/output/df_all.parquet")
#df_all.select(sensor_columns_loc).write.parquet("/datasets/features_labels/output/df_all.parquet")