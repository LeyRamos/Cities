# Databricks notebook source
import pyspark
import numpy as np
import pandas as pd

import  seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

import pyspark
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyspark.sql.functions as f
from pyspark.sql.functions import *
from pyspark.sql.functions import input_file_name
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

# Check the files properly loaded
dbutils.fs.ls('dbfs:/FileStore/')

# COMMAND ----------

# Remove files
#dbutils.fs.rm('dbfs:/FileStore/questions_clnstm_2021.csv/', recurse=True)

# COMMAND ----------

# Input data from Databricks FileStore
df_cities_2018 = spark.read.csv('dbfs:/FileStore/2018_Full_Cities_Dataset.csv', header='true')
df_cities_2019 = spark.read.csv('dbfs:/FileStore/2019_Full_Cities_Dataset.csv', header='true')
df_cities_2020 = spark.read.csv('dbfs:/FileStore/2020_Full_Cities_Dataset.csv', header='true')
df_cities_2021 = spark.read.csv('dbfs:/FileStore/2021_Full_Cities_Dataset.csv', header='true')
df_cities_2022 = spark.read.csv('dbfs:/FileStore/2022_Full_Cities_Dataset.csv', header='true')


# COMMAND ----------

# Original Shape
def sparkShape(df):
    return (df.count(), len(df.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape

print(df_cities_2018.shape())
print(df_cities_2019.shape())
print(df_cities_2020.shape())
print(df_cities_2021.shape())
print(df_cities_2022.shape())

# COMMAND ----------

# DBTITLE 1,1) First Transformations
# Remove other columns and unify colum names
columns_to_drop1 = ["Year Reported to CDP", "Comments","File Name", "Last update"]

df_cities_2018 = df_cities_2018.drop(*columns_to_drop1)
df_cities_2019 = df_cities_2019.drop(*columns_to_drop1)
df_cities_2020 = df_cities_2020.drop(*columns_to_drop1)
df_cities_2021 = df_cities_2021.drop(*columns_to_drop1)

columns_to_drop2 = ["Reporting Authority", "Comments","File Name", "Last update", "City"]
df_cities_2022 = df_cities_2022.drop(*columns_to_drop2) \
  .withColumnRenamed('Organization Number','Account Number') \
  .withColumnRenamed('Organization Name','Organization') 

#.rename(columns = {'Organization Number':'Account Number', 'Organization Name':'Organization'})



# COMMAND ----------

display(stacked_df)

# COMMAND ----------

# DBTITLE 1,Pre-shaping dataset 2018 original
# Apply fixes on df_cities_2018 emissions questions
#df_cities_2018 = spark.read.csv('dbfs:/FileStore/2018_Full_Cities_Dataset.csv', header='true')
df_cities_2018_manualfix_q7_6 = spark.read.csv('dbfs:/FileStore/tables/questions_2018_emissions7_6_manuallyset.csv', header='true')

#columns_to_drop1 = ["Year Reported to CDP", "Comments","File Name", "Last update"]
#df_cities_2018 = df_cities_2018.drop(*columns_to_drop1)

# Seleccionar las filas con valor "7.6" en la columna q_id
df_cities_2018f = df_cities_2018.filter(~col("Question Number").like("%7.6%"))

df_cities_2018 = df_cities_2018f.union(df_cities_2018_manualfix_q7_6)

# COMMAND ----------

#Para llegar al export que uso arriba directamente / NO ejecutar (no hace falta)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Seleccionar las filas con valor "7.6" en la columna q_id
df_cities_2018_sp1 = df_cities_2018_sp.filter(col("q_id").like("%7.6%"))

df_cities_2018_sp2 = df_cities_2018_sp1.groupby('Account Number', 'q_id', 'q_c_r', 'Question Number','Question Name', 'Column Number', 'Column Name', 'Row Number', 'Row Name', 'Response Answer',).count()
display(df_cities_2018_sp2)

# Replace "." by "_" in column q_id in order to manage properly the resulting column names after pivoting ("." cause issues)
from pyspark.sql.functions import regexp_replace
df_cities_2018_sp2 = df_cities_2018_sp2.withColumn("q_id", regexp_replace("q_id", "\\.", "_"))

df_cities_2018_sp2 = df_cities_2018_sp2.withColumn("q_r", concat(col('Question Number'), col('Row Number')))

# Pivot
df_cities_2018_sp3 = df_cities_2018_sp2.groupBy('Account Number', "q_r").pivot("q_id").agg(first(col('Response Answer')))
display(df_cities_2018_sp3)

#Merge

df_cities_2018_sp3 = df_cities_2018_sp3.withColumn("7_6a_q", concat_ws("-", col('7_6a-1'), col('7_6a-2'), col('7_6a-3')))
df_cities_2018_sp3 = df_cities_2018_sp3.withColumn("7_6b_q", concat_ws("-", col('7_6b-1'), col('7_6b-2'), col('7_6b-3')))
df_cities_2018_sp3 = df_cities_2018_sp3.withColumn("7_6c_q", concat_ws("-", col('7_6c-1'), col('7_6c-2'), col('7_6c-3')))
df_cities_2018_sp3 = df_cities_2018_sp3.withColumn("7_6d_q", concat_ws("-", col('7_6d-1'), col('7_6d-2'), col('7_6d-3')))

df_cities_2018_sp3 = df_cities_2018_sp3.drop('7_6a-1','7_6a-2','7_6a-3', '7_6b-1', '7_6b-2', '7_6b-3', '7_6c-1','7_6c-2', '7_6c-3', '7_6d-1', '7_6d-2','7_6d-3')

#
df_cities_2018_sp4 = df_cities_2018_sp3.withColumn("7_6_q", concat(col('7_6a_q'), col('7_6b_q'), col('7_6c_q'), col('7_6d_q')))
df_cities_2018_sp4 = df_cities_2018_sp4.withColumn("Response", concat_ws("", col("7_6a-4"), col("7_6b-4"), col("7_6c-4"), col("7_6d-4")))

df_cities_2018_sp4 = df_cities_2018_sp4.drop('7_6a_q', '7_6b_q', '7_6c_q', '7_6d_q', '7_6a-4', '7_6b-4', '7_6c-4', '7_6d-4')
display(df_cities_2018_sp4)

# Pivot
df_cities_2018_sp5 = df_cities_2018_sp2.groupBy('Account Number', 'q_id', 'Question Name', "q_r").pivot("q_id").agg(first(col('Response Answer')))
display(df_cities_2018_sp5)

# COMMAND ----------

#pandasDF = pysparkDF.toPandas()

pd_cities_2018 = df_cities_2018.toPandas()
pd_cities_2019 = df_cities_2019.toPandas()
pd_cities_2020 = df_cities_2020.toPandas()
pd_cities_2021 = df_cities_2021.toPandas()
pd_cities_2022 = df_cities_2022.toPandas()

# 2.3 min

# COMMAND ----------

# Functions
# Add keys Question Number-Column Number y Question Number-Column Number-Row Number
def add_keys(df):
    df['q_c'] = df['Question Number'].astype(str) + '-' + df['Column Number'].astype(str)
    df['q_c_r'] = df['Question Number'].astype(str) + '-' + df['Column Number'].astype(str) + '-' + df['Row Number'].astype(str)
    return
    df
    
    #df = df.withColumn('q_c', sf.concat(sf.col('colname1'),sf.lit('_'), sf.col('colname2')))
    
# Add text Question Name-Column Name y Question Name-Column Name-Row Name
def add_full_qtext(df):
    df[['Column Name', 'Row Name']] = df[['Column Name', 'Row Name']].fillna('none')
    df['q_c_text'] = df['Question Name'].astype(str) + '-' + df['Column Name'].astype(str)
    df['q_c_r_text'] = df['Question Name'].astype(str) + '-' + df['Column Name'].astype(str) + '-' + df['Row Name'].astype(str)
    df['q_c_text'] = df['q_c_text'].str.lower()
    df['q_c_r_text'] = df['q_c_r_text'].str.lower() 
    df['q_id'] = df.apply(lambda x: x['q_c'] if x['Row Name'] == 'none' else x['q_c_r'], axis=1)
    df['q_text'] = df.apply(lambda x: x['q_c_text'] if x['Row Name'] == 'none' else x['q_c_r_text'], axis=1)
    df['q_type'] = df.apply(lambda x: 'Question-Column' if x['Row Name'] == 'none' else 'Question-Column-Row', axis=1)
    return
    df

# COMMAND ----------

# Add keys and concatenated text
df_list = (pd_cities_2018, pd_cities_2019, pd_cities_2020, pd_cities_2021, pd_cities_2022)

for df in df_list:
    add_keys(df)
    
for df in df_list:
    add_full_qtext(df)

# COMMAND ----------

pd_cities_2018.head(5)

# COMMAND ----------

# DBTITLE 1,Pre-shaping dataset 2022 original
from pyspark.sql.functions import col, first, expr
from pyspark.sql.functions import regexp_replace

df_cities_2022_sp =spark.createDataFrame(pd_cities_2022) 

#Filter rows corresponding to question 5.1a
df_cities_2022_sp_51a = df_cities_2022_sp.filter((col("Question Number") == '5.1a') | (col("Question Number").like("6.1%")) | (col("Question Number") == '7.1'))
df_cities_2022_sp_51a = df_cities_2022_sp_51a.withColumn('id_temp', concat_ws('_', col("Account Number"), col("Row Number")))

# Pivot table
df_cities_2022_sp_51a = df_cities_2022_sp_51a.withColumn("q_id", regexp_replace("q_id", "[\\.-]", "_"))

pivoted_2022 = df_cities_2022_sp_51a.groupBy("Account Number", "id_temp").pivot("q_id").agg(first('Response Answer'))
#display(pivoted_2022)

# Split columns according to target type
pivoted_2022 = pivoted_2022.withColumn("5_1a_3_1", when(col("5_1a_2") == "Base year emissions (absolute) target", col("5_1a_3")).otherwise(None))\
.withColumn("5_1a_7_1", when(col("5_1a_2") == "Base year emissions (absolute) target", col("5_1a_7")).otherwise(None))\
.withColumn("5_1a_9_1", when(col("5_1a_2") == "Base year emissions (absolute) target", col("5_1a_9")).otherwise(None))\
.withColumn("5_1a_12_1", when(col("5_1a_2") == "Base year emissions (absolute) target", col("5_1a_12")).otherwise(None))\
.withColumn("5_1a_10_1", when(col("5_1a_2") == "Base year emissions (absolute) target", col("5_1a_10")).otherwise(None))\
.withColumn("5_1a_13_1", when(col("5_1a_2") == "Base year emissions (absolute) target", col("5_1a_13")).otherwise(None))\
.withColumn("5_1a_14_1", when(col("5_1a_2") == "Base year emissions (absolute) target", col("5_1a_14")).otherwise(None))\
.withColumn("5_1a_1_1", when(col("5_1a_2") == "Base year emissions (absolute) target", col("5_1a_1")).otherwise(None))\
.withColumn("5_1a_3_2", when(col("5_1a_2") == "Fixed_level target", col("5_1a_3")).otherwise(None))\
.withColumn("5_1a_7_2", when(col("5_1a_2") == "Fixed_level target", col("5_1a_7")).otherwise(None))\
.withColumn("5_1a_12_2", when(col("5_1a_2") == "Fixed_level target", col("5_1a_12")).otherwise(None))\
.withColumn("5_1a_13_2", when(col("5_1a_2") == "Fixed_level target", col("5_1a_13")).otherwise(None))\
.withColumn("5_1a_14_2", when(col("5_1a_2") == "Fixed_level target", col("5_1a_14")).otherwise(None))\
.withColumn("5_1a_8_2", when(col("5_1a_2") == "Fixed_level target", col("5_1a_8")).otherwise(None))\
.withColumn("5_1a_16_2", when(col("5_1a_2") == "Fixed_level target", col("5_1a_16")).otherwise(None))\
.withColumn("5_1a_1_2", when(col("5_1a_2") == "Fixed_level target", col("5_1a_1")).otherwise(None))\
.withColumn("5_1a_30_3", when(col("5_1a_2") == "Base year intensity target based on emissions per capita", 'Metric tonnes of CO2e per capita').when(col("5_1a_2") == "Base year intensity target based on emissions per unit GDP", 'Metric tonnes of CO2e per unit GDP').otherwise(None))\
.withColumn("5_1a_3_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_3")).otherwise(None))\
.withColumn("5_1a_7_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_7")).otherwise(None))\
.withColumn("5_1a_9_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_9")).otherwise(None))\
.withColumn("5_1a_12_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_12")).otherwise(None))\
.withColumn("5_1a_10_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_10")).otherwise(None))\
.withColumn("5_1a_13_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_13")).otherwise(None))\
.withColumn("5_1a_14_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_14")).otherwise(None))\
.withColumn("5_1a_11_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_11")).otherwise(None))\
.withColumn("5_1a_1_3", when((col("5_1a_2") == "Base year intensity target based on emissions per capita") | (col("5_1a_2") == "Base year intensity target based on emissions per unit GDP"), col("5_1a_1")).otherwise(None))\
.withColumn("5_1a_3_4", when(col("5_1a_2") == "Baseline scenario target", col("5_1a_3")).otherwise(None))\
.withColumn("5_1a_7_4", when(col("5_1a_2") == "Baseline scenario target", col("5_1a_7")).otherwise(None))\
.withColumn("5_1a_9_4", when(col("5_1a_2") == "Baseline scenario target", col("5_1a_9")).otherwise(None))\
.withColumn("5_1a_12_4", when(col("5_1a_2") == "Baseline scenario target", col("5_1a_12")).otherwise(None))\
.withColumn("5_1a_10_4", when(col("5_1a_2") == "Baseline scenario target", col("5_1a_10")).otherwise(None))\
.withColumn("5_1a_13_4", when(col("5_1a_2") == "Baseline scenario target", col("5_1a_13")).otherwise(None))\
.withColumn("5_1a_14_4", when(col("5_1a_2") == "Baseline scenario target", col("5_1a_14")).otherwise(None))\
.withColumn("5_1a_1_4", when(col("5_1a_2") == "Baseline scenario target", col("5_1a_1")).otherwise(None))\
.withColumn("6_1_1_1", when(col("6_1_1") == "Commercial emissions reduction target", "Yes").otherwise(None))\
.withColumn("6_1_1_2", when(col("6_1_1") == "Municipal emissions reduction target", "Yes").otherwise(None))\
.withColumn("6_1_1_3", when(col("6_1_1") == "Residential buildings emissions reduction target", "Yes").otherwise(None))\
.withColumn("6_1_1_4", when(col("6_1_1") == "New buildings emissions reduction target", "Yes").otherwise(None))\
.withColumn("6_1_1_5", when(col("6_1_1") == "Increase energy efficiency of buildings (commercial buildings)", "Yes").otherwise(None))\
.withColumn("6_1_1_6", when(col("6_1_1") == "Increase energy efficiency of buildings (government-owned buildings)", "Yes").otherwise(None))\
.withColumn("6_1_1_7", when(col("6_1_1") == "Increase energy efficiency of buildings (residential buildings)", "Yes").otherwise(None))\
.withColumn("6_1_1_8", when(col("6_1_1") == "Increase energy efficiency of buildings (all buildings)", "Yes").otherwise(None))\
.withColumn("6_1_1_9", when(col("6_1_1").like("%renewable%"), "Yes").otherwise(None))\
.withColumn("6_1_1_10", when(col("6_1_1").like("%renewable%"), col("6_1_1")).otherwise(None))\
.withColumn("6_1_1_11", when(col("6_1_1").like("%renewable%"), col("6_1_5")).otherwise(None))\
.withColumn("6_1_1_12", when(col("6_1_1").like("%renewable%") | col("6_1_6").like('%Percentage (%)%'), col("6_1_8")).otherwise(None))\
.withColumn("6_1_1_13", when(col("6_1_1").like("%renewable%"), col("6_1_9")).otherwise(None))\
.withColumn("6_1_1_14", when(col("6_1_1").like("%renewable%"), col("6_1_12")).otherwise(None))

#Remove original q_ids
pivoted_2022 = pivoted_2022.drop('5_1a_1','5_1a_10','5_1a_11','5_1a_12','5_1a_13','5_1a_14','5_1a_15','5_1a_16','5_1a_17','5_1a_18','5_1a_19','5_1a_2','5_1a_20','5_1a_21','5_1a_22','5_1a_23','5_1a_3','5_1a_4','5_1a_5','5_1a_6','5_1a_7','5_1a_8','5_1a_9', '6_1_1', '6_1_2', '6_1_3', '6_1_4', '6_1_5', '6_1_6', '6_1_7', '6_1_8', '6_1_9', '6_1_10', '6_1_11', '6_1_12', '6_1_13', '6_1_14')

pivoted_2022 = pivoted_2022.withColumn("7_1_1", col('7_1_0'))

# COMMAND ----------

# Stack back to original scheme
# Generalize q_id format in order to stack
df_cities_2022_sp_51a = df_cities_2022_sp_51a.withColumn("q_id", regexp_replace("q_id", "-", "_"))

#Unpivot
stacked_2022 = pivoted_2022.select("Account Number", 
                                   expr("stack(50, '5_1a_14_3',`5_1a_14_3`,	'5_1a_1_1',`5_1a_1_1`,	'5_1a_1_2',`5_1a_1_2`,	'5_1a_1_3',`5_1a_1_3`,	'5_1a_1_4',`5_1a_1_4`,	'5_1a_10_1',`5_1a_10_1`,	'5_1a_10_3',`5_1a_10_3`,	'5_1a_10_4',`5_1a_10_4`,	'5_1a_11_3',`5_1a_11_3`,	'5_1a_12_1',`5_1a_12_1`,	'5_1a_12_2',`5_1a_12_2`,	'5_1a_12_3',`5_1a_12_3`,	'5_1a_12_4',`5_1a_12_4`,	'5_1a_13_1',`5_1a_13_1`,	'5_1a_13_2',`5_1a_13_2`,	'5_1a_13_3',`5_1a_13_3`,	'5_1a_13_4',`5_1a_13_4`,	'5_1a_14_1',`5_1a_14_1`,	'5_1a_14_2',`5_1a_14_2`,	'5_1a_14_4',`5_1a_14_4`,	'5_1a_16_2',`5_1a_16_2`,	'5_1a_3_1',`5_1a_3_1`,	'5_1a_3_2',`5_1a_3_2`,	'5_1a_3_3',`5_1a_3_3`,	'5_1a_3_4',`5_1a_3_4`,	'5_1a_30_3',`5_1a_30_3`,	'5_1a_7_1',`5_1a_7_1`,	'5_1a_7_2',`5_1a_7_2`,	'5_1a_7_3',`5_1a_7_3`,	'5_1a_7_4',`5_1a_7_4`,	'5_1a_8_2',`5_1a_8_2`,	'5_1a_9_1',`5_1a_9_1`,	'5_1a_9_3',`5_1a_9_3`,	'5_1a_9_4',`5_1a_9_4`,	'6_1_1_1',`6_1_1_1`,	'6_1_1_10',`6_1_1_10`,	'6_1_1_11',`6_1_1_11`,	'6_1_1_12',`6_1_1_12`,	'6_1_1_13',`6_1_1_13`,	'6_1_1_14',`6_1_1_14`,	'6_1_1_2',`6_1_1_2`,	'6_1_1_3',`6_1_1_3`,	'6_1_1_4',`6_1_1_4`,	'6_1_1_5',`6_1_1_5`,	'6_1_1_6',`6_1_1_6`,	'6_1_1_7',`6_1_1_7`,	'6_1_1_8',`6_1_1_8`,	'6_1_1_9',`6_1_1_9`,	'7_1_0',`7_1_0`,	'7_1_1',`7_1_1`) as (q_id, Response)"))


#Re set the q_id format
stacked_2022 = stacked_2022.withColumn("q_id", regexp_replace(col("q_id"), "_", "-"))
# stacked_2022 = stacked_2022.withColumn("q_id", regexp_replace(col("q_id"), "(\d+)-(\w+)-(\d+)-(\d+)", "$1.$2-$3-$4"))
stacked_2022 = stacked_2022.withColumn("q_id", 
                                       when(length(col("q_id")) > 5, regexp_replace(col("q_id"), "(\d+)-(\w+)-(\d+)-(\d+)", "$1.$2-$3-$4"))
                                       .otherwise(regexp_replace(col("q_id"), "(\d+)-(\w+)-(\d+)", "$1.$2-$3")))
stacked_2022 = stacked_2022.withColumnRenamed('Response', 'Response Answer')

#Create columns to match the original dataset schema
stacked_2022 = stacked_2022.withColumn("Questionnaire", lit('Cities 2022'))\
.withColumn("Organization", lit(None))\
.withColumn("City", lit(None))\
.withColumn("Country", lit(None))\
.withColumn("CDP Region", lit(None))\
.withColumn("Parent Section", lit(None))\
.withColumn("Section", lit(None))\
.withColumn("Question Number",
                   when(col("q_id").startswith("5.1a"), lit("5.1a"))
                   .when(col("q_id").startswith("6.1"), lit("6.1"))
                   .when(col("q_id").startswith("7.1"), lit("7.1"))
                   .otherwise(None))\
.withColumn("Question Name", 
                   when(col("q_id").startswith("5.1a"), lit('provide details of your emissions reduction target(s)'))
                   .when(col("q_id").startswith("6.1"), lit("provide details of your jurisdiction's energy-related targets active in the reporting year. in addition, you can report other climate-related targets active in the reporting year"))
                   .when(col("q_id").startswith("7.1"), lit("does your jurisdiction have a climate action plan or strategy?"))
                   .otherwise(None))\
.withColumn("Column Number", split(stacked_2022['q_id'], "-")[1])\
.withColumn("Column Name", lit(None))\
.withColumn("Row Number", split(stacked_2022['q_id'], "-")[2])\
.withColumn("Row Name", lit(None))\
.withColumn("q_c", lit(None))\
.withColumn("q_c_r", lit(None))\
.withColumn("q_c_text", lit(None))\
.withColumn("q_c_r_text", lit(None))\
.withColumn("q_text", lit(None))\
.withColumn("q_type", lit(None))

stacked_2022 = stacked_2022.drop('id_temp').filter(col('Response Answer').isNotNull())
#stacked_2022 = stacked_2022.filter(col('Response Answer').isNotNull())

# COMMAND ----------

# Modify original dataframe by ignoring original questions pre-shaped above and adding new questions
df_cities_2022_sp_orig = df_cities_2022_sp
df_cities_2022f = df_cities_2022_sp.filter((col("Question Number") != '5.1a') & (~col("Question Number").like("6.1%")) & (col("Question Number") != '7.1'))

# Add new rows for question "5.1a"
df_cities_2022_sp = df_cities_2022f.unionByName(stacked_2022)

#Replace original pandas dataframe with shaped one
pd_cities_2022 = df_cities_2022_sp.toPandas()

#display(df_cities_2022_sp.filter((col("Question Number") == '5.1a') | (col("Question Number").like("6.1%")) | (col("Question Number") == '7.1')))

# COMMAND ----------

# DBTITLE 1,2) Questions match analysis
# Group by questions codes SIMPLIFIED
df_cities_2018_q = pd_cities_2018.groupby(['q_id', 'q_text', 'q_type'], as_index=False)['Response Answer'].count()
df_cities_2019_q = pd_cities_2019.groupby(['q_id', 'q_text', 'q_type'], as_index=False)['Response Answer'].count()
df_cities_2020_q = pd_cities_2020.groupby(['q_id', 'q_text', 'q_type'], as_index=False)['Response Answer'].count()
df_cities_2021_q = pd_cities_2021.groupby(['q_id', 'q_text', 'q_type'], as_index=False)['Response Answer'].count() 
df_cities_2022_q = pd_cities_2022.groupby(['q_id', 'q_text', 'q_type'], as_index=False)['Response Answer'].count()

# Clean output dataframes
#df_cities_2018_q = df_cities_2018_q.drop(columns='cualq')
#df_cities_2019_q = df_cities_2019_q.drop(columns='cualq')
#df_cities_2020_q = df_cities_2020_q.drop(columns='cualq')
#df_cities_2021_q = df_cities_2021_q.drop(columns='cualq')
#df_cities_2022_q = df_cities_2022_q.drop(columns='cualq')

# COMMAND ----------

df_cities_2021_q_responses = df_cities_2021_q[df_cities_2021_q['Response Answer'] < 300]
len(df_cities_2021_q_responses)

# COMMAND ----------

pip install nltk

# COMMAND ----------

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# COMMAND ----------

#Functions
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Para descargar modulo 'en_core_web_sm'
# leyramos@Atena:~/TPI$ source TPI/.venv/bin/activate
#(.venv) leyramos@Atena:~/TPI$ python -m spacy info
#(.venv) leyramos@Atena:~/TPI$ python -m spacy download en_core_web_sm

# Group by datasets by question name and column, in order to have unique questions - NOT WORKING
def questions_group(df):
    name =[x for x in globals() if globals()[x] is df][0]
    year = name[-4:]
    q_c_r_year = 'q_c_r' + '_' + str(year)
    q_c_r_text_year = 'q_c_r_text' + '_' + str(year)
    df = df.rename(columns={'q_c_r': q_c_r_year, 'q_c_r_text': q_c_r_text_year})
    df = df.groupby([q_c_r_year, q_c_r_text_year], as_index=False).agg(cualq= ('Column Number', 'sum'))
    df[q_c_r_text_year] = df[q_c_r_text_year].str.lower()  
    df = df.drop(columns='cualq')
    return
    df    

# Clean stopwords and punctuation marks

# Crear lista de stopwords y conjunto de signos de puntuación
# nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))
punctuation = set(string.punctuation)

# Función para limpiar texto
def clean_text(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar signos de puntuación
    texto = ''.join(caracter for caracter in texto if caracter not in punctuation)
    # Eliminar stopwords
    texto = ' '.join(palabra for palabra in texto.split() if palabra not in stop_words)
    return texto

# Stemmer
# Crear objeto Stemmer
stemmer = SnowballStemmer('english')

# Función para aplicar stemming a una columna
def stemmer_text(texto):
    tokens = nltk.word_tokenize(texto)
    stemmed_text = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_text)

# COMMAND ----------

df_cities_2018_q.head(5)

# COMMAND ----------

# SIMPLIFIED
# Cleand and simplify text using Stemmer
df_list = (df_cities_2018_q, df_cities_2019_q, df_cities_2020_q, df_cities_2021_q, df_cities_2022_q)

for df in df_list:
    df['q_text_cln'] = df['q_text'].apply(clean_text)
    df['q_text_cln'] = df['q_text_cln'].apply(stemmer_text)

# COMMAND ----------

pip install pyspark[sql]

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled","true")

df_cities_2018_q_sp =spark.createDataFrame(df_cities_2018_q) 
df_cities_2019_q_sp =spark.createDataFrame(df_cities_2019_q) 
df_cities_2020_q_sp =spark.createDataFrame(df_cities_2020_q) 
df_cities_2021_q_sp =spark.createDataFrame(df_cities_2021_q) 
df_cities_2022_q_sp =spark.createDataFrame(df_cities_2022_q) 

df_cities_2018_q_sp.printSchema()
df_cities_2019_q_sp.printSchema()
df_cities_2020_q_sp.printSchema()
df_cities_2021_q_sp.printSchema()
df_cities_2022_q_sp.printSchema()
#df_cities_2018_q_sp.show()

# COMMAND ----------

# Save result
df_cities_2018_q_sp.write.csv("/FileStore/questions_clnstm_2018_spf.csv", 
                     mode = "overwrite",
                     header = True)

df_cities_2019_q_sp.write.csv("/FileStore/questions_clnstm_2019_spf.csv", 
                     mode = "overwrite",
                     header = True)

df_cities_2020_q_sp.write.csv("/FileStore/questions_clnstm_2020_spf.csv", 
                     mode = "overwrite",
                     header = True)

df_cities_2021_q_sp.write.csv("/FileStore/questions_clnstm_2021_spf.csv", 
                     mode = "overwrite",
                     header = True)

df_cities_2022_q_sp.write.csv("/FileStore/questions_clnstm_2022_spf.csv", 
                     mode = "overwrite",
                     header = True)

# COMMAND ----------

# Input data from Databricks FileStore
df_cities_2018_q = spark.read.csv('dbfs:/FileStore/questions_clnstm_2018_spf.csv', header='true')
df_cities_2019_q = spark.read.csv('dbfs:/FileStore/questions_clnstm_2019_spf.csv', header='true')
df_cities_2020_q = spark.read.csv('dbfs:/FileStore/questions_clnstm_2020_spf.csv', header='true')
df_cities_2021_q = spark.read.csv('dbfs:/FileStore/questions_clnstm_2021_spf.csv', header='true')
df_cities_2022_q = spark.read.csv('dbfs:/FileStore/questions_clnstm_2022_spf.csv', header='true')

# COMMAND ----------

display(df_cities_2022_q)

# COMMAND ----------

# Pipeline to process questions strings in order to enhance the fuzzy matching
# https://medium.com/analytics-vidhya/fuzzy-string-matching-with-spark-in-python-7fcd0c422f71

# Fit on 2021, the "pivot" dataset before restructuring the survey
import pyspark
from pyspark.sql import SparkSession, functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer, NGram, HashingTF, MinHashLSH, RegexTokenizer, SQLTransformer
model = Pipeline(stages=[
    SQLTransformer(statement="SELECT *, lower(q_text_cln) lower FROM __THIS__"),
    Tokenizer(inputCol="lower", outputCol="token"),
    StopWordsRemover(inputCol="token", outputCol="stop"),
    SQLTransformer(statement="SELECT *, concat_ws(' ', stop) concat FROM __THIS__"),
    RegexTokenizer(pattern="", inputCol="concat", outputCol="char", minTokenLength=1),
    NGram(n=2, inputCol="char", outputCol="ngram"),
    HashingTF(inputCol="ngram", outputCol="vector"),
    MinHashLSH(inputCol="vector", outputCol="lsh", numHashTables=3)
]).fit(df_cities_2021_q)

result_2021 = model.transform(df_cities_2021_q)
result_2021 = result_2021.filter(F.size(F.col("ngram")) > 0)    

# COMMAND ----------

# Apply pipeline on other dataframes

result_2018 = model.transform(df_cities_2018_q)
result_2018 = result_2018.filter(F.size(F.col("ngram")) > 0)   

result_2019 = model.transform(df_cities_2019_q)
result_2019 = result_2019.filter(F.size(F.col("ngram")) > 0)   

result_2020 = model.transform(df_cities_2020_q)
result_2020 = result_2020.filter(F.size(F.col("ngram")) > 0)   

result_2022 = model.transform(df_cities_2022_q)
result_2022 = result_2022.filter(F.size(F.col("ngram")) > 0)   

# COMMAND ----------

# Fuzzy match all models to 2021
questions_fuzz_2021_2018 = model.stages[-1].approxSimilarityJoin(result_2021, result_2018, 0.5, "jaccardDist")
questions_fuzz_2021_2019 = model.stages[-1].approxSimilarityJoin(result_2021, result_2019, 0.5, "jaccardDist")
questions_fuzz_2021_2020 = model.stages[-1].approxSimilarityJoin(result_2021, result_2020, 0.5, "jaccardDist")
questions_fuzz_2021_2022 = model.stages[-1].approxSimilarityJoin(result_2021, result_2022, 0.5, "jaccardDist")


# COMMAND ----------

# Select the match option with the minimun Jaccard Distance
from pyspark.sql import Window
w = Window.partitionBy('datasetA.q_id')

questions_fuzz_2021_2018 = (questions_fuzz_2021_2018
             .withColumn('minDist', F.min('jaccardDist').over(w))
             .where(F.col('jaccardDist') == F.col('minDist'))
             .drop('minDist'))

questions_fuzz_2021_2019 = (questions_fuzz_2021_2019
             .withColumn('minDist', F.min('jaccardDist').over(w))
             .where(F.col('jaccardDist') == F.col('minDist'))
             .drop('minDist'))

questions_fuzz_2021_2020 = (questions_fuzz_2021_2020
             .withColumn('minDist', F.min('jaccardDist').over(w))
             .where(F.col('jaccardDist') == F.col('minDist'))
             .drop('minDist'))

questions_fuzz_2021_2022 = (questions_fuzz_2021_2022
             .withColumn('minDist', F.min('jaccardDist').over(w))
             .where(F.col('jaccardDist') == F.col('minDist'))
             .drop('minDist'))

# COMMAND ----------

from pyspark.sql.functions import col

questions_fuzz_2021_2018 = questions_fuzz_2021_2018.select(
  col('datasetA.q_id').alias('q_id_2021'),   \
  col('datasetA.q_text').alias('q_text_2021'),  \
  col('datasetA.q_type').alias('q_type_2021'),   \
  col('datasetB.q_id').alias('q_id_2018'),  \
  col('datasetB.q_text').alias('q_text_2018'),   \
  col('datasetB.q_type').alias('q_type_2018'),   \
  'jaccardDist')

questions_fuzz_2021_2019 = questions_fuzz_2021_2019.select(
  col('datasetA.q_id').alias('q_id_2021'),   \
  col('datasetA.q_text').alias('q_text_2021'),  \
  col('datasetA.q_type').alias('q_type_2021'),   \
  col('datasetB.q_id').alias('q_id_2019'),  \
  col('datasetB.q_text').alias('q_text_2019'),  \
  col('datasetB.q_type').alias('q_type_2019'),   \
  'jaccardDist')

questions_fuzz_2021_2020 = questions_fuzz_2021_2020.select(
  col('datasetA.q_id').alias('q_id_2021'),   \
  col('datasetA.q_text').alias('q_text_2021'),  \
  col('datasetA.q_type').alias('q_type_2021'),   \
  col('datasetB.q_id').alias('q_id_2020'),  \
  col('datasetB.q_text').alias('q_text_2020'),  \
  col('datasetB.q_type').alias('q_type_2020'),   \
  'jaccardDist')

questions_fuzz_2021_2022 = questions_fuzz_2021_2022.select(
  col('datasetA.q_id').alias('q_id_2021'),   \
  col('datasetA.q_text').alias('q_text_2021'),  \
  col('datasetA.q_type').alias('q_type_2021'),   \
  col('datasetB.q_id').alias('q_id_2022'),  \
  col('datasetB.q_text').alias('q_text_2022'),   \
  col('datasetB.q_type').alias('q_type_2021'),   \
  'jaccardDist')

# COMMAND ----------

# Resulting Shape
def sparkShape(df):
    return (df.count(), len(df.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape

print(questions_fuzz_2021_2018.shape())
print(questions_fuzz_2021_2019.shape())
print(questions_fuzz_2021_2020.shape())
print(questions_fuzz_2021_2022.shape())

# COMMAND ----------

display(questions_fuzz_2021_2018)

# COMMAND ----------

display(questions_fuzz_2021_2019)

# COMMAND ----------

display(questions_fuzz_2021_2020) 

# COMMAND ----------

display(questions_fuzz_2021_2022)