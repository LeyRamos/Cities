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
from pyspark.sql.functions import col, regexp_replace, split
from pyspark.sql.functions import col, create_map, lit, when

from pyspark.sql.types import *;
from scipy.stats import *
from scipy import stats

import gzip
# import StringIO --> ModuleNotFoundError: No module named 'StringIO'


from functools import reduce

# unificaion
from pyspark.sql.functions import col, concat, lit, lower, when

# COMMAND ----------

# Create folders to store the input data

dbutils.fs.mkdirs("/datasets/questionnaires")
dbutils.fs.mkdirs("/datasets/cities")
dbutils.fs.mkdirs("/outputs")

# COMMAND ----------

# Check the files properly loaded
dbutils.fs.ls('dbfs:/FileStore/')

# COMMAND ----------

# Remove files
#dbutils.fs.rm('dbfs:/FileStore/df_responses.csv/', recurse=True)

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

# Functions
# Add keys Question Number-Column Number y Question Number-Column Number-Row Number
def add_keys(df):
    df = df.withColumn('q_c', concat(col('Question Number').cast('string'), lit('-'), col('Column Number').cast('string')))
    df = df.withColumn('q_c_r', concat(col('Question Number').cast('string'), lit('-'), col('Column Number').cast('string'), lit('-'), col('Row Number').cast('string')))
    return df

# Add text Question Name-Column Name y Question Name-Column Name-Row Name
def add_full_qtext(df):
    df = df.fillna('none', subset=['Column Name', 'Row Name'])
    df = df.withColumn('q_c_text', concat(col('Question Name').cast('string'), lit('-'), col('Column Name').cast('string')))
    df = df.withColumn('q_c_r_text', concat(col('Question Name').cast('string'), lit('-'), col('Column Name').cast('string'), lit('-'), col('Row Name').cast('string')))
    df = df.withColumn('q_c_text', lower(col('q_c_text')))
    df = df.withColumn('q_c_r_text', lower(col('q_c_r_text')))
    df = df.withColumn('q_id', when(col('Row Name') == 'none', col('q_c')).otherwise(col('q_c_r')))
    df = df.withColumn('q_text', when(col('Row Name') == 'none', col('q_c_text')).otherwise(col('q_c_r_text')))
    df = df.withColumn('q_type', when(col('Row Name') == 'none', 'Question-Column').otherwise('Question-Column-Row'))
    return df

# COMMAND ----------

# Add keys and concatenated text
df_list = [df_cities_2018, df_cities_2019, df_cities_2020, df_cities_2021, df_cities_2022]

df_list = [add_keys(df) for df in df_list]
df_list = [add_full_qtext(df) for df in df_list]

# Back from list to dataframes
df_cities_2018 = df_list[0]
df_cities_2019 = df_list[1]
df_cities_2020 = df_list[2]
df_cities_2021 = df_list[3]
df_cities_2022 = df_list[4]

# COMMAND ----------

display(df_cities_2018)

# COMMAND ----------

# DBTITLE 1,Pre-shaping dataset 2022 original
from pyspark.sql.functions import col, first, expr
from pyspark.sql.functions import regexp_replace 

#Filter rows corresponding to question 5.1a
df_cities_2022_sp_51a = df_cities_2022.filter((col("Question Number") == '5.1a') | (col("Question Number").like("6.1%")) | (col("Question Number") == '7.1'))
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
df_cities_2022_orig = df_cities_2022
df_cities_2022f = df_cities_2022.filter((col("Question Number") != '5.1a') & (~col("Question Number").like("6.1%")) & (col("Question Number") != '7.1'))

stacked_2022 = stacked_2022.drop ('City')

# Add new rows for question "5.1a", 6.1 and 7.1
df_cities_2022_sp = df_cities_2022f.unionByName(stacked_2022)

#Replace original dataframe with shaped one
df_cities_2022 = df_cities_2022_sp

#display(df_cities_2022_sp.filter((col("Question Number") == '5.1a') | (col("Question Number").like("6.1%")) | (col("Question Number") == '7.1')))

# COMMAND ----------

#display(df_cities_2020.filter((col("q_id") == '13.2-0') | (col("q_id") == '4.6a-7-25'))
display(df_cities_2022.filter(col("q_id") == '3.7-3-2'))

# COMMAND ----------

# DBTITLE 1,2) Unify All Dataframes
# 1 - para cada dataset eliminar q_id que no tienen match con 2021 --- ok
# 2 - para cada dataset reemplazar q_id con los de 2021 --- ok
# 3 - append todos los datasets --- ok
# 4 - Generar dataset de catalogo de preguntas (group by) con todas las preguntas que sobrevivieron los q_ids. --- ok
# 5 - Generar dataset de respuestas -- eliminar textos y dejar solo account_id, questionaire, q_id, respuesta para cada año --- ok
# 6 - Para cada columna del dataset de respuestas, generar uniques y unificar categorias, tipos de datos y corregir errores de match -- ok 
# 7 - Generar dataset de catalogo de accounts -- ok

# COMMAND ----------

df_cities_2018 = df_cities_2018.withColumn('q_id_orig', df_cities_2018['q_id'])
df_cities_2019 = df_cities_2019.withColumn('q_id_orig', df_cities_2019['q_id'])
df_cities_2020 = df_cities_2020.withColumn('q_id_orig', df_cities_2020['q_id'])
df_cities_2021 = df_cities_2021.withColumn('q_id_orig', df_cities_2021['q_id'])
df_cities_2022 = df_cities_2022.withColumn('q_id_orig', df_cities_2022['q_id'])

# COMMAND ----------

# Remove rows corresponding to questions not included

# 2018 -- 183 questions
q_tokeep_2018 = ['0.6-4-1',	'0.1-1-1',	'0.6-2-1',	'11.0-6-1',	'11.4-5-1',	'0.8-2-1',	'11.4-5-2',	'11.4-5-3',	'11.4-5-4',	'12.0-0',	'11.4-5-5',	'15.1-0',	'15.3a-2',	'2.0b-6',	'3.1-0',	'2.2a-5',	'11.0-1-1',	'2.2a-3',	'2.2a-8',	'11.0-2-1',	'2.4-3',	'11.0-3-1',	'3.3-2',	'3.3-3',	'3.4-1',	'3.4-3',	'2.2a-4',	'11.0-4-1',	'11.0-5-1',	'2.0-0',	'2.0b-3',	'11.0-7-1',	'2.0b-4',	'2.0b-7',	'11.0-8-1',	'2.2a-1',	'2.2a-7',	'11.4-1-1',	'2.2a-9',	'11.4-1-2',	'2.4-1',	'11.4-1-3',	'2.4-2',	'7.0-0',	'11.4-1-4',	'11.4-1-5',	'11.4-2-1',	'11.4-2-2',	'11.4-2-3',	'11.4-2-4',	'11.4-2-5',	'11.4-3-1',	'11.4-3-2',	'11.4-3-3',	'11.4-3-4',	'11.4-3-5',	'11.4-4-1',	'11.4-4-2',	'11.4-4-3',	'11.4-4-4',	'11.4-4-5',	'14.0-1-1',	'14.0-1-2',	'14.0-1-3',	'14.0-1-4',	'14.0-1-5',	'14.0-1-6', '14.5-2-1', '14.5-2-2', '14.5-2-3', '14.5-2-4', '15.0-0',	'15.3a-1',	'15.3a-3',	'15.3a-4',	'15.4-2',	'15.4-3',	'3.1a-1',	'3.1a-2',	'3.1a-4',	'3.3-4',	'3.4-2',	'8.3-0',	'8.3a-1',	'8.3a-7',	'8.3a-3',	'8.3a-2',	'8.3a-4',	'8.3a-5',	'8.3a-6',	'8.3b-1',	'8.3b-6',	'8.3b-7',	'8.3b-8',	'8.1-3',	'8.2-0',	'7.3a-1-1',	'7.3a-1-10',	'7.3a-1-11',	'7.3a-1-12',	'7.3a-1-13',	'7.3a-1-14',	'7.3a-1-15',	'7.3a-1-16',	'7.3a-1-17',	'7.3a-1-2',	'7.3a-1-3',	'7.3a-1-4',	'7.3a-1-5',	'7.3a-1-6',	'7.3a-1-7',	'7.3a-1-8',	'7.3a-1-9',	'7.4b-1-1',	'7.4b-13-1',	'7.4b-3-1',	'7.4b-8-1',	'7.6a-1',	'7.6a-2',	'7.6a-3',	'7.6a-4',	'7.6c-1',	'7.6c-2',	'7.6c-3',	'7.6c-4',	'7.6d-1',	'7.6d-2',	'7.6d-3',	'7.6d-4',	'8.3b-4',	'8.3b-2',	'8.3b-3',	'8.3b-5',	'8.3c-1',	'8.3c-7',	'8.3c-8',	'8.3c-3',	'8.3c-2',	'8.3c-4',	'8.3c-5',	'8.3c-6',	'5.0a-1',	'5.0a-2',	'5.10-0',	'5.1-0',	'5.1a-1',	'5.1a-2',	'5.2-2',	'5.2-3',	'5.2-4',	'5.2-6',	'6.0-0',	'6.2-0',	'6.3-1-1',	'6.4-0',	'6.7-2-1',	'6.7-3-1',	'6.7-4-1',	'6.8-0',	'6.8a-2',	'9.2-0',	'9.2a-3',	'9.2a-5',	'9.2a-6',	'9.2a-8',	'9.0-1-1',	'9.0-10-1',	'9.0-2-1',	'9.0-3-1',	'9.0-4-1',	'9.0-5-1',	'9.0-6-1',	'9.0-7-1',	'9.0-8-1',	'9.0-9-1',	'10.0-1-1',	'10.0-1-2',	'10.0-1-3',	'10.0-1-4',	'10.0-2-1',	'10.0-2-2',	'10.0-2-3',	'10.0-2-4', '7.6-1-1',	'7.6-1-10',	'7.6-1-13',	'7.6-1-14',	'7.6-1-11',	'7.6-1-15',	'7.6-1-2',	'7.6-1-3',	'7.6-1-4',	'7.6-1-5',	'7.6-1-6',	'7.6-1-7',	'7.6-1-8',	'7.6-1-9',	'7.6-2-1',	'7.6-2-10',	'7.6-2-11',	'7.6-2-13',	'7.6-2-14',	'7.6-2-2',	'7.6-2-3',	'7.6-2-4',	'7.6-2-6',	'7.6-2-7',	'7.6-2-8',	'9.2a-2', '0.6-1-1']

# 2019 -- 407 questions
q_tokeep_2019 = ['0.1-1-1',	'4.6a-1-1',	'0.5-2-1',	'0.6-1-1',	'0.5-3-1',	'4.6a-1-10',	'11.0-0',	'14.2-0',	'4.6a-1-11',	'3.1-0',	'4.6a-1-12',	'10.1-1-1',	'10.1-2-1',	'10.1-8-1',	'10.1-3-1',	'10.1-4-1',	'10.1-5-1',	'10.1-6-1',	'10.1-7-1',	'2.1-8',	'2.1-10',	'13.2-0',	'4.4-0',	'13.3-1-1',	'13.3-1-2',	'13.3-1-3',	'2.0-0',	'13.3-1-4',	'13.3-1-5',	'2.0b-3',	'13.3-1-6',	'2.0b-4',	'2.0b-8',	'2.0b-7',	'2.1-1',	'2.1-6',	'14.0-0',	'10.5-1-1',	'2.1-9',	'2.1-5',	'10.5-1-2',	'2.1-3',	'2.1-4',	'2.1-7',	'2.2-1',	'10.5-1-3',	'2.2-2',	'2.2-3',	'4.0-0',	'10.5-1-4',	'4.6a-1-13',	'4.6a-1-14',	'4.6a-1-15',	'3.0-2',	'10.5-1-5',	'4.6a-1-16',	'3.0-4',	'4.6a-1-17',	'3.0-6',	'10.5-2-1',	'4.6a-1-18',	'10.5-2-2',	'4.6a-1-19',	'4.6a-1-2',	'4.6a-1-20',	'10.5-2-3',	'4.6a-1-22',	'4.6a-1-23',	'10.5-2-4',	'4.6a-1-25',	'4.6a-1-26',	'10.5-2-5',	'4.6a-1-27',	'4.6a-1-28',	'4.6a-1-29',	'10.5-3-1',	'4.6a-1-3',	'4.6a-1-30',	'4.6a-1-9',	'4.6a-1-31',	'10.5-3-2',	'4.6a-1-4',	'4.6a-1-5',	'4.6a-1-6',	'10.5-3-3',	'4.6a-1-7',	'4.6a-1-8',	'4.6a-2-1',	'10.5-3-4',	'4.6a-2-10',	'4.6a-2-11',	'4.6a-2-12',	'4.6a-2-13',	'10.5-3-5',	'4.6a-2-14',	'4.6a-2-15',	'4.6a-4-1',	'4.6a-2-16',	'4.6a-4-11',	'4.6a-2-17',	'10.5-4-1',	'4.6a-4-14',	'4.6a-2-18',	'4.6a-4-18',	'4.6a-2-20',	'4.6a-2-19',	'10.5-4-2',	'4.6a-2-2',	'4.6a-2-25',	'4.6a-4-29',	'4.6a-2-30',	'4.6a-2-22',	'4.6a-4-4',	'4.6a-2-5',	'10.5-4-3',	'4.6a-4-7',	'4.6a-2-8',	'10.5-4-4',	'4.6a-4-9',	'4.6a-3-1',	'4.6a-3-10',	'4.6a-3-11',	'10.5-4-5',	'4.6a-3-12',	'4.6a-3-9',	'10.5-5-1',	'4.6a-3-14',	'4.6a-3-15',	'10.5-5-2',	'4.6a-3-16',	'4.6a-3-18',	'10.5-5-3',	'4.6a-3-17',	'10.5-5-4',	'4.6a-3-19',	'10.5-5-5',	'4.6a-3-2',	'4.6a-3-20',	'4.6a-3-22',	'4.6a-3-23',	'4.6a-3-25',	'4.6a-3-26',	'4.6a-3-27',	'4.6a-3-28',	'4.6a-3-29',	'4.6a-3-3',	'4.6a-3-30',	'4.6a-3-31',	'4.6a-3-4',	'4.6a-3-5',	'4.6a-3-6',	'4.6a-3-7',	'4.6a-3-8',	'4.6a-3-13',	'4.6a-4-10',	'4.6a-4-12',	'4.6a-4-15',	'4.6a-4-16',	'4.6a-4-17',	'4.6a-4-19',	'4.6a-4-2',	'4.6a-4-20',	'4.6a-4-22',	'4.6a-2-23',	'4.6a-4-23',	'4.6a-4-25',	'4.6a-2-26',	'4.6a-4-26',	'4.6a-2-27',	'4.6a-4-27',	'4.6a-2-28',	'4.6a-4-28',	'4.6a-2-29',	'4.6a-2-3',	'4.6a-4-3',	'4.6a-4-30',	'4.6a-2-31',	'4.6a-4-31',	'4.6a-2-4',	'4.6a-4-5',	'4.6a-2-6',	'4.6a-4-6',	'4.6a-2-7',	'4.6a-4-8',	'4.6a-2-9',	'4.6a-4-13',	'4.6a-5-1',	'4.6a-5-10',	'4.6a-5-11',	'4.6a-5-12',	'4.6a-5-13',	'4.6a-5-14',	'14.3a-1',	'4.6a-5-15',	'14.3a-2',	'4.6a-5-16',	'14.3a-3',	'4.6a-5-17',	'14.3a-4',	'4.6a-5-18',	'4.6a-5-19',	'14.4-2',	'4.6a-5-2',	'4.6a-5-20',	'14.4-4',	'4.6a-5-22',	'4.6a-5-23',	'4.6a-5-25',	'4.6a-5-26',	'4.6a-5-27',	'4.6a-5-28',	'4.6a-5-29',	'4.6a-5-3',	'4.6a-5-30',	'4.6a-5-31',	'4.6a-5-4',	'4.6a-5-5',	'4.6a-5-6',	'4.6a-5-7',	'4.6a-5-8',	'4.6a-5-9',	'4.6a-6-1',	'4.6a-6-10',	'4.6a-6-11',	'4.6a-6-12',	'4.6a-6-9',	'4.6a-6-14',	'4.6a-6-15',	'4.6a-6-16',	'4.6a-6-17',	'4.6a-6-18',	'4.6a-6-19',	'4.6a-6-2',	'4.6a-6-20',	'4.6a-6-22',	'4.6a-6-23',	'4.6a-6-25',	'4.6a-6-26',	'4.6a-6-27',	'4.6a-6-28',	'4.6a-6-29',	'4.6a-6-3',	'4.6a-6-30',	'4.6a-6-31',	'4.6a-6-4',	'4.6a-6-5',	'4.6a-6-6',	'4.6a-6-7',	'4.6a-6-8',	'4.6a-6-13',	'3.1a-1',	'3.1a-2',	'3.1a-3',	'3.1a-4',	'4.6b-1-1',	'4.6b-1-10',	'4.6b-1-11',	'4.6b-1-12',	'4.6b-1-13',	'3.1a-7',	'4.6b-1-14',	'4.6b-1-15',	'4.6b-1-16',	'4.6b-1-17',	'4.6b-1-2',	'4.6b-1-3',	'4.6b-1-4',	'3.2-1',	'4.6b-1-5',	'4.6b-1-6',	'3.2-2',	'4.6b-1-7',	'4.6b-1-8',	'3.2-3',	'4.6b-1-9',	'4.6c-1-1',	'4.6c-13-1',	'4.6c-3-1',	'4.6c-8-1',	'4.6d-1',	'4.6d-2',	'4.6d-3',	'4.6d-4',	'4.6e-1',	'4.6e-2',	'4.6e-3',	'4.6e-4',	'4.6f-1',	'4.6f-2',	'4.6f-3',	'4.6f-4',	'4.9-1-1',	'5.0-0',	'5.0a-1',	'5.0a-6',	'5.0a-4',	'5.0a-3',	'5.0a-5',	'5.0a-7',	'5.0a-8',	'5.0a-10',	'5.0b-1',	'5.0b-3',	'5.0b-4',	'5.0b-5',	'5.0b-7',	'5.0b-8',	'5.0b-6',	'5.0c-1',	'5.0c-11',	'5.0c-12',	'5.0c-4',	'5.0c-3',	'5.0c-5',	'5.0c-6',	'5.0c-7',	'5.0c-8',	'5.0c-9',	'5.0c-10',	'5.0d-1',	'5.0d-3',	'5.0d-9',	'5.0d-10',	'5.4-5',	'5.4-6',	'5.5-0',	'5.0d-4',	'5.0d-5',	'5.0d-6',	'5.0d-7',	'5.0d-8',	'6.0-1',	'6.0-2',	'6.11-0',	'6.1-0',	'6.1a-1',	'6.1a-2',	'6.2-3',	'6.2-4',	'6.2-5',	'6.2-7',	'7.0-0',	'7.2-0',	'7.3-1-1',	'7.4-0',	'7.6-2-1',	'7.6-3-1',	'7.6-4-1',	'7.7-0',	'7.7a-2',	'8.0-0',	'8.0a-2',	'8.0a-3',	'8.0a-5',	'8.0a-6',	'8.0a-8',	'8.2-1-1',	'8.2-10-1',	'8.2-2-1',	'8.2-3-1',	'8.2-4-1',	'8.2-5-1',	'8.2-6-1',	'8.2-7-1',	'8.2-8-1',	'8.2-9-1',	'8.5-1-4',	'9.1-1-1',	'9.1-1-2',	'9.1-1-3',	'9.1-1-4',	'9.1-2-1',	'9.1-2-2',	'9.1-2-3',	'9.1-2-4', '0.5-1-1']

# 2020 -- 407 questions
q_tokeep_2020 = ['11.0-0',	'14.1-0',	'3.2-0',	'2.1-11',	'2.1-7',	'4.4-0',	'2.0-0',	'2.0b-2',	'2.0b-4',	'2.0b-7',	'2.0b-6',	'2.1-1',	'2.1-8',	'2.1-6',	'2.1-5',	'2.1-3',	'2.1-4',	'2.1-9',	'2.2-1',	'2.2-2',	'2.2-4',	'4.0-0',	'13.2-0',	'14.0-0',	'14.2a-1',	'14.2a-2',	'14.2a-3',	'14.2a-5',	'14.3-2',	'14.3-4',	'3.0-2',	'3.0-4',	'3.0-8',	'3.2a-1',	'3.2a-3',	'3.2a-5',	'3.2a-6',	'3.2a-9',	'3.3-1',	'3.3-3',	'3.3-4',	'5.0-0',	'5.0a-1',	'5.0a-10',	'5.0a-3',	'5.0a-4',	'5.0a-5',	'5.0a-6',	'5.0a-7',	'5.0a-8',	'5.0b-1',	'5.0b-8',	'5.0b-3',	'5.0b-4',	'5.0b-5',	'5.0b-6',	'5.0b-7',	'5.0c-1',	'5.0c-9',	'5.0c-10',	'5.0c-11',	'5.0c-12',	'5.0c-3',	'5.4-5',	'5.4-6',	'5.5-0',	'4.6d-1',	'4.6d-2',	'4.6d-3',	'4.6d-4',	'4.6e-1',	'4.6e-2',	'4.6e-3',	'4.6e-4',	'4.6f-1',	'4.6f-2',	'4.6f-3',	'4.6f-4',	'5.0c-4',	'5.0c-5',	'5.0c-6',	'5.0c-7',	'5.0c-8',	'5.0d-1',	'5.0d-10',	'5.0d-3',	'5.0d-4',	'5.0d-5',	'5.0d-6',	'5.0d-7',	'5.0d-8',	'5.0d-9',	'6.0-1',	'6.0-2',	'6.2-0',	'6.2a-1',	'6.2a-3',	'6.5-3',	'6.5-4',	'6.5-7',	'6.5-9',	'7.0-0',	'7.2-0',	'7.4-0',	'7.7-0',	'7.7a-2',	'8.0-0',	'8.0a-3',	'8.0a-4',	'8.0a-6',	'8.0a-7',	'8.0a-9',	'0.1-1-1',	'4.6a-1-1',	'0.5-2-1',	'0.6-1-1',	'0.5-3-1',	'4.6a-1-10',	'4.6a-1-11',	'4.6a-1-12',	'10.1-1-1',	'10.1-2-1',	'10.1-3-1',	'10.1-4-1',	'10.1-5-1',	'10.1-6-1',	'10.1-7-1',	'10.1-9-1',	'10.4-1-1',	'10.4-1-2',	'10.4-1-3',	'10.4-1-4',	'4.6a-1-13',	'4.6a-1-14',	'4.6a-1-15',	'10.4-1-5',	'4.6a-1-16',	'4.6a-1-17',	'10.4-2-1',	'4.6a-1-18',	'10.4-2-2',	'4.6a-1-19',	'4.6a-1-2',	'4.6a-1-20',	'10.4-2-3',	'4.6a-1-22',	'4.6a-1-23',	'10.4-2-4',	'4.6a-1-25',	'4.6a-1-26',	'10.4-2-5',	'4.6a-1-27',	'4.6a-1-28',	'4.6a-1-29',	'10.4-3-1',	'4.6a-1-3',	'4.6a-1-30',	'4.6a-1-31',	'10.4-3-2',	'4.6a-1-4',	'4.6a-1-5',	'4.6a-1-6',	'10.4-3-3',	'4.6a-1-7',	'4.6a-1-8',	'4.6a-1-9',	'4.6a-2-1',	'10.4-3-4',	'4.6a-2-10',	'4.6a-2-11',	'4.6a-2-12',	'4.6a-2-13',	'10.4-3-5',	'4.6a-2-14',	'10.4-4-1',	'4.6a-2-15',	'10.4-4-2',	'10.4-4-3',	'4.6a-2-16',	'10.4-4-4',	'10.4-4-5',	'4.6a-2-17',	'10.4-5-1',	'10.4-5-2',	'4.6a-2-18',	'10.4-5-3',	'10.4-5-4',	'4.6a-2-19',	'10.4-5-5',	'4.6a-2-2',	'4.6a-2-20',	'4.6a-2-22',	'13.3-1-1',	'13.3-1-2',	'13.3-1-3',	'13.3-1-4',	'13.3-1-5',	'13.3-1-6',	'4.6a-2-23',	'4.6a-2-25',	'4.6a-2-26',	'4.6a-2-27',	'4.6a-2-28',	'4.6a-2-29',	'4.6a-2-3',	'4.6a-2-30',	'4.6a-2-31',	'4.6a-2-4',	'4.6a-2-5',	'4.6a-2-6',	'4.6a-2-7',	'4.6a-2-8',	'4.6a-2-9',	'4.6a-3-1',	'4.6a-3-10',	'4.6a-3-11',	'4.6a-3-12',	'4.6a-3-13',	'4.6a-3-14',	'4.6a-3-15',	'4.6a-3-16',	'4.6a-3-17',	'4.6a-3-18',	'4.6a-3-19',	'4.6a-3-2',	'4.6a-3-20',	'4.6a-3-22',	'4.6a-3-23',	'4.6a-3-25',	'4.6a-3-26',	'4.6a-3-27',	'4.6a-3-28',	'4.6a-3-29',	'4.6a-3-3',	'4.6a-3-30',	'4.6a-3-31',	'4.6a-3-4',	'4.6a-3-5',	'4.6a-3-6',	'4.6a-3-7',	'4.6a-3-8',	'4.6a-3-9',	'4.6a-4-1',	'4.6a-4-10',	'4.6a-4-11',	'4.6a-4-12',	'4.6a-4-13',	'4.6a-4-14',	'4.6a-4-15',	'4.6a-4-16',	'4.6a-4-17',	'4.6a-4-18',	'4.6a-4-19',	'4.6a-4-2',	'4.6a-4-20',	'4.6a-4-22',	'4.6a-4-23',	'4.6a-4-25',	'4.6a-4-26',	'4.6a-4-27',	'4.6a-4-28',	'4.6a-4-29',	'4.6a-4-3',	'4.6a-4-30',	'4.6a-4-31',	'4.6a-4-4',	'4.6a-4-5',	'4.6a-4-6',	'4.6a-4-7',	'4.6a-4-8',	'4.6a-4-9',	'4.6a-5-1',	'4.6a-5-10',	'4.6a-5-11',	'4.6a-5-12',	'4.6a-5-13',	'4.6a-5-14',	'4.6a-5-15',	'4.6a-5-16',	'4.6a-5-17',	'4.6a-5-18',	'4.6a-5-19',	'4.6a-5-2',	'4.6a-5-20',	'4.6a-5-22',	'4.6a-5-23',	'4.6a-5-25',	'4.6a-5-26',	'4.6a-5-27',	'4.6a-5-28',	'4.6a-5-29',	'4.6a-5-3',	'4.6a-5-30',	'4.6a-5-31',	'4.6a-5-4',	'4.6a-5-5',	'4.6a-5-6',	'4.6a-5-7',	'4.6a-5-8',	'4.6a-5-9',	'4.6a-6-1',	'4.6a-6-10',	'4.6a-6-11',	'4.6a-6-12',	'4.6a-6-13',	'4.6a-6-14',	'4.6a-6-15',	'4.6a-6-16',	'4.6a-6-17',	'4.6a-6-18',	'4.6a-6-19',	'4.6a-6-2',	'4.6a-6-20',	'4.6a-6-22',	'4.6a-6-23',	'4.6a-6-25',	'4.6a-6-26',	'4.6a-6-27',	'4.6a-6-28',	'4.6a-6-29',	'4.6a-6-3',	'4.6a-6-30',	'4.9-1-1',	'4.6a-6-31',	'4.6a-6-4',	'4.6a-6-5',	'4.6a-6-6',	'4.6a-6-7',	'4.6a-6-8',	'4.6a-6-9',	'4.6b-1-1',	'4.6b-1-10',	'4.6b-1-11',	'4.6b-1-12',	'4.6b-1-13',	'4.6b-1-14',	'4.6b-1-15',	'4.6b-1-16',	'4.6b-1-17',	'4.6b-1-2',	'4.6b-1-3',	'4.6b-1-4',	'4.6b-1-5',	'4.6b-1-6',	'4.6b-1-7',	'4.6b-1-8',	'4.6b-1-9',	'4.6c-1-1',	'4.6c-13-1',	'4.6c-3-1',	'4.6c-8-1',	'6.15-1-1',	'7.3-1-1',	'7.6-2-1',	'7.6-3-1',	'7.6-4-1',	'8.1-1-1',	'8.1-10-1',	'8.1-2-1',	'8.1-3-1',	'8.1-4-1',	'8.1-5-1',	'8.1-6-1',	'8.1-7-1',	'8.1-8-1',	'8.1-9-1',	'8.4-1-4',	'9.1-1-1',	'9.1-1-2',	'9.1-1-3',	'9.1-1-4',	'9.1-3-1',	'9.1-3-2',	'9.1-3-3',	'9.1-3-4', '0.5-1-1']

# 2021 -- 407 questions
q_tokeep_2021 = ['0.1-1-1',	'0.5-2-1',	'0.5-3-1',	'0.6-1-1',	'10.1-1-1',	'10.1-2-1',	'10.1-3-1',	'10.1-4-1',	'10.1-5-1',	'10.1-6-1',	'10.1-7-1',	'10.1-9-1',	'10.3-1-1',	'10.3-1-2',	'10.3-1-3',	'10.3-1-4',	'10.3-1-5',	'10.3-2-1',	'10.3-2-2',	'10.3-2-3',	'10.3-2-4',	'10.3-2-5',	'10.3-3-1',	'10.3-3-2',	'10.3-3-3',	'10.3-3-4',	'10.3-3-5',	'10.3-4-1',	'10.3-4-2',	'10.3-4-3',	'10.3-4-4',	'10.3-4-5',	'10.3-5-1',	'10.3-5-2',	'10.3-5-3',	'10.3-5-4',	'10.3-5-5',	'11.0-0',	'13.2-0',	'13.3-1-1',	'13.3-1-2',	'13.3-1-3',	'13.3-1-4',	'13.3-1-5',	'13.3-1-6',	'14.0-0',	'14.1-0',	'14.2a-1',	'14.2a-2',	'14.2a-3',	'14.2a-5',	'14.3-2',	'14.3-4',	'2.0-0',	'2.0b-2',	'2.0b-4',	'2.0b-6',	'2.0b-7',	'2.1-1',	'2.1-11',	'2.1-3',	'2.1-4',	'2.1-5',	'2.1-6',	'2.1-7',	'2.1-8',	'2.1-9',	'2.2-1',	'2.2-2',	'2.2-4', '3.0-2',	'3.0-4',	'3.0-8',	'3.2-0',	'3.2a-1',	'3.2a-3',	'3.2a-5',	'3.2a-6',	'3.2a-9',	'3.3-1',	'3.3-3',	'3.3-4',	'4.0-0',	'4.4-0',	'4.6a-1-1',	'4.6a-1-10',	'4.6a-1-11',	'4.6a-1-12',	'4.6a-1-13',	'4.6a-1-14',	'4.6a-1-15',	'4.6a-1-16',	'4.6a-1-17',	'4.6a-1-18',	'4.6a-1-19',	'4.6a-1-2',	'4.6a-1-20',	'4.6a-1-22',	'4.6a-1-23',	'4.6a-1-25',	'4.6a-1-26',	'4.6a-1-27',	'4.6a-1-28',	'4.6a-1-29',	'4.6a-1-3',	'4.6a-1-30',	'4.6a-1-31',	'4.6a-1-4',	'4.6a-1-5',	'4.6a-1-6',	'4.6a-1-7',	'4.6a-1-8',	'4.6a-1-9',	'4.6a-2-1',	'4.6a-2-10',	'4.6a-2-11',	'4.6a-2-12',	'4.6a-2-13',	'4.6a-2-14',	'4.6a-2-15',	'4.6a-2-16',	'4.6a-2-17',	'4.6a-2-18',	'4.6a-2-19',	'4.6a-2-2',	'4.6a-2-20',	'4.6a-2-22',	'4.6a-2-23',	'4.6a-2-25',	'4.6a-2-26',	'4.6a-2-27',	'4.6a-2-28',	'4.6a-2-29',	'4.6a-2-3',	'4.6a-2-30',	'4.6a-2-31',	'4.6a-2-4',	'4.6a-2-5',	'4.6a-2-6',	'4.6a-2-7',	'4.6a-2-8',	'4.6a-2-9',	'4.6a-3-1',	'4.6a-3-10',	'4.6a-3-11',	'4.6a-3-12',	'4.6a-3-13',	'4.6a-3-14',	'4.6a-3-15',	'4.6a-3-16',	'4.6a-3-17',	'4.6a-3-18',	'4.6a-3-19',	'4.6a-3-2',	'4.6a-3-20',	'4.6a-3-22',	'4.6a-3-23',	'4.6a-3-25',	'4.6a-3-26',	'4.6a-3-27',	'4.6a-3-28',	'4.6a-3-29',	'4.6a-3-3',	'4.6a-3-30',	'4.6a-3-31',	'4.6a-3-4',	'4.6a-3-5',	'4.6a-3-6',	'4.6a-3-7',	'4.6a-3-8',	'4.6a-3-9',	'4.6a-4-1',	'4.6a-4-10',	'4.6a-4-11',	'4.6a-4-12',	'4.6a-4-13',	'4.6a-4-14',	'4.6a-4-15',	'4.6a-4-16',	'4.6a-4-17',	'4.6a-4-18',	'4.6a-4-19',	'4.6a-4-2',	'4.6a-4-20',	'4.6a-4-22',	'4.6a-4-23',	'4.6a-4-25',	'4.6a-4-26',	'4.6a-4-27',	'4.6a-4-28',	'4.6a-4-29',	'4.6a-4-3',	'4.6a-4-30',	'4.6a-4-31',	'4.6a-4-4',	'4.6a-4-5',	'4.6a-4-6',	'4.6a-4-7',	'4.6a-4-8',	'4.6a-4-9',	'4.6a-5-1',	'4.6a-5-10',	'4.6a-5-11',	'4.6a-5-12',	'4.6a-5-13',	'4.6a-5-14',	'4.6a-5-15',	'4.6a-5-16',	'4.6a-5-17',	'4.6a-5-18',	'4.6a-5-19',	'4.6a-5-2',	'4.6a-5-20',	'4.6a-5-22',	'4.6a-5-23',	'4.6a-5-25',	'4.6a-5-26',	'4.6a-5-27',	'4.6a-5-28',	'4.6a-5-29',	'4.6a-5-3',	'4.6a-5-30',	'4.6a-5-31',	'4.6a-5-4',	'4.6a-5-5',	'4.6a-5-6',	'4.6a-5-7',	'4.6a-5-8',	'4.6a-5-9',	'4.6a-6-1',	'4.6a-6-10',	'4.6a-6-11',	'4.6a-6-12',	'4.6a-6-13',	'4.6a-6-14',	'4.6a-6-15',	'4.6a-6-16',	'4.6a-6-17',	'4.6a-6-18',	'4.6a-6-19',	'4.6a-6-2',	'4.6a-6-20',	'4.6a-6-22',	'4.6a-6-23',	'4.6a-6-25',	'4.6a-6-26',	'4.6a-6-27',	'4.6a-6-28',	'4.6a-6-29',	'4.6a-6-3',	'4.6a-6-30',	'4.6a-6-31',	'4.6a-6-4',	'4.6a-6-5',	'4.6a-6-6',	'4.6a-6-7',	'4.6a-6-8',	'4.6a-6-9',	'4.6b-1-1',	'4.6b-1-10',	'4.6b-1-11',	'4.6b-1-12',	'4.6b-1-13',	'4.6b-1-14',	'4.6b-1-15',	'4.6b-1-16',	'4.6b-1-17',	'4.6b-1-2',	'4.6b-1-3',	'4.6b-1-4',	'4.6b-1-5',	'4.6b-1-6',	'4.6b-1-7',	'4.6b-1-8',	'4.6b-1-9',	'4.6c-1-1',	'4.6c-13-1',	'4.6c-3-1',	'4.6c-8-1',	'4.6d-1',	'4.6d-2',	'4.6d-3',	'4.6d-4',	'4.6e-1',	'4.6e-2',	'4.6e-3',	'4.6e-4',	'4.6f-1',	'4.6f-2',	'4.6f-3',	'4.6f-4',	'4.9-1-1',	'5.0-0',	'5.0a-1',	'5.0a-11',	'5.0a-3',	'5.0a-5',	'5.0a-6',	'5.0a-7',	'5.0a-8',	'5.0a-9',	'5.0b-1',	'5.0b-10',	'5.0b-3',	'5.0b-5',	'5.0b-7',	'5.0b-8',	'5.0b-9',	'5.0c-1',	'5.0c-10',	'5.0c-11',	'5.0c-12',	'5.0c-13',	'5.0c-3',	'5.0c-5',	'5.0c-6',	'5.0c-7',	'5.0c-8',	'5.0c-9',	'5.0d-1',	'5.0d-10',	'5.0d-11',	'5.0d-3',	'5.0d-5',	'5.0d-6',	'5.0d-7',	'5.0d-8',	'5.0d-9',	'5.4-7',	'5.4-8',	'5.5-0',	'6.0-1',	'6.0-2',	'6.13-1-1',	'6.2-0',	'6.2a-1',	'6.2a-3',	'6.5-3',	'6.5-4',	'6.5-7',	'6.5-9',	'7.0-0',	'7.2-0',	'7.3-1-1',	'7.4-0',	'7.6-2-1',	'7.6-3-1',	'7.6-4-1',	'7.7-0',	'7.7a-2',	'8.0-0',	'8.0a-3',	'8.0a-4',	'8.0a-6',	'8.0a-7',	'8.0a-9',	'8.1-1-1',	'8.1-11-1',	'8.1-2-1',	'8.1-3-1',	'8.1-4-1',	'8.1-5-1',	'8.1-6-1',	'8.1-7-1',	'8.1-8-1',	'8.1-9-1',	'8.2-1-3',	'9.1-1-1',	'9.1-1-2',	'9.1-1-3',	'9.1-1-4',	'9.1-3-1',	'9.1-3-2',	'9.1-3-3',	'9.1-3-4', '0.5-1-1']

# 2022 -- 372 questions
q_tokeep_2022 = ['0.1-1-1',	'0.1-4-1',	'0.1-5-1',	'0.1-6-1',	'0.1-8-1',	'0.3-3-1', '0.3-3-2', '0.3-3-3', '0.3-4-1',	'0.5-1',	'0.5-3',	'0.5-4',	'1.1-0',	'1.1a-1',	'1.1a-3',	'1.1a-5',	'1.1a-6',	'1.2-1',	'1.2-10',	'1.2-11',	'1.2-2',	'1.2-3',	'1.2-4',	'1.2-7',	'1.2-8',	'1.2-9',	'1.3-1',	'1.3-2',	'1.3-3',	'2.1-0',	'2.1b-4',	'2.1c-1-1',	'2.1c-1-10',	'2.1c-1-11',	'2.1c-1-12',	'2.1c-1-13',	'2.1c-1-14',	'2.1c-1-15',	'2.1c-1-16',	'2.1c-1-17',	'2.1c-1-2',	'2.1c-1-3',	'2.1c-1-4',	'2.1c-1-5',	'2.1c-1-6',	'2.1c-1-7',	'2.1c-1-8',	'2.1c-1-9',	'2.1d-1-1',	'2.1d-1-10',	'2.1d-1-11',	'2.1d-1-12',	'2.1d-1-13',	'2.1d-1-14',	'2.1d-1-15',	'2.1d-1-16',	'2.1d-1-17',	'2.1d-1-18',	'2.1d-1-19',	'2.1d-1-2',	'2.1d-1-20',	'2.1d-1-22',	'2.1d-1-23',	'2.1d-1-25',	'2.1d-1-26',	'2.1d-1-27',	'2.1d-1-28',	'2.1d-1-29',	'2.1d-1-3',	'2.1d-1-30',	'2.1d-1-31',	'2.1d-1-4',	'2.1d-1-5',	'2.1d-1-6',	'2.1d-1-7',	'2.1d-1-8',	'2.1d-1-9',	'2.1d-2-1',	'2.1d-2-10',	'2.1d-2-11',	'2.1d-2-12',	'2.1d-2-13',	'2.1d-2-14',	'2.1d-2-15',	'2.1d-2-16',	'2.1d-2-17',	'2.1d-2-18',	'2.1d-2-19',	'2.1d-2-2',	'2.1d-2-20',	'2.1d-2-22',	'2.1d-2-23',	'2.1d-2-25',	'2.1d-2-26',	'2.1d-2-27',	'2.1d-2-28',	'2.1d-2-29',	'2.1d-2-3',	'2.1d-2-30',	'2.1d-2-31',	'2.1d-2-4',	'2.1d-2-5',	'2.1d-2-6',	'2.1d-2-7',	'2.1d-2-8',	'2.1d-2-9',	'2.1d-3-1',	'2.1d-3-10',	'2.1d-3-11',	'2.1d-3-12',	'2.1d-3-13',	'2.1d-3-14',	'2.1d-3-15',	'2.1d-3-16',	'2.1d-3-17',	'2.1d-3-18',	'2.1d-3-19',	'2.1d-3-2',	'2.1d-3-20',	'2.1d-3-22',	'2.1d-3-23',	'2.1d-3-25',	'2.1d-3-26',	'2.1d-3-27',	'2.1d-3-28',	'2.1d-3-29',	'2.1d-3-3',	'2.1d-3-30',	'2.1d-3-31',	'2.1d-3-4',	'2.1d-3-5',	'2.1d-3-6',	'2.1d-3-7',	'2.1d-3-8',	'2.1d-3-9',	'2.1d-4-1',	'2.1d-4-10',	'2.1d-4-11',	'2.1d-4-12',	'2.1d-4-13',	'2.1d-4-14',	'2.1d-4-15',	'2.1d-4-16',	'2.1d-4-17',	'2.1d-4-18',	'2.1d-4-19',	'2.1d-4-2',	'2.1d-4-20',	'2.1d-4-22',	'2.1d-4-23',	'2.1d-4-25',	'2.1d-4-26',	'2.1d-4-27',	'2.1d-4-28',	'2.1d-4-29',	'2.1d-4-3',	'2.1d-4-30',	'2.1d-4-31',	'2.1d-4-4',	'2.1d-4-5',	'2.1d-4-6',	'2.1d-4-7',	'2.1d-4-8',	'2.1d-4-9',	'2.1d-5-1',	'2.1d-5-10',	'2.1d-5-11',	'2.1d-5-12',	'2.1d-5-13',	'2.1d-5-14',	'2.1d-5-15',	'2.1d-5-16',	'2.1d-5-17',	'2.1d-5-18',	'2.1d-5-19',	'2.1d-5-2',	'2.1d-5-20',	'2.1d-5-22',	'2.1d-5-23',	'2.1d-5-25',	'2.1d-5-26',	'2.1d-5-27',	'2.1d-5-28',	'2.1d-5-29',	'2.1d-5-3',	'2.1d-5-30',	'2.1d-5-31',	'2.1d-5-4',	'2.1d-5-5',	'2.1d-5-6',	'2.1d-5-7',	'2.1d-5-8',	'2.1d-5-9',	'2.1d-6-1',	'2.1d-6-10',	'2.1d-6-11',	'2.1d-6-12',	'2.1d-6-13',	'2.1d-6-14',	'2.1d-6-15',	'2.1d-6-16',	'2.1d-6-17',	'2.1d-6-18',	'2.1d-6-19',	'2.1d-6-2',	'2.1d-6-20',	'2.1d-6-22',	'2.1d-6-23',	'2.1d-6-25',	'2.1d-6-26',	'2.1d-6-27',	'2.1d-6-28',	'2.1d-6-29',	'2.1d-6-3',	'2.1d-6-30',	'2.1d-6-31',	'2.1d-6-4',	'2.1d-6-5',	'2.1d-6-6',	'2.1d-6-7',	'2.1d-6-8',	'2.1d-6-9',	'2.1e-1',	'2.1e-2',	'2.1e-3',	'2.1e-4',	'2.2-1-1',	'2.3-0',	'2.3a-3-1',	'2.3a-5-1',	'2.3b-1-1',	'2.3b-2-1',	'2.3b-3-1',	'2.3b-4-1',	'3.1-10-1',	'3.1-11-1',	'3.11-2-1',	'3.1-14-1',	'3.1-3-1',	'3.14-1',	'3.1-4-1',	'3.1-5-1',	'3.1-6-1',	'3.1-7-1',	'3.1-8-1',	'3.1-9-1',	'3.2-1-3',	'3.5-10-1',	'3.5-2-1',	'3.5-3-1',	'3.5-5-1',	'3.5-6-1',	'3.5-7-1',	'3.5-8-1',	'3.5-9-1',	'3.6-1-2',	'3.6-1-3',	'3.6-1-4',	'3.6-1-5',	'3.6-1-6',	'3.6-2-2',	'3.6-2-3',	'3.6-2-4',	'3.6-2-5',	'3.6-2-6',	'3.6-3-2',	'3.6-3-3',	'3.6-3-4',	'3.6-3-5',	'3.6-3-6',	'3.6-4-2',	'3.6-4-3',	'3.6-4-4',	'3.6-4-5',	'3.6-4-6',	'3.6-5-2',	'3.6-5-3',	'3.6-5-4',	'3.6-5-5',	'3.6-5-6',	'3.7-2-1',	'3.7-2-2',	'4.1a-2',	'4.1a-5',	'4.1a-6',	'5.1a-10-1',	'5.1a-10-3',	'5.1a-10-4',	'5.1a-1-1',	'5.1a-11-3',	'5.1a-1-2',	'5.1a-12-1',	'5.1a-12-2',	'5.1a-12-3',	'5.1a-12-4',	'5.1a-1-3',	'5.1a-13-2',	'5.1a-13-3',	'5.1a-13-4',	'5.1a-1-4',	'5.1a-14-1',	'5.1a-14-2',	'5.1a-14-3',	'5.1a-14-4',	'5.1a-16-2',	'5.1a-30-3',	'5.1a-3-1',	'5.1a-3-2',	'5.1a-3-3',	'5.1a-3-4',	'5.1a-7-1',	'5.1a-7-2',	'5.1a-7-3',	'5.1a-7-4',	'5.1a-9-1',	'5.1a-9-3',	'5.1a-9-4',	'6.1-1-1',	'6.1-1-10',	'6.1-1-11',	'6.1-1-12',	'6.1-1-13',	'6.1-1-14',	'6.1-1-2',	'6.1-1-3',	'6.1-1-4',	'6.1-1-5',	'6.1-1-6',	'6.1-1-7',	'6.1-1-8',	'6.1-1-9',	'7.1-0',	'7.1-1',	'7.1a-1',	'7.1a-11',	'7.1a-14',	'7.1a-2',	'7.1a-4',	'7.4-3',	'7.4-4',	'7.4-6',	'7.4-8',	'9.1-6',	'9.1-7', '0.1-6-1']



# COMMAND ----------

# Filter all dataframes

# filter out records by questions in q_tokeep_2018
df_cities_2018 = df_cities_2018.filter(df_cities_2018.q_id.isin(q_tokeep_2018))
df_cities_2018 = df_cities_2018.drop("q_c","q_c_r","q_c_text", "q_c_r_text", "q_text")

# filter out records by questions in q_tokeep_2019
df_cities_2019 = df_cities_2019.filter(df_cities_2019.q_id.isin(q_tokeep_2019))
df_cities_2019 = df_cities_2019.drop("q_c","q_c_r","q_c_text", "q_c_r_text", "q_text")

# filter out records by questions in q_tokeep_2020
df_cities_2020 = df_cities_2020.filter(df_cities_2020.q_id.isin(q_tokeep_2020))
df_cities_2020 = df_cities_2020.drop("q_c","q_c_r","q_c_text", "q_c_r_text", "q_text")

# filter out records by questions in q_tokeep_2021
df_cities_2021 = df_cities_2021.filter(df_cities_2021.q_id.isin(q_tokeep_2021))
df_cities_2021 = df_cities_2021.drop("q_c","q_c_r","q_c_text", "q_c_r_text", "q_text")

# filter out records by questions in q_tokeep_2022
df_cities_2022 = df_cities_2022.filter(df_cities_2022.q_id.isin(q_tokeep_2022))
df_cities_2022 = df_cities_2022.drop("q_c","q_c_r","q_c_text", "q_c_r_text", "q_text")

# COMMAND ----------

# Replacement dictionaries for each dataframe

replace_2018to2021 = {'0.1-1-1' : '0.1-1-1',	'0.6-2-1' : '0.5-2-1', '0.6-1-1' : '0.5-1-1',	'0.6-4-1' : '0.5-3-1',	'0.8-2-1' : '0.6-1-1',	'10.0-1-1' : '9.1-1-1',	'10.0-1-2' : '9.1-1-2',	'10.0-1-3' : '9.1-1-3',	'10.0-1-4' : '9.1-1-4',	'10.0-2-1' : '9.1-3-1',	'10.0-2-2' : '9.1-3-2',	'10.0-2-3' : '9.1-3-3',	'10.0-2-4' : '9.1-3-4',	'11.0-1-1' : '10.1-1-1',	'11.0-2-1' : '10.1-2-1',	'11.0-3-1' : '10.1-3-1',	'11.0-4-1' : '10.1-4-1',	'11.0-5-1' : '10.1-5-1',	'11.0-6-1' : '10.1-6-1',	'11.0-7-1' : '10.1-7-1',	'11.0-8-1' : '10.1-9-1',	'11.4-1-1' : '10.3-1-1',	'11.4-1-2' : '10.3-1-2',	'11.4-1-3' : '10.3-1-3',	'11.4-1-4' : '10.3-1-4',	'11.4-1-5' : '10.3-1-5',	'11.4-2-1' : '10.3-2-1',	'11.4-2-2' : '10.3-2-2',	'11.4-2-3' : '10.3-2-3',	'11.4-2-4' : '10.3-2-4',	'11.4-2-5' : '10.3-2-5',	'11.4-3-1' : '10.3-3-1',	'11.4-3-2' : '10.3-3-2',	'11.4-3-3' : '10.3-3-3',	'11.4-3-4' : '10.3-3-4',	'11.4-3-5' : '10.3-3-5',	'11.4-4-1' : '10.3-4-1',	'11.4-4-2' : '10.3-4-2',	'11.4-4-3' : '10.3-4-3',	'11.4-4-4' : '10.3-4-4',	'11.4-4-5' : '10.3-4-5',	'11.4-5-1' : '10.3-5-1',	'11.4-5-2' : '10.3-5-2',	'11.4-5-3' : '10.3-5-3',	'11.4-5-4' : '10.3-5-4',	'11.4-5-5' : '10.3-5-5',	'12.0-0' : '11.0-0',	'14.0-1-1' : '13.3-1-1',	'14.0-1-2' : '13.3-1-2',	'14.0-1-3' : '13.3-1-3',	'14.0-1-4' : '13.3-1-4',	'14.0-1-5' : '13.3-1-5',	'14.0-1-6' : '13.3-1-6', '14.5-2-1': '13.2-0', '14.5-2-2': '13.2-0', '14.5-2-3': '13.2-0', '14.5-2-4': '13.2-0', '15.0-0' : '14.0-0',	'15.1-0' : '14.1-0',	'15.3a-1' : '14.2a-1',	'15.3a-2' : '14.2a-2',	'15.3a-3' : '14.2a-3',	'15.3a-4' : '14.2a-5',	'15.4-2' : '14.3-2',	'15.4-3' : '14.3-4',	'2.0-0' : '2.0-0',	'2.0b-3' : '2.0b-2',	'2.0b-4' : '2.0b-4',	'2.0b-6' : '2.0b-6',	'2.0b-7' : '2.0b-7',	'2.2a-1' : '2.1-1',	'2.2a-4' : '2.1-11',	'2.2a-5' : '2.1-3',	'2.2a-3' : '2.1-4',	'2.2a-7' : '2.1-8',	'2.2a-8' : '2.1-9',	'2.2a-9' : '2.1-6',	'2.4-1' : '2.2-1',	'2.4-2' : '2.2-2',	'2.4-3' : '2.2-4',	'3.1-0' : '3.2-0',	'3.1a-1' : '3.2a-1',	'3.1a-2' : '3.2a-5',	'3.1a-4' : '3.2a-6',	'3.3-2' : '3.0-2',	'3.3-3' : '3.0-4',	'3.3-4' : '3.0-8',	'3.4-1' : '3.3-1',	'3.4-2' : '3.3-3',	'3.4-3' : '3.3-4',	'5.0a-1' : '6.0-1',	'5.0a-2' : '6.0-2',	'5.1-0' : '6.2-0',	'5.10-0' : '6.13-1-1',	'5.1a-1' : '6.2a-1',	'5.1a-2' : '6.2a-3',	'5.2-2' : '6.5-3',	'5.2-3' : '6.5-4',	'5.2-4' : '6.5-7',	'5.2-6' : '6.5-9',	'6.0-0' : '7.0-0',	'6.2-0' : '7.2-0',	'6.3-1-1' : '7.3-1-1',	'6.4-0' : '7.4-0',	'6.7-2-1' : '7.6-2-1',	'6.7-3-1' : '7.6-3-1',	'6.7-4-1' : '7.6-4-1',	'6.8-0' : '7.7-0',	'6.8a-2' : '7.7a-2',	'7.0-0' : '4.0-0', '7.3a-1-1' : '4.6b-1-1',	'7.3a-1-10' : '4.6b-1-10',	'7.3a-1-11' : '4.6b-1-11',	'7.3a-1-12' : '4.6b-1-12',	'7.3a-1-13' : '4.6b-1-13',	'7.3a-1-14' : '4.6b-1-14',	'7.3a-1-15' : '4.6b-1-15',	'7.3a-1-16' : '4.6b-1-16',	'7.3a-1-17' : '4.6b-1-17',	'7.3a-1-2' : '4.6b-1-2',	'7.3a-1-3' : '4.6b-1-3',	'7.3a-1-4' : '4.6b-1-4',	'7.3a-1-5' : '4.6b-1-5',	'7.3a-1-6' : '4.6b-1-6',	'7.3a-1-7' : '4.6b-1-7',	'7.3a-1-8' : '4.6b-1-8',	'7.3a-1-9' : '4.6b-1-9',	'7.4b-1-1' : '4.6c-1-1',	'7.4b-13-1' : '4.6c-13-1',	'7.4b-3-1' : '4.6c-3-1',	'7.4b-8-1' : '4.6c-8-1', '7.6-1-1' : '4.6a-1-1',	'7.6-1-10' : '4.6a-1-17',	'7.6-1-13' : '4.6a-1-26',	'7.6-1-14' : '4.6a-1-28',	'7.6-1-15' : '4.6a-1-30',	'7.6-1-2' : '4.6a-1-2',	'7.6-1-3' : '4.6a-1-3',	'7.6-1-5' : '4.6a-1-5',	'7.6-1-6' : '4.6a-1-8',	'7.6-1-7' : '4.6a-1-9',	'7.6-1-8' : '4.6a-1-13',	'7.6-1-9' : '4.6a-1-14',	'7.6-2-1' : '4.6a-3-1',	'7.6-2-10' : '4.6a-3-17',	'7.6-2-11' : '4.6a-3-19',	'7.6-2-13' : '4.6a-3-26',	'7.6-2-14' : '4.6a-3-28',	'7.6-2-2' : '4.6a-3-2',	'7.6-2-3' : '4.6a-3-3',	'7.6-2-4' : '4.6a-3-4',	'7.6-2-6' : '4.6a-3-8',	'7.6-2-7' : '4.6a-3-9',	'7.6-2-8' : '4.6a-3-13', '7.6a-1' : '4.6d-1','7.6c-1' : '4.6d-1','7.6d-1' : '4.6d-1','7.6a-2' : '4.6d-2','7.6c-2' : '4.6d-2','7.6d-2' : '4.6d-2','7.6a-3' : '4.6d-3','7.6c-3' : '4.6d-3','7.6d-3' : '4.6d-3','7.6a-4' : '4.6d-4','7.6c-4' : '4.6d-4','7.6d-4' : '4.6d-4', '8.1-3' : '5.4-7',	'8.2-0' : '5.5-0',	'8.3-0' : '5.0-0',	'8.3a-1' : '5.0a-1',	'8.3a-2' : '5.0a-6',	'8.3a-3' : '5.0a-5',	'8.3a-4' : '5.0a-7',	'8.3a-5' : '5.0a-8',	'8.3a-6' : '5.0a-9',	'8.3a-7' : '5.0a-11',	'8.3b-1' : '5.0c-1',	'8.3b-2' : '5.0c-6',	'8.3b-3' : '5.0c-7',	'8.3b-4' : '5.0c-5',	'8.3b-5' : '5.0c-8',	'8.3b-6' : '5.0c-10',	'8.3b-7' : '5.0c-11',	'8.3b-8' : '5.0c-13',	'8.3c-1' : '5.0d-1',	'8.3c-2' : '5.0d-6',	'8.3c-3' : '5.0d-5',	'8.3c-4' : '5.0d-7',	'8.3c-5' : '5.0d-8',	'8.3c-6' : '5.0d-9',	'8.3c-7' : '5.0d-10',	'8.3c-8' : '5.0d-11',	'9.0-10-1' : '8.1-11-1',	'9.0-1-1' : '8.1-1-1',	'9.0-2-1' : '8.1-2-1',	'9.0-3-1' : '8.1-3-1',	'9.0-4-1' : '8.1-4-1',	'9.0-5-1' : '8.1-5-1',	'9.0-6-1' : '8.1-6-1',	'9.0-7-1' : '8.1-7-1',	'9.0-8-1' : '8.1-8-1',	'9.0-9-1' : '8.1-9-1',	'9.2-0' : '8.0-0',	'9.2a-3' : '8.0a-4',	'9.2a-5' : '8.0a-6',	'9.2a-6' : '8.0a-7',	'9.2a-8' : '8.0a-9', '7.6-1-4' : '4.6a-1-4', '7.6-1-11' : '4.6a-1-19', '9.2a-2' : '8.0a-3'}

replace_2019to2021 = {'0.1-1-1' : '0.1-1-1',	'0.5-2-1' : '0.5-2-1', '0.5-1-1' : '0.5-1-1',	'0.5-3-1' : '0.5-3-1',	'0.6-1-1' : '0.6-1-1',	'10.1-1-1' : '10.1-1-1',	'10.1-2-1' : '10.1-2-1',	'10.1-3-1' : '10.1-3-1',	'10.1-4-1' : '10.1-4-1',	'10.1-5-1' : '10.1-5-1',	'10.1-6-1' : '10.1-6-1',	'10.1-7-1' : '10.1-7-1',	'10.1-8-1' : '10.1-9-1',	'10.5-1-1' : '10.3-1-1',	'10.5-1-2' : '10.3-1-2',	'10.5-1-3' : '10.3-1-3',	'10.5-1-4' : '10.3-1-4',	'10.5-1-5' : '10.3-1-5',	'10.5-2-1' : '10.3-2-1',	'10.5-2-2' : '10.3-2-2',	'10.5-2-3' : '10.3-2-3',	'10.5-2-4' : '10.3-2-4',	'10.5-2-5' : '10.3-2-5',	'10.5-3-1' : '10.3-3-1',	'10.5-3-2' : '10.3-3-2',	'10.5-3-3' : '10.3-3-3',	'10.5-3-4' : '10.3-3-4',	'10.5-3-5' : '10.3-3-5',	'10.5-4-1' : '10.3-4-1',	'10.5-4-2' : '10.3-4-2',	'10.5-4-3' : '10.3-4-3',	'10.5-4-4' : '10.3-4-4',	'10.5-4-5' : '10.3-4-5',	'10.5-5-1' : '10.3-5-1',	'10.5-5-2' : '10.3-5-2',	'10.5-5-3' : '10.3-5-3',	'10.5-5-4' : '10.3-5-4',	'10.5-5-5' : '10.3-5-5',	'11.0-0' : '11.0-0',	'13.2-0' : '13.2-0',	'13.3-1-1' : '13.3-1-1',	'13.3-1-2' : '13.3-1-2',	'13.3-1-3' : '13.3-1-3',	'13.3-1-4' : '13.3-1-4',	'13.3-1-5' : '13.3-1-5',	'13.3-1-6' : '13.3-1-6',	'14.0-0' : '14.0-0',	'14.2-0' : '14.1-0',	'14.3a-1' : '14.2a-1',	'14.3a-2' : '14.2a-2',	'14.3a-3' : '14.2a-3',	'14.3a-4' : '14.2a-5',	'14.4-2' : '14.3-2',	'14.4-4' : '14.3-4',	'2.0-0' : '2.0-0',	'2.0b-3' : '2.0b-2',	'2.0b-4' : '2.0b-4',	'2.0b-7' : '2.0b-6',	'2.0b-8' : '2.0b-7',	'2.1-1' : '2.1-1',	'2.1-10' : '2.1-7',	'2.1-3' : '2.1-3',	'2.1-4' : '2.1-4',	'2.1-5' : '2.1-5',	'2.1-6' : '2.1-8',	'2.1-7' : '2.1-9',	'2.1-8' : '2.1-11',	'2.1-9' : '2.1-6',	'2.2-1' : '2.2-1',	'2.2-2' : '2.2-2',	'2.2-3' : '2.2-4',	'3.0-2' : '3.0-2',	'3.0-4' : '3.0-4',	'3.0-6' : '3.0-8',	'3.1-0' : '3.2-0',	'3.1a-1' : '3.2a-1',	'3.1a-2' : '3.2a-3',	'3.1a-3' : '3.2a-5',	'3.1a-4' : '3.2a-6',	'3.1a-7' : '3.2a-9',	'3.2-1' : '3.3-1',	'3.2-2' : '3.3-3',	'3.2-3' : '3.3-4',	'4.0-0' : '4.0-0',	'4.4-0' : '4.4-0',	'4.6a-1-1' : '4.6a-1-1',	'4.6a-1-10' : '4.6a-1-10',	'4.6a-1-11' : '4.6a-1-11',	'4.6a-1-12' : '4.6a-1-12',	'4.6a-1-13' : '4.6a-1-13',	'4.6a-1-14' : '4.6a-1-14',	'4.6a-1-15' : '4.6a-1-15',	'4.6a-1-16' : '4.6a-1-16',	'4.6a-1-17' : '4.6a-1-17',	'4.6a-1-18' : '4.6a-1-18',	'4.6a-1-19' : '4.6a-1-19',	'4.6a-1-2' : '4.6a-1-2',	'4.6a-1-20' : '4.6a-1-20',	'4.6a-1-22' : '4.6a-1-22',	'4.6a-1-23' : '4.6a-1-23',	'4.6a-1-25' : '4.6a-1-25',	'4.6a-1-26' : '4.6a-1-26',	'4.6a-1-27' : '4.6a-1-27',	'4.6a-1-28' : '4.6a-1-28',	'4.6a-1-29' : '4.6a-1-29',	'4.6a-1-3' : '4.6a-1-3',	'4.6a-1-30' : '4.6a-1-30',	'4.6a-1-31' : '4.6a-1-31',	'4.6a-1-4' : '4.6a-1-4',	'4.6a-1-5' : '4.6a-1-5',	'4.6a-1-6' : '4.6a-1-6',	'4.6a-1-7' : '4.6a-1-7',	'4.6a-1-8' : '4.6a-1-8',	'4.6a-1-9' : '4.6a-1-9',	'4.6a-2-1' : '4.6a-2-1',	'4.6a-2-10' : '4.6a-2-10',	'4.6a-2-11' : '4.6a-2-11',	'4.6a-2-12' : '4.6a-2-12',	'4.6a-2-13' : '4.6a-2-13',	'4.6a-2-14' : '4.6a-2-14',	'4.6a-2-15' : '4.6a-2-15',	'4.6a-2-16' : '4.6a-2-16',	'4.6a-2-17' : '4.6a-2-17',	'4.6a-2-18' : '4.6a-2-18',	'4.6a-2-19' : '4.6a-2-19',	'4.6a-2-2' : '4.6a-2-2',	'4.6a-2-20' : '4.6a-2-20',	'4.6a-2-22' : '4.6a-2-22',	'4.6a-2-23' : '4.6a-2-23',	'4.6a-2-25' : '4.6a-2-25',	'4.6a-2-26' : '4.6a-2-26',	'4.6a-2-27' : '4.6a-2-27',	'4.6a-2-28' : '4.6a-2-28',	'4.6a-2-29' : '4.6a-2-29',	'4.6a-2-3' : '4.6a-2-3',	'4.6a-2-30' : '4.6a-2-30',	'4.6a-2-31' : '4.6a-2-31',	'4.6a-2-4' : '4.6a-2-4',	'4.6a-2-5' : '4.6a-2-5',	'4.6a-2-6' : '4.6a-2-6',	'4.6a-2-7' : '4.6a-2-7',	'4.6a-2-8' : '4.6a-2-8',	'4.6a-2-9' : '4.6a-2-9',	'4.6a-3-1' : '4.6a-3-1',	'4.6a-3-10' : '4.6a-3-10',	'4.6a-3-11' : '4.6a-3-11',	'4.6a-3-12' : '4.6a-3-12',	'4.6a-3-13' : '4.6a-3-13',	'4.6a-3-14' : '4.6a-3-14',	'4.6a-3-15' : '4.6a-3-15',	'4.6a-3-16' : '4.6a-3-16',	'4.6a-3-17' : '4.6a-3-17',	'4.6a-3-18' : '4.6a-3-18',	'4.6a-3-19' : '4.6a-3-19',	'4.6a-3-2' : '4.6a-3-2',	'4.6a-3-20' : '4.6a-3-20',	'4.6a-3-22' : '4.6a-3-22',	'4.6a-3-23' : '4.6a-3-23',	'4.6a-3-25' : '4.6a-3-25',	'4.6a-3-26' : '4.6a-3-26',	'4.6a-3-27' : '4.6a-3-27',	'4.6a-3-28' : '4.6a-3-28',	'4.6a-3-29' : '4.6a-3-29',	'4.6a-3-3' : '4.6a-3-3',	'4.6a-3-30' : '4.6a-3-30',	'4.6a-3-31' : '4.6a-3-31',	'4.6a-3-4' : '4.6a-3-4',	'4.6a-3-5' : '4.6a-3-5',	'4.6a-3-6' : '4.6a-3-6',	'4.6a-3-7' : '4.6a-3-7',	'4.6a-3-8' : '4.6a-3-8',	'4.6a-3-9' : '4.6a-3-9',	'4.6a-4-1' : '4.6a-4-1',	'4.6a-4-10' : '4.6a-4-10',	'4.6a-4-11' : '4.6a-4-11',	'4.6a-4-12' : '4.6a-4-12',	'4.6a-4-13' : '4.6a-4-13',	'4.6a-4-14' : '4.6a-4-14',	'4.6a-4-15' : '4.6a-4-15',	'4.6a-4-16' : '4.6a-4-16',	'4.6a-4-17' : '4.6a-4-17',	'4.6a-4-18' : '4.6a-4-18',	'4.6a-4-19' : '4.6a-4-19',	'4.6a-4-2' : '4.6a-4-2',	'4.6a-4-20' : '4.6a-4-20',	'4.6a-4-22' : '4.6a-4-22',	'4.6a-4-23' : '4.6a-4-23',	'4.6a-4-25' : '4.6a-4-25',	'4.6a-4-26' : '4.6a-4-26',	'4.6a-4-27' : '4.6a-4-27',	'4.6a-4-28' : '4.6a-4-28',	'4.6a-4-29' : '4.6a-4-29',	'4.6a-4-3' : '4.6a-4-3',	'4.6a-4-30' : '4.6a-4-30',	'4.6a-4-31' : '4.6a-4-31',	'4.6a-4-4' : '4.6a-4-4',	'4.6a-4-5' : '4.6a-4-5',	'4.6a-4-6' : '4.6a-4-6',	'4.6a-4-7' : '4.6a-4-7',	'4.6a-4-8' : '4.6a-4-8',	'4.6a-4-9' : '4.6a-4-9',	'4.6a-5-1' : '4.6a-5-1',	'4.6a-5-10' : '4.6a-5-10',	'4.6a-5-11' : '4.6a-5-11',	'4.6a-5-12' : '4.6a-5-12',	'4.6a-5-13' : '4.6a-5-13',	'4.6a-5-14' : '4.6a-5-14',	'4.6a-5-15' : '4.6a-5-15',	'4.6a-5-16' : '4.6a-5-16',	'4.6a-5-17' : '4.6a-5-17',	'4.6a-5-18' : '4.6a-5-18',	'4.6a-5-19' : '4.6a-5-19',	'4.6a-5-2' : '4.6a-5-2',	'4.6a-5-20' : '4.6a-5-20',	'4.6a-5-22' : '4.6a-5-22',	'4.6a-5-23' : '4.6a-5-23',	'4.6a-5-25' : '4.6a-5-25',	'4.6a-5-26' : '4.6a-5-26',	'4.6a-5-27' : '4.6a-5-27',	'4.6a-5-28' : '4.6a-5-28',	'4.6a-5-29' : '4.6a-5-29',	'4.6a-5-3' : '4.6a-5-3',	'4.6a-5-30' : '4.6a-5-30',	'4.6a-5-31' : '4.6a-5-31',	'4.6a-5-4' : '4.6a-5-4',	'4.6a-5-5' : '4.6a-5-5',	'4.6a-5-6' : '4.6a-5-6',	'4.6a-5-7' : '4.6a-5-7',	'4.6a-5-8' : '4.6a-5-8',	'4.6a-5-9' : '4.6a-5-9',	'4.6a-6-1' : '4.6a-6-1',	'4.6a-6-10' : '4.6a-6-10',	'4.6a-6-11' : '4.6a-6-11',	'4.6a-6-12' : '4.6a-6-12',	'4.6a-6-13' : '4.6a-6-13',	'4.6a-6-14' : '4.6a-6-14',	'4.6a-6-15' : '4.6a-6-15',	'4.6a-6-16' : '4.6a-6-16',	'4.6a-6-17' : '4.6a-6-17',	'4.6a-6-18' : '4.6a-6-18',	'4.6a-6-19' : '4.6a-6-19',	'4.6a-6-2' : '4.6a-6-2',	'4.6a-6-20' : '4.6a-6-20',	'4.6a-6-22' : '4.6a-6-22',	'4.6a-6-23' : '4.6a-6-23',	'4.6a-6-25' : '4.6a-6-25',	'4.6a-6-26' : '4.6a-6-26',	'4.6a-6-27' : '4.6a-6-27',	'4.6a-6-28' : '4.6a-6-28',	'4.6a-6-29' : '4.6a-6-29',	'4.6a-6-3' : '4.6a-6-3',	'4.6a-6-30' : '4.6a-6-30',	'4.6a-6-31' : '4.6a-6-31',	'4.6a-6-4' : '4.6a-6-4',	'4.6a-6-5' : '4.6a-6-5',	'4.6a-6-6' : '4.6a-6-6',	'4.6a-6-7' : '4.6a-6-7',	'4.6a-6-8' : '4.6a-6-8',	'4.6a-6-9' : '4.6a-6-9',	'4.6b-1-1' : '4.6b-1-1',	'4.6b-1-10' : '4.6b-1-10',	'4.6b-1-11' : '4.6b-1-11',	'4.6b-1-12' : '4.6b-1-12',	'4.6b-1-13' : '4.6b-1-13',	'4.6b-1-14' : '4.6b-1-14',	'4.6b-1-15' : '4.6b-1-15',	'4.6b-1-16' : '4.6b-1-16',	'4.6b-1-17' : '4.6b-1-17',	'4.6b-1-2' : '4.6b-1-2',	'4.6b-1-3' : '4.6b-1-3',	'4.6b-1-4' : '4.6b-1-4',	'4.6b-1-5' : '4.6b-1-5',	'4.6b-1-6' : '4.6b-1-6',	'4.6b-1-7' : '4.6b-1-7',	'4.6b-1-8' : '4.6b-1-8',	'4.6b-1-9' : '4.6b-1-9',	'4.6c-1-1' : '4.6c-1-1',	'4.6c-13-1' : '4.6c-13-1',	'4.6c-3-1' : '4.6c-3-1',	'4.6c-8-1' : '4.6c-8-1',	'4.6d-1' : '4.6d-1',	'4.6d-2' : '4.6d-2',	'4.6d-3' : '4.6d-3',	'4.6d-4' :'4.6d-4',	'4.6e-1' : '4.6d-1','4.6e-2' : '4.6d-2','4.6e-3' : '4.6d-3','4.6e-4' : '4.6d-4','4.6f-1' : '4.6d-1','4.6f-2' : '4.6d-2','4.6f-3' : '4.6d-3','4.6f-4' : '4.6d-4',	'4.9-1-1' : '4.9-1-1',	'5.0-0' : '5.0-0',	'5.0a-1' : '5.0a-1',	'5.0a-10' : '5.0a-11',	'5.0a-3' : '5.0a-3',	'5.0a-4' : '5.0a-5',	'5.0a-5' : '5.0a-6',	'5.0a-6' : '5.0a-7',	'5.0a-7' : '5.0a-8',	'5.0a-8' : '5.0a-9',	'5.0b-1' : '5.0b-1',	'5.0b-3' : '5.0b-3',	'5.0b-4' : '5.0b-5',	'5.0b-5' : '5.0b-7',	'5.0b-6' : '5.0b-8',	'5.0b-7' : '5.0b-9',	'5.0b-8' : '5.0b-10',	'5.0c-1' : '5.0c-1',	'5.0c-10' : '5.0c-11',	'5.0c-11' : '5.0c-12',	'5.0c-12' : '5.0c-13',	'5.0c-3' : '5.0c-3',	'5.0c-4' : '5.0c-5',	'5.0c-5' : '5.0c-6',	'5.0c-6' : '5.0c-7',	'5.0c-7' : '5.0c-8',	'5.0c-8' : '5.0c-9',	'5.0c-9' : '5.0c-10',	'5.0d-1' : '5.0d-1',	'5.0d-10' : '5.0d-11',	'5.0d-3' : '5.0d-3',	'5.0d-4' : '5.0d-5',	'5.0d-5' : '5.0d-6',	'5.0d-6' : '5.0d-7',	'5.0d-7' : '5.0d-8',	'5.0d-8' : '5.0d-9',	'5.0d-9' : '5.0d-10',	'5.4-5' : '5.4-7',	'5.4-6' : '5.4-8',	'5.5-0' : '5.5-0',	'6.0-1' : '6.0-1',	'6.0-2' : '6.0-2',	'6.1-0' : '6.2-0',	'6.11-0' : '6.13-1-1',	'6.1a-1' : '6.2a-1',	'6.1a-2' : '6.2a-3',	'6.2-3' : '6.5-3',	'6.2-4' : '6.5-4',	'6.2-5' : '6.5-7',	'6.2-7' : '6.5-9',	'7.0-0' : '7.0-0',	'7.2-0' : '7.2-0',	'7.3-1-1' : '7.3-1-1',	'7.4-0' : '7.4-0',	'7.6-2-1' : '7.6-2-1',	'7.6-3-1' : '7.6-3-1',	'7.6-4-1' : '7.6-4-1',	'7.7-0' : '7.7-0',	'7.7a-2' : '7.7a-2',	'8.0-0' : '8.0-0',	'8.0a-2' : '8.0a-3',	'8.0a-3' : '8.0a-4',	'8.0a-5' : '8.0a-6',	'8.0a-6' : '8.0a-7',	'8.0a-8' : '8.0a-9',	'8.2-10-1' : '8.1-11-1',	'8.2-1-1' : '8.1-1-1',	'8.2-2-1' : '8.1-2-1',	'8.2-3-1' : '8.1-3-1',	'8.2-4-1' : '8.1-4-1',	'8.2-5-1' : '8.1-5-1',	'8.2-6-1' : '8.1-6-1',	'8.2-7-1' : '8.1-7-1',	'8.2-8-1' : '8.1-8-1',	'8.2-9-1' : '8.1-9-1',	'8.5-1-4' : '8.2-1-3',	'9.1-1-1' : '9.1-1-1',	'9.1-1-2' : '9.1-1-2',	'9.1-1-3' : '9.1-1-3',	'9.1-1-4' : '9.1-1-4',	'9.1-2-1' : '9.1-3-1',	'9.1-2-2' : '9.1-3-2',	'9.1-2-3' : '9.1-3-3',	'9.1-2-4' : '9.1-3-4'}

replace_2020to2021 = {'0.1-1-1' : '0.1-1-1', '0.5-2-1' : '0.5-2-1',  '0.5-1-1' : '0.5-1-1',	'0.5-3-1' : '0.5-3-1',	'0.6-1-1' : '0.6-1-1',	'10.1-1-1' : '10.1-1-1',	'10.1-2-1' : '10.1-2-1',	'10.1-3-1' : '10.1-3-1',	'10.1-4-1' : '10.1-4-1',	'10.1-5-1' : '10.1-5-1',	'10.1-6-1' : '10.1-6-1',	'10.1-7-1' : '10.1-7-1',	'10.1-9-1' : '10.1-9-1',	'10.4-1-1' : '10.3-1-1',	'10.4-1-2' : '10.3-1-2',	'10.4-1-3' : '10.3-1-3',	'10.4-1-4' : '10.3-1-4',	'10.4-1-5' : '10.3-1-5',	'10.4-2-1' : '10.3-2-1',	'10.4-2-2' : '10.3-2-2',	'10.4-2-3' : '10.3-2-3',	'10.4-2-4' : '10.3-2-4',	'10.4-2-5' : '10.3-2-5',	'10.4-3-1' : '10.3-3-1',	'10.4-3-2' : '10.3-3-2',	'10.4-3-3' : '10.3-3-3',	'10.4-3-4' : '10.3-3-4',	'10.4-3-5' : '10.3-3-5',	'10.4-4-1' : '10.3-4-1',	'10.4-4-2' : '10.3-4-2',	'10.4-4-3' : '10.3-4-3',	'10.4-4-4' : '10.3-4-4',	'10.4-4-5' : '10.3-4-5',	'10.4-5-1' : '10.3-5-1',	'10.4-5-2' : '10.3-5-2',	'10.4-5-3' : '10.3-5-3',	'10.4-5-4' : '10.3-5-4',	'10.4-5-5' : '10.3-5-5',	'11.0-0' : '11.0-0',	'13.2-0' : '13.2-0',	'13.3-1-1' : '13.3-1-1',	'13.3-1-2' : '13.3-1-2',	'13.3-1-3' : '13.3-1-3',	'13.3-1-4' : '13.3-1-4',	'13.3-1-5' : '13.3-1-5',	'13.3-1-6' : '13.3-1-6',	'14.0-0' : '14.0-0',	'14.1-0' : '14.1-0',	'14.2a-1' : '14.2a-1',	'14.2a-2' : '14.2a-2',	'14.2a-3' : '14.2a-3',	'14.2a-5' : '14.2a-5',	'14.3-2' : '14.3-2',	'14.3-4' : '14.3-4',	'2.0-0' : '2.0-0',	'2.0b-2' : '2.0b-2',	'2.0b-4' : '2.0b-4',	'2.0b-6' : '2.0b-6',	'2.0b-7' : '2.0b-7',	'2.1-1' : '2.1-1',	'2.1-11' : '2.1-11',	'2.1-3' : '2.1-3',	'2.1-4' : '2.1-4',	'2.1-5' : '2.1-5',	'2.1-6' : '2.1-6',	'2.1-7' : '2.1-7',	'2.1-8' : '2.1-8',	'2.1-9' : '2.1-9',	'2.2-1' : '2.2-1',	'2.2-2' : '2.2-2',	'2.2-4' : '2.2-4',	'3.0-2' : '3.0-2',	'3.0-4' : '3.0-4',	'3.0-8' : '3.0-8',	'3.2-0' : '3.2-0',	'3.2a-1' : '3.2a-1',	'3.2a-3' : '3.2a-3',	'3.2a-5' : '3.2a-5',	'3.2a-6' : '3.2a-6',	'3.2a-9' : '3.2a-9',	'3.3-1' : '3.3-1',	'3.3-3' : '3.3-3',	'3.3-4' : '3.3-4',	'4.0-0' : '4.0-0',	'4.4-0' : '4.4-0',	'4.6a-1-1' : '4.6a-1-1',	'4.6a-1-10' : '4.6a-1-10',	'4.6a-1-11' : '4.6a-1-11',	'4.6a-1-12' : '4.6a-1-12',	'4.6a-1-13' : '4.6a-1-13',	'4.6a-1-14' : '4.6a-1-14',	'4.6a-1-15' : '4.6a-1-15',	'4.6a-1-16' : '4.6a-1-16',	'4.6a-1-17' : '4.6a-1-17',	'4.6a-1-18' : '4.6a-1-18',	'4.6a-1-19' : '4.6a-1-19',	'4.6a-1-2' : '4.6a-1-2',	'4.6a-1-20' : '4.6a-1-20',	'4.6a-1-22' : '4.6a-1-22',	'4.6a-1-23' : '4.6a-1-23',	'4.6a-1-25' : '4.6a-1-25',	'4.6a-1-26' : '4.6a-1-26',	'4.6a-1-27' : '4.6a-1-27',	'4.6a-1-28' : '4.6a-1-28',	'4.6a-1-29' : '4.6a-1-29',	'4.6a-1-3' : '4.6a-1-3',	'4.6a-1-30' : '4.6a-1-30',	'4.6a-1-31' : '4.6a-1-31',	'4.6a-1-4' : '4.6a-1-4',	'4.6a-1-5' : '4.6a-1-5',	'4.6a-1-6' : '4.6a-1-6',	'4.6a-1-7' : '4.6a-1-7',	'4.6a-1-8' : '4.6a-1-8',	'4.6a-1-9' : '4.6a-1-9',	'4.6a-2-1' : '4.6a-2-1',	'4.6a-2-10' : '4.6a-2-10',	'4.6a-2-11' : '4.6a-2-11',	'4.6a-2-12' : '4.6a-2-12',	'4.6a-2-13' : '4.6a-2-13',	'4.6a-2-14' : '4.6a-2-14',	'4.6a-2-15' : '4.6a-2-15',	'4.6a-2-16' : '4.6a-2-16',	'4.6a-2-17' : '4.6a-2-17',	'4.6a-2-18' : '4.6a-2-18',	'4.6a-2-19' : '4.6a-2-19',	'4.6a-2-2' : '4.6a-2-2',	'4.6a-2-20' : '4.6a-2-20',	'4.6a-2-22' : '4.6a-2-22',	'4.6a-2-23' : '4.6a-2-23',	'4.6a-2-25' : '4.6a-2-25',	'4.6a-2-26' : '4.6a-2-26',	'4.6a-2-27' : '4.6a-2-27',	'4.6a-2-28' : '4.6a-2-28',	'4.6a-2-29' : '4.6a-2-29',	'4.6a-2-3' : '4.6a-2-3',	'4.6a-2-30' : '4.6a-2-30',	'4.6a-2-31' : '4.6a-2-31',	'4.6a-2-4' : '4.6a-2-4',	'4.6a-2-5' : '4.6a-2-5',	'4.6a-2-6' : '4.6a-2-6',	'4.6a-2-7' : '4.6a-2-7',	'4.6a-2-8' : '4.6a-2-8',	'4.6a-2-9' : '4.6a-2-9',	'4.6a-3-1' : '4.6a-3-1',	'4.6a-3-10' : '4.6a-3-10',	'4.6a-3-11' : '4.6a-3-11',	'4.6a-3-12' : '4.6a-3-12',	'4.6a-3-13' : '4.6a-3-13',	'4.6a-3-14' : '4.6a-3-14',	'4.6a-3-15' : '4.6a-3-15',	'4.6a-3-16' : '4.6a-3-16',	'4.6a-3-17' : '4.6a-3-17',	'4.6a-3-18' : '4.6a-3-18',	'4.6a-3-19' : '4.6a-3-19',	'4.6a-3-2' : '4.6a-3-2',	'4.6a-3-20' : '4.6a-3-20',	'4.6a-3-22' : '4.6a-3-22',	'4.6a-3-23' : '4.6a-3-23',	'4.6a-3-25' : '4.6a-3-25',	'4.6a-3-26' : '4.6a-3-26',	'4.6a-3-27' : '4.6a-3-27',	'4.6a-3-28' : '4.6a-3-28',	'4.6a-3-29' : '4.6a-3-29',	'4.6a-3-3' : '4.6a-3-3',	'4.6a-3-30' : '4.6a-3-30',	'4.6a-3-31' : '4.6a-3-31',	'4.6a-3-4' : '4.6a-3-4',	'4.6a-3-5' : '4.6a-3-5',	'4.6a-3-6' : '4.6a-3-6',	'4.6a-3-7' : '4.6a-3-7',	'4.6a-3-8' : '4.6a-3-8',	'4.6a-3-9' : '4.6a-3-9',	'4.6a-4-1' : '4.6a-4-1',	'4.6a-4-10' : '4.6a-4-10',	'4.6a-4-11' : '4.6a-4-11',	'4.6a-4-12' : '4.6a-4-12',	'4.6a-4-13' : '4.6a-4-13',	'4.6a-4-14' : '4.6a-4-14',	'4.6a-4-15' : '4.6a-4-15',	'4.6a-4-16' : '4.6a-4-16',	'4.6a-4-17' : '4.6a-4-17',	'4.6a-4-18' : '4.6a-4-18',	'4.6a-4-19' : '4.6a-4-19',	'4.6a-4-2' : '4.6a-4-2',	'4.6a-4-20' : '4.6a-4-20',	'4.6a-4-22' : '4.6a-4-22',	'4.6a-4-23' : '4.6a-4-23',	'4.6a-4-25' : '4.6a-4-25',	'4.6a-4-26' : '4.6a-4-26',	'4.6a-4-27' : '4.6a-4-27',	'4.6a-4-28' : '4.6a-4-28',	'4.6a-4-29' : '4.6a-4-29',	'4.6a-4-3' : '4.6a-4-3',	'4.6a-4-30' : '4.6a-4-30',	'4.6a-4-31' : '4.6a-4-31',	'4.6a-4-4' : '4.6a-4-4',	'4.6a-4-5' : '4.6a-4-5',	'4.6a-4-6' : '4.6a-4-6',	'4.6a-4-7' : '4.6a-4-7',	'4.6a-4-8' : '4.6a-4-8',	'4.6a-4-9' : '4.6a-4-9',	'4.6a-5-1' : '4.6a-5-1',	'4.6a-5-10' : '4.6a-5-10',	'4.6a-5-11' : '4.6a-5-11',	'4.6a-5-12' : '4.6a-5-12',	'4.6a-5-13' : '4.6a-5-13',	'4.6a-5-14' : '4.6a-5-14',	'4.6a-5-15' : '4.6a-5-15',	'4.6a-5-16' : '4.6a-5-16',	'4.6a-5-17' : '4.6a-5-17',	'4.6a-5-18' : '4.6a-5-18',	'4.6a-5-19' : '4.6a-5-19',	'4.6a-5-2' : '4.6a-5-2',	'4.6a-5-20' : '4.6a-5-20',	'4.6a-5-22' : '4.6a-5-22',	'4.6a-5-23' : '4.6a-5-23',	'4.6a-5-25' : '4.6a-5-25',	'4.6a-5-26' : '4.6a-5-26',	'4.6a-5-27' : '4.6a-5-27',	'4.6a-5-28' : '4.6a-5-28',	'4.6a-5-29' : '4.6a-5-29',	'4.6a-5-3' : '4.6a-5-3',	'4.6a-5-30' : '4.6a-5-30',	'4.6a-5-31' : '4.6a-5-31',	'4.6a-5-4' : '4.6a-5-4',	'4.6a-5-5' : '4.6a-5-5',	'4.6a-5-6' : '4.6a-5-6',	'4.6a-5-7' : '4.6a-5-7',	'4.6a-5-8' : '4.6a-5-8',	'4.6a-5-9' : '4.6a-5-9',	'4.6a-6-1' : '4.6a-6-1',	'4.6a-6-10' : '4.6a-6-10',	'4.6a-6-11' : '4.6a-6-11',	'4.6a-6-12' : '4.6a-6-12',	'4.6a-6-13' : '4.6a-6-13',	'4.6a-6-14' : '4.6a-6-14',	'4.6a-6-15' : '4.6a-6-15',	'4.6a-6-16' : '4.6a-6-16',	'4.6a-6-17' : '4.6a-6-17',	'4.6a-6-18' : '4.6a-6-18',	'4.6a-6-19' : '4.6a-6-19',	'4.6a-6-2' : '4.6a-6-2',	'4.6a-6-20' : '4.6a-6-20',	'4.6a-6-22' : '4.6a-6-22',	'4.6a-6-23' : '4.6a-6-23',	'4.6a-6-25' : '4.6a-6-25',	'4.6a-6-26' : '4.6a-6-26',	'4.6a-6-27' : '4.6a-6-27',	'4.6a-6-28' : '4.6a-6-28',	'4.6a-6-29' : '4.6a-6-29',	'4.6a-6-3' : '4.6a-6-3',	'4.6a-6-30' : '4.6a-6-30',	'4.6a-6-31' : '4.6a-6-31',	'4.6a-6-4' : '4.6a-6-4',	'4.6a-6-5' : '4.6a-6-5',	'4.6a-6-6' : '4.6a-6-6',	'4.6a-6-7' : '4.6a-6-7',	'4.6a-6-8' : '4.6a-6-8',	'4.6a-6-9' : '4.6a-6-9',	'4.6b-1-1' : '4.6b-1-1',	'4.6b-1-10' : '4.6b-1-10',	'4.6b-1-11' : '4.6b-1-11',	'4.6b-1-12' : '4.6b-1-12',	'4.6b-1-13' : '4.6b-1-13',	'4.6b-1-14' : '4.6b-1-14',	'4.6b-1-15' : '4.6b-1-15',	'4.6b-1-16' : '4.6b-1-16',	'4.6b-1-17' : '4.6b-1-17',	'4.6b-1-2' : '4.6b-1-2',	'4.6b-1-3' : '4.6b-1-3',	'4.6b-1-4' : '4.6b-1-4',	'4.6b-1-5' : '4.6b-1-5',	'4.6b-1-6' : '4.6b-1-6',	'4.6b-1-7' : '4.6b-1-7',	'4.6b-1-8' : '4.6b-1-8',	'4.6b-1-9' : '4.6b-1-9',	'4.6c-1-1' : '4.6c-1-1',	'4.6c-13-1' : '4.6c-13-1',	'4.6c-3-1' : '4.6c-3-1',	'4.6c-8-1' : '4.6c-8-1',	'4.6d-1' : '4.6d-1',	'4.6d-2' : '4.6d-2',	'4.6d-3' : '4.6d-3',	'4.6d-4' : '4.6d-4',	'4.6e-1' : '4.6d-1','4.6e-2' : '4.6d-2','4.6e-3' : '4.6d-3','4.6e-4' : '4.6d-4','4.6f-1' : '4.6d-1','4.6f-2' : '4.6d-2','4.6f-3' : '4.6d-3','4.6f-4' : '4.6d-4',	'4.9-1-1' : '4.9-1-1',	'5.0-0' : '5.0-0',	'5.0a-1' : '5.0a-1',	'5.0a-10' : '5.0a-11',	'5.0a-3' : '5.0a-3',	'5.0a-4' : '5.0a-5',	'5.0a-5' : '5.0a-6',	'5.0a-6' : '5.0a-7',	'5.0a-7' : '5.0a-8',	'5.0a-8' : '5.0a-9',	'5.0b-1' : '5.0b-1',	'5.0b-3' : '5.0b-3',	'5.0b-4' : '5.0b-5',	'5.0b-5' : '5.0b-7',	'5.0b-6' : '5.0b-8',	'5.0b-7' : '5.0b-9',	'5.0b-8' : '5.0b-10',	'5.0c-1' : '5.0c-1',	'5.0c-10' : '5.0c-11',	'5.0c-11' : '5.0c-12',	'5.0c-12' : '5.0c-13',	'5.0c-3' : '5.0c-3',	'5.0c-4' : '5.0c-5',	'5.0c-5' : '5.0c-6',	'5.0c-6' : '5.0c-7',	'5.0c-7' : '5.0c-8',	'5.0c-8' : '5.0c-9',	'5.0c-9' : '5.0c-10',	'5.0d-1' : '5.0d-1',	'5.0d-10' : '5.0d-11',	'5.0d-3' : '5.0d-3',	'5.0d-4' : '5.0d-5',	'5.0d-5' : '5.0d-6',	'5.0d-6' : '5.0d-7',	'5.0d-7' : '5.0d-8',	'5.0d-8' : '5.0d-9',	'5.0d-9' : '5.0d-10',	'5.4-5' : '5.4-7',	'5.4-6' : '5.4-8',	'5.5-0' : '5.5-0',	'6.0-1' : '6.0-1',	'6.0-2' : '6.0-2',	'6.15-1-1' : '6.13-1-1',	'6.2-0' : '6.2-0',	'6.2a-1' : '6.2a-1',	'6.2a-3' : '6.2a-3',	'6.5-3' : '6.5-3',	'6.5-4' : '6.5-4',	'6.5-7' : '6.5-7',	'6.5-9' : '6.5-9',	'7.0-0' : '7.0-0',	'7.2-0' : '7.2-0',	'7.3-1-1' : '7.3-1-1',	'7.4-0' : '7.4-0',	'7.6-2-1' : '7.6-2-1',	'7.6-3-1' : '7.6-3-1',	'7.6-4-1' : '7.6-4-1',	'7.7-0' : '7.7-0',	'7.7a-2' : '7.7a-2',	'8.0-0' : '8.0-0',	'8.0a-3' : '8.0a-3',	'8.0a-4' : '8.0a-4',	'8.0a-6' : '8.0a-6',	'8.0a-7' : '8.0a-7',	'8.0a-9' : '8.0a-9',	'8.1-10-1' : '8.1-11-1',	'8.1-1-1' : '8.1-1-1',	'8.1-2-1' : '8.1-2-1',	'8.1-3-1' : '8.1-3-1',	'8.1-4-1' : '8.1-4-1',	'8.1-5-1' : '8.1-5-1',	'8.1-6-1' : '8.1-6-1',	'8.1-7-1' : '8.1-7-1',	'8.1-8-1' : '8.1-8-1',	'8.1-9-1' : '8.1-9-1',	'8.4-1-4' : '8.2-1-3',	'9.1-1-1' : '9.1-1-1',	'9.1-1-2' : '9.1-1-2',	'9.1-1-3' : '9.1-1-3',	'9.1-1-4' : '9.1-1-4',	'9.1-3-1' : '9.1-3-1',	'9.1-3-2' : '9.1-3-2',	'9.1-3-3' : '9.1-3-3',	'9.1-3-4' : '9.1-3-4'}
  
replace_2021 = {'4.6e-1' : '4.6d-1',	'4.6e-2' : '4.6d-2',	'4.6e-3' : '4.6d-3',	'4.6e-4' : '4.6d-4',	'4.6f-1' : '4.6d-1',	'4.6f-2' : '4.6d-2',	'4.6f-3' : '4.6d-3',	'4.6f-4' : '4.6d-4'}
  
replace_2022to2021 = {'0.1-1-1' : '0.1-1-1',	'0.1-4-1' : '0.6-1-1',	'0.1-5-1' : '11.0-0',	'0.1-7-1' : '0.5-2-1',  '0.1-6-1' : '0.5-1-1',	'0.1-8-1' : '0.5-3-1',	'0.3-3-1' : '6.0-1', '0.3-3-2' : '6.0-1', '0.3-3-3' : '6.0-1',	'0.3-4-1' : '6.0-2',	'0.5-1' : '6.2-0',	'0.5-3' : '6.2a-1',	'0.5-4' : '6.2a-3',	'1.1-0' : '2.0-0',	'1.1a-1' : '2.0b-2',	'1.1a-3' : '2.0b-4',	'1.1a-5' : '2.0b-7',	'1.1a-6' : '2.0b-6',	'1.2-1' : '2.1-1',	'1.2-10' : '2.1-8',	'1.2-11' : '2.1-11',	'1.2-2' : '2.1-7',	'1.2-3' : '2.1-6',	'1.2-4' : '2.1-5',	'1.2-7' : '2.1-3',	'1.2-8' : '2.1-4',	'1.2-9' : '2.1-9',	'1.3-1' : '2.2-1',	'1.3-2' : '2.2-2',	'1.3-3' : '2.2-4',	'2.1-0' : '4.0-0',	'2.1b-4' : '4.4-0',	'2.1c-1-1' : '4.6c-1-1',	'2.1c-1-10' : '4.6b-1-7',	'2.1c-1-11' : '4.6b-1-8',	'2.1c-1-12' : '4.6b-1-9',	'2.1c-1-13' : '4.6b-1-10',	'2.1c-1-14' : '4.6b-1-11',	'2.1c-1-15' : '4.6b-1-12',	'2.1c-1-16' : '4.6b-1-16',	'2.1c-1-17' : '4.6b-1-17',	'2.1c-1-2' : '4.6b-1-4',	'2.1c-1-3' : '4.6b-1-14',	'2.1c-1-4' : '4.6b-1-15',	'2.1c-1-5' : '4.6b-1-1',	'2.1c-1-6' : '4.6b-1-2',	'2.1c-1-7' : '4.6b-1-3',	'2.1c-1-8' : '4.6b-1-5',	'2.1c-1-9' : '4.6b-1-6',	'2.1d-1-1' : '4.6a-1-1',	'2.1d-1-10' : '4.6a-1-10',	'2.1d-1-11' : '4.6a-1-11',	'2.1d-1-12' : '4.6a-1-12',	'2.1d-1-13' : '4.6a-1-13',	'2.1d-1-14' : '4.6a-1-14',	'2.1d-1-15' : '4.6a-1-15',	'2.1d-1-16' : '4.6a-1-16',	'2.1d-1-17' : '4.6a-1-17',	'2.1d-1-18' : '4.6a-1-18',	'2.1d-1-19' : '4.6a-1-19',	'2.1d-1-2' : '4.6a-1-2',	'2.1d-1-20' : '4.6a-1-20',	'2.1d-1-22' : '4.6a-1-22',	'2.1d-1-23' : '4.6a-1-23',	'2.1d-1-25' : '4.6a-1-25',	'2.1d-1-26' : '4.6a-1-26',	'2.1d-1-27' : '4.6a-1-27',	'2.1d-1-28' : '4.6a-1-28',	'2.1d-1-29' : '4.6a-1-29',	'2.1d-1-3' : '4.6a-1-3',	'2.1d-1-30' : '4.6a-1-30',	'2.1d-1-31' : '4.6a-1-31',	'2.1d-1-4' : '4.6a-1-4',	'2.1d-1-5' : '4.6a-1-5',	'2.1d-1-6' : '4.6a-1-6',	'2.1d-1-7' : '4.6a-1-7',	'2.1d-1-8' : '4.6a-1-8',	'2.1d-1-9' : '4.6a-1-9',	'2.1d-2-1' : '4.6a-2-1',	'2.1d-2-10' : '4.6a-2-10',	'2.1d-2-11' : '4.6a-2-11',	'2.1d-2-12' : '4.6a-2-12',	'2.1d-2-13' : '4.6a-2-13',	'2.1d-2-14' : '4.6a-2-14',	'2.1d-2-15' : '4.6a-2-15',	'2.1d-2-16' : '4.6a-2-16',	'2.1d-2-17' : '4.6a-2-17',	'2.1d-2-18' : '4.6a-2-18',	'2.1d-2-19' : '4.6a-2-19',	'2.1d-2-2' : '4.6a-2-2',	'2.1d-2-20' : '4.6a-2-20',	'2.1d-2-22' : '4.6a-2-22',	'2.1d-2-23' : '4.6a-2-23',	'2.1d-2-25' : '4.6a-2-25',	'2.1d-2-26' : '4.6a-2-26',	'2.1d-2-27' : '4.6a-2-27',	'2.1d-2-28' : '4.6a-2-28',	'2.1d-2-29' : '4.6a-2-29',	'2.1d-2-3' : '4.6a-2-3',	'2.1d-2-30' : '4.6a-2-30',	'2.1d-2-31' : '4.6a-2-31',	'2.1d-2-4' : '4.6a-2-4',	'2.1d-2-5' : '4.6a-2-5',	'2.1d-2-6' : '4.6a-2-6',	'2.1d-2-7' : '4.6a-2-7',	'2.1d-2-8' : '4.6a-2-8',	'2.1d-2-9' : '4.6a-2-9',	'2.1d-3-1' : '4.6a-3-1',	'2.1d-3-10' : '4.6a-3-10',	'2.1d-3-11' : '4.6a-3-11',	'2.1d-3-12' : '4.6a-3-12',	'2.1d-3-13' : '4.6a-3-13',	'2.1d-3-14' : '4.6a-3-14',	'2.1d-3-15' : '4.6a-3-15',	'2.1d-3-16' : '4.6a-3-16',	'2.1d-3-17' : '4.6a-3-17',	'2.1d-3-18' : '4.6a-3-18',	'2.1d-3-19' : '4.6a-3-19',	'2.1d-3-2' : '4.6a-3-2',	'2.1d-3-20' : '4.6a-3-20',	'2.1d-3-22' : '4.6a-3-22',	'2.1d-3-23' : '4.6a-3-23',	'2.1d-3-25' : '4.6a-3-25',	'2.1d-3-26' : '4.6a-3-26',	'2.1d-3-27' : '4.6a-3-27',	'2.1d-3-28' : '4.6a-3-28',	'2.1d-3-29' : '4.6a-3-29',	'2.1d-3-3' : '4.6a-3-3',	'2.1d-3-30' : '4.6a-3-30',	'2.1d-3-31' : '4.6a-3-31',	'2.1d-3-4' : '4.6a-3-4',	'2.1d-3-5' : '4.6a-3-5',	'2.1d-3-6' : '4.6a-3-6',	'2.1d-3-7' : '4.6a-3-7',	'2.1d-3-8' : '4.6a-3-8',	'2.1d-3-9' : '4.6a-3-9',	'2.1d-4-1' : '4.6a-4-1',	'2.1d-4-10' : '4.6a-4-10',	'2.1d-4-11' : '4.6a-4-11',	'2.1d-4-12' : '4.6a-4-12',	'2.1d-4-13' : '4.6a-4-13',	'2.1d-4-14' : '4.6a-4-14',	'2.1d-4-15' : '4.6a-4-15',	'2.1d-4-16' : '4.6a-4-16',	'2.1d-4-17' : '4.6a-4-17',	'2.1d-4-18' : '4.6a-4-18',	'2.1d-4-19' : '4.6a-4-19',	'2.1d-4-2' : '4.6a-4-2',	'2.1d-4-20' : '4.6a-4-20',	'2.1d-4-22' : '4.6a-4-22',	'2.1d-4-23' : '4.6a-4-23',	'2.1d-4-25' : '4.6a-4-25',	'2.1d-4-26' : '4.6a-4-26',	'2.1d-4-27' : '4.6a-4-27',	'2.1d-4-28' : '4.6a-4-28',	'2.1d-4-29' : '4.6a-4-29',	'2.1d-4-3' : '4.6a-4-3',	'2.1d-4-30' : '4.6a-4-30',	'2.1d-4-31' : '4.6a-4-31',	'2.1d-4-4' : '4.6a-4-4',	'2.1d-4-5' : '4.6a-4-5',	'2.1d-4-6' : '4.6a-4-6',	'2.1d-4-7' : '4.6a-4-7',	'2.1d-4-8' : '4.6a-4-8',	'2.1d-4-9' : '4.6a-4-9',	'2.1d-5-1' : '4.6a-5-1',	'2.1d-5-10' : '4.6a-5-10',	'2.1d-5-11' : '4.6a-5-11',	'2.1d-5-12' : '4.6a-5-12',	'2.1d-5-13' : '4.6a-5-13',	'2.1d-5-14' : '4.6a-5-14',	'2.1d-5-15' : '4.6a-5-15',	'2.1d-5-16' : '4.6a-5-16',	'2.1d-5-17' : '4.6a-5-17',	'2.1d-5-18' : '4.6a-5-18',	'2.1d-5-19' : '4.6a-5-19',	'2.1d-5-2' : '4.6a-5-2',	'2.1d-5-20' : '4.6a-5-20',	'2.1d-5-22' : '4.6a-5-22',	'2.1d-5-23' : '4.6a-5-23',	'2.1d-5-25' : '4.6a-5-25',	'2.1d-5-26' : '4.6a-5-26',	'2.1d-5-27' : '4.6a-5-27',	'2.1d-5-28' : '4.6a-5-28',	'2.1d-5-29' : '4.6a-5-29',	'2.1d-5-3' : '4.6a-5-3',	'2.1d-5-30' : '4.6a-5-30',	'2.1d-5-31' : '4.6a-5-31',	'2.1d-5-4' : '4.6a-5-4',	'2.1d-5-5' : '4.6a-5-5',	'2.1d-5-6' : '4.6a-5-6',	'2.1d-5-7' : '4.6a-5-7',	'2.1d-5-8' : '4.6a-5-8',	'2.1d-5-9' : '4.6a-5-9',	'2.1d-6-1' : '4.6a-6-1',	'2.1d-6-10' : '4.6a-6-10',	'2.1d-6-11' : '4.6a-6-11',	'2.1d-6-12' : '4.6a-6-12',	'2.1d-6-13' : '4.6a-6-13',	'2.1d-6-14' : '4.6a-6-14',	'2.1d-6-15' : '4.6a-6-15',	'2.1d-6-16' : '4.6a-6-16',	'2.1d-6-17' : '4.6a-6-17',	'2.1d-6-18' : '4.6a-6-18',	'2.1d-6-19' : '4.6a-6-19',	'2.1d-6-2' : '4.6a-6-2',	'2.1d-6-20' : '4.6a-6-20',	'2.1d-6-22' : '4.6a-6-22',	'2.1d-6-23' : '4.6a-6-23',	'2.1d-6-25' : '4.6a-6-25',	'2.1d-6-26' : '4.6a-6-26',	'2.1d-6-27' : '4.6a-6-27',	'2.1d-6-28' : '4.6a-6-28',	'2.1d-6-29' : '4.6a-6-29',	'2.1d-6-3' : '4.6a-6-3',	'2.1d-6-30' : '4.6a-6-30',	'2.1d-6-31' : '4.6a-6-31',	'2.1d-6-4' : '4.6a-6-4',	'2.1d-6-5' : '4.6a-6-5',	'2.1d-6-6' : '4.6a-6-6',	'2.1d-6-7' : '4.6a-6-7',	'2.1d-6-8' : '4.6a-6-8',	'2.1d-6-9' : '4.6a-6-9',	'2.1e-1' : '4.6d-1','2.1e-2' : '4.6d-2','2.1e-3' : '4.6d-3','2.1e-4' : '4.6d-4','2.1e-1' : '4.6d-1','2.1e-2' : '4.6d-2','2.1e-3' : '4.6d-3','2.1e-4' : '4.6d-4','2.1e-1' : '4.6d-1','2.1e-2' : '4.6d-2','2.1e-3' : '4.6d-3','2.1e-4' : '4.6d-4',	'2.2-1-1' : '4.9-1-1',	'2.3-0' : '7.0-0',	'2.3a-3-1' : '7.2-0',	'2.3a-5-1' : '7.3-1-1',	'2.3b-1-1' : '7.6-2-1',	'2.3b-2-1' : '7.6-3-1',	'2.3b-3-1' : '7.7a-2',	'2.3b-4-1' : '7.6-4-1',	'3.1-10-1' : '8.1-8-1',	'3.1-11-1' : '8.1-9-1',	'3.11-2-1' : '14.1-0',	'3.1-14-1' : '8.1-11-1',	'3.1-3-1' : '8.1-1-1',	'3.14-1' : '14.0-0',	'3.1-4-1' : '8.1-2-1',	'3.1-5-1' : '8.1-3-1',	'3.1-6-1' : '8.1-4-1',	'3.1-7-1' : '8.1-5-1',	'3.1-8-1' : '8.1-6-1',	'3.1-9-1' : '8.1-7-1',	'3.2-1-3' : '8.2-1-3',	'3.5-10-1' : '10.1-9-1',	'3.5-2-1' : '10.1-5-1',	'3.5-3-1' : '10.1-6-1',	'3.5-5-1' : '10.1-3-1',	'3.5-6-1' : '10.1-2-1',	'3.5-7-1' : '10.1-4-1',	'3.5-8-1' : '10.1-7-1',	'3.5-9-1' : '10.1-1-1',	'3.6-1-2' : '10.3-1-1',	'3.6-1-3' : '10.3-1-2',	'3.6-1-4' : '10.3-1-3',	'3.6-1-5' : '10.3-1-4',	'3.6-1-6' : '10.3-1-5',	'3.6-2-2' : '10.3-2-1',	'3.6-2-3' : '10.3-2-2',	'3.6-2-4' : '10.3-2-3',	'3.6-2-5' : '10.3-2-4',	'3.6-2-6' : '10.3-2-5',	'3.6-3-2' : '10.3-3-1',	'3.6-3-3' : '10.3-3-2',	'3.6-3-4' : '10.3-3-3',	'3.6-3-5' : '10.3-3-4',	'3.6-3-6' : '10.3-3-5',	'3.6-4-2' : '10.3-4-1',	'3.6-4-3' : '10.3-4-2',	'3.6-4-4' : '10.3-4-3',	'3.6-4-5' : '10.3-4-4',	'3.6-4-6' : '10.3-4-5',	'3.6-5-2' : '10.3-5-1',	'3.6-5-3' : '10.3-5-2',	'3.6-5-4' : '10.3-5-3',	'3.6-5-5' : '10.3-5-4',	'3.6-5-6' : '10.3-5-5',	'3.7-2-1' : '13.3-1-1',	'3.7-2-2' : '13.2-0',	'4.1a-2' : '3.3-1',	'4.1a-5' : '3.3-3',	'4.1a-6' : '3.3-4',	'5.1a-10-1' : '5.0a-7',	'5.1a-10-3' : '5.0c-9',	'5.1a-10-4' : '5.0d-7',	'5.1a-1-1' : '5.0a-1',	'5.1a-11-3' : '5.0c-8',	'5.1a-1-2' : '5.0b-1',	'5.1a-12-1' : '5.0a-9',	'5.1a-12-2' : '5.0b-7',	'5.1a-12-3' : '5.0c-11',	'5.1a-12-4' : '5.0d-8',	'5.1a-1-3' : '5.0c-1',	'5.1a-13-2' : '5.0b-9',	'5.1a-13-3' : '5.0c-12',	'5.1a-13-4' : '5.0d-9',	'5.1a-1-4' : '5.0d-1',	'5.1a-14-1' : '5.0a-11',	'5.1a-14-2' : '5.0b-10',	'5.1a-14-3' : '5.0c-13',	'5.1a-14-4' : '5.0d-11',	'5.1a-16-2' : '5.0b-8',	'5.1a-30-3' : '5.0c-7',	'5.1a-3-1' : '5.0a-3',	'5.1a-3-2' : '5.0b-3',	'5.1a-3-3' : '5.0c-3',	'5.1a-3-4' : '5.0d-3',	'5.1a-7-1' : '5.0a-6',	'5.1a-7-2' : '5.0b-5',	'5.1a-7-3' : '5.0c-6',	'5.1a-7-4' : '5.0d-6',	'5.1a-9-1' : '5.0a-5',	'5.1a-9-3' : '5.0c-5',	'5.1a-9-4' : '5.0d-5',	'6.1-1-1' : '9.1-1-1',	'6.1-1-10' : '8.0a-3',	'6.1-1-11' : '8.0a-4',	'6.1-1-12' : '8.0a-6',	'6.1-1-13' : '8.0a-7',	'6.1-1-14' : '8.0a-9',	'6.1-1-2' : '9.1-1-2',	'6.1-1-3' : '9.1-1-3',	'6.1-1-4' : '9.1-1-4',	'6.1-1-5' : '9.1-3-1',	'6.1-1-6' : '9.1-3-2',	'6.1-1-7' : '9.1-3-3',	'6.1-1-8' : '9.1-3-4',	'6.1-1-9' : '8.0-0',	'7.1-0' : '3.2-0',	'7.1-1' : '5.5-0', '7.1a-1' : '3.2a-9',	'7.1a-11' : '3.2a-5',	'7.1a-14' : '3.2a-3',	'7.1a-2' : '3.2a-1',	'7.1a-4' : '3.2a-6',	'7.4-3' : '6.5-3',	'7.4-4' : '6.5-4',	'7.4-6' : '6.5-7',	'7.4-8' : '6.5-9',	'9.1-6' : '5.4-7',	'9.1-7' : '5.4-8'}

# COMMAND ----------

from pyspark.sql.functions import col, expr

# Replacements in 2018
mapping_2018 = "CASE "
for key, value in replace_2018to2021.items():
    mapping_2018 += "WHEN q_id = '{0}' THEN '{1}' ".format(key, value)
mapping_2018 += "ELSE q_id END"
df_cities_2018r = df_cities_2018.withColumn('q_id', expr(mapping_2018))

# Replacements in 2019
mapping_2019 = "CASE "
for key, value in replace_2019to2021.items():
    mapping_2019 += "WHEN q_id = '{0}' THEN '{1}' ".format(key, value)
mapping_2019 += "ELSE q_id END"
df_cities_2019r = df_cities_2019.withColumn('q_id', expr(mapping_2019))

# Replacements in 2020
mapping_2020 = "CASE "
for key, value in replace_2020to2021.items():
    mapping_2020 += "WHEN q_id = '{0}' THEN '{1}' ".format(key, value)
mapping_2020 += "ELSE q_id END"
df_cities_2020r = df_cities_2020.withColumn('q_id', expr(mapping_2020))

# Replacements in 2021
mapping_2021 = "CASE "
for key, value in replace_2021.items():
    mapping_2021 += "WHEN q_id = '{0}' THEN '{1}' ".format(key, value)
mapping_2021 += "ELSE q_id END"
df_cities_2021r = df_cities_2021.withColumn('q_id', expr(mapping_2021))

# Replacements in 2022
mapping_2022 = "CASE "
for key, value in replace_2022to2021.items():
    mapping_2022 += "WHEN q_id = '{0}' THEN '{1}' ".format(key, value)
mapping_2022 += "ELSE q_id END"
df_cities_2022r = df_cities_2022.withColumn('q_id', expr(mapping_2022))

# COMMAND ----------

display(df_cities_2018r.filter(col('q_id_orig').like('14.5-2%')))

# COMMAND ----------

# Append all dataframes
df_cities_all = df_cities_2018r.union(df_cities_2019r)\
                               .union(df_cities_2020r)\
                               .union(df_cities_2021r)\
                               .union(df_cities_2022r)



# COMMAND ----------

# QC de replacements -- Adjustments done
df_cities_all_test = df_cities_all.groupBy('Questionnaire', 'q_id', 'q_id_orig').count()
display(df_cities_all_test)

# COMMAND ----------

display(df_cities_all.filter((df_cities_all['q_id'] == '14.0-0') & (df_cities_all['Account Number'] == '840425')))

# COMMAND ----------

# Replace posible null values in all Responses 
replace_nulls = [None, ' NO ', ' YET', 'DO NOT KNOW', 'DOES NOT HAVE THIS DATA' , 'NOT AVAILABLE', 'NO DISPONIBLE', 'NA', 'N/A', 'NA.', 'N/A.', 'NOT APPLICABLE', 'NOT APPLICABLE.', 'NÃ£O SE APLICA', 'NÃ£O APLICÃ¡VEL', 'NÃ£O SE APLICA.', 'NO APLICA', 'NO TARGET', 'NO TIENE', 'NO WEB LINK AVAILABLE', 'NOT', 'NULL']

df_cities_all = df_cities_all.withColumn("Response Answer", when(upper(col("Response Answer")).isin(replace_nulls), None).otherwise(col("Response Answer")))

# COMMAND ----------

df_cities_all = df_cities_all.withColumn("q_id_fix", 
          when((col('q_id') == '4.4-0') & (col("Response Answer") == "CH4"), '4.4-0-1')
          .when((col('q_id') == '4.4-0') & (col("Response Answer") == "CO2"), '4.4-0-2')
          .when((col('q_id') == '4.4-0') & (col("Response Answer") == "HFCs"), '4.4-0-3')
          .when((col('q_id') == '4.4-0') & ((col("Response Answer") == "N2O") | (col("Response Answer") == "N20")), '4.4-0-4')
          .when((col('q_id') == '4.4-0') & (col("Response Answer") == "NF3"), '4.4-0-5')
          .when((col('q_id') == '4.4-0') & (col("Response Answer") == "PFCs"), '4.4-0-6')
          .when((col('q_id') == '4.4-0') & (col("Response Answer") == "SF6"), '4.4-0-7')
          .when((col('q_id') == '14.0-0') & (upper(col("Response Answer")).like("%GROUND%")), '14.0-0-1')
          .when((col('q_id') == '14.0-0') & (upper(col("Response Answer")).like("%SURFACE%")), '14.0-0-2')
          .when((col('q_id') == '14.0-0') & (upper(col("Response Answer")).like("%RECYCLED%")), '14.0-0-3')
          .when((col('q_id') == '14.0-0') & (upper(col("Response Answer")).like("%SEAWATER%")), '14.0-0-4')
          .when((col('q_id') == '14.0-0') & ((upper(col("Response Answer")).like("%RAINWATER%")) | (upper(col("Response Answer")).like("%LLUVIA%"))), '14.0-0-5')
          .when((col('q_id') == '2.1-3')& ((upper(col("Response Answer")).like("%HIGH%")) | (upper(col("Response Answer")).like("%EXTREME%"))), '2.1-3-1')
          .when((col('q_id') == '2.1-3') & ((upper(col("Response Answer")).like("%MEDIUM%")) | (upper(col("Response Answer")) == "SERIOUS")), '2.1-3-2')
          .when((col('q_id') == '2.1-3') & ((upper(col("Response Answer")).like("%LOW%")) | (upper(col("Response Answer")).like("%LESS%"))), '2.1-3-3')
          .when((col('q_id') == '2.1-3') & (upper(col("Response Answer")).like("%IMPACT%")), '2.1-3-4')
          .when((col('q_id') == '2.1-4')& ((upper(col("Response Answer")).like("%HIGH%")) | (upper(col("Response Answer")).like("%EXTREME%"))), '2.1-4-1')
          .when((col('q_id') == '2.1-4') & ((upper(col("Response Answer")).like("%MEDIUM%")) | (upper(col("Response Answer")) == "SERIOUS")), '2.1-4-2')
          .when((col('q_id') == '2.1-4') & ((upper(col("Response Answer")).like("%LOW%")) | (upper(col("Response Answer")).like("%LESS%"))), '2.1-4-3')
          .when((col('q_id') == '2.1-4') & (upper(col("Response Answer")).like("%IMPACT%")), '2.1-4-4')
          .when((col('q_id') == '2.1-11') & ((upper(col("Response Answer")).like("%IMMEDIATELY%")) | (upper(col("Response Answer")).like("%CURRENT%"))), '2.1-11-1')
          .when((col('q_id') == '2.1-11') & (upper(col("Response Answer")).like("%SHORT%")), '2.1-11-2')
          .when((col('q_id') == '2.1-11') & (upper(col("Response Answer")).like("%MEDIUM%")), '2.1-11_3')
          .when((col('q_id') == '2.1-11') & (upper(col("Response Answer")).like("%LONG%")), '2.1-11-3')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%BIOLOG%")) | (upper(col("Response Answer")).like("%DESEASE%")) | (upper(col("Response Answer")).like("%INFESTATION%")) | (upper(col("Response Answer")).like("%ENFERMEDAD%")) | (upper(col("Response Answer")).like("%INSECTO%"))), '2.1-1-1')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%CHEMICAL CHANGE%")) | (upper(col("Response Answer")).like("%CO2%")) | (upper(col("Response Answer")).like("%SALT%")) | (upper(col("Response Answer")).like("%ACIDIFICATION%")) | (upper(col("Response Answer")).like("%AIR%"))), '2.1-1-2')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%DROUGHT%")) | (upper(col("Response Answer")).like("%SEQU%"))), '2.1-1-3')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%COLD%")) | (upper(col("Response Answer")).like("%WINTER%")) | (upper(col("Response Answer")).like("%FRÃOS%")) | (upper(col("Response Answer")).like("%FREEZ%")) | (upper(col("Response Answer")).like("%HELADA%")) | (upper(col("Response Answer")).like("%INVER%")) | (upper(col("Response Answer")).like("%NEVADA%")) | (upper(col("Response Answer")).like("%NIEVE%"))), '2.1-1-4')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%HOT%")) | (upper(col("Response Answer")).like("%HEAT%")) | (upper(col("Response Answer")).like("%CALOR%")) | (upper(col("Response Answer")).like("%CALURO%"))), '2.1-1-5')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%EXTREME PRECIPITATION%")) | (upper(col("Response Answer")).like("%HEAVY PRECIPITATION%")) | (upper(col("Response Answer")).like("%MONSOON%")) | (upper(col("Response Answer")).like("%SNOW%")) | (upper(col("Response Answer")).like("%HAIL%")) | (upper(col("Response Answer")).like("%FOG%")) | (upper(col("Response Answer")).like("%GRANIZ%")) | (upper(col("Response Answer")).like("%LLUVIA%")) | (upper(col("Response Answer")).like("%NIEBLA%")) | (upper(col("Response Answer")).like("%NEBLIN%"))), '2.1-1-6')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%FLOOD%")) | (upper(col("Response Answer")).like("%INUNDA%")) | (upper(col("Response Answer")).like("%MAR%")) | (upper(col("Response Answer")).like("%SEA LEVEL%")) | (upper(col("Response Answer")).like("%LEVEL RISE%"))), '2.1-1-7')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%MASS%")) | (upper(col("Response Answer")).like("%LANDSLIDE%")) | (upper(col("Response Answer")).like("%SUBSIDENC%")) | (upper(col("Response Answer")).like("%AVALAN%")) | (upper(col("Response Answer")).like("%DESLAVE%")) | (upper(col("Response Answer")).like("%HUNDIMIENTO%")) | (upper(col("Response Answer")).like("%ROCA%")) | (upper(col("Response Answer")).like("%ROCK%"))), '2.1-1-8')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%STORM%")) | (upper(col("Response Answer")).like("%WIND%")) | (upper(col("Response Answer")).like("%TORMENT%")) | (upper(col("Response Answer")).like("%TORNA%")) | (upper(col("Response Answer")).like("%HURRICAN%")) | (upper(col("Response Answer")).like("%HURAC%"))), '2.1-1-9')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%WATER%")) | (upper(col("Response Answer")).like("%AGUA%"))), '2.1-1-10')
          .when((col('q_id') == '2.1-1') & ((upper(col("Response Answer")).like("%FIRE%")) | (upper(col("Response Answer")).like("%INCENDIO%"))), '2.1-1-11')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%CONSTRUC%")) | (upper(col("Response Answer")).like("%INFRA%")) | (upper(col("Response Answer")).like("%BUILDIN%"))), '2.1-6-1')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%COMMERCIAL%")) | (upper(col("Response Answer")).like("%COMERCI%"))), '2.1-6-2')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%EDUCA%")) | (upper(col("Response Answer")).like("%TECHNICAL%"))), '2.1-6-3')
          .when((col('q_id') == '2.1-6') & (upper(col("Response Answer")).like("%EMERGENC%")), '2.1-6-4')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%ENERG%")) | (upper(col("Response Answer")).like("%ELECTRIC%"))), '2.1-6-5')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%ENVIRON%")) | (upper(col("Response Answer")).like("%AMBIENTE%")) | (upper(col("Response Answer")).like("%BIODIVER%")) | (upper(col("Response Answer")).like("%FOREST%"))), '2.1-6-66')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%FOOD%")) | (upper(col("Response Answer")).like("%AGRICULTUR%")) | (upper(col("Response Answer")).like("%ALIMENT%"))), '2.1-6-7')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%INDUSTR%")) | (upper(col("Response Answer")).like("%FABRIC%")) | (upper(col("Response Answer")).like("%MANUFAC%"))), '2.1-6-8')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%INFORMA%")) | (upper(col("Response Answer")).like("%COMMUNICA%"))), '2.1-6-9')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%LAND%")) | (upper(col("Response Answer")).like("%PLANIFIC%")) | (upper(col("Response Answer")).like("%TIERRA%"))), '2.1-6-10')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%HEALTH%")) | (upper(col("Response Answer")).like("%SALUD%"))), '2.1-6-11')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%RESIDEN%")) | (upper(col("Response Answer")).like("%HOUSING%")) | (upper(col("Response Answer")).like("%VIVIEND%")) | (upper(col("Response Answer")).like("%HABITA%"))), '2.1-6-12')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%CULTUR%")) | (upper(col("Response Answer")).like("%SPORT%")) | (upper(col("Response Answer")).like("%COMMUNITY%")) | (upper(col("Response Answer")).like("%ENTRETEN%")) | (upper(col("Response Answer")).like("%ARTS%")) | (upper(col("Response Answer")).like("%COMUNIDAD%")) | (upper(col("Response Answer")).like("%RECREA%"))), '2.1-6-13')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%TOURISM%")) | (upper(col("Response Answer")).like("%TURISMO%"))), '2.1-6-14')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%TRANSPORT%")) | (upper(col("Response Answer")).like("%PORT%"))), '2.1-6-15')
          .when((col('q_id') == '2.1-6') & ((upper(col("Response Answer")).like("%WATER SUPPLY%")) | (upper(col("Response Answer")).like("%AGUA%"))), '2.1-6-16')
          .when((col('q_id') == '2.1-6') & (upper(col("Response Answer")).like("%WASTE%")), '2.1-6-17')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%ALL%")) | (upper(col("Response Answer")).like("%TOD%")) | (upper(col("Response Answer")).like("%GERAL%")) | (upper(col("Response Answer")).like("%POBLACIÃ³N EN GENERAL%"))), '2.1-7-1')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%CHILDREN%")) | (upper(col("Response Answer")).like("%MENORES%")) | (upper(col("Response Answer")).like("%NIÃ±OS%"))), '2.1-7-2')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%ELDERLY%")) | (upper(col("Response Answer")).like("%ADULTOS MAYORES%")) | (upper(col("Response Answer")).like("%TERCERA EDAD%"))), '2.1-7-3')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%BAJOS%")) | (upper(col("Response Answer")).like("%LOW-INCOME%"))), '2.1-7-4')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%MARGINALIZED%")) | (upper(col("Response Answer")).like("%INDIGENOUS%"))), '2.1-7-8')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%OUTDOOR%")) | (upper(col("Response Answer")).like("%WORKER%")) | (upper(col("Response Answer")).like("%TRABAJA%")) | (upper(col("Response Answer")).like("%FARMER%")) | (upper(col("Response Answer")).like("%AGRICU%")) | (upper(col("Response Answer")).like("%AGRO%"))), '2.1-7-9')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%HOUSING%")) | (upper(col("Response Answer")).like("%VIVIENDA%"))), '2.1-7-10')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%DISABILIT%")) | (upper(col("Response Answer")).like("%DISCAPACIDAD%"))), '2.1-7-11')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%UNEMPLOYED%")) | (upper(col("Response Answer")).like("%DESEMPLEA%"))), '2.1-7-12')
          .when((col('q_id') == '2.1-7') & ((upper(col("Response Answer")).like("%CHRONIC%")) | (upper(col("Response Answer")).like("%DISEASE%")) | (upper(col("Response Answer")).like("%ENFERMEDAD%")) | (upper(col("Response Answer")).like("%HEALTH%"))), '2.1-7-13')
          .when((col('q_id') == '2.1-7') & (upper(col("Response Answer")).like("%WOMEN%")), '2.1-7-14')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%WATER EFFICIENCY%")) | (upper(col("Response Answer")).like("%WATER USE%")) | (upper(col("Response Answer")).like("%METERING%"))), '3.0-2-1')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%WATER SUPPLY%")) | (upper(col("Response Answer")).like("%AGUA%"))), '3.0-2-2')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%WATER%")) | (upper(col("Response Answer")).like("%RAIN%"))), '3.0-2-3')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%HEAT%")) | (upper(col("Response Answer")).like("%HOT%"))| (upper(col("Response Answer")).like("%CALOR%")) | (upper(col("Response Answer")).like("%COOL%"))), '3.0-2-4')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%FLOOD%")) | (upper(col("Response Answer")).like("%INUND%"))| (upper(col("Response Answer")).like("%RAIL%"))), '3.0-2-5')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%LAND%")) | (upper(col("Response Answer")).like("%SOIL%")) | (upper(col("Response Answer")).like("%RIVER%"))), '3.0-2-6')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%MONITORING%")) | (upper(col("Response Answer")).like("%WARNING%"))), '3.0-2-7')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%PREPAREDNESS%")) | (upper(col("Response Answer")).like("%EVACUA%")) | (upper(col("Response Answer")).like("%EMERG%"))), '3.0-2-8')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%FOREST%")) | (upper(col("Response Answer")).like("%GREEN%")) | (upper(col("Response Answer")).like("%SHADING%")) | (upper(col("Response Answer")).like("%PLANTIN%")) | (upper(col("Response Answer")).like("%TREE%"))), '3.0-2-9')
          .when((col('q_id') == '3.0-2') & ((upper(col("Response Answer")).like("%INFRA%")) | (upper(col("Response Answer")).like("%BUILD%")) | (upper(col("Response Answer")).like("%CONSTR%"))), '3.0-2-10')
          .when((col('q_id') == '3.0-2') & (upper(col("Response Answer")).like("%EDUCA%")), '3.0-2-11')
          .when((col('q_id') == '3.0-2') & (upper(col("Response Answer")).like("%NO ACTION%")), '3.0-2-12')
          .when((col('q_id') == '2.2-1') & ((upper(col("Response Answer")).like("%ACCESS TO BASIC%")) | (upper(col("Response Answer")).like("%ACCESS TO EDUCATION%")) | (upper(col("Response Answer")).like("%ACCESS TO HEALTH%")) | (upper(col("Response Answer")).like("%POBRE%")) | (upper(col("Response Answer")).like("%PUBLIC HEALTH%"))), '2.2-1-1')
          .when((col('q_id') == '2.2-1') & ((upper(col("Response Answer")).like("%COMPROMISO%")) | (upper(col("Response Answer")).like("%ENGAGEMENT%")) | (upper(col("Response Answer")).like("%POLIT%"))), '2.2-1-2')
          .when((col('q_id') == '2.2-1') & ((upper(col("Response Answer")).like("%ECONOMI%")) | (upper(col("Response Answer")).like("%COST%"))), '2.2-1-3')
          .when((col('q_id') == '2.2-1') & (upper(col("Response Answer")).like("%CAPACITY%")), '2.2-1-4')
          .when((col('q_id') == '2.2-1') & ((upper(col("Response Answer")).like("%HOUSING%")) | (upper(col("Response Answer")).like("%INFORMAL%")) | (upper(col("Response Answer")).like("%VIVIENDA%"))), '2.2-1-5')
          .when((col('q_id') == '2.2-1') & (upper(col("Response Answer")).like("%CONDITIONS%")), '2.2-1-6')
          .when((col('q_id') == '2.2-1') & ((upper(col("Response Answer")).like("%LAND%")) | (upper(col("Response Answer")).like("%ECOSYSTEM%"))), '2.2-1-7')
          .when((col('q_id') == '2.2-1') & (upper(col("Response Answer")).like("%AVAILABILITY%")), '2.2-1-8')
          .when((col('q_id') == '2.2-1') & ((upper(col("Response Answer")).like("%SAFETY%")) | (upper(col("Response Answer")).like("%SEGURIDAD%"))), '2.2-1-9')
          .when((col('q_id') == '2.2-1') & ((upper(col("Response Answer")).like("%EMPLOY%")) | (upper(col("Response Answer")).like("%INEQUALITY%")) | (upper(col("Response Answer")).like("%INFORMAL ACTIVITIES%")) | (upper(col("Response Answer")).like("%MIGRATION%")) | (upper(col("Response Answer")).like("%POPUL%")) | (upper(col("Response Answer")).like("%POVERTY%"))), '2.2-1-10') 
          .when((col('q_id') == '6.2a-1') & ((upper(col("Response Answer")).like("%AGRICUL%")) | (upper(col("Response Answer")).like("%FORES%"))), '6.2a-1-1')
          .when((col('q_id') == '6.2a-1') & ((upper(col("Response Answer")).like("%INFRA%")) | (upper(col("Response Answer")).like("%BUILD%"))), '6.2a-1-2')
          .when((col('q_id') == '6.2a-1') & (upper(col("Response Answer")).like("%ENERG%")), '6.2a-1-3')
          .when((col('q_id') == '6.2a-1') & (upper(col("Response Answer")).like("%INDUSTR%")), '6.2a-1-4')
          .when((col('q_id') == '6.2a-1') & (upper(col("Response Answer")).like("%WATER%")), '6.2a-1-5')
          .when((col('q_id') == '6.2a-1') & (upper(col("Response Answer")).like("%WASTE%")), '6.2a-1-6')
          .when((col('q_id') == '6.2a-1') & (upper(col("Response Answer")).like("%TRANSPORT%")), '6.2a-1-7')
          .when((col('q_id') == '6.2a-1') & (upper(col("Response Answer")).like("%EMISS%")), '6.2a-1-8')
          .when((col('q_id') == '6.2a-1') & ((upper(col("Response Answer")).like("%PLANNING%")) | (upper(col("Response Answer")).like("%LAND%"))), '6.2a-1-9')
          .when((col('q_id') == '6.2a-1') & ((upper(col("Response Answer")).like("%BIOD%")) | (upper(col("Response Answer")).like("%ENVIRO%")) | (upper(col("Response Answer")).like("%ECOSYSTEM%"))), '6.2a-1-10')
          .when((col('q_id') == '6.2a-1') & ((upper(col("Response Answer")).like("%ADAPTATION%")) | (upper(col("Response Answer")).like("%RESILIENCE%"))), '6.2a-1-11')
          .when((col('q_id') == '6.2a-1') & (upper(col("Response Answer")).like("%EDUCA%")), '6.2a-1-12') 
          .when((col('q_id') == '6.0-1') & ((upper(col("Response Answer")).like("%ECONOM%")) | (upper(col("Response Answer")).like("%MARKET%"))), '6.0-1-1')
          .when((col('q_id') == '6.0-1') & ((upper(col("Response Answer")).like("%AMBIEN%")) | (upper(col("Response Answer")).like("%CLIMATE CHANGE RESILIENCY%")) | (upper(col("Response Answer")).like("%BIODIV%")) | (upper(col("Response Answer")).like("%CLIM%")) | (upper(col("Response Answer")).like("%ENVIRON%")) | (upper(col("Response Answer")).like("%NATURAL%")) | (upper(col("Response Answer")).like("%CLIM%"))), '6.0-1-2')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%CLEAN TECHNOLOGY%")), '6.0-1-3')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%ENERGY EFFICIENCY%")), '6.0-1-4')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%ENERGY SECURITY%")), '6.0-1-5')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%AGRIC%")), '6.0-1-6')
          .when((col('q_id') == '6.0-1') & ((upper(col("Response Answer")).like("%FLOOD%")) | (upper(col("Response Answer")).like("%INUND%"))), '6.0-1-7')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%FOOD%")), '6.0-1-8')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%INFRA%")), '6.0-1-9')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%PARTNERSHIP%")), '6.0-1-10')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%CONSERVATION%")), '6.0-1-11')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%CONSTRUC%")), '6.0-1-12')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%SUSTAINABLE TRANSP%")), '6.0-1-13')
          .when((col('q_id') == '6.0-1') & ((upper(col("Response Answer")).like("%TOURIS%")) | (upper(col("Response Answer")).like("%TURISM%"))), '6.0-1-14')
          .when((col('q_id') == '6.0-1') & ((upper(col("Response Answer")).like("%WASTE MANAGEMENT%")) | (upper(col("Response Answer")).like("%WASTE%")) | (upper(col("Response Answer")).like("%RESIDUO%"))), '6.0-1-15')
          .when((col('q_id') == '6.0-1') & ((upper(col("Response Answer")).like("%WATER%")) | (upper(col("Response Answer")).like("%AGUA%"))), '6.0-1-16')
          .when((col('q_id') == '6.0-1') & ((upper(col("Response Answer")).like("%FUND%")) | (upper(col("Response Answer")).like("%INVE%"))), '6.0-1-17')
          .when((col('q_id') == '6.0-1') & ((upper(col("Response Answer")).like("%HEALTH%")) | (upper(col("Response Answer")).like("%SAFETY%"))), '6.0-1-18')
          .when((col('q_id') == '6.0-1') & (upper(col("Response Answer")).like("%OPERA%")), '6.0-1-19')
          .when((col('q_id') == '2.2-2') & ((upper(col("Response Answer")).like("%SUPPORT%")) | (upper(col("Response Answer")).like("%ENHANCE%"))), '2.2-2-1')
          .when((col('q_id') == '2.2-2') & (upper(col("Response Answer")).like("%CHALLENGE%")), '2.2-2-2')
          .when((col('q_id') == '3.0-4') & ((upper(col("Response Answer")).like("%COMPLETE%")) | (upper(col("Response Answer")).like("%MONITOR%"))), '3.0-4-1')
          .when((col('q_id') == '3.0-4') & ((upper(col("Response Answer")).like("%OPERATION%")) | (upper(col("Response Answer")).like("%IMPLEMENTATION%"))), '3.0-4-2')
          .when((col('q_id') == '3.0-4') & ((upper(col("Response Answer")).like("%FEASIBILITY%")) | (upper(col("Response Answer")).like("%SCOPING%"))), '3.0-4-3')
          .when((col('q_id') == '2.1-8') & (upper(col("Response Answer")).like("%DECREASING%")), '2.1-8-1')
          .when((col('q_id') == '2.1-8') & (upper(col("Response Answer")).like("%INCREASING%")), '2.1-8-2')
          .when((col('q_id') == '2.1-8') & ((upper(col("Response Answer")).like("%NONE%")) | (upper(col("Response Answer")).like("%NOT EXPECTED%"))), '2.1-8-3')
          .when((col('q_id') == '2.1-9') & (upper(col("Response Answer")).like("%DECREASING%")), '2.1-9-1')
          .when((col('q_id') == '2.1-9') & (upper(col("Response Answer")).like("%INCREASING%")), '2.1-9-2')
          .when((col('q_id') == '2.1-9') & ((upper(col("Response Answer")).like("%NONE%")) | (upper(col("Response Answer")).like("%NOT EXPECTED%"))), '2.1-9-3')
          .otherwise(col('q_id')))#\
            
# '4.4-0' --> 1 = CH4, 2 = CO2, 3= HFCs, 4 = N2O, 5= NF3, 6= PFCs, 7=SF6 =========================SUMA
# '14.0-0' --> 1= 'Ground water', 2= 'Surface water', 3= 'Recycled water', 4= 'Seawater', 5= Rainwater' 
# '2_1_3' y '2_1_4' --> 1= high, 2= medium, 3= low, 4= not impact ====================================MEDIAM
# '2_1_11_1' --> 1= current, 2= short, 3= medium, 4= long  ====================================MEDIAM

#'2_1_1' --> 1 Biological hazards', 2 'Chemical change', 3 'Drought', 4 'Extreme cold temperature', 5 'Extreme hot temperature',  'Extreme Precipitation', 7 'Flood and sea level rise', 8 'Mass movement', 9 'Storm and wind', 10 'Water stress', 11 'Wild fire'

#'2_1_6' --> 'Building and Infrastructure', 'Commercial', 'Education', 'Emergency services', 'Energy''Environment, biodiversity, forestry', 'Food & agriculture', 'Industrial', 'Information and communication', 'Land use planning', 'Public health', 'Residential', 'Society / community & culture'), 'Tourism', 'Transportation', 'Water supply', 'Waste management'

#'2_1_7' --> 'All population', 'Children and youth', 'Elderly', 'Low-income households', 'Marginalized groups', 'Outdoor workers & Farmers', 'Persons living in sub-standard housing', 'Persons with disabilities', 'Unemployed persons''Vulnerable health groups', 'Women and girls'

#'3_0_2' --> 'Water use efficiency', 'Improve water supply', 'Water resilience', 'Heath action plan', 'Flood risk management', 'Resilience to mass movement', 'Real time risk monitoring', 'Public preparedness', 'Forest management & Green space restoration', ' Resilient infrastructure & building', 'Community engagement/education', 'No action'

# '2.2-1' --> 'Access to basic services/education/health', 'Community/Political engagement', 'Economic', 'Government, Infrastructure, Budgetary capacity', 'Housing', 'Infrastructure conditions / maintenance', 'Housing', 'Resource availability', 'Safety and security', 'Social'

# '6.2a-1' --> 'Agriculture and Forestry', 'Building and Infrastructure', 'Energy', 'Industry', 'Water', 'Waste', 'Transportation', 'Emissions reduction', 'Spatial Planning', 'Natural environment', 'Adaptation', 'Education'

# '6.0-1' --> 'Circular economy / Markets', 'Climate change resiliency projects', 'Clean technology businesses', 'Energy efficiency measures and technologies', 'Energy security', 'Extended agricultural seasons', 'Flood risk mitigation', 'Food security and sustainable', 'Infrastructure investment', 'Partnerships', 'Resource conservation and management', 'Sustainable construction/real estate sector', 'Sustainable transport sector', 'Tourism and eco-tourism sector', 'Waste management', 'Water management sector', 'Funding opportunities', 'Health and Safety', 'Efficiency of operations'

#df_cities_all = df_cities_all.drop('q_id')
#df_cities_all = df_cities_all.withColumnRename("q_id_fix", 'q_id')



# COMMAND ----------

df_cities_all = df_cities_all.drop('q_id')
df_cities_all = df_cities_all.withColumnRenamed("q_id_fix", 'q_id')

# COMMAND ----------

# Save as parquet file
df_cities_all.write.parquet("/FileStore/df_cities_all.parquet", 
                            mode="overwrite")

# COMMAND ----------

# DBTITLE 1,4) Create Questions Dataframe
import pyspark.sql.functions as F

# Group by unique q_ids
df_questions = df_cities_all.groupBy('q_id').count()

# Group unique q_ids for Question-Column-Row (292 questions)
df_cities_2021_questions1 = df_cities_2021.filter(df_cities_2021['q_type'] == 'Question-Column-Row').groupBy('q_id', 'Question Number', 'Column Number', 'Row Number', 'Parent Section', 'Section', 'Question Name', 'Column Name', 'Row Name', 'q_type').count()

# Group unique q_ids for Question-Column (115 questions)
df_cities_2021_questions2 = df_cities_2021.filter(df_cities_2021['q_type'] == 'Question-Column').groupBy('q_id', 'Question Number', 'Column Number', 'Parent Section', 'Section', 'Question Name', 'Column Name', 'q_type').count()
df_cities_2021_questions2 = df_cities_2021_questions2.withColumn('Row Number', F.lit('none')) \
                                                     .withColumn('Row Name', F.lit('none'))
df_cities_2021_questions2 = df_cities_2021_questions2.select('q_id', 'Question Number', 'Column Number', 'Row Number', 'Parent Section', 'Section', 'Question Name', 'Column Name', 'Row Name', 'q_type', 'count')

# Unify scenarios
df_cities_2021_questions = df_cities_2021_questions1.union(df_cities_2021_questions2)
df_cities_2021_questions = df_cities_2021_questions.withColumn('q_id2', df_cities_2021_questions['q_id'])
df_cities_2021_questions = df_cities_2021_questions.drop('count', 'q_id')

# COMMAND ----------

#Create df_questions
df_questions = df_questions.join(df_cities_2021_questions, df_questions.q_id == df_cities_2021_questions.q_id2, "left")
df_questions = df_questions.drop('q_id2')

# Set nulls to null value
df_questions = df_questions.withColumn("Row Number", coalesce(col("Row Number"), lit(None)))\
                           .withColumn("Row Name", coalesce(col("Row Name"), lit(None)))\
                           .withColumn("Column Name", coalesce(col("Column Name"), lit(None)))\

# Save as parquet file
df_questions.write.parquet("/FileStore/df_questions.parquet", 
                            mode="overwrite")

#print(df_questions.shape())
display(df_questions)

# COMMAND ----------

# DBTITLE 1,5) Create Responses Dataframe (Fact Table)
from pyspark.sql.functions import col, coalesce

df_responses = df_cities_all.select("Account Number","Questionnaire","q_id", "Response Answer")
df_responses = df_responses.withColumn('year', df_responses.Questionnaire.substr(-4,4))\
                           .withColumn("Response Answer", when(col("Response Answer") == "Question not applicable", lit(None)).otherwise(col("Response Answer")))

# Unify 'None' and 'null' as null values
df_responses = df_responses.withColumn("Response Answer", coalesce(col("Response Answer"), lit(None)))

# Add rows IDs
df_responses = df_responses.select(concat_ws('_', df_responses['Account Number'], df_responses['year'])
              .alias('account_year'), "Account Number", "year", "q_id", "Response Answer")

# Save as parquet file
df_responses.write.parquet("/FileStore/df_responses.parquet", 
                            mode="overwrite")

#print(df_responses.shape())

# COMMAND ----------

display(df_responses)

# COMMAND ----------

df_responses_unique = df_responses.groupby("q_id", "Response Answer").count()
#display(df_responses_unique.filter(df_responses_unique['count'] > 1))

# COMMAND ----------

from pyspark.sql.functions import sum

# Filtrar valores numéricos
df_responses_unique_numeric = df_responses_unique.filter(col("Response Answer").cast("double").isNotNull())
df_responses_unique_numeric = df_responses_unique_numeric.groupby("q_id").agg(sum("count").alias("total_count"))

# Filtrar valores de texto
df_responses_unique_text = df_responses_unique.filter(col("Response Answer").cast("double").isNull())

# COMMAND ----------

# DBTITLE 1,From here!
# Input data from Databricks FileStore
df_responses = spark.read.format('parquet').options(header=True,inferSchema=True).load('dbfs:/FileStore/df_responses.parquet/')
df_account = spark.read.format('parquet').options(header=True,inferSchema=True).load('dbfs:/FileStore/df_account.parquet/')

# COMMAND ----------

# Replace Null values
columns_free_text = ['13.2-0', '14.2a-5', '14.3-4', '2.1-5', '2.2-4', '3.0-8', '3.3-1', '3.3-4', '6.0-2', '6.2a-3', '6.5-7', '7.6-4-1']

df_responses_free = df_responses.filter(col("q_id").isin(columns_free_text))
df_responses_nofree = df_responses.filter(~col("q_id").isin(columns_free_text))

# Replace posible null values in all Responses 
replace_nulls = [None, ' NO ', ' YET', 'DO NOT KNOW', 'DOES NOT HAVE THIS DATA' , 'NOT AVAILABLE', 'NO DISPONIBLE', 'NA', 'N/A', 'NA.', 'N/A.', 'NOT APPLICABLE', 'NOT APPLICABLE.', 'NÃ£O SE APLICA', 'NÃ£O APLICÃ¡VEL', 'NÃ£O SE APLICA.', 'NO APLICA', 'NO TARGET', 'NO TIENE', 'NO WEB LINK AVAILABLE', 'NOT', 'NULL']

df_responses_nofree = df_responses_nofree.withColumn("Response Answer", when(upper(col("Response Answer")).isin(replace_nulls), None).otherwise(col("Response Answer")))

# Rebuild df_responses
df_responses = df_responses_nofree.unionByName(df_responses_free)

# COMMAND ----------

# Create pivot dataframe

# Replace "." by "_" in column q_id in order to manage properly the resulting column names after pivoting ("." cause issues)
from pyspark.sql.functions import regexp_replace
#df_responses = df_responses.withColumn("q_id", regexp_replace("q_id", "\\.", "_"))
df_responses = df_responses.withColumn("q_id", regexp_replace("q_id", "[\\.-]", "_"))
df_responses = df_responses.withColumnRenamed("Account Number", "account")

# Pivot
#df_responses_piv = df_responses.groupBy('account_year', "account", "year").pivot("q_id").agg(first(col("Response Answer")))
#display(df_responses_piv)


# COMMAND ----------

df_responses_piv_test = df_responses.groupBy('account_year', "account", "year", "Response Answer").pivot("q_id").agg(first(col("Response Answer")))
display(df_responses_piv_test)

# COMMAND ----------

df_responses_piv = df_responses_piv_test

# COMMAND ----------

columns_ids = ['account_year', 'Account Number', 'year']
# Questions columns
columns_questions = [column for column in df_responses_piv.columns if column not in columns_ids]

# Questions dataframe (pivot)
df_responses_piv_q = df_responses_piv.select(columns_questions)

# COMMAND ----------

# Replace Yes/No columns 1
columns_yes_no = ["2_0_0", "3_2_0", "4_0_0", "4_9_1_1", "5_5_0", "7_0_0", "7_7_0", "8_0_0"]

df_responses_piv_q2 = df_responses_piv
for c in df_responses_piv.columns:
    if c in columns_yes_no:
        new_column_name = "c" + c
        df_responses_piv_q2 = df_responses_piv_q2.withColumn(new_column_name,
                                                             when(upper(col(c)).like("%YES%"), '1')
                                                             .when(upper(col(c)).like("%IN PROGRESS%"), '1')
                                                             .when(col(c).isNull(), None)
                                                             .otherwise('0'))

# Replace Yes/No columns 2
# List of replacement for questio 6.2-0
p6_2_0 = ['%academ%', '%build%','%business%','%clima%','%Communication Services%', '%Consumer Discretionary%', '%Consumer Staples%', '%educa%', '%energ%', '%Financials%', '%health%', '%In progress%', '%Industr%', '%information%', '%material%', '%mobil%', '%real%', '%Transport%', '%utilit%', '%waste%', '%yes%']

# 6.2-0 conditions of matchs
condiciones = [col("6_2_0").like(patron) for patron in p6_2_0]
condicion_final = condiciones[0]
for condicion in condiciones[1:]:
    condicion_final = condicion_final | condicion

df_responses_piv_q2 = df_responses_piv_q2.withColumn("c2_0b_2", 
                                                    when(length(col("2_0b_2")) != 0, '1')
                                                    .when(col("2_0b_2").isNull(), None)
                                                    .otherwise('0'))\
.withColumn("c2_0b_7", 
            when(length(col("2_0b_7")) != 0, '1')
            .when(col("2_0b_7").isNull(), None)
            .otherwise('0'))\
.withColumn("c6_2_0",
              when(condicion_final, "1")
              .when(col("6_2_0").isNull(), None)
              .otherwise("0"))\
.withColumn("c5_0_0",
              when(upper(col("5_0_0")) == "NO TARGET", '0')
              .when(col("5_0_0").isNull(), None)
              .otherwise("1"))

# COMMAND ----------

#Replace other category columns
df_responses_piv_q2 = df_responses_piv_q2.withColumn("c0_1_1_1", 
          when((upper(col("0_1_1_1")).like("%DISTRI%")), 'District')
          .when((upper(col("0_1_1_1")).like("%COUNTY%")), 'County')
          .when((upper(col("0_1_1_1")).like("%METROPOL%")), 'Greater City')
          .when((upper(col("0_1_1_1")).like("%CITY%")) | (upper(col("0_1_1_1")).like("%CIUDAD%")) | (upper(col("0_1_1_1")).like("%MUNIC%")), 'City')
          .otherwise(None))\
.withColumn("c5_0_sector",
          when(length(col("5_0a_1")) != 0, col("5_0a_1"))
          .when(length(col("5_0c_1")) != 0, col("5_0c_1"))
          .when(length(col("5_0d_1")) != 0, col("5_0d_1"))
          .otherwise(None))\
.withColumn("csector_",
          when(length(col("3_2a_3")) != 0, col("3_2a_3"))
          .when(length(col("4_6d_2")) != 0, col("4_6d_2"))
          .when(length(col("c5_0_sector")) != 0, col("c5_0_sector"))
          .otherwise(None))\
.withColumn("cboundary_",
          when(length(col("2_0b_4")) != 0, col("2_0b_4"))
          .when(length(col("3_2a_6")) != 0, col("3_2a_6"))
          .when(length(col("5_0a_3")) != 0, col("5_0a_3"))
          .when(length(col("5_0b_3")) != 0, col("5_0b_3"))
          .when(length(col("5_0c_3")) != 0, col("5_0c_3"))
          .when(length(col("5_0d_3")) != 0, col("5_0d_3"))
          .otherwise(None))\
.withColumn("c6_5_3", 
          when((upper(col("6_5_3")).like("%COMPLETE%")) | (upper(col("6_5_3")).like("%MONITOR%")), 'Complete')
          .when((upper(col("6_5_3")).like("%OPERATION%")) | (upper(col("6_5_3")).like("%IMPLEMENTATION%")) | (upper(col("6_5_3")).like("%TRANSAC%")), 'Implementation')
          .when((upper(col("6_5_3")).like("%FEASIBILITY%")) | (upper(col("6_5_3")).like("%SCOPING%")) | (upper(col("6_5_3")).like("%STRUCT%")), 'Scoping')
          .when(col("6_5_3").isNull(), None)
          .otherwise("Other"))\
.withColumn("c6_5_4", 
          when((upper(col("6_5_4")).like("%NOT FUNDED%")) | (upper(col("6_5_4")).like("%PROJECT NOT FUNDED%")) | (upper(col("6_5_4")).like("%PROJETO NÃ£O FINANCIADO%")), 'Not funded')
          .when((upper(col("6_5_4")).like("%FUNDED%")) | (upper(col("6_5_4")).like("%FINANCED%")) | (upper(col("6_5_4")).like("%FINANCIADO%")) | (upper(col("6_5_4")).like("%FULLY FUNDED%")) | (upper(col("6_5_4")).like("%FULLY FINANCED%")) | (upper(col("6_5_4")).like("%PARTIALLY FUNDED%")), 'Funded')
          .when(col("6_5_4").isNull(), None)
          .otherwise("Other"))\
.withColumn("c3_2a_9", 
          when((upper(col("3_2a_9")).like("%ADDRESSED%")), 'Addressed')
          .when((upper(col("3_2a_9")).like("%INTEGRATED%")), 'Integrated')
          .when((upper(col("3_2a_9")).like("%STANDALONE%")), 'Standalone')
          .when(col("3_2a_9").isNull(), None)
          .otherwise("Other"))\
.withColumn("csector", 
          when((upper(col("csector_")).like("%AGRICUL%")) | (upper(col("csector_")).like("%FORESTRY%")), 'Agriculture and Forestry')
          .when((upper(col("csector_")).like("%INFRA%")) | (upper(col("csector_")).like("%BUILD%")), 'Building and Infrastructure')
          .when((upper(col("csector_")).like("%ENERG%")) | (upper(col("csector_")).like("%ELECT%")), 'Energy')
          .when((upper(col("csector_")).like("%INDUSTR%")) | (upper(col("csector_")).like("%MANUF%")), 'Industry')
          .when((upper(col("csector_")).like("%WATER%")) | (upper(col("csector_")).like("%AGUA%")), 'Water')
          .when((upper(col("csector_")).like("%WASTE%")) | (upper(col("csector_")).like("%RESID%"))| (upper(col("csector_")).like("%DESECH%")), 'Waste')
          .when((upper(col("csector_")).like("%RESIDENTIAL%")), 'Residential')
          .when((upper(col("csector_")).like("%TRANSPORT%")) | (upper(col("csector_")).like("%ROAD%"))| (upper(col("csector_")).like("%RAIL%")), 'Transportation')
          .when((upper(col("csector_")).like("%TARGET%")), 'Target')
          .when((upper(col("csector_")).like("%PLANNING%")), 'Spatial Planning')
          .when((upper(col("csector_")).like("%HEALTH%")), 'Public Health and Safety')
          .when((upper(col("csector_")).like("%SOCIAL%")), 'Social Services')
          .when((upper(col("csector_")).like("%ALL%")) | (upper(col("csector_")).like("%TOTAL%")), 'All')
          .when(col("csector_").isNull(), None)
          .otherwise("Other"))\
.withColumn("cboundary", 
          when((upper(col("cboundary_")).like("%LARGER%")), 'Larger')
          .when((upper(col("cboundary_")).like("%PARTIAL%")), 'Partial')
          .when((upper(col("cboundary_")).like("%SMALLER%")), 'Smaller')
          .when(col("cboundary_").isNull(), None)
          .otherwise("Same"))\
.withColumn("c5_0_base_year",
          when(length(col("5_0a_5")) != 0, col("5_0a_5"))
          .when(length(col("5_0c_5")) != 0, col("5_0c_5"))
          .when(length(col("5_0d_5")) != 0, col("5_0d_5"))
          .otherwise(None))\
.withColumn("c5_0_target_year",
          when(length(col('5_0a_9')) != 0, col('5_0a_9'))
          .when(length(col('5_0b_7')) != 0, col('5_0b_7'))
          .when(length(col('5_0c_11')) != 0, col('5_0c_11'))
          .when(length(col('5_0d_8')) != 0, col('5_0d_8'))
          .otherwise(None))\
.withColumn("c5_0_target_year_set",
          when(length(col('5_0a_6')) != 0, col('5_0a_6'))
          .when(length(col('5_0b_5')) != 0, col('5_0b_5'))
          .when(length(col('5_0c_6')) != 0, col('5_0c_6'))
          .when(length(col('5_0d_6')) != 0, col('5_0d_6'))
          .otherwise(None))#\


# COMMAND ----------

#Replace other category columns
#df_responses_piv_q2 = df_responses_piv_q2.withColumn("c2_1_3", 
                                                    when((upper(col("2_1_3")).like("%HIGH%")) | (upper(col("2_1_3")).like("%EXTREME%")), '4')
                                                    .when((upper(col("2_1_3")).like("%MEDIUM%")) | (upper(col("2_1_3")) == "SERIOUS"), '3')
                                                    .when((upper(col("2_1_3")).like("%LOW%")) | (upper(col("2_1_3")).like("%LESS%")), '2')
                                                    .when((upper(col("2_1_3")).like("%IMPACT%")), '1')
                                                    .when(col("2_1_3").isNull(), None)
                                                    .otherwise('0'))\
#.withColumn("c2_1_4", 
          when((upper(col("2_1_4")).like("%HIGH%")) | (upper(col("2_1_4")).like("%EXTREME%")), '4')
          .when((upper(col("2_1_4")).like("%MEDIUM%")) | (upper(col("2_1_4")) == "SERIOUS"), '3')
          .when((upper(col("2_1_4")).like("%LOW%")) | (upper(col("2_1_4")).like("%LESS%")), '2')
          .when((upper(col("2_1_4")).like("%IMPACT%")), '1')
          .when(col("2_1_4").isNull(), None)
          .otherwise('0'))\
#.withColumn("c2_1_11", 
          when((upper(col("2_1_11")).like("%IMMEDIATELY%")) | (upper(col("2_1_11")).like("%CURRENT%")), '4')
          .when((upper(col("2_1_11")).like("%SHORT%")), '3')
          .when((upper(col("2_1_11")).like("%MEDIUM%")), '2')
          .when((upper(col("2_1_11")).like("%LONG%")), '1')
          .when(col("2_1_11").isNull(), None)
          .otherwise('0'))\
.withColumn("c0_1_1_1", 
          when((upper(col("0_1_1_1")).like("%DISTRI%")), 'District')
          .when((upper(col("0_1_1_1")).like("%COUNTY%")), 'County')
          .when((upper(col("0_1_1_1")).like("%METROPOL%")), 'Greater City')
          .when((upper(col("0_1_1_1")).like("%CITY%")) | (upper(col("0_1_1_1")).like("%CIUDAD%")) | (upper(col("0_1_1_1")).like("%MUNIC%")), 'City')
          .otherwise(None))\
#.withColumn("c14_0_0", 
          when((upper(col("14_0_0")).like("%GROUND%")), 'Ground water')
          .when((upper(col("14_0_0")).like("%SURFACE%")), 'Surface water')
          #.when((upper(col("14_0_0")).like("%RECYCLED%")), 'Recycled water')
          #.when((upper(col("14_0_0")).like("%SEAWATER%")), 'Seawater')
          #.when((upper(col("14_0_0")).like("%RAINWATER%")) | (upper(col("0_1_1_1")).like("%LLUVIA%")), 'Rainwater')
          .when(col("14_0_0").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c4_4_0", 
          when((upper(col("4_4_0")).like("%N2%")), 'N2O')
          .when((upper(col("4_4_0")) == "CH4"), 'CH4')
          .when((upper(col("4_4_0")) == "CO2"), 'CO2')
          .when(col("4_4_0").isNull(), None)
          .otherwise("Other"))\
.withColumn("c5_0_sector",
          when(length(col("5_0a_1")) != 0, col("5_0a_1"))
          .when(length(col("5_0c_1")) != 0, col("5_0c_1"))
          .when(length(col("5_0d_1")) != 0, col("5_0d_1"))
          .otherwise(None))\
.withColumn("csector_",
          when(length(col("3_2a_3")) != 0, col("3_2a_3"))
          .when(length(col("4_6d_2")) != 0, col("4_6d_2"))
          .when(length(col("c5_0_sector")) != 0, col("c5_0_sector"))
          .otherwise(None))\
.withColumn("cboundary_",
          when(length(col("2_0b_4")) != 0, col("2_0b_4"))
          .when(length(col("3_2a_6")) != 0, col("3_2a_6"))
          .when(length(col("5_0a_3")) != 0, col("5_0a_3"))
          .when(length(col("5_0b_3")) != 0, col("5_0b_3"))
          .when(length(col("5_0c_3")) != 0, col("5_0c_3"))
          .when(length(col("5_0d_3")) != 0, col("5_0d_3"))
          .otherwise(None))\
#.withColumn("c3_0_2", 
          when((upper(col("3_0_2")).like("%WATER EFFICIENCY%")) | (upper(col("3_0_2")).like("%WATER USE%")) | (upper(col("3_0_2")).like("%METERING%")), 'Water use efficiency')
          .when((upper(col("3_0_2")).like("%WATER SUPPLY%")) | (upper(col("3_0_2")).like("%AGUA%")), 'Improve water supply')
          .when((upper(col("3_0_2")).like("%WATER%")) | (upper(col("3_0_2")).like("%RAIN%")), 'Water resilience')
          .when((upper(col("3_0_2")).like("%HEAT%")) | (upper(col("3_0_2")).like("%HOT%"))| (upper(col("3_0_2")).like("%CALOR%")) | (upper(col("3_0_2")).like("%COOL%")), 'Heath action plan')
          .when((upper(col("3_0_2")).like("%FLOOD%")) | (upper(col("3_0_2")).like("%INUND%"))| (upper(col("3_0_2")).like("%RAIL%")), 'Flood risk management')
          .when((upper(col("3_0_2")).like("%LAND%")) | (upper(col("3_0_2")).like("%SOIL%")) | (upper(col("3_0_2")).like("%RIVER%")), 'Resilience to mass movement')
          .when((upper(col("3_0_2")).like("%MONITORING%")) | (upper(col("3_0_2")).like("%WARNING%")), 'Real time risk monitoring')
          .when((upper(col("3_0_2")).like("%PREPAREDNESS%")) | (upper(col("3_0_2")).like("%EVACUA%")) | (upper(col("3_0_2")).like("%EMERG%")), 'Public preparedness')
          .when((upper(col("3_0_2")).like("%FOREST%")) | (upper(col("3_0_2")).like("%GREEN%")) | (upper(col("3_0_2")).like("%SHADING%")) | (upper(col("3_0_2")).like("%PLANTIN%")) | (upper(col("3_0_2")).like("%TREE%")), 'Forest management & Green space restoration')
          .when((upper(col("3_0_2")).like("%INFRA%")) | (upper(col("3_0_2")).like("%BUILD%")) | (upper(col("3_0_2")).like("%CONSTR%")), ' Resilient infrastructure & building')
          .when((upper(col("3_0_2")).like("%EDUCA%")), 'Community engagement/education')
          .when((upper(col("3_0_2")).like("%NO ACTION%")), 'No action')
          .when(col("3_0_2").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c6_2a_1", 
          when((upper(col("6_2a_1")).like("%AGRICUL%")) | (upper(col("6_2a_1")).like("%FORES%")), 'Agriculture and Forestry')
          .when((upper(col("6_2a_1")).like("%INFRA%")) | (upper(col("6_2a_1")).like("%BUILD%")), 'Building and Infrastructure')
          .when((upper(col("6_2a_1")).like("%ENERG%")), 'Energy')
          .when((upper(col("6_2a_1")).like("%INDUSTR%")), 'Industry')
          .when((upper(col("6_2a_1")).like("%WATER%")), 'Water')
          .when((upper(col("6_2a_1")).like("%WASTE%")), 'Waste')
          .when((upper(col("6_2a_1")).like("%TRANSPORT%")), 'Transportation')
          .when((upper(col("6_2a_1")).like("%EMISS%")), 'Emissions reduction')
          .when((upper(col("6_2a_1")).like("%PLANNING%")) | (upper(col("6_2a_1")).like("%LAND%")), 'Spatial Planning')
          .when((upper(col("6_2a_1")).like("%BIOD%")) | (upper(col("6_2a_1")).like("%ENVIRO%")) | (upper(col("6_2a_1")).like("%ECOSYSTEM%")), 'Natural environment')
          .when((upper(col("6_2a_1")).like("%ADAPTATION%")) | (upper(col("6_2a_1")).like("%RESILIENCE%")), 'Adaptation')
          #.when((upper(col("6_2a_1")).like("%HEALTH%")), 'Public Health')
          .when((upper(col("6_2a_1")).like("%EDUCA%")), 'Education') 
          .when(col("6_2a_1").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c3_0_4", 
          when((upper(col("3_0_4")).like("%COMPLETE%")) | (upper(col("3_0_4")).like("%MONITOR%")), 'Complete')
          .when((upper(col("3_0_4")).like("%OPERATION%")) | (upper(col("3_0_4")).like("%IMPLEMENTATION%")), 'Implementation')
          .when((upper(col("3_0_4")).like("%FEASIBILITY%")) | (upper(col("3_0_4")).like("%SCOPING%")), 'Scoping')
          .when(col("3_0_4").isNull(), None)
          .otherwise("Other"))\
.withColumn("c6_5_3", 
          when((upper(col("6_5_3")).like("%COMPLETE%")) | (upper(col("6_5_3")).like("%MONITOR%")), 'Complete')
          .when((upper(col("6_5_3")).like("%OPERATION%")) | (upper(col("6_5_3")).like("%IMPLEMENTATION%")) | (upper(col("6_5_3")).like("%TRANSAC%")), 'Implementation')
          .when((upper(col("6_5_3")).like("%FEASIBILITY%")) | (upper(col("6_5_3")).like("%SCOPING%")) | (upper(col("6_5_3")).like("%STRUCT%")), 'Scoping')
          .when(col("6_5_3").isNull(), None)
          .otherwise("Other"))\
.withColumn("c6_5_4", 
          when((upper(col("6_5_4")).like("%NOT FUNDED%")) | (upper(col("6_5_4")).like("%PROJECT NOT FUNDED%")) | (upper(col("6_5_4")).like("%PROJETO NÃ£O FINANCIADO%")), 'Not funded')
          .when((upper(col("6_5_4")).like("%FUNDED%")) | (upper(col("6_5_4")).like("%FINANCED%")) | (upper(col("6_5_4")).like("%FINANCIADO%")) | (upper(col("6_5_4")).like("%FULLY FUNDED%")) | (upper(col("6_5_4")).like("%FULLY FINANCED%")) | (upper(col("6_5_4")).like("%PARTIALLY FUNDED%")), 'Funded')
          .when(col("6_5_4").isNull(), None)
          .otherwise("Other"))\
.withColumn("c3_2a_9", 
          when((upper(col("3_2a_9")).like("%ADDRESSED%")), 'Addressed')
          .when((upper(col("3_2a_9")).like("%INTEGRATED%")), 'Integrated')
          .when((upper(col("3_2a_9")).like("%STANDALONE%")), 'Standalone')
          .when(col("3_2a_9").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c2_2_1", 
          when((upper(col("2_2_1")).like("%ACCESS TO BASIC%")) | (upper(col("2_2_1")).like("%ACCESS TO EDUCATION%")) | (upper(col("2_2_1")).like("%ACCESS TO HEALTH%")) | (upper(col("2_2_1")).like("%POBRE%")) | (upper(col("2_2_1")).like("%PUBLIC HEALTH%")), 'Access to basic services/education/health')
          .when((upper(col("2_2_1")).like("%COMPROMISO%")) | (upper(col("2_2_1")).like("%ENGAGEMENT%")) | (upper(col("2_2_1")).like("%POLIT%")), 'Community/Political engagement')
          .when((upper(col("2_2_1")).like("%ECONOMI%")) | (upper(col("2_2_1")).like("%COST%")), 'Economic')
          .when((upper(col("2_2_1")).like("%CAPACITY%")), 'Government, Infrastructure, Budgetary capacity')
          .when((upper(col("2_2_1")).like("%HOUSING%")) | (upper(col("2_2_1")).like("%INFORMAL%")) | (upper(col("2_2_1")).like("%VIVIENDA%")), 'Housing')
          .when((upper(col("2_2_1")).like("%CONDITIONS%")), 'Infrastructure conditions / maintenance')
          .when((upper(col("2_2_1")).like("%LAND%")) | (upper(col("2_2_1")).like("%ECOSYSTEM%")), 'Housing')
          .when((upper(col("2_2_1")).like("%AVAILABILITY%")), 'Resource availability')
          .when((upper(col("2_2_1")).like("%SAFETY%")) | (upper(col("2_2_1")).like("%SEGURIDAD%")), 'Safety and security')
          .when((upper(col("2_2_1")).like("%EMPLOY%")) | (upper(col("2_2_1")).like("%INEQUALITY%")) | (upper(col("2_2_1")).like("%INFORMAL ACTIVITIES%")) | (upper(col("2_2_1")).like("%MIGRATION%")) | (upper(col("2_2_1")).like("%POPUL%")) | (upper(col("2_2_1")).like("%POVERTY%")), 'Social')
          .when(col("2_2_1").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c2_1_8", 
          when((upper(col("2_1_8")).like("%NONE%")) | (upper(col("2_1_8")).like("%NOT EXPECTED%")), 'Not expected')
          .when(col("2_1_8").isNull(), None)
          .otherwise(col("2_1_8")))\
#.withColumn("c2_1_9", 
          when((upper(col("2_1_9")).like("%NONE%")) | (upper(col("2_1_9")).like("%NOT EXPECTED%")), 'Not expected')
          .when(col("2_1_9").isNull(), None)
          .otherwise(col("2_1_9")))\
#.withColumn("c2_1_1", 
          when((upper(col("2_1_1")).like("%BIOLOG%")) | (upper(col("2_1_1")).like("%DESEASE%")) | (upper(col("2_1_1")).like("%INFESTATION%")) | (upper(col("2_1_1")).like("%ENFERMEDAD%")) | (upper(col("2_1_1")).like("%INSECTO%")), 'Biological hazards')
          .when((upper(col("2_1_1")).like("%CHEMICAL CHANGE%")) | (upper(col("2_1_1")).like("%CO2%")) | (upper(col("2_1_1")).like("%SALT%")) | (upper(col("2_1_1")).like("%ACIDIFICATION%")) | (upper(col("2_1_1")).like("%AIR%")), 'Chemical change')
          .when((upper(col("2_1_1")).like("%DROUGHT%")) | (upper(col("2_1_1")).like("%SEQU%")), 'Drought')
          .when((upper(col("2_1_1")).like("%COLD%")) | (upper(col("2_1_1")).like("%WINTER%")) | (upper(col("2_1_1")).like("%FRÃOS%")) | (upper(col("2_1_1")).like("%FREEZ%")) | (upper(col("2_1_1")).like("%HELADA%")) | (upper(col("2_1_1")).like("%INVER%")) | (upper(col("2_1_1")).like("%NEVADA%")) | (upper(col("2_1_1")).like("%NIEVE%")), 'Extreme cold temperature')
          .when((upper(col("2_1_1")).like("%HOT%")) | (upper(col("2_1_1")).like("%HEAT%")) | (upper(col("2_1_1")).like("%CALOR%")) | (upper(col("2_1_1")).like("%CALURO%")), 'Extreme hot temperature')
          .when((upper(col("2_1_1")).like("%EXTREME PRECIPITATION%")) | (upper(col("2_1_1")).like("%HEAVY PRECIPITATION%")) | (upper(col("2_1_1")).like("%MONSOON%")) | (upper(col("2_1_1")).like("%SNOW%")) | (upper(col("2_1_1")).like("%HAIL%")) | (upper(col("2_1_1")).like("%FOG%")) | (upper(col("2_1_1")).like("%GRANIZ%")) | (upper(col("2_1_1")).like("%LLUVIA%")) | (upper(col("2_1_1")).like("%NIEBLA%")) | (upper(col("2_1_1")).like("%NEBLIN%")), 'Extreme Precipitation')
          .when((upper(col("2_1_1")).like("%FLOOD%")) | (upper(col("2_1_1")).like("%INUNDA%")) | (upper(col("2_1_1")).like("%MAR%")) | (upper(col("2_1_1")).like("%SEA LEVEL%")) | (upper(col("2_1_1")).like("%LEVEL RISE%")), 'Flood and sea level rise')
          .when((upper(col("2_1_1")).like("%MASS%")) | (upper(col("2_1_1")).like("%LANDSLIDE%")) | (upper(col("2_1_1")).like("%SUBSIDENC%")) | (upper(col("2_1_1")).like("%AVALAN%")) | (upper(col("2_1_1")).like("%DESLAVE%")) | (upper(col("2_1_1")).like("%HUNDIMIENTO%")) | (upper(col("2_1_1")).like("%ROCA%")) | (upper(col("2_1_1")).like("%ROCK%")), 'Mass movement')
          .when((upper(col("2_1_1")).like("%STORM%")) | (upper(col("2_1_1")).like("%WIND%")) | (upper(col("2_1_1")).like("%TORMENT%")) | (upper(col("2_1_1")).like("%TORNA%")) | (upper(col("2_1_1")).like("%HURRICAN%")) | (upper(col("2_1_1")).like("%HURAC%")), 'Storm and wind')
          .when((upper(col("2_1_1")).like("%WATER%")) | (upper(col("2_1_1")).like("%AGUA%")), 'Water stress')
          .when((upper(col("2_1_1")).like("%FIRE%")) | (upper(col("2_1_1")).like("%INCENDIO%")), 'Wild fire')
          .when(col("2_1_1").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c2_1_6", 
          when((upper(col("2_1_6")).like("%CONSTRUC%")) | (upper(col("2_1_6")).like("%INFRA%")) | (upper(col("2_1_6")).like("%BUILDIN%")), 'Building and Infrastructure')
          .when((upper(col("2_1_6")).like("%COMMERCIAL%")) | (upper(col("2_1_6")).like("%COMERCI%")), 'Commercial')
          .when((upper(col("2_1_6")).like("%EDUCA%")) | (upper(col("2_1_6")).like("%TECHNICAL%")), 'Education')
          .when((upper(col("2_1_6")).like("%EMERGENC%")), 'Emergency services')
          .when((upper(col("2_1_6")).like("%ENERG%")) | (upper(col("2_1_6")).like("%ELECTRIC%")), 'Energy')
          .when((upper(col("2_1_6")).like("%ENVIRON%")) | (upper(col("2_1_6")).like("%AMBIENTE%")) | (upper(col("2_1_6")).like("%BIODIVER%")) | (upper(col("2_1_6")).like("%FOREST%")), 'Environment, biodiversity, forestry')
          .when((upper(col("2_1_6")).like("%FOOD%")) | (upper(col("2_1_6")).like("%AGRICULTUR%")) | (upper(col("2_1_6")).like("%ALIMENT%")), 'Food & agriculture')
          .when((upper(col("2_1_6")).like("%INDUSTR%")) | (upper(col("2_1_6")).like("%FABRIC%")) | (upper(col("2_1_6")).like("%MANUFAC%")), 'Industrial')
          .when((upper(col("2_1_6")).like("%INFORMA%")) | (upper(col("2_1_6")).like("%COMMUNICA%")), 'Information and communication')
          .when((upper(col("2_1_6")).like("%LAND%")) | (upper(col("2_1_6")).like("%PLANIFIC%")) | (upper(col("2_1_6")).like("%TIERRA%")), 'Land use planning')
          .when((upper(col("2_1_6")).like("%HEALTH%")) | (upper(col("2_1_6")).like("%SALUD%")), 'Public health')
          .when((upper(col("2_1_6")).like("%RESIDEN%")) | (upper(col("2_1_6")).like("%HOUSING%")) | (upper(col("2_1_6")).like("%VIVIEND%")) | (upper(col("2_1_6")).like("%HABITA%")), 'Residential')
          .when((upper(col("2_1_6")).like("%CULTUR%")) | (upper(col("2_1_6")).like("%SPORT%")) | (upper(col("2_1_6")).like("%COMMUNITY%")) | (upper(col("2_1_6")).like("%ENTRETEN%")) | (upper(col("2_1_6")).like("%ARTS%")) | (upper(col("2_1_6")).like("%COMUNIDAD%")) | (upper(col("2_1_6")).like("%RECREA%")), 'Society / community & culture')
          .when((upper(col("2_1_6")).like("%TOURISM%")) | (upper(col("2_1_6")).like("%TURISMO%")), 'Tourism')
          ##.when((upper(col("2_1_6")).like("%TRANSPORT%")) | (upper(col("2_1_6")).like("%PORT%")), 'Transportation')
          .when((upper(col("2_1_6")).like("%WATER SUPPLY%")) | (upper(col("2_1_6")).like("%AGUA%")), 'Water supply')
          .when((upper(col("2_1_6")).like("%WASTE%")), 'Waste management')
          .when(col("2_1_6").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c2_1_7", 
          when((upper(col("2_1_7")).like("%ALL%")) | (upper(col("2_1_7")).like("%TOD%")) | (upper(col("2_1_7")).like("%GERAL%")) | (upper(col("2_1_7")).like("%POBLACIÃ³N EN GENERAL%")), 'All population')
          .when((upper(col("2_1_7")).like("%CHILDREN%")) | (upper(col("2_1_7")).like("%MENORES%")) | (upper(col("2_1_7")).like("%NIÃ±OS%")), 'Children and youth')
          .when((upper(col("2_1_7")).like("%ELDERLY%")) | (upper(col("2_1_7")).like("%ADULTOS MAYORES%")) | (upper(col("2_1_7")).like("%TERCERA EDAD%")), 'Elderly')
          ##.when((upper(col("2_1_7")).like("%HOMELESS%")) | (upper(col("2_1_7")).like("%UNHOUSE%")) | (upper(col("2_1_7")).like("%CALLE%")), 'Homeless persons')
          .when((upper(col("2_1_7")).like("%BAJOS%")) | (upper(col("2_1_7")).like("%LOW-INCOME%")), 'Low-income households')
          .when((upper(col("2_1_7")).like("%MARGINALIZED%")) | (upper(col("2_1_7")).like("%INDIGENOUS%")), 'Marginalized groups')
          .when((upper(col("2_1_7")).like("%OUTDOOR%")) | (upper(col("2_1_7")).like("%WORKER%")) | (upper(col("2_1_7")).like("%TRABAJA%")) | (upper(col("2_1_7")).like("%FARMER%")) | (upper(col("2_1_7")).like("%AGRICU%")) | (upper(col("2_1_7")).like("%AGRO%")), 'Outdoor workers & Farmers')
          .when((upper(col("2_1_7")).like("%HOUSING%")) | (upper(col("2_1_7")).like("%VIVIENDA%")), 'Persons living in sub-standard housing')
          .when((upper(col("2_1_7")).like("%DISABILIT%")) | (upper(col("2_1_7")).like("%DISCAPACIDAD%")), 'Persons with disabilities')
          .when((upper(col("2_1_7")).like("%UNEMPLOYED%")) | (upper(col("2_1_7")).like("%DESEMPLEA%")), 'Unemployed persons')
          .when((upper(col("2_1_7")).like("%CHRONIC%")) | (upper(col("2_1_7")).like("%DISEASE%")) | (upper(col("2_1_7")).like("%ENFERMEDAD%")) | (upper(col("2_1_7")).like("%HEALTH%")), 'Vulnerable health groups')
          .when((upper(col("2_1_7")).like("%WOMEN%")), 'Women and girls')
          .when(col("2_1_7").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c2_2_2", 
          when((upper(col("2_2_2")).like("%SUPPORT%")) | (upper(col("2_2_2")).like("%ENHANCE%")), 'Supports')
          .when((upper(col("2_2_2")).like("%CHALLENGE%")), 'Challenges')
          .when(col("2_2_2").isNull(), None)
          .otherwise("Other"))\
#.withColumn("c6_0_1", 
          when((upper(col("6_0_1")).like("%ECONOM%")) | (upper(col("6_0_1")).like("%MARKET%")), 'Circular economy / Markets')
          .when((upper(col("6_0_1")).like("%AMBIEN%")) | (upper(col("6_0_1")).like("%CLIMATE CHANGE RESILIENCY%")) | (upper(col("6_0_1")).like("%BIODIV%")) | (upper(col("6_0_1")).like("%CLIM%")) | (upper(col("6_0_1")).like("%ENVIRON%")) | (upper(col("6_0_1")).like("%NATURAL%")) | (upper(col("6_0_1")).like("%CLIM%")), 'Climate change resiliency projects')
          .when((upper(col("6_0_1")).like("%CLEAN TECHNOLOGY%")), 'Clean technology businesses')
          .when((upper(col("6_0_1")).like("%ENERGY EFFICIENCY%")), 'Energy efficiency measures and technologies')
          .when((upper(col("6_0_1")).like("%ENERGY SECURITY%")), 'Energy security')
          .when((upper(col("6_0_1")).like("%AGRIC%")), 'Extended agricultural seasons')
          .when((upper(col("6_0_1")).like("%FLOOD%")) | (upper(col("6_0_1")).like("%INUND%")), 'Flood risk mitigation')
          .when((upper(col("6_0_1")).like("%FOOD%")), 'Food security and sustainable')
          .when((upper(col("6_0_1")).like("%INFRA%")), 'Infrastructure investment')
          .when((upper(col("6_0_1")).like("%PARTNERSHIP%")), 'Partnerships')
          .when((upper(col("6_0_1")).like("%CONSERVATION%")), 'Resource conservation and management')
          .when((upper(col("6_0_1")).like("%CONSTRUC%")), 'Sustainable construction/real estate sector')
          .when((upper(col("6_0_1")).like("%SUSTAINABLE TRANSP%")), 'Sustainable transport sector')
          .when((upper(col("6_0_1")).like("%TOURIS%")) | (upper(col("6_0_1")).like("%TURISM%")), 'Tourism and eco-tourism sector')
          .when((upper(col("6_0_1")).like("%WASTE MANAGEMENT%")) | (upper(col("6_0_1")).like("%WASTE%")) | (upper(col("6_0_1")).like("%RESIDUO%")), 'Waste management')
          .when((upper(col("6_0_1")).like("%WATER%")) | (upper(col("6_0_1")).like("%AGUA%")), 'Water management sector')
          .when((upper(col("6_0_1")).like("%FUND%")) | (upper(col("6_0_1")).like("%INVE%")), 'Funding opportunities')
          .when((upper(col("6_0_1")).like("%HEALTH%")) | (upper(col("6_0_1")).like("%SAFETY%")), 'Health and Safety')
          .when((upper(col("6_0_1")).like("%OPERA%")), 'Efficiency of operations')
          .when(col("6_0_1").isNull(), None)
          .otherwise("Other"))\
.withColumn("csector", 
          when((upper(col("csector_")).like("%AGRICUL%")) | (upper(col("csector_")).like("%FORESTRY%")), 'Agriculture and Forestry')
          .when((upper(col("csector_")).like("%INFRA%")) | (upper(col("csector_")).like("%BUILD%")), 'Building and Infrastructure')
          .when((upper(col("csector_")).like("%ENERG%")) | (upper(col("csector_")).like("%ELECT%")), 'Energy')
          .when((upper(col("csector_")).like("%INDUSTR%")) | (upper(col("csector_")).like("%MANUF%")), 'Industry')
          .when((upper(col("csector_")).like("%WATER%")) | (upper(col("csector_")).like("%AGUA%")), 'Water')
          .when((upper(col("csector_")).like("%WASTE%")) | (upper(col("csector_")).like("%RESID%"))| (upper(col("csector_")).like("%DESECH%")), 'Waste')
          .when((upper(col("csector_")).like("%RESIDENTIAL%")), 'Residential')
          .when((upper(col("csector_")).like("%TRANSPORT%")) | (upper(col("csector_")).like("%ROAD%"))| (upper(col("csector_")).like("%RAIL%")), 'Transportation')
          .when((upper(col("csector_")).like("%TARGET%")), 'Target')
          .when((upper(col("csector_")).like("%PLANNING%")), 'Spatial Planning')
          .when((upper(col("csector_")).like("%HEALTH%")), 'Public Health and Safety')
          .when((upper(col("csector_")).like("%SOCIAL%")), 'Social Services')
          .when((upper(col("csector_")).like("%ALL%")) | (upper(col("csector_")).like("%TOTAL%")), 'All')
          .when(col("csector_").isNull(), None)
          .otherwise("Other"))\
.withColumn("cboundary", 
          when((upper(col("cboundary_")).like("%LARGER%")), 'Larger')
          .when((upper(col("cboundary_")).like("%PARTIAL%")), 'Partial')
          .when((upper(col("cboundary_")).like("%SMALLER%")), 'Smaller')
          .when(col("cboundary_").isNull(), None)
          .otherwise("Same"))\
.withColumn("c5_0_base_year",
          when(length(col("5_0a_5")) != 0, col("5_0a_5"))
          .when(length(col("5_0c_5")) != 0, col("5_0c_5"))
          .when(length(col("5_0d_5")) != 0, col("5_0d_5"))
          .otherwise(None))\
.withColumn("c5_0_target_year",
          when(length(col('5_0a_9')) != 0, col('5_0a_9'))
          .when(length(col('5_0b_7')) != 0, col('5_0b_7'))
          .when(length(col('5_0c_11')) != 0, col('5_0c_11'))
          .when(length(col('5_0d_8')) != 0, col('5_0d_8'))
          .otherwise(None))\
.withColumn("c5_0_target_year_set",
          when(length(col('5_0a_6')) != 0, col('5_0a_6'))
          .when(length(col('5_0b_5')) != 0, col('5_0b_5'))
          .when(length(col('5_0c_6')) != 0, col('5_0c_6'))
          .when(length(col('5_0d_6')) != 0, col('5_0d_6'))
          .otherwise(None))#\


# COMMAND ----------

df_responses_piv_q2 = df_responses_piv_q2.withColumn('c4_4_0_1',when(length(col('4_4_0_1')) != 0, '1').otherwise('0'))\
.withColumn('c4_4_0_2',when(length(col('4_4_0_2')) != 0, '1').otherwise('0'))\
.withColumn('c4_4_0_3',when(length(col('4_4_0_3')) != 0, '1').otherwise('0'))\
.withColumn('c4_4_0_4',when(length(col('4_4_0_4')) != 0, '1').otherwise('0'))\
.withColumn('c4_4_0_5',when(length(col('4_4_0_5')) != 0, '1').otherwise('0'))\
.withColumn('c4_4_0_6',when(length(col('4_4_0_6')) != 0, '1').otherwise('0'))\
.withColumn('c4_4_0_7',when(length(col('4_4_0_7')) != 0, '1').otherwise('0'))\
.withColumn('c14_0_0_1',when(length(col('14_0_0_1')) != 0, '1').otherwise('0'))\
.withColumn('c14_0_0_2',when(length(col('14_0_0_2')) != 0, '1').otherwise('0'))\
.withColumn('c14_0_0_3',when(length(col('14_0_0_3')) != 0, '1').otherwise('0'))\
.withColumn('c14_0_0_4',when(length(col('14_0_0_4')) != 0, '1').otherwise('0'))\
.withColumn('c14_0_0_5',when(length(col('14_0_0_5')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_3_1',when(length(col('2_1_3_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_3_2',when(length(col('2_1_3_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_3_3',when(length(col('2_1_3_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_3_4',when(length(col('2_1_3_4')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_4_1',when(length(col('2_1_4_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_4_2',when(length(col('2_1_4_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_4_3',when(length(col('2_1_4_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_4_4',when(length(col('2_1_4_4')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_11_1',when(length(col('2_1_11_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_11_2',when(length(col('2_1_11_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_11_3',when(length(col('2_1_11_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_1',when(length(col('2_1_1_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_2',when(length(col('2_1_1_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_3',when(length(col('2_1_1_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_4',when(length(col('2_1_1_4')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_5',when(length(col('2_1_1_5')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_6',when(length(col('2_1_1_6')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_7',when(length(col('2_1_1_7')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_8',when(length(col('2_1_1_8')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_9',when(length(col('2_1_1_9')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_10',when(length(col('2_1_1_10')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_1_11',when(length(col('2_1_1_11')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_1',when(length(col('2_1_6_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_2',when(length(col('2_1_6_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_3',when(length(col('2_1_6_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_4',when(length(col('2_1_6_4')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_5',when(length(col('2_1_6_5')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_66',when(length(col('2_1_6_66')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_7',when(length(col('2_1_6_7')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_8',when(length(col('2_1_6_8')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_9',when(length(col('2_1_6_9')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_10',when(length(col('2_1_6_10')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_11',when(length(col('2_1_6_11')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_12',when(length(col('2_1_6_12')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_13',when(length(col('2_1_6_13')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_14',when(length(col('2_1_6_14')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_15',when(length(col('2_1_6_15')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_16',when(length(col('2_1_6_16')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_6_17',when(length(col('2_1_6_17')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_1',when(length(col('2_1_7_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_2',when(length(col('2_1_7_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_3',when(length(col('2_1_7_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_4',when(length(col('2_1_7_4')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_8',when(length(col('2_1_7_8')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_9',when(length(col('2_1_7_9')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_10',when(length(col('2_1_7_10')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_11',when(length(col('2_1_7_11')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_12',when(length(col('2_1_7_12')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_13',when(length(col('2_1_7_13')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_7_14',when(length(col('2_1_7_14')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_1',when(length(col('3_0_2_1')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_2',when(length(col('3_0_2_2')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_3',when(length(col('3_0_2_3')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_4',when(length(col('3_0_2_4')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_5',when(length(col('3_0_2_5')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_6',when(length(col('3_0_2_6')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_7',when(length(col('3_0_2_7')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_8',when(length(col('3_0_2_8')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_9',when(length(col('3_0_2_9')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_10',when(length(col('3_0_2_10')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_11',when(length(col('3_0_2_11')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_2_12',when(length(col('3_0_2_12')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_1',when(length(col('2_2_1_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_2',when(length(col('2_2_1_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_3',when(length(col('2_2_1_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_4',when(length(col('2_2_1_4')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_5',when(length(col('2_2_1_5')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_6',when(length(col('2_2_1_6')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_7',when(length(col('2_2_1_7')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_8',when(length(col('2_2_1_8')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_9',when(length(col('2_2_1_9')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_1_10',when(length(col('2_2_1_10')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_1',when(length(col('6_2a_1_1')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_2',when(length(col('6_2a_1_2')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_3',when(length(col('6_2a_1_3')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_4',when(length(col('6_2a_1_4')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_5',when(length(col('6_2a_1_5')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_6',when(length(col('6_2a_1_6')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_7',when(length(col('6_2a_1_7')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_8',when(length(col('6_2a_1_8')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_9',when(length(col('6_2a_1_9')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_10',when(length(col('6_2a_1_10')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_11',when(length(col('6_2a_1_11')) != 0, '1').otherwise('0'))\
.withColumn('c6_2a_1_12',when(length(col('6_2a_1_12')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_1',when(length(col('6_0_1_1')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_2',when(length(col('6_0_1_2')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_3',when(length(col('6_0_1_3')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_4',when(length(col('6_0_1_4')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_5',when(length(col('6_0_1_5')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_6',when(length(col('6_0_1_6')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_7',when(length(col('6_0_1_7')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_8',when(length(col('6_0_1_8')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_9',when(length(col('6_0_1_9')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_10',when(length(col('6_0_1_10')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_11',when(length(col('6_0_1_11')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_12',when(length(col('6_0_1_12')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_13',when(length(col('6_0_1_13')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_14',when(length(col('6_0_1_14')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_15',when(length(col('6_0_1_15')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_16',when(length(col('6_0_1_16')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_17',when(length(col('6_0_1_17')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_18',when(length(col('6_0_1_18')) != 0, '1').otherwise('0'))\
.withColumn('c6_0_1_19',when(length(col('6_0_1_19')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_2_1',when(length(col('2_2_2_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_2_2_2',when(length(col('2_2_2_2')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_4_1',when(length(col('3_0_4_1')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_4_2',when(length(col('3_0_4_2')) != 0, '1').otherwise('0'))\
.withColumn('c3_0_4_3',when(length(col('3_0_4_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_8_1',when(length(col('2_1_8_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_8_2',when(length(col('2_1_8_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_8_3',when(length(col('2_1_8_3')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_9_1',when(length(col('2_1_9_1')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_9_2',when(length(col('2_1_9_2')) != 0, '1').otherwise('0'))\
.withColumn('c2_1_9_3',when(length(col('2_1_9_3')) != 0, '1').otherwise('0'))



# COMMAND ----------

# Remove original replaced columns
columns_category_replaced = ["Response Answer", "2_0_0", "3_2_0", "4_0_0", "4_9_1_1", "5_5_0", "7_0_0", "7_7_0", "8_0_0", "2_0b_2","2_0b_7",'6_2_0','5_0_0','2_1_3','2_1_4','2_1_11','14_0_0','0_1_1_1', '6_5_3', '6_5_4', '3_2a_9', "6_0_1", "csector_", "cboundary_", '4_4_0_1', 	'4_4_0_2', 	'4_4_0_3', 	'4_4_0_4', 	'4_4_0_5', 	'4_4_0_6', 	'4_4_0_7', 	'14_0_0_1', 	'14_0_0_2', 	'14_0_0_3', 	'14_0_0_4', 	'14_0_0_5', 	'2_1_3_1', 	'2_1_3_2', 	'2_1_3_3', 	'2_1_3_4', 	'2_1_4_1', 	'2_1_4_2', 	'2_1_4_3', 	'2_1_4_4', 	'2_1_11_1', 	'2_1_11_2', 	'2_1_11_3', 	'2_1_11_3', 	'2_1_1_1', 	'2_1_1_2', 	'2_1_1_3', 	'2_1_1_4', 	'2_1_1_5', 	'2_1_1_6', 	'2_1_1_7', 	'2_1_1_8', 	'2_1_1_9', 	'2_1_1_10', 	'2_1_1_11', 	'2_1_6_1', 	'2_1_6_2', 	'2_1_6_3', 	'2_1_6_4', 	'2_1_6_5', 	'2_1_6_66', 	'2_1_6_7', 	'2_1_6_8', 	'2_1_6_9', 	'2_1_6_10', 	'2_1_6_11', 	'2_1_6_12', 	'2_1_6_13', 	'2_1_6_14', 	'2_1_6_15', 	'2_1_6_16', 	'2_1_6_17', 	'2_1_7_1', 	'2_1_7_2', 	'2_1_7_3', 	'2_1_7_4', 	'2_1_7_8', 	'2_1_7_9', 	'2_1_7_10', 	'2_1_7_11', 	'2_1_7_12', 	'2_1_7_13', 	'2_1_7_14', 	'3_0_2_1', 	'3_0_2_2', 	'3_0_2_3', 	'3_0_2_4', 	'3_0_2_5', 	'3_0_2_6', 	'3_0_2_7', 	'3_0_2_8', 	'3_0_2_9', 	'3_0_2_10', 	'3_0_2_11', 	'3_0_2_12', 	'2_2_1_1', 	'2_2_1_2', 	'2_2_1_3', 	'2_2_1_4', 	'2_2_1_5', 	'2_2_1_6', 	'2_2_1_7', 	'2_2_1_8', 	'2_2_1_9', 	'2_2_1_10', 	'6_2a_1_1', 	'6_2a_1_2', 	'6_2a_1_3', 	'6_2a_1_4', 	'6_2a_1_5', 	'6_2a_1_6', 	'6_2a_1_7', 	'6_2a_1_8', 	'6_2a_1_9', 	'6_2a_1_10', 	'6_2a_1_11', 	'6_2a_1_12', 	'6_0_1_1', 	'6_0_1_2', 	'6_0_1_3', 	'6_0_1_4', 	'6_0_1_5', 	'6_0_1_6', 	'6_0_1_7', 	'6_0_1_8', 	'6_0_1_9', 	'6_0_1_10', 	'6_0_1_11', 	'6_0_1_12', 	'6_0_1_13', 	'6_0_1_14', 	'6_0_1_15', 	'6_0_1_16', 	'6_0_1_17', 	'6_0_1_18', 	'6_0_1_19', 	'2_2_2_1', 	'2_2_2_2', 	'3_0_4_1', 	'3_0_4_2', 	'3_0_4_3', 	'2_1_8_1', 	'2_1_8_2', 	'2_1_8_3', 	'2_1_9_1', 	'2_1_9_2', 	'2_1_9_3']

df_responses_piv_q2 = df_responses_piv_q2.drop(*columns_category_replaced)

# COMMAND ----------

# Original Shape
def sparkShape(df):
    return (df.count(), len(df.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape

print(df_responses_piv_q2.shape())

# COMMAND ----------

from pyspark.sql.types import DecimalType

df_responses_piv_q2 = df_responses_piv_q2.withColumn('c2_0_0', df_responses_piv_q2['c2_0_0'].cast(DecimalType()))\
.withColumn('c3_2_0', df_responses_piv_q2['c3_2_0'].cast(DecimalType()))\
.withColumn('c2_0b_7', df_responses_piv_q2['c2_0b_7'].cast(DecimalType()))\
.withColumn('c4_0_0', df_responses_piv_q2['c4_0_0'].cast(DecimalType()))\
.withColumn('c4_9_1_1', df_responses_piv_q2['c4_9_1_1'].cast(DecimalType()))\
.withColumn('c5_5_0', df_responses_piv_q2['c5_5_0'].cast(DecimalType()))\
.withColumn('c5_0_0', df_responses_piv_q2['c5_0_0'].cast(DecimalType()))\
.withColumn('c6_2_0', df_responses_piv_q2['c6_2_0'].cast(DecimalType()))\
.withColumn('c7_0_0', df_responses_piv_q2['c7_0_0'].cast(DecimalType()))\
.withColumn('c7_7_0', df_responses_piv_q2['c7_7_0'].cast(DecimalType()))\
.withColumn('c8_0_0', df_responses_piv_q2['c8_0_0'].cast(DecimalType()))\
.withColumn('10_1_1_1', df_responses_piv_q2['10_1_1_1'].cast(DecimalType()))\
.withColumn('10_1_2_1', df_responses_piv_q2['10_1_2_1'].cast(DecimalType()))\
.withColumn('10_1_3_1', df_responses_piv_q2['10_1_3_1'].cast(DecimalType()))\
.withColumn('10_1_4_1', df_responses_piv_q2['10_1_4_1'].cast(DecimalType()))\
.withColumn('10_1_5_1', df_responses_piv_q2['10_1_5_1'].cast(DecimalType()))\
.withColumn('10_1_6_1', df_responses_piv_q2['10_1_6_1'].cast(DecimalType()))\
.withColumn('10_1_7_1', df_responses_piv_q2['10_1_7_1'].cast(DecimalType()))\
.withColumn('10_1_9_1', df_responses_piv_q2['10_1_9_1'].cast(DecimalType()))\
.withColumn('10_3_1_1', df_responses_piv_q2['10_3_1_1'].cast(DecimalType()))\
.withColumn('10_3_1_2', df_responses_piv_q2['10_3_1_2'].cast(DecimalType()))\
.withColumn('10_3_1_3', df_responses_piv_q2['10_3_1_3'].cast(DecimalType()))\
.withColumn('10_3_1_4', df_responses_piv_q2['10_3_1_4'].cast(DecimalType()))\
.withColumn('10_3_1_5', df_responses_piv_q2['10_3_1_5'].cast(DecimalType()))\
.withColumn('10_3_2_1', df_responses_piv_q2['10_3_2_1'].cast(DecimalType()))\
.withColumn('10_3_2_2', df_responses_piv_q2['10_3_2_2'].cast(DecimalType()))\
.withColumn('10_3_2_3', df_responses_piv_q2['10_3_2_3'].cast(DecimalType()))\
.withColumn('10_3_2_4', df_responses_piv_q2['10_3_2_4'].cast(DecimalType()))\
.withColumn('10_3_2_5', df_responses_piv_q2['10_3_2_5'].cast(DecimalType()))\
.withColumn('10_3_3_1', df_responses_piv_q2['10_3_3_1'].cast(DecimalType()))\
.withColumn('10_3_3_2', df_responses_piv_q2['10_3_3_2'].cast(DecimalType()))\
.withColumn('10_3_3_3', df_responses_piv_q2['10_3_3_3'].cast(DecimalType()))\
.withColumn('10_3_3_4', df_responses_piv_q2['10_3_3_4'].cast(DecimalType()))\
.withColumn('10_3_3_5', df_responses_piv_q2['10_3_3_5'].cast(DecimalType()))\
.withColumn('10_3_4_1', df_responses_piv_q2['10_3_4_1'].cast(DecimalType()))\
.withColumn('10_3_4_2', df_responses_piv_q2['10_3_4_2'].cast(DecimalType()))\
.withColumn('10_3_4_3', df_responses_piv_q2['10_3_4_3'].cast(DecimalType()))\
.withColumn('10_3_4_4', df_responses_piv_q2['10_3_4_4'].cast(DecimalType()))\
.withColumn('10_3_4_5', df_responses_piv_q2['10_3_4_5'].cast(DecimalType()))\
.withColumn('10_3_5_1', df_responses_piv_q2['10_3_5_1'].cast(DecimalType()))\
.withColumn('10_3_5_2', df_responses_piv_q2['10_3_5_2'].cast(DecimalType()))\
.withColumn('10_3_5_3', df_responses_piv_q2['10_3_5_3'].cast(DecimalType()))\
.withColumn('10_3_5_4', df_responses_piv_q2['10_3_5_4'].cast(DecimalType()))\
.withColumn('10_3_5_5', df_responses_piv_q2['10_3_5_5'].cast(DecimalType()))\
.withColumn('13_3_1_1', df_responses_piv_q2['13_3_1_1'].cast(DecimalType()))\
.withColumn('13_3_1_2', df_responses_piv_q2['13_3_1_2'].cast(DecimalType()))\
.withColumn('13_3_1_3', df_responses_piv_q2['13_3_1_3'].cast(DecimalType()))\
.withColumn('13_3_1_4', df_responses_piv_q2['13_3_1_4'].cast(DecimalType()))\
.withColumn('13_3_1_5', df_responses_piv_q2['13_3_1_5'].cast(DecimalType()))\
.withColumn('13_3_1_6', df_responses_piv_q2['13_3_1_6'].cast(DecimalType()))\
.withColumn('4_6a_1_1', df_responses_piv_q2['4_6a_1_1'].cast(DecimalType()))\
.withColumn('4_6a_1_10', df_responses_piv_q2['4_6a_1_10'].cast(DecimalType()))\
.withColumn('4_6a_1_11', df_responses_piv_q2['4_6a_1_11'].cast(DecimalType()))\
.withColumn('4_6a_1_12', df_responses_piv_q2['4_6a_1_12'].cast(DecimalType()))\
.withColumn('4_6a_1_13', df_responses_piv_q2['4_6a_1_13'].cast(DecimalType()))\
.withColumn('4_6a_1_14', df_responses_piv_q2['4_6a_1_14'].cast(DecimalType()))\
.withColumn('4_6a_1_15', df_responses_piv_q2['4_6a_1_15'].cast(DecimalType()))\
.withColumn('4_6a_1_16', df_responses_piv_q2['4_6a_1_16'].cast(DecimalType()))\
.withColumn('4_6a_1_17', df_responses_piv_q2['4_6a_1_17'].cast(DecimalType()))\
.withColumn('4_6a_1_18', df_responses_piv_q2['4_6a_1_18'].cast(DecimalType()))\
.withColumn('4_6a_1_19', df_responses_piv_q2['4_6a_1_19'].cast(DecimalType()))\
.withColumn('4_6a_1_2', df_responses_piv_q2['4_6a_1_2'].cast(DecimalType()))\
.withColumn('4_6a_1_20', df_responses_piv_q2['4_6a_1_20'].cast(DecimalType()))\
.withColumn('4_6a_1_22', df_responses_piv_q2['4_6a_1_22'].cast(DecimalType()))\
.withColumn('4_6a_1_23', df_responses_piv_q2['4_6a_1_23'].cast(DecimalType()))\
.withColumn('4_6a_1_25', df_responses_piv_q2['4_6a_1_25'].cast(DecimalType()))\
.withColumn('4_6a_1_26', df_responses_piv_q2['4_6a_1_26'].cast(DecimalType()))\
.withColumn('4_6a_1_27', df_responses_piv_q2['4_6a_1_27'].cast(DecimalType()))\
.withColumn('4_6a_1_28', df_responses_piv_q2['4_6a_1_28'].cast(DecimalType()))\
.withColumn('4_6a_1_29', df_responses_piv_q2['4_6a_1_29'].cast(DecimalType()))\
.withColumn('4_6a_1_3', df_responses_piv_q2['4_6a_1_3'].cast(DecimalType()))\
.withColumn('4_6a_1_30', df_responses_piv_q2['4_6a_1_30'].cast(DecimalType()))\
.withColumn('4_6a_1_31', df_responses_piv_q2['4_6a_1_31'].cast(DecimalType()))\
.withColumn('4_6a_1_4', df_responses_piv_q2['4_6a_1_4'].cast(DecimalType()))\
.withColumn('4_6a_1_5', df_responses_piv_q2['4_6a_1_5'].cast(DecimalType()))\
.withColumn('4_6a_1_6', df_responses_piv_q2['4_6a_1_6'].cast(DecimalType()))\
.withColumn('4_6a_1_7', df_responses_piv_q2['4_6a_1_7'].cast(DecimalType()))\
.withColumn('4_6a_1_8', df_responses_piv_q2['4_6a_1_8'].cast(DecimalType()))\
.withColumn('4_6a_1_9', df_responses_piv_q2['4_6a_1_9'].cast(DecimalType()))\
.withColumn('4_6a_3_1', df_responses_piv_q2['4_6a_3_1'].cast(DecimalType()))\
.withColumn('4_6a_3_10', df_responses_piv_q2['4_6a_3_10'].cast(DecimalType()))\
.withColumn('4_6a_3_11', df_responses_piv_q2['4_6a_3_11'].cast(DecimalType()))\
.withColumn('4_6a_3_12', df_responses_piv_q2['4_6a_3_12'].cast(DecimalType()))\
.withColumn('4_6a_3_13', df_responses_piv_q2['4_6a_3_13'].cast(DecimalType()))\
.withColumn('4_6a_3_14', df_responses_piv_q2['4_6a_3_14'].cast(DecimalType()))\
.withColumn('4_6a_3_15', df_responses_piv_q2['4_6a_3_15'].cast(DecimalType()))\
.withColumn('4_6a_3_16', df_responses_piv_q2['4_6a_3_16'].cast(DecimalType()))\
.withColumn('4_6a_3_17', df_responses_piv_q2['4_6a_3_17'].cast(DecimalType()))\
.withColumn('4_6a_3_18', df_responses_piv_q2['4_6a_3_18'].cast(DecimalType()))\
.withColumn('4_6a_3_19', df_responses_piv_q2['4_6a_3_19'].cast(DecimalType()))\
.withColumn('4_6a_3_2', df_responses_piv_q2['4_6a_3_2'].cast(DecimalType()))\
.withColumn('4_6a_3_20', df_responses_piv_q2['4_6a_3_20'].cast(DecimalType()))\
.withColumn('4_6a_3_22', df_responses_piv_q2['4_6a_3_22'].cast(DecimalType()))\
.withColumn('4_6a_3_23', df_responses_piv_q2['4_6a_3_23'].cast(DecimalType()))\
.withColumn('4_6a_3_25', df_responses_piv_q2['4_6a_3_25'].cast(DecimalType()))\
.withColumn('4_6a_3_26', df_responses_piv_q2['4_6a_3_26'].cast(DecimalType()))\
.withColumn('4_6a_3_27', df_responses_piv_q2['4_6a_3_27'].cast(DecimalType()))\
.withColumn('4_6a_3_28', df_responses_piv_q2['4_6a_3_28'].cast(DecimalType()))\
.withColumn('4_6a_3_29', df_responses_piv_q2['4_6a_3_29'].cast(DecimalType()))\
.withColumn('4_6a_3_3', df_responses_piv_q2['4_6a_3_3'].cast(DecimalType()))\
.withColumn('4_6a_3_30', df_responses_piv_q2['4_6a_3_30'].cast(DecimalType()))\
.withColumn('4_6a_3_31', df_responses_piv_q2['4_6a_3_31'].cast(DecimalType()))\
.withColumn('4_6a_3_4', df_responses_piv_q2['4_6a_3_4'].cast(DecimalType()))\
.withColumn('4_6a_3_5', df_responses_piv_q2['4_6a_3_5'].cast(DecimalType()))\
.withColumn('4_6a_3_6', df_responses_piv_q2['4_6a_3_6'].cast(DecimalType()))\
.withColumn('4_6a_3_7', df_responses_piv_q2['4_6a_3_7'].cast(DecimalType()))\
.withColumn('4_6a_3_8', df_responses_piv_q2['4_6a_3_8'].cast(DecimalType()))\
.withColumn('4_6a_3_9', df_responses_piv_q2['4_6a_3_9'].cast(DecimalType()))\
.withColumn('4_6a_5_1', df_responses_piv_q2['4_6a_5_1'].cast(DecimalType()))\
.withColumn('4_6a_5_10', df_responses_piv_q2['4_6a_5_10'].cast(DecimalType()))\
.withColumn('4_6a_5_11', df_responses_piv_q2['4_6a_5_11'].cast(DecimalType()))\
.withColumn('4_6a_5_12', df_responses_piv_q2['4_6a_5_12'].cast(DecimalType()))\
.withColumn('4_6a_5_13', df_responses_piv_q2['4_6a_5_13'].cast(DecimalType()))\
.withColumn('4_6a_5_14', df_responses_piv_q2['4_6a_5_14'].cast(DecimalType()))\
.withColumn('4_6a_5_15', df_responses_piv_q2['4_6a_5_15'].cast(DecimalType()))\
.withColumn('4_6a_5_16', df_responses_piv_q2['4_6a_5_16'].cast(DecimalType()))\
.withColumn('4_6a_5_17', df_responses_piv_q2['4_6a_5_17'].cast(DecimalType()))\
.withColumn('4_6a_5_18', df_responses_piv_q2['4_6a_5_18'].cast(DecimalType()))\
.withColumn('4_6a_5_19', df_responses_piv_q2['4_6a_5_19'].cast(DecimalType()))\
.withColumn('4_6a_5_2', df_responses_piv_q2['4_6a_5_2'].cast(DecimalType()))\
.withColumn('4_6a_5_20', df_responses_piv_q2['4_6a_5_20'].cast(DecimalType()))\
.withColumn('4_6a_5_22', df_responses_piv_q2['4_6a_5_22'].cast(DecimalType()))\
.withColumn('4_6a_5_23', df_responses_piv_q2['4_6a_5_23'].cast(DecimalType()))\
.withColumn('4_6a_5_25', df_responses_piv_q2['4_6a_5_25'].cast(DecimalType()))\
.withColumn('4_6a_5_26', df_responses_piv_q2['4_6a_5_26'].cast(DecimalType()))\
.withColumn('4_6a_5_27', df_responses_piv_q2['4_6a_5_27'].cast(DecimalType()))\
.withColumn('4_6a_5_28', df_responses_piv_q2['4_6a_5_28'].cast(DecimalType()))\
.withColumn('4_6a_5_29', df_responses_piv_q2['4_6a_5_29'].cast(DecimalType()))\
.withColumn('4_6a_5_3', df_responses_piv_q2['4_6a_5_3'].cast(DecimalType()))\
.withColumn('4_6a_5_30', df_responses_piv_q2['4_6a_5_30'].cast(DecimalType()))\
.withColumn('4_6a_5_31', df_responses_piv_q2['4_6a_5_31'].cast(DecimalType()))\
.withColumn('4_6a_5_4', df_responses_piv_q2['4_6a_5_4'].cast(DecimalType()))\
.withColumn('4_6a_5_5', df_responses_piv_q2['4_6a_5_5'].cast(DecimalType()))\
.withColumn('4_6a_5_6', df_responses_piv_q2['4_6a_5_6'].cast(DecimalType()))\
.withColumn('4_6a_5_7', df_responses_piv_q2['4_6a_5_7'].cast(DecimalType()))\
.withColumn('4_6a_5_8', df_responses_piv_q2['4_6a_5_8'].cast(DecimalType()))\
.withColumn('4_6a_5_9', df_responses_piv_q2['4_6a_5_9'].cast(DecimalType()))\
.withColumn("4_6b_1_1", df_responses_piv_q2["4_6b_1_1"].cast(DecimalType()))\
.withColumn("4_6b_1_2", df_responses_piv_q2["4_6b_1_2"].cast(DecimalType()))\
.withColumn("4_6b_1_3", df_responses_piv_q2["4_6b_1_3"].cast(DecimalType()))\
.withColumn("4_6b_1_5", df_responses_piv_q2["4_6b_1_5"].cast(DecimalType()))\
.withColumn("4_6b_1_6", df_responses_piv_q2["4_6b_1_6"].cast(DecimalType()))\
.withColumn("4_6b_1_7", df_responses_piv_q2["4_6b_1_7"].cast(DecimalType()))\
.withColumn("4_6b_1_8", df_responses_piv_q2["4_6b_1_8"].cast(DecimalType()))\
.withColumn("4_6b_1_9", df_responses_piv_q2["4_6b_1_9"].cast(DecimalType()))\
.withColumn("4_6b_1_10", df_responses_piv_q2["4_6b_1_10"].cast(DecimalType()))\
.withColumn("4_6b_1_13", df_responses_piv_q2["4_6b_1_13"].cast(DecimalType()))\
.withColumn("4_6b_1_14", df_responses_piv_q2["4_6b_1_14"].cast(DecimalType()))\
.withColumn("4_6b_1_15", df_responses_piv_q2["4_6b_1_15"].cast(DecimalType()))\
.withColumn('8_1_1_1', df_responses_piv_q2['8_1_1_1'].cast(DecimalType()))\
.withColumn('8_1_11_1', df_responses_piv_q2['8_1_11_1'].cast(DecimalType()))\
.withColumn('8_1_2_1', df_responses_piv_q2['8_1_2_1'].cast(DecimalType()))\
.withColumn('8_1_3_1', df_responses_piv_q2['8_1_3_1'].cast(DecimalType()))\
.withColumn('8_1_4_1', df_responses_piv_q2['8_1_4_1'].cast(DecimalType()))\
.withColumn('8_1_5_1', df_responses_piv_q2['8_1_5_1'].cast(DecimalType()))\
.withColumn('8_1_6_1', df_responses_piv_q2['8_1_6_1'].cast(DecimalType()))\
.withColumn('8_1_7_1', df_responses_piv_q2['8_1_7_1'].cast(DecimalType()))\
.withColumn('8_1_8_1', df_responses_piv_q2['8_1_8_1'].cast(DecimalType()))\
.withColumn('8_1_9_1', df_responses_piv_q2['8_1_9_1'].cast(DecimalType()))\
.withColumn('8_2_1_3', df_responses_piv_q2['8_2_1_3'].cast(DecimalType()))\
.withColumn('5_0a_11', df_responses_piv_q2['5_0a_11'].cast(DecimalType()))\
.withColumn('5_0a_7', df_responses_piv_q2['5_0a_7'].cast(DecimalType()))\
.withColumn('5_0a_8', df_responses_piv_q2['5_0a_8'].cast(DecimalType()))\
.withColumn('5_0b_10', df_responses_piv_q2['5_0b_10'].cast(DecimalType()))\
.withColumn('5_0b_8', df_responses_piv_q2['5_0b_8'].cast(DecimalType()))\
.withColumn('5_0b_9', df_responses_piv_q2['5_0b_9'].cast(DecimalType()))\
.withColumn('5_0c_10', df_responses_piv_q2['5_0c_10'].cast(DecimalType()))\
.withColumn('5_0c_12', df_responses_piv_q2['5_0c_12'].cast(DecimalType()))\
.withColumn('5_0c_13', df_responses_piv_q2['5_0c_13'].cast(DecimalType()))\
.withColumn('5_0c_8', df_responses_piv_q2['5_0c_8'].cast(DecimalType()))\
.withColumn('5_0c_9', df_responses_piv_q2['5_0c_9'].cast(DecimalType()))\
.withColumn('5_0d_10', df_responses_piv_q2['5_0d_10'].cast(DecimalType()))\
.withColumn('5_0d_11', df_responses_piv_q2['5_0d_11'].cast(DecimalType()))\
.withColumn('5_0d_7', df_responses_piv_q2['5_0d_7'].cast(DecimalType()))\
.withColumn('5_0d_9', df_responses_piv_q2['5_0d_9'].cast(DecimalType()))\
.withColumn('c4_4_0_1', df_responses_piv_q2['c4_4_0_1'].cast(IntegerType()))\
.withColumn('c4_4_0_2', df_responses_piv_q2['c4_4_0_2'].cast(IntegerType()))\
.withColumn('c4_4_0_3', df_responses_piv_q2['c4_4_0_3'].cast(IntegerType()))\
.withColumn('c4_4_0_4', df_responses_piv_q2['c4_4_0_4'].cast(IntegerType()))\
.withColumn('c4_4_0_5', df_responses_piv_q2['c4_4_0_5'].cast(IntegerType()))\
.withColumn('c4_4_0_6', df_responses_piv_q2['c4_4_0_6'].cast(IntegerType()))\
.withColumn('c4_4_0_7', df_responses_piv_q2['c4_4_0_7'].cast(IntegerType()))\
.withColumn('c14_0_0_1', df_responses_piv_q2['c14_0_0_1'].cast(IntegerType()))\
.withColumn('c14_0_0_2', df_responses_piv_q2['c14_0_0_2'].cast(IntegerType()))\
.withColumn('c14_0_0_3', df_responses_piv_q2['c14_0_0_3'].cast(IntegerType()))\
.withColumn('c14_0_0_4', df_responses_piv_q2['c14_0_0_4'].cast(IntegerType()))\
.withColumn('c14_0_0_5', df_responses_piv_q2['c14_0_0_5'].cast(IntegerType()))\
.withColumn('c2_1_3_1', df_responses_piv_q2['c2_1_3_1'].cast(IntegerType()))\
.withColumn('c2_1_3_2', df_responses_piv_q2['c2_1_3_2'].cast(IntegerType()))\
.withColumn('c2_1_3_3', df_responses_piv_q2['c2_1_3_3'].cast(IntegerType()))\
.withColumn('c2_1_3_4', df_responses_piv_q2['c2_1_3_4'].cast(IntegerType()))\
.withColumn('c2_1_4_1', df_responses_piv_q2['c2_1_4_1'].cast(IntegerType()))\
.withColumn('c2_1_4_2', df_responses_piv_q2['c2_1_4_2'].cast(IntegerType()))\
.withColumn('c2_1_4_3', df_responses_piv_q2['c2_1_4_3'].cast(IntegerType()))\
.withColumn('c2_1_4_4', df_responses_piv_q2['c2_1_4_4'].cast(IntegerType()))\
.withColumn('c2_1_11_1', df_responses_piv_q2['c2_1_11_1'].cast(IntegerType()))\
.withColumn('c2_1_11_2', df_responses_piv_q2['c2_1_11_2'].cast(IntegerType()))\
.withColumn('c2_1_11_3', df_responses_piv_q2['c2_1_11_3'].cast(IntegerType()))\
.withColumn('c2_1_1_1', df_responses_piv_q2['c2_1_1_1'].cast(IntegerType()))\
.withColumn('c2_1_1_2', df_responses_piv_q2['c2_1_1_2'].cast(IntegerType()))\
.withColumn('c2_1_1_3', df_responses_piv_q2['c2_1_1_3'].cast(IntegerType()))\
.withColumn('c2_1_1_4', df_responses_piv_q2['c2_1_1_4'].cast(IntegerType()))\
.withColumn('c2_1_1_5', df_responses_piv_q2['c2_1_1_5'].cast(IntegerType()))\
.withColumn('c2_1_1_6', df_responses_piv_q2['c2_1_1_6'].cast(IntegerType()))\
.withColumn('c2_1_1_7', df_responses_piv_q2['c2_1_1_7'].cast(IntegerType()))\
.withColumn('c2_1_1_8', df_responses_piv_q2['c2_1_1_8'].cast(IntegerType()))\
.withColumn('c2_1_1_9', df_responses_piv_q2['c2_1_1_9'].cast(IntegerType()))\
.withColumn('c2_1_1_10', df_responses_piv_q2['c2_1_1_10'].cast(IntegerType()))\
.withColumn('c2_1_1_11', df_responses_piv_q2['c2_1_1_11'].cast(IntegerType()))\
.withColumn('c2_1_6_1', df_responses_piv_q2['c2_1_6_1'].cast(IntegerType()))\
.withColumn('c2_1_6_2', df_responses_piv_q2['c2_1_6_2'].cast(IntegerType()))\
.withColumn('c2_1_6_3', df_responses_piv_q2['c2_1_6_3'].cast(IntegerType()))\
.withColumn('c2_1_6_4', df_responses_piv_q2['c2_1_6_4'].cast(IntegerType()))\
.withColumn('c2_1_6_5', df_responses_piv_q2['c2_1_6_5'].cast(IntegerType()))\
.withColumn('c2_1_6_66', df_responses_piv_q2['c2_1_6_66'].cast(IntegerType()))\
.withColumn('c2_1_6_7', df_responses_piv_q2['c2_1_6_7'].cast(IntegerType()))\
.withColumn('c2_1_6_8', df_responses_piv_q2['c2_1_6_8'].cast(IntegerType()))\
.withColumn('c2_1_6_9', df_responses_piv_q2['c2_1_6_9'].cast(IntegerType()))\
.withColumn('c2_1_6_10', df_responses_piv_q2['c2_1_6_10'].cast(IntegerType()))\
.withColumn('c2_1_6_11', df_responses_piv_q2['c2_1_6_11'].cast(IntegerType()))\
.withColumn('c2_1_6_12', df_responses_piv_q2['c2_1_6_12'].cast(IntegerType()))\
.withColumn('c2_1_6_13', df_responses_piv_q2['c2_1_6_13'].cast(IntegerType()))\
.withColumn('c2_1_6_14', df_responses_piv_q2['c2_1_6_14'].cast(IntegerType()))\
.withColumn('c2_1_6_15', df_responses_piv_q2['c2_1_6_15'].cast(IntegerType()))\
.withColumn('c2_1_6_16', df_responses_piv_q2['c2_1_6_16'].cast(IntegerType()))\
.withColumn('c2_1_6_17', df_responses_piv_q2['c2_1_6_17'].cast(IntegerType()))\
.withColumn('c2_1_7_1', df_responses_piv_q2['c2_1_7_1'].cast(IntegerType()))\
.withColumn('c2_1_7_2', df_responses_piv_q2['c2_1_7_2'].cast(IntegerType()))\
.withColumn('c2_1_7_3', df_responses_piv_q2['c2_1_7_3'].cast(IntegerType()))\
.withColumn('c2_1_7_4', df_responses_piv_q2['c2_1_7_4'].cast(IntegerType()))\
.withColumn('c2_1_7_8', df_responses_piv_q2['c2_1_7_8'].cast(IntegerType()))\
.withColumn('c2_1_7_9', df_responses_piv_q2['c2_1_7_9'].cast(IntegerType()))\
.withColumn('c2_1_7_10', df_responses_piv_q2['c2_1_7_10'].cast(IntegerType()))\
.withColumn('c2_1_7_11', df_responses_piv_q2['c2_1_7_11'].cast(IntegerType()))\
.withColumn('c2_1_7_12', df_responses_piv_q2['c2_1_7_12'].cast(IntegerType()))\
.withColumn('c2_1_7_13', df_responses_piv_q2['c2_1_7_13'].cast(IntegerType()))\
.withColumn('c2_1_7_14', df_responses_piv_q2['c2_1_7_14'].cast(IntegerType()))\
.withColumn('c3_0_2_1', df_responses_piv_q2['c3_0_2_1'].cast(IntegerType()))\
.withColumn('c3_0_2_2', df_responses_piv_q2['c3_0_2_2'].cast(IntegerType()))\
.withColumn('c3_0_2_3', df_responses_piv_q2['c3_0_2_3'].cast(IntegerType()))\
.withColumn('c3_0_2_4', df_responses_piv_q2['c3_0_2_4'].cast(IntegerType()))\
.withColumn('c3_0_2_5', df_responses_piv_q2['c3_0_2_5'].cast(IntegerType()))\
.withColumn('c3_0_2_6', df_responses_piv_q2['c3_0_2_6'].cast(IntegerType()))\
.withColumn('c3_0_2_7', df_responses_piv_q2['c3_0_2_7'].cast(IntegerType()))\
.withColumn('c3_0_2_8', df_responses_piv_q2['c3_0_2_8'].cast(IntegerType()))\
.withColumn('c3_0_2_9', df_responses_piv_q2['c3_0_2_9'].cast(IntegerType()))\
.withColumn('c3_0_2_10', df_responses_piv_q2['c3_0_2_10'].cast(IntegerType()))\
.withColumn('c3_0_2_11', df_responses_piv_q2['c3_0_2_11'].cast(IntegerType()))\
.withColumn('c3_0_2_12', df_responses_piv_q2['c3_0_2_12'].cast(IntegerType()))\
.withColumn('c2_2_1_1', df_responses_piv_q2['c2_2_1_1'].cast(IntegerType()))\
.withColumn('c2_2_1_2', df_responses_piv_q2['c2_2_1_2'].cast(IntegerType()))\
.withColumn('c2_2_1_3', df_responses_piv_q2['c2_2_1_3'].cast(IntegerType()))\
.withColumn('c2_2_1_4', df_responses_piv_q2['c2_2_1_4'].cast(IntegerType()))\
.withColumn('c2_2_1_5', df_responses_piv_q2['c2_2_1_5'].cast(IntegerType()))\
.withColumn('c2_2_1_6', df_responses_piv_q2['c2_2_1_6'].cast(IntegerType()))\
.withColumn('c2_2_1_7', df_responses_piv_q2['c2_2_1_7'].cast(IntegerType()))\
.withColumn('c2_2_1_8', df_responses_piv_q2['c2_2_1_8'].cast(IntegerType()))\
.withColumn('c2_2_1_9', df_responses_piv_q2['c2_2_1_9'].cast(IntegerType()))\
.withColumn('c2_2_1_10', df_responses_piv_q2['c2_2_1_10'].cast(IntegerType()))\
.withColumn('c6_2a_1_1', df_responses_piv_q2['c6_2a_1_1'].cast(IntegerType()))\
.withColumn('c6_2a_1_2', df_responses_piv_q2['c6_2a_1_2'].cast(IntegerType()))\
.withColumn('c6_2a_1_3', df_responses_piv_q2['c6_2a_1_3'].cast(IntegerType()))\
.withColumn('c6_2a_1_4', df_responses_piv_q2['c6_2a_1_4'].cast(IntegerType()))\
.withColumn('c6_2a_1_5', df_responses_piv_q2['c6_2a_1_5'].cast(IntegerType()))\
.withColumn('c6_2a_1_6', df_responses_piv_q2['c6_2a_1_6'].cast(IntegerType()))\
.withColumn('c6_2a_1_7', df_responses_piv_q2['c6_2a_1_7'].cast(IntegerType()))\
.withColumn('c6_2a_1_8', df_responses_piv_q2['c6_2a_1_8'].cast(IntegerType()))\
.withColumn('c6_2a_1_9', df_responses_piv_q2['c6_2a_1_9'].cast(IntegerType()))\
.withColumn('c6_2a_1_10', df_responses_piv_q2['c6_2a_1_10'].cast(IntegerType()))\
.withColumn('c6_2a_1_11', df_responses_piv_q2['c6_2a_1_11'].cast(IntegerType()))\
.withColumn('c6_2a_1_12', df_responses_piv_q2['c6_2a_1_12'].cast(IntegerType()))\
.withColumn('c6_0_1_1', df_responses_piv_q2['c6_0_1_1'].cast(IntegerType()))\
.withColumn('c6_0_1_2', df_responses_piv_q2['c6_0_1_2'].cast(IntegerType()))\
.withColumn('c6_0_1_3', df_responses_piv_q2['c6_0_1_3'].cast(IntegerType()))\
.withColumn('c6_0_1_4', df_responses_piv_q2['c6_0_1_4'].cast(IntegerType()))\
.withColumn('c6_0_1_5', df_responses_piv_q2['c6_0_1_5'].cast(IntegerType()))\
.withColumn('c6_0_1_6', df_responses_piv_q2['c6_0_1_6'].cast(IntegerType()))\
.withColumn('c6_0_1_7', df_responses_piv_q2['c6_0_1_7'].cast(IntegerType()))\
.withColumn('c6_0_1_8', df_responses_piv_q2['c6_0_1_8'].cast(IntegerType()))\
.withColumn('c6_0_1_9', df_responses_piv_q2['c6_0_1_9'].cast(IntegerType()))\
.withColumn('c6_0_1_10', df_responses_piv_q2['c6_0_1_10'].cast(IntegerType()))\
.withColumn('c6_0_1_11', df_responses_piv_q2['c6_0_1_11'].cast(IntegerType()))\
.withColumn('c6_0_1_12', df_responses_piv_q2['c6_0_1_12'].cast(IntegerType()))\
.withColumn('c6_0_1_13', df_responses_piv_q2['c6_0_1_13'].cast(IntegerType()))\
.withColumn('c6_0_1_14', df_responses_piv_q2['c6_0_1_14'].cast(IntegerType()))\
.withColumn('c6_0_1_15', df_responses_piv_q2['c6_0_1_15'].cast(IntegerType()))\
.withColumn('c6_0_1_16', df_responses_piv_q2['c6_0_1_16'].cast(IntegerType()))\
.withColumn('c6_0_1_17', df_responses_piv_q2['c6_0_1_17'].cast(IntegerType()))\
.withColumn('c6_0_1_18', df_responses_piv_q2['c6_0_1_18'].cast(IntegerType()))\
.withColumn('c6_0_1_19', df_responses_piv_q2['c6_0_1_19'].cast(IntegerType()))\
.withColumn('c2_2_2_1', df_responses_piv_q2['c2_2_2_1'].cast(IntegerType()))\
.withColumn('c2_2_2_2', df_responses_piv_q2['c2_2_2_2'].cast(IntegerType()))\
.withColumn('c3_0_4_1', df_responses_piv_q2['c3_0_4_1'].cast(IntegerType()))\
.withColumn('c3_0_4_2', df_responses_piv_q2['c3_0_4_2'].cast(IntegerType()))\
.withColumn('c3_0_4_3', df_responses_piv_q2['c3_0_4_3'].cast(IntegerType()))\
.withColumn('c2_1_8_1', df_responses_piv_q2['c2_1_8_1'].cast(IntegerType()))\
.withColumn('c2_1_8_2', df_responses_piv_q2['c2_1_8_2'].cast(IntegerType()))\
.withColumn('c2_1_8_3', df_responses_piv_q2['c2_1_8_3'].cast(IntegerType()))\
.withColumn('c2_1_9_1', df_responses_piv_q2['c2_1_9_1'].cast(IntegerType()))\
.withColumn('c2_1_9_2', df_responses_piv_q2['c2_1_9_2'].cast(IntegerType()))\
.withColumn('c2_1_9_3', df_responses_piv_q2['c2_1_9_3'].cast(IntegerType()))


# COMMAND ----------

df_responses_piv_q2 = df_responses_piv_q2.fillna(0, subset=["c2_0_0", "c2_0b_7","c3_2_0", "c4_0_0", "c4_9_1_1", "c5_0_0", "c5_5_0", "c6_2_0", "c7_0_0", "c7_7_0","c8_0_0", "10_1_1_1", 	"10_1_2_1", 	"10_1_3_1", 	"10_1_4_1", 	"10_1_5_1", 	"10_1_6_1", 	"10_1_7_1", 	"10_1_9_1", 	"10_3_1_1", 	"10_3_1_2", 	"10_3_1_3", 	"10_3_1_4", 	"10_3_1_5", 	"10_3_2_1", 	"10_3_2_2", 	"10_3_2_3", 	"10_3_2_4", 	"10_3_2_5", 	"10_3_3_1", 	"10_3_3_2", 	"10_3_3_3", 	"10_3_3_4", 	"10_3_3_5", 	"10_3_4_1", 	"10_3_4_2", 	"10_3_4_3", 	"10_3_4_4", 	"10_3_4_5", 	"10_3_5_1", 	"10_3_5_2", 	"10_3_5_3", 	"10_3_5_4", 	"10_3_5_5", "13_3_1_1", 	"13_3_1_2", 	"13_3_1_3", 	"13_3_1_4", 	"13_3_1_5", 	"13_3_1_6", "4_6a_1_1","4_6a_1_2","4_6a_1_3","4_6a_1_4","4_6a_1_5","4_6a_1_6","4_6a_1_7","4_6a_1_8","4_6a_1_9","4_6a_1_10","4_6a_1_11","4_6a_1_12","4_6a_1_13","4_6a_1_14","4_6a_1_15","4_6a_1_16","4_6a_1_17","4_6a_1_18","4_6a_1_19","4_6a_1_20","4_6a_1_22","4_6a_1_23","4_6a_1_25","4_6a_1_26","4_6a_1_27","4_6a_1_28","4_6a_1_29","4_6a_1_30","4_6a_1_31", "4_6a_3_1","4_6a_3_2","4_6a_3_3","4_6a_3_4","4_6a_3_5","4_6a_3_6","4_6a_3_7","4_6a_3_8","4_6a_3_9","4_6a_3_10","4_6a_3_11","4_6a_3_12","4_6a_3_13","4_6a_3_14","4_6a_3_15","4_6a_3_16","4_6a_3_17","4_6a_3_18","4_6a_3_19","4_6a_3_20","4_6a_3_22","4_6a_3_23","4_6a_3_25","4_6a_3_26","4_6a_3_27","4_6a_3_28","4_6a_3_29","4_6a_3_30","4_6a_3_31", 	"4_6a_5_1", 	"4_6a_5_10", 	"4_6a_5_11", 	"4_6a_5_12", 	"4_6a_5_13", 	"4_6a_5_14", 	"4_6a_5_15", 	"4_6a_5_16", 	"4_6a_5_17", 	"4_6a_5_18", 	"4_6a_5_19", 	"4_6a_5_2", 	"4_6a_5_20", 	"4_6a_5_22", 	"4_6a_5_23", 	"4_6a_5_25", 	"4_6a_5_26", 	"4_6a_5_27", 	"4_6a_5_28", 	"4_6a_5_29", 	"4_6a_5_3", 	"4_6a_5_30", 	"4_6a_5_31", 	"4_6a_5_4", 	"4_6a_5_5", 	"4_6a_5_6", 	"4_6a_5_7", 	"4_6a_5_8", 	"4_6a_5_9", "4_6b_1_1", 	"4_6b_1_10", 	"4_6b_1_13", 	"4_6b_1_14", 	"4_6b_1_15", 	"4_6b_1_2", 	"4_6b_1_3", 	"4_6b_1_5", 	"4_6b_1_6", 	"4_6b_1_7", 	"4_6b_1_8", 	"4_6b_1_9", "8_1_1_1", 	"8_1_11_1", 	"8_1_2_1", 	"8_1_3_1", 	"8_1_4_1", 	"8_1_5_1", 	"8_1_6_1", 	"8_1_7_1", 	"8_1_8_1", 	"8_1_9_1", 	"8_2_1_3", "5_0a_11", "5_0b_10", "5_0c_13", "5_0d_11", "5_0a_8", "5_0c_10", "5_0d_10","5_0a_7","5_0c_9","5_0d_7", 'c4_4_0_1', 	'c4_4_0_2', 	'c4_4_0_3', 	'c4_4_0_4', 	'c4_4_0_5', 	'c4_4_0_6', 	'c4_4_0_7', 	'c14_0_0_1', 	'c14_0_0_2', 	'c14_0_0_3', 	'c14_0_0_4', 	'c14_0_0_5', 	'c2_1_3_1', 	'c2_1_3_2', 	'c2_1_3_3', 	'c2_1_3_4', 	'c2_1_4_1', 	'c2_1_4_2', 	'c2_1_4_3', 	'c2_1_4_4', 	'c2_1_11_1', 	'c2_1_11_2', 	'c2_1_11_3', 	'c2_1_11_3', 	'c2_1_1_1', 	'c2_1_1_2', 	'c2_1_1_3', 	'c2_1_1_4', 	'c2_1_1_5', 	'c2_1_1_6', 	'c2_1_1_7', 	'c2_1_1_8', 	'c2_1_1_9', 	'c2_1_1_10', 	'c2_1_1_11', 	'c2_1_6_1', 	'c2_1_6_2', 	'c2_1_6_3', 	'c2_1_6_4', 	'c2_1_6_5', 	'c2_1_6_66', 	'c2_1_6_7', 	'c2_1_6_8', 	'c2_1_6_9', 	'c2_1_6_10', 	'c2_1_6_11', 	'c2_1_6_12', 	'c2_1_6_13', 	'c2_1_6_14', 	'c2_1_6_15', 	'c2_1_6_16', 	'c2_1_6_17', 	'c2_1_7_1', 	'c2_1_7_2', 	'c2_1_7_3', 	'c2_1_7_4', 	'c2_1_7_8', 	'c2_1_7_9', 	'c2_1_7_10', 	'c2_1_7_11', 	'c2_1_7_12', 	'c2_1_7_13', 	'c2_1_7_14', 	'c3_0_2_1', 	'c3_0_2_2', 	'c3_0_2_3', 	'c3_0_2_4', 	'c3_0_2_5', 	'c3_0_2_6', 	'c3_0_2_7', 	'c3_0_2_8', 	'c3_0_2_9', 	'c3_0_2_10', 	'c3_0_2_11', 	'c3_0_2_12', 	'c2_2_1_1', 	'c2_2_1_2', 	'c2_2_1_3', 	'c2_2_1_4', 	'c2_2_1_5', 	'c2_2_1_6', 	'c2_2_1_7', 	'c2_2_1_8', 	'c2_2_1_9', 	'c2_2_1_10', 	'c6_2a_1_1', 	'c6_2a_1_2', 	'c6_2a_1_3', 	'c6_2a_1_4', 	'c6_2a_1_5', 	'c6_2a_1_6', 	'c6_2a_1_7', 	'c6_2a_1_8', 	'c6_2a_1_9', 	'c6_2a_1_10', 	'c6_2a_1_11', 	'c6_2a_1_12', 	'c6_0_1_1', 	'c6_0_1_2', 	'c6_0_1_3', 	'c6_0_1_4', 	'c6_0_1_5', 	'c6_0_1_6', 	'c6_0_1_7', 	'c6_0_1_8', 	'c6_0_1_9', 	'c6_0_1_10', 	'c6_0_1_11', 	'c6_0_1_12', 	'c6_0_1_13', 	'c6_0_1_14', 	'c6_0_1_15', 	'c6_0_1_16', 	'c6_0_1_17', 	'c6_0_1_18', 	'c6_0_1_19', 	'c2_2_2_1', 	'c2_2_2_2', 	'c3_0_4_1', 	'c3_0_4_2', 	'c3_0_4_3', 	'c2_1_8_1', 	'c2_1_8_2', 	'c2_1_8_3', 	'c2_1_9_1', 	'c2_1_9_2', 	'c2_1_9_3'])


# COMMAND ----------

# DBTITLE 1,Group by Account-Year-q_id 
df_responses_piv_q2 = df_responses_piv_q2.groupBy('account_year', "account", "year").agg(first('0_5_1_1', ignoreNulls = True).alias('0_5_1_1'),\
first('0_5_2_1', ignoreNulls = True).alias('0_5_2_1'),\
first('0_5_3_1', ignoreNulls = True).alias('0_5_3_1'),\
first('0_6_1_1', ignoreNulls = True).alias('0_6_1_1'),\
sum('10_1_1_1').alias('10_1_1_1'),\
sum('10_1_2_1').alias('10_1_2_1'),\
sum('10_1_3_1').alias('10_1_3_1'),\
sum('10_1_4_1').alias('10_1_4_1'),\
sum('10_1_5_1').alias('10_1_5_1'),\
sum('10_1_6_1').alias('10_1_6_1'),\
sum('10_1_7_1').alias('10_1_7_1'),\
sum('10_1_9_1').alias('10_1_9_1'),\
sum('10_3_1_1').alias('10_3_1_1'),\
sum('10_3_1_2').alias('10_3_1_2'),\
sum('10_3_1_3').alias('10_3_1_3'),\
sum('10_3_1_4').alias('10_3_1_4'),\
sum('10_3_1_5').alias('10_3_1_5'),\
sum('10_3_2_1').alias('10_3_2_1'),\
sum('10_3_2_2').alias('10_3_2_2'),\
sum('10_3_2_3').alias('10_3_2_3'),\
sum('10_3_2_4').alias('10_3_2_4'),\
sum('10_3_2_5').alias('10_3_2_5'),\
sum('10_3_3_1').alias('10_3_3_1'),\
sum('10_3_3_2').alias('10_3_3_2'),\
sum('10_3_3_3').alias('10_3_3_3'),\
sum('10_3_3_4').alias('10_3_3_4'),\
sum('10_3_3_5').alias('10_3_3_5'),\
sum('10_3_4_1').alias('10_3_4_1'),\
sum('10_3_4_2').alias('10_3_4_2'),\
sum('10_3_4_3').alias('10_3_4_3'),\
sum('10_3_4_4').alias('10_3_4_4'),\
sum('10_3_4_5').alias('10_3_4_5'),\
sum('10_3_5_1').alias('10_3_5_1'),\
sum('10_3_5_2').alias('10_3_5_2'),\
sum('10_3_5_3').alias('10_3_5_3'),\
sum('10_3_5_4').alias('10_3_5_4'),\
sum('10_3_5_5').alias('10_3_5_5'),\
first('11_0_0', ignoreNulls = True).alias('11_0_0'),\
first('13_2_0', ignoreNulls = True).alias('13_2_0'),\
sum('13_3_1_1').alias('13_3_1_1'),\
sum('13_3_1_2').alias('13_3_1_2'),\
sum('13_3_1_3').alias('13_3_1_3'),\
sum('13_3_1_4').alias('13_3_1_4'),\
sum('13_3_1_5').alias('13_3_1_5'),\
sum('13_3_1_6').alias('13_3_1_6'),\
first('14_1_0', ignoreNulls = True).alias('14_1_0'),\
first('14_2a_1', ignoreNulls = True).alias('14_2a_1'),\
first('14_2a_2', ignoreNulls = True).alias('14_2a_2'),\
first('14_2a_3', ignoreNulls = True).alias('14_2a_3'),\
first('14_2a_5', ignoreNulls = True).alias('14_2a_5'),\
first('14_3_2', ignoreNulls = True).alias('14_3_2'),\
first('14_3_4', ignoreNulls = True).alias('14_3_4'),\
first('2_0b_4', ignoreNulls = True).alias('2_0b_4'),\
first('2_0b_6', ignoreNulls = True).alias('2_0b_6'),\
first('2_1_1', ignoreNulls = True).alias('2_1_1'),\
first('2_1_5', ignoreNulls = True).alias('2_1_5'),\
first('2_1_6', ignoreNulls = True).alias('2_1_6'),\
first('2_1_7', ignoreNulls = True).alias('2_1_7'),\
first('2_1_8', ignoreNulls = True).alias('2_1_8'),\
first('2_1_9', ignoreNulls = True).alias('2_1_9'),\
first('2_2_1', ignoreNulls = True).alias('2_2_1'),\
first('2_2_2', ignoreNulls = True).alias('2_2_2'),\
first('2_2_4', ignoreNulls = True).alias('2_2_4'),\
first('3_0_2', ignoreNulls = True).alias('3_0_2'),\
first('3_0_4', ignoreNulls = True).alias('3_0_4'),\
first('3_0_8', ignoreNulls = True).alias('3_0_8'),\
first('3_2a_1', ignoreNulls = True).alias('3_2a_1'),\
first('3_2a_3', ignoreNulls = True).alias('3_2a_3'),\
first('3_2a_5', ignoreNulls = True).alias('3_2a_5'),\
first('3_2a_6', ignoreNulls = True).alias('3_2a_6'),\
first('3_3_1', ignoreNulls = True).alias('3_3_1'),\
first('3_3_3', ignoreNulls = True).alias('3_3_3'),\
first('3_3_4', ignoreNulls = True).alias('3_3_4'),\
first('4_4_0', ignoreNulls = True).alias('4_4_0'),\
sum('4_6a_1_1').alias('4_6a_1_1'),\
sum('4_6a_1_10').alias('4_6a_1_10'),\
sum('4_6a_1_11').alias('4_6a_1_11'),\
sum('4_6a_1_12').alias('4_6a_1_12'),\
sum('4_6a_1_13').alias('4_6a_1_13'),\
sum('4_6a_1_14').alias('4_6a_1_14'),\
sum('4_6a_1_15').alias('4_6a_1_15'),\
sum('4_6a_1_16').alias('4_6a_1_16'),\
sum('4_6a_1_17').alias('4_6a_1_17'),\
sum('4_6a_1_18').alias('4_6a_1_18'),\
sum('4_6a_1_19').alias('4_6a_1_19'),\
sum('4_6a_1_2').alias('4_6a_1_2'),\
sum('4_6a_1_20').alias('4_6a_1_20'),\
sum('4_6a_1_22').alias('4_6a_1_22'),\
sum('4_6a_1_23').alias('4_6a_1_23'),\
sum('4_6a_1_25').alias('4_6a_1_25'),\
sum('4_6a_1_26').alias('4_6a_1_26'),\
sum('4_6a_1_27').alias('4_6a_1_27'),\
sum('4_6a_1_28').alias('4_6a_1_28'),\
sum('4_6a_1_29').alias('4_6a_1_29'),\
sum('4_6a_1_3').alias('4_6a_1_3'),\
sum('4_6a_1_30').alias('4_6a_1_30'),\
sum('4_6a_1_31').alias('4_6a_1_31'),\
sum('4_6a_1_4').alias('4_6a_1_4'),\
sum('4_6a_1_5').alias('4_6a_1_5'),\
sum('4_6a_1_6').alias('4_6a_1_6'),\
sum('4_6a_1_7').alias('4_6a_1_7'),\
sum('4_6a_1_8').alias('4_6a_1_8'),\
sum('4_6a_1_9').alias('4_6a_1_9'),\
first('4_6a_2_1', ignoreNulls = True).alias('4_6a_2_1'),\
first('4_6a_2_10', ignoreNulls = True).alias('4_6a_2_10'),\
first('4_6a_2_11', ignoreNulls = True).alias('4_6a_2_11'),\
first('4_6a_2_12', ignoreNulls = True).alias('4_6a_2_12'),\
first('4_6a_2_13', ignoreNulls = True).alias('4_6a_2_13'),\
first('4_6a_2_14', ignoreNulls = True).alias('4_6a_2_14'),\
first('4_6a_2_15', ignoreNulls = True).alias('4_6a_2_15'),\
first('4_6a_2_16', ignoreNulls = True).alias('4_6a_2_16'),\
first('4_6a_2_17', ignoreNulls = True).alias('4_6a_2_17'),\
first('4_6a_2_18', ignoreNulls = True).alias('4_6a_2_18'),\
first('4_6a_2_19', ignoreNulls = True).alias('4_6a_2_19'),\
first('4_6a_2_2', ignoreNulls = True).alias('4_6a_2_2'),\
first('4_6a_2_20', ignoreNulls = True).alias('4_6a_2_20'),\
first('4_6a_2_22', ignoreNulls = True).alias('4_6a_2_22'),\
first('4_6a_2_23', ignoreNulls = True).alias('4_6a_2_23'),\
first('4_6a_2_25', ignoreNulls = True).alias('4_6a_2_25'),\
first('4_6a_2_26', ignoreNulls = True).alias('4_6a_2_26'),\
first('4_6a_2_27', ignoreNulls = True).alias('4_6a_2_27'),\
first('4_6a_2_28', ignoreNulls = True).alias('4_6a_2_28'),\
first('4_6a_2_29', ignoreNulls = True).alias('4_6a_2_29'),\
first('4_6a_2_3', ignoreNulls = True).alias('4_6a_2_3'),\
first('4_6a_2_30', ignoreNulls = True).alias('4_6a_2_30'),\
first('4_6a_2_31', ignoreNulls = True).alias('4_6a_2_31'),\
first('4_6a_2_4', ignoreNulls = True).alias('4_6a_2_4'),\
first('4_6a_2_5', ignoreNulls = True).alias('4_6a_2_5'),\
first('4_6a_2_6', ignoreNulls = True).alias('4_6a_2_6'),\
first('4_6a_2_7', ignoreNulls = True).alias('4_6a_2_7'),\
first('4_6a_2_8', ignoreNulls = True).alias('4_6a_2_8'),\
first('4_6a_2_9', ignoreNulls = True).alias('4_6a_2_9'),\
sum('4_6a_3_1').alias('4_6a_3_1'),\
sum('4_6a_3_10').alias('4_6a_3_10'),\
sum('4_6a_3_11').alias('4_6a_3_11'),\
sum('4_6a_3_12').alias('4_6a_3_12'),\
sum('4_6a_3_13').alias('4_6a_3_13'),\
sum('4_6a_3_14').alias('4_6a_3_14'),\
sum('4_6a_3_15').alias('4_6a_3_15'),\
sum('4_6a_3_16').alias('4_6a_3_16'),\
sum('4_6a_3_17').alias('4_6a_3_17'),\
sum('4_6a_3_18').alias('4_6a_3_18'),\
sum('4_6a_3_19').alias('4_6a_3_19'),\
sum('4_6a_3_2').alias('4_6a_3_2'),\
sum('4_6a_3_20').alias('4_6a_3_20'),\
sum('4_6a_3_22').alias('4_6a_3_22'),\
sum('4_6a_3_23').alias('4_6a_3_23'),\
sum('4_6a_3_25').alias('4_6a_3_25'),\
sum('4_6a_3_26').alias('4_6a_3_26'),\
sum('4_6a_3_27').alias('4_6a_3_27'),\
sum('4_6a_3_28').alias('4_6a_3_28'),\
sum('4_6a_3_29').alias('4_6a_3_29'),\
sum('4_6a_3_3').alias('4_6a_3_3'),\
sum('4_6a_3_30').alias('4_6a_3_30'),\
sum('4_6a_3_31').alias('4_6a_3_31'),\
sum('4_6a_3_4').alias('4_6a_3_4'),\
sum('4_6a_3_5').alias('4_6a_3_5'),\
sum('4_6a_3_6').alias('4_6a_3_6'),\
sum('4_6a_3_7').alias('4_6a_3_7'),\
sum('4_6a_3_8').alias('4_6a_3_8'),\
sum('4_6a_3_9').alias('4_6a_3_9'),\
first('4_6a_4_1', ignoreNulls = True).alias('4_6a_4_1'),\
first('4_6a_4_10', ignoreNulls = True).alias('4_6a_4_10'),\
first('4_6a_4_11', ignoreNulls = True).alias('4_6a_4_11'),\
first('4_6a_4_12', ignoreNulls = True).alias('4_6a_4_12'),\
first('4_6a_4_13', ignoreNulls = True).alias('4_6a_4_13'),\
first('4_6a_4_14', ignoreNulls = True).alias('4_6a_4_14'),\
first('4_6a_4_15', ignoreNulls = True).alias('4_6a_4_15'),\
first('4_6a_4_16', ignoreNulls = True).alias('4_6a_4_16'),\
first('4_6a_4_17', ignoreNulls = True).alias('4_6a_4_17'),\
first('4_6a_4_18', ignoreNulls = True).alias('4_6a_4_18'),\
first('4_6a_4_19', ignoreNulls = True).alias('4_6a_4_19'),\
first('4_6a_4_2', ignoreNulls = True).alias('4_6a_4_2'),\
first('4_6a_4_20', ignoreNulls = True).alias('4_6a_4_20'),\
first('4_6a_4_22', ignoreNulls = True).alias('4_6a_4_22'),\
first('4_6a_4_23', ignoreNulls = True).alias('4_6a_4_23'),\
first('4_6a_4_25', ignoreNulls = True).alias('4_6a_4_25'),\
first('4_6a_4_26', ignoreNulls = True).alias('4_6a_4_26'),\
first('4_6a_4_27', ignoreNulls = True).alias('4_6a_4_27'),\
first('4_6a_4_28', ignoreNulls = True).alias('4_6a_4_28'),\
first('4_6a_4_29', ignoreNulls = True).alias('4_6a_4_29'),\
first('4_6a_4_3', ignoreNulls = True).alias('4_6a_4_3'),\
first('4_6a_4_30', ignoreNulls = True).alias('4_6a_4_30'),\
first('4_6a_4_31', ignoreNulls = True).alias('4_6a_4_31'),\
first('4_6a_4_4', ignoreNulls = True).alias('4_6a_4_4'),\
first('4_6a_4_5', ignoreNulls = True).alias('4_6a_4_5'),\
first('4_6a_4_6', ignoreNulls = True).alias('4_6a_4_6'),\
first('4_6a_4_7', ignoreNulls = True).alias('4_6a_4_7'),\
first('4_6a_4_8', ignoreNulls = True).alias('4_6a_4_8'),\
first('4_6a_4_9', ignoreNulls = True).alias('4_6a_4_9'),\
sum('4_6a_5_1').alias('4_6a_5_1'),\
sum('4_6a_5_10').alias('4_6a_5_10'),\
sum('4_6a_5_11').alias('4_6a_5_11'),\
sum('4_6a_5_12').alias('4_6a_5_12'),\
sum('4_6a_5_13').alias('4_6a_5_13'),\
sum('4_6a_5_14').alias('4_6a_5_14'),\
sum('4_6a_5_15').alias('4_6a_5_15'),\
sum('4_6a_5_16').alias('4_6a_5_16'),\
sum('4_6a_5_17').alias('4_6a_5_17'),\
sum('4_6a_5_18').alias('4_6a_5_18'),\
sum('4_6a_5_19').alias('4_6a_5_19'),\
sum('4_6a_5_2').alias('4_6a_5_2'),\
sum('4_6a_5_20').alias('4_6a_5_20'),\
sum('4_6a_5_22').alias('4_6a_5_22'),\
sum('4_6a_5_23').alias('4_6a_5_23'),\
sum('4_6a_5_25').alias('4_6a_5_25'),\
sum('4_6a_5_26').alias('4_6a_5_26'),\
sum('4_6a_5_27').alias('4_6a_5_27'),\
sum('4_6a_5_28').alias('4_6a_5_28'),\
sum('4_6a_5_29').alias('4_6a_5_29'),\
sum('4_6a_5_3').alias('4_6a_5_3'),\
sum('4_6a_5_30').alias('4_6a_5_30'),\
sum('4_6a_5_31').alias('4_6a_5_31'),\
sum('4_6a_5_4').alias('4_6a_5_4'),\
sum('4_6a_5_5').alias('4_6a_5_5'),\
sum('4_6a_5_6').alias('4_6a_5_6'),\
sum('4_6a_5_7').alias('4_6a_5_7'),\
sum('4_6a_5_8').alias('4_6a_5_8'),\
sum('4_6a_5_9').alias('4_6a_5_9'),\
first('4_6a_6_1', ignoreNulls = True).alias('4_6a_6_1'),\
first('4_6a_6_10', ignoreNulls = True).alias('4_6a_6_10'),\
first('4_6a_6_11', ignoreNulls = True).alias('4_6a_6_11'),\
first('4_6a_6_12', ignoreNulls = True).alias('4_6a_6_12'),\
first('4_6a_6_13', ignoreNulls = True).alias('4_6a_6_13'),\
first('4_6a_6_14', ignoreNulls = True).alias('4_6a_6_14'),\
first('4_6a_6_15', ignoreNulls = True).alias('4_6a_6_15'),\
first('4_6a_6_16', ignoreNulls = True).alias('4_6a_6_16'),\
first('4_6a_6_17', ignoreNulls = True).alias('4_6a_6_17'),\
first('4_6a_6_18', ignoreNulls = True).alias('4_6a_6_18'),\
first('4_6a_6_19', ignoreNulls = True).alias('4_6a_6_19'),\
first('4_6a_6_2', ignoreNulls = True).alias('4_6a_6_2'),\
first('4_6a_6_20', ignoreNulls = True).alias('4_6a_6_20'),\
first('4_6a_6_22', ignoreNulls = True).alias('4_6a_6_22'),\
first('4_6a_6_23', ignoreNulls = True).alias('4_6a_6_23'),\
first('4_6a_6_25', ignoreNulls = True).alias('4_6a_6_25'),\
first('4_6a_6_26', ignoreNulls = True).alias('4_6a_6_26'),\
first('4_6a_6_27', ignoreNulls = True).alias('4_6a_6_27'),\
first('4_6a_6_28', ignoreNulls = True).alias('4_6a_6_28'),\
first('4_6a_6_29', ignoreNulls = True).alias('4_6a_6_29'),\
first('4_6a_6_3', ignoreNulls = True).alias('4_6a_6_3'),\
first('4_6a_6_30', ignoreNulls = True).alias('4_6a_6_30'),\
first('4_6a_6_31', ignoreNulls = True).alias('4_6a_6_31'),\
first('4_6a_6_4', ignoreNulls = True).alias('4_6a_6_4'),\
first('4_6a_6_5', ignoreNulls = True).alias('4_6a_6_5'),\
first('4_6a_6_6', ignoreNulls = True).alias('4_6a_6_6'),\
first('4_6a_6_7', ignoreNulls = True).alias('4_6a_6_7'),\
first('4_6a_6_8', ignoreNulls = True).alias('4_6a_6_8'),\
first('4_6a_6_9', ignoreNulls = True).alias('4_6a_6_9'),\
sum('4_6b_1_1').alias('4_6b_1_1'),\
sum('4_6b_1_10').alias('4_6b_1_10'),\
first('4_6b_1_11', ignoreNulls = True).alias('4_6b_1_11'),\
first('4_6b_1_12', ignoreNulls = True).alias('4_6b_1_12'),\
sum('4_6b_1_13').alias('4_6b_1_13'),\
sum('4_6b_1_14').alias('4_6b_1_14'),\
sum('4_6b_1_15').alias('4_6b_1_15'),\
first('4_6b_1_16', ignoreNulls = True).alias('4_6b_1_16'),\
first('4_6b_1_17', ignoreNulls = True).alias('4_6b_1_17'),\
sum('4_6b_1_2').alias('4_6b_1_2'),\
sum('4_6b_1_3').alias('4_6b_1_3'),\
first('4_6b_1_4', ignoreNulls = True).alias('4_6b_1_4'),\
sum('4_6b_1_5').alias('4_6b_1_5'),\
sum('4_6b_1_6').alias('4_6b_1_6'),\
sum('4_6b_1_7').alias('4_6b_1_7'),\
sum('4_6b_1_8').alias('4_6b_1_8'),\
sum('4_6b_1_9').alias('4_6b_1_9'),\
first('4_6c_13_1', ignoreNulls = True).alias('4_6c_13_1'),\
first('4_6c_1_1', ignoreNulls = True).alias('4_6c_1_1'),\
first('4_6c_3_1', ignoreNulls = True).alias('4_6c_3_1'),\
first('4_6c_8_1', ignoreNulls = True).alias('4_6c_8_1'),\
first('4_6d_1', ignoreNulls = True).alias('4_6d_1'),\
first('4_6d_2', ignoreNulls = True).alias('4_6d_2'),\
first('4_6d_3', ignoreNulls = True).alias('4_6d_3'),\
first('4_6d_4', ignoreNulls = True).alias('4_6d_4'),\
first('5_0a_1', ignoreNulls = True).alias('5_0a_1'),\
avg('5_0a_11').alias('5_0a_11'),\
first('5_0a_3', ignoreNulls = True).alias('5_0a_3'),\
first('5_0a_5', ignoreNulls = True).alias('5_0a_5'),\
first('5_0a_6', ignoreNulls = True).alias('5_0a_6'),\
sum('5_0a_7').alias('5_0a_7'),\
avg('5_0a_8').alias('5_0a_8'),\
first('5_0a_9', ignoreNulls = True).alias('5_0a_9'),\
first('5_0b_1', ignoreNulls = True).alias('5_0b_1'),\
avg('5_0b_10').alias('5_0b_10'),\
first('5_0b_3', ignoreNulls = True).alias('5_0b_3'),\
first('5_0b_5', ignoreNulls = True).alias('5_0b_5'),\
first('5_0b_7', ignoreNulls = True).alias('5_0b_7'),\
sum('5_0b_8').alias('5_0b_8'),\
sum('5_0b_9').alias('5_0b_9'),\
first('5_0c_1', ignoreNulls = True).alias('5_0c_1'),\
avg('5_0c_10').alias('5_0c_10'),\
first('5_0c_11', ignoreNulls = True).alias('5_0c_11'),\
sum('5_0c_12').alias('5_0c_12'),\
avg('5_0c_13').alias('5_0c_13'),\
first('5_0c_3', ignoreNulls = True).alias('5_0c_3'),\
first('5_0c_5', ignoreNulls = True).alias('5_0c_5'),\
first('5_0c_6', ignoreNulls = True).alias('5_0c_6'),\
first('5_0c_7', ignoreNulls = True).alias('5_0c_7'),\
sum('5_0c_8').alias('5_0c_8'),\
sum('5_0c_9').alias('5_0c_9'),\
first('5_0d_1', ignoreNulls = True).alias('5_0d_1'),\
avg('5_0d_10').alias('5_0d_10'),\
avg('5_0d_11').alias('5_0d_11'),\
first('5_0d_3', ignoreNulls = True).alias('5_0d_3'),\
first('5_0d_5', ignoreNulls = True).alias('5_0d_5'),\
first('5_0d_6', ignoreNulls = True).alias('5_0d_6'),\
sum('5_0d_7').alias('5_0d_7'),\
first('5_0d_8', ignoreNulls = True).alias('5_0d_8'),\
sum('5_0d_9').alias('5_0d_9'),\
first('5_4_7', ignoreNulls = True).alias('5_4_7'),\
first('5_4_8', ignoreNulls = True).alias('5_4_8'),\
first('6_0_2', ignoreNulls = True).alias('6_0_2'),\
first('6_13_1_1', ignoreNulls = True).alias('6_13_1_1'),\
first('6_2a_1', ignoreNulls = True).alias('6_2a_1'),\
first('6_2a_3', ignoreNulls = True).alias('6_2a_3'),\
first('6_5_7', ignoreNulls = True).alias('6_5_7'),\
first('6_5_9', ignoreNulls = True).alias('6_5_9'),\
first('7_2_0', ignoreNulls = True).alias('7_2_0'),\
first('7_3_1_1', ignoreNulls = True).alias('7_3_1_1'),\
first('7_4_0', ignoreNulls = True).alias('7_4_0'),\
first('7_6_2_1', ignoreNulls = True).alias('7_6_2_1'),\
first('7_6_3_1', ignoreNulls = True).alias('7_6_3_1'),\
first('7_6_4_1', ignoreNulls = True).alias('7_6_4_1'),\
first('7_7a_2', ignoreNulls = True).alias('7_7a_2'),\
first('8_0a_3', ignoreNulls = True).alias('8_0a_3'),\
first('8_0a_4', ignoreNulls = True).alias('8_0a_4'),\
first('8_0a_6', ignoreNulls = True).alias('8_0a_6'),\
first('8_0a_7', ignoreNulls = True).alias('8_0a_7'),\
first('8_0a_9', ignoreNulls = True).alias('8_0a_9'),\
sum('8_1_11_1').alias('8_1_11_1'),\
sum('8_1_1_1').alias('8_1_1_1'),\
sum('8_1_2_1').alias('8_1_2_1'),\
sum('8_1_3_1').alias('8_1_3_1'),\
sum('8_1_4_1').alias('8_1_4_1'),\
sum('8_1_5_1').alias('8_1_5_1'),\
sum('8_1_6_1').alias('8_1_6_1'),\
sum('8_1_7_1').alias('8_1_7_1'),\
sum('8_1_8_1').alias('8_1_8_1'),\
sum('8_1_9_1').alias('8_1_9_1'),\
sum('8_2_1_3').alias('8_2_1_3'),\
first('9_1_1_1', ignoreNulls = True).alias('9_1_1_1'),\
first('9_1_1_2', ignoreNulls = True).alias('9_1_1_2'),\
first('9_1_1_3', ignoreNulls = True).alias('9_1_1_3'),\
first('9_1_1_4', ignoreNulls = True).alias('9_1_1_4'),\
first('9_1_3_1', ignoreNulls = True).alias('9_1_3_1'),\
first('9_1_3_2', ignoreNulls = True).alias('9_1_3_2'),\
first('9_1_3_3', ignoreNulls = True).alias('9_1_3_3'),\
first('9_1_3_4', ignoreNulls = True).alias('9_1_3_4'),\
sum('c2_0_0').alias('c2_0_0'),\
sum('c3_2_0').alias('c3_2_0'),\
sum('c4_0_0').alias('c4_0_0'),\
sum('c4_9_1_1').alias('c4_9_1_1'),\
sum('c5_5_0').alias('c5_5_0'),\
sum('c7_0_0').alias('c7_0_0'),\
sum('c7_7_0').alias('c7_7_0'),\
sum('c8_0_0').alias('c8_0_0'),\
first('c2_0b_2', ignoreNulls = True).alias('c2_0b_2'),\
sum('c2_0b_7').alias('c2_0b_7'),\
sum('c6_2_0').alias('c6_2_0'),\
sum('c5_0_0').alias('c5_0_0'),\
first('c0_1_1_1', ignoreNulls = True).alias('c0_1_1_1'),\
first('c5_0_sector', ignoreNulls = True).alias('c5_0_sector'),\
first('c6_5_3', ignoreNulls = True).alias('c6_5_3'),\
first('c6_5_4', ignoreNulls = True).alias('c6_5_4'),\
first('c3_2a_9', ignoreNulls = True).alias('c3_2a_9'),\
first('csector', ignoreNulls = True).alias('csector'),\
first('cboundary', ignoreNulls = True).alias('cboundary'),\
first('c5_0_base_year', ignoreNulls = True).alias('c5_0_base_year'),\
first('c5_0_target_year', ignoreNulls = True).alias('c5_0_target_year'),\
first('c5_0_target_year_set', ignoreNulls = True).alias('c5_0_target_year_set'),\
sum('c4_4_0_1').alias('c4_4_0_1'),\
sum('c4_4_0_2').alias('c4_4_0_2'),\
sum('c4_4_0_3').alias('c4_4_0_3'),\
sum('c4_4_0_4').alias('c4_4_0_4'),\
sum('c4_4_0_5').alias('c4_4_0_5'),\
sum('c4_4_0_6').alias('c4_4_0_6'),\
sum('c4_4_0_7').alias('c4_4_0_7'),\
sum('c14_0_0_1').alias('c14_0_0_1'),\
sum('c14_0_0_2').alias('c14_0_0_2'),\
sum('c14_0_0_3').alias('c14_0_0_3'),\
sum('c14_0_0_4').alias('c14_0_0_4'),\
sum('c14_0_0_5').alias('c14_0_0_5'),\
sum('c2_1_3_1').alias('c2_1_3_1'),\
sum('c2_1_3_2').alias('c2_1_3_2'),\
sum('c2_1_3_3').alias('c2_1_3_3'),\
sum('c2_1_3_4').alias('c2_1_3_4'),\
sum('c2_1_4_1').alias('c2_1_4_1'),\
sum('c2_1_4_2').alias('c2_1_4_2'),\
sum('c2_1_4_3').alias('c2_1_4_3'),\
sum('c2_1_4_4').alias('c2_1_4_4'),\
sum('c2_1_11_1').alias('c2_1_11_1'),\
sum('c2_1_11_2').alias('c2_1_11_2'),\
sum('c2_1_11_3').alias('c2_1_11_3'),\
sum('c2_1_1_1').alias('c2_1_1_1'),\
sum('c2_1_1_2').alias('c2_1_1_2'),\
sum('c2_1_1_3').alias('c2_1_1_3'),\
sum('c2_1_1_4').alias('c2_1_1_4'),\
sum('c2_1_1_5').alias('c2_1_1_5'),\
sum('c2_1_1_6').alias('c2_1_1_6'),\
sum('c2_1_1_7').alias('c2_1_1_7'),\
sum('c2_1_1_8').alias('c2_1_1_8'),\
sum('c2_1_1_9').alias('c2_1_1_9'),\
sum('c2_1_1_10').alias('c2_1_1_10'),\
sum('c2_1_1_11').alias('c2_1_1_11'),\
sum('c2_1_6_1').alias('c2_1_6_1'),\
sum('c2_1_6_2').alias('c2_1_6_2'),\
sum('c2_1_6_3').alias('c2_1_6_3'),\
sum('c2_1_6_4').alias('c2_1_6_4'),\
sum('c2_1_6_5').alias('c2_1_6_5'),\
sum('c2_1_6_66').alias('c2_1_6_66'),\
sum('c2_1_6_7').alias('c2_1_6_7'),\
sum('c2_1_6_8').alias('c2_1_6_8'),\
sum('c2_1_6_9').alias('c2_1_6_9'),\
sum('c2_1_6_10').alias('c2_1_6_10'),\
sum('c2_1_6_11').alias('c2_1_6_11'),\
sum('c2_1_6_12').alias('c2_1_6_12'),\
sum('c2_1_6_13').alias('c2_1_6_13'),\
sum('c2_1_6_14').alias('c2_1_6_14'),\
sum('c2_1_6_15').alias('c2_1_6_15'),\
sum('c2_1_6_16').alias('c2_1_6_16'),\
sum('c2_1_6_17').alias('c2_1_6_17'),\
sum('c2_1_7_1').alias('c2_1_7_1'),\
sum('c2_1_7_2').alias('c2_1_7_2'),\
sum('c2_1_7_3').alias('c2_1_7_3'),\
sum('c2_1_7_4').alias('c2_1_7_4'),\
sum('c2_1_7_8').alias('c2_1_7_8'),\
sum('c2_1_7_9').alias('c2_1_7_9'),\
sum('c2_1_7_10').alias('c2_1_7_10'),\
sum('c2_1_7_11').alias('c2_1_7_11'),\
sum('c2_1_7_12').alias('c2_1_7_12'),\
sum('c2_1_7_13').alias('c2_1_7_13'),\
sum('c2_1_7_14').alias('c2_1_7_14'),\
sum('c3_0_2_1').alias('c3_0_2_1'),\
sum('c3_0_2_2').alias('c3_0_2_2'),\
sum('c3_0_2_3').alias('c3_0_2_3'),\
sum('c3_0_2_4').alias('c3_0_2_4'),\
sum('c3_0_2_5').alias('c3_0_2_5'),\
sum('c3_0_2_6').alias('c3_0_2_6'),\
sum('c3_0_2_7').alias('c3_0_2_7'),\
sum('c3_0_2_8').alias('c3_0_2_8'),\
sum('c3_0_2_9').alias('c3_0_2_9'),\
sum('c3_0_2_10').alias('c3_0_2_10'),\
sum('c3_0_2_11').alias('c3_0_2_11'),\
sum('c3_0_2_12').alias('c3_0_2_12'),\
sum('c2_2_1_1').alias('c2_2_1_1'),\
sum('c2_2_1_2').alias('c2_2_1_2'),\
sum('c2_2_1_3').alias('c2_2_1_3'),\
sum('c2_2_1_4').alias('c2_2_1_4'),\
sum('c2_2_1_5').alias('c2_2_1_5'),\
sum('c2_2_1_6').alias('c2_2_1_6'),\
sum('c2_2_1_7').alias('c2_2_1_7'),\
sum('c2_2_1_8').alias('c2_2_1_8'),\
sum('c2_2_1_9').alias('c2_2_1_9'),\
sum('c2_2_1_10').alias('c2_2_1_10'),\
sum('c6_2a_1_1').alias('c6_2a_1_1'),\
sum('c6_2a_1_2').alias('c6_2a_1_2'),\
sum('c6_2a_1_3').alias('c6_2a_1_3'),\
sum('c6_2a_1_4').alias('c6_2a_1_4'),\
sum('c6_2a_1_5').alias('c6_2a_1_5'),\
sum('c6_2a_1_6').alias('c6_2a_1_6'),\
sum('c6_2a_1_7').alias('c6_2a_1_7'),\
sum('c6_2a_1_8').alias('c6_2a_1_8'),\
sum('c6_2a_1_9').alias('c6_2a_1_9'),\
sum('c6_2a_1_10').alias('c6_2a_1_10'),\
sum('c6_2a_1_11').alias('c6_2a_1_11'),\
sum('c6_2a_1_12').alias('c6_2a_1_12'),\
sum('c6_0_1_1').alias('c6_0_1_1'),\
sum('c6_0_1_2').alias('c6_0_1_2'),\
sum('c6_0_1_3').alias('c6_0_1_3'),\
sum('c6_0_1_4').alias('c6_0_1_4'),\
sum('c6_0_1_5').alias('c6_0_1_5'),\
sum('c6_0_1_6').alias('c6_0_1_6'),\
sum('c6_0_1_7').alias('c6_0_1_7'),\
sum('c6_0_1_8').alias('c6_0_1_8'),\
sum('c6_0_1_9').alias('c6_0_1_9'),\
sum('c6_0_1_10').alias('c6_0_1_10'),\
sum('c6_0_1_11').alias('c6_0_1_11'),\
sum('c6_0_1_12').alias('c6_0_1_12'),\
sum('c6_0_1_13').alias('c6_0_1_13'),\
sum('c6_0_1_14').alias('c6_0_1_14'),\
sum('c6_0_1_15').alias('c6_0_1_15'),\
sum('c6_0_1_16').alias('c6_0_1_16'),\
sum('c6_0_1_17').alias('c6_0_1_17'),\
sum('c6_0_1_18').alias('c6_0_1_18'),\
sum('c6_0_1_19').alias('c6_0_1_19'),\
sum('c2_2_2_1').alias('c2_2_2_1'),\
sum('c2_2_2_2').alias('c2_2_2_2'),\
sum('c3_0_4_1').alias('c3_0_4_1'),\
sum('c3_0_4_2').alias('c3_0_4_2'),\
sum('c3_0_4_3').alias('c3_0_4_3'),\
sum('c2_1_8_1').alias('c2_1_8_1'),\
sum('c2_1_8_2').alias('c2_1_8_2'),\
sum('c2_1_8_3').alias('c2_1_8_3'),\
sum('c2_1_9_1').alias('c2_1_9_1'),\
sum('c2_1_9_2').alias('c2_1_9_2'),\
sum('c2_1_9_3').alias('c2_1_9_3'))


# COMMAND ----------

print(df_responses_piv_q2.shape())

# COMMAND ----------

# DBTITLE 1,FE Numeric
df_responses_piv_q2 = df_responses_piv_q2.withColumn("c4_6a_1_sum", (col("4_6a_1_1")+col("4_6a_1_2")+col("4_6a_1_3")+col("4_6a_1_4")+col("4_6a_1_5")+col("4_6a_1_6")+col("4_6a_1_7")+col("4_6a_1_8")+col("4_6a_1_9")+col("4_6a_1_10")+col("4_6a_1_11")+col("4_6a_1_12")+col("4_6a_1_13")+col("4_6a_1_14")+col("4_6a_1_15")+col("4_6a_1_16")+col("4_6a_1_17")+col("4_6a_1_18")+col("4_6a_1_19")+col("4_6a_1_20")+col("4_6a_1_22")+col("4_6a_1_23")+col("4_6a_1_25")+col("4_6a_1_26")+col("4_6a_1_27")+col("4_6a_1_28")+col("4_6a_1_29")+col("4_6a_1_30")+col("4_6a_1_31")))\
.withColumn("c4_6a_1_avg", ((col("4_6a_1_1")+col("4_6a_1_2")+col("4_6a_1_3")+col("4_6a_1_4")+col("4_6a_1_5")+col("4_6a_1_6")+col("4_6a_1_7")+col("4_6a_1_8")+col("4_6a_1_9")+col("4_6a_1_10")+col("4_6a_1_11")+col("4_6a_1_12")+col("4_6a_1_13")+col("4_6a_1_14")+col("4_6a_1_15")+col("4_6a_1_16")+col("4_6a_1_17")+col("4_6a_1_18")+col("4_6a_1_19")+col("4_6a_1_20")+col("4_6a_1_22")+col("4_6a_1_23")+col("4_6a_1_25")+col("4_6a_1_26")+col("4_6a_1_27")+col("4_6a_1_28")+col("4_6a_1_29")+col("4_6a_1_30")+col("4_6a_1_31"))/29))\
.withColumn("c4_6a_sum", (col("4_6a_1_1")+col("4_6a_1_2")+col("4_6a_1_3")+col("4_6a_1_4")+col("4_6a_1_5")+col("4_6a_1_6")+col("4_6a_1_7")+col("4_6a_1_8")+col("4_6a_1_9")+col("4_6a_1_10")+col("4_6a_1_11")+col("4_6a_1_12")+col("4_6a_1_13")+col("4_6a_1_14")+col("4_6a_1_15")+col("4_6a_1_16")+col("4_6a_1_17")+col("4_6a_1_18")+col("4_6a_1_19")+col("4_6a_1_20")+col("4_6a_1_22")+col("4_6a_1_23")+col("4_6a_1_25")+col("4_6a_1_26")+col("4_6a_1_27")+col("4_6a_1_28")+col("4_6a_1_29")+col("4_6a_1_30")+col("4_6a_1_31")+col("4_6a_3_1")+col("4_6a_3_2")+col("4_6a_3_3")+col("4_6a_3_4")+col("4_6a_3_5")+col("4_6a_3_6")+col("4_6a_3_7")+col("4_6a_3_8")+col("4_6a_3_9")+col("4_6a_3_10")+col("4_6a_3_11")+col("4_6a_3_12")+col("4_6a_3_13")+col("4_6a_3_14")+col("4_6a_3_15")+col("4_6a_3_16")+col("4_6a_3_17")+col("4_6a_3_18")+col("4_6a_3_19")+col("4_6a_3_20")+col("4_6a_3_22")+col("4_6a_3_23")+col("4_6a_3_25")+col("4_6a_3_26")+col("4_6a_3_27")+col("4_6a_3_28")+col("4_6a_3_29")+col("4_6a_3_30")+col("4_6a_3_31")+col("4_6a_5_1")+col("4_6a_5_2")+col("4_6a_5_3")+col("4_6a_5_4")+col("4_6a_5_5")+col("4_6a_5_6")+col("4_6a_5_7")+col("4_6a_5_8")+col("4_6a_5_9")+col("4_6a_5_10")+col("4_6a_5_11")+col("4_6a_5_12")+col("4_6a_5_13")+col("4_6a_5_14")+col("4_6a_5_15")+col("4_6a_5_16")+col("4_6a_5_17")+col("4_6a_5_18")+col("4_6a_5_19")+col("4_6a_5_20")+col("4_6a_5_22")+col("4_6a_5_23")+col("4_6a_5_25")+col("4_6a_5_26")+col("4_6a_5_27")+col("4_6a_5_28")+col("4_6a_5_29")+col("4_6a_5_30")+col("4_6a_5_31")))\
.withColumn("c10_1_sum_1", (col('10_1_1_1')+col('10_1_2_1')+col('10_1_3_1')+col('10_1_4_1')+col('10_1_5_1')+col('10_1_6_1')+col('10_1_7_1')+col('10_1_9_1')+col('10_3_1_1')))\
.withColumn("c10_3_sum_1", (col('10_3_1_1')+col('10_3_2_1')+col('10_3_3_1')+col('10_3_4_1')+col('10_3_5_1')))\
.withColumn("c8_1_sum_1", (col('8_1_1_1')+col('8_1_11_1')+col('8_1_2_1')+col('8_1_3_1')+col('8_1_4_1')+col('8_1_5_1')+col('8_1_6_1')+col('8_1_7_1')+col('8_1_8_1')+col('8_1_9_1')))\
.withColumn("cyes_no_sum", (col('c2_0_0')+col('c3_2_0')+col('c4_0_0')+col('c4_9_1_1')+col('c5_5_0')+col('c5_0_0')+col('c6_2_0')+col('c7_0_0')+col('c7_7_0')+col('c8_0_0')+col('c2_0b_7')))\
.withColumn("c5_0_percentage_achived", ((col('5_0a_11')+col('5_0b_10')+col('5_0c_13')+col('5_0d_11'))/4))\
.withColumn("c5_0_percentage_reduction", ((col('5_0a_8')+col('5_0c_10')+col('5_0d_10'))/3))\
.withColumn("c5_0_base_year_emi", ((col('5_0a_7')+col('5_0c_9')+col('5_0d_7'))/3))\
.withColumn("c4_4_0_sum", (col('c4_4_0_1')+col('c4_4_0_2')+col('c4_4_0_3')+col('c4_4_0_4')+col('c4_4_0_5')+col('c4_4_0_6')+col('c4_4_0_7')))\
.withColumn("c14_0_0_sum", (col('c14_0_0_1')+col('c14_0_0_2')+col('c14_0_0_3')+col('c14_0_0_4')+col('c14_0_0_5')))\
.withColumn("c2_1_3_avg", ((col('c2_1_3_1')+col('c2_1_3_2')+col('c2_1_3_3'))/3))\
.withColumn("c2_1_4_avg", ((col('c2_1_4_1')+col('c2_1_4_2')+col('c2_1_4_3'))/3))\
.withColumn("c2_1_high", (col('c2_1_3_1')+col('c2_1_4_1')))\
.withColumn("c2_1_medium", (col('c2_1_3_2')+col('c2_1_4_2')))\
.withColumn("c2_1_low", (col('c2_1_3_3')+col('c2_1_4_3')))\
.withColumn("c2_1_11_sum", ((col('c2_1_11_1')+col('c2_1_11_2')+col('c2_1_11_3'))/3))\
.withColumn("c2_1_1_sum", (col('c2_1_1_1') +col('c2_1_1_2') +col('c2_1_1_3') +col('c2_1_1_4') +col('c2_1_1_5') +col('c2_1_1_6') +col('c2_1_1_7') +col('c2_1_1_8') +col('c2_1_1_9') +col('c2_1_1_10') +col('c2_1_1_11')))\
.withColumn("c2_1_6_sum", (col('c2_1_6_1') +col('c2_1_6_2') +col('c2_1_6_3') +col('c2_1_6_4') +col('c2_1_6_5') +col('c2_1_6_66') +col('c2_1_6_7') +col('c2_1_6_8') +col('c2_1_6_9') +col('c2_1_6_10') +col('c2_1_6_11') +col('c2_1_6_12') +col('c2_1_6_13') +col('c2_1_6_14') +col('c2_1_6_15') +col('c2_1_6_16') +col('c2_1_6_17')))\
.withColumn("c2_1_7_sum", (col('c2_1_7_1') +col('c2_1_7_2') +col('c2_1_7_3') +col('c2_1_7_4') +col('c2_1_7_8') +col('c2_1_7_9') +col('c2_1_7_10') +col('c2_1_7_11') +col('c2_1_7_12') +col('c2_1_7_13') +col('c2_1_7_14')))\
.withColumn("c3_0_2_sum", (col('c3_0_2_1') +col('c3_0_2_2') +col('c3_0_2_3') +col('c3_0_2_4') +col('c3_0_2_5') +col('c3_0_2_6') +col('c3_0_2_7') +col('c3_0_2_8') +col('c3_0_2_9') +col('c3_0_2_10') +col('c3_0_2_11') +col('c3_0_2_12')))\
.withColumn("c2_2_1_sum", (col('c2_2_1_1') +col('c2_2_1_2') +col('c2_2_1_3') +col('c2_2_1_4') +col('c2_2_1_5') +col('c2_2_1_6') +col('c2_2_1_7') +col('c2_2_1_8') +col('c2_2_1_9') +col('c2_2_1_10')))\
.withColumn("c6_2a_1_sum", (col('c6_2a_1_1') +col('c6_2a_1_2') +col('c6_2a_1_3') +col('c6_2a_1_4') +col('c6_2a_1_5') +col('c6_2a_1_6') +col('c6_2a_1_7') +col('c6_2a_1_8') +col('c6_2a_1_9') +col('c6_2a_1_10') +col('c6_2a_1_11') +col('c6_2a_1_12')))\
.withColumn("c6_0_1_sum", (col('c6_0_1_1') +col('c6_0_1_2') +col('c6_0_1_3') +col('c6_0_1_4') +col('c6_0_1_5') +col('c6_0_1_6') +col('c6_0_1_7') +col('c6_0_1_8') +col('c6_0_1_9') +col('c6_0_1_10') +col('c6_0_1_11') +col('c6_0_1_12') +col('c6_0_1_13') +col('c6_0_1_14') +col('c6_0_1_15') +col('c6_0_1_16') +col('c6_0_1_17') +col('c6_0_1_18') +col('c6_0_1_19')))\
.withColumn("c2_1_8_avg", ((col('c2_1_8_1') +col('c2_1_8_2') +col('c2_1_8_3'))/3))\
.withColumn("c2_1_9_avg", ((col('c2_1_9_1') +col('c2_1_9_2') +col('c2_1_9_3'))/3))

# COMMAND ----------


df_responses_piv_q2 = df_responses_piv_q2.replace(0, None,  subset=['c4_9_1_1', 'c6_2_0', 'c7_0_0', 'c7_7_0','c8_0_0', '10_1_1_1', 	'10_1_2_1', 	'10_1_3_1', 	'10_1_4_1', 	'10_1_5_1', 	'10_1_6_1', 	'10_1_7_1', 	'10_1_9_1', 	'10_3_1_1', 	'10_3_1_2', 	'10_3_1_3', 	'10_3_1_4', 	'10_3_1_5', 	'10_3_2_1', 	'10_3_2_2', 	'10_3_2_3', 	'10_3_2_4', 	'10_3_2_5', 	'10_3_3_1', 	'10_3_3_2', 	'10_3_3_3', 	'10_3_3_4', 	'10_3_3_5', 	'10_3_4_1', 	'10_3_4_2', 	'10_3_4_3', 	'10_3_4_4', 	'10_3_4_5', 	'10_3_5_1', 	'10_3_5_2', 	'10_3_5_3', 	'10_3_5_4', 	'10_3_5_5', '13_3_1_1', 	'13_3_1_2', 	'13_3_1_3', 	'13_3_1_4', 	'13_3_1_5', 	'13_3_1_6', 		'4_6a_1_1', '4_6a_1_10', '4_6a_1_11','4_6a_1_12', 	'4_6a_1_13', 	'4_6a_1_14', 	'4_6a_1_15', 	'4_6a_1_16', 	'4_6a_1_17', 	'4_6a_1_18', 	'4_6a_1_19', 	'4_6a_1_2', 	'4_6a_1_20', 	'4_6a_1_22', 	'4_6a_1_23', 	'4_6a_1_25', 	'4_6a_1_26', 	'4_6a_1_27', 	'4_6a_1_28', 	'4_6a_1_29', 	'4_6a_1_3', 	'4_6a_1_30', 	'4_6a_1_31', 	'4_6a_1_4', 	'4_6a_1_5', 	'4_6a_1_6', 	'4_6a_1_7', 	'4_6a_1_8', 	'4_6a_1_9', 	'4_6a_3_1', 	'4_6a_3_10', 	'4_6a_3_11', 	'4_6a_3_12', 	'4_6a_3_13', 	'4_6a_3_14', 	'4_6a_3_15', 	'4_6a_3_16', 	'4_6a_3_17', 	'4_6a_3_18', 	'4_6a_3_19', 	'4_6a_3_2', 	'4_6a_3_20', 	'4_6a_3_22', 	'4_6a_3_23', 	'4_6a_3_25', 	'4_6a_3_26', 	'4_6a_3_27', 	'4_6a_3_28', 	'4_6a_3_29', 	'4_6a_3_3', 	'4_6a_3_30', 	'4_6a_3_31', 	'4_6a_3_4', 	'4_6a_3_5', 	'4_6a_3_6', 	'4_6a_3_7', 	'4_6a_3_8', 	'4_6a_3_9', 	'4_6a_5_1', 	'4_6a_5_10', 	'4_6a_5_11', 	'4_6a_5_12', 	'4_6a_5_13', 	'4_6a_5_14', 	'4_6a_5_15', 	'4_6a_5_16', 	'4_6a_5_17', 	'4_6a_5_18', 	'4_6a_5_19', 	'4_6a_5_2', 	'4_6a_5_20', 	'4_6a_5_22', 	'4_6a_5_23', 	'4_6a_5_25', 	'4_6a_5_26', 	'4_6a_5_27', 	'4_6a_5_28', 	'4_6a_5_29', 	'4_6a_5_3', 	'4_6a_5_30', 	'4_6a_5_31', 	'4_6a_5_4', 	'4_6a_5_5', 	'4_6a_5_6', 	'4_6a_5_7', 	'4_6a_5_8', 	'4_6a_5_9', '4_6b_1_1', 	'4_6b_1_10', 	'4_6b_1_13', 	'4_6b_1_14', 	'4_6b_1_15', 	'4_6b_1_2', 	'4_6b_1_3', 	'4_6b_1_5', 	'4_6b_1_6', 	'4_6b_1_7', 	'4_6b_1_8', 	'4_6b_1_9', '8_1_1_1', 	'8_1_11_1', 	'8_1_2_1', 	'8_1_3_1', 	'8_1_4_1', 	'8_1_5_1', 	'8_1_6_1', 	'8_1_7_1', 	'8_1_8_1', 	'8_1_9_1', 	'8_2_1_3', '5_0a_11', '5_0b_10', '5_0c_13', '5_0d_11', '5_0a_8', '5_0c_10', '5_0d_10', '5_0a_5', '5_0c_5', '5_0d_5','5_0a_9','5_0b_7','5_0c_11','5_0d_8','5_0a_6','5_0b_5','5_0c_6','5_0d_6','5_0a_7','5_0c_9','5_0d_7', 'c4_4_0_1', 	'c4_4_0_2', 	'c4_4_0_3', 	'c4_4_0_4', 	'c4_4_0_5', 	'c4_4_0_6', 	'c4_4_0_7', 	'c14_0_0_1', 	'c14_0_0_2', 	'c14_0_0_3', 	'c14_0_0_4', 	'c14_0_0_5', 	'c2_1_3_1', 	'c2_1_3_2', 	'c2_1_3_3', 	'c2_1_3_4', 	'c2_1_4_1', 	'c2_1_4_2', 	'c2_1_4_3', 	'c2_1_4_4', 	'c2_1_11_1', 	'c2_1_11_2', 	'c2_1_11_3', 	'c2_1_11_3', 	'c2_1_1_1', 	'c2_1_1_2', 	'c2_1_1_3', 	'c2_1_1_4', 	'c2_1_1_5', 	'c2_1_1_6', 	'c2_1_1_7', 	'c2_1_1_8', 	'c2_1_1_9', 	'c2_1_1_10', 	'c2_1_1_11', 	'c2_1_6_1', 	'c2_1_6_2', 	'c2_1_6_3', 	'c2_1_6_4', 	'c2_1_6_5', 	'c2_1_6_66', 	'c2_1_6_7', 	'c2_1_6_8', 	'c2_1_6_9', 	'c2_1_6_10', 	'c2_1_6_11', 	'c2_1_6_12', 	'c2_1_6_13', 	'c2_1_6_14', 	'c2_1_6_15', 	'c2_1_6_16', 	'c2_1_6_17', 	'c2_1_7_1', 	'c2_1_7_2', 	'c2_1_7_3', 	'c2_1_7_4', 	'c2_1_7_8', 	'c2_1_7_9', 	'c2_1_7_10', 	'c2_1_7_11', 	'c2_1_7_12', 	'c2_1_7_13', 	'c2_1_7_14', 	'c3_0_2_1', 	'c3_0_2_2', 	'c3_0_2_3', 	'c3_0_2_4', 	'c3_0_2_5', 	'c3_0_2_6', 	'c3_0_2_7', 	'c3_0_2_8', 	'c3_0_2_9', 	'c3_0_2_10', 	'c3_0_2_11', 	'c3_0_2_12', 	'c2_2_1_1', 	'c2_2_1_2', 	'c2_2_1_3', 	'c2_2_1_4', 	'c2_2_1_5', 	'c2_2_1_6', 	'c2_2_1_7', 	'c2_2_1_8', 	'c2_2_1_9', 	'c2_2_1_10', 	'c6_2a_1_1', 	'c6_2a_1_2', 	'c6_2a_1_3', 	'c6_2a_1_4', 	'c6_2a_1_5', 	'c6_2a_1_6', 	'c6_2a_1_7', 	'c6_2a_1_8', 	'c6_2a_1_9', 	'c6_2a_1_10', 	'c6_2a_1_11', 	'c6_2a_1_12', 	'c6_0_1_1', 	'c6_0_1_2', 	'c6_0_1_3', 	'c6_0_1_4', 	'c6_0_1_5', 	'c6_0_1_6', 	'c6_0_1_7', 	'c6_0_1_8', 	'c6_0_1_9', 	'c6_0_1_10', 	'c6_0_1_11', 	'c6_0_1_12', 	'c6_0_1_13', 	'c6_0_1_14', 	'c6_0_1_15', 	'c6_0_1_16', 	'c6_0_1_17', 	'c6_0_1_18', 	'c6_0_1_19', 	'c2_2_2_1', 	'c2_2_2_2', 	'c3_0_4_1', 	'c3_0_4_2', 	'c3_0_4_3', 	'c2_1_8_1', 	'c2_1_8_2', 	'c2_1_8_3', 	'c2_1_9_1', 	'c2_1_9_2', 	'c2_1_9_3', "c4_6a_1_sum", 'c4_6a_1_avg', "c4_6a_sum", "c10_1_sum_1", "c10_3_sum_1", "c8_1_sum_1", "c5_0_percentage_achived", "c5_0_percentage_reduction", "c5_0_base_year_emi", 'c4_4_0_sum', 'c14_0_0_sum','c2_1_3_avg','c2_1_4_avg','c2_1_high','c2_1_medium','c2_1_low','c2_1_11_sum','c2_1_1_sum','c2_1_6_sum','c2_1_7_sum','c3_0_2_sum','c2_2_1_sum','c6_2a_1_sum','c6_0_1_sum','c2_1_8_avg','c2_1_9_avg'])


# COMMAND ----------

#df_responses_piv_q2_nulls = df_responses_piv_q2.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_responses_piv_q2.columns])

# COMMAND ----------

#display(df_responses_piv.select('4_4_0').groupby('4_4_0').count())

# COMMAND ----------

#display(df_responses_piv_q2)
colList = ['c4_6a_3_sum', 'c4_6a_3_avg', "4_6a_3_1","4_6a_3_2","4_6a_3_3","4_6a_3_4","4_6a_3_5","4_6a_3_6","4_6a_3_7","4_6a_3_8","4_6a_3_9","4_6a_3_10","4_6a_3_11","4_6a_3_12","4_6a_3_13","4_6a_3_14","4_6a_3_15","4_6a_3_16","4_6a_3_17","4_6a_3_18","4_6a_3_19","4_6a_3_20","4_6a_3_22","4_6a_3_23","4_6a_3_25","4_6a_3_26","4_6a_3_27","4_6a_3_28","4_6a_3_29","4_6a_3_30","4_6a_3_31"]

display(df_responses_piv_q2.select(*[col(c) for c in df_responses_piv_q2.columns if c in colList]))

# COMMAND ----------

display(df_responses_piv_q2)

# COMMAND ----------

#df_account2 = df_account.withColumn('account_year_join', col('account_year'))
columns_account = ['account_year', 'Population', 'population_year_bin', 'first_time']

# Realizar el join entre df1 y df2 usando un inner join
# Selección de las columnas del primer DataFrame (df1.*)
# Selección de las columnas del segundo DataFrame (df2.col2, df2.col3)
# Puedes ajustar el tipo de join según tus necesidades (inner, left, right, etc.)
df_responses_piv_q3 = df_responses_piv_q2.join(df_account.select(*columns_account), on='account_year', how='left')


# COMMAND ----------

# DBTITLE 1,Nulls dataframe
null_counts = df_responses_piv_q3.select([sum(col(column).isNull().cast("int")).alias(column) for column in df_responses_piv_q3.columns])
from pyspark.sql.functions import expr
null_counts2 = null_counts.selectExpr("stack(528, 'account_year', account_year,	'account', account,	'year', year,	'0_5_1_1', 0_5_1_1,	'0_5_2_1', 0_5_2_1,	'0_5_3_1', 0_5_3_1,	'0_6_1_1', 0_6_1_1,	'10_1_1_1', 10_1_1_1,	'10_1_2_1', 10_1_2_1,	'10_1_3_1', 10_1_3_1,	'10_1_4_1', 10_1_4_1,	'10_1_5_1', 10_1_5_1,	'10_1_6_1', 10_1_6_1,	'10_1_7_1', 10_1_7_1,	'10_1_9_1', 10_1_9_1,	'10_3_1_1', 10_3_1_1,	'10_3_1_2', 10_3_1_2,	'10_3_1_3', 10_3_1_3,	'10_3_1_4', 10_3_1_4,	'10_3_1_5', 10_3_1_5,	'10_3_2_1', 10_3_2_1,	'10_3_2_2', 10_3_2_2,	'10_3_2_3', 10_3_2_3,	'10_3_2_4', 10_3_2_4,	'10_3_2_5', 10_3_2_5,	'10_3_3_1', 10_3_3_1,	'10_3_3_2', 10_3_3_2,	'10_3_3_3', 10_3_3_3,	'10_3_3_4', 10_3_3_4,	'10_3_3_5', 10_3_3_5,	'10_3_4_1', 10_3_4_1,	'10_3_4_2', 10_3_4_2,	'10_3_4_3', 10_3_4_3,	'10_3_4_4', 10_3_4_4,	'10_3_4_5', 10_3_4_5,	'10_3_5_1', 10_3_5_1,	'10_3_5_2', 10_3_5_2,	'10_3_5_3', 10_3_5_3,	'10_3_5_4', 10_3_5_4,	'10_3_5_5', 10_3_5_5,	'11_0_0', 11_0_0,	'13_2_0', 13_2_0,	'13_3_1_1', 13_3_1_1,	'13_3_1_2', 13_3_1_2,	'13_3_1_3', 13_3_1_3,	'13_3_1_4', 13_3_1_4,	'13_3_1_5', 13_3_1_5,	'13_3_1_6', 13_3_1_6,	'14_1_0', 14_1_0,	'14_2a_1', 14_2a_1,	'14_2a_2', 14_2a_2,	'14_2a_3', 14_2a_3,	'14_2a_5', 14_2a_5,	'14_3_2', 14_3_2,	'14_3_4', 14_3_4,	'2_0b_4', 2_0b_4,	'2_0b_6', 2_0b_6,	'2_1_1', 2_1_1,	'2_1_5', 2_1_5,	'2_1_6', 2_1_6,	'2_1_7', 2_1_7,	'2_1_8', 2_1_8,	'2_1_9', 2_1_9,	'2_2_1', 2_2_1,	'2_2_2', 2_2_2,	'2_2_4', 2_2_4,	'3_0_2', 3_0_2,	'3_0_4', 3_0_4,	'3_0_8', 3_0_8,	'3_2a_1', 3_2a_1,	'3_2a_3', 3_2a_3,	'3_2a_5', 3_2a_5,	'3_2a_6', 3_2a_6,	'3_3_1', 3_3_1,	'3_3_3', 3_3_3,	'3_3_4', 3_3_4,	'4_4_0', 4_4_0,	'4_6a_1_1', 4_6a_1_1,	'4_6a_1_10', 4_6a_1_10,	'4_6a_1_11', 4_6a_1_11,	'4_6a_1_12', 4_6a_1_12,	'4_6a_1_13', 4_6a_1_13,	'4_6a_1_14', 4_6a_1_14,	'4_6a_1_15', 4_6a_1_15,	'4_6a_1_16', 4_6a_1_16,	'4_6a_1_17', 4_6a_1_17,	'4_6a_1_18', 4_6a_1_18,	'4_6a_1_19', 4_6a_1_19,	'4_6a_1_2', 4_6a_1_2,	'4_6a_1_20', 4_6a_1_20,	'4_6a_1_22', 4_6a_1_22,	'4_6a_1_23', 4_6a_1_23,	'4_6a_1_25', 4_6a_1_25,	'4_6a_1_26', 4_6a_1_26,	'4_6a_1_27', 4_6a_1_27,	'4_6a_1_28', 4_6a_1_28,	'4_6a_1_29', 4_6a_1_29,	'4_6a_1_3', 4_6a_1_3,	'4_6a_1_30', 4_6a_1_30,	'4_6a_1_31', 4_6a_1_31,	'4_6a_1_4', 4_6a_1_4,	'4_6a_1_5', 4_6a_1_5,	'4_6a_1_6', 4_6a_1_6,	'4_6a_1_7', 4_6a_1_7,	'4_6a_1_8', 4_6a_1_8,	'4_6a_1_9', 4_6a_1_9,	'4_6a_2_1', 4_6a_2_1,	'4_6a_2_10', 4_6a_2_10,	'4_6a_2_11', 4_6a_2_11,	'4_6a_2_12', 4_6a_2_12,	'4_6a_2_13', 4_6a_2_13,	'4_6a_2_14', 4_6a_2_14,	'4_6a_2_15', 4_6a_2_15,	'4_6a_2_16', 4_6a_2_16,	'4_6a_2_17', 4_6a_2_17,	'4_6a_2_18', 4_6a_2_18,	'4_6a_2_19', 4_6a_2_19,	'4_6a_2_2', 4_6a_2_2,	'4_6a_2_20', 4_6a_2_20,	'4_6a_2_22', 4_6a_2_22,	'4_6a_2_23', 4_6a_2_23,	'4_6a_2_25', 4_6a_2_25,	'4_6a_2_26', 4_6a_2_26,	'4_6a_2_27', 4_6a_2_27,	'4_6a_2_28', 4_6a_2_28,	'4_6a_2_29', 4_6a_2_29,	'4_6a_2_3', 4_6a_2_3,	'4_6a_2_30', 4_6a_2_30,	'4_6a_2_31', 4_6a_2_31,	'4_6a_2_4', 4_6a_2_4,	'4_6a_2_5', 4_6a_2_5,	'4_6a_2_6', 4_6a_2_6,	'4_6a_2_7', 4_6a_2_7,	'4_6a_2_8', 4_6a_2_8,	'4_6a_2_9', 4_6a_2_9,	'4_6a_3_1', 4_6a_3_1,	'4_6a_3_10', 4_6a_3_10,	'4_6a_3_11', 4_6a_3_11,	'4_6a_3_12', 4_6a_3_12,	'4_6a_3_13', 4_6a_3_13,	'4_6a_3_14', 4_6a_3_14,	'4_6a_3_15', 4_6a_3_15,	'4_6a_3_16', 4_6a_3_16,	'4_6a_3_17', 4_6a_3_17,	'4_6a_3_18', 4_6a_3_18,	'4_6a_3_19', 4_6a_3_19,	'4_6a_3_2', 4_6a_3_2,	'4_6a_3_20', 4_6a_3_20,	'4_6a_3_22', 4_6a_3_22,	'4_6a_3_23', 4_6a_3_23,	'4_6a_3_25', 4_6a_3_25,	'4_6a_3_26', 4_6a_3_26,	'4_6a_3_27', 4_6a_3_27,	'4_6a_3_28', 4_6a_3_28,	'4_6a_3_29', 4_6a_3_29,	'4_6a_3_3', 4_6a_3_3,	'4_6a_3_30', 4_6a_3_30,	'4_6a_3_31', 4_6a_3_31,	'4_6a_3_4', 4_6a_3_4,	'4_6a_3_5', 4_6a_3_5,	'4_6a_3_6', 4_6a_3_6,	'4_6a_3_7', 4_6a_3_7,	'4_6a_3_8', 4_6a_3_8,	'4_6a_3_9', 4_6a_3_9,	'4_6a_4_1', 4_6a_4_1,	'4_6a_4_10', 4_6a_4_10,	'4_6a_4_11', 4_6a_4_11,	'4_6a_4_12', 4_6a_4_12,	'4_6a_4_13', 4_6a_4_13,	'4_6a_4_14', 4_6a_4_14,	'4_6a_4_15', 4_6a_4_15,	'4_6a_4_16', 4_6a_4_16,	'4_6a_4_17', 4_6a_4_17,	'4_6a_4_18', 4_6a_4_18,	'4_6a_4_19', 4_6a_4_19,	'4_6a_4_2', 4_6a_4_2,	'4_6a_4_20', 4_6a_4_20,	'4_6a_4_22', 4_6a_4_22,	'4_6a_4_23', 4_6a_4_23,	'4_6a_4_25', 4_6a_4_25,	'4_6a_4_26', 4_6a_4_26,	'4_6a_4_27', 4_6a_4_27,	'4_6a_4_28', 4_6a_4_28,	'4_6a_4_29', 4_6a_4_29,	'4_6a_4_3', 4_6a_4_3,	'4_6a_4_30', 4_6a_4_30,	'4_6a_4_31', 4_6a_4_31,	'4_6a_4_4', 4_6a_4_4,	'4_6a_4_5', 4_6a_4_5,	'4_6a_4_6', 4_6a_4_6,	'4_6a_4_7', 4_6a_4_7,	'4_6a_4_8', 4_6a_4_8,	'4_6a_4_9', 4_6a_4_9,	'4_6a_5_1', 4_6a_5_1,	'4_6a_5_10', 4_6a_5_10,	'4_6a_5_11', 4_6a_5_11,	'4_6a_5_12', 4_6a_5_12,	'4_6a_5_13', 4_6a_5_13,	'4_6a_5_14', 4_6a_5_14,	'4_6a_5_15', 4_6a_5_15,	'4_6a_5_16', 4_6a_5_16,	'4_6a_5_17', 4_6a_5_17,	'4_6a_5_18', 4_6a_5_18,	'4_6a_5_19', 4_6a_5_19,	'4_6a_5_2', 4_6a_5_2,	'4_6a_5_20', 4_6a_5_20,	'4_6a_5_22', 4_6a_5_22,	'4_6a_5_23', 4_6a_5_23,	'4_6a_5_25', 4_6a_5_25,	'4_6a_5_26', 4_6a_5_26,	'4_6a_5_27', 4_6a_5_27,	'4_6a_5_28', 4_6a_5_28,	'4_6a_5_29', 4_6a_5_29,	'4_6a_5_3', 4_6a_5_3,	'4_6a_5_30', 4_6a_5_30,	'4_6a_5_31', 4_6a_5_31,	'4_6a_5_4', 4_6a_5_4,	'4_6a_5_5', 4_6a_5_5,	'4_6a_5_6', 4_6a_5_6,	'4_6a_5_7', 4_6a_5_7,	'4_6a_5_8', 4_6a_5_8,	'4_6a_5_9', 4_6a_5_9,	'4_6a_6_1', 4_6a_6_1,	'4_6a_6_10', 4_6a_6_10,	'4_6a_6_11', 4_6a_6_11,	'4_6a_6_12', 4_6a_6_12,	'4_6a_6_13', 4_6a_6_13,	'4_6a_6_14', 4_6a_6_14,	'4_6a_6_15', 4_6a_6_15,	'4_6a_6_16', 4_6a_6_16,	'4_6a_6_17', 4_6a_6_17,	'4_6a_6_18', 4_6a_6_18,	'4_6a_6_19', 4_6a_6_19,	'4_6a_6_2', 4_6a_6_2,	'4_6a_6_20', 4_6a_6_20,	'4_6a_6_22', 4_6a_6_22,	'4_6a_6_23', 4_6a_6_23,	'4_6a_6_25', 4_6a_6_25,	'4_6a_6_26', 4_6a_6_26,	'4_6a_6_27', 4_6a_6_27,	'4_6a_6_28', 4_6a_6_28,	'4_6a_6_29', 4_6a_6_29,	'4_6a_6_3', 4_6a_6_3,	'4_6a_6_30', 4_6a_6_30,	'4_6a_6_31', 4_6a_6_31,	'4_6a_6_4', 4_6a_6_4,	'4_6a_6_5', 4_6a_6_5,	'4_6a_6_6', 4_6a_6_6,	'4_6a_6_7', 4_6a_6_7,	'4_6a_6_8', 4_6a_6_8,	'4_6a_6_9', 4_6a_6_9,	'4_6b_1_1', 4_6b_1_1,	'4_6b_1_10', 4_6b_1_10,	'4_6b_1_11', 4_6b_1_11,	'4_6b_1_12', 4_6b_1_12,	'4_6b_1_13', 4_6b_1_13,	'4_6b_1_14', 4_6b_1_14,	'4_6b_1_15', 4_6b_1_15,	'4_6b_1_16', 4_6b_1_16,	'4_6b_1_17', 4_6b_1_17,	'4_6b_1_2', 4_6b_1_2,	'4_6b_1_3', 4_6b_1_3,	'4_6b_1_4', 4_6b_1_4,	'4_6b_1_5', 4_6b_1_5,	'4_6b_1_6', 4_6b_1_6,	'4_6b_1_7', 4_6b_1_7,	'4_6b_1_8', 4_6b_1_8,	'4_6b_1_9', 4_6b_1_9,	'4_6c_13_1', 4_6c_13_1,	'4_6c_1_1', 4_6c_1_1,	'4_6c_3_1', 4_6c_3_1,	'4_6c_8_1', 4_6c_8_1,	'4_6d_1', 4_6d_1,	'4_6d_2', 4_6d_2,	'4_6d_3', 4_6d_3,	'4_6d_4', 4_6d_4,	'5_0a_1', 5_0a_1,	'5_0a_11', 5_0a_11,	'5_0a_3', 5_0a_3,	'5_0a_5', 5_0a_5,	'5_0a_6', 5_0a_6,	'5_0a_7', 5_0a_7,	'5_0a_8', 5_0a_8,	'5_0a_9', 5_0a_9,	'5_0b_1', 5_0b_1,	'5_0b_10', 5_0b_10,	'5_0b_3', 5_0b_3,	'5_0b_5', 5_0b_5,	'5_0b_7', 5_0b_7,	'5_0b_8', 5_0b_8,	'5_0b_9', 5_0b_9,	'5_0c_1', 5_0c_1,	'5_0c_10', 5_0c_10,	'5_0c_11', 5_0c_11,	'5_0c_12', 5_0c_12,	'5_0c_13', 5_0c_13,	'5_0c_3', 5_0c_3,	'5_0c_5', 5_0c_5,	'5_0c_6', 5_0c_6,	'5_0c_7', 5_0c_7,	'5_0c_8', 5_0c_8,	'5_0c_9', 5_0c_9,	'5_0d_1', 5_0d_1,	'5_0d_10', 5_0d_10,	'5_0d_11', 5_0d_11,	'5_0d_3', 5_0d_3,	'5_0d_5', 5_0d_5,	'5_0d_6', 5_0d_6,	'5_0d_7', 5_0d_7,	'5_0d_8', 5_0d_8,	'5_0d_9', 5_0d_9,	'5_4_7', 5_4_7,	'5_4_8', 5_4_8,	'6_0_2', 6_0_2,	'6_13_1_1', 6_13_1_1,	'6_2a_1', 6_2a_1,	'6_2a_3', 6_2a_3,	'6_5_7', 6_5_7,	'6_5_9', 6_5_9,	'7_2_0', 7_2_0,	'7_3_1_1', 7_3_1_1,	'7_4_0', 7_4_0,	'7_6_2_1', 7_6_2_1,	'7_6_3_1', 7_6_3_1,	'7_6_4_1', 7_6_4_1,	'7_7a_2', 7_7a_2,	'8_0a_3', 8_0a_3,	'8_0a_4', 8_0a_4,	'8_0a_6', 8_0a_6,	'8_0a_7', 8_0a_7,	'8_0a_9', 8_0a_9,	'8_1_11_1', 8_1_11_1,	'8_1_1_1', 8_1_1_1,	'8_1_2_1', 8_1_2_1,	'8_1_3_1', 8_1_3_1,	'8_1_4_1', 8_1_4_1,	'8_1_5_1', 8_1_5_1,	'8_1_6_1', 8_1_6_1,	'8_1_7_1', 8_1_7_1,	'8_1_8_1', 8_1_8_1,	'8_1_9_1', 8_1_9_1,	'8_2_1_3', 8_2_1_3,	'9_1_1_1', 9_1_1_1,	'9_1_1_2', 9_1_1_2,	'9_1_1_3', 9_1_1_3,	'9_1_1_4', 9_1_1_4,	'9_1_3_1', 9_1_3_1,	'9_1_3_2', 9_1_3_2,	'9_1_3_3', 9_1_3_3,	'9_1_3_4', 9_1_3_4,	'c2_0_0', c2_0_0,	'c3_2_0', c3_2_0,	'c4_0_0', c4_0_0,	'c4_9_1_1', c4_9_1_1,	'c5_5_0', c5_5_0,	'c7_0_0', c7_0_0,	'c7_7_0', c7_7_0,	'c8_0_0', c8_0_0,	'c2_0b_2', c2_0b_2,	'c2_0b_7', c2_0b_7,	'c6_2_0', c6_2_0,	'c5_0_0', c5_0_0,	'c0_1_1_1', c0_1_1_1,	'c5_0_sector', c5_0_sector,	'c6_5_3', c6_5_3,	'c6_5_4', c6_5_4,	'c3_2a_9', c3_2a_9,	'csector', csector,	'cboundary', cboundary,	'c5_0_base_year', c5_0_base_year,	'c5_0_target_year', c5_0_target_year,	'c5_0_target_year_set', c5_0_target_year_set,	'c4_4_0_1', c4_4_0_1,	'c4_4_0_2', c4_4_0_2,	'c4_4_0_3', c4_4_0_3,	'c4_4_0_4', c4_4_0_4,	'c4_4_0_5', c4_4_0_5,	'c4_4_0_6', c4_4_0_6,	'c4_4_0_7', c4_4_0_7,	'c14_0_0_1', c14_0_0_1,	'c14_0_0_2', c14_0_0_2,	'c14_0_0_3', c14_0_0_3,	'c14_0_0_4', c14_0_0_4,	'c14_0_0_5', c14_0_0_5,	'c2_1_3_1', c2_1_3_1,	'c2_1_3_2', c2_1_3_2,	'c2_1_3_3', c2_1_3_3,	'c2_1_3_4', c2_1_3_4,	'c2_1_4_1', c2_1_4_1,	'c2_1_4_2', c2_1_4_2,	'c2_1_4_3', c2_1_4_3,	'c2_1_4_4', c2_1_4_4,	'c2_1_11_1', c2_1_11_1,	'c2_1_11_2', c2_1_11_2,	'c2_1_11_3', c2_1_11_3,	'c2_1_1_1', c2_1_1_1,	'c2_1_1_2', c2_1_1_2,	'c2_1_1_3', c2_1_1_3,	'c2_1_1_4', c2_1_1_4,	'c2_1_1_5', c2_1_1_5,	'c2_1_1_6', c2_1_1_6,	'c2_1_1_7', c2_1_1_7,	'c2_1_1_8', c2_1_1_8,	'c2_1_1_9', c2_1_1_9,	'c2_1_1_10', c2_1_1_10,	'c2_1_1_11', c2_1_1_11,	'c2_1_6_1', c2_1_6_1,	'c2_1_6_2', c2_1_6_2,	'c2_1_6_3', c2_1_6_3,	'c2_1_6_4', c2_1_6_4,	'c2_1_6_5', c2_1_6_5,	'c2_1_6_66', c2_1_6_66,	'c2_1_6_7', c2_1_6_7,	'c2_1_6_8', c2_1_6_8,	'c2_1_6_9', c2_1_6_9,	'c2_1_6_10', c2_1_6_10,	'c2_1_6_11', c2_1_6_11,	'c2_1_6_12', c2_1_6_12,	'c2_1_6_13', c2_1_6_13,	'c2_1_6_14', c2_1_6_14,	'c2_1_6_15', c2_1_6_15,	'c2_1_6_16', c2_1_6_16,	'c2_1_6_17', c2_1_6_17,	'c2_1_7_1', c2_1_7_1,	'c2_1_7_2', c2_1_7_2,	'c2_1_7_3', c2_1_7_3,	'c2_1_7_4', c2_1_7_4,	'c2_1_7_8', c2_1_7_8,	'c2_1_7_9', c2_1_7_9,	'c2_1_7_10', c2_1_7_10,	'c2_1_7_11', c2_1_7_11,	'c2_1_7_12', c2_1_7_12,	'c2_1_7_13', c2_1_7_13,	'c2_1_7_14', c2_1_7_14,	'c3_0_2_1', c3_0_2_1,	'c3_0_2_2', c3_0_2_2,	'c3_0_2_3', c3_0_2_3,	'c3_0_2_4', c3_0_2_4,	'c3_0_2_5', c3_0_2_5,	'c3_0_2_6', c3_0_2_6,	'c3_0_2_7', c3_0_2_7,	'c3_0_2_8', c3_0_2_8,	'c3_0_2_9', c3_0_2_9,	'c3_0_2_10', c3_0_2_10,	'c3_0_2_11', c3_0_2_11,	'c3_0_2_12', c3_0_2_12,	'c2_2_1_1', c2_2_1_1,	'c2_2_1_2', c2_2_1_2,	'c2_2_1_3', c2_2_1_3,	'c2_2_1_4', c2_2_1_4,	'c2_2_1_5', c2_2_1_5,	'c2_2_1_6', c2_2_1_6,	'c2_2_1_7', c2_2_1_7,	'c2_2_1_8', c2_2_1_8,	'c2_2_1_9', c2_2_1_9,	'c2_2_1_10', c2_2_1_10,	'c6_2a_1_1', c6_2a_1_1,	'c6_2a_1_2', c6_2a_1_2,	'c6_2a_1_3', c6_2a_1_3,	'c6_2a_1_4', c6_2a_1_4,	'c6_2a_1_5', c6_2a_1_5,	'c6_2a_1_6', c6_2a_1_6,	'c6_2a_1_7', c6_2a_1_7,	'c6_2a_1_8', c6_2a_1_8,	'c6_2a_1_9', c6_2a_1_9,	'c6_2a_1_10', c6_2a_1_10,	'c6_2a_1_11', c6_2a_1_11,	'c6_2a_1_12', c6_2a_1_12,	'c6_0_1_1', c6_0_1_1,	'c6_0_1_2', c6_0_1_2,	'c6_0_1_3', c6_0_1_3,	'c6_0_1_4', c6_0_1_4,	'c6_0_1_5', c6_0_1_5,	'c6_0_1_6', c6_0_1_6,	'c6_0_1_7', c6_0_1_7,	'c6_0_1_8', c6_0_1_8,	'c6_0_1_9', c6_0_1_9,	'c6_0_1_10', c6_0_1_10,	'c6_0_1_11', c6_0_1_11,	'c6_0_1_12', c6_0_1_12,	'c6_0_1_13', c6_0_1_13,	'c6_0_1_14', c6_0_1_14,	'c6_0_1_15', c6_0_1_15,	'c6_0_1_16', c6_0_1_16,	'c6_0_1_17', c6_0_1_17,	'c6_0_1_18', c6_0_1_18,	'c6_0_1_19', c6_0_1_19,	'c2_2_2_1', c2_2_2_1,	'c2_2_2_2', c2_2_2_2,	'c3_0_4_1', c3_0_4_1,	'c3_0_4_2', c3_0_4_2,	'c3_0_4_3', c3_0_4_3,	'c2_1_8_1', c2_1_8_1,	'c2_1_8_2', c2_1_8_2,	'c2_1_8_3', c2_1_8_3,	'c2_1_9_1', c2_1_9_1,	'c2_1_9_2', c2_1_9_2,	'c2_1_9_3', c2_1_9_3,	'c4_6a_1_sum', c4_6a_1_sum,	'c4_6a_1_avg', c4_6a_1_avg,	'c4_6a_sum', c4_6a_sum,	'c10_1_sum_1', c10_1_sum_1,	'c10_3_sum_1', c10_3_sum_1,	'c8_1_sum_1', c8_1_sum_1,	'cyes_no_sum', cyes_no_sum,	'c5_0_percentage_achived', c5_0_percentage_achived,	'c5_0_percentage_reduction', c5_0_percentage_reduction,	'c5_0_base_year_emi', c5_0_base_year_emi,	'c4_4_0_sum', c4_4_0_sum,	'c14_0_0_sum', c14_0_0_sum,	'c2_1_3_avg', c2_1_3_avg,	'c2_1_4_avg', c2_1_4_avg,	'c2_1_high', c2_1_high,	'c2_1_medium', c2_1_medium,	'c2_1_low', c2_1_low,	'c2_1_11_sum', c2_1_11_sum,	'c2_1_1_sum', c2_1_1_sum,	'c2_1_6_sum', c2_1_6_sum,	'c2_1_7_sum', c2_1_7_sum,	'c3_0_2_sum', c3_0_2_sum,	'c2_2_1_sum', c2_2_1_sum,	'c6_2a_1_sum', c6_2a_1_sum,	'c6_0_1_sum', c6_0_1_sum,	'c2_1_8_avg', c2_1_8_avg,	'c2_1_9_avg', c2_1_9_avg,	'Population', Population,	'population_year_bin', population_year_bin,	'first_time', first_time) as (columna, valor)")

null_counts2 = null_counts2.withColumn('percentage', (col('valor') / 4134)*100)
# Mostrar el resultado
display(null_counts2)

# COMMAND ----------

# Select columns to keep

from pyspark.sql.functions import collect_list

column_to_keep_nulls_40perc = null_counts2.filter((col("percentage") <= 45)).select(collect_list("columna")).first()[0]
column_to_keep_nulls_others = ['14_1_0', '3_2a_5', 'c4_4_0','c3_0_2','c3_0_4','c6_5_3','c6_5_4','c8_1_sum_1','c2_0b_7','c5_0_target_year','c10_1_sum_1','c6_2a_1','c3_2a_9','c5_0_base_year','c5_0_target_year_set','c5_0_0','c5_0_base_year_emi','c10_3_sum_1','c4_6a_sum','c4_6a_1_avg','c4_6a_1_sum','c5_0_percentage_achived', 'csector', 'c4_4_0_sum', 'c14_0_0_sum', 'c2_1_3_avg', 'c2_1_4_avg',	'c2_1_11_sum',	'c2_1_1_sum', 'c2_1_6_sum', 'c2_1_7_sum', 'c3_0_2_sum', 	'c2_2_1_sum', 'c6_2a_1_sum', 'c6_0_1_sum', 'c2_1_8_avg', 'c2_1_9_avg', 'Population', 'population_year_bin', 'first_time']

column_to_keep_nulls = column_to_keep_nulls_40perc + column_to_keep_nulls_others

#Get columns with > 70% of null values
#column_to_drop_nulls = null_counts2.filter((col("percentage") > 70)).select(collect_list("columna")).first()[0]

# Remove keep valid columns
df_responses_piv_q4 = df_responses_piv_q3.select(*[col(c) for c in df_responses_piv_q3.columns if c in column_to_keep_nulls])

# Remove free text columns '6_0_2', '2_1_5', '2_2_4'
columns_to_drop = ('6_0_2', '2_1_5', '2_2_4', '0_5_2_1', '0_5_1_1') # '0_5_2_1', '0_5_1_1' repetidas
df_responses_piv_q4 = df_responses_piv_q4.drop(*columns_to_drop)

print(column_to_keep_nulls)



# COMMAND ----------

# Original Shape
def sparkShape(df):
    return (df.count(), len(df.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape

print(df_responses_piv_q3.shape())
print(df_responses_piv_q4.shape())

# COMMAND ----------

display(df_responses_piv_q4)

# COMMAND ----------

# Save as parquet file
df_responses_piv_q4.write.parquet("/FileStore/df_responses_final.parquet", 
                            mode="overwrite")

# COMMAND ----------

#REVISAR, AREA, EMISSIONES, TRANSPORT Y ELECT CONSUMED PARA VERIFICAR UNIDADES CORRECTAS
# PCA COMO ESTAN
# REEMPLAZAR OUTLIERS BASADO EN IQR, CON MEDIA

# COMMAND ----------

display(df_responses_piv_q3.groupby('c4_6a_1_sum').count())

# COMMAND ----------

display(df_responses_piv_q3.groupby('c4_6a_1_avg').count())

# COMMAND ----------

display(df_responses_piv_q3.groupby('c4_6a_sum').count())

# COMMAND ----------

display(df_responses_piv_q3.groupby('0_5_3_1').count())

# Revisar Unidades

# COMMAND ----------

display(df_responses_piv_q3.groupby('0_5_1_1').count())

# Revisar Unidades

# COMMAND ----------

display(df_responses_piv_q3.groupby('0_6_1_1').count())
# Revisar Unidades

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


#CONTAR VALORES <> DE 0 = NULLS
# RELLENAR CON MEDIA?


# COMMAND ----------

display(df_responses_piv_q2.select(col("4_6a_2_4")).groupby("4_6a_2_4").count())

# COMMAND ----------

df_responses2 = df_responses.withColumn("6_1_1_7", when(col("6_1_1") == "Increase energy efficiency of buildings (residential buildings)", "Yes").otherwise(None))\

.withColumn("6_1_1_8", when(col("6_1_1") == "Increase energy efficiency of buildings (all buildings)", "Yes").otherwise(None))\
.withColumn("6_1_1_9", when(col("6_1_1").like("%renewable%"), "Yes").otherwise(None))\
.withColumn("6_1_1_10", when(col("6_1_1").like("%renewable%"), col("6_1_1")).otherwise(None))\

.withColumn("6_1_1_8", when(col("6_1_1") == "Increase energy efficiency of buildings (all buildings)", "Yes").otherwise(None))\
.withColumn("6_1_1_9", when(col("6_1_1").like("%renewable%"), "Yes").otherwise(None))\
.withColumn("6_1_1_10", when(col("6_1_1").like("%renewable%"), col("6_1_1")).otherwise(None))\
.withColumn("6_1_1_11", when(col("6_1_1").like("%renewable%"), col("6_1_5")).otherwise(None))\
.withColumn("6_1_1_12", when(col("6_1_1").like("%renewable%") | col("6_1_6").like('%Percentage (%)%'), col("6_1_8")).otherwise(None))\
.withColumn("6_1_1_13", when(col("6_1_1").like("%renewable%"), col("6_1_9")).otherwise(None))\
.withColumn("6_1_1_14", when(col("6_1_1").like("%renewable%"), col("6_1_12")).otherwise(None))

# COMMAND ----------

# Original Shape
def sparkShape(df):
    return (df.count(), len(df.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape

#display(df_responses.shape())
display(df_responses_piv_q.shape())

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Count unique values for each column
distinct_responses = df_responses_piv.agg(*[countDistinct(col).alias(col) for col in df_responses_piv.columns])

display(distinct_responses)

# COMMAND ----------

# Set datatypes
from pyspark.sql.functions import col

data_types = {
    "account_year": "integer",
    "Account Number": "string",
    "year": "double",
    # Agrega más columnas y tipos de datos según sea necesario
}

for column, data_type in data_types.items():
    df_responses_piv = df_responses_piv.withColumn(column, col(column).cast(data_type))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,6) Create Account Dataframe
# Create dataframe with account related columns
df_accounts = df_cities_all.groupby("Account Number", 'Organization', "Country", "CDP Region").count()

# COMMAND ----------

# Check the files properly loaded
dbutils.fs.ls('dbfs:/FileStore/')

# COMMAND ----------

# Input data from Databricks FileStore
df_account_2018 = spark.read.csv('dbfs:/FileStore/2018_Cities_Disclosing_to_CDP.csv', header='true')
df_account_2019 = spark.read.csv('dbfs:/FileStore/2019_Cities_Disclosing_to_CDP.csv', header='true')
df_account_2020 = spark.read.csv('dbfs:/FileStore/2020_Cities_Disclosing_to_CDP.csv', header='true')
df_account_2021 = spark.read.csv('dbfs:/FileStore/2021_Cities_Disclosing_to_CDP.csv', header='true')
df_account_2022 = spark.read.csv('dbfs:/FileStore/2022_Cities_Disclosing_to_CDP.csv', header='true')

# COMMAND ----------

# Original Shape
def sparkShape(df):
    return (df.count(), len(df.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape

print(df_account_2018.shape())
print(df_account_2019.shape())
print(df_account_2020.shape())
print(df_account_2021.shape())
print(df_account_2022.shape())

# COMMAND ----------

# Fix 2022 dataset
columns_to_drop4 = ["Number of times reporting", "View Response"]
df_account_2022 = df_account_2022.drop(*columns_to_drop4)\
  .filter(col('Questionnaire') == 'Cities 2022')\
  .withColumnRenamed('Questionnaire','Year Reported to CDP')\
  .withColumnRenamed('Organization Number','Account Number')\
  .withColumnRenamed('Organization Name','Organization')\
  .withColumnRenamed('First time reporting','First Time Discloser')\
  .withColumnRenamed('Organization Name','Organization')

# Remove other columns and unify colum names
columns_to_drop5 = ["Reporting Authority", "Access", "Last update"]

df_account_2018 = df_account_2018.drop(*columns_to_drop5)
df_account_2019 = df_account_2019.drop(*columns_to_drop5)
df_account_2020 = df_account_2020.drop(*columns_to_drop5)
df_account_2021 = df_account_2021.drop(*columns_to_drop5)
df_account_2022 = df_account_2022.drop(*columns_to_drop5)

# COMMAND ----------

# Append all dataframes
df_account = df_account_2018.union(df_account_2019)\
                               .union(df_account_2020)\
                               .union(df_account_2021)\
                               .union(df_account_2022)

print(df_account.shape())

# COMMAND ----------

# Fix columns and data types
df_account = df_account.withColumn('year', col('Year Reported to CDP').substr(-4,4))\
.withColumn('Population', df_account['Population'].cast(IntegerType()))\
.withColumn("first_time", 
          when((upper(col("First Time Discloser")) == "YES"), 1)
          .when((upper(col("First Time Discloser")) == "NO"), 0)
          .otherwise(None))\
.withColumn("population_year", 
          when((col("Population Year") == "21"), "2010")
          .when((col("Population Year") == "211"), "2011")
          .when((col("Population Year") == "214"), "2014")
          .when((col("Population Year") == "216"), "2016")
          .when((col("Population Year") == "217"), "2017")
          .when((col("Population Year") == "218"), "2018")
          .when((col("Population Year") == "219"), "2019")
          .when((col("Population Year") == "1537") | (col("Population Year") == "19") | (col("Population Year") == "7") | (col("Population Year") == "800"), None)
          .otherwise(col("Population Year")))\
.withColumn('city_fix',
          when(length(col('City')) != None, col('City'))
          .otherwise(col('Organization')))

df_account = df_account.withColumn("population_year", df_account["population_year"].cast(IntegerType()))\
.withColumn("population_year_bin", 
          when((col("population_year") >= 2020), ">2020")
          .when((col("population_year") >= 2015) & (col("population_year") < 2020), "2015-2019")
          .when((col("population_year") >= 2010) & (col("population_year") < 2015), "2010-2014")
          .when((col("population_year") < 2010), "<2010")
          .otherwise(None))\
          .withColumn("account_year", concat_ws('_', df_account['Account Number'], df_account['year']))

# Extract latitud and longitud from City location point
df_account = df_account.withColumn("City Location", regexp_replace("City Location", "POINT \\(|\\)", ""))\
.withColumn("location_latitud", split("City Location", " ").getItem(0).cast(DecimalType()))\
.withColumn("location_longitud", split("City Location", " ").getItem(1).cast(DecimalType()))

# Drop columns
df_account = df_account.drop('Year Reported to CDP', 'First Time Discloser', 'City Location', 'Population Year', 'City', 'Organization')

df_account = df_account.withColumnRenamed('Account Number', 'account')\
.withColumnRenamed('CDP Region', 'region')

# COMMAND ----------

replace_region = {
'United States of America' : 'North America',
'Canada' : 'North America',
'Oceania' : 'Southeast Asia and Oceania',
'Southeast Asia' : 'Southeast Asia and Oceania',
'South Asia' : 'South and West Asia'
}

replace_country = {
'Taiwan, China' : 'Taiwan, Greater China',
"Côte d'Ivoire" : "Cote d'Ivoire"
}

replace_city = {
'Ajuntament de Barcelona' : 'City of Barcelona',
'Stockholms stad' : 'City of Stockholm',
'Helsingin kaupunki' : 'City of Helsinki',
'Oslo Municipality' : 'City of Oslo',
'Oslo kommune' : 'City of Oslo',
'Southend on Sea City Council' : 'Southend on Sea Borough Council',
'MunicÃ­pio de Torres Vedras' : 'Municipality of Torres Vedras',
'Copenhagen Municipality' : 'City of Copenhagen',
'KÃ¸benhavn Kommune' : 'City of Copenhagen',
'Comune di Roma Capitale' : 'City of Rome',
'Roma Capitale' : 'City of Rome',
'Gemeente Amsterdam' : 'City of Amsterdam',
'Municipality of Amsterdam' : 'City of Amsterdam',
'DÃ­mos AthinaÃ­on' : 'City of Athens',
'Municipality of Athens' : 'City of Athens',
'Stadt Basel' : 'City of Basel-Stadt',
'Bundeshauptstadt Berlin' : 'City of Berlin',
'Stadt Heidelberg' : 'City of Heidelberg',
'Ayuntamiento de Madrid' : 'City of Madrid',
'Comune di Milano' : 'City of Milan',
'Ville de Paris' : 'City of Paris',
'Municipality of Rotterdam' : 'Gemeente Rotterdam',
'Miasto StoÅ‚eczne Warszawa' : 'City of Warsaw',
'City of ZÃ¼rich' : 'City of Zurich',
'Stadt ZÃ¼rich' : 'City of Zurich',
'MedellÃ­n' : 'Medellin',
'Municipality of MedellÃ­n' : 'Medellin',
'Comune di Torino' : 'City of Turin',
'Ajuntament de ValÃ¨ncia' : 'Municipality of Valencia',
'Bengaluru' : 'Bangalore',
'Municipality of BelÃ©m' : 'Prefeitura de Belém',
'Prefeitura de BelÃ©m' : 'Prefeitura de Belém',
'Comune di Napoli' : 'City of Naples',
'MunicÃ­pio de Lisboa' : 'City of Lisbon',
'Comune di Venezia' : 'City of Venice',
'Comune di Genova' : 'City of Genoa',
'Comune di Firenze' : 'City of Florence',
'Comune di Ferrara' : 'City of Ferrara',
'City of RÄ«ga' : 'City of Riga',
'Riga City' : 'City of Riga',
'RÄ«gas valstspilsÄ“tas paÅ¡valdÄ«ba' : 'City of Riga',
"Comune dell'Aquila" : "City of L'Aquila",
'Comune di Parma' : 'City of Parma',
'Comune di Padova' : 'City of Padua',
'Comune di Prato' : 'City of Prato',
'GÃ¶teborgs Stad' : 'City of Gothenburg',
'GÃ¶teborgs stad' : 'City of Gothenburg',
'Obshtina Sofia' : 'Capital Municipality of Sofia',
'Sofia Municipality' : 'Capital Municipality of Sofia',
'City of Ljubljana' : 'City Municipality of Ljubljana',
'Mestna obÄina Ljubljana' : 'City Municipality of Ljubljana',
'Grad Zagreb' : 'City of Zagreb',
'City of Hannover' : 'City of Hanover',
'Stadt Mannheim' : 'City of Mannheim',
'Gemeente Den Haag' : 'Municipality of The Hague',
'The Hague' : 'Municipality of The Hague',
'City of MalmÃ¶' : 'City of Malmo',
'MalmÃ¶ Stad' : 'City of Malmo',
'MalmÃ¶ stad' : 'City of Malmo',
'Hwaseong City' : 'City of Hwaseong',
'Hwaseong Metropolitan Government' : 'City of Hwaseong',
'Yeosu City' : 'City of Yeosu',
'Yeosu Metropolitan Government' : 'City of Yeosu',
'Ayuntamiento de Vitoria-Gasteiz' : 'Municipality of Vitoria-Gasteiz',
'Ayuntamiento de Zaragoza' : 'City of Zaragoza',
'Municipality of Saragossa (Spain)' : 'City of Zaragoza',
'MunicÃ­pio do Porto' : 'City of Porto',
'Kyoto' : 'City of Kyoto',
'Kyoto City' : 'City of Kyoto',
'Turun kaupunki' : 'City of Turku',
"MÃ©tropole Nice CÃ´te d'Azur" : 'Metropole de Nice',
'MÃ©tropole de Nice' : 'Metropole de Nice',
'Gobierno AutÃ³nomo Municipal de la Paz' : 'Municipalidad de La Paz',
'Municipalidad  de Rosario' : 'Municipalidad de Rosario',
'MunicÃ­pio de Faro' : 'Municipality of Faro',
'MunicÃ­pio de Viseu' : 'Municipality of Viseu',
'Municipality of Ã‰vora' : 'Municipality of Evora',
'MunicÃ­pio de Ã‰vora' : 'Municipality of Evora',
'MunicÃ­pio de Cascais' : 'Municipality of Cascais',
'Dhaka North City' : 'City of Dhaka',
'Dhaka City' : 'City of Dhaka',
'Commune de Monaco' : 'Municipality of Monaco',
'Ville de Monaco' : 'Municipality of Monaco',
'Lahden kaupunki' : 'City of Lahti',
'Tampereen kaupunki' : 'City of Tampere',
'Aarhus Kommune' : 'Aarhus Municipality',
'Espoon kaupunki' : 'City of Espoo',
'Hanse- und UniversitÃ¤tsstadt Rostock' : 'Hansestadt Rostock',
'City of ReykjavÃ­k' : 'City of Reykjavik',
'ReykjavÃ­kurborg' : 'City of Reykjavik',
'Gemeente Nijmegen' : 'Municipality of Nijmegen',
'Trondheim Municipality' : 'Municipality of Trondheim',
'Trondheim kommune' : 'Municipality of Trondheim',
'Bergen Municipality' : 'Municipality of Bergen',
'Bergen kommune' : 'Municipality of Bergen',
'Miasto WrocÅ‚aw' : 'City of Wroclaw',
'Ayuntamiento de Murcia' : 'Municipality of Murcia',
'UmeÃ¥ kommun' : 'Municipality of Umea',
'UmeÃ¥ municipality' : 'Municipality of Umea',
'Uppsala Municipality' : 'Municipality of Uppsala',
'Uppsala kommun' : 'Municipality of Uppsala',
'Helsingborgs stad' : 'City of Helsingborg',
'Lund Municipality' : 'City of Lund',
'Lunds kommun' : 'City of Lund',
'Municipality of GuimarÃ£es' : 'Municipality of Guimaraes',
'MunicÃ­pio de GuimarÃ£es' : 'Municipality of Guimaraes',
'Municipality of Ãgueda' : 'Municipality of Agueda',
'MunicÃ­pio de Ãgueda' : 'Municipality of Agueda',
'MunicÃ­pio de Braga' : 'Municipality of Braga',
'Prefeitura NiterÃ³i' : 'Prefeitura Niteroi',
'Prefeitura de NiterÃ³i' : 'Prefeitura Niteroi',
'BÃ¦rum Kommune' : 'Municipality of Baerum',
'BÃ¦rum Municipality' : 'Municipality of Baerum',
'Hoeje-Taastrup Kommune' : 'Municipality of Hoeje-Taastrup',
'HÃ¸je-Taastrup Kommune' : 'Municipality of Hoeje-Taastrup',
'HÃ¸je-Taastrup Municipality' : 'Municipality of Hoeje-Taastrup',
'Podgorica Capital City' : 'City of Podgorica',
'HelsingÃ¸r Kommune' : 'Municipality of Helsingor ',
'HelsingÃ¸r Kommune / Elsinore Municipality' : 'Municipality of Helsingor ',
'HelsingÃ¸r Municipality' : 'Municipality of Helsingor ',
'AkureyrarbÃ¦r' : 'City of Akureyri',
'Town of Akureyri' : 'City of Akureyri',
'Municipality of Tirana' : 'City of Tirana',
'Gladsaxe Kommune' : 'Municipality of Gladsaxe',
'Gladsaxe Municipality' : 'Municipality of Gladsaxe',
'Middelfart Kommune' : 'Municipality of Middelfart',
'Middelfart Municipality' : 'Municipality of Middelfart',
'City of YaoundÃ© 6' : 'City of Yaounde',
'YaoundÃ© 6' : 'City of Yaounde',
'Porvoon kaupunki' : 'City of Porvoo',
'Miasto Gdynia' : 'City of Gdynia',
'Klaipeda City Municipality' : 'City of Klaipeda',
'KlaipÄ—da City Municipality' : 'City of Klaipeda',
'KlaipÄ—dos miesto savivaldybÄ—' : 'City of Klaipeda',
'LinkÃ¶ping Municipality' : 'Municipality of Linkoping',
'LinkÃ¶pings kommun' : 'Municipality of Linkoping',
'Vantaan kaupunki' : 'City of Vantaa',
'City of VÃ¤xjÃ¶' : 'City of Vaxjo',
'VÃ¤xjÃ¶ Municipality' : 'City of Vaxjo',
'VÃ¤xjÃ¶ kommun' : 'City of Vaxjo',
'Karlskrona Municipality' : 'Municipality of Karlskrona',
'Municipality of Karlskrona' : 'Municipality of Karlskrona',
'PanevÄ—Å¾io miesto savivaldybÄ—' : 'Municipality of Panevezys',
'PanevÄ—Å¾ys City Municipality' : 'Municipality of Panevezys',
'Arendal Municipality' : 'Municipality of Arendal',
'Arendal kommune' : 'Municipality of Arendal',
'Municipality of Arendal' : 'Municipality of Arendal',
'Trelleborg Municipality' : 'Municipality of Trelleborg',
'Trelleborgs kommun' : 'Municipality of Trelleborg',
'AlcaldÃ­a de Sincelejo' : 'Municipality of Sincelejo',
'AlcaldÃ­ade Sincelejo' : 'Municipality of Sincelejo',
'Hvidovre Kommune' : 'City of Hvidovre',
'Hvidovre Municipality' : 'City of Hvidovre',
'Municipiul Alba Iulia' : 'City of Alba-Iulia',
'MÃ©tropole de Rouen' : 'Metropolitan Municipality of Rouen',
'Egedal Kommune' : 'Municipality of Egedal',
'Egedal Municipality' : 'Municipality of Egedal',
'Municipio La Chorrera' : 'Municipio de Chorrera',
'Kristianstad' : 'Municipality of Kristianstad',
'Kristianstads kommun' : 'Municipality of Kristianstad',
'VÃ¤stervik Municipality' : 'Municipality of Västervik',
'VÃ¤sterviks kommun' : 'Municipality of Västervik',
'Kemi' : 'City of Kemi',
'Kemin kaupunki' : 'City of Kemi',
'Greifswald' : 'City of Greifswald',
'TauragÄ—' : 'Taurage District Municipality',
'TauragÄ— District Municipality' : 'Taurage District Municipality',
'TauragÄ—s rajono savivaldybÄ—' : 'Taurage District Municipality',
'Armstrong' : 'Municipalidad de Armstrong',
'Roskilde' : 'Municipality of Roskilde',
'Roskilde Kommune' : 'Municipality of Roskilde',
'RoskildeÂ Municipality' : 'Municipality of Roskilde',
'MunÃ­cipio de Sintra' : 'Municipality of Sintra',
'MunicÃ­pio de Valongo' : 'Municipality of Valongo',
'MunicÃ­pio de Figueira da Foz' : 'Municipality of Figueira da Foz',
'MunicÃ­pio de Amarante' : 'Municipality of Amarante',
'Comune di Massa Marittima' : 'City of Massa Marittima',
'MunicÃ­pio de Mafra' : 'Municipality of Mafra',
'Municipality of Lagos (Portugal)' : 'Municipality of Lagos',
'MunicÃ­pio de Lagos' : 'Municipality of Lagos',
'MunicÃ­pio de Odemira' : 'Municipality of Odemira',
'MunicÃ­pio de Coruche' : 'Municipality of Coruche',
'Dobong-gu District of Seoul' : 'Municipality of Dobong-gu',
'Gangdong-gu District of Seoul' : 'Municipality of Gangdong-gu',
'San Justo (Argentina)' : 'San Justo',
'Michuhol-gu District of Incheon' : 'Municipality of Michuhol-gu',
'Michuhol-gu Municipal Government of Incheon' : 'Municipality of Michuhol-gu',
'Falkoping Kommun' : 'Municipality of Falkoping',
'FalkÃ¶pings kommun' : 'Municipality of Falkoping',
'CittÃ  Metropolitana di Milano' : 'Metropolitan Municipality of Milan',
'Uppvidinge Municipality' : 'Municipality of Uppvidinge',
'Uppvidinge kommun' : 'Municipality of Uppvidinge',
'Comune di Segrate' : 'City of Segrate'
}

# COMMAND ----------

# Replace region ONLY
from pyspark.sql.functions import col, create_map, lit, when
from itertools import chain

mapping_region = "CASE "
for key, value in replace_region.items():
    mapping_region += "WHEN region = '{0}' THEN '{1}' ".format(key, value)
mapping_region += "ELSE region END"
df_account = df_account.withColumn('region', expr(mapping_region))

# COMMAND ----------

# NOT WORKING

# Replacements in CDP Region
mapping_region = "CASE "
for key, value in replace_region.items():
    mapping_region += "WHEN region = '{0}' THEN '{1}' ".format(key, value)
mapping_region += "ELSE region END"
df_account = df_account.withColumn('region', expr(mapping_region))

mapping_country = create_map([lit(x) for x in chain(*replace_country.items())])
df_account = df_account.withColumn('Country', mapping_country[df_account['Country']])

mapping_city = create_map([lit(x) for x in chain(*replace_city.items())])
df_account = df_account.withColumn('city_fix', mapping_city[df_account['city_fix']])

# COMMAND ----------

display(df_account)

# COMMAND ----------

display(df_account.filter(df_account.account == '62171'))

# COMMAND ----------

# Simplify Organization field
df_accounts = df_accounts.withColumn('Organization1', split(df_accounts['Organization'], ",")[0])  \
                         .withColumn('Organization2', split(df_accounts['Organization'], ",")[1])
df_accounts = df_accounts.drop('Organization', 'Organization2', 'count')
df_accounts = df_accounts.withColumnRenamed('Organization1', 'Organization')

# COMMAND ----------

# Save as parquet file
df_account.write.parquet("/FileStore/df_account.parquet", 
                            mode="overwrite")

# COMMAND ----------

Countrydf_accounts_test = df_accounts.groupby('Country').count()
display(df_accounts_test)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Pre-shaping dataset 2022 original


# COMMAND ----------



# COMMAND ----------

.withColumn("c14_2a_3", 
          when((upper(col("14_2a_3")).like("%HIGH%")) | (upper(col("14_2a_3")).like("%EXTREME%")), '4')
          .when((upper(col("14_2a_3")).like("%MEDIUM%")) | (upper(col("14_2a_3")) == "SERIOUS"), '3')
          .when((upper(col("14_2a_3")).like("%LOW%")) | (upper(col("14_2a_3")).like("%LESS%")), '2')
          .when((upper(col("14_2a_3")).like("%IMPACT%")), '1')
          .when(col("14_2a_3").isNull(), None)
          .otherwise('0'))\
.withColumn("c14_2a_2", 
          when((upper(col("14_2a_2")).like("%IMMEDIATELY%")) | (upper(col("14_2a_2")).like("%CURRENT%")), '4')
          .when((upper(col("14_2a_2")).like("%SHORT%")), '3')
          .when((upper(col("14_2a_2")).like("%MEDIUM%")), '2')
          .when((upper(col("14_2a_2")).like("%LONG%")), '1')
          .when(col("14_2a_2").isNull(), None)
          .otherwise('0'))\
.withColumn("c14_2a_1", 
          when(condicion_final2, 'Water quality')
          .when(condicion_final3, 'Water scarcity')
          .when((upper(col("14_2a_1")).like("%WATER DEMAND%")), 'Water demand')
          .when((upper(col("14_2a_1")).like("%DROUGHT%")), 'Drought')
          .when((upper(col("14_2a_1")).like("%WEATHER%")), 'Weather')
          #.when((upper(col("14_2a_1")).like("%REGULAT%")), 'Regulatory')
          #.when((upper(col("14_2a_1")).like("%ENERG%")), 'Energy')
          #.when((upper(col("14_2a_1")).like("%POLLUTION%")), 'Pollution incidents')
          #.when((upper(col("14_2a_1")).like("%CHANGE IN LAND-USE%")), 'Change in land-use')
          .when((upper(col("14_2a_1")).like("%INADEQUATE OR AGEING INFRA%")), 'Inadequate infrastructure')
          .when((upper(col("14_2a_1")).like("%FLOOD%")), 'Flood')
          .when((upper(col("14_2a_1")).like("%RAIN%")), 'Rain')
          .when(col("14_2a_1").isNull(), None)
          .otherwise("Other"))\