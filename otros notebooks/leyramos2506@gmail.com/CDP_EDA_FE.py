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

# Input data from Databricks FileStore
df_responses = spark.read.format('parquet').options(header=True,inferSchema=True).load('dbfs:/FileStore/df_responses_final.parquet/')

# COMMAND ----------

from pyspark.sql.types import DecimalType

df_responses = df_responses.withColumn('0_5_3_1', df_responses['0_5_3_1'].cast(IntegerType()))\
.withColumn('0_6_1_1', df_responses['0_6_1_1'].cast(DecimalType()))\
.withColumn('14_1_0', df_responses['14_1_0'].cast(DecimalType()))\
.withColumn('c2_0_0', df_responses['c2_0_0'].cast(DecimalType()))\
.withColumn('c3_2_0', df_responses['c3_2_0'].cast(DecimalType()))\
.withColumn('c4_0_0', df_responses['c4_0_0'].cast(DecimalType()))\
.withColumn('c2_0b_7', df_responses['c2_0b_7'].cast(DecimalType()))\
.withColumn('c5_5_0', df_responses['c5_5_0'].cast(DecimalType()))\
.withColumn('c5_0_0', df_responses['c5_0_0'].cast(DecimalType()))\
.withColumn('c2_1_3', df_responses['c2_1_3'].cast(DecimalType()))\
.withColumn('c2_1_4', df_responses['c2_1_4'].cast(DecimalType()))\
.withColumn('c2_1_11', df_responses['c2_1_11'].cast(DecimalType()))\
.withColumn('c4_6a_1_sum', df_responses['c4_6a_1_sum'].cast(DecimalType()))\
.withColumn('c4_6a_1_avg', df_responses['c4_6a_1_avg'].cast(DecimalType()))\
.withColumn('c4_6a_sum', df_responses['c4_6a_sum'].cast(DecimalType()))\
.withColumn('c10_1_sum_1', df_responses['c10_1_sum_1'].cast(DecimalType()))\
.withColumn('c10_3_sum_1', df_responses['c10_3_sum_1'].cast(DecimalType()))\
.withColumn('c8_1_sum_1', df_responses['c8_1_sum_1'].cast(DecimalType()))\
.withColumn('cyes_no_sum', df_responses['cyes_no_sum'].cast(DecimalType()))\
.withColumn('c5_0_percentage_achived', df_responses['c5_0_percentage_achived'].cast(DecimalType()))\
.withColumn('c5_0_base_year_emi', df_responses['c5_0_base_year_emi'].cast(DecimalType()))\
.withColumn('Population', df_responses['Population'].cast(IntegerType()))\
.withColumn('first_time', df_responses['first_time'].cast(IntegerType()))

# COMMAND ----------


# Rename columns to descriptive names
df_responses = df_responses.withColumnRenamed( '2_2_4', 'cat_supports/challenges_how')\
.withColumnRenamed( '3_2a_5', 'cat_adaptation_plan_year')\
.withColumnRenamed( 'c6_0_1', 'cat_opportunity')\
.withColumnRenamed( 'c0_1_1_1', 'cat_boundary')\
.withColumnRenamed( 'c14_0_0', 'cat_water_supply')\
.withColumnRenamed( 'c2_1_1', 'cat_climate_hazard')\
.withColumnRenamed( 'c2_1_6', 'cat_services_affected')\
.withColumnRenamed( 'c2_1_7', 'cat_vulnerable_population')\
.withColumnRenamed( 'c2_1_8', 'cat_change_frequency')\
.withColumnRenamed( 'c2_1_9', 'cat_chance_intensity')\
.withColumnRenamed( 'c2_2_1', 'cat_factors_affecting')\
.withColumnRenamed( 'c2_2_2', 'cat_supports/challenges')\
.withColumnRenamed( 'c3_0_2', 'cat_actions')\
.withColumnRenamed( 'c3_0_4', 'cat_actions_status')\
.withColumnRenamed( 'c3_2a_9', 'cat_adaptation_plan_type')\
.withColumnRenamed( 'c4_4_0', 'cat_gases')\
.withColumnRenamed( 'c6_2a_1', 'cat_collaboration_area')\
.withColumnRenamed( 'c6_5_3', 'cat_project_status')\
.withColumnRenamed( 'c6_5_4', 'cat_financing_status')\
.withColumnRenamed( 'cboundary', 'cat_boundary_type')\
.withColumnRenamed( 'csector', 'cat_sector_general')\
.withColumnRenamed( '0_5_3_1', 'num_population_projected')\
.withColumnRenamed( '0_6_1_1', 'num_area_sq2km')\
.withColumnRenamed( '14_1_0', 'num_%_potable_water')\
.withColumnRenamed( 'c10_1_sum_1', 'num_transport_mode_share')\
.withColumnRenamed( 'c10_3_sum_1', 'num_transport_total_fleet')\
.withColumnRenamed( 'c2_0_0', 'num_vulnerability_assessment')\
.withColumnRenamed( 'c2_0b_7', 'num_vulnerable_population_assessment')\
.withColumnRenamed( 'c2_1_11', 'num_when_changes_expected')\
.withColumnRenamed( 'c2_1_3', 'num_hazard_probability')\
.withColumnRenamed( 'c2_1_4', 'num_hazard_magnitude')\
.withColumnRenamed( 'c3_2_0', 'num_adaptation_plan')\
.withColumnRenamed( 'c4_0_0', 'num_emissions_inventory')\
.withColumnRenamed( 'c4_6a_1_avg', 'num_direct_emissions_avg')\
.withColumnRenamed( 'c4_6a_1_sum', 'num_direct_emissions_sum')\
.withColumnRenamed( 'c4_6a_sum', 'num_all_emissions_sum')\
.withColumnRenamed( 'c5_0_0', 'num_reduction_target')\
.withColumnRenamed( 'c5_0_base_year', 'cat_reduction_target_base_year')\
.withColumnRenamed( 'c5_0_base_year_emi', 'num_reduction_target_base_emi')\
.withColumnRenamed( 'c5_0_percentage_achived', 'num_reduction_target_%_achieved')\
.withColumnRenamed( 'c5_0_target_year', 'cat_reduction_target_year')\
.withColumnRenamed( 'c5_0_target_year_set', 'cat_reduction_target_year_set')\
.withColumnRenamed( 'c5_5_0', 'num_change_adaptation')\
.withColumnRenamed( 'c8_1_sum_1', 'num_electricity_consumed')\
.withColumnRenamed( 'cyes_no_sum', 'num_yes_no_sum')\
.withColumnRenamed( 'Population', 'num_population')\
.withColumnRenamed( 'population_year_bin', 'cat_population_year_bin')\
.withColumnRenamed( 'first_time', 'num_first_time')

# COMMAND ----------

display(df_responses)

# COMMAND ----------

display(df_responses.filter(df_responses.year == '2021'))

# COMMAND ----------

display(df_responses.filter(df_responses.account == '831618'))

# COMMAND ----------

'62171_2021' reemplazar num_area_sq2km 664000000 por 664

# COMMAND ----------

## 3_2a_5 bin <2010, 2010-2015, 2015-2020, >2020

# COMMAND ----------

# Inizialite columns
numeric_columns = []
categorical_columns = []

# Split numerical of category columns
for column in df_responses.columns:
    column_type = df_responses.schema[column].dataType
    if isinstance(column_type, DecimalType):
        numeric_columns.append(column)
    else:
        categorical_columns.append(column)

#Columns
print("Numerical columns:", numeric_columns)
print("Category columns:", categorical_columns)

# COMMAND ----------

display(df_responses.select(*categorical_columns))

# COMMAND ----------



# COMMAND ----------

# Summarize numeric
dbutils.data.summarize(df_responses.select(*numeric_columns), precise= True)

# COMMAND ----------

display(df_responses.select(*numeric_columns))

# COMMAND ----------

#REVISAR POPULATION, AREA, EMISSIONES, TRANSPORT Y ELECT CONSUMED PARA VERIFICAR UNIDADES CORRECTAS
# ONE HOT ENCODING CATEGORICAS
# BINNIG DE ANOS
# PCA COMO ESTAN
# REEMPLAZAR OUTLIERS BASADO EN IQR, CON MEDIA

# COMMAND ----------

# Los valores que están fuera del rango [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] se consideran outliers

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, approxQuantile

# Crear una instancia de SparkSession
spark = SparkSession.builder.appName("Calculating IQR").getOrCreate()

# DataFrame de ejemplo con columnas numéricas
data = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
columns = ["col1", "col2"]
df = spark.createDataFrame(data, columns)

# Lista de columnas numéricas para calcular IQR, Q1 y Q3
numeric_columns = ["col1", "col2"]

# Calcular aproximadamente los cuantiles 25% (Q1) y 75% (Q3)
quantiles = df.approxQuantile(numeric_columns, [0.25, 0.75], 0.01)

# Obtener los valores Q1 y Q3 de las columnas numéricas
Q1_values = quantiles[0]
Q3_values = quantiles[1]

# Calcular el Rango Intercuartílico (IQR)
IQR_values = [Q3 - Q1 for Q1, Q3 in zip(Q1_values, Q3_values)]

print("Q1 values:", Q1_values)
print("Q3 values:", Q3_values)
print("IQR values:", IQR_values)


# COMMAND ----------



# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import col

# Obtener los datos de las columnas en un DataFrame de pandas
df_responses_numeric_pd = df_responses.select(*numeric_columns).toPandas()

# Iterar a través de las columnas y graficar histogramas
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df_responses_numeric_pd[column], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histograma de {column}')
    plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

