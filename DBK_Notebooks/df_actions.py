# Databricks notebook source
# DBTITLE 1,Libraries
import pyspark
import numpy as np
import pandas as pd

import  seaborn as sns
import matplotlib.pyplot as plt

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
from pyspark.sql.functions import col, coalesce
from pyspark.sql.types import *;
from scipy.stats import *
from scipy import stats
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import col, collect_list
from pyspark.sql.functions import regexp_replace

import gzip
# import StringIO --> ModuleNotFoundError: No module named 'StringIO'


from functools import reduce

# unificaion
from pyspark.sql.functions import col, concat, lit, lower, when

# COMMAND ----------

# DBTITLE 1,Functions
#-----------------------------------------------------------------------#
# Get Shape
#-----------------------------------------------------------------------#
def sparkShape(df):
    return (df.count(), len(df.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape

#display(df_responses.shape())


#-----------------------------------------------------------------------#
# Append a list of dfs for selected columns
#-----------------------------------------------------------------------#
def append_dfs(dataframes: List[DataFrame], columns: List[str]) -> DataFrame:
    # Select columns
    df_union = dataframes[0].select(*columns)
    
    # Iterate and append all dfs
    for df in dataframes[1:]:
        df_selected = df.select(*columns)
        df_union = df_union.unionByName(df_selected)
    
    return df_union

#-----------------------------------------------------------------------#
# Replace spaces in column names
#-----------------------------------------------------------------------#
def remplazeSpacesInColumnName(df):
    columns = df.columns
    # Reemplazar espacios y puntos en los nombres de las columnas
    modified_columns = [col_name.replace(' ', '_').replace('.', '_') for col_name in columns]
    # Renombrar las columnas en el DataFrame
    renamed_df = df.select([col(col_name).alias(modified_col_name) for col_name, modified_col_name in zip(columns, modified_columns)])
    
    return renamed_df

#-----------------------------------------------------------------------#
# Calculate Null % for each column in df
#-----------------------------------------------------------------------#
def calcular_porcentaje_nulos(df):
    # Calcular el número de valores nulos por columna
    null_counts = df.select([sum(col(column).isNull().cast("int")).alias(column) for column in df.columns])
    
    # Convertir el DataFrame de recuento de nulos en un formato apilado
    expr_columns = ", ".join(["'{}', cast(`{}` as string)".format(col_name.replace(' ', '_'), col_name) for col_name in df.columns])
    stack_expr = "stack({num_columns}, {columns}) as (columna, valor)".format(
        num_columns=len(df.columns),
        columns=expr_columns
    )
    null_counts2 = null_counts.selectExpr(stack_expr)
    
    # Calcular el porcentaje de nulos por columna
    total_filas = df.count()
    null_counts2 = null_counts2.withColumn('percentage', (col('valor') / total_filas) * 100)
    
    return null_counts2
  
#-----------------------------------------------------------------------#
# Remove columns in df if Null % < percentage_threshold
#-----------------------------------------------------------------------#
def filter_columns_by_null_percentage(df_piv, df_null, percentage_threshold, column_to_keep_nulls_others):
    # Filtrar las columnas que cumplen el umbral de porcentaje
    columns_to_keep_nulls = df_null.filter(col("percentage") <= percentage_threshold) \
                                  .select(collect_list("columna")) \
                                  .first()[0]
                                  
    # Combinar las columnas adicionales con las columnas a mantener por el porcentaje de nulos
    columns_to_keep_nulls += column_to_keep_nulls_others

    #Preparar df antes de filtrar
    df_piv = remplazeSpacesInColumnName(df_piv)
    # Seleccionar solo las columnas que se van a mantener
    df_piv_filtered = df_piv.select(*[col(c) for c in df_piv.columns if c in columns_to_keep_nulls])
    
    return df_piv_filtered

#-----------------------------------------------------------------------#
# Add a prefix for a list of columns in df
#-----------------------------------------------------------------------#
def AddColumnPrefix(df, prefix, exclude_columns=[]):
    current_columns = df.columns
    # Lista de columnas a renombrar
    columns_to_rename = [col for col in current_columns if col not in exclude_columns]
    # Aplicar los cambios al DataFrame
    renamed_df = df.select([col(col_name).alias(prefix + col_name) if col_name in columns_to_rename else col(col_name) for col_name in current_columns])

    return renamed_df

# COMMAND ----------

#Funciones para manejar NULLS






##############



##############################


####################
def convertirDecimal(df, columns_to_exclude=[]):
    all_columns = df.columns
    columns_to_convert = [col_name for col_name in all_columns if col_name not in columns_to_exclude]
    
    # Aplicar la conversión a tipo decimal a las columnas seleccionadas
    for col_name in columns_to_convert:
        df = df.withColumn(col_name, df[col_name].cast("decimal"))
    # Completar los valores nulos con 0 en todas las columnas
    df = df.fillna(0)
    
    return df

######################
# Set 1 if value is not null in columns
def ColumntoFlag(df, excepciones):
    for columna in df.columns:
        if columna not in excepciones:
            df = df.withColumn(columna, when(col(columna).isNull() | (col(columna) == ""), 0).otherwise(1))
            df = df.withColumnRenamed(columna, columna)  # Mantener el mismo nombre de columna
    return df

# COMMAND ----------

# DBTITLE 1,Import data from Storage Account
storage_account_name = 'stgaclnrmlab'
storage_account_access_key = 'rf/4ogYc/eG+oqVD8K9xjsVamosf1qO1s0Kab+ujHsTt0GjaGY2XHfXFNVdti4iaUndCJjNSqizi+ASt8IWqHw=='
spark.conf.set('fs.azure.account.key.' + storage_account_name + '.blob.core.windows.net', storage_account_access_key)
blob_container = 'datasets'

path_2012 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2012_Cities_Emissions_Reduction_Actions.csv"
path_2013 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2013_Cities_Emissions_Reduction_Actions.csv"
path_2014 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2014_Cities_Emissions_Reduction_Actions.csv"
path_2015 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2015_Cities_Emissions_Reduction_Actions.csv"
path_2016 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2016_Cities_Emissions_Reduction_Actions.csv"
path_2017 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2017_Cities_Emissions_Reduction_Actions.csv"
#path_2018 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2018_Cities_Emissions_Reduction_Actions.csv"
path_2019 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2019_Cities_Emissions_Reduction_Actions.csv"
path_2020 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2020_Cities_Emissions_Reduction_Actions.csv"
path_2021 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2021_Cities_Emissions_Reduction_Actions.csv"
path_2022 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2022_Cities_Emissions_Reduction_Actions.csv"
path_2023 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions_reduction_actions/2023_Cities_Emissions_Reduction_Actions.csv"

actions_2012 = spark.read.format("csv").load(path_2012, inferSchema = True, header = True)
actions_2013 = spark.read.format("csv").load(path_2013, inferSchema = True, header = True)
actions_2014 = spark.read.format("csv").load(path_2014, inferSchema = True, header = True)
actions_2015 = spark.read.format("csv").load(path_2015, inferSchema = True, header = True)
actions_2016 = spark.read.format("csv").load(path_2016, inferSchema = True, header = True)
actions_2017 = spark.read.format("csv").load(path_2017, inferSchema = True, header = True)
#actions_2018 = spark.read.format("csv").load(path_2018, inferSchema = True, header = True)
#actions_2019 = spark.read.format("csv").load(path_2019, inferSchema = True, header = True)
actions_2020 = spark.read.format("csv").load(path_2020, inferSchema = True, header = True)
actions_2021 = spark.read.format("csv").load(path_2021, inferSchema = True, header = True)
actions_2022 = spark.read.format("csv").load(path_2022, inferSchema = True, header = True)
actions_2023 = spark.read.format("csv").load(path_2023, inferSchema = True, header = True)

# COMMAND ----------

display(actions_2017)

# COMMAND ----------

# Buscar datos el Full dataset de 2018 y 2019

# COMMAND ----------

# Rename columns
actions_2012 = actions_2012.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Effects of Climate Change", "hazard")\
.withColumnRenamed("Risk Level", "hazard_magnitude")\
.withColumnRenamed("Risk Timescale", "hazard_timescale")
actions_2013 = actions_2013.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Effects of Climate Change", "hazard")\
.withColumnRenamed("Risk Level", "hazard_magnitude")\
.withColumnRenamed("Risk Timescale", "hazard_timescale")
actions_2014 = actions_2014.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Effects of Climate Change", "hazard")\
.withColumnRenamed("Risk Level", "hazard_magnitude")\
.withColumnRenamed("Risk Timescale", "hazard_timescale")
actions_2015 = actions_2015.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Effects of climate change", "hazard")\
.withColumnRenamed("Magnitude", "hazard_magnitude")\
.withColumnRenamed("Anticipated timescale in years", "hazard_timescale")
actions_2016= actions_2016.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed(" Climate hazards​", "hazard")\
.withColumnRenamed("Magnitude of impact", "hazard_magnitude")
actions_2017 = actions_2017.withColumnRenamed("Account number", "account")\
.withColumnRenamed("Project Year", "year")\
.withColumnRenamed("Climate Hazards", "hazard")\
.withColumnRenamed("Magnitude of Impact", "hazard_magnitude")
actions_2018 = actions_2018.withColumnRenamed("Questionnaire", "year")\
.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Anticipated timescale", "hazard_timescale")\
.withColumnRenamed("Climate Hazards", "hazard")\
.withColumnRenamed("Magnitude of impact", "hazard_magnitude")
actions_2019 = actions_2019.withColumnRenamed("Questionnaire Name", "year")\
.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Climate Hazards", "hazard")\
.withColumnRenamed("Current consequence of hazard", "hazard_magnitude")\
.withColumnRenamed("When do you first expect to experience those changes?", "hazard_timescale")
actions_2020 = actions_2020.withColumnRenamed("Questionnaire Name", "year")\
.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Climate Hazards", "hazard")\
.withColumnRenamed("Current magnitude of hazard", "hazard_magnitude")\
.withColumnRenamed("When do you first expect to experience those changes in frequency and intensity?", "hazard_timescale")
actions_2021 = actions_2021.withColumnRenamed("Questionnaire Name", "year")\
.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Climate Hazards", "hazard")\
.withColumnRenamed("Current magnitude of hazard", "hazard_magnitude")\
.withColumnRenamed("When do you first expect to experience those changes in frequency and intensity?", "hazard_timescale")
actions_2022 = actions_2022.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Organization Number", "account")\
.withColumnRenamed("Climate-related hazards", "hazard")\
.withColumnRenamed("Current magnitude of impact of hazard", "hazard_magnitude")\
.withColumnRenamed("Timeframe of expected future changes", "hazard_timescale")
actions_2023 = actions_2023.withColumnRenamed("Questionnaire", "year")\
.withColumnRenamed("Organization Number", "account")\
.withColumnRenamed("Climate-related hazards", "hazard")\
.withColumnRenamed("Current magnitude of impact of hazard", "hazard_magnitude")\
.withColumnRenamed("Timeframe of expected future changes", "hazard_timescale")

# Create missing columns 
actions_2016= actions_2016.withColumn("hazard_timescale", lit(None))
actions_2017 = actions_2017.withColumn("hazard_timescale", lit(None))


# COMMAND ----------

display(actions_2016)

# COMMAND ----------

# DBTITLE 1,Create climate hazards DF
columns = ["account","year","hazard","hazard_magnitude","hazard_timescale"]

# Dfs to append
dataframes = [actions_2012,
actions_2013,
actions_2014,
actions_2015,
actions_2016,
actions_2017,
actions_2018,
actions_2019,
actions_2020,
actions_2021,
actions_2022,
actions_2023]

# Append all dfs
climate_hazards = append_dfs(dataframes, columns)

# COMMAND ----------

# DBTITLE 1,DF transformations
# Extract year
climate_hazards = climate_hazards.withColumn("year", regexp_replace("year", "Cities ", ""))

# Filter wrong rows
climate_hazards = climate_hazards.filter(col("account").isNotNull())
years = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]
climate_hazards = climate_hazards.filter(col("year").isin(years))

# Add unique id
climate_hazards = climate_hazards.withColumn("account_year", concat(climate_hazards["account"], lit("_"), climate_hazards["year"]))

# COMMAND ----------

display(climate_hazards.select("hazard").distinct())

# COMMAND ----------

# DBTITLE 1,Normalize Categories
# Normalize categories
climate_hazards = climate_hazards.withColumn("hazard", 
            when((upper(col("hazard")).like("%ENFERM%")) 
                 | (upper(col("hazard")).like("%DISEASE%"))
                 | (upper(col("hazard")).like("%ILLN%"))
                 | (upper(col("hazard")).like("%INFLUEN%"))
                 | (upper(col("hazard")).like("%ALLERG%"))
                 | (upper(col("hazard")).like("%POLLEN%")), 'Disease')
            .when((upper(col("hazard")).like("%INFEST%"))
                 | (upper(col("hazard")).like("%INVAS%"))
                 | (upper(col("hazard")).like("%PEST%"))
                 | (upper(col("hazard")).like("%PLAG%")), 'Insect infestation')
            .when((upper(col("hazard")).like("%BIO%")), 'Other biological')
            .when((upper(col("hazard")).like("%ATMOS%"))
                 | (upper(col("hazard")).like("%CO2%")), 'Atmospheric CO2 concentrations')
            .when((upper(col("hazard")).like("%AIR%")), 'Air pollution')
            .when((upper(col("hazard")).like("%OCEAN%")), 'Ocean acidification')
            .when((upper(col("hazard")).like("%INTRUSI%"))
                 | (upper(col("hazard")).like("%SALIN%")), 'Salt water intrusion')  
            .when((upper(col("hazard")).like("%RAIN%"))
                 | (upper(col("hazard")).like("%LLUVIA%"))
                 | (upper(col("hazard")).like("%PRECIP%"))
                 | (upper(col("hazard")).like("%MONSOON%"))
                 | (upper(col("hazard")).like("%HAIL%"))
                 | (upper(col("hazard")).like("%GRANIZ%"))
                 | (upper(col("hazard")).like("%FOG%"))
                 | (upper(col("hazard")).like("%NIEB%")), 'Heavy precipitation')
            .when((upper(col("hazard")).like("%SNOW%"))
                 | (upper(col("hazard")).like("%NEVA%"))
                 | (upper(col("hazard")).like("%NIEVE%")), 'Heavy snow')
            .when((upper(col("hazard")).like("%COLD%"))
                 | (upper(col("hazard")).like("%FREEZ%"))
                 | (upper(col("hazard")).like("%FRÍO%"))
                 | (upper(col("hazard")).like("%FROST%"))
                 | (upper(col("hazard")).like("%HELAD%"))
                 | (upper(col("hazard")).like("%IVERN%"))
                 | (upper(col("hazard")).like("%WINTER%")), 'Extreme cold')
            .when((upper(col("hazard")).like("%CALOR%"))
                 | (upper(col("hazard")).like("%CALUR%"))
                 | (upper(col("hazard")).like("%HEAT%"))
                 | (upper(col("hazard")).like("%HIGH%"))
                 | (upper(col("hazard")).like("%HOT%"))
                 | (upper(col("hazard")).like("%TEMP%"))
                 | (upper(col("hazard")).like("%WARM%")), 'Extreme heat')
            .when((upper(col("hazard")).like("%WEATHE%"))
                 | (upper(col("hazard")).like("%CLIMA%")), 'Extreme weather')
            .when((upper(col("hazard")).like("%SUPERF%"))
                 | (upper(col("hazard")).like("%SURFACE%")), 'Flash/surface flood')
            .when((upper(col("hazard")).like("%GROUNDWATER%"))
                 | (upper(col("hazard")).like("%SUBTERR%")), 'Groundwater flood')
            .when((upper(col("hazard")).like("%RIVER%"))
                 | (upper(col("hazard")).like("%FLUVIAL%")), 'River flood')
            .when((upper(col("hazard")).like("%COASTAL%"))
                 | (upper(col("hazard")).like("%MAR%"))
                 | (upper(col("hazard")).like("%SEE%"))
                 | (upper(col("hazard")).like("%SEA LEVEL RISE%")), 'Coastal flood')
            .when((upper(col("hazard")).like("%FLOOD%"))
                 | (upper(col("hazard")).like("%INUND%"))
                 | (upper(col("hazard")).like("%PLUVIAL%")), 'Other flood')
            .when((upper(col("hazard")).like("%AVALAN%")), 'Avalanche')
            .when((upper(col("hazard")).like("%SUBSIDEN%")), 'Subsidence')
            .when((upper(col("hazard")).like("%LANDSL%"))
                 | (upper(col("hazard")).like("%DESLIZ%"))
                 | (upper(col("hazard")).like("%DESLAV%"))
                 | (upper(col("hazard")).like("%ROCK%"))
                 | (upper(col("hazard")).like("%ROCA%"))
                 | (upper(col("hazard")).like("%MASS%")), 'Landslide')
            .when((upper(col("hazard")).like("%SOIL%"))
                 | (upper(col("hazard")).like("%EROSI%")), 'Soil degradation/erosion')
            .when((upper(col("hazard")).like("%CYCLON%"))
                 | (upper(col("hazard")).like("%HURAC%"))
                 | (upper(col("hazard")).like("%TORNADO%")), 'Cyclone/Hurricane/Tornado')
            .when((upper(col("hazard")).like("%WIDN%"))
                 | (upper(col("hazard")).like("%VIENT%")), 'Extreme wind')
            .when((upper(col("hazard")).like("%STORM%"))
                 | (upper(col("hazard")).like("%TORMENT%"))
                 | (upper(col("hazard")).like("%EXTRA%")), 'Storm')
            .when((upper(col("hazard")).like("%DROUGHT%"))
                 | (upper(col("hazard")).like("%SEQU%"))
                 | (upper(col("hazard")).like("%DESERT%"))
                 | (upper(col("hazard")).like("%DRY%"))
                 | (upper(col("hazard")).like("%ESCAZES%")), 'Drought')
            .when((upper(col("hazard")).like("%WATER%"))
                 | (upper(col("hazard")).like("%AGUA%")), 'Water stress')
            .when((upper(col("hazard")).like("%FIRE%"))
                 | (upper(col("hazard")).like("%FUEGO%"))
                 | (upper(col("hazard")).like("%INCEND%")), 'Wildfire')
            .when(col("hazard").isNull(), None)
            .otherwise('Other'))

climate_hazards = climate_hazards.withColumn("hazard_group", 
            when((upper(col("hazard")).like("%ENFERM%")) 
                 | (upper(col("hazard")).like("%DISEASE%"))
                 | (upper(col("hazard")).like("%ILLN%"))
                 | (upper(col("hazard")).like("%INFLUEN%"))
                 | (upper(col("hazard")).like("%ALLERG%"))
                 | (upper(col("hazard")).like("%POLLEN%"))
                 | (upper(col("hazard")).like("%INFEST%"))
                 | (upper(col("hazard")).like("%INVAS%"))
                 | (upper(col("hazard")).like("%PEST%"))
                 | (upper(col("hazard")).like("%PLAG%"))
                 | (upper(col("hazard")).like("%BIO%")), 'Biological')
            .when((upper(col("hazard")).like("%ATMOS%"))
                 | (upper(col("hazard")).like("%CO2%"))
                 | (upper(col("hazard")).like("%AIR%"))
                 | (upper(col("hazard")).like("%OCEAN%"))
                 | (upper(col("hazard")).like("%INTRUSI%"))
                 | (upper(col("hazard")).like("%SALIN%")), 'Chemical change')
            .when((upper(col("hazard")).like("%RAIN%"))
                 | (upper(col("hazard")).like("%LLUVIA%"))
                 | (upper(col("hazard")).like("%PRECIP%"))
                 | (upper(col("hazard")).like("%MONSOON%"))
                 | (upper(col("hazard")).like("%HAIL%"))
                 | (upper(col("hazard")).like("%GRANIZ%"))
                 | (upper(col("hazard")).like("%FOG%"))
                 | (upper(col("hazard")).like("%NIEB%"))
                 | (upper(col("hazard")).like("%SNOW%"))
                 | (upper(col("hazard")).like("%NEVA%"))
                 | (upper(col("hazard")).like("%NIEVE%")), 'Extreme Precipitation')
            .when((upper(col("hazard")).like("%COLD%"))
                 | (upper(col("hazard")).like("%FREEZ%"))
                 | (upper(col("hazard")).like("%FRÍO%"))
                 | (upper(col("hazard")).like("%FROST%"))
                 | (upper(col("hazard")).like("%HELAD%"))
                 | (upper(col("hazard")).like("%IVERN%"))
                 | (upper(col("hazard")).like("%WINTER%"))
                 | (upper(col("hazard")).like("%CALOR%"))
                 | (upper(col("hazard")).like("%CALUR%"))
                 | (upper(col("hazard")).like("%HEAT%"))
                 | (upper(col("hazard")).like("%HIGH%"))
                 | (upper(col("hazard")).like("%HOT%"))
                 | (upper(col("hazard")).like("%TEMP%"))
                 | (upper(col("hazard")).like("%WARM%"))
                 | (upper(col("hazard")).like("%WEATHE%"))
                 | (upper(col("hazard")).like("%CLIMA%")), 'Extreme temperatures')
            .when((upper(col("hazard")).like("%SUPERF%"))
                 | (upper(col("hazard")).like("%SURFACE%"))
                 | (upper(col("hazard")).like("%GROUNDWATER%"))
                 | (upper(col("hazard")).like("%SUBTERR%"))
                 | (upper(col("hazard")).like("%RIVER%"))
                 | (upper(col("hazard")).like("%FLUVIAL%"))
                 | (upper(col("hazard")).like("%COASTAL%"))
                 | (upper(col("hazard")).like("%MAR%"))
                 | (upper(col("hazard")).like("%SEE%"))
                 | (upper(col("hazard")).like("%SEA LEVEL RISE%"))
                 | (upper(col("hazard")).like("%FLOOD%"))
                 | (upper(col("hazard")).like("%INUND%"))
                 | (upper(col("hazard")).like("%PLUVIAL%")), 'Flood and sea level rise')
            .when((upper(col("hazard")).like("%AVALAN%"))
                 | (upper(col("hazard")).like("%SUBSIDEN%"))
                 | (upper(col("hazard")).like("%LANDSL%"))
                 | (upper(col("hazard")).like("%DESLIZ%"))
                 | (upper(col("hazard")).like("%DESLAV%"))
                 | (upper(col("hazard")).like("%ROCK%"))
                 | (upper(col("hazard")).like("%ROCA%"))
                 | (upper(col("hazard")).like("%MASS%"))
                 | (upper(col("hazard")).like("%SOIL%"))
                 | (upper(col("hazard")).like("%EROSI%")), 'Mass movement ')
            .when((upper(col("hazard")).like("%CYCLON%"))
                 | (upper(col("hazard")).like("%HURAC%"))
                 | (upper(col("hazard")).like("%TORNADO%"))
                 | (upper(col("hazard")).like("%WIDN%"))
                 | (upper(col("hazard")).like("%VIENT%"))
                 | (upper(col("hazard")).like("%STORM%"))
                 | (upper(col("hazard")).like("%TORMENT%"))
                 | (upper(col("hazard")).like("%EXTRA%")), 'Storm and wind')
            .when((upper(col("hazard")).like("%DROUGHT%"))
                 | (upper(col("hazard")).like("%SEQU%"))
                 | (upper(col("hazard")).like("%DESERT%"))
                 | (upper(col("hazard")).like("%DRY%"))
                 | (upper(col("hazard")).like("%ESCAZES%"))
                 | (upper(col("hazard")).like("%WATER%"))
                 | (upper(col("hazard")).like("%AGUA%")), 'Water scarcity')
            .when((upper(col("hazard")).like("%FIRE%"))
                 | (upper(col("hazard")).like("%FUEGO%"))
                 | (upper(col("hazard")).like("%INCEND%")), 'Wildfire')
            .when(col("hazard").isNull(), None)
            .otherwise('Other'))

climate_hazards = climate_hazards.withColumn("hazard_magnitude", 
            when((upper(col("hazard_magnitude")) == "SERIOUS") | (upper(col("hazard_magnitude")) == "MEDIUM HIGH"), '3')
            .when((upper(col("hazard_magnitude")).like("%MEDIUM%")), '2')
            .when((upper(col("hazard_magnitude")).like("%HIGH%")) | (upper(col("hazard_magnitude")).like("%EXTREME%")), '4')
            .when((upper(col("hazard_magnitude")).like("%LOW%")) | (upper(col("hazard_magnitude")).like("%LESS%")), '1')
            .when((upper(col("hazard_magnitude")).like("%NOT%")), '0')
            .when(col("hazard_magnitude").isNull(), None)
            .otherwise('0'))

climate_hazards = climate_hazards.withColumn("hazard_timescale", 
          when((upper(col("hazard_timescale")).like("%IMMEDIATELY%")) | (upper(col("hazard_timescale")).like("%CURRENT%")) | (upper(col("hazard_timescale")).like("%ALREADY%")) | (upper(col("hazard_timescale")).like("%INTERMITTENLY%")), '4')
          .when((upper(col("hazard_timescale")).like("%SHORT%")) | (upper(col("hazard_timescale")).like("%HIGH%")), '3')
          .when((upper(col("hazard_timescale")).like("%MEDIUM%")), '2')
          .when((upper(col("hazard_timescale")).like("%LONG%")) | (upper(col("hazard_timescale")).like("%LOW%")), '1')
          .when(col("hazard_timescale").isNull(), None)
          .otherwise('0'))

# COMMAND ----------

display(climate_hazards)

# COMMAND ----------

# DBTITLE 1,Aggregations

# Group by categorical columns according to new categories defined earlier
climate_hazards = climate_hazards.groupBy('account_year', 'account', 'year', 'hazard_group', 'hazard').agg(
    count('*').alias('hazard_count'), 
    avg('hazard_magnitude').alias('hazard_magnitude_avg'),
    sum('hazard_magnitude').alias('hazard_magnitude_sum'),
    avg('hazard_timescale').alias('hazard_timescale_avg'),
    sum('hazard_timescale').alias('hazard_timescale_sum')
)

# Generate General measures
actions_1 = climate_hazards.groupBy('account_year', 'account', 'year') \
    .agg(
    count('*').alias('hazard_count'), 
    avg('hazard_magnitude_avg').alias('hazard_magnitude_avg'),
    sum('hazard_magnitude_sum').alias('hazard_magnitude_sum'),
    avg('hazard_timescale_avg').alias('hazard_timescale_avg'),
    sum('hazard_timescale_sum').alias('hazard_timescale_sum'))

# Generate Specific measures by hazard_group
actions_2 = climate_hazards.groupBy('account_year').pivot('hazard_group') \
    .agg(first('hazard_count').alias('count'), 
         first('hazard_magnitude_avg').alias('magnitude_avg'), 
         first('hazard_magnitude_sum').alias('magnitude_sum'), 
         first('hazard_timescale_avg').alias('timescale_avg'), 
         first('hazard_timescale_sum').alias('timescale_sum')
    )

# Normalize column names
actions_2 = remplazeSpacesInColumnName(actions_2)

# COMMAND ----------

# Calculate null % for each column
actions_2_nulls = calcular_porcentaje_nulos(actions_2)

# Remove columnas with null % < percentage_threshold
percentage_threshold = 51
column_to_keep_nulls_others = []

actions_2 = filter_columns_by_null_percentage(actions_2, actions_2_nulls, percentage_threshold, column_to_keep_nulls_others)

display(actions_2_nulls)

# COMMAND ----------

#Join columns to prepare final df for this question group
df_climate_hazards = actions_1.join(actions_2, on='account_year')

# Fill Nulls with 0
df_climate_hazards = df_climate_hazards.na.fill(0)


df_climate_hazards = AddColumnPrefix(df_climate_hazards, "hazard_", exclude_columns=['account_year',
'account',
'year',
'hazard_count',
'hazard_magnitude_avg',
'hazard_magnitude_sum',
'hazard_timescale_avg',
'hazard_timescale_sum'])



# COMMAND ----------

display(df_climate_hazards)

# COMMAND ----------

# DBTITLE 1,Save df
# Save dataframe as table in catalog default
df_climate_hazards.write.mode("overwrite").saveAsTable("default.df_climate_hazards")

# Save as CSV file in blob container
#storage_account_name = 'stgaclnrmlab'
#storage_account_access_key = 'rf/4ogYc/eG+oqVD8K9xjsVamosf1qO1s0Kab+ujHsTt0GjaGY2XHfXFNVdti4iaUndCJjNSqizi+ASt8IWqHw=='
#spark.conf.set('fs.azure.account.key.' + storage_account_name + '.blob.core.windows.net', storage_account_access_key)
#
#blob_container = 'citiesrespones'
#file_path = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/df_climate_hazards.csv.gz"
#
## Escribir el DataFrame en un archivo CSV comprimido en Azure Blob Storage
#df_climate_hazards.write.csv(file_path, compression='gzip', header=True)

# COMMAND ----------

df_actions_c = df_climate_hazards.groupBy('account') \
    .agg(
    count('*').alias('count'))

df_actions_c.count()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

