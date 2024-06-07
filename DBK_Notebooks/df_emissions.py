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

#-----------------------------------------------------------------------#
# Replace '.' in column names
#-----------------------------------------------------------------------#
def replace_dots(df):
    for col_name in df.columns:
        new_col_name = col_name.replace('.', '_')
        df = df.withColumnRenamed(col_name, new_col_name)
    return df

#-----------------------------------------------------------------------#
# Replace 'Not Applicable' by Null value
#-----------------------------------------------------------------------#
def replace_not_applicable_with_zero(df):
    # Recorre todas las columnas del DataFrame
    for column in df.columns:
        # Reemplaza 'Not Applicable' por 0 en cada columna
        df = df.withColumn(column, when(col(column) == 'Not Applicable', None).otherwise(col(column)))
    return df

#-----------------------------------------------------------------------#
# Replace Non numeric values by Null value
#-----------------------------------------------------------------------#
def replace_non_numeric_with_null(df, exclude_columns=[]):
    # Recorre todas las columnas del DataFrame
    for column in df.columns:
        if column not in exclude_columns:
            # Reemplaza valores no numéricos por null en cada columna que no está en la lista de exclusión
            df = df.withColumn(column, when(col(column).cast(FloatType()).isNotNull(), col(column)).otherwise(None))
    return df

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

path_2012 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2012_Citywide_Emissions_GHG.csv"
path_2013 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2013_Citywide_Emissions_GHG_Map.csv"
path_2014 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2014_Citywide_Emissions_GHG.csv"
path_2015 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2015_Citywide_Emissions.csv"
path_2016 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2016_Citywide_Emissions_GHG.csv"
path_2017 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2017_Citywide_Emissions_GPC.csv"
path_2018 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2018_Citywide_Emissions.csv"
path_2019 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2019_Citywide_Emissions.csv"
path_2020 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2020_Citywide_Emissions.csv"
path_2021 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2021_Citywide_Emissions.csv"
path_2022 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2022_Citywide_Emissions.csv"
path_2023 = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/Emissions/2023_Citywide_Emissions.csv"


emissions_2012 = spark.read.format("csv").load(path_2012, inferSchema = True, header = True)
emissions_2013 = spark.read.format("csv").load(path_2013, inferSchema = True, header = True)
emissions_2014 = spark.read.format("csv").load(path_2014, inferSchema = True, header = True)
emissions_2015 = spark.read.format("csv").load(path_2015, inferSchema = True, header = True)
emissions_2016 = spark.read.format("csv").load(path_2016, inferSchema = True, header = True)
emissions_2017 = spark.read.format("csv").load(path_2017, inferSchema = True, header = True)
emissions_2018 = spark.read.format("csv").load(path_2018, inferSchema = True, header = True)
emissions_2019 = spark.read.format("csv").load(path_2019, inferSchema = True, header = True)
emissions_2020 = spark.read.format("csv").load(path_2020, inferSchema = True, header = True)
emissions_2021 = spark.read.format("csv").load(path_2021, inferSchema = True, header = True)
emissions_2022 = spark.read.format("csv").load(path_2022, inferSchema = True, header = True)
emissions_2023 = spark.read.format("csv").load(path_2023, inferSchema = True, header = True)

# COMMAND ----------

# DBTITLE 1,Tranform emissions_2022

emissions_2022 = replace_not_applicable_with_zero(emissions_2022)
emissions_2022 = replace_non_numeric_with_null(emissions_2022 , exclude_columns=["Questionnaire", "Organization Number", "Emissions reporting framework or protocol", "Inventory Year", "Gases Included", "Inventory boundary (relative to jurisdiction boundary)", "Population in inventory year", "Emissions Row Number", "Emissions Row Name", "Emissions Column Name"])
emissions_2022 = emissions_2022.na.fill(0)

# Pivot column name to columns for 2022 dataset
emissions_2022_ = emissions_2022.groupBy("Questionnaire", "Organization Number", "Emissions reporting framework or protocol", "Inventory Year", "Gases Included", "Inventory boundary (relative to jurisdiction boundary)", "Population in inventory year", "Emissions Row Number", "Emissions Row Name") \
             .pivot("Emissions Column Name") \
             .agg({"Emissions Response Answer": "first"}) \
             .orderBy("Questionnaire", "Organization Number", "Emissions Row Number", "Emissions Row Name")


# Pivot dataset for GPC report format
emissions_2022_gpc = emissions_2022_.filter(emissions_2022_["Emissions (metric tonnes CO2e)"].isNotNull())\
    .groupBy("Questionnaire", "Organization Number", "Emissions reporting framework or protocol", "Inventory Year", "Gases Included", "Inventory boundary (relative to jurisdiction boundary)", "Population in inventory year") \
             .pivot("Emissions Row Name") \
             .agg({"Emissions (metric tonnes CO2e)": "first"}) \
             .orderBy("Questionnaire", "Organization Number")

emissions_2022_gpc = emissions_2022_gpc.withColumn("Total emissions", col('TOTAL BASIC emissions') + col('TOTAL BASIC+ emissions'))\
                                        .withColumn("Total Scope 1 emissions", col('Total Scope 1 emissions (excluding generation of grid-supplied energy)') + col('Scope 1 emissions from generation of grid-supplied energy'))

# Keep only columns of interest
emissions_2022_gpc = emissions_2022_gpc.select(
'Questionnaire',
'Organization Number',
'Emissions reporting framework or protocol',
'Inventory Year',
'Gases Included',
'Inventory boundary (relative to jurisdiction boundary)',
'Population in inventory year',
'Total emissions',
'Total Scope 1 emissions',
'Total Scope 2 emissions',
'Total Scope 3 emissions')


# Pivot dataset for CRF format
emissions_2022_crf = emissions_2022_.filter(emissions_2022_["Emissions (metric tonnes CO2e)"].isNull())\
    .groupBy("Questionnaire", "Organization Number", "Emissions reporting framework or protocol", "Inventory Year", "Gases Included", "Inventory boundary (relative to jurisdiction boundary)", "Population in inventory year") \
             .pivot("Emissions Row Name") \
             .agg({"Direct emissions (metric tonnes CO2e)^": "first"}) \
             .orderBy("Questionnaire", "Organization Number")

emissions_2022_crf = emissions_2022_crf.withColumn("Total emissions", 
col('Total AFOLU') +col('Total Emissions (excluding generation of grid-supplied energy)') +col('Total IPPU') +col('Total Stationary Energy') +col('Total Transport') +col('Total Waste') +col('Total generation of grid-supplied energy'))

# Keep only columns of interest
emissions_2022_crf = emissions_2022_crf.select(
'Questionnaire',
'Organization Number',
'Emissions reporting framework or protocol',
'Inventory Year',
'Gases Included',
'Inventory boundary (relative to jurisdiction boundary)',
'Population in inventory year',
'Total emissions')

# COMMAND ----------

# DBTITLE 1,Tranform emissions_2023

emissions_2023 = replace_not_applicable_with_zero(emissions_2023)
emissions_2023 = replace_non_numeric_with_null(emissions_2023 , exclude_columns=["Questionnaire", "Organization Number", "Primary protocol/framework used to compile main inventory", "Year covered by main inventory", "Gases included in main inventory", "Boundary of main inventory relative to jurisdiction boundary", "Population in year covered by main inventory", "Emissions Row Number", "Emissions Row Name", "Emissions Column Name"])
emissions_2023 = emissions_2023.na.fill(0)

# Pivot column name to columns for 2023 dataset
emissions_2023_ = emissions_2023.groupBy("Questionnaire", "Organization Number", "Primary protocol/framework used to compile main inventory", "Year covered by main inventory", "Gases included in main inventory", "Boundary of main inventory relative to jurisdiction boundary", "Population in year covered by main inventory", "Emissions Row Number", "Emissions Row Name") \
             .pivot("Emissions Column Name") \
             .agg({"Emissions Response Answer": "first"}) \
             .orderBy("Questionnaire", "Organization Number", "Emissions Row Number", "Emissions Row Name")

# Pivot dataset for GPC report format
emissions_2023_gpc = emissions_2023_.filter(emissions_2023_["Emissions (metric tonnes CO2e)"].isNotNull())\
    .groupBy("Questionnaire", "Organization Number", "Primary protocol/framework used to compile main inventory", "Year covered by main inventory", "Gases included in main inventory", "Boundary of main inventory relative to jurisdiction boundary", "Population in year covered by main inventory") \
             .pivot("Emissions Row Name") \
             .agg({"Emissions (metric tonnes CO2e)": "first"}) \
             .orderBy("Questionnaire", "Organization Number")

emissions_2023_gpc = emissions_2023_gpc.withColumn("Total emissions", col('TOTAL BASIC emissions') + col('TOTAL BASIC+ emissions'))\
                                        .withColumn("Total Scope 1 emissions", col('Total Scope 1 emissions (excluding generation of grid-supplied energy)') + col('Scope 1 emissions from generation of grid-supplied energy'))

# Keep only columns of interest
emissions_2023_gpc = emissions_2023_gpc.select(
'Questionnaire',
'Organization Number',
'Primary protocol/framework used to compile main inventory',
'Year covered by main inventory',
'Gases included in main inventory',
'Boundary of main inventory relative to jurisdiction boundary',
'Population in year covered by main inventory',
'Total emissions',
'Total Scope 1 emissions',
'Total Scope 2 emissions',
'Total Scope 3 emissions')


# Pivot dataset for CRF format
emissions_2023_crf = emissions_2023_.filter(emissions_2023_["Emissions (metric tonnes CO2e)"].isNull())\
    .groupBy("Questionnaire", "Organization Number", "Primary protocol/framework used to compile main inventory", "Year covered by main inventory", "Gases included in main inventory", "Boundary of main inventory relative to jurisdiction boundary", "Population in year covered by main inventory") \
             .pivot("Emissions Row Name") \
             .agg({"Direct emissions (metric tonnes CO2e)^": "first"}) \
             .orderBy("Questionnaire", "Organization Number")

emissions_2023_crf = emissions_2023_crf.withColumn("Total emissions", 
col('Total AFOLU') +col('Total Emissions (excluding generation of grid-supplied energy)') +col('Total IPPU') +col('Total Stationary Energy') +col('Total Transport') +col('Total Waste') +col('Total generation of grid-supplied energy'))

# Keep only columns of interest
emissions_2023_crf = emissions_2023_crf.select(
'Questionnaire',
'Organization Number',
'Primary protocol/framework used to compile main inventory',
'Year covered by main inventory',
'Gases included in main inventory',
'Boundary of main inventory relative to jurisdiction boundary',
'Population in year covered by main inventory',
'Total emissions')

# COMMAND ----------

# DBTITLE 1,Tranform emissions_2017
# Pivot column name to columns for 2017 dataset
emissions_2017_ = emissions_2017.filter(emissions_2017["Emissions (metric tonnes CO2e)"].isNotNull())\
    .groupBy("Reporting year", "Account number", "Protocol", "Account year", "Population", "Population Year") \
             .pivot("Sector and scope (GPC reference number)") \
             .agg({"Emissions (metric tonnes CO2e)": "first"}) \
             .orderBy("Reporting year", "Account number")

emissions_2017_ = replace_dots(emissions_2017_)

emissions_2017_  = emissions_2017_.withColumn("Total Scope 2 emissions", col('Stationary Energy: energy use – Scope 2 (I_X_2)') + col('Transportation – Scope 2 (II_X_2)'))\
    .withColumn("Total Scope 3 emissions", col('Stationary Energy: energy use – Scope 3 (I_X_3)') + col('Transportation – Scope 3 (II_X_3)') + col('Waste: waste generated within the city boundary – Scope 3 (III_X_2)'))

# Keep only columns of interest
emissions_2017 = emissions_2017_.select(
"Reporting year", 
"Account number", 
"Protocol", 
"Account year", 
"Population", 
"Population Year",
'TOTAL BASIC and BASIC+ emissions',
'TOTAL BASIC emissions',
'TOTAL Scope 1 (Territorial) emissions',
"Total Scope 2 emissions",
"Total Scope 3 emissions")

# COMMAND ----------

# DBTITLE 1,Replace null values
# Replace 'Not Applicable' values
emissions_2012 = replace_not_applicable_with_zero(emissions_2012)
emissions_2013 = replace_not_applicable_with_zero(emissions_2013)
emissions_2014 = replace_not_applicable_with_zero(emissions_2014)
emissions_2015 = replace_not_applicable_with_zero(emissions_2015)
emissions_2016 = replace_not_applicable_with_zero(emissions_2016)
emissions_2017 = replace_not_applicable_with_zero(emissions_2017)
emissions_2018 = replace_not_applicable_with_zero(emissions_2018)
emissions_2019 = replace_not_applicable_with_zero(emissions_2019)
emissions_2020 = replace_not_applicable_with_zero(emissions_2020)
emissions_2021 = replace_not_applicable_with_zero(emissions_2021)

emissions_2012 = replace_non_numeric_with_null(emissions_2012 , exclude_columns=["Account No", "Primary Methodology", "Measurement Year"])
emissions_2013 = replace_non_numeric_with_null(emissions_2013 , exclude_columns=["Account No", "Primary Methodology", "Measurement Year"])
emissions_2014 = replace_non_numeric_with_null(emissions_2014 , exclude_columns=["Account No", "Primary Methodology", "Measurement Year"])
emissions_2015 = replace_non_numeric_with_null(emissions_2015 , exclude_columns=["Account No", "Primary Methodology", "Measurement Year"])
emissions_2016 = replace_non_numeric_with_null(emissions_2016 , exclude_columns=["Account Number", "Primary Methodology", "Gases included", "Measurement Year"])
emissions_2017 = replace_non_numeric_with_null(emissions_2017 , exclude_columns=["Account number", "Protocol", "Account year"])
emissions_2018 = replace_non_numeric_with_null(emissions_2018 , exclude_columns=["Account Number", "Primary Protocol", "Gases Included", "Accounting Year"])
emissions_2019 = replace_non_numeric_with_null(emissions_2019 , exclude_columns=["Account Number", "Primary Protocol", "Gases Included", "Accounting Year"])
emissions_2020 = replace_non_numeric_with_null(emissions_2020 , exclude_columns=["Account Number", "Primary Protocol", "Gases Included", "Accounting Year"])
emissions_2021 = replace_non_numeric_with_null(emissions_2021 , exclude_columns=["Account Number", "Primary Protocol", "Gases Included", "Accounting Year"])
emissions_2022_gpc = replace_non_numeric_with_null(emissions_2022_gpc , exclude_columns=["Questionnaire", "Organization Number", "Emissions reporting framework or protocol", "Gases Included", "Inventory Year"])
emissions_2022_crf = replace_non_numeric_with_null(emissions_2022_crf , exclude_columns=["Questionnaire", "Organization Number", "Emissions reporting framework or protocol", "Gases Included", "Inventory Year"])
emissions_2023_gpc = replace_non_numeric_with_null(emissions_2023_gpc , exclude_columns=["Questionnaire", "Organization Number", "Primary protocol/framework used to compile main inventory", "Gases included in main inventory", "Year covered by main inventory"])
emissions_2023_crf = replace_non_numeric_with_null(emissions_2023_crf , exclude_columns=["Questionnaire", "Organization Number", "Primary protocol/framework used to compile main inventory", "Gases included in main inventory", "Year covered by main inventory"])


# Replace null values
emissions_2012 = emissions_2012.na.fill(0)
emissions_2013 = emissions_2013.na.fill(0)
emissions_2014 = emissions_2014.na.fill(0)
emissions_2015 = emissions_2015.na.fill(0)
emissions_2016 = emissions_2016.na.fill(0)
emissions_2017 = emissions_2017.na.fill(0)
emissions_2018 = emissions_2018.na.fill(0)
emissions_2019 = emissions_2019.na.fill(0)
emissions_2020 = emissions_2020.na.fill(0)
emissions_2021 = emissions_2021.na.fill(0)
emissions_2022_gpc = emissions_2022_gpc.na.fill(0)
emissions_2022_crf = emissions_2022_crf.na.fill(0)
emissions_2023_gpc = emissions_2023_gpc.na.fill(0)
emissions_2023_crf = emissions_2023_crf.na.fill(0)


# COMMAND ----------

# DBTITLE 1,Calculate Total Emissions
emissions_2017 = emissions_2017.withColumn("Total emissions", 
                                          when(col('TOTAL BASIC and BASIC+ emissions').isNotNull() | col('TOTAL BASIC emissions').isNotNull(),
                                            col('TOTAL BASIC and BASIC+ emissions')+col('TOTAL BASIC emissions'))
                                          .when(col('TOTAL Scope 1 (Territorial) emissions').isNotNull() | col('Total Scope 2 emissions').isNotNull() | col('Total Scope 3 emissions').isNotNull(),
                                            col('TOTAL Scope 1 (Territorial) emissions')+col('Total Scope 2 emissions')+ col('Total Scope 3 emissions'))
                                          .otherwise(0))

emissions_2018 = emissions_2018.withColumn("Total emissions", 
                                          when(col('Total Scope 1 Emissions (metric tonnes CO2e)').isNotNull() | col('Total Scope 2 Emissions (metric tonnes CO2e)').isNotNull() | col('Total Scope 3 Emissions (metric tonnes CO2e)').isNotNull(),
                                            col('Total Scope 1 Emissions (metric tonnes CO2e)')+col('Total Scope 2 Emissions (metric tonnes CO2e)')+col('Total Scope 3 emissions (metric tonnes CO2e)'))
                                          .when(col('Total BASIC Emissions (GPC)').isNotNull() | col('Total BASIC+ Emissions (GPC)').isNotNull(),
                                            col('Total BASIC Emissions (GPC)')+col('Total BASIC+ Emissions (GPC)'))
                                          .otherwise(col('Direct emissions/ Scope 1 (metric tonnes CO2e) for Total generation of grid supplied energy ')+
                                                     col('Direct emissions/ Scope 1 (metric tonnes CO2e) for Total emissions (excluding generation of grid-supplied energy) ')+
                                                     col('Indirect emissions from use of grid supplied energy/Scope 2 (metric tonnes CO2e) for Total generation of grid supplied energy')+
                                                     col('Indirect emissions from use of grid supplied energy/Scope 2 (metric tonnes CO2e) for Total emissions (excluding generation of grid-supplied energy)')+
                                                     col('Emissions occurring outside city boundary/ Scope 3 (metric tonnes CO2e) for Total generation of grid supplied energy ')+
                                                     col('Emissions occurring outside city boundary/ Scope 3 (metric tonnes CO2e) for Total emissions (excluding generation of grid-supplied energy) ')))

emissions_2019 = emissions_2019.withColumn("Total emissions", 
                                          when(col('Total Scope 1 Emissions (metric tonnes CO2e)').isNotNull() | col('Total Scope 2 Emissions (metric tonnes CO2e)').isNotNull() | col('Total Scope 3 Emissions (metric tonnes CO2e)').isNotNull(),
                                            col('Total Scope 1 Emissions (metric tonnes CO2e)')+col('Total Scope 2 Emissions (metric tonnes CO2e)')+col('Total Scope 3 emissions (metric tonnes CO2e)'))
                                          .when(col('Total BASIC Emissions (GPC)').isNotNull() | col('Total BASIC+ Emissions (GPC)').isNotNull(),
                                            col('Total BASIC Emissions (GPC)')+col('Total BASIC+ Emissions (GPC)'))
                                          .otherwise(col('Direct emissions/ Scope 1 (metric tonnes CO2e) for Total generation of grid supplied energy ')+
                                                     col('Direct emissions/ Scope 1 (metric tonnes CO2e) for Total emissions (excluding generation of grid-supplied energy) ')+
                                                     col('Indirect emissions from use of grid supplied energy/Scope 2 (metric tonnes CO2e) for Total generation of grid supplied energy')+
                                                     col('Indirect emissions from use of grid supplied energy/Scope 2 (metric tonnes CO2e) for Total emissions (excluding generation of grid-supplied energy)')+
                                                     col('Emissions occurring outside city boundary/ Scope 3 (metric tonnes CO2e) for Total generation of grid supplied energy ')+
                                                     col('Emissions occurring outside city boundary/ Scope 3 (metric tonnes CO2e) for Total emissions (excluding generation of grid-supplied energy) ')))   

emissions_2020 = emissions_2020.withColumn("Total emissions", 
                                          when(col('TOTAL Scope 1 Emissions (metric tonnes CO2e)').isNotNull() | col('TOTAL Scope 2 emissions (metric tonnes CO2e)').isNotNull() | col('TOTAL Scope 3 Emissions').isNotNull(),
                                            col('TOTAL Scope 1 Emissions (metric tonnes CO2e)')+col('TOTAL Scope 2 emissions (metric tonnes CO2e)')+col('TOTAL Scope 3 Emissions'))
                                          .when(col('TOTAL BASIC Emissions (GPC)').isNotNull() | col('TOTAL BASIC+ Emissions (GPC)').isNotNull(),
                                            col('TOTAL BASIC Emissions (GPC)')+col('TOTAL BASIC+ Emissions (GPC)'))
                                          .otherwise(col('Direct emissions (metric tonnes CO2e) for Total generation of grid-supplied energy')+
                                                     col('Direct emissions (metric tonnes CO2e) for Total emissions (excluding generation of grid-supplied energy)')+
                                                     col('Indirect emissions from use of grid supplied energy (metric tonnes CO2e) for Total generation of grid supplied energy')+
                                                     col('Indirect emissions from use of grid supplied energy (metric tonnes CO2e) for Total Emissions (excluding generation of grid-supplied energy)')+
                                                     col('Emissions occurring outside city boundary (metric tonnes CO2e) for Total Generation of grid supplied energy')+
                                                     col('Emissions occurring outside city boundary (metric tonnes CO2e) for Total Emissions (excluding generation of grid-supplied energy)')))  

emissions_2021 = emissions_2021.withColumn("Total emissions", 
                                          when(col('TOTAL Scope 1 Emissions (metric tonnes CO2e)').isNotNull() | col('TOTAL Scope 2 emissions (metric tonnes CO2e)').isNotNull() | col('TOTAL Scope 3 Emissions').isNotNull(),
                                            col('TOTAL Scope 1 Emissions (metric tonnes CO2e)')+col('TOTAL Scope 2 emissions (metric tonnes CO2e)')+col('TOTAL Scope 3 Emissions'))
                                          .when(col('TOTAL BASIC Emissions (GPC)').isNotNull() | col('TOTAL BASIC+ Emissions (GPC)').isNotNull(),
                                            col('TOTAL BASIC Emissions (GPC)')+col('TOTAL BASIC+ Emissions (GPC)'))
                                          .otherwise(col('Direct emissions (metric tonnes CO2e) for Total generation of grid-supplied energy')+
                                                     col('Direct emissions (metric tonnes CO2e) for Total emissions (excluding generation of grid-supplied energy)')+
                                                     col('Indirect emissions from use of grid supplied energy (metric tonnes CO2e) for Total generation of grid supplied energy')+
                                                     col('Indirect emissions from use of grid supplied energy (metric tonnes CO2e) for Total Emissions (excluding generation of grid-supplied energy)')+
                                                     col('Emissions occurring outside city boundary (metric tonnes CO2e) for Total Generation of grid supplied energy')+
                                                     col('Emissions occurring outside city boundary (metric tonnes CO2e) for Total Emissions (excluding generation of grid-supplied energy)')))                                                

# COMMAND ----------

# DBTITLE 1,Normalize colum names
# Rename columns
emissions_2012 = emissions_2012.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Primary Methodology", "protocol")\
.withColumnRenamed("Measurement Year", "inventory_year")\
.withColumnRenamed("Total City-wide Emissions (metric tonnes CO2e)", "total_emissions_mtco2e")
emissions_2013 = emissions_2013.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Primary Methodology", "protocol")\
.withColumnRenamed("Total City-wide Emissions (metric tonnes CO2e)", "total_emissions_mtco2e")
emissions_2014 = emissions_2014.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Primary Methodology", "protocol")\
.withColumnRenamed("Measurement Year", "inventory_year")\
.withColumnRenamed("Total City-wide Emissions (metric tonnes CO2e)", "total_emissions_mtco2e")
emissions_2015 = emissions_2015.withColumnRenamed("Account No", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Primary Methodology", "protocol")\
.withColumnRenamed("Measurement Year", "inventory_year")\
.withColumnRenamed("Total City-wide Emissions (metric tonnes CO2e)", "total_emissions_mtco2e")
emissions_2016= emissions_2016.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Primary Methodology", "protocol")\
.withColumnRenamed("Measurement Year", "inventory_year")\
.withColumnRenamed("Total City-wide Emissions (metric tonnes CO2e)", "total_emissions_mtco2e")\
.withColumnRenamed("Total Scope 1 Emissions (metric tonnes CO2e)", "total_scope1_mtco2e")\
.withColumnRenamed("Total Scope 2 Emissions (metric tonnes CO2e)", "total_scope2_mtco2e")\
.withColumnRenamed("Gases included", "gases")
emissions_2017 = emissions_2017.withColumnRenamed("Account number", "account")\
.withColumnRenamed("Reporting Year", "year")\
.withColumnRenamed("Protocol", "protocol")\
.withColumnRenamed("Account year", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("TOTAL Scope 1 (Territorial) emissions", "total_scope1_mtco2e")\
.withColumnRenamed("Total Scope 2 emissions", "total_scope2_mtco2e")\
.withColumnRenamed("Total Scope 3 emissions", "total_scope3_mtco2e")
emissions_2018 = emissions_2018.withColumnRenamed("Year Reported to CDP", "year")\
.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Primary Protocol", "protocol")\
.withColumnRenamed("Accounting Year", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("Total Scope 1 Emissions (metric tonnes CO2e)", "total_scope1_mtco2e")\
.withColumnRenamed("Total Scope 2 Emissions (metric tonnes CO2e)", "total_scope2_mtco2e")\
.withColumnRenamed("Total Scope 3 Emissions (metric tonnes CO2e)", "total_scope3_mtco2e")\
.withColumnRenamed("Gases Included", "gases")
emissions_2019 = emissions_2019.withColumnRenamed("Year Reported to CDP", "year")\
.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Primary Protocol", "protocol")\
.withColumnRenamed("Accounting Year", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("Total Scope 1 Emissions (metric tonnes CO2e)", "total_scope1_mtco2e")\
.withColumnRenamed("Total Scope 2 Emissions (metric tonnes CO2e)", "total_scope2_mtco2e")\
.withColumnRenamed("Total Scope 3 emissions (metric tonnes CO2e)", "total_scope3_mtco2e")\
.withColumnRenamed("Gases Included", "gases")
emissions_2020 = emissions_2020.withColumnRenamed("Year Reported to CDP", "year")\
.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Primary Protocol", "protocol")\
.withColumnRenamed("Accounting Year", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("TOTAL Scope 1 Emissions (metric tonnes CO2e)", "total_scope1_mtco2e")\
.withColumnRenamed("TOTAL Scope 2 emissions (metric tonnes CO2e)", "total_scope2_mtco2e")\
.withColumnRenamed("TOTAL Scope 3 Emissions", "total_scope3_mtco2e")\
.withColumnRenamed("Gases Included", "gases")
emissions_2021 = emissions_2021.withColumnRenamed("Year Reported to CDP", "year")\
.withColumnRenamed("Account Number", "account")\
.withColumnRenamed("Primary Protocol", "protocol")\
.withColumnRenamed("Accounting Year", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("TOTAL Scope 1 Emissions (metric tonnes CO2e)", "total_scope1_mtco2e")\
.withColumnRenamed("TOTAL Scope 2 emissions (metric tonnes CO2e)", "total_scope2_mtco2e")\
.withColumnRenamed("TOTAL Scope 3 Emissions", "total_scope3_mtco2e")\
.withColumnRenamed("Gases Included", "gases")
emissions_2022_gpc = emissions_2022_gpc.withColumnRenamed("Questionnaire", "year")\
.withColumnRenamed("Organization Number", "account")\
.withColumnRenamed("Emissions reporting framework or protocol", "protocol")\
.withColumnRenamed("Inventory Year", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("Total Scope 1 emissions", "total_scope1_mtco2e")\
.withColumnRenamed("Total Scope 2 emissions", "total_scope2_mtco2e")\
.withColumnRenamed("Total Scope 3 emissions", "total_scope3_mtco2e")\
.withColumnRenamed("Gases Included", "gases")
emissions_2022_crf = emissions_2022_crf.withColumnRenamed("Questionnaire", "year")\
.withColumnRenamed("Organization Number", "account")\
.withColumnRenamed("Emissions reporting framework or protocol", "protocol")\
.withColumnRenamed("Inventory Year", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("Gases Included", "gases")
emissions_2023_gpc = emissions_2023_gpc.withColumnRenamed("Questionnaire", "year")\
.withColumnRenamed("Organization Number", "account")\
.withColumnRenamed("Primary protocol/framework used to compile main inventory", "protocol")\
.withColumnRenamed("Year covered by main inventory", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("Total Scope 1 emissions", "total_scope1_mtco2e")\
.withColumnRenamed("Total Scope 2 emissions", "total_scope2_mtco2e")\
.withColumnRenamed("Total Scope 3 emissions", "total_scope3_mtco2e")\
.withColumnRenamed("Gases included in main inventory", "gases")
emissions_2023_crf = emissions_2023_crf.withColumnRenamed("Questionnaire", "year")\
.withColumnRenamed("Organization Number", "account")\
.withColumnRenamed("Primary protocol/framework used to compile main inventory", "protocol")\
.withColumnRenamed("Year covered by main inventory", "inventory_year")\
.withColumnRenamed("Total emissions", "total_emissions_mtco2e")\
.withColumnRenamed("Gases included in main inventory", "gases")


# Create missing columns 
emissions_2012 = emissions_2012.withColumn("total_scope1_mtco2e", lit(None))\
                                .withColumn("total_scope2_mtco2e", lit(None))\
                                .withColumn("total_scope3_mtco2e", lit(None))\
                                .withColumn("gases", lit(None))
emissions_2013 = emissions_2013.withColumn("total_scope1_mtco2e", lit(None))\
                                .withColumn("total_scope2_mtco2e", lit(None))\
                                .withColumn("total_scope3_mtco2e", lit(None))\
                                .withColumn("gases", lit(None))\
                                .withColumn("inventory_year", col("year"))
emissions_2014 = emissions_2014.withColumn("total_scope1_mtco2e", lit(None))\
                                .withColumn("total_scope2_mtco2e", lit(None))\
                                .withColumn("total_scope3_mtco2e", lit(None))\
                                .withColumn("gases", lit(None))
emissions_2015 = emissions_2015.withColumn("total_scope1_mtco2e", lit(None))\
                                .withColumn("total_scope2_mtco2e", lit(None))\
                                .withColumn("total_scope3_mtco2e", lit(None))\
                                .withColumn("gases", lit(None))
emissions_2016 = emissions_2016.withColumn("total_scope3_mtco2e", lit(None))                               
emissions_2017 = emissions_2017.withColumn("gases", lit(None))
emissions_2022_crf = emissions_2022_crf.withColumn("total_scope1_mtco2e", lit(None))\
                                .withColumn("total_scope2_mtco2e", lit(None))\
                                .withColumn("total_scope3_mtco2e", lit(None))
emissions_2023_crf = emissions_2023_crf.withColumn("total_scope1_mtco2e", lit(None))\
                                .withColumn("total_scope2_mtco2e", lit(None))\
                                .withColumn("total_scope3_mtco2e", lit(None))

# COMMAND ----------

# DBTITLE 1,Filter rows
emissions_2012 = emissions_2012.filter(emissions_2012["year"].isNotNull() & emissions_2012["total_emissions_mtco2e"].isNotNull())
emissions_2013 = emissions_2013.filter(emissions_2013["year"].isNotNull() & emissions_2013["total_emissions_mtco2e"].isNotNull()) 
emissions_2014 = emissions_2014.filter(emissions_2014["year"].isNotNull() & emissions_2014["total_emissions_mtco2e"].isNotNull()) 
emissions_2015 = emissions_2015.filter(emissions_2015["year"].isNotNull() & emissions_2015["total_emissions_mtco2e"].isNotNull()) 
emissions_2016 = emissions_2016.filter(emissions_2016["year"].isNotNull() & emissions_2016["total_emissions_mtco2e"].isNotNull())
emissions_2017 = emissions_2017.filter(emissions_2017["year"].isNotNull() & emissions_2017["total_emissions_mtco2e"].isNotNull()) 
emissions_2018 = emissions_2018.filter(emissions_2018["year"].isNotNull() & emissions_2018["total_emissions_mtco2e"].isNotNull()) 
emissions_2019 = emissions_2019.filter(emissions_2019["year"].isNotNull() & emissions_2019["total_emissions_mtco2e"].isNotNull()) 
emissions_2020 = emissions_2020.filter(emissions_2020["year"].isNotNull() & emissions_2020["total_emissions_mtco2e"].isNotNull()) 
emissions_2021 = emissions_2021.filter(emissions_2021["year"].isNotNull() & emissions_2021["total_emissions_mtco2e"].isNotNull()) 

# COMMAND ----------

display(emissions_2012)

# COMMAND ----------

# DBTITLE 1,Create climate hazards DF
columns = ['year','account','protocol','inventory_year','total_emissions_mtco2e','total_scope1_mtco2e','total_scope2_mtco2e','total_scope3_mtco2e','gases']

# Dfs to append
dataframes = [emissions_2012,
emissions_2013,
emissions_2014,
emissions_2015,
emissions_2016,
emissions_2017,
emissions_2018,
emissions_2019,
emissions_2020,
emissions_2021,
emissions_2022_gpc,
emissions_2022_crf,
emissions_2023_gpc,
emissions_2023_crf]

# Append all dfs
emissions = append_dfs(dataframes, columns)

# COMMAND ----------

# DBTITLE 1,DF transformations
# Extract year
emissions = emissions.withColumn("year", regexp_replace("year", "Cities ", ""))

# Filter wrong rows
emissions = emissions.filter(col("account").isNotNull())
years = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]
emissions = emissions.filter(col("year").isin(years))

# Add unique id
emissions = emissions.withColumn("account_year", concat(emissions["account"], lit("_"), emissions["year"]))

# COMMAND ----------

display(emissions_.select("inventory_year","inventory_year_").distinct())

# COMMAND ----------

# DBTITLE 1,Normalize Categories
# Normalize categories
# Regular expressions for each case
regex_1 = r'.*(\d{4})[-/]\d{2}[-/]\d{2}.*'  # '1990-01-01 - 1990-12-31'
regex_2 = r'(\d{4})/\d{4}'                 # '2005/2006'
regex_3 = r'.* - (\d{4})[-/]\d{2}[-/]\d{2}' # '- 2018-12-30'
regex_4 = r'^\d{4}$'                       # '2023'
regex_5 = r'\d{2}/\d{2}/(\d{4}) .*'        # '12/31/2005 12:00:00 AM'

# Apply regex to extract year only
emissions = emissions.withColumn("inventory_year", 
    when(regexp_extract(col("inventory_year"), regex_1, 1) != "", regexp_extract(col("inventory_year"), regex_1, 1)).
    when(regexp_extract(col("inventory_year"), regex_2, 1) != "", regexp_extract(col("inventory_year"), regex_2, 1)).
    when(regexp_extract(col("inventory_year"), regex_3, 1) != "", regexp_extract(col("inventory_year"), regex_3, 1)).
    when(regexp_extract(col("inventory_year"), regex_4, 0) != "", regexp_extract(col("inventory_year"), regex_4, 0)).
    when(regexp_extract(col("inventory_year"), regex_5, 1) != "", regexp_extract(col("inventory_year"), regex_5, 1)).
    otherwise(None)
)

emissions = emissions.withColumn("protocol", 
            when((upper(col("protocol")).like("%GPC%")), 'GPC')
            .when((upper(col("protocol")).like("%CRF%")), 'GCom_CRF')
            .when((upper(col("protocol")).like("%ICLEI%")), 'ICLEI')
            .when((upper(col("protocol")).like("%IPCC%")), 'IPCC')
            .when((upper(col("protocol")).like("%SPECIFIC%")), 'Jurisdiction_specific')
            .when(col("protocol").isNull(), None)
            .otherwise('Other'))

emissions = emissions.withColumn("gases_CH4", 
            when((upper(col("gases")).like("%CH4%")), 1)
            .when(col("gases").isNull(), None)
            .otherwise(0))\
.withColumn("gases_HFCs", 
            when((upper(col("gases")).like("%HFCs%")), 1)
            .when(col("gases").isNull(), None)
            .otherwise(0))\
.withColumn("gases_N2O", 
            when((upper(col("gases")).like("%N2O%")), 1)
            .when(col("gases").isNull(), None)
            .otherwise(0))\
.withColumn("gases_NF3", 
            when((upper(col("gases")).like("%NF3%")), 1)
            .when(col("gases").isNull(), None)
            .otherwise(0))\
.withColumn("gases_PFCs", 
            when((upper(col("gases")).like("%PFCs%")), 1)
            .when(col("gases").isNull(), None)
            .otherwise(0))\
.withColumn("gases_SF6", 
            when((upper(col("gases")).like("%SF6%")), 1)
            .when(col("gases").isNull(), None)
            .otherwise(0))               

emissions = emissions.drop("gases")

# COMMAND ----------

display(emissions)

# COMMAND ----------

display(emissions.groupBy("account_year").count().filter(col("count") > 1))

# COMMAND ----------

# Calculate null % for each column
emissions_nulls = calcular_porcentaje_nulos(emissions)

# Remove columnas with null % < percentage_threshold
percentage_threshold = 51
column_to_keep_nulls_others = []

#emissions_2 = filter_columns_by_null_percentage(emissions, emissions_nulls, percentage_threshold, column_to_keep_nulls_others)

display(emissions_nulls)

# COMMAND ----------

# DBTITLE 1,Save df
df_emissions = emissions

# Save dataframe as table in catalog default
df_emissions.write.mode("overwrite").saveAsTable("default.df_emissions")

# Save as CSV file in blob container
#storage_account_name = 'stgaclnrmlab'
#storage_account_access_key = 'rf/4ogYc/eG+oqVD8K9xjsVamosf1qO1s0Kab+ujHsTt0GjaGY2XHfXFNVdti4iaUndCJjNSqizi+ASt8IWqHw=='
#spark.conf.set('fs.azure.account.key.' + storage_account_name + '.blob.core.windows.net', storage_account_access_key)
#
#blob_container = 'citiesrespones'
#file_path = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/df_emissions.csv.gz"
#
## Escribir el DataFrame en un archivo CSV comprimido en Azure Blob Storage
#df_emissions.write.csv(file_path, compression='gzip', header=True)

# COMMAND ----------

df_emissions_c = df_emissions.groupBy('account') \
    .agg(
    count('*').alias('count'))

df_emissions_c.count()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

