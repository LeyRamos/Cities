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
from pyspark.sql.types import DecimalType, IntegerType, DoubleType

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
# One Hot encoding
#-----------------------------------------------------------------------#
def create_binary_columns(df, category_column):
    categories = df.filter(col(category_column).isNotNull()) \
                   .select(category_column).distinct().rdd.flatMap(lambda x: x).collect()
    
    # One Hot encoding for each category in category_column
    for category in categories:
        df = df.withColumn(category, when(col(category_column) == category, 1).otherwise(0))
    
    return df

# COMMAND ----------

df_climate_hazards = spark.table("default.df_climate_hazards")
df_emissions = spark.table("default.df_emissions")

# COMMAND ----------


# Set data types
df_emissions = df_emissions \
    .withColumn("year", col("year").cast(IntegerType())) \
    .withColumn("inventory_year", col("inventory_year").cast(IntegerType())) \
    .withColumn("total_emissions_mtco2e", col("total_emissions_mtco2e").cast(DoubleType())) \
    .withColumn("total_scope1_mtco2e", col("total_scope1_mtco2e").cast(DoubleType())) \
    .withColumn("total_scope2_mtco2e", col("total_scope2_mtco2e").cast(DoubleType())) \
    .withColumn("total_scope3_mtco2e", col("total_scope3_mtco2e").cast(DoubleType()))

# Replace nulls on 'inventory_year'
df_emissions = df_emissions.withColumn('inventory_year', 
                                       when((col('total_emissions_mtco2e').isNotNull() & col('inventory_year').isNull()), col('year'))\
                                       .otherwise(col('inventory_year')))

# Create binary columns for category 'protocol'
df_emissions = create_binary_columns(df_emissions, 'protocol')

#Tranform high double values


# Fill Nulls with 0
df_emissions = df_emissions.na.fill(0)

df_emissions = df_emissions.drop('account', 'year', 'inventory_year', 'protocol')

# Add prefix on columns
df_emissions = AddColumnPrefix(df_emissions, "emissions_", exclude_columns=['account_year'])

# COMMAND ----------

display(df_emissions)

# COMMAND ----------

df_all = df_climate_hazards.join(df_emissions, on='account_year', how='left')

# COMMAND ----------

display(df_climate_hazards.shape())
display(df_emissions.shape())
display(df_all.shape())

# COMMAND ----------

display(df_all)

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import when, col, expr, mean
from pyspark.sql import functions as F

def reemplazar_valores(df: DataFrame, columnas_a_procesar: list, metodo: str) -> DataFrame:
    """
    Reemplaza los valores nulos o infinitos en las columnas especificadas por la media o la mediana.
    
    :param df: El DataFrame de entrada.
    :param columnas_a_procesar: Lista de columnas en las que se desea realizar el reemplazo.
    :param metodo: Método a utilizar para el reemplazo ('media' o 'mediana').
    :return: DataFrame con los valores reemplazados.
    """
    
    if metodo not in ['media', 'mediana']:
        raise ValueError("El método debe ser 'media' o 'mediana'")
    
    if metodo == 'mediana':
        # Calcular la mediana de cada columna
        estadisticas = df.select([F.expr(f'percentile_approx({col_name}, 0.5)').alias(col_name) for col_name in columnas_a_procesar])
    elif metodo == 'media':
        # Calcular la media de cada columna
        estadisticas = df.select([F.mean(col_name).alias(col_name) for col_name in columnas_a_procesar])
    
    # Recoger los valores calculados
    estadisticas = estadisticas.collect()[0]
    
    for col_name in columnas_a_procesar:
        valor_reemplazo = estadisticas[col_name]
        
        # Reemplazar los valores nulos o infinitos por el valor correspondiente
        df = df.withColumn(col_name, when(col(col_name).isNull() | col(col_name).isin(float('inf'), float('-inf')), valor_reemplazo).otherwise(col(col_name)))
    
    return df




# COMMAND ----------

# Uso de la función
#columnas_a_procesar = ['total_emissions_mtco2e', 'total_scope1_mtco2e', 'total_scope2_mtco2e', 'total_scope3_mtco2e']
columnas_a_procesar = ['total_emissions_mtco2e',]
df_emissions_ = reemplazar_valores(df_emissions.filter(df_emissions['total_emissions_mtco2e'] >0), columnas_a_procesar, metodo='mediana')


# COMMAND ----------

# DBTITLE 1,K-means
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler


#df_pca_sp = spark.createDataFrame(df_pca_pd)
#df_pca_sp = df_pca_sp.drop("features", "scaledFeatures", "pcaFeatures")

# Assemble las características en un solo vector
feature_columns = [col for col in df_pca_sp.columns if col not in ("account_year")]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="keep")
df_assembled = assembler.transform(df_pca_sp)

# Entrenar el modelo K-means
kmeans = KMeans().setK(3)  # Definir el número de clusters
model = kmeans.fit(df_assembled)

# Predecir los clusters para los datos de entrada
predictions = model.transform(df_assembled)

# Mostrar los resultados
#predictions.select('account_year', 'prediction').show()


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler

# Step 1: Create a VectorAssembler object
feature_columns = [col for col in df_emissions.columns if col not in ("account_year")]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Step 2: Transform the dataframe using the VectorAssembler
assembled_df = assembler.transform(df_emissions)

# Step 3: Fit the StandardScaler object to the assembled vector column dataframe
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(assembled_df)

# Step 4: Use the transform method of the StandardScalerModel object to scale the assembled vector column
scaled_features = scaler_model.transform(assembled_df)

#StandardScaler class. This class implements a type of feature scaling called standardization. Standardization scales, or shifts, the values for each numerical feature in your dataset so that the features have a mean of 0 and standard deviation of 1

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# Entrenar el modelo K-means
kmeans = KMeans().setK(3)  # Definir el número de clusters
model = kmeans.fit(scaled_features)

# Predecir los clusters para los datos de entrada
predictions = model.transform(scaled_features)

# Mostrar los resultados
#predictions.select('account_year', 'prediction').show()

# COMMAND ----------

display(predictions)

# COMMAND ----------

display(scaled_features)