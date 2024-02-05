# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Ejemplo de procesamiento de datos utilizando el "million song dataset"
# MAGIC
# MAGIC Vamos a arrancar por listar los contenidos del filesystem. Para esto se puede usar el magic command **%fs** o el objeto **dbutils**

# COMMAND ----------

# MAGIC %fs 
# MAGIC
# MAGIC ls /databricks-datasets/

# COMMAND ----------

# MAGIC %fs 
# MAGIC
# MAGIC ls /databricks-datasets/songs/data-001/

# COMMAND ----------

dbutils.fs.ls('/databricks-datasets/songs/data-001/')

# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets/songs/data-001/'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Examinando los datos

# COMMAND ----------

dataRDD = sc.textFile("/databricks-datasets/songs/data-001/part-00001")
dataRDD.take(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cargando el archivo de header para entender las columnas

# COMMAND ----------

header = sc\
.textFile("/databricks-datasets/songs/data-001/header.txt")\
.map(lambda line: line.split(":"))\
.collect()
# print(sc.textFile("/databricks-datasets/songs/data-001/header.txt").collect())
header[0:5]

# COMMAND ----------

def parseLine(line):
  tokens = zip(line.split("\t"), header)
  parsed_tokens = []
  for token in tokens:
    token_type = token[1][1]
    if token_type == 'double':
      parsed_tokens.append(float(token[0]))
    elif token_type == 'int':
      parsed_tokens.append(-1 if '-' in token[0] else int(token[0])) # Taking care of fields with --
    else:
      parsed_tokens.append(token[0])
  return parsed_tokens

# COMMAND ----------

# MAGIC %md
# MAGIC Generamos una función que mapea del string al tipo de dato

# COMMAND ----------

from pyspark.sql.types import *

def strToType(str):
  if str == 'int':
    return IntegerType()
  elif str == 'double':
    return DoubleType()
  else:
    return StringType()

# COMMAND ----------

schema = StructType([StructField(t[0], strToType(t[1]), True) for t in header])
print(schema)

# COMMAND ----------

## Create Dataframe
parsedRDD = dataRDD.map(parseLine)
# print(dataRDD.map(parseLine).take(3)[0])
df = sqlContext.createDataFrame(parsedRDD,schema)

# COMMAND ----------

## Si lo quieren guardar en un archivo formato parquet 
# df.write \
#   .mode("overwrite") \
#   .format("parquet") \
#   .save("/datasets/songs-parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escribiendo en una tabla

# COMMAND ----------

# %sql 
# CREATE DATABASE IF NOT EXISTS SONGS_DB
# LOCATION "dbfs:/FileStore/songs_db/"

# COMMAND ----------

# %sql
# USE SONGS_DB

# COMMAND ----------

# df.write.saveAsTable("songs_table_saved", format = "parquet", mode = "overwrite", path = "dbfs:/FileStore/songs_db/songs_table")

# COMMAND ----------

# %sql DESCRIBE songs_table_saved

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Trabajar con tablas temporales y Spark SQL

# COMMAND ----------

## Registrar como tabla temporal llamada songsTable
df.registerTempTable("songsTable")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Ya que vamos a consultar muchas veces la tabla es mejor "guardarla" en memoria (cachearla)

# COMMAND ----------

# MAGIC %sql cache table songsTable;

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Ver primeras 10 filas de la tabla
# MAGIC select * from songsTable limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Explorando los datos

# COMMAND ----------

## Ver el esquema de la tabla
df.printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Contar cantidad de filas en la tabla
# MAGIC select count(*) from songsTable

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Calcular el promedio de duración de las canciones por año sobre la tabla songsTable entre 1950 y 2010
# MAGIC
# MAGIC select year, avg(duration) as duration
# MAGIC from songsTable
# MAGIC where year between 1950 and 2010
# MAGIC group by year

# COMMAND ----------

df_all = sqlContext.createDataFrame(sc.textFile("/databricks-datasets/songs/data-001/part-*").map(parseLine), schema)

df_all.registerTempTable("all_songs_table")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Calcular el promedio de duración de las canciones por año sobre la tabla all_songs_table entre 1950 y 2010
# MAGIC
# MAGIC select year, avg(duration) as duration
# MAGIC from all_songs_table
# MAGIC where year between 1950 and 2010
# MAGIC group by year

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Contar la cantidad de líneas en la tabla all_songs_table
# MAGIC select count(*)
# MAGIC from all_songs_table

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Cantidad de canciones por año
# MAGIC select year, count(1) 
# MAGIC from all_songs_table 
# MAGIC where year between 1950 and 2010 
# MAGIC group by year
# MAGIC order by year asc

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Cantidad de canciones por década
# MAGIC select substr(year,1,3), count(1) 
# MAGIC from all_songs_table 
# MAGIC where year between 1950 and 2010 
# MAGIC group by substr(year,1,3) 
# MAGIC order by 1 asc

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# Utilizando el VectorAssembler, crear una nueva columna features.
# Features debe ser un vector conformado por duration, tempo y loudness.
# Utilizar la tabla songsTable
# https://stackoverflow.com/questions/47585723/kmeans-clustering-in-pyspark
trainingData = 

# COMMAND ----------

trainingData.take(3)

# COMMAND ----------

display(trainingData.select("duration", "tempo", "loudness", "features").limit(2))

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# COMMAND ----------

# DBTITLE 1,Fiteando un model KMeans con los parámetros default y k = 4
model = KMeans().setK(4).fit(trainingData)

# COMMAND ----------

model.clusterCenters()

# COMMAND ----------

# DBTITLE 1,Agregamos una columna con la predicción de cada canción a que cluster pertenece
modelTransformed = model.transform(trainingData)
display(modelTransformed)

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(modelTransformed)
print("Silhouette with squared euclidean distance = {s}".format(s=silhouette))

# COMMAND ----------

modelTransformed.printSchema()

# COMMAND ----------

data_with_prediction = modelTransformed.select("duration", "tempo", "loudness", "prediction")

# COMMAND ----------

# DBTITLE 1,Sampleamos para visualizar las predicciones
display(data_with_prediction.sample(fraction = 0.05))


# COMMAND ----------

# Graficar las predicciones en un Scatter Plot usando sample_size de 0.05


# COMMAND ----------

# Entrenar un nuevo modelo (model2) con K = 2 y plotear las predicciones con sample de 0.5

# COMMAND ----------

evaluator2 = ClusteringEvaluator()
silhouette = evaluator2.evaluate(modelTransformed2)
print("Silhouette with squared euclidean distance = {s}".format(s=silhouette))

# COMMAND ----------

WSSSE = model.computeCost(trainingData)
print("Within Set Sum of Squared Errors = {w}".format(w=WSSSE))

# COMMAND ----------

WSSSE2 = model2.computeCost(trainingData)
print("Within Set Sum of Squared Errors 2 = {w}".format(w=WSSSE2))

# COMMAND ----------

# Crear el vector de features sobre la tabla all_songs_table

# COMMAND ----------

modelFull = KMeans().setK(3).fit(fulltrainingData)
modelTransformedFull = modelFull.transform(fulltrainingData)
display(modelTransformedFull.select("duration", "tempo", "loudness", "prediction").sample(fraction = 0.5))