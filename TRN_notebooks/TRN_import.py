# Databricks notebook source
# MAGIC %pip install folium
# MAGIC !pip install geopandas folium
# MAGIC %pip install azure-storage-blob

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType, TimestampType

from pyspark.sql.functions import explode
from pyspark.sql.functions import posexplode

# COMMAND ----------

#-----------------------------------------------------------------------#
# Get Shape
#-----------------------------------------------------------------------#
def sparkShape(df):
    return (df.count(), len(df.columns))
pyspark.sql.dataframe.DataFrame.shape = sparkShape

#display(df_responses.shape())


# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("LoadCSVFiles").getOrCreate()

storage_account_name = 'stgaclnrmlab'
storage_account_access_key = 'rf/4ogYc/eG+oqVD8K9xjsVamosf1qO1s0Kab+ujHsTt0GjaGY2XHfXFNVdti4iaUndCJjNSqizi+ASt8IWqHw=='

# Configurate access to Blob Storage
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_access_key)

# Blob Storage directory
blob_container = 'data'
directory_path_bel = f"wasbs://{blob_container}@{storage_account_name}.blob.core.windows.net/source/Bel"
directory_path_and = f"wasbs://{blob_container}@{storage_account_name}.blob.core.windows.net/source/And"
directory_path_bxl = f"wasbs://{blob_container}@{storage_account_name}.blob.core.windows.net/source/Bxl"


# COMMAND ----------

# Listar archivos en el directorio
file_list = dbutils.fs.ls(directory_path_bel)
file_list

# COMMAND ----------

# Define schema
schema = StructType([
    StructField("datetime", TimestampType(), True),
    StructField("street_id", StringType(), True),
    StructField("count", IntegerType(), True),
    StructField("speed", FloatType(), True)
])

# Inicializar el DataFrame
df_bel_stream_05m = None
df_bel_stream_15m = None
df_bel_stream_30m = None
df_bel_stream_60m = None

# Recorrer la lista de archivos
for file_info in file_list:
    file_name = file_info.name
    if file_name.startswith("Bel_05min") and file_name.endswith(".csv"):
        file_path = file_info.path
        # Cargar el archivo CSV en un DataFrame temporal
        temp_df = spark.read.format("csv").option("header", "false").schema(schema).load(file_path)
        # Hacer append al DataFrame principal
        if df_bel_stream_05m is None:
            df_bel_stream_05m = temp_df
        else:
            df_bel_stream_05m = df_bel_stream_05m.union(temp_df)
    if file_name.startswith("Bel_15min") and file_name.endswith(".csv"):
        file_path = file_info.path
        # Cargar el archivo CSV en un DataFrame temporal
        temp_df = spark.read.format("csv").option("header", "false").schema(schema).load(file_path)
        # Hacer append al DataFrame principal
        if df_bel_stream_15m is None:
            df_bel_stream_15m = temp_df
        else:
            df_bel_stream_15m = df_bel_stream_15m.union(temp_df)
    if file_name.startswith("Bel_30min") and file_name.endswith(".csv"):
        file_path = file_info.path
        # Cargar el archivo CSV en un DataFrame temporal
        temp_df = spark.read.format("csv").option("header", "false").schema(schema).load(file_path)
        # Hacer append al DataFrame principal
        if df_bel_stream_30m is None:
            df_bel_stream_30m = temp_df
        else:
            df_bel_stream_30m = df_bel_stream_30m.union(temp_df)

# COMMAND ----------

display(df_bel_stream_05m.shape())
display(df_bel_stream_15m.shape())
display(df_bel_stream_30m.shape())

# COMMAND ----------

display(df_bel_stream_15m.filter(df_bel_stream_15m['street_id']== '3265.0'))

# COMMAND ----------

from pyspark.sql.functions import min, max
max_datetime = df_bel_stream_15m.select(max('datetime')).collect()[0][0]
min_datetime = df_bel_stream_15m.select(min('datetime')).collect()[0][0]

print("Máximo datetime:", max_datetime)
print("Mínimo datetime:", min_datetime)

# COMMAND ----------

# Importar las bibliotecas necesarias
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as sum
import matplotlib.pyplot as plt

# Ordenar los datos por la columna 'datetime'
df_bel_stream_15m = df_bel_stream_15m.orderBy('datetime')

# Agrupar los datos por la columna 'datetime' y agregar la suma de la columna 'count'
df_bel_stream_15m_grouped = df_bel_stream_15m.filter(df_bel_stream_15m['datetime'] <= '2019-01-15T13:00:00.000+00:00').groupBy('datetime').agg(sum('count').alias('count'))

# Recopilar los datos para visualizarlos localmente
data = df_bel_stream_15m_grouped.toPandas()

# Graficar los datos
plt.figure(figsize=(20,5))
plt.plot(data['datetime'], data['count'], color='red')
plt.xticks(rotation=45)
plt.title('Belgium')
plt.show()



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def getDuplicateColumns(df):
    # Convertir DataFrame a RDD para manipulación
    rdd = df.rdd

    # Crear un diccionario que agrupe las columnas por sus tipos de datos
    groups = {}
    for field in df.schema.fields:
        if field.dataType not in groups:
            groups[field.dataType] = []
        groups[field.dataType].append(field.name)

    # Lista para almacenar las columnas duplicadas
    dups = []


# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("LoadCSVFiles").getOrCreate()

storage_account_name = 'stgaclnrmlab'
storage_account_access_key = 'rf/4ogYc/eG+oqVD8K9xjsVamosf1qO1s0Kab+ujHsTt0GjaGY2XHfXFNVdti4iaUndCJjNSqizi+ASt8IWqHw=='

# Configurate access to Blob Storage
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_access_key)

# Blob Storage directory
blob_container = 'data'
archivo_geojson = f"wasbs://{blob_container}@{storage_account_name}.blob.core.windows.net/source/geo_data/Belgium_streets.json"

df_bruxelles = gpd.read_file(archivo_geojson)
print('Anderlecht total number of streets ' + str(df_bruxelles.shape[0]))

# Visualizar los datos con Folium
polygons = df_bruxelles

m = folium.Map([50.85045, 4.34878], zoom_start=13, tiles='cartodbpositron')
folium.GeoJson(polygons).add_to(m)

# Guardar el mapa en un archivo HTML
m.save('map.html')

# Mostrar el mapa en un entorno Jupyter Notebook (opcional)
from IPython.display import display, HTML
display(HTML('map.html'))

# COMMAND ----------

import geopandas as gpd
import folium
from azure.storage.blob import BlobServiceClient
import os

# Información de conexión
blob_container = "data"
storage_account_name = "stgaclnrmlab"
storage_account_key = 'rf/4ogYc/eG+oqVD8K9xjsVamosf1qO1s0Kab+ujHsTt0GjaGY2XHfXFNVdti4iaUndCJjNSqizi+ASt8IWqHw=='
blob_name = "source/geo_data/Bruxelles_streets.json"

# Crear un cliente de BlobService
blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=storage_account_key)
container_client = blob_service_client.get_container_client(blob_container)
blob_client = container_client.get_blob_client(blob_name)

# Descargar el archivo localmente
download_file_path = f"./{blob_name.split('/')[-1]}"
with open(download_file_path, "wb") as download_file:
    download_file.write(blob_client.download_blob().readall())

# Leer el archivo GeoJSON con GeoPandas
df_bruxelles = gpd.read_file(download_file_path)
print('Anderlecht total number of streets ' + str(df_bruxelles.shape[0]))

# Visualizar los datos con Folium
polygons = df_bruxelles
m = folium.Map([50.85045, 4.34878], zoom_start=13, tiles='cartodbpositron')
folium.GeoJson(polygons).add_to(m)
m.save('map.html')

# Mostrar el mapa en un entorno Jupyter Notebook (opcional)
from IPython.display import display, HTML
display(HTML('map.html'))

# COMMAND ----------

from pyspark.sql import SparkSession
import geopandas as gpd
import folium
from pyspark.sql import functions as F



# Leer el archivo GeoJSON con GeoPandas
df_bruxelles = gpd.read_file(archivo_geojson)
print('Anderlecht total number of streets ' + str(df_bruxelles.shape[0]))

# Si necesitas convertir el GeoDataFrame a un DataFrame de PySpark
df_bruxelles_spark = spark.createDataFrame(df_bruxelles)

# Convertir las geometrías a WKT (Well-Known Text) para almacenarlas en un DataFrame de PySpark
df_bruxelles_spark = df_bruxelles_spark.withColumn("geometry", F.col("geometry").cast("string"))

# Mostrar algunas filas del DataFrame de PySpark
df_bruxelles_spark.show(5)

# Visualizar los datos con Folium
polygons = df_bruxelles

m = folium.Map([50.85045, 4.34878], zoom_start=13, tiles='cartodbpositron')
folium.GeoJson(polygons).add_to(m)

# Guardar el mapa en un archivo HTML
m.save('map.html')

# Mostrar el mapa en un entorno Jupyter Notebook (opcional)
from IPython.display import display, HTML
display(HTML('map.html'))

# Detener la sesión de Spark
spark.stop()


# COMMAND ----------




# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("LoadCSVFiles").getOrCreate()

storage_account_name = 'stgaclnrmlab'
storage_account_access_key = 'rf/4ogYc/eG+oqVD8K9xjsVamosf1qO1s0Kab+ujHsTt0GjaGY2XHfXFNVdti4iaUndCJjNSqizi+ASt8IWqHw=='

# Configurate access to Blob Storage
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_access_key)

# Blob Storage directory
blob_container = 'data'
archivo_geojson = f"wasbs://{blob_container}@{storage_account_name}.blob.core.windows.net/source/geo_data/Belgium_streets.json"

df_bel_geo = spark.read.json(archivo_geojson)

# COMMAND ----------

df_bel_geo = df_bel_geo.drop('_corrupt_record', 'type')
df_bel_geo = df_bel_geo.filter(df_bel_geo['geometry'].isNotNull())


df_bel_geo = df_bel_geo.select(explode("geometry.coordinates").alias("geometry"))

df_bel_geo = df_bel_geo.select(posexplode("geometry").alias("element", "coordinates")
).select(
    col("element").alias("element"),
    col("coordinates").alias("coordinates")
)

#df_bel_geo = df_bel_geo.withColumn("latitude", col("coordinates").getItem(1)) \
#                 .withColumn("longitude", col("coordinates").getItem(0)) \
 #                .drop("coordinates")

# COMMAND ----------

df_pandas = df_bel_geo.toPandas()

df_pandas['geometry'] = gpd.points_from_xy(df_pandas['coordinates'].apply(lambda x: x[0]), df_pandas['coordinates'].apply(lambda x: x[1]))

# Convertir el DataFrame de pandas con geometrías a un GeoDataFrame
gdf = gpd.GeoDataFrame(df_pandas, geometry='geometry')

gdf = gpd.set_crs(epsg=4326)
print(gdf)

# COMMAND ----------

from fiona.crs import from_epsg
gdf = gdf.set_crs(epsg=4326)

# COMMAND ----------

import folium
import json

# Convert the NumPy array to a Python list
gdf_list = gdf.values.tolist()

# Create a map centered on Belgium
m = folium.Map([50.85045, 4.34878], zoom_start=7, tiles='cartodbpositron')

# Add the GeoDataFrame to the map
folium.GeoJson(json.loads(json.dumps(gdf_list))).add_to(m)

# Add layer controls
folium.LayerControl().add_to(m)

# Show the map
m.save('map.html')


# COMMAND ----------

display(df_bel_geo)

# COMMAND ----------

import pandas as pd

# COMMAND ----------


import geopandas as gpd
import shapely
from pyspark.sql import SparkSession
from shapely.geometry import Point

df_pandas = df_bel_geo.toPandas()

# Crear una GeoDataFrame de GeoPandas a partir del DataFrame de pandas usando las columnas de latitud y longitud
gdf = gpd.GeoDataFrame(df_pandas, geometry=gpd.points_from_xy(df_pandas.longitude, df_pandas.latitude))

# COMMAND ----------

import geopandas as gpd
import matplotlib.pyplot as plt

# Suponiendo que tu GeoDataFrame se llama gdf

# Cargar un mapa base
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plotear el mapa base
ax = world.plot(color='white', edgecolor='black')

# Plotear los puntos del GeoDataFrame
gdf.plot(ax=ax, marker='o', color='red', markersize=5)

# Mostrar el mapa
plt.show()
