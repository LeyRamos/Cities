# Databricks notebook source
# MAGIC %md # Análisis de dataset del fútbol Europeo
# MAGIC
# MAGIC El dataset fue obtenido de [**Kaggle**](https://www.kaggle.com/secareanualin/football-events). 
# MAGIC Provee una visión granular de 9,074 partidos, de las 5 ligas más importantes de Europa (Inglaterra, España, Alemania, Italia y Francia).
# MAGIC Comprende las termoradas desde 2011/2012 hasta 2016/2017. 
# MAGIC
# MAGIC Debajo se muestra una lista de las columnas:
# MAGIC
# MAGIC | Column Name | Colum Description |
# MAGIC | ----------- | ----------------- |
# MAGIC | id_odsp | unique identifier of game (odsp stands from oddsportal.com) |
# MAGIC | id_event | unique identifier of event (id_odsp + sort_order) |
# MAGIC | sort_order | chronological sequence of events in a game |
# MAGIC | time | minute of the game |
# MAGIC | text | text commentary |
# MAGIC | event_type | primary event, 11 unique events |
# MAGIC | event_type2 | secondary event, 4 unique events |
# MAGIC | side | Home or Away team |
# MAGIC | event_team | team that produced the event. In case of Own goals, event team is the team that benefited from the own goal |
# MAGIC | opponent | opposing team |
# MAGIC | player | name of the player involved in main event |
# MAGIC | player2 | name of player involved in secondary event |
# MAGIC | player_in | player that came in (only applies to substitutions) |
# MAGIC | player_out | player substituted (only applies to substitutions) |
# MAGIC | shot_place | placement of the shot, 13 possible placement locations |
# MAGIC | shot_outcome | 4 possible outcomes |
# MAGIC | is_goal | binary variable if the shot resulted in a goal (own goals included) |
# MAGIC | location | location on the pitch where the event happened, 19 possible locations |
# MAGIC | bodypart | 3 body parts |
# MAGIC | assist_method | in case of an assisted shot, 5 possible assist methods |
# MAGIC | situation | 4 types |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ##1. Carga de los datos
# MAGIC

# COMMAND ----------

# DBTITLE 1,Crear las carpetas donde se van almacenar los datos
dbutils.fs.mkdirs("/data/eu-league-events")
dbutils.fs.mkdirs("/data/eu-league-events/input")
dbutils.fs.mkdirs("/data/eu-league-events/interm")

# COMMAND ----------

dbutils.fs.ls("/data/eu-league-events/")

# COMMAND ----------

import urllib

with urllib.request.urlopen('https://github.com/juanpampliega/datasets/raw/master/events.csv.gz') as response:
  gzipcontent = response.read()
  
with urllib.request.urlopen('https://github.com/juanpampliega/datasets/raw/master/ginf.csv.gz') as response:
  gzipcontent = response.read()


# COMMAND ----------

# Implementar la función dowload_file que sirva para bajar los archivos que se van a utilizar para el análisis de
# https://github.com/juanpampliega/datasets/raw/master/events.csv.gz
# https://github.com/juanpampliega/datasets/raw/master/ginf.csv.gz
# deben guardarse en el directorio /data/eu-league-events/input/




import urllib

with urllib.request.urlopen('https://github.com/juanpampliega/datasets/raw/master/events.csv.gz') as response:
  gzipcontent = response.read()

with open("/tmp/don-quijote.txt.gz", 'wb') as f:
  f.write(gzipcontent)

dbutils.fs.cp("file:/tmp/don-quijote.txt.gz",'/tmp/')


# COMMAND ----------

files = ['events.csv.gz', 'ginf.csv.gz']
for f in files:
  download_file(f)

# COMMAND ----------

# DBTITLE 1,Check si los archivos se bajaron correctamente
dbutils.fs.ls('/data/eu-league-events/input/')

# COMMAND ----------

# DBTITLE 1,Revisar el CSV de eventos a ver que tiene
events = sc.textFile("dbfs:/data/eu-league-events/input/events.csv.gz")
print(events.take(10))

# COMMAND ----------

# DBTITLE 1,Especificar el esquema de la tabla events.csv
from pyspark.sql.types import *

schema = (StructType().
          add("id_odsp", StringType()).add("id_event", StringType()).add("sort_order", IntegerType()).
          add("time", IntegerType()).add("text", StringType()).add("event_type", IntegerType()).
          add("event_type2", IntegerType()).add("side", IntegerType()).add("event_team", StringType()).
          add("opponent", StringType()).add("player", StringType()).add("player2", StringType()).
          add("player_in", StringType()).add("player_out", StringType()).add("shot_place", IntegerType()).
          add("shot_outcome", IntegerType()).add("is_goal", IntegerType()).add("location", IntegerType()).
          add("bodypart", IntegerType()).add("assist_method", IntegerType()).add("situation", IntegerType()).
          add("fast_break", IntegerType())
         )

# COMMAND ----------

# DBTITLE 1,Crear un DataFrame con el contenido del archivo events.csv 
eventsDf = (spark.read.csv("/data/eu-league-events/input/events.csv.gz", 
                         schema=schema, header=True, 
                         ignoreLeadingWhiteSpace=True, 
                         ignoreTrailingWhiteSpace=True,
                         nullValue='NA'))

# rellenar los NA con el siguiente diccionario de valores para cada columna
# {'player': 'NA', 'event_team': 'NA', 'opponent': 'NA', 
# 'event_type': 99, 'event_type2': 99, 'shot_place': 99, 
# 'shot_outcome': 99, 'location': 99, 'bodypart': 99, 
# 'assist_method': 99, 'situation': 99}

display(eventsDf)

# COMMAND ----------

# DBTITLE 1,Cargar a un Dataframe el CSV con la información de los partidos 
# Cargar el CSV a Dataframe en la variable gameInfDf usando la inferencia de esquema
# gameInfDf 
display(gameInfDf)

# COMMAND ----------

# MAGIC %md ## 2. Transformación de los datos

# COMMAND ----------

# DBTITLE 1,Definimos una función genérica de clave a valor de mapa
def mapKeyToVal(mapping):
    def mapKeyToVal_(col):
        return mapping.get(col)
    return udf(mapKeyToVal_, StringType())

# COMMAND ----------

# DBTITLE 1,Mapeo de diccionarios (dictionary.txt de Kaggle)
evtTypeMap = {0:'Announcement', 1:'Attempt', 2:'Corner', 3:'Foul', 4:'Yellow card', 5:'Second yellow card', 6:'Red card', 7:'Substitution', 8:'Free kick won', 9:'Offside', 10:'Hand ball', 11:'Penalty conceded', 99:'NA'}

evtTyp2Map = {12:'Key Pass', 13:'Failed through ball', 14:'Sending off', 15:'Own goal', 99:'NA'}

sideMap = {1:'Home', 2:'Away'}

shotPlaceMap = {1:'Bit too high', 2:'Blocked', 3:'Bottom left corner', 4:'Bottom right corner', 5:'Centre of the goal', 6:'High and wide', 7:'Hits the bar', 8:'Misses to the left', 9:'Misses to the right', 10:'Too high', 11:'Top centre of the goal', 12:'Top left corner', 13:'Top right corner', 99:'NA'}

shotOutcomeMap = {1:'On target', 2:'Off target', 3:'Blocked', 4:'Hit the bar', 99:'NA'}

locationMap = {1:'Attacking half', 2:'Defensive half', 3:'Centre of the box', 4:'Left wing', 5:'Right wing', 6:'Difficult angle and long range', 7:'Difficult angle on the left', 8:'Difficult angle on the right', 9:'Left side of the box', 10:'Left side of the six yard box', 11:'Right side of the box', 12:'Right side of the six yard box', 13:'Very close range', 14:'Penalty spot', 15:'Outside the box', 16:'Long range', 17:'More than 35 yards', 18:'More than 40 yards', 19:'Not recorded', 99:'NA'}

bodyPartMap = {1:'Right foot', 2:'Left foot', 3:'Head', 99:'NA'}

assistMethodMap = {0:'None', 1:'Pass', 2:'Cross', 3:'Headed pass', 4:'Through ball', 99:'NA'}

situationMap = {1:'Open play', 2:'Set piece', 3:'Corner', 4:'Free kick', 99:'NA'}

countryCodeMap = {'germany':'DEU', 'france':'FRA', 'england':'GBR', 'spain':'ESP', 'italy':'ITA'}

# COMMAND ----------

# DBTITLE 1,Map country names to codes
gameInfDf = gameInfDf.withColumn("country_code", mapKeyToVal(countryCodeMap)("country"))

display(gameInfDf['id_odsp','country','country_code'])

# COMMAND ----------

# DBTITLE 1,Transform game events data using lookups and join with high-level info
eventsDf = (
             eventsDf.
             withColumn("event_type_str", mapKeyToVal(evtTypeMap)("event_type")).
             withColumn("event_type2_str", mapKeyToVal(evtTyp2Map)("event_type2")).
             withColumn("side_str", mapKeyToVal(sideMap)("side")).
             withColumn("shot_place_str", mapKeyToVal(shotPlaceMap)("shot_place")).
             withColumn("shot_outcome_str", mapKeyToVal(shotOutcomeMap)("shot_outcome")).
             withColumn("location_str", mapKeyToVal(locationMap)("location")).
             withColumn("bodypart_str", mapKeyToVal(bodyPartMap)("bodypart")).
             withColumn("assist_method_str", mapKeyToVal(assistMethodMap)("assist_method")).
             withColumn("situation_str", mapKeyToVal(situationMap)("situation"))
           )

# Crear un nuevo DataFrame nombrado joinedDF realizando un join entre los dataframes gameInfDf y eventsDf con los siguientes campos:
# eventsDf.id_odsp, eventsDf.id_event, eventsDf.sort_order, eventsDf.time, eventsDf.event_type, eventsDf.event_type_str, eventsDf.event_type2, eventsDf.event_type2_str, eventsDf.side, eventsDf.side_str, eventsDf.event_team, eventsDf.opponent, eventsDf.player, eventsDf.player2, eventsDf.player_in, eventsDf.player_out, eventsDf.shot_place, eventsDf.shot_place_str, eventsDf.shot_outcome, eventsDf.shot_outcome_str, eventsDf.is_goal, eventsDf.location, eventsDf.location_str, eventsDf.bodypart, eventsDf.bodypart_str, eventsDf.assist_method, eventsDf.assist_method_str, eventsDf.situation, eventsDf.situation_str, gameInfDf.country_code 

#joinedDf = 


# COMMAND ----------

# DBTITLE 1,Create time bins for game events
from pyspark.ml.feature import QuantileDiscretizer

# Utilizar el QuantileDiscretizer para crear una nueva columna llamada time_bin que genere 10 bins para el valor de la columna time https://spark.apache.org/docs/latest/ml-features.html#quantilediscretizer
display(joinedDf)

# COMMAND ----------

# MAGIC %md ## 3. Carga de los datos a una tabla

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE DATABASE IF NOT EXISTS EURO_LEAGUE_DB
# MAGIC LOCATION "dbfs:/FileStore/itba-hpgvd/eu-league-events/interm"

# COMMAND ----------

# DBTITLE 0,Set the database in session
# MAGIC %sql
# MAGIC USE EURO_LEAGUE_DB

# COMMAND ----------

# DBTITLE 1,Crear la tabla GAME_EVENTS
# Cargar el Dataframe resultante a la tabla GAME_EVENTS en formato Parquet y particionada por el campo COUNTRY_CODE en el path 
# dbfs:/FileStore/itba-hpgvd/eu-league-events/interm/tr-events
joinedDf.write.saveAsTable()

# COMMAND ----------

# MAGIC %sql DESCRIBE GAME_EVENTS

# COMMAND ----------

# MAGIC %md ## 4. Ejercicio: Análisis de datos
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Funciones de procesamiento de datos
# MAGIC
# MAGIC
# MAGIC En esta sección deberá desarrolar una función por punto que utilizando la tabla `GAME_EVENTS` como fuente de datos retorne lo pedido en cada punto. La función a su vez puede llamar a funciona auxiliares secundarias. Agregue una demostración del uso de la función y su correcto funcionamiento. Para completar esta sección debe resolver al menos 4 de los 5 puntos.
# MAGIC
# MAGIC
# MAGIC 1. Recibe un `id_odsp` de un partido y devuelve un dataframe con todos los eventos que ocurrieron en el mismo ordenados cronológicamente.
# MAGIC 2. Recibe un `id_odsp` y retorna el nombre del equipo que ganó el partido o en el caso de ser empate el string 'Draw'.
# MAGIC 3. Para un `COUNTRY_CODE`, crear una tabla en formato parquet que tiene una fila por partido con la siguientes columnas: id_odsp, winner (nombre del equipo o draw), goals_home, goals_away, corners_home, corners_away, fouls_home, fouls_awas, yellow_cards_home, yellow_cards_away. 
# MAGIC 4. Para un `COUTRY_CODE` retorna una lista de nombres de equipos tuvieron al menos 10 victorias de "visitantes". (ayudas: https://databricks.com/blog/2016/02/09/reshaping-data-with-pivot-in-apache-spark.html https://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html )
# MAGIC 5. Recibe un nombre de equipo y devuelve la cantidad máxima de tiempo que estuvo sin realizar un gol.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Visualizaciones
# MAGIC
# MAGIC Utilizando Spark SQL sobre la tabla creada y las visualizaciones de Databricks (función display()), recrear al menos 4 visualizaciones de las que se muestran en el siguiente notebook de Kaggle https://www.kaggle.com/ahmedyoussef/the-beautiful-game-analysis-of-football-events