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

# DBTITLE 1,Definicion de funcion para descargar
# Implementar la función dowload_file que sirva para bajar los archivos que se van a utilizar para el análisis de
# https://github.com/juanpampliega/datasets/raw/master/events.csv.gz
# https://github.com/juanpampliega/datasets/raw/master/ginf.csv.gz
# deben guardarse en el directorio /data/eu-league-events/input/

def download_file(file):
  import urllib
  import tempfile
  with urllib.request.urlopen('https://github.com/juanpampliega/datasets/raw/master/' + file) as response:
    gzipcontent = response.read()
  
  # Persiste archivo en tmp
  with open('/tmp/'+file, 'wb') as f:
    f.write(gzipcontent)
  # Copia archivo al path solicitado
  path = '/data/eu-league-events/input/'
  dbutils.fs.cp('file:/tmp/' + file, path + file)

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
          add("id_odsp", StringType()).
          add("id_event", StringType()).
          add("sort_order", IntegerType()).
          add("time", IntegerType()).
          add("text", StringType()).
          add("event_type", IntegerType()).
          add("event_type2", IntegerType()).
          add("side", IntegerType()).
          add("event_team", StringType()).
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

# DBTITLE 1,Reemplazar NAs con valores del diccionario (menos elegante pero anda)
# rellenar los NA con el siguiente diccionario de valores para cada columna
# {'player': 'NA', 'event_team': 'NA', 'opponent': 'NA', 
# 'event_type': 99, 'event_type2': 99, 'shot_place': 99, 
# 'shot_outcome': 99, 'location': 99, 'bodypart': 99, 
# 'assist_method': 99, 'situation': 99}

from pyspark.sql.functions import col, asc, count
diccNA = {'player': 'NA', 'event_team': 'NA', 'opponent': 'NA',  'event_type': 99, 'event_type2': 99, 'shot_place': 99,  
          'shot_outcome': 99, 'location': 99, 'bodypart': 99,  'assist_method': 99, 'situation': 99}
diccNA_list = list(diccNA.keys())

eventsDf_sinNA_tmp = eventsDf.na.fill(value='NA',subset=['player', 'event_team','opponent'])
eventsDf = eventsDf_sinNA_tmp.na.fill(value=99,subset=['event_type', 'event_type2','shot_place', 'shot_outcome', 'location', 'bodypart', 'assist_method', 'situation'])

# Verificar que la cantida de nulos en las columnas del diccionario son nulas
Dict_Null = {col:eventsDf.filter(eventsDf[col].isNull()).count() for col in diccNA_list}
Dict_Null

# COMMAND ----------

display(eventsDf)

# COMMAND ----------

# DBTITLE 1,Revisar el CSV de información de los partidos
partidos = sc.textFile("dbfs:/data/eu-league-events/input/ginf.csv.gz")
print(partidos.take(10))

# COMMAND ----------

# DBTITLE 1,Especificar el esquema de la tabla ginf.csv
from pyspark.sql.types import *

schema_partidos = (StructType().
                    add("id_odsp", StringType()).
                    add("link_odsp", StringType()).
                    add("adv_stats", BooleanType()).
                    add("date", DateType()).
                    add("league", StringType()).
                    add("season", StringType()).
                    add("country", StringType()).
                    add("ht", StringType()).
                    add("at", StringType()).
                    add("fthg", IntegerType()).
                    add("ftag", IntegerType()).
                    add("odd_h", FloatType()).         
                    add("odd_d", FloatType()).
                    add("odd_a", FloatType()).
                    add("odd_over", IntegerType()).
                    add("odd_under", IntegerType()).
                    add("odd_bts", IntegerType()).
                    add("odd_bts_n", IntegerType())
                  )

# COMMAND ----------

# DBTITLE 1,Cargar a un Dataframe el CSV con la información de los partidos 
# Cargar el CSV a Dataframe en la variable gameInfDf usando la inferencia de esquema
# gameInfDf 
gameInfDf = (spark.read.csv("/data/eu-league-events/input/ginf.csv.gz", 
                         schema=schema_partidos, header=True, 
                         ignoreLeadingWhiteSpace=True, 
                         ignoreTrailingWhiteSpace=True,
                         nullValue='NA'))
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


# COMMAND ----------

# DBTITLE 1,Crear joinedDf
# Crear un nuevo DataFrame nombrado joinedDF realizando un join entre los dataframes gameInfDf y eventsDf con los siguientes campos:
# eventsDf.id_odsp, eventsDf.id_event, eventsDf.sort_order, eventsDf.time, eventsDf.event_type, eventsDf.event_type_str, eventsDf.event_type2, eventsDf.event_type2_str, eventsDf.side, eventsDf.side_str, eventsDf.event_team, eventsDf.opponent, eventsDf.player, eventsDf.player2, eventsDf.player_in, eventsDf.player_out, eventsDf.shot_place, eventsDf.shot_place_str, eventsDf.shot_outcome, eventsDf.shot_outcome_str, eventsDf.is_goal, eventsDf.location, eventsDf.location_str, eventsDf.bodypart, eventsDf.bodypart_str, eventsDf.assist_method, eventsDf.assist_method_str, eventsDf.situation, eventsDf.situation_str, gameInfDf.country_code 

joinedDf = eventsDf.join(gameInfDf, eventsDf.id_odsp == gameInfDf.id_odsp, 'inner').select(
eventsDf.id_odsp, eventsDf.id_event, eventsDf.sort_order, eventsDf.time, eventsDf.event_type, eventsDf.event_type_str, eventsDf.event_type2, eventsDf.event_type2_str, eventsDf.side, eventsDf.side_str, eventsDf.event_team, eventsDf.opponent, eventsDf.player, eventsDf.player2, eventsDf.player_in, eventsDf.player_out, eventsDf.shot_place, eventsDf.shot_place_str, eventsDf.shot_outcome, eventsDf.shot_outcome_str, eventsDf.is_goal, eventsDf.location, eventsDf.location_str, eventsDf.bodypart, eventsDf.bodypart_str, eventsDf.assist_method, eventsDf.assist_method_str, eventsDf.situation, eventsDf.situation_str, gameInfDf.country_code, gameInfDf.league, gameInfDf.season)
display(joinedDf)

# COMMAND ----------

# DBTITLE 1,Create time bins for game events
from pyspark.ml.feature import QuantileDiscretizer

# Utilizar el QuantileDiscretizer para crear una nueva columna llamada time_bin que genere 10 bins para el valor de la columna time https://spark.apache.org/docs/latest/ml-features.html#quantilediscretizer
discretizer = QuantileDiscretizer(numBuckets=10, inputCol="time", outputCol="time_bin")

joinedDf = discretizer.fit(joinedDf).transform(joinedDf)

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
#joinedDf.write.saveAsTable()

#Crear tabla external .format("parquet") not supported by Spark 2.2
#joinedDfparq.write.mode('overwrite').partitionBy("country_code").option("path", "/FileStore/itba-hpgvd/eu-league-events/interm/tr-events").saveAsTable("GAME_EVENTS")

joinedDf.write.saveAsTable("GAME_EVENTS" , format = "parquet", partitionBy = "country_code", mode = "overwrite", path = "/FileStore/itba-hpgvd/eu-league-events/interm/tr-events")

#Metadata refreshing. Spark SQL caches Parquet metadata for better performance.
spark.catalog.refreshTable("GAME_EVENTS")

# Verificar que se creó en el catálogo
#spark.catalog.listTables()

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

# DBTITLE 1,Función 1
# Recibe un id_odsp de un partido y devuelve un dataframe con todos los eventos que ocurrieron en el mismo ordenados cronológicamente.
# Para ordenamiento cronológico se usó el campo sort_order, que indica la secuencia cronológica de eventos

def events_finder (var_id_odsp):
  df_result1 = spark.sql("SELECT * FROM GAME_EVENTS WHERE id_odsp = '{}' ORDER BY sort_order".format(var_id_odsp))
  display(df_result1)
  return df_result1


# COMMAND ----------

# Uso de la función 1 para encontrar todos los eventos del id_odsp = Wn69eU5B/
var_id_odsp = "Wn69eU5B/"
df_result1 = events_finder (var_id_odsp)

# COMMAND ----------

# DBTITLE 1,Funcion generica que genera tabla de partidos con su equipo ganador, perdedor y scores
def winner_def (input_df):
  #Registrar df como tabla temporal para usarla
  input_df.registerTempTable('input_df_table')
  df_winners = spark.sql("WITH  \
event_team1 AS ( SELECT id_odsp,  event_team,  SUM(is_goal) AS team_goals FROM input_df_table GROUP BY id_odsp, event_team ), \
event_team2 AS ( SELECT id_odsp,  event_team,  SUM(is_goal) AS team_goals FROM input_df_table GROUP BY id_odsp, event_team ), \
event_join_winner AS ( SELECT E1.id_odsp AS id_odsp, E1.event_team AS winner_, E2.event_team AS looser_, E1.team_goals AS winner_goals, E2.team_goals AS looser_goals FROM event_team1 AS E1, event_team2 AS E2 WHERE E1.id_odsp = E2.id_odsp AND E1.team_goals > E2.team_goals GROUP BY E1.id_odsp, E1.event_team, E2.event_team, E1.team_goals, E2.team_goals ORDER BY E1.id_odsp ), \
event_join_draw AS( SELECT E1.id_odsp AS id_odsp, FIRST(E1.event_team) AS winner_, FIRST(E2.event_team) AS looser_, FIRST(E1.team_goals) AS winner_goals, FIRST(E2.team_goals) AS looser_goals FROM event_team1 AS E1, event_team2 AS E2 WHERE E1.id_odsp = E2.id_odsp AND E1.event_team <> E2.event_team AND E1.team_goals = E2.team_goals GROUP BY E1.id_odsp ), \
events_union AS ( SELECT W.id_odsp, W.winner_, W.looser_, W.winner_goals, W.looser_goals FROM event_join_winner AS W UNION  SELECT D.id_odsp, D.winner_, D.looser_, D.winner_goals, D.looser_goals FROM event_join_draw AS D) \
SELECT id_odsp, winner_, looser_, winner_goals, looser_goals, CASE WHEN (winner_goals = looser_goals) THEN 'Draw' ELSE winner_ END as winner_label FROM events_union ORDER BY id_odsp")
  return df_winners

# COMMAND ----------

# Prueba de funcion que genera tabla de partidos con su equipo ganador, perdedor y scores
testwinner = winner_def (df_result1)

# COMMAND ----------

# DBTITLE 1,Función 2
# Recibe un id_odsp y retorna el nombre del equipo que ganó el partido o en el caso de ser empate el string 'Draw'.
from pyspark.sql.functions import col

def winner_finder (var_id_odsp):
  df_result2 = spark.sql("SELECT * FROM GAME_EVENTS WHERE id_odsp = '{}'".format(var_id_odsp))

  # Aplicon funcion generica de buscar ganador
  winner_table = winner_def (df_result2)
  return print (winner_table.collect()[0][5])

# COMMAND ----------

# Uso de la función 2 para encontrar el equipo ganador / Caso Ganador
var_id_odsp = "02zs6b5s/"
winner_finder (var_id_odsp)

# COMMAND ----------

# Uso de la función 2 para encontrar el equipo ganador / Caso Empate
var_id_odsp = "00OX4xFp/"
winner_finder (var_id_odsp)

# COMMAND ----------

# DBTITLE 1,Funcion 3
#Para un COUNTRY_CODE, crear una tabla en formato parquet que tiene una fila por partido con la siguientes columnas: id_odsp, winner (nombre del equipo o draw), goals_home, goals_away, corners_home, corners_away, fouls_home, fouls_awas, yellow_cards_home, yellow_cards_away.
from pyspark.sql import functions as F

def country_summary (var_country):
  df_result3 = spark.sql("SELECT * FROM GAME_EVENTS WHERE country_code = '{}' ORDER BY id_odsp".format(var_country))
  
  # Preparar dataframe con categorias input para funcion pivot
  # Diccionario para reemplazar concatenacion de campos 
  events_side_Dic={'Yellow card_Away':'yellow_cards_away','Yellow card_Home':'yellow_cards_home','Foul_Home':'fouls_home','Foul_Away':'fouls_away','Corner_Home':'corners_home','Corner_Away':'corners_away'}
  # Agrego columna concatenada con valores validos usando la funcion de mapping
  df_result3 = df_result3.withColumn("events_side", mapKeyToVal(events_side_Dic) (F.concat(df_result3["event_type_str"],F.lit('_'),df_result3["side_str"])))
 
  # Genero primera tabla pivot que suma los goles del partido
  df_rslt3_piv1 = df_result3.withColumn("side_str",F.concat(F.lit('goals_'),df_result3["side_str"])).groupby("id_odsp").pivot("side_str").sum("is_goal")
  
  # Genero segunda tabla pivot que suma otros eventos del partido
  df_rslt3_piv2 = df_result3.groupBy("id_odsp").pivot("events_side", ['corners_away','corners_home', 'fouls_away', 'fouls_home', 'yellow_cards_home', 'yellow_cards_away']).count()

  # Join de ambas tablas pivot ,df_rslt3_piv1.winner
  df_rslt3_joined = df_rslt3_piv1.join(df_rslt3_piv2, df_rslt3_piv1.id_odsp == df_rslt3_piv2.id_odsp). select(
  df_rslt3_piv1.id_odsp,
  df_rslt3_piv1.goals_Home,
  df_rslt3_piv1.goals_Away,
  df_rslt3_piv2.corners_home,
  df_rslt3_piv2.corners_away, 
  df_rslt3_piv2.fouls_home,
  df_rslt3_piv2.fouls_away,
  df_rslt3_piv2.yellow_cards_home,
  df_rslt3_piv2.yellow_cards_away)
    
  #Agregar el winner
  df_rslt3_winner = winner_def(df_result3)
  df_rslt3_joined_winner = df_rslt3_joined.join(df_rslt3_winner, df_rslt3_joined.id_odsp == df_rslt3_winner.id_odsp).select(
  df_rslt3_joined.id_odsp,
  df_rslt3_winner.winner_label,
  df_rslt3_joined.goals_Home,
  df_rslt3_joined.goals_Away,
  df_rslt3_joined.corners_home,
  df_rslt3_joined.corners_away, 
  df_rslt3_joined.fouls_home,
  df_rslt3_joined.fouls_away,
  df_rslt3_joined.yellow_cards_home,
  df_rslt3_joined.yellow_cards_away)
  
  return df_rslt3_joined_winner

# COMMAND ----------

# Uso de Funcion 3
var_country = "DEU"
display(country_summary(var_country))

# COMMAND ----------

# DBTITLE 1,Funcion 4
# Para un COUTRY_CODE retorna una lista de nombres de equipos tuvieron al menos 10 victorias de "visitantes".
from pyspark.sql import functions as F

def winners_visitors (var_country):
  # Usa funcion 3 para generar la tabla pivot para el pais solicitado. ESta tabla se usara como base para calcular la salida de esta funcion
  country_summary(var_country).registerTempTable('df_teams_table')
  df_teams_winners = spark.sql("SELECT winner_label AS team, COUNT(winner_label) AS victories_as_visitor FROM df_teams_table WHERE winner_label <> 'Draw' AND goals_Away > goals_Home GROUP BY winner_label HAVING victories_as_visitor > 10 ORDER BY victories_as_visitor DESC")
  return df_teams_winners

# COMMAND ----------

# Uso de Funcion 4
display(winners_visitors ("FRA"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Visualizaciones
# MAGIC
# MAGIC Utilizando Spark SQL sobre la tabla creada y las visualizaciones de Databricks (función display()), recrear al menos 4 visualizaciones de las que se muestran en el siguiente notebook de Kaggle https://www.kaggle.com/ahmedyoussef/the-beautiful-game-analysis-of-football-events

# COMMAND ----------

# DBTITLE 1,Visualizacion 1: Cantidad de tarjetas promedio por Partido para cada Liga
# MAGIC %sql
# MAGIC WITH
# MAGIC match_total_cards AS(
# MAGIC SELECT league, id_odsp, COUNT(id_odsp) AS match_cards
# MAGIC FROM GAME_EVENTS
# MAGIC WHERE event_type_str = "Red card" OR event_type_str = "Yellow card" OR event_type_str = "Second yellow card"
# MAGIC GROUP BY league, id_odsp
# MAGIC )
# MAGIC SELECT league AS League, ROUND(AVG(match_cards),1) AS AvgCardsPerMatch
# MAGIC FROM match_total_cards
# MAGIC GROUP BY league
# MAGIC ORDER BY league
# MAGIC
# MAGIC -- Incluye Tarjetas Amarillas (1era y 2da) y Rojas

# COMMAND ----------

# DBTITLE 1,Visualizacion 2 : Goles por cantidad de tiros para equipos mas eficientes por Liga
# MAGIC %sql
# MAGIC WITH
# MAGIC team_goals_shots AS(
# MAGIC SELECT league, event_team,
# MAGIC        COUNT(event_type_str) AS count_events,
# MAGIC        SUM(is_goal) AS count_goals, 
# MAGIC        ROUND(SUM(is_goal)/COUNT(event_type_str),2) AS goals_per_shots,
# MAGIC        ROW_NUMBER() OVER(PARTITION BY league ORDER BY ROUND(SUM(is_goal)/COUNT(event_type_str),2) DESC) AS row_number
# MAGIC FROM GAME_EVENTS
# MAGIC WHERE event_type_str = "Attempt" -- Segun la data, los unicos eventos asociados a goles son de este tipo
# MAGIC GROUP BY league, event_team
# MAGIC )
# MAGIC
# MAGIC SELECT league AS League, event_team AS Team, count_events, count_goals, goals_per_shots AS GoalsPerShots, row_number
# MAGIC FROM team_goals_shots
# MAGIC WHERE row_number < 10 -- Se muestran los primeros 10 equipos por League, de acuerdo al indice de eficiencia de goles por tiros goals_per_shots (como en la página de referencia)
# MAGIC ORDER BY goals_per_shots DESC

# COMMAND ----------

# DBTITLE 1,Visualizacion 3: Top goleadores por liga
# MAGIC %sql
# MAGIC WITH
# MAGIC player_goals AS(
# MAGIC SELECT player, league, SUM(is_goal) AS count_goals, ROW_NUMBER() OVER(PARTITION BY league ORDER BY SUM(is_goal) DESC) AS row_number
# MAGIC   FROM GAME_EVENTS 
# MAGIC   GROUP BY player,league
# MAGIC )
# MAGIC
# MAGIC SELECT league AS League, player AS Player, count_goals AS Goals
# MAGIC FROM player_goals
# MAGIC WHERE row_number < 4 -- Se muestran los primeros 4 jugadores goleadores por League, de acuerdo a la cantidad de goles (como en la página de referencia)
# MAGIC ORDER BY count_goals DESC
# MAGIC