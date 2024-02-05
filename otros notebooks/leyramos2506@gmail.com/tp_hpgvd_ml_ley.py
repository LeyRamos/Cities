# Databricks notebook source
# MAGIC %md ## Machine Learning sobre el dataset del fútbol Europeo
# MAGIC
# MAGIC Para poder ejecutar este notebook se requiere haber procesado la información y generado la base de datos y la tabla correspondiente con el primer notebook.

# COMMAND ----------

# MAGIC %sql USE EURO_LEAGUE_DB

# COMMAND ----------

# MAGIC %sql SELECT * FROM GAME_EVENTS LIMIT 30

# COMMAND ----------

# MAGIC %sql cache table GAME_EVENTS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejercicio:
# MAGIC
# MAGIC Clusterizar los eventos utilizando KMeans usando al menos 3 features basadas en las columnas de la tabla GAME_EVENTS.
# MAGIC Evaluar la clusterización utilizando la métrica de Silhouette y elegir el mejor K entre 2 y 12.
# MAGIC Graficar la clusterización resultante utilizando distintos gráficos que muestren la combinación de features. (0.5 puntos)
# MAGIC
# MAGIC https://spark.apache.org/docs/latest/ml-clustering.html
# MAGIC
# MAGIC https://stackoverflow.com/questions/47585723/kmeans-clustering-in-pyspark

# COMMAND ----------

# Guardar tabla en df
df_game_events = spark.sql("SELECT * FROM GAME_EVENTS")
df_game_events.printSchema()

# COMMAND ----------

# DBTITLE 1,Crear vector feature - variables numericas
from pyspark.ml.feature import VectorAssembler

#vec1 = VectorAssembler(inputCols=["event_type","event_type2", "is_goal", "side"], outputCol="features")
#trainingData = vec1.transform(df_game_events)

vec2 = VectorAssembler(inputCols=["event_type","event_type2", "is_goal", "shot_place", "time"], outputCol="features")
trainingData = vec2.transform(df_game_events)


#display(trainingData.select("event_type", "event_type2","is_goal", "side", "features").limit(10))
display(trainingData.select("event_type","event_type2", "is_goal", "shot_place", "time").limit(10))

# COMMAND ----------

# DBTITLE 1,Generacion del modelo KMeans para diferentes k. Calculo de metrica Silhouette
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

k_Dic = {}
for k in range(2, 12):
    model = KMeans().setK(k).setSeed(1).fit(trainingData)
    #Generar predicciones
    predictions = model.transform(trainingData)
    
    # Evaluar el clustering mediante métrica de Silhouette
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette) + " for K= "+ str(k))
    
    #Acumular valores de silhouette para cada K
    k_Dic[k] = silhouette
    
    # Centroides de cada cluster
    #centers = model.clusterCenters()
    #print("Cluster Centers: ")
    #for center in centers:
    #    print(center)
        
# Modelo superador
max_k = max(k_Dic, key=lambda q: k_Dic[q])
print("El modelo con mayor valor de Silhouette es para k = " + str(max_k))
#valor alto de silhouette indica que el objeto está bien emparejado con su propio cluster y mal emparejado con los clusters vecinos.

# COMMAND ----------

# Modelo con k = 4, máximo valor de Silhouette
model_k2 = KMeans().setK(4).setSeed(1).fit(trainingData)
predictions_k2 = model_k2.transform(trainingData)

# COMMAND ----------

# DBTITLE 1,Grafico del los clusters resultantes para todas las combinaciones de features
display(predictions_k2.select("event_type","event_type2", "is_goal", "shot_place", "time", "prediction").sample(fraction = 0.5))

# COMMAND ----------

display(predictions_k2.select("event_type","event_type2", "is_goal", "shot_place", "time", "prediction").sample(fraction = 0.5))

# COMMAND ----------

# DBTITLE 1,Distribucion de predicciones en el conjunto
display(predictions_k2.sample(fraction = 0.5))

# COMMAND ----------

# MAGIC %md ## Ejercicio: GBT Classifier
# MAGIC A continuación vamos a utilizar [Gradient-boosted tree](https://spark.apache.org/docs/2.3.0/ml-classification-regression.html#gradient-boosted-tree-classifier) para fitear un modelo y predecir la combinación de condiciones de un evento que pueden llevar a un gol.

# COMMAND ----------

# DBTITLE 1,Seleccionamos una lista de columnas utilizadas para generar las features
gameEventsDf = spark.sql("select event_type_str, event_team, shot_place_str, location_str, assist_method_str, situation_str, country_code, is_goal from game_events")

# COMMAND ----------

# DBTITLE 1,Imports para Spark ML pipelines
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# DBTITLE 1,Features categóricas
categFeatures = ["event_type_str", "event_team", "shot_place_str", "location_str", "assist_method_str", "situation_str", "country_code"]

# COMMAND ----------

# DBTITLE 1,Encode categorical string cols to label indices
stringIndexers = [StringIndexer().setInputCol(baseFeature).setOutputCol(baseFeature + "_idx") for baseFeature in categFeatures]
stringIndexers

# COMMAND ----------

# DBTITLE 1,Convert categorical label indices to binary vectors
encoders = [OneHotEncoder().setInputCol(baseFeature + "_idx").setOutputCol(baseFeature + "_vec") for baseFeature in categFeatures]
encoders

# COMMAND ----------

# DBTITLE 1,Combinar las columnas a un vector de features
featureAssembler = VectorAssembler()
featureAssembler.setInputCols([baseFeature + "_vec" for baseFeature in categFeatures])
featureAssembler.setOutputCol("features")

# COMMAND ----------

# DBTITLE 1,Crear un Spark ML pipeline usando un GBT classifier
gbtClassifier = GBTClassifier(labelCol="is_goal", featuresCol="features", maxDepth=5, maxIter=20)

pipelineStages = stringIndexers + encoders + [featureAssembler, gbtClassifier]
pipeline = Pipeline(stages=pipelineStages)

# COMMAND ----------

# MAGIC %md
# MAGIC Siguiendo la siguiente documentación, completar los párrafos vacíos
# MAGIC http://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-regression

# COMMAND ----------

# DBTITLE 1,Split en training/test 75/25 y fitear el modelo
# Generar conjuntos de training y testing
(trainingData2, testData2) = gameEventsDf.randomSplit([0.75, 0.25])

# Entrenar el modelo
model_gbt = pipeline.fit(trainingData2)

# COMMAND ----------

# DBTITLE 1,Generar las predicciones y mostrarlas en una tabla con el vector de features
#Generar predicciones
predictions2 = model_gbt.transform(testData2)

#Predicciones y vector de features
#predictions2.select("prediction", "label", "features").show(10)
display(predictions2.select("prediction", "is_goal", "features"))

# COMMAND ----------

# DBTITLE 1,Evaluar el modelo utilizando la métrica de area debajo de la curva ROC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# utilizar BinaryClassificationEvaluator para evaluar el modelo

predict_roc = predictions2.select("rawPrediction", "prediction", "is_goal").withColumnRenamed("is_goal","label")

# Evaluar el modelo mediante métrica de area bajo la curva ROC
evaluator = BinaryClassificationEvaluator()
evaluator.setRawPredictionCol("rawPrediction")
area_roc = evaluator.evaluate(predict_4met, {evaluator.metricName: "areaUnderPR"})

print("Area bajo la curva ROC = " + str(area_roc))
# Se considera que por encima del 70% (0.7) el modelo de predicion es aceptable

# COMMAND ----------

# MAGIC %md ## Ejercicios:
# MAGIC
# MAGIC Implementar algunos de los siguientes puntos:
# MAGIC
# MAGIC 1. Implementar el tracking de corridas de entrenamiento del modelo con MLFlow (1.5 puntos)
# MAGIC https://docs.databricks.com/applications/mlflow/index.html
# MAGIC https://docs.databricks.com/_static/notebooks/mlflow/mlflow-quick-start-python.html
# MAGIC https://docs.databricks.com/_static/notebooks/mlflow/mlflow-end-to-end-example-aws.html
# MAGIC
# MAGIC 2. Evaluar el GBT classifier utilizando alguna otra métrica además de el área bajo a curva ROC (0.5 puntos)
# MAGIC 5. Entrenar una RandomForestClassifier con las el de GBT pero con RF. Trackear las corridas con MLFlow. Qué diferencias nota en las métricas de evaluación? Qué ventajas tiene RF sobre GBT (1.5 puntos)
# MAGIC 4. Realizar un ML Pipeline para tuneo de hiperparámetros de cualquiera de los dos clasificadores usando crossvalidation como se observa en los notebooks de referencia. Trackear las corridas con MLFlow y persistir el mejor modelo. (3 puntos)
# MAGIC https://docs.databricks.com/_static/notebooks/gbt-regression.html
# MAGIC https://docs.databricks.com/_static/notebooks/mllib-mlflow-integration.html
# MAGIC
# MAGIC 4. Realizar analytics sobre las predicciones de los modelos, tratar de encontrar patrones para los casos donde las conclusiones fueron erróneas. (2 puntos)
# MAGIC 5. Utilizando el siguiente notebook como referencia escribir el dataset de GAME_EVENTS particionado en muchos archivos y luego evaluar el modelo entrenado en streaming leyendo da a un archivo por trigger. https://docs.databricks.com/_static/notebooks/using-mllib-with-structured-streaming.html (2.5 puntos)

# COMMAND ----------

# DBTITLE 1,Ejercicio 4
# 4. Realizar un ML Pipeline para tuneo de hiperparámetros de cualquiera de los dos clasificadores usando crossvalidation
# Importar librerias

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow


# COMMAND ----------

# DBTITLE 1,1 - Create the VectorAssembler and VectorIndexer
# Features categoricos
categFeatures = ["event_type_str", "event_team", "shot_place_str", "location_str", "assist_method_str", "situation_str", "country_code"]

#Crear indices
stringIndexers = [StringIndexer().setInputCol(baseFeature).setOutputCol(baseFeature + "_idx") for baseFeature in categFeatures]

#Crear vectores
encoders = [OneHotEncoder().setInputCol(baseFeature + "_idx").setOutputCol(baseFeature + "_vec") for baseFeature in categFeatures]

# COMMAND ----------

# DBTITLE 1,2 - Create feature assemblers
featureAssembler = VectorAssembler()
featureAssembler.setInputCols([baseFeature + "_vec" for baseFeature in categFeatures])
featureAssembler.setOutputCol("features")

# COMMAND ----------

# DBTITLE 1,3 - Create the model
gbtClass = GBTClassifier(labelCol="is_goal", featuresCol="features")

# COMMAND ----------

# DBTITLE 1,4 - Create Crossvalidator
# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(gbtClass.maxDepth, [2, 5])\
  .addGrid(gbtClass.maxIter, [10, 100])\
  .build()

# COMMAND ----------

# DBTITLE 1,5 - Create the metric evaluator
# Definir metrica de evaluacion del modelo mediante métrica de area bajo la curva ROC - para elegir el mejor modelo
evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR", labelCol= gbtClass.getLabelCol(), rawPredictionCol= gbtClass.getRawPredictionCol())


# COMMAND ----------

# DBTITLE 1,6 - Model tuning 
# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=gbtClass, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# DBTITLE 1,7 - Create Spark ML pipeline
pipelineStages = stringIndexers + encoders + [featureAssembler, gbtClassifier, cv]
pipeline = Pipeline(stages=pipelineStages)

# COMMAND ----------

# DBTITLE 1,Entrenar el modelo usando el pipeline
# Generar conjuntos de training y testing
(trainingData3, testData3) = gameEventsDf.randomSplit([0.75, 0.25])

# Entrenar el modelo
pipelineModel = pipeline.fit(trainingData3)

# COMMAND ----------

display(trainingData3)

# COMMAND ----------

# DBTITLE 1,Generar predicciones usando el pipeline
predictions = pipelineModel.transform(testData3)
display(predictions.select("is_goal", "prediction", "features"))

# COMMAND ----------

# DBTITLE 1,Evaluar modelo
area_roc = evaluator.evaluate(predictions)
print("Area bajo la curva ROC = " + str(area_roc))

# COMMAND ----------

display(predictions.select("event_type", "prediction"))

# COMMAND ----------


import pyspark.sql.functions as F
predictions_with_residuals = predictions.withColumn("residual", (F.col("cnt") - F.col("prediction")))