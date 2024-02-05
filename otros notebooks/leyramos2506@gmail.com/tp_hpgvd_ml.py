# Databricks notebook source
# MAGIC %md ## Machine Learning sobre el dataset del fútbol Europeo
# MAGIC
# MAGIC Para poder ejecutar este notebook se requiere haber procesado la información y generado la base de datos y la tabla correspondiente con el primer notebook.

# COMMAND ----------

# MAGIC %sql USE EURO_LEAGUE_DB

# COMMAND ----------

# MAGIC %sql SELECT * FROM GAME_EVENTS LIMIT 30

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

# COMMAND ----------

# DBTITLE 1,Convert categorical label indices to binary vectors
encoders = [OneHotEncoder().setInputCol(baseFeature + "_idx").setOutputCol(baseFeature + "_vec") for baseFeature in categFeatures]

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


# COMMAND ----------

# DBTITLE 1,Generar las predicciones y mostrarlas en una tabla con el vector de features


# COMMAND ----------

# DBTITLE 1,Evaluar el modelo utilizando la métrica de area debajo de la curva ROC
# utilizar BinaryClassificationEvaluator para evaluar el modelo

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