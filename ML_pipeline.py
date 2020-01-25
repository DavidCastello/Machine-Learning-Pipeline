## CONFIGURACIÓN INICIAL DEL ENTORNO

import os.path
import random

# ************ SPARK & SQL ********************

import findspark
import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import rand


# ************ MACHINE LEARNING ********************

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# ************** Visualización **********************

from matplotlib import pyplot as plt
import pandas as pd

findspark.init()
sc = pyspark.SparkContext(master="local", appName="myApp")

## Cargamos el archivo csv original

path_rawFile = "user/data/table.csv" # path a nuestro fichero en HDFS

data = sc.textFile(path_rawFile) # creamos elemento RDD a partir del fichero csv

## Formato del ejemplo utilizado:

print("\nFormato de los datos originales cargados:\n")
print (data.take(5))

## Definición del DataFrame

sqlContext = SQLContext(sc)
# Para parar spark context sc.stop()
powerPlantDF = sqlContext.read.format("csv").options(delimiter=',',header='true',inferschema='true').load(path_rawFile)

# Exploramos los datos

print("\nData types de los datos cargados:\n")
print(powerPlantDF.dtypes)
print("\nVisualización de la tabla:\n")
powerPlantDF.show()

# Creamos un schema a manualmente

customSchema = StructType([ \
    StructField('AT', DoubleType(), True), \
    StructField('V', DoubleType(), True), \
    StructField('AP', DoubleType(), True), \
    StructField('RH', DoubleType(), True), \
    StructField('PE', DoubleType(), True)])
    
altPowerPlantDF = sqlContext.read.format('csv').options(delimiter=',',header='true').schema("col0 DOUBLE, col1 DOUBLE, col2 DOUBLE, col3 DOUBLE, col4 DOUBLE").load("/aula_M2.858/data/pec2",schema = customSchema)

print("Data types de los datos cargados con configuración manual:\n")
print(powerPlantDF.dtypes)
print("\nVisualización de la tabla con configuración manual:\n")
powerPlantDF.show()

## Creamos una tabla SQL a partir de los datos

sqlContext.sql('DROP TABLE IF EXISTS power_plant')
sqlContext.registerDataFrameAsTable(altPowerPlantDF, 'power_plant')

##############!!!!!!!!!!BORRAR dfALL=sqlContext.sql("SELECT AT AS AT, V as V, AP as AP, RH as RH, PE as PE from power_plant") # Volvemos a un DF comprobando nuestra transformación

# Visualización de datos estadísticos básicos de nuestro dataset

df = sqlContext.table('power_plant')
df.describe().show()

## Análisis inicial de las variables de nuestro modelo

# Estudiamos como la potencia "PE" varía en función de las otras variables

# Visualización de la influencia de Temperatura "AT"

x_y_AT = sqlContext.sql("SELECT AT as AT, PE as PE from power_plant")
x_y_DF_AT = pd.DataFrame(x_y_AT.toPandas().sample(n=1000),columns=['AT','PE'])

x_y_DF_AT.plot(kind='scatter',x='AT',y='PE',color='red')
plt.show()

# Visualización de la influencia de Presión Atmosférica PE

x_y_AP = sqlContext.sql("SELECT AP as AP, PE as PE from power_plant")
x_y_DF_AP = pd.DataFrame(x_y_AP.toPandas().sample(n=1000),columns=['AP','PE'])

x_y_DF_AP.plot(kind='scatter',x='AP',y='PE',color='green')
plt.show()

# Visualización de la influencia de Humedad RH

x_y_RH = sqlContext.sql("SELECT RH as RH, PE as PE from power_plant")
x_y_DF_RH = pd.DataFrame(x_y_RH.toPandas().sample(n=1000),columns=['RH','PE'])

x_y_DF_RH.plot(kind='scatter',x='RH',y='PE',color='yellow')
plt.show()

## Preparación de los datos para aprendizaje automático

# Utilizatremos el formato VectorAssembler()

datasetDF = sqlContext.table('power_plant')

vectorizer = VectorAssembler()
vectorizer.setInputCols(["AT", "V", "AP", "RH"])
vectorizer.setOutputCol("features") # Aquí guardaremos nuestra variable objetivo Potencia (PE)

# Dividimos el dataset en entrenamiento (80%) y test (20%)

seed = 1800009193
(split20DF, split80DF) = datasetDF.randomSplit([0.2,0.8],1800009193)
trainingSetDF = split80DF
testSetDF = split20DF

# Guardamos en cache los datos para agilizar los cáluclos

trainingSetDF.cache()
testSetDF.cache()

# Árboles de decisión

rf = RandomForestRegressor()

# Para información sobre los parametros: print(rf.explainParams())

rf.setPredictionCol('Predicted_PE')\
  .setLabelCol('PE')\
  .setNumTrees(20)\
  .setMaxDepth(5)

# Forest Pipeline

pipeline = Pipeline(stages = [vectorizer, rf])

# Entrenamos el modelo

model = pipeline.fit(trainingSetDF)

# Podemos ver los detalles del árbol creado:

"""
    print("Nodos: " + str(model.stages[-1]._java_obj.parent().getNumTrees()))
    print("Profundidad: "+ str(model.stages[-1]._java_obj.parent().getMaxDepth()))  

    print(model.stages[-1]._java_obj.toDebugString())

"""

# Aplicamos el modelo de RandomForest al test data y hacemos una predicción del output

print("\nPredicted output with RandomForest:\n")

predictions = model.transform(testSetDF)
predictions.show()

## Evaluamos nuestro modelo con RMSE

regEval = RegressionEvaluator(predictionCol='Predicted_PE', labelCol='PE', metricName='rmse')
rmse = regEval.evaluate(predictions)
print("\nRoot Mean Squared Error (RMSE): %.2f\n" % rmse)

## Evaluamos con el coeficiente de determinación R2

r2 = regEval.evaluate(predictions, {regEval.metricName: "r2"})
print("\nCoeficiente de Determinación (r2): {0:.2f}\n".format(r2))

## Podemos visualizar el error residual en un historiograma

predictions.selectExpr("PE", "Predicted_PE", "PE - Predicted_PE Residual_Error", "(PE - Predicted_PE) / {} Within_RSME".format(rmse)).registerTempTable("Power_Plant_RMSE_Evaluation")
x = sqlContext.sql('SELECT Residual_Error as Residual_Error from Power_Plant_RMSE_Evaluation')
df_error = x.toPandas()

# Mostramos el histograma con el Error Residual_Error

fig = plt.figure()
ax = plt.gca()
ax.hist(df_error["Residual_Error"], bins=100, color="salmon", edgecolor="black")
ax.set_title("Histograma Error Residual")
ax.set_xlabel("frequencia")
ax.set_ylabel("Error residual")
plt.show() 

# *************** HYPERPARAMETER TUNING *****************

# Podemos construir un Cross Validator para optimizar los parametros de nuestro modelo

pipeline = Pipeline(stages = [vectorizer, rf]) # Datos preparados para ML en vectorizer y el arbol de decisión definido anteriormente
evaluator = RegressionEvaluator(predictionCol="Predicted_PE", labelCol="PE", metricName="rmse")

grid = (ParamGridBuilder()
               .addGrid(rf.maxDepth, [5, 1, 2, 6, 7]) # Probaremos 5 profundidades diferentes
               .addGrid(rf.numTrees, [20, 15, 22, 25, 18]) # Probaremos 5 números de arboles diferntes
             .build())

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5) # Doblamos los datos 5 veces (5 distribuciones diferentes de training/test data) 

model = cv.fit(trainingSetDF)

print("\nPredicciones con el Cross Validator:\n")

predictions = model.transform(testSetDF)
print('RMSE:', evaluator.evaluate(predictions))
predictions.show()




