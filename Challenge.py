#Se importan librerias, se define Schema y se carga .csv
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, IntegerType, StringType, DoubleType,LongType
import pyspark.sql.functions as psf

spark = SparkSession.builder.appName('Nubimetrics').getOrCreate()

schema = StructType([
    StructField("ID", IntegerType()),
    StructField("Title", StringType()),
    StructField("Sales", DoubleType()),
    StructField("Unit Sales", LongType())
])


df = spark.read.option("delimiter", "").csv('cellphoneslisting.csv', schema=schema, header=True, encoding='utf-8')


# In[ ]:


# Se verifica que se haya aplicado correctamente el Schema
df.printSchema()


# In[ ]:


# Se da un primer vistazo a como esta compuesto los datos
df.show(30, False)


# In[ ]:


# Se recorre con una expresion regular, los items de 'Title', buscando alguna de las marcas mas conocidas
# Suma una nueva columna 'Brand', donde pondra la marca si es que la encuentra, caso contrario deja un string vacio
brand_list = ['Samsung', 'Iphone', 'Xiaomi', 'Motorola', 'Huawei', 'Alcatel', 'Nokia', 'Lg', 'Blu', 'Ken Brown', 'Sony', 'Noblex', 'Bgh', 'Philco', 'Caterpillar', 'Kanji', 'Blackberry', 'Philips', 'Nextel', 'Hyundai' ]
df1 = df.withColumn('Brand', psf.regexp_extract('Title', '(?=^|.*\s)(' + '|'.join(brand_list) + ')(?=\s|$)', 0))


# In[ ]:


#Se chequea cuales son los titulos que no pudo clasificar, para evalular si corresponde ponerlos en la lista
df1.where(df1.Brand == '').show(30, False)


# In[ ]:


# Se verifica como queda una porcion del nuevo DataFrame ya clasificado
df1.show(30, False)


# In[ ]:


# Se renombra lo clasificado como un string vacio a 'Otros'
df_null = df1.withColumn("Marca", psf.when(psf.col('Brand') != '', psf.col('Brand')).otherwise(None)).drop('Brand')
df2 = df_null.na.fill({'Marca': 'Otros'})


# In[ ]:


# Se chequea cual es la proporcion de elementos no clasificados con el total, se encuentra aceptable.
df2.where(df2.Marca == 'Otros').count() / df.count()


# In[ ]:


# Se arma el DataFrame final, donde se agrupa por marcas, sumando el total de ventas y la cantidad.
df_final = df2.groupBy(df2.Marca).agg(psf.round(psf.sum("Sales"),2).alias('Total de Ventas'), psf.sum("Unit Sales").alias('Cantidad Vendida'))


# In[ ]:


# Se muestra ordenada por total de ventas
df_final.orderBy('Total de Ventas', ascending=False).show()


# In[ ]:


#Se importan librerias necesarias para poder graficar el DataFrame anterior
import pandas as pd
import matplotlib.pyplot as plt

#Necesario para correr en Jupyter Notebook
from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Para operar mas facil, se pasa de DataFrame de Spark a uno de Pandas
df_pandas = df_final.toPandas()


# In[ ]:


#Preparo los datos para poder graficarlos, por un tema de espacio, solo se tomaran los 5 valores mas altos teniendo
# en cuenta el total de las ventas, el resto se sumara a 'Otros'

df_top = df_pandas.sort_values('Total de Ventas', ascending=False)[:6]
suma_ventas = df_pandas.sort_values('Total de Ventas', ascending=False)["Total de Ventas"][6:].sum()
suma_cant = df_pandas.sort_values('Cantidad Vendida', ascending=False)["Cantidad Vendida"][6:].sum()

#Indexo el DF
df_top.set_index("Marca", inplace=True)
#df_top.head()

#Obtengo el valor actual de 'Otros'
value_ventas = df_top.loc[["Otros"], ['Total de Ventas']].values[0][0]
value_cant = df_top.loc[["Otros"], ['Cantidad Vendida']].values[0][0]

#Reemplazo en el nuevo DF a graficar
df_top['Total de Ventas']['Otros'] = value_ventas + suma_ventas
df_top['Cantidad Vendida']['Otros'] = value_cant + suma_cant

#Una vez que termino de operar, reseteo el Index para poder graficar
df_top.reset_index(inplace=True)


# In[ ]:


# Crear grafico de torta, tanto de total de ventas como de cantidad vendida

plt.figure(0)
plt.pie(
    df_top['Total de Ventas'],
    labels=df_top['Marca'],
    shadow=False,
    startangle=90,
    autopct='%1.1f%%',
    )
plt.title('Total de Ventas')

plt.figure(1)
plt.pie(
    df_top['Cantidad Vendida'],
    labels=df_top['Marca'],
    shadow=False,
    startangle=90,
    autopct='%1.1f%%',
    )
plt.title('Cantidad Vendida')

plt.show()



