
# coding: utf-8

# # Big Data Platforms
# 
# ## PySpark Machine Learning
# 
# ### MLlib applied to Wine reviews data 
# 
# **Dataset:**
# https://www.kaggle.com/zynicide/wine-reviews
# 
# 
# Copyright: 2018 [Ashish Pujari](apujari@uchicago.edu)

# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#create Spark session
spark = SparkSession.builder.appName('WineReviewsML').getOrCreate()

#change configuration settings on Spark 
conf = spark.sparkContext._conf.setAll([('spark.executor.memory', '5g'), ('spark.app.name', 'Spark Updated Conf'), ('spark.executor.cores', '4'), ('spark.cores.max', '4'), ('spark.driver.memory','8g')])

#print spark configuration settings
spark.sparkContext.getConf().getAll()


# ## Read Data

# In[3]:


df = spark.read     .option("quote", "\"")      .option("escape", "\"")     .option("ignoreLeadingWhiteSpace",True)     .csv("T:\\courses\\BigData\\data\\wine-reviews\\winemag-data_first150k.csv",inferSchema=True, header=True )


# In[22]:


df.printSchema()


# In[21]:


df = spark.read.csv("winemag-data_first150k.csv", header=True)
df2 = spark.read.csv("winemag-data-130k-v2.csv", header=True)


# In[24]:


import pyspark.sql.types as types


# In[25]:


df = df.withColumn('points', df['points'].cast(types.IntegerType()))
df2 = df2.withColumn('points', df2['points'].cast(types.IntegerType()))


# In[5]:


df2 = spark.read     .option("quote", "\"")      .option("escape", "\"")     .option("ignoreLeadingWhiteSpace",True)     .csv("T:\\courses\\BigData\\data\\wine-reviews\\winemag-data-130k-v2.csv",inferSchema=True, header=True )


# In[17]:


df2.printSchema()


# In[11]:


#combine the two datasets
df = df.union(df2.drop("taster_name", "taster_twitter_handle", "title"))


# ## Data Exploration

# In[12]:


df.count()


# In[14]:


df.show(5)


# In[15]:


#Count rows with missing values
df.dropna().count()

df = df.dropna()


# ##  Feature Engineering

# In[26]:


from pyspark.ml.feature import QuantileDiscretizer

#High Medium Low
discretizer = QuantileDiscretizer(numBuckets=3, inputCol="points", outputCol="points_category")
df = discretizer.fit(df).transform(df)
df.show(3)


# ## Natural Language Processing

# In[27]:


from pyspark.ml.feature import HashingTF, IDF, Tokenizer

#tokenize words
tokenizer = Tokenizer(inputCol="description", outputCol="words")
df = tokenizer.transform(df)

#drop the redundant source column
df= df.drop("description")
df.show(5)


# In[28]:


from pyspark.ml.feature import StopWordsRemover

#remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
df = remover.transform(df)

#drop the redundant source column
df= df.drop("words")
df.show(5)


# In[29]:


#Maps a sequence of terms to their term frequencies using the hashing trick. 
#alternatively, CountVectorizer can also be used to get term frequency vectors
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
nlpdf = idfModel.transform(featurizedData)
nlpdf.select("points_category", "features").show(10)


# In[15]:


nlpdf.show(5)


# In[30]:


#split data into train and test
splits = nlpdf.randomSplit([0.8, 0.2])
train_df = splits[0]
test_df = splits[1]

train_df.show(1)


# ### Logistic Regression Model

# In[31]:


from pyspark.ml.classification import LogisticRegression

# Set parameters for Logistic Regression
lgr = LogisticRegression(maxIter=10, featuresCol = 'features', labelCol='points_category')

# Fit the model to the data.
lgrm = lgr.fit(train_df)

# Given a dataset, predict each point's label, and show the results.
predictions = lgrm.transform(test_df)


# In[18]:


predictions.show(3)


# In[32]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#print evaluation metrics
evaluator = MulticlassClassificationEvaluator(labelCol="points_category", predictionCol="prediction")

print(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}))
print(evaluator.evaluate(predictions, {evaluator.metricName: "f1"}))


# ### Word2Vec

# In[33]:


# Learn a mapping from words to Vectors
from pyspark.ml.feature import Word2Vec
word2Vec = Word2Vec(vectorSize=2, minCount=0, inputCol="filtered", outputCol="wordVectors")
w2VM = word2Vec.fit(df)
nlpdf = w2VM.transform(df)


# In[34]:


nlpdf.select("points_category", "wordVectors").show(2, truncate=False)


# In[35]:


#split data into train and test
splits = nlpdf.randomSplit([0.8, 0.2])
train_df = splits[0]
test_df = splits[1]

train_df.show(1)


# In[36]:

from pyspark.ml.classification import RandomForestClassifier

# Set parameters for Logistic Regression
rf = RandomForestClassifier(featuresCol='wordVectors',
                            labelCol='points_category')
# Fit the model to the data.
rf_fit = rf.fit(train_df)


# Set parameters for Logistic Regression
lgr = LogisticRegression(maxIter=10, featuresCol='wordVectors', labelCol='points_category')

# Fit the model to the data.
lgrm = lgr.fit(train_df)

# Given a dataset, predict each point's label, and show the results.
predictions = lgrm.transform(test_df)


# In[24]:


#print evaluation metrics
evaluator = MulticlassClassificationEvaluator(labelCol="points_category", predictionCol="prediction")

print(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}))
print(evaluator.evaluate(predictions, {evaluator.metricName: "f1"}))


# <b>Exercise</b>: <font color='red'>Fine tune the Word2vec method to improve model accuracy </font>
