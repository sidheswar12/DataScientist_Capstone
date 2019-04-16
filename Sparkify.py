#!/usr/bin/env python
# coding: utf-8

# # Sparkify Project Workspace
# This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.
# 
# You can follow the steps below to guide your data analysis and model building portion of this project.

# In[1]:


# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, isnan, count, when, desc, sort_array, asc, avg, lag, floor
from pyspark.sql.types import IntegerType, DateType
from pyspark.sql.window import Window
from pyspark.sql.functions import sum as Fsum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.types import DoubleType
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import datetime
from pyspark.ml import Pipeline 
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics


# In[2]:


# Create a Spark session
spark = SparkSession.builder     .master("local")     .appName("Identify Features")     .getOrCreate()

df = spark.read.json('mini_sparkify_event_data.json')

# Print schema for future reference
df.printSchema()


# # Load and Clean Dataset
# In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 

# In[3]:


#Clean dataset with Empty userid
df = df.filter(df.userId != "")
print(f"Empty userIds: {df.filter(df.userId == '').count()}")


# # Exploratory Data Analysis
# When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.
# 
# ### Define Churn
# 
# Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.
# 
# ### Explore Data
# Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

# In[4]:


# Observation
df.printSchema()


# In[5]:


# Look at 2 rows for sample
df.take(2)


# In[6]:


# Get number of users
df.select("userId").dropDuplicates().count()


# In[7]:


# Get number of sessions
df.select("sessionId").dropDuplicates().count()


# In[8]:


churn = udf(lambda x: int(x=="Cancellation Confirmation"), IntegerType())
downgrade_churn = udf(lambda x: int(x=="Submit Downgrade"), IntegerType())

df = df.withColumn("downgraded", downgrade_churn("page")).withColumn("cancelled", churn("page"))


# In[9]:


#Distribution of users downgrades and cancellations
df.select(['userId', 'downgraded', 'cancelled'])    .groupBy('userId').sum()    .withColumnRenamed('sum(downgraded)', 'downgraded')    .withColumnRenamed('sum(cancelled)', 'cancelled').describe().show()


# In[10]:


windowvalue = Window.partitionBy("userId").orderBy(desc("ts")).rangeBetween(Window.unboundedPreceding, 0)
df = df.withColumn("churn_phase", Fsum("cancelled").over(windowvalue))    .withColumn("downgrade_phase", Fsum("downgraded").over(windowvalue))


# In[11]:


get_day = udf(lambda x: datetime.datetime.fromtimestamp(x/1000), DateType())


# In[12]:


song=udf(lambda x : int(x=='NextSong'), IntegerType())
home_visit=udf(lambda x : int(x=='Home'), IntegerType())
df = df.withColumn('date', get_day(col('ts')))


# In[13]:


#Difference in number of songs played between users who churned or not
df.filter(col('churn_phase')==1).withColumn('songPlayed', song(col('page'))).agg({'songPlayed':'mean'}).show()
df.filter(col('churn_phase')==0).withColumn('songPlayed', song(col('page'))).agg({'songPlayed':'mean'}).show()


# In[14]:


#number of songs played between home visits
cusum = df.filter((df.page == 'NextSong') | (df.page == 'Home'))     .select('userID', 'page', 'ts', 'churn_phase')     .withColumn('homevisit', home_visit(col('page')))     .withColumn('songPeriod', Fsum('homevisit').over(windowvalue))

cusum.filter((cusum.churn_phase == 1) &(cusum.page == 'NextSong'))     .groupBy('userID', 'songPeriod')     .agg({'songPeriod':'count'})     .agg({'count(songPeriod)':'avg'}).show()

cusum.filter((cusum.churn_phase == 0) &(cusum.page == 'NextSong'))     .groupBy('userID', 'songPeriod')     .agg({'songPeriod':'count'})     .agg({'count(songPeriod)':'avg'}).show()


# In[15]:


days = lambda i: i * 86400 
daywindow = Window.partitionBy('userId', 'date').orderBy(desc('ts')).rangeBetween(Window.unboundedPreceding, 0)
get_day = udf(lambda x: datetime.datetime.fromtimestamp(x/1000), DateType())


# In[16]:


#Number of songs played daily
df.filter((df.page=='NextSong')&(col('churn_phase')==1)).select('userId', 'page', 'ts')    .withColumn('date', get_day(col('ts'))).groupBy('userId', 'date').count().describe().show()

df.filter((df.page=='NextSong')&(col('churn_phase')==0)).select('userId', 'page', 'ts')    .withColumn('date', get_day(col('ts'))).groupBy('userId', 'date').count().describe().show()


# In[17]:


#number of songs couldn't be played due to errors
df.filter((df.page=='Error')&(df.churn_phase==1)).select('userId', 'page', 'ts', 'length')    .withColumn('date', get_day(col('ts')))    .groupBy('userId', 'date').agg({'page':'count'}).select('count(page)').describe().show()

df.filter((df.page=='Error')&(df.churn_phase==0)).select('userId', 'page', 'ts', 'length')    .withColumn('date', get_day(col('ts')))    .groupBy('userId', 'date').agg({'page':'count'}).select('count(page)').describe().show()


# In[18]:


#Number of times user opted for help
df.filter((df.page=='Help')&(df.churn_phase==1)).select('userId', 'page', 'ts', 'length')    .withColumn('date', get_day(col('ts')))    .groupBy('userId', 'date').agg({'page':'count'}).describe().show()

df.filter((df.page=='Help')&(df.churn_phase==0)).select('userId', 'page', 'ts', 'length')    .withColumn('date', get_day(col('ts')))    .groupBy('userId', 'date').agg({'page':'count'}).describe().show()


# In[19]:


#Ratio of those who cancelled subscriptions both free and paid
print(df.filter((df.page=='Cancellation Confirmation') & (df.level=='paid')).count(),
df.filter((df.page=='Cancellation Confirmation') & (df.level=='free')).count())


# In[20]:


#Number of users who downgraded & Number of users to cancel
print(df.filter(col('downgraded')==1).select('userId').dropDuplicates().count(), 
      df.filter(col('cancelled')==1).select('userId').dropDuplicates().count())


# In[21]:


#Users who downgraded and then cancelled
df.select(['userId', 'downgraded', 'cancelled'])    .groupBy('userId').sum()    .withColumnRenamed('sum(downgraded)', 'downgraded')    .withColumnRenamed('sum(cancelled)', 'cancelled')    .filter((col("downgraded")==1)&(col("cancelled")==1))    .count()


# In[22]:


#Number of users to cancel who downgrade
df.select(['userId', 'downgraded', 'cancelled'])    .groupBy('userId').sum()    .withColumnRenamed('sum(downgraded)', 'downgraded')    .withColumnRenamed('sum(cancelled)', 'cancelled')    .filter((col("downgraded")==0)&(col("cancelled")==1))    .count()


# In[23]:


#Number of paid users to drop without downgrading
print(df.filter((col('cancelled')==1) & (col('downgraded')==0) & (col('level')=='paid'))      .select('userId').dropDuplicates().count())


# In[24]:


#Those who churn or not have different listening habits?
df.filter(col('cancelled')==1).agg({'length':'mean'}).show()
df.filter(col('cancelled')==0).agg({'length':'mean'}).show()


# # Feature Engineering
# Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
# - Write a script to extract the necessary features from the smaller subset of data
# - Ensure that your script is scalable, using the best practices discussed in Lesson 3
# - Try your script on the full data set, debugging your script if necessary
# 
# If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.

# In[25]:


# Remov duplicate and return user
def user_info(df):
    return df.where((df.userId != "") | (df.sessionId != "")).select('userId').dropDuplicates()


# In[26]:


# Return average thumbs up 
def average_thumbs_up(df):
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x/1000), DateType())
    return df.filter(df.page=='Thumbs Up').select('userId', 'page', 'ts')                    .withColumn('date', get_day(col('ts'))).groupBy('userId', 'date')                    .agg({'page':'count'}).groupBy('userId').mean()                    .withColumnRenamed('avg(count(page))', 'avgThumbsUp')


# In[27]:


# Return average thumbs down
def average_thumbs_down(df):
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x/1000), DateType())
    return df.filter(df.page=='Thumbs Down')        .select('userId', 'page', 'ts')        .withColumn('date', get_day(col('ts')))        .groupBy('userId', 'date').agg({'page':'count'})        .groupBy('userId').mean()        .withColumnRenamed('avg(count(page))', 'avgThumbsDown')


# In[28]:


# Return number of friends
def number_of_friends(df):
    return df.filter(df.page=='Add Friend')        .select('userId', 'page')        .groupBy('userId').count().withColumnRenamed('count', 'numFriends')


# In[29]:


# Return skipped attributes
def skipp_attributes(df):
    song = udf(lambda x: int(x=='NextSong'), IntegerType())
    skipped = udf(lambda x: int(x!=0), IntegerType())
    session = Window.partitionBy("userId", "sessionId").orderBy(desc("ts"))
    return df.select('userId', 'page', 'ts', 'length', 'sessionId', 'itemInSession')        .where((df.page != 'Thumbs Up') & (df.page != 'Thumbs Down'))        .withColumn('song', song('page')).orderBy('userId', 'sessionId', 'itemInSession')        .withColumn('nextActSong', lag(col('song')).over(session))        .withColumn('tsDiff', (lag('ts').over(session)-col('ts'))/1000)        .withColumn('timeSkipped', (floor('length')-col('tsDiff')))        .withColumn('roundedLength', floor('length'))        .where((col('song')==1) & ((col('nextActSong')!=0)&(col('timeSkipped')>=0)))        .withColumn('skipped', skipped('timeSkipped'))        .select('userId', 'timeSkipped', 'skipped', 'length', 'ts', 'tsDiff')        .groupBy('userId').agg({'skipped':'avg', 'timeSkipped':'avg'})        .withColumnRenamed('avg(skipped)', 'skipRate')        .withColumnRenamed('avg(timeSkipped)', 'avgTimeSkipped')


# In[30]:


# Return who all are regular visiter of help site
def daily_visit_help_site(df):
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x/1000), DateType())
    home_visit=udf(lambda x : int(x=='Home'), IntegerType())
    return df.filter(df.page=='Help')        .select('userId', 'page', 'ts', 'length')        .withColumn('date', get_day(col('ts')))        .groupBy('userId', 'date').agg({'page':'count'})        .groupBy('userId').mean()         .withColumnRenamed('avg(count(page))', 'dailyHelpVisits')


# In[31]:


# Return daily error
def get_daily_error(df):
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x/1000), DateType())
    return df.filter(df.page=='Error')        .select('userId', 'page', 'ts', 'length')        .withColumn('date', get_day(col('ts')))        .groupBy('userId', 'date').agg({'page':'count'})        .groupBy('userId').mean()        .withColumnRenamed('avg(count(page))', 'dailyErrors')


# In[32]:


# Return churn
def churn_user(df):
    #Define custom functions
    churn = udf(lambda x: int(x=="Cancellation Confirmation"), IntegerType())
    downgrade_churn = udf(lambda x: int(x=="Submit Downgrade"), IntegerType())
    visited_downgrade = udf(lambda x: int(x=='Downgrade'), IntegerType())
    visited_cancel = udf(lambda x: int(x=='Cancel'), IntegerType())
    
    return df.withColumn("downgraded", downgrade_churn("page"))        .withColumn("cancelled", churn("page"))        .withColumn('visited_cancel', visited_cancel('page'))        .withColumn('visited_downgrade', visited_downgrade('page'))        .select(['userId', 'downgraded', 'cancelled', 'visited_cancel', 'visited_downgrade'])        .groupBy('userId').sum()        .withColumnRenamed('sum(downgraded)', 'downgraded')        .withColumnRenamed('sum(cancelled)', 'cancelled')        .withColumnRenamed('sum(visited_cancel)', 'visited_cancel')        .withColumnRenamed('sum(visited_downgrade)', 'visited_downgrade')

    


# In[33]:


# Return user level
def get_user_level(df):
    free = udf(lambda x: int(x=='free'), IntegerType())
    paid = udf(lambda x: int(x=='paid'), IntegerType())
    return df.select('userId', 'level')        .where((df.level=='free')|(df.level=='paid'))        .dropDuplicates()        .withColumn('free', free('level'))        .withColumn('paid', paid('level')).drop('level')

    


# In[34]:


# Return cusum
def get_cusum(df):
    windowval = Window.partitionBy("userId").orderBy(desc("ts")).rangeBetween(Window.unboundedPreceding, 0)
    return df.filter((df.page == 'NextSong') | (df.page == 'Home'))         .select('userID', 'page', 'ts')         .withColumn('homevisit', home_visit(col('page')))         .withColumn('songPeriod', Fsum('homevisit').over(windowval))    
   


# In[35]:


# Return average songs till home
def average_songs_till_home(cusum):
    return cusum.filter((cusum.page=='NextSong'))        .groupBy('userId', 'songPeriod')        .agg({'songPeriod':'count'}).drop('songPeriod')        .groupby('userId').mean()        .withColumnRenamed('avg(count(songPeriod))', 'avgSongsTillHome')


# In[36]:


def feature_engineering(filepath):
    '''
    Create necessary features to use machine learning algorithms.
    First loads data set from file
    
    Resulting DF Strucutre:    
    root
     |-- userId: string
     |-- downgraded: long
     |-- cancelled: long
     |-- visited_cancel: long
     |-- visited_downgrade: long
     |-- dailyHelpVisits: double
     |-- dailyErrors: double
     |-- free: integer
     |-- paid: integer
     |-- avgThumbsUp: double
     |-- avgThumbsDOwn: double
     |-- numFriends: long
     |-- avgSongsTillHome: double
     |-- avgTimeSkipped: double
     |-- skipRate: double
    
    Inputs:
        filepath - path to json dataset on file
        
    Outputs:
        data - engineered dataset
    '''
    #Dataframe of user ids to merge onto
    df = spark.read.json(filepath)
    users = user_info(df)
       
    #Define windows
    windowval = Window.partitionBy("userId").orderBy(desc("ts")).rangeBetween(Window.unboundedPreceding, 0)
    #session = Window.partitionBy("userId", "sessionId").orderBy(desc("ts"))
    daywindow = Window.partitionBy('userId', 'date').orderBy(desc('ts')).rangeBetween(Window.unboundedPreceding, 0)

    avg_thumbs_up = average_thumbs_up(df)
    
    avg_thumbs_down = average_thumbs_down(df)
    
    num_friends = number_of_friends(df)
    
    '''
    Process to calculate skipping variables

    1. Dont include thumbs up and down pages because that usually occurs 
       while playing and does not change song
    2. Create variable for if action is song
    3. Check if next action is song - this will check to see if someone is 
       skipping song or just leaving page
    4. Get the difference in timestamp for next action song playing
    5. Subtract the difference in timestamp from song length to see 
       how much of song was skipped
    6. Get descriptive stats
    '''

    skipping = skipp_attributes(df)
    
    #Avg daily visits to help site
    daily_help_visit = daily_visit_help_site(df)

    daily_errors = get_daily_error(df)
    
    #Whether a user has downgraded
    churn = churn_user(df)

    user_level = get_user_level(df)

    cusum = get_cusum(df)
    
    avg_songs_till_home = average_songs_till_home(cusum)
    
    df = users.join(churn, on='userId')        .join(daily_help_visit, on='userId')        .join(daily_errors, on='userId')        .join(user_level, on='userId')        .join(avg_thumbs_up, on='userId')        .join(avg_thumbs_down, on='userId')        .join(num_friends, on='userId')        .join(avg_songs_till_home, on='userId')        .join(skipping, on='userId')
    
    return df


# In[37]:


def feature_scaling(df):
    feature_cols = df.drop('userId', 'cancelled').columns
    assembler = VectorAssembler(inputCols=feature_cols,                                outputCol='feature_vec')
    
    #Pyspark.ml expects target column to be names: 'labelCol', type: Double
    df = df.withColumn("label", df["cancelled"].cast(DoubleType()))
    
    #Pyspark default name for features vector column: 'featuresCol'
    minmaxscaler = MinMaxScaler(inputCol="feature_vec", outputCol="features")
    
    df = assembler.transform(df)
    minmaxscaler_model = minmaxscaler.fit(df)
    scaled_df = minmaxscaler_model.transform(df)
    return scaled_df


# # Modeling
# Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.

# In[38]:


data_small = feature_engineering("mini_sparkify_event_data.json")
data_small = feature_scaling(data_small)
data_small.persist()


# In[39]:


data_small.take(1)


# In[40]:


train, rest = data_small.randomSplit([0.85, 0.15], seed=42)
validation, test = rest.randomSplit([0.5,0.5], seed=42)


# In[41]:


pr = BinaryClassificationEvaluator(metricName='areaUnderPR')
roc = BinaryClassificationEvaluator(metricName='areaUnderROC')


# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt

def custom_evaluation(pred, model_name):
    '''
    Perform custom evaluation of predictions
    
    1. Inspect with PySpark.ML evaluator will use for pipeline
    2. Use RDD-API; PySpark.MLLib to get metrics based on predictions 
    3. Display confusion matrix
    
    Inputs:
        preds PySpark.ml.DataFrame - predictions from model
    '''
    pr = BinaryClassificationEvaluator(metricName='areaUnderPR')    
    pr_auc=pr.evaluate(pred)    
    print(f"{model_name} -> PR AUC: {pr_auc}")
    predictionRDD = pred.select(['label', 'prediction']).rdd                        .map(lambda line: (line[1], line[0]))
    metrics = MulticlassMetrics(predictionRDD)

    print(f"{model_name}\n | precision = {metrics.precision()}")
    print(f" | recall = {metrics.recall()}\n | F1-Score = {metrics.fMeasure()}")
    
    conf_matrix = metrics.confusionMatrix().toArray()
    sns.set(font_scale=1.4)#for label size
    ax = sns.heatmap(conf_matrix, annot=True,annot_kws={"size": 16})
    ax.set(xlabel='Predicted Label', ylabel='True Label', title='Confusion Mtx')
    plt.show()


# In[43]:


#Random forest classifier model
rando_forest = RandomForestClassifier(numTrees=10)
rando_forest_model = rando_forest.fit(train)
rando_forest_preds = rando_forest_model.transform(validation)
custom_evaluation(rando_forest_preds, 'Random Forest')


# In[44]:


#Gradient boosted trees (ie ada boost)
gbtrees = GBTClassifier(maxIter=10)
gbtree_model = gbtrees.fit(train)
gbtree_preds = gbtree_model.transform(validation)
custom_evaluation(gbtree_preds, 'Gradient Boosted Trees')


# In[45]:


#SVM
svm = LinearSVC(maxIter=10, regParam=0.1)
svm_model=svm.fit(train)
svm_preds=svm_model.transform(validation)
custom_evaluation(svm_preds, 'Support Vector Machine')


# In[46]:


#Logistic regression model
logReg = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = logReg.fit(train)
lr_preds = lrModel.transform(validation)
custom_evaluation(lr_preds, 'Logistic Regression')


# In[47]:


#Visual check for predictions
for x in [svm_preds, lr_preds, gbtree_preds, rando_forest_preds]:
    x.select('features', 'rawPrediction', 'prediction', 'label').show()


# In[48]:


# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", metricName = 'areaUnderPR')
for x in [svm_preds, lr_preds, gbtree_preds, rando_forest_preds]:
    print(x,evaluator.evaluate(x))


# # Final Steps
# Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.

# In[ ]:




