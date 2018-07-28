# Data Analytics Machine Learning Pipeline

To build a data pipeline that takes a document as an input and uses Apache Spark MapReduce and Machine Learning libraries to classify in one of the topics – Business, Sports, Politics and Technology. This pipeline is built using the following procedure:

• Data Collection: A large number of articles (1000 approx.) is collected from NY Times API using a python script on topics - Business, Sports, Politics and Technology.

• Feature Engineering: Apache Spark MapReduce framework is used in python programming language to get the top 40 words (representing each class) from each class using word count algorithm. Cumulatively these 120 words (approx.) will be used as feature in classification algorithms.

• The data gathered from NY Times is split into 80% training set and 20% testing set. Using the features extracted, a machine learning model is built (Naïve Bayes and Multi-Layer Perceptron) and the accuracy is determined.

• Once the model is built, a random document given to the model will be classified into one of the 4 classes - Business, Sports, Politics and Technology.


• Refer "README.pdf" for detailed description
