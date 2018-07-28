# Aditya Subramanian Muralidaran
 
# To Run: spark-submit FileClassify_multilayer_perceptron.py FileToLearnModel/ fileToClassify/

from __future__ import print_function


from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession
import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder.appName("multilayer_perceptron_classification_example").getOrCreate()


    # Load training data
    data = spark.read.format("libsvm")\
        .load(sys.argv[1])

    # Split the data into train and test
    splits = data.randomSplit([1.0, 0.0])
    train = splits[0]
    test = splits[1]

    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    layers = [114, 5, 4, 4]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=200, layers=layers, blockSize=128)

    # train the model
    model = trainer.fit(train)

    # compute accuracy on the test set
    #result = model.transform(test)

    # select example rows to display.
    
    classify_file = spark.read.format("libsvm") \
        .load(sys.argv[2])
    predictions = model.transform(classify_file)
    predictions.show()
    print("---------- Output Classes ----------")
    print("0 - Business")
    print("1 - Politics")
    print("2 - Sports")
    print("3 - Technology")
    print("------------------------------------")

    #predictionAndLabels = result.select("prediction", "label")
    #evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    #print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


    spark.stop()
