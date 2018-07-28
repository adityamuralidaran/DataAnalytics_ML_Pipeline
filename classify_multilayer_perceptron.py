# Aditya Subramanian Muralidaran

# To Run: spark-submit classify_multilayer_perceptron.py classify_input/

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
    splits = data.randomSplit([0.8, 0.2])
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
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


    spark.stop()
