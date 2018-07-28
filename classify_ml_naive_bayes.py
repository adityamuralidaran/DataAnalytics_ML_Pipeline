# Aditya Subramanian Muralidaran

# To Run : spark-submit classify_ml_naive_bayes.py classify_input/

from __future__ import print_function


from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession
import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("NaiveBayesExample")\
        .getOrCreate()


    # Load training data
    data = spark.read.format("libsvm") \
        .load(sys.argv[1])

    # Split the data into train and test
    splits = data.randomSplit([0.8, 0.2])
    train = splits[0]
    test = splits[1]

    # create the trainer and set its parameters
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

    # train the model
    model = nb.fit(train)

    # select example rows to display.
    predictions = model.transform(test)
    predictions.show()

    # compute accuracy on the test set
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = " + str(accuracy))
    # $example off$

    spark.stop()
