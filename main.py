import numpy
import sklearn.model_selection
import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder
tf.disable_v2_behavior()
# import nltk
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
import numpy as np
import csv
import pandas as pd

if __name__ == '__main__':

    address = r'/Users/nnussbaum/PycharmProjects/pythonProject6/venv/lib/data.csv'
    df = pd.read_csv(address)

    # change features with words  to numbers
    labelencoder = LabelEncoder()

    df['Attrition'] = labelencoder.fit_transform(df['Attrition'])
    df['BusinessTravel'] = labelencoder.fit_transform(df['BusinessTravel'])
    df['Department'] = labelencoder.fit_transform(df['Department'])
    df['EducationField'] = labelencoder.fit_transform(df['EducationField'])
    df['Gender'] = labelencoder.fit_transform(df['Gender'])
    df['JobRole'] = labelencoder.fit_transform(df['JobRole'])
    df['MaritalStatus'] = labelencoder.fit_transform(df['MaritalStatus'])
    df['Over18'] = labelencoder.fit_transform(df['Over18'])
    df['OverTime'] = labelencoder.fit_transform(df['OverTime'])

    # df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'])

    # Split data into training and testing sets

    x = df.drop('JobSatisfaction', axis = 1)
    y = df.JobSatisfaction

    # Split your dataset
    train = df[:70]
    test = df[70:]

    xtrain = train.drop(columns=['JobSatisfaction'], axis=1).values
    xtest = test.drop(columns=['JobSatisfaction'], axis=1).values

    ytrain = train['JobSatisfaction'].values
    ytest = test['JobSatisfaction'].values

    ytrain = ytrain[:,np.newaxis]
    ytest = ytest[:,np.newaxis]

    # Linear regression
    features = 34
    classes = 4

    epochs = 1000

    X = tf.placeholder(tf.float64, [None, features])
    Y = tf.placeholder(tf.float64, [None, classes])

    W = tf.Variable(tf.zeros([features, classes], dtype=tf.dtypes.float64, name="weight"))
    b = tf.Variable(tf.zeros([classes], dtype=tf.dtypes.float64, name="bias"))

    pred = tf.matmul(X,W) + b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=pred))

    train_prediction = tf.nn.softmax(pred)

    optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

    # Convert y column to vectors
    def converter(yList):
        y_ = []
        for i in range(0, len(yList)):
            if yList[i] == 1:
                y_.append((1, 0, 0, 0))
            if yList[i] == 2:
                y_.append((0, 1, 0, 0))
            if yList[i] == 3:
                y_.append((0, 0, 1, 0))
            if yList[i] == 4:
                y_.append((0, 0, 0, 1))
        return numpy.array(y_)

    yListTrain = converter(ytrain)
    yListTest = converter(ytest)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(epochs):

            sess.run(optimizer, feed_dict={X: xtrain, Y: yListTrain})

        print("\nOptimization Finished!\n")

        correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={X: xtest, Y: yListTest}))











