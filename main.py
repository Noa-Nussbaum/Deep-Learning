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

#         Let's add a neural network

    xholder = tf.placeholder(tf.float32, [None, features])
    yholder = tf.placeholder(tf.float32, [None, classes])

    # Layer 1
    layer_1 = 70
    W1 = tf.Variable(tf.truncated_normal([features, layer_1], dtype=tf.dtypes.float32, stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1,dtype=tf.dtypes.float32, shape=[layer_1]))
    z1 = tf.add(tf.matmul(xholder, W1), b1)
    # a1 = tf.nn.leaky_relu(z1)
    a1 = tf.nn.elu(z1)

    # Layer 2
    layer_2 = 70
    W2 = tf.Variable(tf.truncated_normal([layer_1, layer_2], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[layer_2]))
    z2 = tf.add(tf.matmul(a1, W2), b2)
    # a2 = tf.nn.leaky_relu(z2)
    a2 = tf.nn.elu(z2)

    # Layer 3
    layer_3 = 70
    W3 = tf.Variable(tf.truncated_normal([layer_2, layer_3], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[layer_3]))
    z3 = tf.add(tf.matmul(a2, W3), b3)
    a3 = tf.nn.elu(z3)

    # Layer 4
    layer_4 = 34
    W4 = tf.Variable(tf.truncated_normal([layer_3, layer_4], stddev=0.1))
    b4 = tf.Variable(tf.constant(0.1, shape=[layer_4]))
    z4 = tf.add(tf.matmul(a3, W4), b4)
    a4 = tf.nn.elu(z4)


    # Output layer 2
    # W3 = tf.Variable(tf.truncated_normal([layer_2, classes], stddev=0.1))
    # b3 = tf.Variable(tf.constant(0.1, shape=[classes]))
    # predLayers = tf.add(tf.matmul(z2, W3), b3)

    # Output layer 3
    # W4 = tf.Variable(tf.truncated_normal([layer_3, classes], stddev=0.1))
    # b4 = tf.Variable(tf.constant(0.1, shape=[classes]))
    # predLayers = tf.add(tf.matmul(z3, W4), b4)

    # Output layer 4
    W5 = tf.Variable(tf.truncated_normal([layer_4, classes], stddev=0.1))
    b5 = tf.Variable(tf.constant(0.1, shape=[classes]))
    predLayers = tf.add(tf.matmul(z4, W5), b5)

    #      Run Softmax

    lossLayers = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yholder, logits=predLayers))

    train_prediction_layers = tf.nn.softmax(predLayers)

    optimizerLayers = tf.train.GradientDescentOptimizer(0.00001).minimize(lossLayers)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(epochs):

            sess.run(optimizerLayers, feed_dict={xholder: xtrain, yholder: yListTrain})

        print("\nWith layers Optimization Finished!\n")

        correct_prediction_layers = tf.equal(tf.argmax(train_prediction_layers, 1), tf.argmax(yholder, 1))
        accuracyLayers = tf.reduce_mean(tf.cast(correct_prediction_layers, tf.float32))
        print("Layers result:",sess.run(accuracyLayers, feed_dict={xholder: xtest, yholder: yListTest}))












