import tensorflow.compat.v1 as tf
from sklearn.preprocessing import LabelEncoder
tf.disable_v2_behavior()
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import pandas as pd
if _name_ == '_main_':

    address = r'C:\Users\USER\PycharmProjects\pythonProject4\Project\data.csv'
    df = pd.read_csv(address)
    print(df.shape)

    #change features with words  to numbers
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
    print(df.head(3))


    train_ds, test_ds = nltk.load('mnist', split=['train', 'test[:30%]'])

    # 35 feature
    categories = 4
    features = 35
    # x = tf.placeholder(tf.float32, [None, features])
    # y_ = tf.placeholder(tf.float32, [None, categories])
    x = df.drop('JobSatisfaction', axis=1)
    y = df.JobSatisfaction
    W = tf.Variable(tf.zeros([features, categories]))
    b = tf.Variable(tf.zeros([categories]))

    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    # loss = -tf.reduce_mean(y_ * tf.log(pred))
    loss = -tf.reduce_mean(y * tf.log(pred))


    update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # data_x = np.array([convert2vec(data[i]) for i in range(len(df))])
    # data_y = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])

    for i in range(len(df)):
        print('Prediction for: "' + test_ds[i] + ': "', nltk.sess.run(y, feed_dict={x:(test_ds[i])}))