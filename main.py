import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
tf.disable_v2_behavior()
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import pandas as pd


if __name__ == '__main__':

    address = r'C:\Users\USER\PycharmProjects\pythonProject4\Project\data.csv'
    df = pd.read_csv(address)

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

    x = df.drop('JobSatisfaction', axis=1)

    y = df.JobSatisfaction
    #[1470 rows x 34 columns]
    initial = x.astype(np.float32)
    yList = y.tolist()
    y_ = {}
    for i in range(0 , len(yList)):
        if yList[i]==1:
            y_[i] =(1,0,0,0)
        if yList[i]==2:
            y_[i] =(0,1,0,0)
        if yList[i]==3:
            y_[i] =(0,0,1,0)
        if yList[i]==4:
            y_[i] =(0,0,0,1)


    # 35 feature
    categories = 4
    features = 35
    xPlace = tf.placeholder(tf.float32, [None, features])
    yPlace = tf.placeholder(tf.float32, [None, categories])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    #(34,4)
    W = tf.Variable(tf.zeros([34, categories]))
    #(1470,4)
    b = tf.Variable(tf.zeros([1470,4]))
    #[1470 rows x 34 columns]

    pred = tf.nn.softmax(tf.matmul(initial, W) + b)
    #loss = -tf.reduce_mean(tf.log(pred) *y )
    mat = tf.matmul( initial,W)
    # print(mat.shape)
    # print(type(mat))
    # print(b.shape)
    yMat = np.array([y_[i] for i in y_.keys()])

    entro = tf.nn.softmax_cross_entropy_with_logits_v2(yMat,b+ mat )
    # print(entro)
    # print(yMat)
    # labels, logits, axis = None, name = None, dim = None
    loss = tf.math.reduce_mean(entro,y_)
   # loss = -tf.reduce_mean(y_ * tf.log(pred))

    # loss = tf.reduce_mean(-(y_ * tf.log(pred) + (1 - y) * tf.log(1 -pred)))


    update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # accuracy = tf.reduce_mean(tf.cast(update, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0, 10000):

        sess.run(update, feed_dict={x: x_train, y: y_train})

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

