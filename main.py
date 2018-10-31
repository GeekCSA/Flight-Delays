import tensorflow as tf
import numpy as np
import pandas as pd

full_df = pd.read_csv("dataset/mini_flights.csv")

features = len(full_df.columns)
print(full_df.values)
x_df = full_df.drop(['ARRIVAL_DELAY'], axis=1)

print(60*"-")

print(x_df.values)


# x = tf.placeholder(tf.float32, [None, features])
# y_ = tf.placeholder(tf.float32, [None, 1])
# W = tf.Variable(tf.zeros([features,1]))
# b = tf.Variable(tf.zeros([1]))
# data_x = df.values
# data_y = np.array(df.loc[:,['ARRIVAL_DELAY']])