import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv("dataset/mini_flights.csv")
titles = df.values
first = titles[0]

features = len(df.columns)

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
# data_x = np.array([[2,4],[3,9],[4,16],[6,36],[7,49]])
# data_y = np.array([[70],[110],[165],[390],[550]])