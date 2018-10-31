import tensorflow as tf
import numpy as np
import pandas as pd

y_data_column_name = 'ARRIVAL_DELAY'

#Read dataset file
full_df = pd.read_csv("dataset/mini_flights.csv")

# Number of features
features = len(full_df.columns)

# Drop the data_y from complete table.
data_x = np.array(full_df.drop([y_data_column_name], axis=1))
data_y = np.array(full_df.loc[:,[y_data_column_name]])

# Create a table in size #features on n, n will be determined later in the program
# and initialize W to table in size #featuresX1 because there are #features features for each point in the train data.

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))

# initialize the loss function and update function that runs the gradient descent with alpha 0.001 and our loss function.

y = tf.matmul(x,W) + b
loss = tf.reduce_mean(tf.pow(y - y_, 2))
update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100000):
    sess.run(update, feed_dict = {x:data_x, y_:data_y})
    if i % 10000 == 0 :
        print('Iteration:' , i , ' W:' , sess.run(W) , ' b:' , sess.run(b), ' loss:', loss.eval(session=sess, feed_dict = {x:data_x, y_:data_y}))

