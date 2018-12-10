import csv
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import math

def split_train_test(np_full_df, train_percent = 0.7, validation_percent = 0.15, test_percent = 0.15):

    # Number of features
    dataset_size = np_full_df.shape[0]

    train_size = math.ceil(train_percent * dataset_size)
    test_validation_size = dataset_size - train_size
    validation_size = math.ceil(validation_percent/(1-train_percent) * test_validation_size)
    test_size = test_validation_size - validation_size

    train_index = train_size
    validation_index = train_index + validation_size
    test_index = dataset_size

    st = np.split(np_full_df, [train_index,validation_index])

    return st[0], st[1], st[2]

def my_func(x):
    return 73*tf.nn.tanh(x*0.0135)

def train(data_x, data_y, x_test, y_test):

    features = data_x.shape[1]

    # Create a table in size #features on n, n will be determined later in the program
    # and initialize W to table in size #featuresX1 because there are #features features for each point in the train data.

    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])

    number_of_neurons_each_layer = [features, 100, 50, 25 ,12, 6, 1]
    matrix_size = list(zip(number_of_neurons_each_layer, number_of_neurons_each_layer[1:]))
    input_into_hidden_layers = [x]

    W_i = []
    b_i = []

    for i, mat_size in enumerate(matrix_size):
        if i == len(matrix_size) - 1:
            W = tf.Variable(tf.truncated_normal([mat_size[0], mat_size[1]], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[mat_size[1]]))

        else:
            W = tf.Variable(tf.truncated_normal([mat_size[0], mat_size[1]], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[mat_size[1]]))
            a = tf.matmul(input_into_hidden_layers[-1], W) + b
            z = tf.nn.leaky_relu(a)

            input_into_hidden_layers.append(z)
        W_i.append(W)
        b_i.append(b)

    w_final = W_i[-1]
    b_final = b_i[-1]
    z_final = input_into_hidden_layers[-1]
    y = tf.matmul(z_final, w_final) + b_final

    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.AdamOptimizer(0.0005).minimize(loss)

    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            sess.run(update, feed_dict = {x:data_x, y_:data_y})
            if i % (epochs/10) == 0 :
                print('\nIteration:', i)
                # ypretrain = y.eval(feed_dict={x: data_x})
                # ypretest = y.eval(feed_dict={x: x_test})

                # costabstest = sess.run(tf.reduce_mean(tf.abs(ypretest - y_test)))
                # costabstrain = sess.run(tf.reduce_mean(tf.abs(ypretrain - data_y)))
        		# print('\nIteration:', i, "MAETest: ", costabstest)
                # print('\nIteration:', i, "MAETrain: ", costabstrain)

        ypretest = y.eval(feed_dict={x: x_test})
        costabstest = sess.run(tf.reduce_mean(tf.abs(ypretest - y_test)))

        detailes = costabstest, number_of_neurons_each_layer, epochs
        with open('nnStrecture.csv', 'a') as fd:
            fd.write(str(detailes))
            fd.write("\n")

        print('\nIteration:', i, "MAETest: ", costabstest)
        return costabstest 

def test(x_test, W, b):

    y_prediction = np.matmul(x_test,W) + b

    return y_prediction

def normalization(data_x):

    x = data_x

    columns_mean = np.mean(x, axis=0) # calculates the mean of each column
    x -= columns_mean

    columns_std = np.std(x, axis=0) # this calcualtes the standard deviation of each column
    columns_std[columns_std == 0.] = 1.
    x /= columns_std

    return x, columns_mean, columns_std

def print_MAE_MSE(x_data, y_data, W, b, y_mean_columns, y_std_columns):
    # Conversion of the prediction time delay to minutes and conversion of the real delay time to minutes.
    # This conversion must be performed because we have normalized the dataset and now we are returning the values to their real value.

    y_prediction = test(x_data, W, b)
    y_prediction_min = y_prediction * y_std_columns
    y_prediction_min += y_mean_columns

    y_real_min = y_data * y_std_columns
    y_real_min += y_mean_columns

    diff_pow = (y_prediction_min - y_real_min)*(y_prediction_min - y_real_min)
    diff_abs = np.absolute(y_prediction_min - y_real_min)
    diff_random = np.absolute(5.858338583 - y_real_min)

    print("\n\n\tMAE: ", np.mean(diff_abs))
    print("\tMSE: ", np.mean(diff_pow))
    print("\tRansom prediction: ", np.mean(diff_random))

def main():
    y_data_column_name = 'ARRIVAL_DELAY'

    # Read dataset file
    print("\n\nStart reading")
    full_df = pd.read_csv("dataset/ROW3k.csv")
    np_full_df = np.array(full_df, dtype='f')
    print("\tFinish reading")

    np.random.seed(52)
    tf.set_random_seed(52)
    np.random.shuffle(np_full_df)

    print("\n\nStart divide to data_x and data_y")
    # Drop the data_y from complete table.
    data_x = np.array(full_df.drop([y_data_column_name], axis=1), dtype='f')
    data_y = np.array(full_df.loc[:,[y_data_column_name]], dtype='f')
    print("\n\nFinish divide to data_x and data_y")

    print("\n\nStart normalization")
    # data_x, x_mean_columns, x_std_columns = normalization(data_x)
    # data_y, y_mean_columns, y_std_columns = normalization(data_y)
    print("\tFinish normalization")

    print("\n\nStart split the data to train and test by 70%-30%")
    x_train, x_validation, x_test = split_train_test(data_x, 0.7, 0.0, 0.3)
    y_train, y_validation, y_test = split_train_test(data_y, 0.7, 0.0, 0.3)
    print("\n\nFinish split the data to train and test by 70%-30%")

    print("\n\nStart training")
    W_b = train(x_train, y_train, x_test, y_test)
    print("\tFinish training")

if __name__== "__main__":
    main()
