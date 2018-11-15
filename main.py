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

def train(data_x, data_y):

    features = data_x.shape[1]

    # Create a table in size #features on n, n will be determined later in the program
    # and initialize W to table in size #featuresX1 because there are #features features for each point in the train data.

    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.zeros([features,1]))
    b = tf.Variable(tf.zeros([1]))

    # Split 'value' into 3 tensors along dimension 1

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

    return sess.run(W), sess.run(b)

def test(x_test, y_test, W, b):
    y_prediction = np.matmul(x_test,W) + b
    diff = abs(y_test - y_prediction)

    correctResults = diff <= 1.5
    sumOfCorrectResults = np.sum(diff)

    return y_prediction

def normalization(data_x):

    x = data_x

    columns_mean = np.mean(x, axis=0) # calculates the mean of each column
    x -= columns_mean

    # print("\n\ncolumns_mean: " ,columns_mean, "\t\tlen(): " ,len(columns_mean))
    # print("\n\n\t\t np.mean(x): " , np.mean(x))

    columns_std = np.std(x, axis=0) # this calcualtes the standard deviation of each column
    columns_std[columns_std == 0.] = 1.
    x /= columns_std

    # print("\n\ncolumns_std: ", columns_std, "\t\tlen(): " ,len(columns_std))
    # print("\n\n\t\t np.std(x): " , np.std(x))



    return x, columns_mean, columns_std

def main():
    y_data_column_name = 'ARRIVAL_DELAY'

    # Read dataset file
    print("\n\nStart reading")
    full_df = pd.read_csv("dataset/Medium_Flight_delete_some_columns.csv")
    np_full_df = np.array(full_df,dtype='f')
    print("\tFinish reading")

    # Drop the data_y from complete table.
    data_x = np.array(full_df.drop([y_data_column_name], axis=1), dtype='f')
    data_y = np.array(full_df.loc[:,[y_data_column_name]],dtype='f')

    # print(data_x)
    # print(data_y)

    print("\n\nStart normalization")
    data_x, x_mean_columns, x_std_columns = normalization(data_x)
    data_y, y_mean_columns, y_std_columns = normalization(data_y)
    print("\tFinish normalization")

    # print(data_x)
    # print(data_y)

    x_train, x_validation, x_test = split_train_test(data_x, 0.7, 0.15, 0.15)
    y_train, y_validation, y_test = split_train_test(data_y, 0.7, 0.15, 0.15)

    # print(x_train)
    # print(x_test)
    # print(x_validation)
    #
    # print(y_train)
    # print(y_test)
    # print(y_validation)

    print("\n\nStart training")
    W, b = train(x_train, y_train)
    print("\tFinish training")

    print("\n\nStart testing")


    # Conversion of the prediction time delay to minutes and conversion of the real delay time to minutes. This conversion must be performed because we have normalized the dataset and now we are returning the values ​​to their real value.
    y_prediction = test(x_test, y_test, W, b)
    y_prediction_min = y_prediction * y_std_columns
    y_prediction_min += y_mean_columns

    y_real_min = y_test * y_std_columns
    y_real_min += y_mean_columns

    print("\n\n\t\tResult:\n" , np.c_[y_prediction_min, y_real_min])

    print("\tFinish testing")

if __name__== "__main__":
    main()