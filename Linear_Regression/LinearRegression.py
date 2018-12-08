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

def train(data_x, data_y, x_test, y_test):

    features = data_x.shape[1]

    # Create a table in size #features on n, n will be determined later in the program
    # and initialize W to table in size #featuresX1 because there are #features features for each point in the train data.

    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.truncated_normal([features,1], stddev=0.1))
    b = tf.Variable(tf.zeros([1]))

    # Split 'value' into 3 tensors along dimension 1

    # initialize the loss function and update function that runs the gradient descent with alpha 0.001 and our loss function.

    y = tf.matmul(x,W) + b
    # loss = tf.reduce_mean(tf.pow(y - y_, 2)) + 0.1 * tf.nn.l2_loss(W) # Ridge
    loss = tf.reduce_mean(tf.pow(y - y_, 2)) # Regular
    update = tf.train.AdamOptimizer(0.0005).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 100000
    for i in range(epochs):
        sess.run(update, feed_dict = {x:data_x, y_:data_y})
        if i % (epochs/10) == 0 :
            # ypretrain = sess.run(y, feed_dict={x: data_x})
            print('Iteration:' , i , ' W:' , sess.run(W) , ' b:' , sess.run(b), ' loss:', loss.eval(session=sess, feed_dict = {x:data_x, y_:data_y}))
    # ypretest = sess.run(y, feed_dict={x: x_test})
    ypretest = y.eval(session=sess,feed_dict={x: x_test})
    costabstest = sess.run(tf.reduce_mean(tf.abs(ypretest - y_test)))
    print('\nIteration:', i, "MAETest: ", costabstest)
    return sess.run(W), sess.run(b)

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
    # print("\tMSE: ", np.mean(diff_pow))
    print("\tRansom prediction: ", np.mean(diff_random))

def main():
    y_data_column_name = 'ARRIVAL_DELAY'

    # Read dataset file
    print("\n\nStart reading")
    full_df = pd.read_csv("../dataset/ROW100k.csv")
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

    # print("\n\nStart normalization")
    # data_x, x_mean_columns, x_std_columns = normalization(data_x)
    # data_y, y_mean_columns, y_std_columns = normalization(data_y)
    # print("\tFinish normalization")

    print("\n\nStart split the data to train and test by 80%-20%")
    x_train, x_validation, x_test = split_train_test(data_x, 0.7, 0.0, 0.3)
    y_train, y_validation, y_test = split_train_test(data_y, 0.7, 0.0, 0.3)
    print("\n\nFinish split the data to train and test by 80%-20%")

    print("\n\nStart training")
    W, b = train(x_train, y_train, x_test, y_test)
    print("\tFinish training")

    # # Check accuracy of the model on train data
    # print("\n\nStart testing on train-data")
    # # print_MAE_MSE(x_train, y_train, W, b, y_mean_columns, y_std_columns)
    # print("\tFinish testing on train-data")
    #
    # # Check accuracy of the model on test data
    # print("\n\nStart testing on test-data")
    # print_MAE_MSE(x_test, y_test, W, b, 0, 1)
    # print("\tFinish testing on test-data")

if __name__== "__main__":
    main()
