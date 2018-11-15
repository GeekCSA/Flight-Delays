<div class="text-align-center">Project name: predicte delay time of aircraft landing in USA.</div>

The purpose of the project is to predict the delay of the plane landing due to the weather and geographical location of the airport on domestic flights in the USA.

## Dependencies:

*	Python 3.5
*	TensorFlow 1.11.0
*	Numpy 1.15.3
*	Pandas 0.23.4

## Building dataset:

Our dataset consists of two parts.
*   dataset of scheduled and scheduled times and departures and the actual airport in which the plane and airport landed on each of the 2015 days of domestic flights in the United States (source: [Kaggle](https://www.kaggle.com/usdot/flight-delays)) .
*   dataset of weather conditions at each of the US airports every day in 2015 (source: [NOAA](https://www.ncdc.noaa.gov)).
*   Geographical coordinates of airports (source: [Kaggle](https://www.kaggle.com/usdot/flight-delays)).

We've merged the tables by place and time, and saved it as a CSV file

The complete dataset is in the set: dataset.

## Explanation of the structure of the project:

The project consists of several stages:
1.  Reading the dataset
```python
    full_df = pd.read_csv("dataset/dataset_file.csv")
```
2.  Separation of the dataset to x_data, y_data
```python
    data_x = np.array(full_df.drop([y_data_column_name], axis=1), dtype='f')
    data_y = np.array(full_df.loc[:,[y_data_column_name]],dtype='f')
```
3.  Normalization of the dataset. The function returns the normalized data, the mean of each column, and the standard deviation of each column.
```python
    data_x, x_mean_columns, x_std_columns = normalization(data_x)
    data_y, y_mean_columns, y_std_columns = normalization(data_y)
```
4.  Separation of x_data, y_data to train, validation, test. (70% train, 15% validation and 15% test)
```python
    x_train, x_validation, x_test = split_train_test(data_x, 0.7, 0.15, 0.15)
    y_train, y_validation, y_test = split_train_test(data_y, 0.7, 0.15, 0.15)
```
5.  Coaching of the model on x_train and y_train. The function returns the W (weights) and b.
```python
    W, b = train(x_train, y_train)
```
6.  Examine the system on x_test and y_test.  The function returns the values ​​it predicted according to W and b, these values are normalized!
```python
y_prediction = test(x_test, y_test, W, b)
```
7.  Conversion of the prediction time delay to minutes and conversion of the real delay time to minutes. This conversion must be performed because we have normalized the dataset and now we are returning the values to their real value.
```python
    y_prediction_min = y_prediction * y_std_columns
    y_prediction_min += y_mean_columns

    y_real_min = y_test * y_std_columns
    y_real_min += y_mean_columns
```
