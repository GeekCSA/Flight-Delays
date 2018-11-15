# Project name: predicte delay time of aircraft landing in USA.

The purpose of the project is to predict the delay of the plane landing due to the weather and geographical location of the airport on domestic flights in the USA.

Dependencies:

*	Python 3.5
*	TensorFlow 1.11.0
*	Numpy 1.15.3
*	Pandas 0.23.4

Building dataset:

Our dataset consists of two parts.
*   dataset of scheduled and scheduled times and departures and the actual airport in which the plane and airport landed on each of the 2015 days of domestic flights in the United States (source: [Kaggle](https://www.kaggle.com/usdot/flight-delays)) .
*   dataset of weather conditions at each of the US airports every day in 2015 (source: [NOAA](https://www.ncdc.noaa.gov)).
*   Geographical coordinates of airports (source: [Kaggle](https://www.kaggle.com/usdot/flight-delays)).

We've merged the tables by place and time, and saved it as a CSV file

The complete dataset is in the set: datasets.

Explanation of the structure of the project:

The project consists of several stages:
1.  Reading the dataset
2.  Separation of the dataset to x_data, y_data
3.  Normalization of the dataset.
4.  Separation of x_data, y_data to train, validation, test.
5.  Coaching of the model on x_train and y_train.
6.  Examine the system on x_test and y_test.
7.  Conversion of the prediction time delay to minutes and conversion of the real delay time to minutes. This conversion must be performed because we have normalized the dataset and now we are returning the values ​​to their real value.
