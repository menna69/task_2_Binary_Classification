Task2 Binary classification problem,
import necessary libraries for data which are numpy and pandas, load the data from training and validation csv files.
data preprocessing, 
split columns that contain coordinate data into two new columns.
deal with missing data by applying forward fill on training and validation datasets.
split datasets into features and target data.
data normalization,
feature scaling the numericl data to all range between 0 and 1 using MinMaxScalar function.
one hot encode non-numeric data so all values be in numerical form.
change the target data to be 0 for no and 1 for yes since it's binary data.
add the extra columns in the training data to the validation data and give them zero value.
train the model,
using logistic regression model as the best model for training binary data.
using grid search to tune hyperparameters and determine the optimal values for training the model.
train the model on the training data and predict the target varible on the validation data.
measure the accuracy score and f_score of the model as our performance metrics.
