# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:41:23 2019

@author: Menna El-Zahaby
"""

#import necessary libraries and load the data
import numpy as np
import pandas as pd

training_data = pd.read_csv('training.csv', sep = ';')
validation_data = pd.read_csv('validation.csv', sep = ';')

#preprocessing data

#split coordinate columns to separate columns
new = training_data['variable2'].str.split(',', n = 1, expand = True)
training_data['variable2x'] = new[0]
training_data['variable2y'] = new[1]
training_data.drop(columns = ['variable2'], inplace = True)

new = training_data['variable3'].str.split(',', n = 1, expand = True)
training_data['variable3x'] = new[0]
training_data['variable3y'] = new[1]
training_data.drop(columns = ['variable3'], inplace = True)

new = training_data['variable8'].str.split(',', n = 1, expand = True)
training_data['variable8x'] = new[0]
training_data['variable8y'] = new[1]
training_data.drop(columns = ['variable8'], inplace = True)

new = validation_data['variable2'].str.split(',', n = 1, expand = True)
validation_data['variable2x'] = new[0]
validation_data['variable2y'] = new[1]
validation_data.drop(columns = ['variable2'], inplace = True)

new = validation_data['variable3'].str.split(',', n = 1, expand = True)
validation_data['variable3x'] = new[0]
validation_data['variable3y'] = new[1]
validation_data.drop(columns = ['variable3'], inplace = True)

new = validation_data['variable8'].str.split(',', n = 1, expand = True)
validation_data['variable8x'] = new[0]
validation_data['variable8y'] = new[1]
validation_data.drop(columns = ['variable8'], inplace = True)

# dealing with missing data using forward fill method
training_data.fillna(method='ffill',inplace=True)
validation_data.fillna(method='ffill',inplace=True)

#split data to features and target
train_target = training_data['classLabel']
train_features = training_data.drop('classLabel', axis = 1)

valid_target = validation_data['classLabel']
valid_features = validation_data.drop('classLabel', axis = 1)

#normalizing numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical = ['variable11', 'variable14', 'variable15', 'variable17', 'variable19',
             'variable2x', 'variable2y', 'variable3x', 'variable3y', 'variable8x',
             'variable8y']

train_features[numerical] = scaler.fit_transform(train_features[numerical])
valid_features[numerical] = scaler.fit_transform(valid_features[numerical])

#one hot encoding for non numeric variables
training_features_final = pd.get_dummies(train_features)
training_target = train_target.map({'no.': 0, 'yes.': 1})

validation_features_final = pd.get_dummies(valid_features)
validation_target = valid_target.map({'no.': 0, 'yes.': 1})


#adding missing columns to validation set
feature_difference = set(training_features_final) - set(validation_features_final)
feature_difference_df = pd.DataFrame(data=np.zeros((validation_features_final.shape[0],
                                                    len(feature_difference))),
                                     columns=list(feature_difference))
validation_features_final = validation_features_final.join(feature_difference_df)



#training model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score
from sklearn.model_selection import GridSearchCV

model = LogisticRegression(random_state = 0)
parameters = {'C': [0.1, 1, 10]}
scorer = make_scorer(fbeta_score, beta = 0.5)
grid_obj = GridSearchCV(model, parameters, scoring = scorer)
grid_fit = grid_obj.fit(training_features_final, training_target)
best_model = grid_fit.best_estimator_

#make predictions on validation data
valid_predictions = best_model.predict(validation_features_final)

# print accuracy score and F score on validarion data
print("\nValidation Model\n------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(validation_target, valid_predictions)))
print("Final F-score on the validation data: {:.4f}".format(fbeta_score(validation_target, valid_predictions, beta = 0.5)))


