# Hotel_Booking_Cancellation_Prediction_with_Random_Forest_Alghorithm
The provided code appears to be a Python script for data preprocessing, feature selection, and modeling using the Random Forest Classifier algorithm. Here is a breakdown of the code:
#### Importing Relevant Libraries
The script starts by importing the necessary libraries, including pandas, numpy, matplotlib.pyplot, and seaborn.

#### Loading Dataset
The script loads a dataset from a CSV file named 'booking.csv' using the pandas library.

#### Data Preprocessing
The script performs various data preprocessing steps, including:
- Dropping the 'Booking_ID' column from the dataset.
- Converting the 'booking status' column to binary values (0 for 'Not_Canceled' and 1 for 'Canceled').
- Dropping columns with correlation coefficients lower than the average correlation coefficient with the 'booking status' column.
- Converting the 'date of reservation' column to datetime format.
- Handling outliers by capping values above the upper bound and below the lower bound.

#### Data Conversion using One-Hot Encoder
The script uses one-hot encoding to convert categorical variables into binary columns.

#### Data Scaling
The script applies standard scaling to the numerical features using the StandardScaler from sklearn.preprocessing.

#### Data Modeling
The script splits the data into training and testing sets using train_test_split from sklearn.model_selection. It then fits a Random Forest Classifier model to the training data and evaluates the model's performance on the testing data.

#### Applying SelectFromModel
The script applies feature selection using SelectFromModel from sklearn.feature_selection. It selects the most important features based on the Random Forest Classifier model's feature importances.

#### Hyperparameter Tuning
The script performs hyperparameter tuning using RandomizedSearchCV from sklearn.model_selection. It searches for the best combination of hyperparameters for the Random Forest Classifier model.

#### Univariate Analysis
The script performs univariate analysis by evaluating the Random Forest Classifier model's performance on each individual feature separately.
