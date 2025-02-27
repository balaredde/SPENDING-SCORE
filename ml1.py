import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error  # Corrected import statement
from sklearn.model_selection import train_test_split

labelencoder = LabelEncoder()
lr = LinearRegression()

a = pd.read_csv('new.csv')

y = a['Spending_Score']
x = a.drop(columns='Spending_Score')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

if 'Gender' in x_train.columns:
    x_train['Gender'] = labelencoder.fit_transform(x_train['Gender'])
    x_test['Gender'] = labelencoder.transform(x_test['Gender'])  # Use transform instead of fit_transform

# Use One-Hot Encoding
x_train = pd.get_dummies(x_train, columns=['Gender'], drop_first=True)
x_test = pd.get_dummies(x_test, columns=['Gender'], drop_first=True)

lr.fit(x_train, y_train)
lr_prediction = lr.predict(x_test)
print(lr_prediction)
lr_score = lr.score(x_test, y_test)
lr_mae = mean_absolute_error(y_test, lr_prediction)
print('Score:', lr_score)
print('Mean Absolute Error:', lr_mae)
