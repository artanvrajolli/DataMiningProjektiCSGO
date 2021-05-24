import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#
#
#

#pd.set_option('display.max_columns', None)
data = pd.read_csv("test.csv")

# data.drop(['Map_1'],axis=1)
# data.drop(['Map_2'],axis=1)
# data.drop(['Map_3'],axis=1)

final = pd.get_dummies(data,prefix=['Team_1','Team_2'],columns=['Team_1','Team_2'])

X = final.drop(['winner'],axis=1)
y = final["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.30,random_state = 42)

logreg = LogisticRegression(solver='lbfgs', max_iter=100)
logreg.fit(X_train,y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print("Training set accuracy:"+str(score))
print("Test set accuracy:"+str(score2))




pred_set = pd.DataFrame([{'Team_1':'SAW','Team_2':'Invictus'}])
pred_set = pd.get_dummies(pred_set, prefix=['Team_1','Team_2'],columns=['Team_1','Team_2'])

missing_cols2 = set(final.columns) - set(pred_set.columns)
for c in missing_cols2:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

pred_set = pred_set.drop(['winner'],axis=1)
predictions = logreg.predict(pred_set)
for x in predictions:
    print(x)

# data.drop(['link'], axis = 1)

# data['winner'] = np.where(data['Team_score_1'] > data['Team_score_2'], "1", "2")
#
# data['Map3_Team1_Score'] = np.where(data['Map3_Team1_Score'] == '-', 0, data['Map3_Team1_Score'])
# data['Map3_Team2_Score'] = np.where(data['Map3_Team2_Score'] == '-', 0, data['Map3_Team2_Score'])

# data.fillna(0, inplace=True)
# data.drop(data[data['Map_1'] == "Default"].index, inplace = True)

#print(data.head())

#data.to_csv(r'test2.csv',index=False)
#print(data.isnull().sum())

# show secife Team rows
# df = data[(data['Team_1'] == "Imperial") | (data['Team_2'] == "Imperial")]
# Imperial = df.iloc[:]
# print(Imperial.head())

#print(data.head())
print("Test")