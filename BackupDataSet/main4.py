import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("Table_1.csv")
#data.drop(data[data['Map_1'] == "Default"].index, inplace = True)
data['winner'] = np.where(data['Team_score_1'] > data['Team_score_2'], data['Team_1'], data['Team_2'])
data['Map1_Team1_Score'] = np.where(data['Map1_Team1_Score'] is None or data['Map1_Team1_Score'] == '-', 0, data['Map1_Team1_Score'])
data['Map1_Team2_Score'] = np.where(data['Map1_Team2_Score'] is None or data['Map1_Team2_Score'] == '-', 0, data['Map1_Team2_Score'])
data['Map2_Team1_Score'] = np.where(data['Map2_Team1_Score'] is None or data['Map2_Team1_Score'] == '-', 0, data['Map2_Team1_Score'])
data['Map2_Team2_Score'] = np.where(data['Map2_Team2_Score'] is None or data['Map2_Team2_Score'] == '-', 0, data['Map2_Team2_Score'])
data['Map3_Team1_Score'] = np.where(data['Map3_Team1_Score'] is None or data['Map3_Team1_Score'] == '-', 0, data['Map3_Team1_Score'])
data['Map3_Team2_Score'] = np.where(data['Map3_Team2_Score'] is None or data['Map3_Team2_Score'] == '-', 0, data['Map3_Team2_Score'])
data = data.drop(["map","link","Map_1","Map_2","Map_3","Team_score_1","Team_score_2"],axis=1)
data.replace(np.nan,0)
data = data.dropna()

data.to_csv(r'debugTest.csv',index=False)
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

#print(data)


#data.to_csv(r'debugTest.csv',index=False)


print("TEST")