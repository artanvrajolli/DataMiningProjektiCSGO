import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from IPython.display import display
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("Table_1.csv")

data['winner'] = np.where(data['Team_score_1'] >  data['Team_score_2'], "Team1Winner","Team2Winner")
data['Map_3'] = np.where(data['Map_3'] == '-', 0, data['Map_3'])
data['Map3_Team1_Score'] = np.where(data['Map3_Team1_Score'] == '-', 0, data['Map3_Team1_Score'])
data['Map3_Team2_Score'] = np.where(data['Map3_Team2_Score'] == '-', 0, data['Map3_Team2_Score'])

# data.loc[0:10,'winner'] = 'H'
# print(data.loc[0:10,'winner'])
#pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)
print(scatter_matrix(data[['Team_score_1','Team_score_2','Map1_Team1_Score','Map1_Team2_Score','Map2_Team1_Score',
                     'Map2_Team2_Score','Map3_Team1_Score','Map3_Team2_Score']], figsize=(10,10)))
# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.float64)
#


# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
# data.drop(['Map_1'],1)
# data.drop(['Map_2'],1)
# data.drop(['Map_3'],1)
# data.drop(['Team_1'],1)
# data.drop(['Team_2'],1)
data.drop(['Team_score_1'],1)
data.drop(['Team_score_2'],1)
X_all = data.drop(['winner'],1)
y_all = data['winner']

#Center to the mean and component wise scale to unit variance.
cols = [['Map1_Team1_Score','Map1_Team2_Score','Map2_Team1_Score','Map2_Team2_Score','Map3_Team1_Score','Map3_Team2_Score']]
for col in cols:
    X_all[col] = scale(X_all[col])
#last 3 wins for both sides
X_all.Map_1 = X_all.Map_1.astype('str')
X_all.Map_2 = X_all.Map_2.astype('str')
X_all.Map_3 = X_all.Map_3.astype('str')
X_all.Team_1 = X_all.Team_1.astype('str')
X_all.Team_2 = X_all.Team_2.astype('str')

# clean_dataset(X_all)


# we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
    output = pd.DataFrame(index=X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        output = output.join(col_data)
    return output
X_all = preprocess_features(X_all)
#print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify = y_all)

# print(data.isnull().sum())
#
# print(data['Map_3'])

def train_classifier(clf, X_train, y_train):
    # np.any(np.isnan(X_train))
    # np.all(np.isfinite(X_train))
    # np.any(np.isnan(y_train))
    # np.all(np.isfinite(y_train))
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Trained model in {:.4f} seconds".format(end - start))

def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target, y_pred, average='micro'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    train_classifier(clf, X_train, y_train)
    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))
    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))

# Initialize the three models (XGBoost is initialized later)
# clf_A = LogisticRegression(random_state = 20)
# clf_B = SVC(random_state = 812, kernel='rbf')
#
# train_predict(clf_A, X_train, y_train, X_test, y_test)
# train_predict(clf_B, X_train, y_train, X_test, y_test)

clf = RandomForestClassifier(max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)

print(score)