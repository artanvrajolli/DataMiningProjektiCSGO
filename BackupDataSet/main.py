import numpy as np
import pandas as pd


TableCSGOMatch = pd.read_csv("Table_1.csv")
# 'Map_1','Map1_Team1_Score','Map1_Team2_Score',"Map_2","Map2_Team1_Sc'ore",'Map2_Team2_Score','Map_3','Map3_Team1_Score',Map3_Team2_Score
# create a dataframe #Team_1,Team_2,Team_score_1,Team_score_2
dataframe = pd.DataFrame(TableCSGOMatch, columns=[
'Team_1', 'Team_2', 'Team_score_1', 'Team_score_2',
'Map_1','Map1_Team1_Score','Map1_Team2_Score',
"Map_2","Map2_Team1_Score",'Map2_Team2_Score',
'Map_3','Map3_Team1_Score',"Map3_Team2_Score"
])

def predict2(t1,t2):
    dataFrameTeam1 = dataframe[((dataframe['Team_1'] == t1) & (dataframe['Team_2'] == t2))]
    maps1Score = []
    maps2Score = []
    maps3Score = []
    mapsScoreResult = []
    for x in dataFrameTeam1['Map1_Team1_Score']:
        maps1Score.append(x) # map 1
    for x in dataFrameTeam1['Map2_Team1_Score']:
        maps2Score.append(x) # map 2
    for x in dataFrameTeam1['Map3_Team1_Score']:
        maps3Score.append(x) # map 3
    i=0
    while i < len(maps1Score):
        if(maps1Score[i] != "-"):
            mapsScoreResult.append(maps1Score[i])
        if (maps2Score[i] != "-"):
            mapsScoreResult.append(maps2Score[i])
        if (maps3Score[i] != "-"):
            mapsScoreResult.append(maps3Score[i])
        i+=1
    return mapsScoreResult

c1Astralis = predict2("Astralis", "Natus Vincere")
c2Navi = predict2("Natus Vincere", "Astralis")
c1Navi = predict2()
output = []
for x in range(0,len(c1)):
    output.append(np.std([int(c1[x]),int(c2[x])]))

print(np.std(output))
#x = numpy.mean(speed)
def predict(t1,t2):
    #| ((dataframe['Team_1'] == t2) & (dataframe['Team_2'] == t1))
    dataFrameTeam1 = dataframe[((dataframe['Team_1'] == t1) & (dataframe['Team_2'] == t2))]
    return np.mean(dataFrameTeam1['Team_score_1']),np.mean(dataFrameTeam1['Team_score_2'])

def meanAll(t1,t2):
    Mean1_team1, Mean1_team2 = predict(t1, t2)
    Mean2_team2, Mean2_team1 = predict(t2, t1)
    Mean1F = (Mean1_team1 + Mean2_team1) / 2
    Mean2F = (Mean1_team2 + Mean2_team2) / 2
    return Mean1F,Mean2F


#print(meanAll("Astralis", "Natus Vincere"))


#print(dataFrame1['Team_1'])
#dataframe['1'] == t1 && data['2'] = t2
#dataframe['1'] == t2 && data[2] = t1
#a || b
# TeamOne      Third Impact  ...                 -                 -