from libs import *


def getStats(Team):
    stats = {"win":0,"los":0,"total":0}
    TeamDataframeHome = data[data['Team_1'] == Team]
    TeamDataframeAway = data[data['Team_2'] == Team]

    for index, row in TeamDataframeHome.iterrows():
        if(row["Team_score_1"] > row["Team_score_2"]):
            stats["win"] += 1
        else:
            stats["los"] += 1
        stats["total"] += 1


    for index, row in TeamDataframeAway.iterrows():
        if (row["Team_score_1"] < row["Team_score_2"]):
            stats["win"] += 1
        else:
            stats["los"] += 1
        stats["total"] += 1

    return stats

def getStatsDirect(Team1,Team2):
    stats = {
        "Team1":{"win":0,"los":0},
        "Team2":{"win":0,"los":0},
        "total":0
    }
    TeamDataframeHome = data[(data['Team_1'] == Team1) & (data['Team_2'] == Team2)]
    for index, row in TeamDataframeHome.iterrows():
        if(row["Team_score_1"] > row["Team_score_2"]):
            stats["Team1"]["win"] += 1
            stats["Team2"]["los"] += 1
        else:
            stats["Team1"]["los"] += 1
            stats["Team2"]["win"] += 1
        stats["total"] += 1

    TeamDataframeAway = data[(data['Team_1'] == Team2) & (data['Team_2'] == Team1)]
    for index, row in TeamDataframeAway.iterrows():
        if (row["Team_score_1"] < row["Team_score_2"]):
            stats["Team1"]["win"] += 1
            stats["Team2"]["los"] += 1
        else:
            stats["Team1"]["los"] += 1
            stats["Team2"]["win"] += 1
        stats["total"] += 1
    return stats

def ProbabilityofWinning(Ra, Rb):
        prob =  1 / (1 + pow(10, (Rb - Ra) / 400))  # Ea
        formatted_string = "{:.5f}".format(prob)
        float_value = float(formatted_string)
        return float_value
def predict(team1, team2):
    team1stats = getStats(team1)
    team2stats = getStats(team2)

    team1performance = (1000*team1stats["total"] + 400 * (team1stats["win"]-team1stats["los"])) / team1stats["total"]
    team2performance = (1000*team2stats["total"] + 400 * (team2stats["win"]-team2stats["los"])) / team2stats["total"]

    directteamstats = getStatsDirect(team1,team2)

    if directteamstats["total"] != 0 :
        directteam1performance = (1000*directteamstats["total"] + 400 * (directteamstats["Team1"]["win"]-directteamstats["Team1"]["los"])) / directteamstats["total"]
        directteam2performance = (1000*directteamstats["total"] + 400 * (directteamstats["Team2"]["win"]-directteamstats["Team2"]["los"])) / directteamstats["total"]
        team1ratio = team1performance * 0.45 + directteam1performance * 0.55
        team2ratio = team2performance * 0.45 + directteam2performance * 0.55
    else:
        team1ratio = team1performance
        team2ratio = team2performance

    return {"Team1Prob":ProbabilityofWinning(team1ratio,team2ratio),
            "Team2Prob":ProbabilityofWinning(team2ratio,team1ratio),
            "Team_1":team1,"Team_2":team2}


def firstBracket(upperbracket,lowerbracket):
    i =0
    while i < len(upperbracket):
        pred = predict(upperbracket[0],upperbracket[1])
        if pred["Team1Prob"] > pred["Team2Prob"]:
            upperbracket.append(upperbracket[0])
            lowerbracket.append(upperbracket[1])
        else:
            upperbracket.append(upperbracket[1])
            lowerbracket.append(upperbracket[0])

        print("UpperBraket R1:",pred)
        upperbracket = upperbracket[2:]
        return upperbracket,lowerbracket

def bracketSystem(data,upperBracket,lowerBracket):
    data = data
    upperBracket,lowerBracket  = firstBracket(upperBracket,lowerBracket)
    while len(upperBracket) > 1:
        pred = predict(upperBracket[0],upperBracket[1])
        if pred["Team1Prob"] > pred["Team2Prob"]:
            upperBracket.append(upperBracket[0])
            opponed = upperBracket[1]
        else:
            upperBracket.append(upperBracket[1])
            opponed = upperBracket[0]
        upperBracket = upperBracket[2:]
        print("UpperBracket RN:",pred)
        pred2 = predict(lowerBracket[0],opponed)
        if pred2["Team1Prob"] > pred2["Team2Prob"]:
            lowerBracket.append(lowerBracket[0])
        else:
            lowerBracket.append(opponed)
        print("LowerBracket:",pred2)
        lowerBracket = lowerBracket[1:]

    while len(lowerBracket) > 1:
        pred = predict(lowerBracket[0],lowerBracket[1])
        if pred["Team1Prob"] > pred["Team2Prob"] :
            lowerBracket.append(lowerBracket[0])
        else:
            lowerBracket.append(lowerBracket[1])
        print("LowerBracket:",pred)
        lowerBracket = lowerBracket[2:]
    # Grand Final
    pred = predict(upperBracket[0],lowerBracket[0])
    print("GrandFinal:",pred)