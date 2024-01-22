import pandas as pd

def PercentToRank(data, yscore):
    count = 0

    if isinstance(data, pd.DataFrame):
        scores = data.values.flatten()
    elif isinstance(data, list):
        scores = data
    else:
        raise ValueError("O argumento 'data' deve ser uma lista ou um DataFrame do Pandas.")

    for i in scores:
        if i <= yscore:
            count += 1

    rank = 100.0 * count / len(scores)
    return round(rank, 2)

def RankToPercent(scores, percent_rank):
    scores.sort()
    index = percent_rank * (len(scores) -1) // 100
    return scores[index]


# Exemplo de uso com uma lista
scores = [34,78,56,17,47,23,45,65,18,39,74,52,4,6,8,15,17,14]
yscore = 65
prank = 90

print('\n------------')
print(f'Percent to Rank -> Value {yscore} is equal/above {PercentToRank(scores, yscore)}%')
print(f'Rank to Percent -> Percent {prank}% is equal/above value {RankToPercent(scores, prank)}')
print('------------')


# Exemplo de uso com um DataFrame do Pandas
# data = pd.DataFrame({'Pontuacoes': [34,78,56,17,47,23,45,65,18,39,74,52,4,6,8,15,17,14]})
# yscore = 45
# print(PercentRank(data, yscore))
