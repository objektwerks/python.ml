# Eclat

install.packages('arules')
library(arules)

dataset = read.csv('./../../../data/shopping.csv')
dataset = read.transactions('./../../../data/shopping.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))
inspect(sort(rules, by = 'support')[1 : 10])