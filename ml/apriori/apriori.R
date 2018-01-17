# Apriori

install.packages('arules')
library(arules)
dataset = read.csv('./../../data/shopping.csv', header = FALSE)
dataset = read.transactions('./../../data/shopping.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))
inspect(sort(rules, by = 'lift')[1:10])