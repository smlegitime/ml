#Eclat

#Importing the datset
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv')
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

#Training Eclat on the dataset
sets = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

#Visualizing the results
inspect(sort(sets, by = 'support')[1:10])