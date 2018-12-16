setwd("D:/R studio/Laboratorium/dom2")

install.packages("editrules")
install.packages("Hmisc")
install.packages("deducorrect")
install.packages("caret")
install.packages("ggvis")
install.packages("class")
install.packages("gmodels")
install.packages('e1071', dependencies=TRUE)
install.packages("party")
install.packages("partykit")
install.packages("stats")
install.packages("arules")
install.packages("cluster")
install.packages("ROCR")
install.packages("pROC")
library(editrules)
library(Hmisc)
library(deducorrect)
library(caret)
library(ggvis)
library(class)
library(gmodels)
library(e1071)
library(party)
library(partykit)
library(stats)
library(arules)
require(randomForest)
library(cluster)
library(ROCR)
library(pROC)

telco <- read.csv("Telco-Customer-Churn.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)


# Check how many records are invalid #
valid <- nrow(telco.corrected[complete.cases(telco.corrected), ])
invalid <- nrow(telco.corrected) - valid


# Check all possible values for all attributes, if numeric show min and max SUMMARY(TELCO) #
uniqueValues <- function(data) {
  values <- NULL
  str <- NULL
  for(i in names(data)){
    values <- unique(data$i)
    str <- as.character(i)
    if(i != "customerID" & i != "MonthlyCharges" & i != "TotalCharges" & i != "tenure") {
      print(unique(data[i]))
    } else if(i != "customerID") {
      print(str)
      print(min(as.numeric(as.character(data[,i])), na.rm=T))
      print(max(as.numeric(as.character(data[,i])), na.rm=T))
    }
  }
}
uniqueValues(telco)

# Check rules for all attributes #
E <- editset(c("gender %in% c('Female','Male')",
               "SeniorCitizen %in% c('0','1')",
               "Partner %in% c('Yes','No')",
               "Dependents %in% c('Yes','No')",
               "tenure >= 0",
               "tenure <= 72",
               "PhoneService %in% c('Yes','No')",
               "MultipleLines %in% c('Yes','No','No phone service')",
               "InternetService %in% c('DSL','Fiber optic','No')",
               "OnlineSecurity %in% c('Yes','No','No internet service')",
               "OnlineBackup %in% c('Yes','No','No internet service')",
               "DeviceProtection %in% c('Yes','No','No internet service')",
               "TechSupport %in% c('Yes','No','No internet service')",
               "StreamingTV %in% c('Yes','No','No internet service')",
               "StreamingMovies %in% c('Yes','No','No internet service')",
               "Contract %in% c('Month-to-month','One year','Two year')",
               "PaperlessBilling %in% c('Yes','No')",
               "PaymentMethod %in% c('Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)')",
               "MonthlyCharges >= 18.25",
               "MonthlyCharges <= 118.75",
               "TotalCharges >= 18.8",
               "MonthlyCharges <= 8684.8",
               "Churn %in% c('Yes','No')"
               ))
ve <- violatedEdits(E, telco)
summary(ve)

# Clean NA and replace with mean #
telco.clean <- telco
Mean = mean(telco.clean[, 20], na.rm = TRUE)
telco.clean[, 20][is.na(telco.clean[, 20])] <- Mean
write.csv(telco.clean, file = "telco-clean.csv")

# Correction rules #
rules <- correctionRules("rules2.txt")
corrected <- correctWithRules(rules, telco.clean)
telco.corrected <- corrected$corrected

# Remove customerID column and save corrected data #
telco.corrected$customerID <- NULL
write.csv(telco.corrected, file = "telco-corrected.csv")
telco.corrected <- read.csv("telco-corrected.csv", header=TRUE, sep=",")
telco.corrected$X <- NULL

# Standarize between 0 1 and normalize #
normalize <- function(x) {
  x <- as.numeric(x)
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

telco.normalized <- telco.corrected
telco.normalized <- as.data.frame(lapply(telco.corrected[,1:20], normalize))
write.csv(telco.normalized, file = "telco-normalized.csv")

telco.normalized <- read.csv("telco-normalized.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)
telco.normalized$Churn = as.factor(telco.normalized$Churn)
telco.normalized$X <- NULL

telco.normalized <- read.csv("telco-normalized-with-categories.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)
telco.normalized$Churn = as.factor(telco.normalized$Churn)
telco.normalized$X <- NULL


# Training and test dataset #
set.seed(1234)
ind <- sample(2, nrow(telco.normalized), replace=TRUE, prob=c(0.67, 0.33))

# Compose training set
telco.training <- telco.normalized[ind==1, 1:20]

# Inspect training set
head(telco.training)

# Compose test set
telco.test <- telco.normalized[ind==2, 1:20]

# Inspect test set
head(telco.test)

# Compose training labels
telco.trainLabels <- telco.normalized[ind==1,20]

# Inspect result
print(telco.trainLabels)

# Compose test labels
telco.testLabels <- telco.normalized[ind==2, 20]

# Inspect result
print(telco.testLabels)



##### KNN #####

# Build the model
telco.KNNprediction <- knn(train = telco.training, test = telco.test, cl = telco.trainLabels, k=2)

# Inspect
telco.KNNprediction

# Put in a data frame
telcoTestLabels <- data.frame(telco.testLabels)

# Merge pred and testLabels 
telco.merge <- data.frame(telco.KNNprediction, telco.testLabels)

# Specify column names for `merge`
names(telco.merge) <- c("Predicted Churn", "Observed Churn")

# Inspect `merge` 
telco.merge

correct <- function(merge) {
  count <- nrow(merge)
  rows <- c(1:count)
  result <- 0
  
  for(r in rows) {
    predicted <- as.character(merge[r, "Predicted Churn"])
    observed <- as.character(merge[r, "Observed Churn"])
    if(predicted == observed) {
      result <- result + 1
    }
  }
  return(cat("Correct predictions: ", result, "/", count, " = ", (result*100)/count, "%"))
}

correct(telco.merge)
# Correct predictions:  2253 / 2313  =  97.40597 %

KNNConfusion <- confusionMatrix(telco.KNNprediction,telco.test[,20])
KNNAccuracy <- KNNConfusion$overall[["Accuracy"]]

table(telco.KNNprediction)
confusionMatrix(telco.KNNprediction,telco.test[,20])
CrossTable(x = telco.testLabels, y = telco.KNNprediction, prop.chisq=FALSE)

#TPR = TP/TP + FN
#FPR = FP/FP + TN
TPR_KNN = KNNConfusion[["table"]][[2,2]] / (KNNConfusion[["table"]][[2,2]] + KNNConfusion[["table"]][[1,2]])
FPR_KNN = KNNConfusion[["table"]][[2,1]] / (KNNConfusion[["table"]][[2,1]] + KNNConfusion[["table"]][[1,1]])
TPR_KNN
FPR_KNN

pred <- prediction(as.numeric(telco.KNNprediction), as.numeric(telco.test$Churn))
perf <- performance(pred,"tpr","fpr")
plot(perf,col="blue", type="l")
abline(0,1)

# Caret #
# Train a model
model_knn <- train(telco.training[, 1:19], telco.training[, 20], method='knn')

# Predict the labels of the test set
telco.KNN2prediction <- predict(object=model_knn,telco.test[,1:19])

# Evaluate the predictions
table(telco.KNN2prediction)

KNN2Confusion <- confusionMatrix(telco.KNN2prediction,telco.test[,20])
KNN2Accuracy <- KNN2Confusion$overall[["Accuracy"]]

# Confusion matrix 
confusionMatrix(telco.KNN2prediction,telco.test[,20])
CrossTable(x = telco.testLabels, y = telco.KNN2prediction, prop.chisq=FALSE)

#TPR = TP/TP + FN
#FPR = FP/FP + TN
TPR_KNN2 = KNN2Confusion[["table"]][[2,2]] / (KNN2Confusion[["table"]][[2,2]] + KNN2Confusion[["table"]][[1,2]])
FPR_KNN2 = KNN2Confusion[["table"]][[2,1]] / (KNN2Confusion[["table"]][[2,1]] + KNN2Confusion[["table"]][[1,1]])
TPR_KNN2
FPR_KNN2

pred <- prediction(as.numeric(telco.KNN2prediction), as.numeric(telco.test$Churn))
perf <- performance(pred,"tpr","fpr")
plot(perf,col="blue", type="l")
abline(0,1)


##### Naive Bayes #####

model_bayes <- naiveBayes(Churn ~ ., data = telco.training)
telco.BAYESprediction <- predict(object=model_bayes,telco.test[,1:19])

bayesConfusion <- confusionMatrix(telco.BAYESprediction,telco.test[,20])
bayesAccuracy <- bayesConfusion$overall[["Accuracy"]]

table(telco.BAYESprediction)
confusionMatrix(telco.BAYESprediction,telco.test[,20])
CrossTable(x = telco.testLabels, y = telco.BAYESprediction, prop.chisq=FALSE)

#TPR = TP/TP + FN
#FPR = FP/FP + TN
TPR_bayes = bayesConfusion[["table"]][[2,2]] / (bayesConfusion[["table"]][[2,2]] + bayesConfusion[["table"]][[1,2]])
FPR_bayes = bayesConfusion[["table"]][[2,1]] / (bayesConfusion[["table"]][[2,1]] + bayesConfusion[["table"]][[1,1]])
TPR_bayes
FPR_bayes

pred <- prediction(as.numeric(telco.BAYESprediction), as.numeric(telco.test$Churn))
perf <- performance(pred,"tpr","fpr")
plot(perf,col="blue", type="l")
abline(0,1)


##### C4.5 Tree #####

model_C45 <- ctree(Churn ~ ., data=telco.training)
telco.C45prediction <- predict(object=model_C45,telco.test[,1:19])

C45Confusion <- confusionMatrix(telco.C45prediction,telco.test[,20])
C45Accuracy <- C45Confusion$overall[["Accuracy"]]

table(telco.C45prediction)
confusionMatrix(telco.C45prediction,telco.test[,20])
CrossTable(x = telco.testLabels, y = telco.C45prediction, prop.chisq=FALSE)

plot(model_C45)

#TPR = TP/TP + FN
#FPR = FP/FP + TN
TPR_C45 = C45Confusion[["table"]][[2,2]] / (C45Confusion[["table"]][[2,2]] + C45Confusion[["table"]][[1,2]])
FPR_C45 = C45Confusion[["table"]][[2,1]] / (C45Confusion[["table"]][[2,1]] + C45Confusion[["table"]][[1,1]])
TPR_C45
FPR_C45

pred <- prediction(as.numeric(telco.C45prediction), as.numeric(telco.test$Churn))
perf <- performance(pred,"tpr","fpr")
plot(perf,col="blue", type="l")
abline(0,1)

##### Random Forest #####
model_forest <- randomForest(Churn ~ ., data=telco.training)
telco.FORESTprediction <- predict(object=model_forest,telco.test[,1:19])

forestConfusion <- confusionMatrix(telco.FORESTprediction,telco.test[,20])
forestAccuracy <- forestConfusion$overall[["Accuracy"]]

table(telco.FORESTprediction)
confusionMatrix(telco.FORESTprediction,telco.test[,20])
CrossTable(x = telco.testLabels, y = telco.FORESTprediction, prop.chisq=FALSE)

#TPR = TP/TP + FN
#FPR = FP/FP + TN
TPR_forest = forestConfusion[["table"]][[2,2]] / (forestConfusion[["table"]][[2,2]] + forestConfusion[["table"]][[1,2]])
FPR_forest = forestConfusion[["table"]][[2,1]] / (forestConfusion[["table"]][[2,1]] + forestConfusion[["table"]][[1,1]])
TPR_forest
FPR_forest

pred <- prediction(as.numeric(telco.FORESTprediction), as.numeric(telco.test$Churn))
perf <- performance(pred,"tpr","fpr")
plot(perf,col="blue", type="l")
abline(0,1)

# Plots #
accuracies <- c(KNNAccuracy, KNN2Accuracy, bayesAccuracy, C45Accuracy, forestAccuracy)
barplot(accuracies, main="All methods accuracy", 
        space=1, 
        xlab="Method", 
        ylab="Accuracy",
        col = rainbow(18),
        border = NA, 
        names.arg=c("KNN","KNN2","Naive Bayes","C4.5","Random Forest"),
        ann=FALSE,
        axes=FALSE
        )
axis(2,at=seq(0,max(accuracies),0.05))



data <- structure(list(U = c(KNNConfusion[["table"]][[1,1]],
                             KNNConfusion[["table"]][[1,2]],
                             KNNConfusion[["table"]][[2,1]],
                             KNNConfusion[["table"]][[2,2]]),
                       W = c(KNN2Confusion[["table"]][[1,1]],
                            KNN2Confusion[["table"]][[1,2]],
                            KNN2Confusion[["table"]][[2,1]],
                            KNN2Confusion[["table"]][[2,2]]),
                       X = c(bayesConfusion[["table"]][[1,1]],
                             bayesConfusion[["table"]][[1,2]],
                             bayesConfusion[["table"]][[2,1]],
                             bayesConfusion[["table"]][[2,2]]),
                       Y = c(C45Confusion[["table"]][[1,1]],
                             C45Confusion[["table"]][[1,2]],
                             C45Confusion[["table"]][[2,1]],
                             C45Confusion[["table"]][[2,2]]), 
                       Z = c(forestConfusion[["table"]][[1,1]],
                             forestConfusion[["table"]][[1,2]],
                             forestConfusion[["table"]][[2,1]],
                             forestConfusion[["table"]][[2,2]])),
                  .Names = c("KNN","KNN2","Naive Bayes","C4.5","Random Forest"), 
                  class = "data.frame", 
                  row.names = c(NA, -4L))
attach(data)
print(data)

colours <- c("firebrick", "indianred1", "darkolivegreen3", "forestgreen")
barplot(as.matrix(data), 
        main="Confusion Matrix values", 
        ylab = "Amount", 
        cex.lab = 1.5, 
        cex.main = 1.4, 
        beside=TRUE, 
        col=colours,
        border=NA,
        ann=FALSE,
        axes=FALSE)
axis(2,at=seq(0,max(data),50))
legend("topright", 
       c("TN","FN","FP","TP"), 
       cex=0.8, 
       bty="n", 
       fill=colours)



# K-means #
telco.scale <-  scale(telco.normalized[,-20], center=TRUE)
telco.pca <- prcomp(telco.scale)
telco.final <- predict(telco.pca)[,1:19]

set.seed(76964057) #Set the seed for reproducibility
k <-kmeans(telco.final, centers=3) 
k$centers 
table(k$cluster)

(cl <- kmeans(telco.final, 3))
plot(telco.final, col = cl$cluster)
points(cl$centers, col = 1:19, pch = 8, cex = 3)


# Add payment categories #
rules <- correctionRules("rules3.txt")
corrected <- correctWithRules(rules, telco.normalized)
telco.correctedAssoc <- corrected$corrected
telco.correctedAssoc$X <- NULL
write.csv(telco.correctedAssoc, file = "telco-normalized-with-categories.csv")


telco.correctedAssoc[] <- lapply(telco.correctedAssoc, factor)
assocRules <- apriori(telco.correctedAssoc)

assocRules <- apriori(telco.correctedAssoc, 
                      parameter = list(minlen=2, supp=0.2, conf=0.25), 
                      appearance = list(rhs=c("Churn=1"), 
                                        default="lhs"), 
                      control = list(verbose=F))
assocRules.sorted <- sort(assocRules, by="lift")
inspect(assocRules.sorted)

subset.matrix <- is.subset(assocRules.sorted, assocRules.sorted)
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1

# remove redundant rules
assocRules.pruned <- assocRules.sorted[!redundant]
inspect(assocRules.pruned)