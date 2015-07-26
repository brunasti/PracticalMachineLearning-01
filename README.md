# PracticalMachineLearning-01
Practical Machine Learning - Course Project 01
Paolo Brunasti - 26/7/2015

Given both training and test data from the study :
- Qualitative Activity Recognition of Weight Lifting Exercises
predict the manner in which they did the exercise.

### Index of the present document:
- Introduction
- Question
- Input Data
- Features
- Algorithm
- Conclusion


## INTRODUCTION
The program implementing the algorythm is in the file:
PML-01.R


## QUESTION
Six participants participated in a lifting exercise in five different ways. 
The five ways were:
A) exactly according to the specification
B) throwing the elbows to the front
C) lifting the dumbbell only halfway
D) lowering the dumbbell only halfway
E) throwing the hips to the front

Case A corresponds to the correct execution of the exercise, the other 4 correspond to different kind of mistakes.

By processing data gathered, the question is: can the appropriate activity quality (A-E) be deducted?


## INPUT DATA
The training data are from: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

and saved locally as 

    pml-training.csv

The test data are instead from: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

and saved locally as 

    pml-testing.csv

First I load the appropriate packages and set the seed for reproduceable results.

    library(AppliedPredictiveModeling)
    library(caret)
    library(rattle)
    library(rpart.plot)
    library(randomForest)

If needed import the packages in the R environemnt.

    install.packages("rattle")
    install.packages("rpart.plot"")
    install.packages("randomForest")

Then I need to import the data

Download data.

    file_dest_training <- "pml-training.csv"
    file_dest_testing <- "pml-testing.csv"

Import the data treating empty values as NA.

    df_training <- read.csv(file_dest_training, na.strings=c("NA",""), header=TRUE)
    colnames_train <- colnames(df_training)
    df_testing <- read.csv(file_dest_testing, na.strings=c("NA",""), header=TRUE)
    colnames_test <- colnames(df_testing)

Verify that the column names (excluding classe and problem_id) are identical in the training and test set.

    all.equal(colnames_train[1:length(colnames_train)-1], colnames_test[1:length(colnames_train)-1])



## FEATURES
Having verified that the schema of both the training and testing sets are identical (excluding the final column representing the A-E class), I decided to eliminate all the NA columns and other extraneous columns.

    # Utility function that count the number of non-NAs in each col.
    nonNAs <- function(x) {
        as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
    }

    # Build vector of missing data or NA columns to drop.
    colcnts <- nonNAs(df_training)
    drops <- c()
    for (cnt in 1:length(colcnts)) {
        if (colcnts[cnt] < nrow(df_training)) {
            drops <- c(drops, colnames_train[cnt])
        }
    }

Drop NA data and the first 7 columns as they're unnecessary for predicting.

    df_training <- df_training[,!(names(df_training) %in% drops)]
    df_training <- df_training[,8:length(colnames(df_training))]

    df_testing <- df_testing[,!(names(df_testing) %in% drops)]
    df_testing <- df_testing[,8:length(colnames(df_testing))]

    # Show remaining columns.
    print(colnames(df_training))
    print(colnames(df_testing))

I check for covariates that have virtually no variablility.

    nsv <- nearZeroVar(df_training, saveMetrics=TRUE)
    print(nsv)

Given that all of the near zero variance variables (nsv) are FALSE, there's no need to eliminate any covariates due to lack of variablility.



## ALGORITHM

A large training set was provided, and a small final testing set (20 entries). 
Due to time constrains I reduced the size of the training set. 
Instead of performing the algorithm on the entire training set, I divided it into four smaller sets.

After some test I decided to use both preprocessing and cross validation

Train on training set with both preprocessing and cross validation.

    modFit <- train(df_small_training$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training)
    print(modFit, digits=3)

    # Run against small testing set.
    predictions <- predict(modFit, newdata=df_small_testing)
    print(confusionMatrix(predictions, df_small_testing$classe), digits=4)


    # Reporting for Evaluation
    print(modFit$finalModel, digits=3)
    fancyRpartPlot(modFit$finalModel)


    # Run against 20 testing set provided
    print("Predict: ")
    print(predict(modFit, newdata=df_testing))


## CONCLUSION

This is the diagram of the prediction tree:

![RPlot.png)(https://github.com/brunasti/PracticalMachineLearning-01/blob/master/Rplot.png "Rplot.png")

I received these predictions by appling the chosen model for the 20 items of the final training set:

 B A A A A E D D A A B C B A E E A B B B