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

![RPlot.png](https://github.com/brunasti/PracticalMachineLearning-01/blob/master/Rplot.png "Rplot.png")

And this is the model fit:

```   
CART 

3437 samples
  52 predictor
   5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 

Summary of sample sizes: 3437, 3437, 3437, 3437, 3437, 3437, ... 

Resampling results across tuning parameters:

  cp      Accuracy  Kappa   Accuracy SD  Kappa SD
  0.0297  0.541     0.4123  0.0503       0.0801  
  0.0482  0.469     0.3027  0.0651       0.1131  
  0.1142  0.320     0.0517  0.0377       0.0597  

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was cp = 0.0297. 
n= 3437 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

 1) root 3437 2460 A (0.28 0.19 0.17 0.16 0.18)  
   2) roll_belt< 130 3154 2180 A (0.31 0.21 0.19 0.18 0.11)  
     4) pitch_forearm< -34.2 272    1 A (1 0.0037 0 0 0) *
     5) pitch_forearm>=-34.2 2882 2180 A (0.24 0.23 0.21 0.2 0.12)  
      10) magnet_dumbbell_y< 424 2383 1700 A (0.29 0.17 0.24 0.19 0.11)  
        20) roll_forearm< 120 1438  831 A (0.42 0.16 0.2 0.16 0.06) *
        21) roll_forearm>=120 945  650 C (0.078 0.18 0.31 0.25 0.19) *
      11) magnet_dumbbell_y>=424 499  238 B (0.048 0.52 0.042 0.21 0.18) *
   3) roll_belt>=130 283    1 E (0.0035 0 0 0 1) *
```


I received these predictions by appling the chosen model for the 20 items of the final training set:

B A A A A E D D A A B C B A E E A B A B