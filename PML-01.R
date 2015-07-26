# PML-01.R
# Paolo Brunasti
library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)


# Defining file names
file_dest_training <- "pml-training.csv"
file_dest_testing <- "pml-testing.csv"

# Import the data treating empty values as NA.
df_training <- read.csv(file_dest_training, na.strings=c("NA",""), header=TRUE)
colnames_train <- colnames(df_training)

df_testing <- read.csv(file_dest_testing, na.strings=c("NA",""), header=TRUE)
colnames_test <- colnames(df_testing)

# Verify that the column names (excluding classe and problem_id) are identical in the training and test set.
all.equal(colnames_train[1:length(colnames_train)-1], colnames_test[1:length(colnames_train)-1])


print("Loaded data: Done")



# Defining utility functions
# Count the number of non-NAs in each given col.
nonNAs <- function(x) {
  as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}

# Cleaning data
# Build vector of missing data or columns to drop.
colcnts <- nonNAs(df_training)
drops <- c()
for (cnt in 1:length(colcnts)) {
  if (colcnts[cnt] < nrow(df_training)) {
    drops <- c(drops, colnames_train[cnt])
  }
}

# Creating the training and test dataframe


# Drop NA data and the first 7 columns that are not needed for the experiment.
df_training <- df_training[,!(names(df_training) %in% drops)]
df_training <- df_training[,8:length(colnames(df_training))]

df_testing <- df_testing[,!(names(df_testing) %in% drops)]
df_testing <- df_testing[,8:length(colnames(df_testing))]

# Show remaining columns.
print("Training Columns")
print(colnames(df_training))

print("Testing Columns")
print(colnames(df_testing))




print("Check for covariates that have pratically no variablility:")
nsv <- nearZeroVar(df_training, saveMetrics=TRUE)
print(nsv)




# Reduce the given training set into a smaller one.
set.seed(123)
ids_small <- createDataPartition(y=df_training$classe, p=0.25, list=FALSE)
df_small1 <- df_training[ids_small,]

# Divide the small set into training (70%) and test (30%) sets.
inTrain <- createDataPartition(y=df_small1$classe, p=0.7, list=FALSE)
df_small_training <- df_small1[inTrain,]
df_small_testing <- df_small1[-inTrain,]


print("Training: ")

# Train on training set with both preprocessing and cross validation.
modFit <- train(df_small_training$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training)
print(modFit, digits=3)


# Run against small testing set.
predictions <- predict(modFit, newdata=df_small_testing)
print(confusionMatrix(predictions, df_small_testing$classe), digits=4)
#predictions <- predict(modFit, newdata=df_testing)
#print(confusionMatrix(predictions, df_testing$classe), digits=4)


# Run against 20 testing set provided
print("Predict: ")
print(predict(modFit, newdata=df_testing))




