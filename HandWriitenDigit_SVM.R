############################ SVM handwritten digit recognition #####################################
#Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other 
#digital devices. The goal is to develop a model that can correctly identify the digit (between 0-9) 
#written in an image.

####################################################################################################

#Installing Neccessary packages
#install.packages("caret")
#install.packages("kernlab")
#install.packages("dplyr")
#install.packages("readr")
#install.packages("ggplot2")
#install.packages("gridExtra")
#install.packages("e1071")
#install.packages("pbkrtest")
#install.packages("DEoptimR",dependencies = T)

#Loading Neccessary libraries
library(dplyr)
library(ggplot2)
library(gridExtra)
library(kernlab)
library(readr)
library(caret)
library(e1071)
library(caTools)


#Loading Data
trainRaw <- read.csv("mnist_train.csv", header=FALSE)
testRaw <- read.csv("mnist_test.csv", header=FALSE)

#Understanding Dimensions
dim(trainRaw)  #60000 rows #785 columns
dim(testRaw)   #10000 rows #785 columns

#Structure of the dataset
str(trainRaw)
str(testRaw)

#printing first few rows
head(trainRaw)
head(testRaw)

#Exploring the data
summary(trainRaw)
summary(testRaw)

#checking missing value
#no NA values detected
sapply(trainRaw, function(x) sum(is.na(x)))
sapply(testRaw, function(x) sum(is.na(x)))

#check for unique entries, all rows unique
nrow(unique(trainRaw))
nrow(unique(testRaw))

#Changing target name V1 to digit
colnames(trainRaw)[colnames(trainRaw)=="V1"] <- "digit"
colnames(testRaw)[colnames(testRaw)=="V1"] <- "digit"

#Making our target class to factor 
trainRaw$digit<-factor(trainRaw$digit)
testRaw$digit<-factor(testRaw$digit)

#look at some digits
digit <- matrix(as.numeric(trainRaw[10,-1]), nrow = 28) 
image(digit, col = grey.colors(255))

digit <- matrix(as.numeric(trainRaw[13,-1]), nrow = 28) 
image(digit, col = grey.colors(255))

# for constant and repeatable results
set.seed(100)
train.indices = sample(1:nrow(trainRaw), 0.3*nrow(trainRaw))
train_sample = trainRaw[train.indices,]
dim(train_sample) #18000 rows #785 columns

#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(digit~ ., data = train_sample, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, testRaw)


#confusion matrix - Linear Kernel
confusion_linear <- confusionMatrix(Eval_linear,testRaw$digit)
acc_linear <- confusion_linear$overall[1]  #0.913
sens_linear <- confusion_linear$byClass[1] #0.970
spec_linear <- confusion_linear$byClass[2] #0.983
acc_linear
sens_linear
spec_linear

#Using RBF Kernel
Model_RBF <- ksvm(train_sample$digit~ ., data = train_sample, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, testRaw)

#confusion matrix - RBF Kernel
confusion_rbf <- confusionMatrix(Eval_RBF,testRaw$digit)
acc_RBF <- confusion_rbf$overall[1]  #0.965
sens_RBF <- confusion_rbf$byClass[1] #0.988
spec_RBF <- confusion_rbf$byClass[2] #0.990
acc_RBF
sens_RBF
spec_RBF


############   Hyperparameter tuning and Cross Validation-Non Linear #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=3)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
set.seed(7)
grid <- expand.grid(.sigma=c(1.63851698339032e-07, 4.63851698339032e-07), .C=c(1,2) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(digit~., data=train_sample, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)
plot(fit.svm)


# Validating the model results on test data
evaluate_non_linear<- predict(fit.svm, testRaw)
#confusion matrix
confusion_NL <- confusionMatrix(evaluate_non_linear, testRaw$digit)
acc_NL <- confusion_NL$overall[1]  
sens_NL <- confusion_NL$byClass[1] 
spec_NL <- confusion_NL$byClass[2] 
acc_NL
sens_NL
spec_NL


#####################################################################
# Hyperparameter tuning and Cross Validation  - Linear - SVM 
######################################################################

# We will use the train function from caret package to perform crossvalidation

trainControl <- trainControl(method="cv", number=5)
# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"

set.seed(100)

# making a grid of C values. 
grid <- expand.grid(C=seq(1, 5, by=1))

# Performing 5-fold cross validation
fit.svm_linear <- train(digit~., data=train_sample, method="svmLinear", metric=metric, 
                        tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm_linear)
#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was C = 1.

# Plotting "fit.svm_linear" results
plot(fit.svm_linear)

###############################################################################

# Valdiating the model after cross validation on test data

evaluate_linear_test<- predict(fit.svm_linear, testRaw)
confusionMatrix(evaluate_linear_test, testRaw$digit)

#confusion matrix
confusion_L <- confusionMatrix(evaluate_linear_test, testRaw$digit)
acc_L <- confusion_L$overall[1]  #0.913
sens_L <- confusion_L$byClass[1] #0.970
spec_L <- confusion_L$byClass[2] #0.983
acc_L
sens_L
spec_L
