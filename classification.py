import xgboost as xgb
import preprocessing as pr_kaggle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing as pre
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def check_classifier(Xtrain, ytrain, Xtest, ytest, clf):
    clf.fit(Xtrain, ytrain)
    print clf.score(Xtrain, ytrain)
    print clf.score(Xtest, ytest)

def check_xgb_classifier(Xtrain, ytrain, Xtest, ytest, clf):
    clf.fit(Xtrain, ytrain,
       eval_metric = "auc",
            #early_stopping_rounds = 150,
            #eval_set=((Xtrain, ytrain), (Xvalidation, yvalidation), ),
            #verbose = False
           )
    print clf.score(Xtrain, ytrain)
    print clf.score(Xtest, ytest)

def test_gcb(Xy = None, n_estimators = 100, max_depth = 10, test_size = 0.1):
    if Xy is None:
        X, y = pr_kaggle.load_data()
    else:
        X, y = Xy
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
        test_size=test_size, random_state=36)
    dc = lambda: GBC(learning_rate = 0.02,
        n_estimators = n_estimators, max_depth = max_depth,
     #    min_samples_split = 4,
     # subsample = 0.8, max_features = 0.66
    )
    clf = dc()
    check_classifier(Xtrain, ytrain, Xtest, ytest,clf )

    clf = dc()
    clfbag = BaggingClassifier(clf, n_estimators=5)
    check_classifier(Xtrain, ytrain, Xtest, ytest,clfbag )

    clf = dc()
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    check_classifier(Xtrain, ytrain, Xtest, ytest,clf_isotonic )

def test_xgb(Xy = None, n_estimators = 100, max_depth = 10, test_size = 0.1):
    if Xy is None:
        X, y = pr_kaggle.load_data()
    else:
        X, y = Xy
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
        test_size=test_size, random_state=36)
    dc = lambda: xgb.XGBClassifier(n_estimators=n_estimators,
        learning_rate=0.02, max_depth = max_depth, subsample =0.85,
        colsample_bytree = 0.66)
    clf = dc()
    check_xgb_classifier(Xtrain, ytrain, Xtest, ytest,clf )
#
    clf = dc()
    clfbag = BaggingClassifier(clf, n_estimators=5)
    check_classifier(Xtrain, ytrain, Xtest, ytest,clfbag )

    clf = dc()
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    check_classifier(Xtrain, ytrain, Xtest, ytest,clf_isotonic )

def xgb_for_submission(n_estimators = 100, max_depth = 10, cat2vectors = False):
  X, y, Xtest, test_df = pr_kaggle.load_data(load_test = True, cat2vectors = True)
  dc = lambda: xgb.XGBClassifier(n_estimators=n_estimators,
                                 learning_rate=0.02, max_depth = max_depth,
                                 subsample =0.85, colsample_bytree = 0.66)
  clf = dc()
  clf.fit(X, y,
       eval_metric = "auc",
            #early_stopping_rounds = 150,
            #eval_set=((Xtrain, ytrain), (Xvalidation, yvalidation), ),
            #verbose = False
           )
  print clf.score(X, y)
  t = clf.predict_proba(Xtest)
  print t.shape
  score = test_df['QuoteConversion_Flag'] = t[:,1]
  test_df[['QuoteNumber', 'QuoteConversion_Flag']].to_csv('submission_xgb_%d_%d.csv' %(n_estimators, max_depth), index=False)
"""
Original R code:
# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

library(readr)
library(xgboost)

#my favorite seed^^
set.seed(88888)

cat("reading the train and test data\n")
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)


# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Original_Quote_Date, "%m"))
train$year <- as.integer(format(train$Original_Quote_Date, "%y"))
train$day <- weekdays(as.Date(train$Original_Quote_Date))

# removing the date column
train <- train[,-c(2)]

# seperating out the elements of the date column for the train set
test$month <- as.integer(format(test$Original_Quote_Date, "%m"))
test$year <- as.integer(format(test$Original_Quote_Date, "%y"))
test$day <- weekdays(as.Date(test$Original_Quote_Date))

# removing the date column
test <- test[,-c(2)]


feature.names <- names(train)[c(3:301)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)

set.seed(9)
train <- train[sample(nrow(train), 5000),]
gc()

tra<-train[,feature.names]
tra<-tra[,c(1:50,65:105,110:165,180:230,245:290)]
dim(tra)
dim(test)

nrow(train)
h<-sample(nrow(train),2000)



dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
#dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$QuoteConversion_Flag[-h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[,]),label=train$QuoteConversion_Flag)

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "binary:logistic",
                booster = "gbtree",
                eval_metric = "auc",
                eta                 = 0.02, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.85, # 0.7
                colsample_bytree    = 0.66 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001,
                # lambda = 1
)

clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = 30, #1500,
                    verbose             = 0,  #1
                    #early.stop.round    = 150,
                    #watchlist           = watchlist,
                    maximize            = FALSE
)

test1<-test[,feature.names]
dim(test)
test1<-test1[,c(1:50,65:105,110:165,180:230,245:290)]

pred1 <- predict(clf, data.matrix(test1))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb_Shize_stop_1500_8.csv")
"""
