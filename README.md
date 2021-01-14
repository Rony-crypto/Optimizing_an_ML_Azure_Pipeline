# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains information about the marketing campaign of a Portuguese Banking Institute. The classification goal was to predict whether a client will subscribe to a term deposit or not. The target column is named as 'y'.
Dataset source: "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

The best perfroming model was "StackEnsemble" which gave an accuracy of 91.74%. The model was found using the AutoML technique provided by Azure.


## Scikit-learn Pipeline
**Following steps were majorly caried out to create the Scikit-learn Pipeline:**
1. The dataset is loaded using TabularDatasetFactory.
2. The raw data obtained is thus preprocessed in training ('train.py') script.
   Some of the processes performed for the same are:
   - Dropping null values
   - Using one-hot-encoding on some features
   - Separating the target column ('y')
3. The preprocessed data was then split in 10-90 ratio using test_train_split() of SKLearn
4. The training data thus obtained was then fit into LogisticRegression model.
5. Hyperdrive is used to tune the two hyperparameters namely 'c' and 'number_of_operations'.

**Hyperparameter tuning**
The Hyperdrive used for hyperparameter tuning after creating the workspace and defining the cluster. This can be seen in 'udacity-project.ipynb'. The optimized results thus obtained for the hyperparameters are:
1. Inverse regularization parameter (c) - 72.0066014
2. No of iterations: 60

**Benefits of the parameter sampler**
RandomParameterSampler is choosen for this purpose as it allows us to randomly select hyperparameters in a searchspace, whether discrete or continuous.

**Benefits of the early stopping policy**
Bandit Policy is choosen as the early stopping policy. For every given number of iterations, it checks if the accuracy lies outside given slack-factor percentage. If it does, it terminates the job. Thus saving a lot of time and resources.


## AutoML
Automated Machine Learning is the process of automating the time-consuming, iterative tasks of ML model development. It allows to build the models with high scale efficiency & productivity all while sustaining the model quality. In case of classification problem, many models such as XGBoost, RandomForest, StackEnsemble, VotingEnsemble etc. are compared.
The dataset had to be concatenated with previously removed target labels for setting into automl config.
The number of cross-validations was set to 5.

The best performing model thus obtained was "LightGBMClassifier" with an accuracy of 91.74%. And some of the parameters optimized are:
1. boosting_type='gbdt'
2. learning_rate=0.1
3. max_depth=-1
4. min_child_samples=20,
5. min_child_weight=0.001,
    ...and many more.

## Pipeline comparison
The accuracies obtained by the hyperdrive and automl pipelines are 90% and 91.74% respectively. 
As expected, AutoML performed better as it compared 100s of models as well as hyperparameters to get the best model, on the other hand model was fixed as Logistic Regresssion in case of HyperDrive and thus only parameters wer tuned.

But another point to be noted is that HyperDrive run took less than a minutes to complete while AutoML run took approx 30 minutes, using the same Standard_DS3_V2 cluster. Thus we can say that HyperDRive pipeline provided pretty much good results in a very low span of time as compared to AutoML run.

Also, Automl is highly efficient where as hyperdrive is resource intensive. AutoML can automtically detect and work itself with automatic

## Future work
Another appraoch, to use automl first to get the best model and then use hyperdrive to tune the hyperparameters on that model can be obtained and the results can be compared to the individual accuracies. This might lead to increase in accuracy of the model, but I'm not sure. Also 
Another thing that I would like to dig deep into is n_cross_validations used in AutoML and the Early Stopping Policy by changing the parameters and then comparing parameters, as I'm not much familiar with these.
