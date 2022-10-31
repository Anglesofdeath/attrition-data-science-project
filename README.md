### AIAP12 Assessment 2 Submission   
Name: Joshua Ashwin Thomas;  Email: joshua_thomas@live.com   

##### Overview of folders      
&emsp; 1. **.github folder** (provided by AIAP)   
&emsp; 2. **src folder**    
&emsp;&emsp; - **Conf folder** (contains config.yaml file)   
&emsp;&emsp;&emsp; - **config.yaml** (contains the configurable settings for the machine learning pipeline)    
&emsp;&emsp; - __init__.py    
&emsp;&emsp; - **cleaning_engineering_functions.py** (this folder contains the various functions that are used during DataCleaning and feature Engineering)   
&emsp;&emsp; - **data_processing.py** (this file contains 2 classes. The first one: DataProcessing is where the db is downloaded from the link provided. The second one: DataCleaning is where the data is cleaned and new features are engineered based on EDA)   
&emsp;&emsp; - **model.py** (this file contains the functions where our 3 Ml models are defined)   
&emsp; 3. **eda.ipynb** (this file contains the EDA done on the dataset)   
&emsp; 4. **main.py** (this file is the main file which executes all the modules above. Is run by the run.sh file)   
&emsp; 5. **README.md** (this file)    
&emsp; 6. **requirements.txt** list of python modules required    
&emsp; 7. **run.sh** executable file   

##### Instructions for executing pipeline: 
The pipeline can be executed by either running the executable **run.sh** file or by just running the **main.py** file directly. I have set up the pipeline such that certain parameters are able to be edited within the **config.yaml** file. We are able to:    
&emsp; 1. Choose which algorithms to run:   
&emsp;&emsp; Choice of 3 algorithms:    
&emsp;&emsp;&emsp; - Random Forest Classifier   
&emsp;&emsp;&emsp; - Logistic Regression   
&emsp;&emsp;&emsp; - KNN classifier     
&emsp;&emsp; Can either choose to run 1/2/all 3 algorithms within the pipeline. The selection of algorithm to run in the pipeline is done by setting the ['Run'] parameter under each algorithm name to ['True']. If ['Run] == ['True], the pipeline will run the algorithm. If ['Run'] == ['False'] the pipeline will skip the algorithm.     
&emsp; 2. Choose whether to run Grid Search Cross Validation (Grid CV) when running the algorithm. If the ['GridCV'] parameter under each algorithm is set to ['True'], a Grid Search CV will be run to find the best hyper-parameters for the algorithm. If ['GridCV'] == ['False'], it will skip the Grid Search and run the algorithm using the hyperparameters already listed in the config file. It is by default set to ['False'] as a 2-fold Grid Search CV was already run for the hyper-parameters listed within the config file and the ideal hyper-parameters found from that run are already being used as the input. 

##### Description of logical steps/flow of Pipeline  
The pipeline works by executing the run.sh file. This will then run the main.py file. This will then start the running of the pipeline.   
&emsp; 1. Download the data from the url provided in the assessment doc provided. If db already exists then delete db and download and extract out all columns from the table.   
&emsp; 2. Clean the data as was described in the EDA and create new feature: Client Age, Weekly Hours   
&emsp; 3. The main.py file will then run all 3 functions that run the different ML algorithms. If the ['Run'] field in the Algorithm's class in the config file is set to False it will skip the running of the algorithm. If the ['Run'] == ['True'] and ['GridCV'] == ['False'] it will run the algorithm with the predefined hyperparameters that were chosen after performing the Grid Search CV shown in **model.py**. If ['GridCV'] == ['True'], It will run a Grid Search Cross Validation and then run the algorithm using the hyper-parameters that resulted in the highest recall (evaluation metric chosen). It will then log the results of the recall, accuracy and f1 test so that the results can be seen by the user. Accuracy and f1 are given for user info but evaluation metric used is recall. 
##### Evaluation Metrics used
The goal is to help the country club reduce attrition. Therefore, the model's goal is to be able to predict based on the given features which members are likely to Attrition: Yes (1). Therefore, the cost associated with a false negative (so predicting that a member is likely to stay but the member leaves) is high. Therefore, the metric of choice is recall. This is because, so long we minimize false negatives, the country club can focus on retaining the members the model predicts as going to leave (Attrition = 1) without having to worry that they might lose members from the No output (Attrition = 0). The recall scores for the models were:

&emsp; Random Forest: 0.93   
&emsp; Logistic Regression: 1.0  
&emsp; KNN: 0.87

With accuracy scores of: 

&emsp; Random Forest: 80.8%   
&emsp; Logistic Regression: 83.6%  
&emsp; KNN: 75.7%

Based on the fact that recall scores are the same for both the Random Forest and Logistic Regression classifier, my choice would the Logistic Regression classifier. This is because, I would prefer the simpler classifier in this case as its likely to generalize better. 
##### Reason for choice of models: 
This is a binary classification so one of the best models to use for that is logistic regression. 
I chose Random Forest classifier because interpretability of the results is not a big deal as we are simply trying to predict if a customer is a no-show or show. Hence, its useful in this scenario.    
I chose KNN because it is useful for classifying in this scenario as there really isn't a lot of correlation between a lot of the features and the target (shown in EDA). Therefore, i thought KNN might be useful in this scenario. 

##### Key findings from EDA: 
The main takeaway is that none of the features on their own is well correlated with the target. We also found out that of the features chosen for the modelling there was minimal correlation between them (can be seen in EDA). 

##### Improvements that can be made
1. One of the biggest improvements I would like to make would be to implement a K-fold cross validation for the dataset when choosing our hyperparameters (on top of the Grid CV). We only have around 2000+ datapoints which is not a lot for training a ML model considering we are using 8 features. 
2. I would also have preferred considering the size of the dataset, to further reduce the number of features used. However, I am unsure what to take away. 
