# lets-make-concrete

Lets make concrete is a great intro into machine learning. It was my first project to test a real world application of ussing a regressive model to sole a real world problem.

Using ensemble techniques this Machine Learning project utilises scikitlearns RandomForestRegressor to learn what composition of concrete is likely to
have a high compressive strength value. 

The idea is that a user can than try untested concrete compositions first. Results that predict a higher compressive strength 

Users can customise the training of the model. Or use it for inference providing they run the training locally first (5-10 minutes with default settings)


The dataset is included in this repo. It was orginally taken from Kaggle  (https://www.kaggle.com/c/dat300-2018-concrete)

There are 2 main parts to this project

ConcreteTreeRegression.py
This is the script you should run if you wish to start training the model.

models.py
This is  where model parameters can be set. If you wish to train the model with different settings to achieve a better result you can here.


