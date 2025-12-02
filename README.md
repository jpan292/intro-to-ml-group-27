# intro-to-ml-group-27
Problem statement:
Social media bots operated by bad actors have caused damage to society and need to be stopped from tainting social media for humans.

Motivation:
Social media bots operated by bad actors spread disinformation surrounding health and climate, spread conspiracy theories, sway public opinion by inflating likes and posts, and even organize potentially dangerous protests creating civil tension. 
Creating an AI that detects bot accounts can reduce the power of bad actors on social media, and prevent the damage that they can do.

Solution: 
Utilize Kaggle's VK.com dataset for bot detection to train the best ML classifier to detect bot accounts on new data.

All three models are trained on 70% of the original dataset. The models are going to be compared by their results from the validation set (15% of original dataset). The choosen model will be documented in our final report with performance metrics on the test dataset (15% of original dataset). 

Data_module file contains three function definitions to preprocess the data and ensure all models being evaluated see the same datasets.

