# QRT (2020)

This project is part of École Centrale de Lyon Machine Learning course shared with École Normale Supérieure de Lyon, and was done using Qube Research and Technologies (QRT) data within the ChallengeData national ML competition (<a href="https://challengedata.ens.fr/">see here</a>). 

## Problem 
Use information on illiquid assets to predict performances on liquid assets. Projection from a `d` dimensional space (`d` predictors on illiquid assets) to a `q` dimensional space (`q` performance measures on liquid assets) at each time t. Note that for storage constraints, the uploaded datasets are subsets of the original datasets. The aim is therefore to develop an accurate regression model which may be either seen as one single multi-output regression problem or `k` (the number of liquid assets) smaller single-output regression problems.

## Methodology 
The benchmark used by QRT relies on simples correlation between each liquid and illiquid assets: for each liquid asset, we use the train set to find the most correlated illiquid asset and predict the performance of the liquid asset as being equal to the performance of the illiquid asset. The multi-output regression performs slightly better but requires more computational power. Finally, dividing the problem in 100 smaller problems, we train a ensemble learning methods (Stacking, Voting) using simple base models (boosted and bagged trees, naive Bayes).  

## Results
The benchmark has a 69% weighted accuracy on test set; the multi-ouput regressor performs slightly better with 71% weighted accuracy; the final (divided) model performs even better with more than 73% weighted accuracy on test set. Ranked first (and 20/20 obtained) among all École Centrale de Lyon ML projects. 
