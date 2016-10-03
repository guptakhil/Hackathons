#Approach

First of all, the categorical and continuous features were identified, to be treated differently.<br>
It was a <b>Regression</b> problem, where given the details of parks, dates and other environment variables, it is required to predict the Footfall for future.<br>
As the 'date' was given in 'DD-MM-YYYY' format, each entity i.e. date, month and year were extracted for proper time-series analysis.
###Missing Value Imputation
Train and test data contained missing values for 90% of continuous features, and they weren't that small to be neglected.<br>So, missing value imputation became very important.
By printing values of features on each date for different parks in a particular location, it was observed that parks in any location are dependent on each other.<br>
Initially, I tried this for few features, but the trend was found to be common.
###Noise Removal
Next, I used visualisations like Boxplot and Scatter plot to identify noise or outliers present in both train and test, and got rid of them by scaling each feature appropriately.
###Feature Engineering
As the features were distributed differently i.e. some ranged from 0 to 2000 and some from 0 to 300, it was important to bring them to a common level. For the same, I performed following feature transformations:<br>
* DOW_Bin -> Direction_Of_Wind/60
* Average_Mois_Bin -> (Average_Moisture_In_Park-100)/20
* Min_Mois_Bin -> (Min_Moisture_In_Park-50)/25
* Max_Mois_Bin -> (Max_Moisture_In_Park-150)/15
* Aver_Brez_Speed -> Average_Breeze_Speed/30
* Min_Brez_Speed -> Min_Breeze_Speed/30
* Max_Brez_Speed -> Max_Breeze_Speed/30
* Min_Ambi_Poll -> Min_Ambient_Pollution/40
* Max_Ambi_Poll -> Max_Ambient_Pollution/40
* Var1_Bin -> Var1/20

As far as categorical features are concerned, they were binned using the median of Footfall (using Boxplot):

[![Screen Shot 2016-10-03 at 5.45.48 PM.png](https://s15.postimg.org/7kggipnff/Screen_Shot_2016_10_03_at_5_45_48_PM.png)](https://postimg.org/image/u95nia4t3/)

It can be clearly seen that footfall for months '7' and '8' are almost similar, and for months '1', '2' and '12' are similar.<br>
Hence, they were grouped. Dates and parks were also grouped using the same strategy. <br>
It gave me a jump of 5 points on the public LB.

###Model Implementation
As it was a regression problem with many variables of contrasting nature, I tried RandomForestRegressor first. But, it wasn't performing as good as GradientBoostingRegressor.<br>
Due to lack of time, I couldn't try XGBRegressor which would have surely increased the LB score. <br>
One of the important thing in any ML problem is "Cross-validation" to avoid overfitting. For a better validation, I performed 5-fold CV and also validated the model taking 2000-01 as test data.<br>
It gave me a better idea of how my model will perform on the private leaderboard.<br>
Park_ID 19 was dropped from the model because it wasn't present in the test data, and there was no point in feeding noise to the model.

###Result
#####Public LB Score: 105.82 (Rank 4)<br>
#####Private LB Score: 92.70 (Rank 4)

