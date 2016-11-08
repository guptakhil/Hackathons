#Code for YDS'16
#Akhil Gupta

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from random import uniform
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import operator

#Loading Data
train = pd.read_csv('train 2.csv') #Provide the path to train file
test = pd.read_csv('test.csv') #Provide the path to test file

#Categorical features dropped because same for all values in both train and test
#77 and 127 - Missing values
#72 - Highly correlated with 71
feat_cat_drop = [134,138,140,141,142,143,145,146,148,150,161,162,183,184,185,77,127,72]
for x in feat_cat_drop:
    feat = 'feature'+str(x)
    train = train.drop(feat,1)
    test = test.drop(feat,1)

#Infinity Treatment
df = [train,test]
for d in df:
    for index,row in d.iterrows():
        x = row['feature103']
        if(x==float('+inf')):
            y = 50.0 #Higher than maximum
            d.set_value(index,'feature103',y)

#cat - Categorical Variables
#con - Continuous Variables
cate = train.select_dtypes(include=['int64']).columns
cont = train.select_dtypes(include=['float64']).columns
cat = []
total = []
for i in cate:
    if(i=='attrite'):
        continue
    elif(i=='agentID'):
        continue
    else:
        cat.append(i.encode('ascii','ignore'))
con = []
for i in cont:
    con.append(i.encode('ascii','ignore'))

#Standardization of continuous variables
df = [train,test]
for i in con:
    for d in df:
        stdscaler = StandardScaler()
        d[i] = stdscaler.fit_transform(d[i])

#Imputing missing values by using random number generation
df = [train,test]
for d in df:
    for index,row in d.iterrows():
        u = row['feature69']
        if(u=='H'):
            y = 0
        elif(u=='L'):
            y = 1
        elif(u=='M'):
            y = 2
        else:
            s = uniform(0,1) #Random number generation
            if(s<=0.5):
                y = 0
            elif(s<=0.8):
                y = 2
            else:
                y = 1
        d.set_value(index,'feature69',y)

var = []
for i in con:
    var.append(i)
for i in cat:
    var.append(i)

dtrain = train[var]
dtest = test[var]

#Logistic Regression
lr = LogisticRegression(solver='newton-cg')
lr.fit(dtrain,train['attrite'])
pred = lr.predict_proba(dtest) #Finding probabilities
i = 0
lr_prob = [] #Storing probabilities
lr_label = [] #Storing labels
for tt in pred:
    lr_prob.append(tt[0])
for x in lr_prob:
    if(x<=np.mean(lr_prob)):
        y = 1
    else:
        y = 0
    lr_label.append(y)

#Categorical Binning using Crosstabs in Python
df = [train,test]
feat = 'feature75'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x==3):
            y = 2
        else:
            y = x
        d.set_value(index,feat,y)

df = [train,test]
feat = 'feature87'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x>2):
            y = 2
        else:
            y = x
        d.set_value(index,feat,y)

df = [train,test]
feat = 'feature101'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x==8 or x==9):
            y = 8
        elif(x==10 or x==11):
            y = 9
        else:
            y = x
        d.set_value(index,feat,y)

df = [train,test]
feat = 'feature76'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x>7):
            y = 7
        else:
            y = x
        d.set_value(index,feat,y)

df = [train,test]
feat = 'feature151'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x==0):
            y = 0
        elif(x<=12):
            y = int((x+1)/2)
        elif(x>=13 and x<=17):
            y = 7
        elif(x>=18 and x<=22):
            y = 8
        elif(x>=23 and x<=28):
            y = 9
        elif(x>=29 and x<=36):
            y = 10
        elif(x>=37 and x<=44):
            y = 11
        elif(x>=45 and x<=67):
            y = 12
        elif(x>=68 and x<=90):
            y = 13
        else:
            y = 14
        d.set_value(index,feat,y)

df = [train,test]
feat = 'feature84'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x==0):
            y = 0
        elif(x<=5):
            y = 1
        elif(x==6):
            y = 2
        elif(x>=7 and x<=11):
            y = 3
        elif(x>=12 and x<=17):
            y = 4
        elif(x>=18 and x<=23):
            y = 5
        elif(x>=24 and x<=30):
            y = 6
        elif(x>=31 and x<=40):
            y = 7
        elif(x>=41 and x<=60):
            y = 8
        elif(x>=61 and x<=260):
            y = 9
        else:
            y = 10
        d.set_value(index,feat,y)

df = [train,test]
feat = 'feature147'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x<10):
            y = x
        elif(x>=10 and x<=11):
            y = 10
        elif(x>=12 and x<=14):
            y = 11
        elif(x>=15 and x<=17):
            y = 12
        elif(x>=18 and x<=23):
            y = 13
        elif(x>=24 and x<=32):
            y = 14
        elif(x>=33 and x<=45):
            y = 15
        elif(x>=46 and x<=53):
            y = 16
        elif(x>=54 and x<=64):
            y = 17
        elif(x>=65 and x<=80):
            y = 18
        elif(x>=81 and x<=100):
            y = 19
        else:
            y = 20
        d.set_value(index,feat,y)

#These 3 features have been scaled down because they had huge number of classes
f = ['feature96','feature67','feature94']
df = [train,test]
for fe in f:
    for d in df:
        for index,row in d.iterrows():
            x = row[fe]
            y = x/15.0
            d.set_value(index,feat,y)

df = [train,test]
feat = 'feature71'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x==0):
            y = 0
        elif(x<=51):
            y = int((x-1)/3)+1
        elif(x==52):
            y = 18
        elif(x>=53 and x<=57):
            y = 19
        elif(x>=58 and x<=63):
            y = 20
        elif(x==64):
            y = 21
        elif(x>=65 and x<=69):
            y = 22
        elif(x>=70 and x<=75):
            y = 23
        elif(x==76):
            y = 24
        elif(x>=77 and x<=80):
            y = 25
        elif(x>=81 and x<=87):
            y = 26
        elif(x==88):
            y = 27
        elif(x>=89 and x<=100):
            y = 28
        elif(x>=101 and x<=115):
            y = 29
        elif(x>=116 and x<=122):
            y = 30
        elif(x>=123 and x<=135):
            y = 31
        elif(x>=136 and x<=145):
            y = 32
        elif(x>=146 and x<=151):
            y = 33
        else:
            y = 34
        d.set_value(index,feat,y)

df = [train,test]
feat = 'feature149'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x<=6):
            y = x
        elif(x>=7 and x<=8):
            y = 7
        elif(x>=9 and x<=10):
            y = 8
        elif(x>=11 and x<=13):
            y = 9
        elif(x>=14 and x<=16):
            y = 10
        elif(x>=17 and x<=20):
            y = 11
        elif(x>=21 and x<=25):
            y = 12
        elif(x>=26 and x<=31):
            y = 13
        elif(x>=32 and x<=36):
            y = 14
        elif(x>=37 and x<=42):
            y = 15
        elif(x>=43 and x<=50):
            y = 16
        elif(x>=51 and x<=70):
            y = 17
        elif(x>=71 and x<=90):
            y = 18
        elif(x>=91 and x<=120):
            y = 19
        elif(x>=121 and x<=180):
            y = 20
        else:
            y = 21
        d.set_value(index,feat,y)

df = [train,test]
feat = 'feature73'
for d in df:
    for index,row in d.iterrows():
        x = row[feat]
        if(x==1):
            y = 0
        elif(x==0):
            y = 1
        elif(x>=2 and x<=9):
            y = 1
        elif(x>=10 and x<=14):
            y = 2
        elif(x>=15 and x<=19):
            y = 3
        elif(x>=20 and x<=31):
            y = 4
        elif(x==32):
            y = 5
        elif(x>=33 and x<=59):
            y = 6
        elif(x==60):
            y = 7
        elif(x>=61 and x<=62):
            y = 8
        elif(x>=63 and x<=90):
            y = 9
        elif(x==91):
            y = 10
        elif(x==92):
            y = 11
        elif(x>=93 and x<=120):
            y = 12
        elif(x==121):
            y = 13
        elif(x==122):
            y = 14
        elif(x>=123 and x<=151):
            y = 15
        elif(x==152):
            y = 16
        elif(x==153):
            y = 17
        elif(x>=154 and x<=181):
            y = 18
        elif(x==182):
            y = 19
        elif(x==183):
            y = 20
        elif(x>=184 and x<=212):
            y = 21
        elif(x==213):
            y = 22
        elif(x==214):
            y = 23
        elif(x>=215 and x<=243):
            y = 24
        elif(x==244):
            y = 25
        elif(x==245):
            y = 26
        elif(x>=246 and x<=273):
            y = 27
        elif(x==274):
            y = 28
        elif(x==275):
            y = 29
        elif(x>=276 and x<=304):
            y = 30
        elif(x==305):
            y = 31
        elif(x==306):
            y = 32
        elif(x>=307 and x<=334):
            y = 33
        elif(x==335):
            y = 34
        elif(x==336):
            y = 35
        else:
            y = 36
        d.set_value(index,feat,y)

#XGB (Softmax)
w = range(1,11,1)
test['attrite'] = 0.0
for rann in w:
    #print rann
    us = RandomUnderSampler(ratio = 0.9, random_state=rann,replacement=False) #Undersampling using ratio 0.9
    train_new, train_new_target = us.fit_sample(train[var],train['attrite'])

    X = pd.DataFrame(train_new)
    y = pd.DataFrame(train_new_target)
    X_test = test[var]
    X.columns = var
    params = {"objective": "multi:softmax",
            "num_class":2,
          "booster" : "gbtree",
          "eta": 0.092,
          "max_depth": 3,
          "subsample": 1.0,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301,
          "min_child_weight": 30
          }
    num_boost_round = 200

    dtrain = xgb.DMatrix(X, y)

    gbm = xgb.train(params, dtrain, num_boost_round)

    test['attrite'] += gbm.predict(xgb.DMatrix(X_test))
test['attrite'] = test['attrite']/10.0
xgb_ms_prob = [] #Probabilities for xgb softmax
xgb_ms_label = [] #Labels for xgb softmax
for index,row in test.iterrows():
    y = row['attrite']
    if(y>=0.5):
        x = 1
    else:
        x = 0
    xgb_ms_label.append(x)
    xgb_ms_prob.append(y)

#Execution of model for generation of Feature map
us = RandomUnderSampler(ratio = 0.9, random_state=10,replacement=False)
X_train_res, y_train_res = us.fit_sample(train[var],train['attrite'])

params = {"objective": "multi:softmax",
          'num_class':2,
          "booster" : "gbtree",
          "eta": 0.092,
          "max_depth": 3,
          "subsample": 1,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301,
          "min_child_weight":60
          }
num_boost_round = 200

dtrain = xgb.DMatrix(X_train_res,y_train_res)

gbm = xgb.train(params, dtrain, num_boost_round)
X_test = test[var]

predicted = gbm.predict(xgb.DMatrix(X_test.as_matrix()))

def create_feature_map(features): #Function for creation of feature map
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
create_feature_map(var)
import operator
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore']) #Feature Importance
df['fscore'] = df['fscore'] / df['fscore'].sum()
#print df

varr = [] #Important features
for index,row in df.iterrows():
    varr.append(row['feature'])

#XGB(Logistic) on 125 features
w = range(1,11,1)
test['attrite'] = 0.0
for rann in w:
    #print rann
    us = RandomUnderSampler(ratio = 0.9, random_state=rann,replacement=False) #Undersampling
    train_new, train_new_target = us.fit_sample(train[varr],train['attrite'])

    X = pd.DataFrame(train_new)
    y = pd.DataFrame(train_new_target)
    X_test = test[varr]
    X.columns = varr #125 features
    params = {"objective": "binary:logistic",
          "booster" : "gbtree",
          "eta": 0.092,
          "max_depth": 7,
          "subsample": 1.0,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301,
          "min_child_weight": 30
          }
    num_boost_round = 200

    dtrain = xgb.DMatrix(X, y)

    gbm = xgb.train(params, dtrain, num_boost_round)

    test['attrite'] += gbm.predict(xgb.DMatrix(X_test))
test['attrite'] = test['attrite']/10.0
xgb_log_prob = [] #Probabilities for xgb logistic
xgb_log_label = [] #Labels for xgb logistic
for index,row in test.iterrows():
    y = row['attrite']
    if(y>=0.5):
        x = 1
    else:
        x = 0
    xgb_log_label.append(x)
    xgb_log_prob.append(y)

#Averaging of all three models prepared
final_pred = []
u = 0
for i in lr_label:
    e = i + xgb_log_label[u] + xgb_ms_label[u]
    if(e<=1):
        final_pred.append(0)
    else:
        final_pred.append(1)
    u = u + 1

test['attrite'] = -1
i = 0
for index,row in test.iterrows():
    test.set_value(index,'attrite',final_pred[i])
    i = i + 1

#Generation of Submission file
test.to_csv('submission.csv',columns=('agentID','attrite'),index=False)                                                                                      
