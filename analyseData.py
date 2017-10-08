import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.externals import joblib 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV 
import re
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from PseudoLabeler import * 

from getPath import *
pardir = getparentdir()
train_path = pardir+'/data/train.csv'
test_path = pardir+'/data/test.csv'
res_path = pardir+'/data/res.csv'

# def encoder(arr):
    # ohe = OneHotEncoder(sparse=False)#categorical_features='all',
    # ohe.fit(arr)
    # return ohe.transform(arr)
    
def convert_continues(bininterval,arr):
    bins = int((np.max(arr)-np.min(arr))/bininterval)
    label = [a for a in range(bins)]
    out = pd.cut(arr, bins, labels = label)
    out = [o for o in out]
    return out

def onhotcoder(traindata,testdata):
    trainarr = traindata[['Age','Embarked','Cabin','Name','Ticket','Pclass','Sex']]
    testarr = testdata[['Age','Embarked','Cabin','Name','Ticket','Pclass','Sex']]
    combine = pd.concat([trainarr,testarr])
    ages = combine['Age']
    getticketletter(combine['Ticket'])
    na_arr = np.array(ages.isnull())
    no_na_ages = ages[na_arr==False]
    no_na_out = convert_continues(5,no_na_ages)
    ages[na_arr==False] = no_na_out
    age_dummy = pd.get_dummies(ages,dummy_na=True).rename(columns=lambda x: 'age_'+str(x))

    embark = combine["Embarked"].fillna("C")
    embark_dummy = pd.get_dummies(embark).rename(columns=lambda x: 'embark_'+str(x))
    callist = getcallist(combine)
    call_dummy = pd.get_dummies(callist).rename(columns=lambda x: 'call_'+str(x))
    cabindf = getcabininfo(combine['Cabin'])
    cabinnum_dummy = pd.get_dummies(cabindf['num']).rename(columns=lambda x: 'cabinnum_'+str(x))
    cabinletter_dummy = pd.get_dummies(cabindf['letter']).rename(columns=lambda x: 'cabinletter_'+str(x))
    ticketdf = getticketletter(combine['Ticket'])
    ticket_dummy = pd.get_dummies(ticketdf).rename(columns=lambda x: 'ticket_'+str(x))
    pclass_dummy = pd.get_dummies(combine['Pclass']).rename(columns=lambda x: 'pclass_'+str(x))
    sex_dummy =pd.get_dummies(combine['Sex']).rename(columns=lambda x: 'sex_'+str(x))
    
    
    call_dummy.reset_index(inplace=True,drop=True)
    age_dummy.reset_index(inplace=True,drop=True)
    embark_dummy.reset_index(inplace=True,drop=True)
    cabinletter_dummy.reset_index(inplace=True,drop=True)
    cabinnum_dummy.reset_index(inplace=True,drop=True)
    ticket_dummy.reset_index(inplace=True,drop=True)
    pclass_dummy.reset_index(inplace=True,drop=True)
    sex_dummy.reset_index(inplace=True,drop=True)
    combine_dummy = pd.concat([age_dummy,embark_dummy,call_dummy,cabinnum_dummy,cabinletter_dummy,ticket_dummy,pclass_dummy,sex_dummy],axis = 1)
    return combine_dummy[:891],combine_dummy[891:]
    
def getcabinletter(cabin):
    match = re.compile("([a-zA-Z]+)").search(cabin)
    if match:
        return match.group(0)
    else:
        return 'U'
    
def getcabinnum(cabin):
    match = re.compile("([1-9]+)").search(cabin)
    if match:
        return match.group(0)
    else:
        return '0'
  
def getcabininfo(cabinlist):
    cabinlist = cabinlist.fillna('0')
    df = pd.DataFrame()
    df['letter'] = cabinlist.map(lambda x :getcabinletter(str(x)))
    df['num'] = cabinlist.map(lambda x :getcabinnum(str(x)))
    return df
    
def getticketletter(ticketlist):
    ticketlist = ticketlist.fillna('0')
    return ticketlist.map(lambda x :getcabinletter(str(x)))
    
def convertdf(arr):
    dfarr = []
    for a in arr:
        dfarr.append(pd.DataFrame(a))
    return dfarr

def preprocess(data,istest,combine_dummy):
    # print(combine_dummy)
    classout = data['Pclass'].rename("pclass")
    sibout = data['SibSp'].rename("sib")
    parchout = data['Parch'].rename("parch")
    fareout = data['Fare'].fillna(8).rename("fare")
    parchratio= (parchout/(parchout+sibout+1)).rename("parchratio")
    sibratio = (sibout/(parchout+sibout+1)).rename( "sibratio")
    average_fare = (fareout/(parchout+sibout+1)).rename("fare_avg")
    # fare = fare.fillna(0)
    value = (data['Pclass']*(sibout+parchout+1)).rename( "value")
    if not istest:
        # if issvm:
        min_max_scaler = preprocessing.MinMaxScaler() 
        classnorm = min_max_scaler.fit_transform(classout)
        joblib.dump(min_max_scaler,pardir+'/model/class.pkl')
        
        min_max_scaler = preprocessing.MinMaxScaler() 
        sibnorm = min_max_scaler.fit_transform(sibout)
        joblib.dump(min_max_scaler,pardir+'/model/sib.pkl')
        
        min_max_scaler = preprocessing.MinMaxScaler()
        parchnorm = min_max_scaler.fit_transform(parchout)
        joblib.dump(min_max_scaler,pardir+'/model/parch.pkl')
        
        standard_scaler = preprocessing.StandardScaler()
        farenorm = standard_scaler.fit_transform(fareout)
        joblib.dump(standard_scaler,pardir+'/model/fare.pkl')
        normarr = convertdf([classnorm,sibnorm,parchnorm,farenorm])
        
    else:
        # if issvm:
        min_max_scaler = joblib.load(pardir+'/model/class.pkl')
        classnorm = min_max_scaler.transform(classout)
        min_max_scaler = joblib.load(pardir+'/model/sib.pkl')
        sibnorm = min_max_scaler.transform(sibout)
        min_max_scaler = joblib.load(pardir+'/model/parch.pkl')
        parchnorm = min_max_scaler.transform(parchout)
        standard_scaler = joblib.load(pardir+'/model/fare.pkl')
        farenorm = standard_scaler.transform(fareout)
        normarr = convertdf([classnorm,sibnorm,parchnorm,farenorm])
    # sibout = pd.DataFrame(sib)
    # parchout = pd.DataFrame(parch)
    # fareout = pd.DataFrame(fare)
    combine_dummy.reset_index(inplace=True,drop=True)
    features = pd.concat([combine_dummy,sibout,parchout,fareout,parchratio,sibratio,average_fare,value,classout],axis=1)
    svmfeatures = pd.concat([combine_dummy,normarr[1],normarr[2],normarr[3],parchratio,sibratio,average_fare,value,normarr[0]],axis=1)
    return features,svmfeatures
    
def getcallist(data):
    call = data['Name']
    call = list(call)
    callist = [i.split(',')[1].split('.')[0].replace(" ","") for i in call]
    return callist

def analyze_relation():
    data = pd.read_csv(train_path, encoding='utf-8')
    embark = data['Ticket']
    # print(len(embark[embark.isnull()]))
    # print(len(embark))
    # pclass = data['Pclass']
    # fare = data['Fare']
    # plt.scatter(pclass,fare)
    # plt.show()
    
    call = data['Name']
    call = list(call)
    callist = [i.split(',')[1].split('.')[0].replace(" ","") for i in call]
    print(callist)
    
def trainmodel(features,labels):
    sample_leaf_options = [1,5,10,50,100,200]
    finalmodel = 0
    maxscore = 0
    # for leaf_size in sample_leaf_options:
        # model = RandomForestClassifier(n_estimators = 200, oob_score = True,n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = leaf_size)
        # model.fit(features,labels)
        # score = model.oob_score_ 
        # print(score)
        # if maxscore<score:
            # maxscore =score
            # finalmodel = model
            # print(leaf_size)
    xgb1 = XGBClassifier(reg_alpha=1e-5,learning_rate = 0.05,n_estimators=73,max_depth =3, min_child_weight=1,gamma=0.4,subsample=0.8,colsample_bytree=0.8,
    objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
    xgb1.fit(features,labels)
    # finalmodel.fit(features, labels)
    joblib.dump(xgb1,pardir+'/model/rf.pkl')
   
def predict(test_data,features):
    clf = joblib.load(pardir+'/model/rf.pkl')
    labels = clf.predict(features)
    pid = test_data['PassengerId']
    res = pd.DataFrame()
    res['PassengerId'] = pid
    # print(len(pid))
    # print(len(labels))
    res['Survived'] = labels
    res.to_csv(res_path,encoding='utf-8',index=False) 
    
def modelfit(alg, features, labels,useTrainCV=True, cv_folds=5, early_stopping_rounds=50): 
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(features, label=labels)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(features, labels,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(features)
    dtrain_predprob = alg.predict_proba(features)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(labels, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(labels, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    # plt.show()
    
def tuneParam(features, labels):
    xgb_train = xgb.DMatrix(features, label=labels)
    param_test1 = {
     'max_depth':list(range(3,10,2)),
     'min_child_weight':list(range(1,6,2))
    }
    param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[1,2]
}
    param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
    param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
    param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
    xgb1 = XGBClassifier(reg_alpha = 1e-5,learning_rate=0.1,n_estimators=77,gamma = 0,max_depth =5,min_child_weight = 1,seed=27,subsample=0.8,colsample_bytree=0.6,objective= 'binary:logistic')
    # modelfit(xgb1, features, labels)
    # gsearch1 = GridSearchCV(estimator = xgb1,param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    # gsearch1.fit(features, labels)
    # print(gsearch1.grid_scores_)
    # print(gsearch1.best_params_)
    # print(gsearch1.best_score_)

def getfeatureimportance(X,y):
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
    # scores = cross_val_score(clf, X, y)
    clf.fit(X, y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    indexs = X.columns.values
    select = []
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, indexs[indices[f]], importances[indices[f]]))
        if importances[indices[f]]>0:
            select.append(indexs[indices[f]])
    return select
    
    
def validata(valid_features, valid_labels):
    clf = joblib.load(pardir+'/model/rf.pkl')
    labels = clf.predict(valid_features)
    print(accuracy_score(valid_labels,labels))

def total():
    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data = pd.read_csv(test_path, encoding='utf-8')
    train_dummy,test_dummy= onhotcoder(train_data,test_data)
    train_features = preprocess(train_data,False,train_dummy)
    # index = [agelen,embarklen,calllen,pclasslen,sexlen,1,1,1,1,1,1,1,1]
   
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_data['Survived'], test_size=0.2, random_state=42)
    test_features= preprocess(test_data,True,test_dummy)
   
    # print(train_features)
    select = getfeatureimportance(train_features,train_data['Survived'])
    # tuneParam(train_features[select],train_data['Survived'])
    trainmodel(train_features[select],train_data['Survived'])
    # trainmodel(X_train[select],y_train)
    # validata(X_test[select], y_test)
    predict(test_data,test_features[select])
    
def submodel(train_features,labels,test_features,train_indexs,test_indexs, model):
    print(len(train_features))
    print(len(test_features))
    print(len(labels))
    length = len(train_features)
    layer = np.array([1.0]*length)
    for i in range(len(train_indexs)):
        print("first train"+str(i))
        train = train_features[train_indexs[i]]
        label = labels[train_indexs[i]]
        model.fit(train,label)
        res = model.predict(train_features[test_indexs[i]])
        layer[test_indexs[i]] = res
    layer = np.array([[f] for f in layer])
    model.fit(train_features,labels)
    testappend = model.predict(test_features)
    testappend = np.array([[t] for t in testappend])
    return layer,testappend

def stackingmodel(isvalid):
    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data = pd.read_csv(test_path, encoding='utf-8')
    train_dummy,test_dummy= onhotcoder(train_data,test_data)
    train_features,svm_train_features = preprocess(train_data,False,train_dummy)
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_data['Survived'], test_size=0.2, random_state=42)
    test_features,svm_test_features= preprocess(test_data,True,test_dummy)
    if isvalid: 
        select = getfeatureimportance(X_train,y_train)
        trainfeatures = X_train[select]
        labels = y_train
        test_features = X_test[select]
        train_indexs,test_indexs = get_k_fold(trainfeatures)
        
    else:
        select = getfeatureimportance(train_features,train_data['Survived'])
        trainfeatures = train_features[select]
        labels = train_data['Survived']
        test_features = test_features[select]
        train_indexs,test_indexs = get_k_fold(trainfeatures)
    
    model1 = SVC(random_state = 11)
    model2 = RandomForestClassifier(random_state=11)
    model3 = AdaBoostClassifier(random_state=12)
    model4 = ExtraTreesClassifier(random_state = 12)
    model5 = GradientBoostingClassifier(random_state = 12)
    
    models = [model2,model3,model4,model5]
    
    trainfeatures = np.array(trainfeatures)
    test_features = np.array(test_features)
    
    svmtrainfeatures = np.array(svm_train_features)
    svmtestfeatures = np.array(svm_test_features)
    train = np.array(trainfeatures)
    test = np.array(test_features)
    # train = np.array([])
    # test = np.array([])
    labels = np.array(labels)
    for model in models:
        layer,testappend = submodel(trainfeatures,labels,test_features,train_indexs,test_indexs, model)
        train = np.hstack((train,layer))
        test = np.hstack((test,testappend))
    layer,testappend = submodel(svmtrainfeatures,labels,svmtestfeatures,train_indexs,test_indexs, model1)
    train = np.hstack((train,layer))
    test = np.hstack((test,testappend))
    # tuneParam(train, labels)
    # clf = XGBClassifier(learning_rate=0.1,n_estimators=100,gamma = 0,max_depth =3,min_child_weight = 1,seed=27,objective= 'binary:logistic')
    # modelfit(clf, train, labels)
    clf = XGBClassifier()
    clf.fit(train, labels)
    predict_res = clf.predict(test)
    if isvalid:
        print(accuracy_score(y_test,predict_res))
    else:
        pid = test_data['PassengerId']
        res = pd.DataFrame()
        res['PassengerId'] = pid
        res['Survived'] = predict_res
        res.to_csv(res_path,encoding='utf-8',index=False) 

def get_k_fold(data):
    kf = KFold(n_splits=5,random_state=1,shuffle=True)
    train_indexs = []
    test_indexs = []
    for train_index, test_index in kf.split(data):
        train_indexs.append(train_index)
        test_indexs.append(test_index)
    return train_indexs,test_indexs
    
def getTraintest():
    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data = pd.read_csv(test_path, encoding='utf-8')
    train_dummy,test_dummy= onhotcoder(train_data,test_data)
    train_features = preprocess(train_data,False,train_dummy)
    # X_train, X_test, y_train, y_test = train_test_split(train_features, train_data['Survived'], test_size=0.2, random_state=42)
    test_features= preprocess(test_data,True,test_dummy)
    # if isvalid: 
    select = getfeatureimportance(train_features,train_data['Survived'])
    train_features = train_features[select]
    labels = train_data['Survived']
    test_features = test_features[select]
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    labels = np.array(labels)
    return train_features,labels, test_features
    
    
def pseudomodel():
    model = XGBClassifier(learning_rate=0.05,n_estimators=100,gamma = 0,max_depth =3,reg_alpha=1e-5)
    # model = XGBClassifier()
    # model = RandomForestClassifier()
    trainfeatures,labels, test_features = getTraintest()
   
    sample_rates = np.linspace(0,1,10)
    rates = []
    scors = []
    
    # for s in sample_rates:
        # model = PseudoLabeler( model,test_features,sample_rate = s)
        # scores = cross_val_score(model, trainfeatures, labels, cv=5, scoring='roc_auc')
        # m = np.mean(scores)
        # rates.append(s)
        # scors.append(m)
        # print(m)
    # plt.plot(rates,scors)
    # plt.show() 
    model = PseudoLabeler( model,test_features,sample_rate = 0.44)
    scores = cross_val_score(model, trainfeatures, labels, cv=5, scoring='roc_auc')
    print(np.mean(scores))
    test_data = pd.read_csv(test_path, encoding='utf-8')
    # model = PseudoLabeler( model,test_features,sample_rate = 0.44)
    model.fit(trainfeatures,labels)
    pid = test_data['PassengerId']
    res = pd.DataFrame()
    res['PassengerId'] = pid
    res['Survived'] = [int(a) for a in model.predict(test_features)]
    res.to_csv(res_path,encoding='utf-8',index=False)

def getmissinginfo(train_data):
    pclass = len(train_data[(train_data['Pclass'].isnull()==False)])
    print("pclass" + str(pclass))
    name = len(train_data[(train_data['Name'].isnull()==False)])
    print("Name" + str(name))
    name = len(train_data[(train_data['Sex'].isnull()==False)])
    print("sex" + str(name))
    name = len(train_data[(train_data['Age'].isnull()==False)])
    print("age" + str(name))
    name = len(train_data[(train_data['SibSp'].isnull()==False)])
    print("SibSp" + str(name))
    name = len(train_data[(train_data['Parch'].isnull()==False)])
    print("Parch" + str(name))
    name = len(train_data[(train_data['Ticket'].isnull()==False)])
    print("ticket" + str(name))
    name = len(train_data[(train_data['Fare'].isnull()==False)])
    print("fare" + str(name))
    name = len(train_data[(train_data['Cabin'].isnull()==False)])
    print("Cabin" + str(name))
    name = len(train_data[(train_data['Embarked'].isnull()==False)])
    print("Embarked" + str(name))
    # print(train_data['Embarked'].median())
    
def get_train_test_combine():
    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data = pd.read_csv(test_path, encoding='utf-8')
    combine = pd.concat([train_data,test_data])
    return combine
    
def plotbox():
    combine = get_train_test_combine()
    combine.boxplot(column = "Fare",by=["Embarked","Pclass"])
    plt.show()


if __name__=="__main__":
    # preprocess(train_path)
    # analyze_relation()
    stackingmodel(0)
    # pseudomodel()
    
    # getmissinginfo(train_data)
    # combine =get_train_test_combine()
    # getmissinginfo(combine)
    # arr = combine[(combine['Fare'].isnull())]
    # print(arr)
    # onhotcoder(train_data,test_data)
    # plotbox()