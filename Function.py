import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


def fillNANValue(traindata):
    for i in np.arange(40):
        traindata.iloc[:,i] = traindata.iloc[:,i].fillna(method="bfill")
        traindata.iloc[:,i] = traindata.iloc[:,i].fillna(method="ffill")
        #print(traindata)
        traindata["O2Sat"] = traindata["O2Sat"].fillna(97.5)
        traindata["Temp"] = traindata["Temp"].fillna(36.5)
        traindata["BaseExcess"] = traindata["BaseExcess"].fillna(26)
        traindata["SBP"] = traindata["SBP"].fillna(110)
        traindata["DBP"] = traindata["DBP"].fillna(75)
        traindata["EtCO2"] = traindata["EtCO2"].fillna(40)
        traindata["Resp"] = traindata["Resp"].fillna(16)
        traindata["MAP"] = traindata["MAP"].fillna(85)
        traindata["HCO3"] = traindata["HCO3"].fillna(26)
        traindata["pH"] = traindata["pH"].fillna(7.4)
        traindata["PaCO2"] = traindata["PaCO2"].fillna(40)
        traindata["FiO2"] = traindata["FiO2"].fillna(0.21)
        traindata["SaO2"] = traindata["SaO2"].fillna(97)
        traindata["AST"] = traindata["AST"].fillna(25)
        traindata["BUN"] = traindata["BUN"].fillna(15)
        traindata["Alkalinephos"] = traindata["Alkalinephos"].fillna(95.5)
        traindata["Calcium"] = traindata["Calcium"].fillna(9.5)
        traindata["Chloride"] = traindata["Chloride"].fillna(102)
        traindata["Creatinine"] = traindata["Creatinine"].fillna(0.9)
        traindata["Bilirubin_direct"] = traindata["Bilirubin_direct"].fillna(0.2)
        traindata["Glucose"] = traindata["Glucose"].fillna(100.0)
        traindata["Lactate"] = traindata["Lactate"].fillna(1.35)
        traindata["Magnesium"] = traindata["Magnesium"].fillna(2.0)
        traindata["Phosphate"] = traindata["Phosphate"].fillna(3.5)
        traindata["Potassium"] = traindata["Potassium"].fillna(4.4)
        traindata["Bilirubin_total"] = traindata["Bilirubin_total"].fillna(0.6)
        traindata["Unit1"] = traindata["Unit1"].fillna(1)
        traindata["Unit2"] = traindata["Unit2"].fillna(1)
        gender = traindata.iloc[1,35]
        if gender==1:   
            traindata["Hct"] = traindata["Hct"].fillna(48.5)
            traindata["Hgb"] = traindata["Hgb"].fillna(15.5)
        else:
            traindata["Hct"] = traindata["Hct"].fillna(42.5)
            traindata["Hgb"] = traindata["Hgb"].fillna(13.7)
        
        traindata["TroponinI"] = traindata["TroponinI"].fillna(0.2)
        traindata["PTT"] = traindata["PTT"].fillna(30)
        traindata["WBC"] = traindata["WBC"].fillna(9)
        traindata["Fibrinogen"] = traindata["Fibrinogen"].fillna(300)
        traindata["Platelets"] = traindata["Platelets"].fillna(275)
    return traindata;


def loadfiles(start,end,step,path,initial):
    for file in os.listdir(path)[start:end:step]:
        pathofFile = os.path.join(path,file)
        files = pd.read_csv(pathofFile ,sep='|' )
        files = fillNANValue(files)
        x = files.copy()
        initial = pd.concat([initial,x],ignore_index=True)
    Train_set = initial.copy()
    return Train_set;

def loadfilesnew(path,files,initial):
    c = 0
    for file in files:
        c += 1
        pathofFile = os.path.join(path,file)
        file_loaded = pd.read_csv(pathofFile ,sep='|' )
        file_loaded = fillNANValue(file_loaded)
        initial = pd.concat([initial,file_loaded],ignore_index=True)
        if c % 500 == 0:
            print('Done',c,end = ' ')
    return initial;

def result(clf,X,y,test_X,test_y):
    predicted_test_y = clf.predict(test_X)
    predicted_train_y = clf.predict(X)
    
    #for test
    confmat_test = confusion_matrix(test_y,predicted_test_y)
    accuracy_test = accuracy_score(test_y,predicted_test_y)
    precision_test = precision_score(test_y,predicted_test_y)
    recall_test = recall_score(test_y,predicted_test_y)
    f1_score_test = f1_score(test_y,predicted_test_y)
    print("Confusion matrix\n" ,confmat_test)
    print("Accuracy on test set" , accuracy_test)
    print("Precision on test set" , precision_test)
    print("Recall on test set" , recall_test)
    print("F1 score on test set" , f1_score_test)
    print("Classification report on test\n")
    #clasification report
    print(classification_report(test_y, predicted_test_y, target_names=['not sepsis', 'sepsis']))
    
    # for train 
    confmat_train = confusion_matrix(y,predicted_train_y)
    accuracy_train = accuracy_score(y,predicted_train_y)
    precision_train = precision_score(y,predicted_train_y)
    recall_train = recall_score(y,predicted_train_y)
    f1_score_train = f1_score(y,predicted_train_y)
    print("Confusion matrix on train set\n",confmat_train )
    print("Accuracy on train set" , accuracy_train)
    print("Precision on train set" , precision_train)
    print("Recall on train set" , recall_train)
    print("F1 score on train set" , f1_score_train)
    #clasification report
    print("Classification report on train\n")
    print(classification_report(y, predicted_train_y, target_names=['not sepsis', 'sepsis']))
    
    
def prcurve(clf,X,y,test_X,test_y):
    #precision recall curve
    y_scores_clf = clf.decision_function(test_X)
    precision, recall, thresholds = precision_recall_curve(test_y, y_scores_clf)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    #print(precision, recall, thresholds,closest_zero_p,closest_zero_r,closest_zero,np.abs(thresholds))
    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show() 
    
    
def ROC_AUCcurve(clf,X,y,test_X,test_y):
    # ROC AUC curve
    y_score_clf = clf.decision_function(test_X)
    fpr_clf, tpr_clf, thresholds = roc_curve(test_y, y_score_clf)
    roc_auc_clf = auc(fpr_clf, tpr_clf)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = fpr_clf[closest_zero]
    closest_zero_r = tpr_clf[closest_zero]
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_clf, tpr_clf, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_clf))
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    
    
def plot_feature_importances(clf, feature_names):
    c_features = len(feature_names)
    plt.barh(range(c_features), clf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), feature_names)    
    
    
def Grid_result(clf,X_train,y_train,X_test,y_test,scoring,grid_values):
    # alternative metric to optimize over grid parameters: AUC
    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring = scoring)
    grid_clf.fit(X_train, y_train)
    y_predicted = grid_clf.predict(X_test) 
    y_decision_fn_scores = grid_clf.decision_function(X_test)
    if scoring == None:
        print('Test set Accuracy: ', accuracy_score(y_test, y_predicted))
        print('Grid best parameter (max. Accuracy): ', grid_clf.best_params_)
        print('Grid best score (Accuracy): ', grid_clf.best_score_)
        #print('accuracy',grid_clf.cv_results_)  
    else:
        if scoring == "precision":
            print('Test set {}: '.format(scoring), precision_score(y_test, y_predicted)) 
        elif scoring == "recall":
            print('Test set {}: '.format(scoring), recall_score(y_test, y_predicted))
        elif scoring == "f1":
            print('Test set {}: '.format(scoring), f1_score(y_test, y_predicted))
        else:
            print('Test set {}: '.format(scoring), roc_auc_score(y_test, y_decision_fn_scores))
            
        print('Grid best parameter (max. {}): '.format(scoring), grid_clf.best_params_)
        print('Grid best score ({}): '.format(scoring), grid_clf.best_score_)    
        
        
def pca(X_train,y_train,X_test,dimention,feature_names):
        
    pca = PCA(n_components = dimention).fit(X_train)
    X_pca = pca.transform(X_train)
    X_pca_test = pca.transform(X_test)
    print("shape before and after PCA\n",X_train.shape, X_pca.shape)

    fig = plt.figure(figsize=(18, 7))
    plt.imshow(pca.components_, interpolation = 'none', cmap = 'plasma')
    
    plt.gca().set_xticks(np.arange(-.5, len(feature_names)));
    plt.gca().set_yticks(np.arange(0.5, 2));
    plt.gca().set_xticklabels(feature_names, rotation=90, ha='left', fontsize=12);
    
    plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0, 
                                                  pca.components_.max()], pad=0.65);
    return X_pca,X_pca_test;