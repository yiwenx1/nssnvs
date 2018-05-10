import copy
import math

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.metrics import recall_score,matthews_corrcoef
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split


# MutationAssessor, PolyPhen 2, SIFT, FATHMM-W, CADD, Condel
score_index = [1,3,5,7,9,10]
# CADD only have score, does not have a prediction
pred_index = [2,4,6,8,-1,11]
file_names = ["humVar_scores.csv",
              "exovar_scores.csv",
              "predictSNPSelected_scores.csv"]
validate_data = ["variBenchSelected_scores.csv"]


def load_data(file_names):
    """ load data from csv files
    Return:
        Scores of tools: n*6 matrix (n: number of variants, 6: number of tools)
        Predictions of tools: n*6 matrix (n: number of variants, 6: number of tools)
        True labels: numpy array of length n (n: number of variants)
    """

    true_label = []
    data_matrix = []
    tool_label = []

    for file_name in file_names:
        file = open(file_name)
        for i, line in enumerate(file):
            # exclude the head line
            if i == 0:
                continue
            else:
                items = line.strip().split(",")
                true_label.append(int(items[0]))
                data_line = np.ones(len(score_index))
                pred_line = np.ones(len(pred_index))
                for j in range(len(score_index)):
                    # normalize the scores into an interval [0,1]
                    # SIFT
                    if score_index[j] == 5:
                        data_line[j] = 1-float(items[score_index[j]])
                    # Mutation Accessor
                    elif score_index[j] == 1:
                        ori_score = float(items[score_index[j]])
                        if ori_score <= 1.938:
                            data_line[j] = 0.5*(ori_score+5.545)/7.483
                        else:
                            data_line[j] = 0.5 +0.5 * (ori_score -1.938) / 3.999
                    # FATHMM
                    elif score_index[j] == 7:
                        ori_score = float(items[score_index[j]])
                        if ori_score >-1.5:
                            data_line[j] = 0.5 * (10.64-ori_score) / 12.14
                        else:
                            data_line[j] = 0.5 - 0.5 * (ori_score +1.5) / 14.63
                    # CADD
                    elif score_index[j] == 9:
                        ori_score = float(items[score_index[j]])
                        if ori_score > 15:
                            pred_line[j] = 1
                        else:
                            pred_line[j] = -1
                        data_line[j] = ori_score/40
                    # PolyPhen 2 or Condel
                    else:
                        data_line[j] = float(items[score_index[j]])
                    # data_line[j] = float(items[score_index[j]])
                    if score_index[j] != 20:
                        pred_line[j] = float(items[pred_index[j]])
                data_matrix.append(data_line)
                tool_label.append(pred_line)
        file.close()
    return np.array(data_matrix),np.array(tool_label),np.array(true_label)


def plot_ROC_curve(y_true, y_score):
    """ get the x-axis and y-axis for plotting the ROC curve by varying a threshold on y_score.
    Assume that the positive class is labeled truly as 1, and that if given a threshold value,
    we can classify the examples with: (y_score > threshold).astype(int)
    Return:
        The area under your curve (AUROC)
        X-axis data and Y-axis date used to plot ROC curve
    """

    thresholds = np.unique(y_score)
    x_axis = []
    y_axis = []
    f_score = []
    for i in range(0,len(thresholds)):
        y_prediction = (y_score > thresholds[i]).astype(int)

        TP = np.sum((y_true == y_prediction) & (y_true == 1))
        FN = np.sum((y_true != y_prediction) & (y_true == 1))
        FP = np.sum((y_true != y_prediction) & (y_true == 0))
        TN = len(y_true)-TP-FN-FP

        true_positive = TP/(TP+FN)
        # precision = TP/(TP+FP)
        false_positive = FP/(TN+FP)

        x_axis.append(false_positive)
        y_axis.append(true_positive)
        # use this expression instead of recall and precision
        # to avoid the denominator of precision to be 0
        f_score.append(2*TP/(2*TP+FP+FN))

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = sklearn.metrics.auc(fpr, tpr)

    return auc,x_axis,y_axis


def get_metrics(y_true, y_prediction):
    """ print out and return the metrics for evaluating the performance of tools
    Return:
        Accuracy
        Precision
        Recall
        Specificity
        F-score
        Negative predictive value
        Matthews Correlation Coefficient
        The area under ROC curve
    """

    TP = np.sum((y_true == y_prediction) & (y_true == 1))
    FN = np.sum((y_true != y_prediction) & (y_true == 1))
    FP = np.sum((y_true != y_prediction) & (y_true == -1))
    TN = len(y_true) - TP - FN - FP
    print(len(y_true),TP,FN,FP,TN)

    accuracy = (TP+TN)/(TP+FP+TN+FN)
    precision = TP / (TP + FP)
    # recall/sensitivity/true positive
    recall = TP/(TP+FN)
    # specificity/false positive
    specificity = TN / (TN + FP)
    f_score = 2 * TP / (2 * TP + FP + FN)
    # negative predictive value
    NPV = TN/(TN+FN)
    mcc = (TP*TN-FP*FN)/(math.sqrt(TP+FP)*math.sqrt(TP+FN)*math.sqrt(TN+FP)*math.sqrt(TN+FN))

    fpr, tpr, _ = metrics.roc_curve(y_true, y_prediction)
    auc = sklearn.metrics.auc(fpr, tpr)

    print("accuracy:",accuracy,",precision:",precision,",recall:",recall,",specificity:",specificity)
    print("f score:",f_score,",NPV:",NPV,",MCC:",mcc,",AUC",auc)

    return accuracy,precision,recall,specificity,f_score,NPV,mcc,auc


def print_metrics(accuracy,precision,recall,f_score,mcc,auc):
    print("accuracy:%0.2f" % accuracy, ",precision:%0.2f" % precision,
          ",recall:%0.2f" % recall, ",f-score:%0.2f" % f_score,
          ",mcc:%0.2f" % mcc, ",auc:%0.2f" % auc)


if __name__ == '__main__':
    data, tool_label, true_label = load_data(file_names)
    x_list = []
    y_list = []
    auc_list = []
    legend = ['Mutation Assessor', 'PolyPhen 2', 'SIFT', 'FATHMM-W', 'CADD', 'Condel'
        , 'Logistic Regression', 'SVM', 'Random Forest', 'Adaboost']
    for i in range(len(tool_label[0])):
        print("----"+legend[i]+"-----")
        y_score = data[np.logical_not(np.isnan(data[:, i])),i]
        y_true = true_label[np.logical_not(np.isnan(data[:, i]))]
        y_prediction = tool_label[np.logical_not(np.isnan(data[:, i])),i]
        y_bin = copy.copy(y_true)
        y_bin[y_bin == -1] = 0
        auc,x_axis,y_axis=plot_ROC_curve(y_bin, y_score)
        get_metrics(y_true,y_prediction)
        x_list.append(x_axis)
        y_list.append(y_axis)
        auc_list.append(auc)
    full_data = data[np.sum(np.isnan(data[:, :6]), 1) == 0, :4]
    full_label = true_label[np.sum(np.isnan(data[:, :6]), 1) == 0]

    y_bin = copy.copy(full_label)
    y_bin[y_bin==-1]=0

    # plot ROC curves
    plt.figure()
    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i], '-')
        legend[i] += ", auc={0:.3f}".format(auc_list[i])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(legend, loc='lower right')

    plt.plot(x_list[0],x_list[0],'grey',ls='--')
    plt.show()

    '''
    cross-validation
    '''
    scoring = {'accuracy': 'accuracy',
               'prec': 'precision',
               'recall': 'recall',
               'f': 'f1',
               'mcc': make_scorer(matthews_corrcoef),
               'auc:':make_scorer(sklearn.metrics.auc)}
    full_label[full_label == -1] = 0

    print("----Logistic regression-----")
    logistic = LogisticRegression()
    scores = cross_validate(logistic, full_data, full_label, scoring=scoring,
                            cv = 10, return_train_score = False)
    print_metrics(scores['test_accuracy'].mean(),scores['test_prec'].mean(),scores['test_recall'].mean(),
                  scores['test_f'].mean(),scores['test_mcc'].mean(),scores['test_auc'].mean())
    log_scores = cross_val_score(logistic, full_data, full_label, cv=10)
    auc, x_axis, y_axis = plot_ROC_curve(y_bin, log_scores)
    x_list.append(x_axis)
    y_list.append(y_axis)
    auc_list.append(auc)

    print("----svm-----")
    clf = svm.SVC()
    scores = cross_validate(clf, full_data, full_label, scoring=scoring,
                            cv = 10, return_train_score = False)
    print_metrics(scores['test_accuracy'].mean(), scores['test_prec'].mean(), scores['test_recall'].mean(),
                  scores['test_f'].mean(), scores['test_mcc'].mean(), scores['test_auc'].mean())
    svm_scores = cross_val_score(clf, full_data, full_label, cv=10)
    auc, x_axis, y_axis = plot_ROC_curve(y_bin, svm_scores)
    x_list.append(x_axis)
    y_list.append(y_axis)
    auc_list.append(auc)

    print("----Random Forest-----")
    rf = RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=1)
    scores = cross_validate(rf, full_data, full_label, scoring=scoring,
                            cv = 10, return_train_score = False)
    print_metrics(scores['test_accuracy'].mean(), scores['test_prec'].mean(), scores['test_recall'].mean(),
                  scores['test_f'].mean(), scores['test_mcc'].mean(), scores['test_auc'].mean())
    rf_scores = cross_val_score(rf, full_data, full_label, cv=10)
    auc, x_axis, y_axis = plot_ROC_curve(y_bin, rf_scores)
    x_list.append(x_axis)
    y_list.append(y_axis)
    auc_list.append(auc)

    print("----Adaboost-----")
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                              algorithm="SAMME.R",
                              n_estimators=1000,
                              learning_rate=0.07)
    scores = cross_validate(bdt, full_data, full_label, scoring=scoring,
                            cv = 10, return_train_score = False)
    print_metrics(scores['test_accuracy'].mean(), scores['test_prec'].mean(), scores['test_recall'].mean(),
                  scores['test_f'].mean(), scores['test_mcc'].mean(), scores['test_auc'].mean())
    bdt_scores = cross_val_score(bdt, full_data, full_label, cv=10)
    auc, x_axis, y_axis = plot_ROC_curve(y_bin, bdt_scores)
    x_list.append(x_axis)
    y_list.append(y_axis)
    auc_list.append(auc)

    # plot ROC curve
    plt.figure()
    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i], '-')
        legend[i] += ", auc={0:.3f}".format(auc_list[i])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(legend, loc='lower right')

    plt.plot(x_list[0],x_list[0],'grey',ls='--')
    plt.show()

    '''
        external validation
    '''
    data, _, true_label = load_data(validate_data)
    test_data = data[np.sum(np.isnan(data[:, :6]), 1) == 0, :4]
    test_label = true_label[np.sum(np.isnan(data[:, :6]), 1) == 0]
    test_bin = copy.copy(test_label)
    test_bin[test_bin==-1]=0

    print("----Logistic regression-----")
    logistic = LogisticRegression()
    logistic.fit(full_data, full_label)
    get_metrics(test_label, logistic.predict(test_data))
    log_score = logistic.predict_proba(test_data)[:, 1]
    auc, x_axis, y_axis = plot_ROC_curve(test_bin, log_scores)
    x_list[6]=x_axis
    y_list[6]=y_axis
    auc_list[6]=auc

    print("----svm-----")
    clf = svm.SVC()
    clf.fit(full_data, full_label)
    get_metrics(test_label, clf.predict(test_data))
    svm_score = clf.decision_function(test_data)
    auc, x_axis, y_axis = plot_ROC_curve(test_bin, svm_scores)
    x_list[7] = x_axis
    y_list[7] = y_axis
    auc_list[7] = auc

    print("----Random Forest-----")
    rf = RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=1)
    rf.fit(full_data, full_label)
    get_metrics(test_label, rf.predict(test_data))
    rf_score = rf.predict_proba(test_data)[:, 1]
    auc, x_axis, y_axis = plot_ROC_curve(test_bin, rf_scores)
    x_list[8] = x_axis
    y_list[8] = y_axis
    auc_list[8] = auc

    print("----Adaboost-----")
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME.R",
                             n_estimators=1000,
                             learning_rate=0.08)
    bdt.fit(full_data, full_label)
    get_metrics(test_label, bdt.predict(test_data))
    ada_score = bdt.predict_proba(test_data)[:, 1]
    auc, x_axis, y_axis = plot_ROC_curve(test_bin, bdt_scores)
    x_list[9] = x_axis
    y_list[9] = y_axis
    auc_list[9] = auc

    # plot ROC curve
    plt.figure()
    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i], '-')
        legend[i] += ", auc={0:.3f}".format(auc_list[i])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(legend, loc='lower right')

    plt.plot(x_list[0],x_list[0],'grey',ls='--')
    plt.show()

