# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:37 2019

@author: XYZ
"""

import numpy as np
import time
import sys, os

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc, precision_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif

# from xgboost import XGBClassifier as RandomForestClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.evaluate import feature_importance_permutation
import mlxtend


########################################################################################################################

# def SortFeature(X, y, ratio):
#     # TODO(X_index)  直接计算每一个列向量和label的关系 or 计算训练模型后预测值与label的关系
#     print(X.shape)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#     clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=1)
#     rf = clf.fit(X_train, y_train)
#     imp_vals, _ = feature_importance_permutation(
#         predict_method=rf.predict,
#         X=X_test,
#         y=y_test,
#         metric='accuracy',
#         num_rounds=1,
#         seed=1)
#     new_index = np.argsort(imp_vals)[::-1]
#     print(new_index.shape)
#     X_transition = X[:, new_index][:, 0:ratio]
#     print(X_transition.shape)
#     print('Feature has been sorted by feature_importance, return a new X.')
#     return X_transition

def SortFeatures(X, y, ratio, dim):
    samp = int(dim * ratio)
    selector = SelectPercentile(f_classif, percentile=100)
    X_new = selector.fit_transform(X, y)
    X_transition = X_new[:, 0:samp]
    # scores = -np.log10(selector1.pvalues_)
    # scores /= scores.max()
    # new_index = np.argsort(scores)
    # X_transition = X[:, new_index][:, 0:samp]
    # print(X_transition.shape)

    return X_transition


# def GetOptimalSubset(param_grid, X_train, y_train, clf, feature):
#     clf = RandomForestClassifier(n_estimators=500, oobscore=True,max_depth=50,min_samples_leaf=5)
#
#     sfs1 = SFS(estimator=clf,
#                k_features=10,
#                forward=True,
#                floating=False,
#                scoring='roc_auc',
#                cv=5,
#                n_jobs=-1)
#
#     pipe = Pipeline([('sfs', sfs1),
#                      ('clf', clf)])
#
#     tic = time.perf_counter()
#     print('Start Gridsearch, this process usually take lots of time.')
#     gs = GridSearchCV(estimator=pipe,
#                       param_grid=param_grid,
#                       scoring='roc_auc',
#                       n_jobs=-1,
#                       cv=5,
#                       iid=True,
#                       refit=True)
#     gs = gs.fit(X_train, y_train)
#     toc = time.perf_counter()
#
#     print("Gridsearch has been processed, used time %d min" % (toc - tic) / 60)
#
#     print("Best parameters via GridSearch", gs.best_params_)
#     print('Best score:', gs.best_score_)
#
#     joblib.dump(gs, "Human.RandomForest." + feature + ".model")
#     print('The model has been refit by the optimal parameter,and saved in your current working directory.')
#     return gs, probas

########################################################################################################################

def MakePrediction(X, y, clf_list, feature_index, iteration_round):
    if feature_index is not None:
        clf0 = clf_list[feature_index - 1]

    if iteration_round is not None:
        clf0 = clf_list[5 + iteration_round]
        # clf0 = clf_list[iteration_round-1]

    y_test_proba = clf0.predict_proba(X)[:, 1]
    # AUC, Accuracy, Sensitivity, Specificity, F1, Precision = EvaluatePerformances(y, y_proba, y_pred)

    return y_test_proba


def EvaluatePerformances(real_labels, predicted_probas, predicted_labels):
    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probas, pos_label=1)
    AUC = auc(fpr, tpr)
    Accuracy = accuracy_score(real_labels, predicted_labels)
    Sensitivity = recall_score(real_labels, predicted_labels)
    sample_num = len(real_labels)
    posi_num = sum(real_labels)
    Specificity = (Accuracy * sample_num - Sensitivity * posi_num) / (sample_num - posi_num)
    Precision = precision_score(real_labels, predicted_labels)
    F1 = f1_score(real_labels, predicted_labels)
    return AUC, Accuracy, Sensitivity, Specificity, F1, Precision


def CrossValidation(X, y, seed, folds, cv_round, clf_list):
    probas = np.zeros(len(y))
    labels = np.zeros(len(y))
    AUCs = []
    Accuracys = []
    Sensitivitys = []
    Specificitys = []
    F1s = []
    Precisions = []
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=1, n_jobs=-1)
    clf0 = clf.fit(X, y)
    clf_list.append(clf0)

    kf = KFold(n_splits=folds,
               shuffle=True,
               random_state=np.random.RandomState(seed))

    for train_index, vali_index in kf.split(X, y):
        start = time.perf_counter()
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_vali_fold = X[vali_index]
        y_vali_fold = y[vali_index]
        clf1 = clf.fit(X_train_folds, y_train_folds)
        y_pred = clf1.predict(X_vali_fold)
        y_predicted_probas = clf1.predict_proba(X_vali_fold)[:, 1]
        labels[vali_index] = y_pred
        probas[vali_index] = y_predicted_probas
        AUC, Accuracy, Sensitivity, Specificity, F1, Precision = EvaluatePerformances(y_vali_fold, y_predicted_probas,
                                                                                      y_pred)

        AUCs.append(AUC)
        Accuracys.append(Accuracy)
        Sensitivitys.append(Sensitivity)
        Specificitys.append(Specificity)
        F1s.append(F1)
        Precisions.append(Precision)
        end = time.perf_counter()

        # print('AUC %.4f, Accuracy %.4f,Sensitivity %.4f, Specificity %.4f,F1 %.4f, Precision %.4f' % (
        #     AUC, Accuracy, Sensitivity, Specificity, F1, Precision))
        # print('round %d, running time: %.4f minutes' % (cv_round, (end - start) / 60))
        # print('..........................................................................\n')
        cv_round += 1

    return np.mean(AUCs), np.mean(Accuracys), np.mean(Sensitivitys), np.mean(Specificitys), \
           np.mean(F1s), np.mean(Precisions), probas, labels, clf_list


########################################################################################################################


if __name__ == '__main__':
    n_trees = 500  # the initial number of trees for the random forest
    all_features = ['Spectrum', 'RevCSpectrum', 'PseudoComposition', 'AutoCorrelation', 'Mismatch', 'Genomic']
    train_probas_vector = None
    test_probas_vector = None
    iterate_round = 1
    max_iteration = 51
    feature_index = 1
    cv_round = 1
    seed = 1
    score = []
    clf_list = []
    folds = 10
    y_train = None
    y_test = None

    start = time.perf_counter()

    for feature in all_features:
        posi_features_file = sys.argv[1]
        nega_features_file = sys.argv[2]
        posi_prefix, nega_prefix = os.path.splitext(posi_features_file)[0], os.path.splitext(nega_features_file)[0]
        fp = np.loadtxt(posi_prefix + '.' + feature)
        fn = np.loadtxt(nega_prefix + '.' + feature)
        # fn = fn[:len(fp)]
        fp = fp[:10000]
        fn = fn[:20000]
        posi_samples, nega_samples, dim = np.shape(fp)[0], np.shape(fn)[0], np.shape(fp)[1]
        print("For feature", feature, ', start constructing the initial probability matrix, which has shape[samples,6]')
        print('The number of positive and negative samples: %d, %d' % (posi_samples, nega_samples))

        X1 = np.vstack((fp, fn))
        y = np.array([1] * posi_samples + [0] * nega_samples)
        selector = SelectPercentile(f_classif, percentile=100)
        X = selector.fit_transform(X1, y)
        # for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        #     samp = int(ratio * dim)
        #     print("Extract %.2f features of all features" % ratio)
        #     X = X_new[:, 0:samp]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

        AUC, Accuracy, Sensitivity, Specificity, F1, Precision, predict_probas, predict_labels, clf_list = \
            CrossValidation(X_train, y_train, seed, folds, cv_round, clf_list)

        y_test_proba = MakePrediction(X_test, y_test, clf_list, feature_index, iteration_round=None)

        print("AUC:", AUC)
        print("Accuracy:", Accuracy)
        print("Sensitivity:", Sensitivity)
        print("Specificity:", Specificity)
        print("F1:", F1)
        print("Precision:", Precision)

        # clf2 = clf.fit(X, y)
        # joblib.dump(clf2, "Human.RandomForest." + feature + ".model")
        # print("The initial feature model has been refitted and saved in your current directory.")

        if train_probas_vector is None:
            train_probas_vector = predict_probas
        else:
            print(train_probas_vector.shape)
            print(predict_probas)
            train_probas_vector = np.vstack((train_probas_vector, predict_probas))

        if test_probas_vector is None:
            test_probas_vector = y_test_proba
        else:
            test_probas_vector = np.vstack((test_probas_vector, y_test_proba))

        feature_index += 1

        print("For feature", feature, ', The new probability matrix has been created, waiting for iterating.')
        print("###################################################################################################")

    X_train_probas = train_probas_vector.T
    X_test_probas = test_probas_vector.T
    y_train_probas = y_train
    y_test_probas = y_test

    # np.savetxt('X_train_human.txt', X_train_probas)
    # np.savetxt('y_train_human.txt', y_train_probas)
    # np.savetxt('X_test_human.txt', X_test_probas)
    # np.savetxt('y_test_human.txt', y_test_probas)
    #
    # X_train_probas = np.loadtxt('X_train_mouse.txt')
    # X_test_probas = np.loadtxt('X_test_mouse.txt')
    # y_train_probas = np.loadtxt('y_train_mouse.txt')
    # y_test_probas = np.loadtxt('y_test_mouse.txt')

    for round in range(1, max_iteration):
        print("Start iterating probability matrix, current round is %d." % round)

        AUC, Accuracy, Sensitivity, Specificity, F1, Precision, new_train_probas, predict_labels, clf_list = \
            CrossValidation(X_train_probas, y_train_probas, seed, folds, cv_round, clf_list)
        new_test_probas = MakePrediction(X_test_probas, y_test_probas, clf_list, feature_index=None,
                                         iteration_round=round)

        new_train_probas = np.transpose([new_train_probas])
        X_train_probas = np.hstack((X_train_probas, new_train_probas))
        new_test_probas = np.transpose([new_test_probas])
        X_test_probas = np.hstack((X_test_probas, new_test_probas))

        print(X_train_probas.shape)
        print(X_test_probas.shape)

        score.append(AUC)

        print("For iteration round:" + str(round) + ', The AUC is %.4f' % AUC)
        print("Accuracy:", Accuracy)
        print("Sensitivity:", Sensitivity)
        print("Specificity:", Specificity)
        print("F1:", F1)
        print("Precision:", Precision)

        print(score)
        print("#######################################################################################################")
    score = np.array(score)
    max_score = np.max(score)
    max_index = np.argmax(score)

    final_X_test = X_test_probas[:, 0:max_index]
    final_clf = RandomForestClassifier(n_estimators=n_trees, random_state=1, n_jobs=-1).fit(X_train_probas[:,0:max_index], y_train_probas)
    final_proba = final_clf.predict_proba(final_X_test)[:, 1]
    final_predict_label = final_clf.predict(final_X_test)
    AUC, Accuracy, Sensitivity, Specificity, F1, Precision = EvaluatePerformances(y_test_probas, final_proba, final_predict_label)
    print("For final result, scores are show as follows:")
    print("AUC", AUC)
    print("Accuracy:", Accuracy)
    print("Sensitivity:", Sensitivity)
    print("Specificity:", Specificity)
    print("F1:", F1)
    print("Precision:", Precision)

    print(final_proba)


    print('For all iteration round, the best auc is', max_score, ', best round is', max_index)

    # X_train_backend = X_train_probas[:, 7:max_index]
    # y_train_backend = y_train
    # X_test_backend = X_test_probas[:, 7:max_index]
    # y_test_backend = y_test
    #
    # final_clf = RandomForestClassifier(n_estimators=n_trees, random_state=1, n_jobs=-1).fit(X_train_backend, y_train_backend)
    # final_proba = final_clf.predict_proba(X_test_backend)[:, 1]
    # final_predict_label = final_clf.predict(X_test_backend)
    # AUC, Accuracy, Sensitivity, Specificity, F1, Precision = EvaluatePerformances(y_test_backend, final_proba, final_predict_label)
    #
    # print("For final result, scores are show as follows:")
    # print("AUC", AUC)
    # print("Accuracy:", Accuracy)
    # print("Sensitivity:", Sensitivity)
    # print("Specificity:", Specificity)
    # print("F1:", F1)
    # print("Precision:", Precision)

    np.savetxt('X_iteration.txt', X_train_probas[:, 0:5+max_index])



    end = time.perf_counter()
    print('All process used time %.2f hours' % ((end - start) / 3600))
