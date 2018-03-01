#!/usr/bin/python

import sys
import pickle
sys.path.append("C:/Users/Administrator/ud120-projects-master/tools/")
sys.path.append("C:/Users/Administrator/ud120-projects-master/choose_your_own")
sys.path.append("C:/Users/Administrator/ud120-projects-master/final_project/")

import os
os.chdir('C:/Users/Administrator/ud120-projects-master/final_project')

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','long_term_incentive','deferred_income',
                 'deferral_payments','total_payments','exercised_stock_options',
                 'restricted_stock','restricted_stock_deferred','total_stock_value'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict.pop("TOTAL",0)

for i in data_dict:
    person = data_dict[i]
    if (all([person['from_poi_to_this_person'] != 'NaN',
             person['from_this_person_to_poi'] != 'NaN',
             person['to_messages'] != 'NaN',
             person['from_messages'] != 'NaN'])):
        fraction_from_poi = float(person["from_poi_to_this_person"]) / float(person["to_messages"])
        person["fraction_from_poi"] = fraction_from_poi
        fraction_to_poi = float(person["from_this_person_to_poi"]) / float(person["from_messages"])
        person["fraction_to_poi"] = fraction_to_poi
    else:
        person["fraction_from_poi"] = person["fraction_to_poi"] = 0

my_features_list = ['poi','salary','bonus','long_term_incentive','deferred_income',
                    'deferral_payments','total_payments','exercised_stock_options',
                    'restricted_stock','restricted_stock_deferred','total_stock_value',
                    'fraction_from_poi','fraction_to_poi']

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# features selection


from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

#pca = PCA(n_components=2)

#estimators = [('pca',pca), ('kernel_pca', KernelPCA())]
#combined_features = FeatureUnion(estimators)


# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#from sklearn.pipeline import Pipeline
#clf = GaussianNB()
#pipe_nb=Pipeline([("features", combined_features),('clf',GaussianNB())])
#test_classifier(pipe_nb, my_dataset, my_features_list)



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#LogisticRegression
pipe_lr = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression())])   #random_state=1
pipe_lr.fit(features_train, labels_train)
print('Test accuracy_lr: %.3f' % pipe_lr.score(features_test, labels_test))
test_classifier(pipe_lr, my_dataset, my_features_list)

#GaussianNB
from sklearn.naive_bayes import GaussianNB
pipe_nb = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', GaussianNB()) ])
pipe_nb.fit(features_train, labels_train)
print('Test accuracy_nb: %.3f' % pipe_nb.score(features_test, labels_test))
test_classifier(pipe_nb, my_dataset, my_features_list)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
pipe_tr = Pipeline([ ('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', tree)])
pipe_tr.fit(features_train, labels_train)
print('Test accuracy_tr: %.3f' % pipe_tr.score(features_test, labels_test))
test_classifier(pipe_tr, my_dataset, my_features_list)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()   #max_depth=2, random_state=0
pipe_rf = Pipeline([('pca', PCA(n_components=2)),
                    ('clf', clf)])
pipe_rf.fit(features_train, labels_train)
print('Test accuracy_rf: %.3f' % pipe_rf.score(features_test, labels_test))
test_classifier(pipe_rf, my_dataset, my_features_list)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


from sklearn.model_selection import GridSearchCV


# LogisticRegression
param_grid =[{'penalty': ('l1', 'l2'),
               'C': [0.1, 1.0, 10, 100]}]

lr = GridSearchCV( LogisticRegression(),
                  param_grid = param_grid, scoring = 'f1', cv = 10)
lr.fit(features_train, labels_train)
lr_best = lr.best_estimator_ #LogisticRegression
print lr_best
test_classifier(lr_best, my_dataset, my_features_list)


#RandomForestClassifier
param_grid =[{ 'n_estimators': [5, 10, 50],
               'criterion': ('gini', 'entropy'),
               'min_samples_split': [2, 5, 10, 20, 40, 70] }]

rf = GridSearchCV( RandomForestClassifier(),
                  param_grid = param_grid, scoring = 'f1', cv = 10)
rf.fit(features_train, labels_train)
rf_best = rf.best_estimator_ #RandomForestClassifier
print rf_best
test_classifier(rf_best, my_dataset, my_features_list)


#DecisionTreeClassifier
param_grid =[{ 'splitter': ('best', 'random'),
               'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
               'max_depth': [None, 10, 30, 50, 70, 100] }]

tr = GridSearchCV( DecisionTreeClassifier(),
                  param_grid = param_grid, scoring = 'f1', cv = 10)
tr.fit(features_train, labels_train)
tr_best = tr.best_estimator_ #DecisionTreeClassifier
print tr_best
test_classifier(tr_best, my_dataset, my_features_list)



# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

################################	没加新特征########
#LogisticRegression
pipe_lr = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression())])   #random_state=1
pipe_lr.fit(features_train, labels_train)
test_classifier(pipe_lr, my_dataset, features_list)

#GaussianNB
from sklearn.naive_bayes import GaussianNB
pipe_nb = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', GaussianNB()) ])
pipe_nb.fit(features_train, labels_train)
test_classifier(pipe_nb, my_dataset, features_list)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
pipe_tr = Pipeline([ ('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', tree)])
pipe_tr.fit(features_train, labels_train)
test_classifier(pipe_tr, my_dataset, features_list)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()   #max_depth=2, random_state=0
pipe_rf = Pipeline([('pca', PCA(n_components=2)),
                    ('clf', clf)])
pipe_rf.fit(features_train, labels_train)
test_classifier(pipe_rf, my_dataset, features_list)

###############	没加新特征以上################

###########所以算法增加selectkbest#########
#LogisticRegression
pipe_lr = Pipeline([('univ_select',SelectKBest(k=7)),
                    ('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression())])   #random_state=1
pipe_lr.fit(features_train, labels_train)
test_classifier(pipe_lr, my_dataset, my_features_list)

#GaussianNB
from sklearn.naive_bayes import GaussianNB
pipe_nb = Pipeline([('univ_select',SelectKBest(k=7)),
                    ('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', GaussianNB())])
pipe_nb.fit(features_train, labels_train)
test_classifier(pipe_nb, my_dataset, my_features_list)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
pipe_tr = Pipeline([('univ_select',SelectKBest(k=7)),
                    ('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', tree)])
pipe_tr.fit(features_train, labels_train)
test_classifier(pipe_tr, my_dataset, my_features_list)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()   #max_depth=2, random_state=0
pipe_rf = Pipeline([('univ_select',SelectKBest(k=7)),
                    ('pca', PCA(n_components=2)),
                    ('clf', clf)])
pipe_rf.fit(features_train, labels_train)
test_classifier(pipe_rf, my_dataset, my_features_list)

###########增加selectkbest以上部分###############

# KNN算法
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(features_train, labels_train)
test_classifier(clf, my_dataset, my_features_list)

#调参后KNN算法
clf = NearestCentroid()
pipe_knn = Pipeline([('univ_select',SelectKBest(k=7)),
                    ('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', NearestCentroid())])
pipe_knn.fit(features_train, labels_train)
test_classifier(pipe_knn, my_dataset, my_features_list)
	
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)