
# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv('iris.data', names=names)

# Show instance and attribute count
print("Instances and Attributes: ")
print(dataset.shape)

# Summarize the attributes
print("\nSummary: ")
print(dataset.describe())

# Class distribution
print("\nClass distribution: ")
print(dataset.groupby('class').size())

# Box and Whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.savefig('Boxplot.png')
plt.close()
# Histograms
dataset.hist()
plt.savefig("Histogram.png")
plt.close()
# Scatter plot matrix
scatter_matrix(dataset)
plt.savefig("Scatterplot.png")
plt.close()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDF', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# Evaluate each model in turn
results = []
names = []
highestMean = 0
bestClassifier = ""
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring= scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    if cv_results.mean() > highestMean:
        bestClassifier=name

print("Best Classifier: " + bestClassifier)
# Compare Algorithms
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig("AlgorithmComparison.png")
plt.close()
# Pick the best Classifier and run a Prediction
if bestClassifier=="LR":
    classifier = LogisticRegression()
elif bestClassifier=="LDA":
    classifier = LinearDiscriminantAnalysis()
elif bestClassifier=="KNN":
    classifier = KNeighborsClassifier()
elif bestClassifier=="CART":
    classifier = DecisionTreeClassifier()
elif bestClassifier=="NB":
    classifier = GaussianNB()
else:
    classifier = SVC()

classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
