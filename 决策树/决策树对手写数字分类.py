from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

digits = load_digits()
features, labels = digits.data, digits.target

#随机0.33作为测试集，其余作为训练集
train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.33)

clf = DecisionTreeClassifier()
clf = clf.fit(train_features,train_labels)
test_predict = clf.predict(test_features)

print('CART 分类树准确率 %.4lf'%accuracy_score(test_labels,test_predict))
