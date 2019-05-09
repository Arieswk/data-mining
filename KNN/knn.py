#先探索数据，分别使用KNN,SVM,多项式朴素贝叶斯以及CART决策树对手写数字进行分类，并比较四种算法的准确率(accuracy)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

#加载数据
digits = load_digits()
data = digits.data
# 数据探索
# print(data.shape)
# #查看第一幅图像
# print(digits.images[0])
# #第一幅图像代表的数字含义
# print(digits.target[0])
# #将第一幅图像显示出来
# plt.gray()
# plt.imshow(digits.images[0])
# plt.show()

#分割数据，将25%的数据作为测试集，其余作为训练集
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,test_size=0.25)
#采取Z-score规范化
ss = StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

#创建KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_x,train_y)
predict_y = knn.predict(test_ss_x)
print('KNN准确率：%.4lf' % metrics.accuracy_score(predict_y,test_y))

#创建SVM分类器
svm = SVC()
svm.fit(train_ss_x,train_y)
predict_y = svm.predict(test_ss_x)
print('SVM准确率：%.4lf' % metrics.accuracy_score(predict_y,test_y))

#采用Min-Max规范化
mm = MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)
#创建Naive Bayes 分类器
mnb = MultinomialNB()
mnb.fit(train_mm_x,train_y)
predict_y = mnb.predict(test_mm_x)
print('多项式朴素贝叶斯准确率：%.4lf' % metrics.accuracy_score(predict_y,test_y))
#创建CART决策树
dtc = DecisionTreeClassifier()
dtc.fit(train_mm_x,train_y)
predict_y = dtc.predict(test_mm_x)
print('CART决策树准确率：%.4lf' % metrics.accuracy_score(predict_y,test_y))
