from sklearn import svm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#加载数据集
data = pd.read_csv('data.csv')
pd.set_option('display.max_columns',None)
# print(data.columns)
# print(data.head(5))
# print(data.describe())

#将特征字段分为3组
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])
#数据清洗
#ID列没有用，删除该列
data.drop('id',axis=1,inplace=True)
#将B良性替换为0，M恶性替换为1
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

#将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'],label='Count')
plt.show()
#用热力图呈现 features_mean 字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
#annot = True 显示每个方格的数据
sns.heatmap(corr,annot=True)
plt.show()

#特征选择
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']

#准备训练集和测试集,抽取30%作为测试集，其余为训练集
train,test = train_test_split(data,test_size=0.3)
#抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_y = test['diagnosis']

#对数据进行规范化，避免因维度造成数据误差
#采用Z-score 规范化数据，保证每个特征维度的数据均值为0,方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

#SVM做训练和预测
#创建SVM分类器
model = svm.SVC()
#用训练集做训练
model.fit(train_X,train_y)
#用测试集做预测
prediction = model.predict(test_X)
print('准确率:',metrics.accuracy_score(prediction,test_y))
