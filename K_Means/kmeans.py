from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
#输入数据
data = pd.read_csv('data.csv',encoding='gbk')
train_x = data[['2019年国际排名','2018世界杯','2015亚洲杯']]
df = pd.DataFrame(train_x)
kmeans = KMeans(n_clusters=3)
#规范化到[0,1]
mm = preprocessing.MinMaxScaler()
train_x = mm.fit_transform(train_x)
#kmeans算法
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
#合并聚合结果，插入到原数据中
res = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
res.rename({0:u'聚类'},axis=1,inplace=True)
res.sort_values('聚类',inplace=True)
print(res)
