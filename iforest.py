import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=100,
                      n_jobs=-1,
                      contamination=0.3,
                      verbose=2,
                      ) #contamination是异常值的比例
data = pd.read_csv('data.csv')
clf.fit(data.values)
scores_pred = clf.decision_function(data.values) #返回异常的评分，值越小越可能是异常
output = clf.predict(data.values) #输出1表示正常-1表示异常
#也可以分段检测
all = []
shape = data.shape[0]
batch = 10**6
for i in range(shape/batch+1):
    start = i * batch
    end = (i+1) * batch
    test = data[start:end]
    # 预测
    pred = clf.predict(test)
    all.extend(pred)