from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer, make_column_transformer
import csv
import numpy as np
data = list(csv.reader(open('final_dataset.csv')))
test = list(csv.reader(open('./targetData/final_dataset.csv')))
for i in range(len(data)):
    for j in range(len(data[i])):
        if(j>=2):
            data[i][j]=float(data[i][j])
            if i in range(len(test)):
                test[i][j]=float(test[i][j])

prep = make_column_transformer((OneHotEncoder(), [0]))
xt = [row[1:-1] for row in data]
temp = prep.fit_transform(np.array(xt)).toarray()
xt = [row[1:] for row in xt]
yt = [row[-1] for row in data]
x = []
y = []
xt_test = [row[1:-1] for row in test]
temp2 = prep.transform(xt_test).toarray()
xt_test = [row[1:] for row in xt_test]
x_teste = []
y_test = []
for i in range(len(xt)):
    if(xt[i][2]==0.0):
        xt[i][2] = xt[i][3]
    x.append([xt[i][0]/xt[i][4], xt[i][2]/xt[i][3], xt[i][3], xt[i][4]])
    y.append([yt[i]/xt[i][3]])
temt=[]
for i in range(len(xt_test)):
    if(xt_test[i][2]==0.0):
        continue
    x_teste.append([xt_test[i][0]/xt_test[i][4], xt_test[i][2]/xt_test[i][3], xt_test[i][3], xt_test[i][4]])
    temt.append(temp2[i])
temp2 = temt
mmx = MinMaxScaler()
mmy = MinMaxScaler()
x = mmx.fit_transform(x)
x_test = mmx.transform(x_teste)
y = mmy.fit_transform(y)
X_train, X_test, y_train, y_test, temp_train, temp_test = train_test_split(x, y, temp, test_size=0.2, random_state=42)
print('SVR')
svr = SVR(C=10).fit(X_train,y_train.ravel())
print(svr.score(X_test,y_test))
prediction1 = svr.predict(X_test)
train_pred1 = svr.predict(X_train)
print(mean_absolute_error(y_test, prediction1))
print(mean_squared_error(y_test, prediction1))
#print(mmy.inverse_transform(y_test))
#print(mmy.inverse_transform(prediction.reshape(-1,1)))
print('NN')
nn = MLPRegressor(max_iter=1000000, hidden_layer_sizes=(200,200)).fit(X_train, y_train.ravel())
print(nn.score(X_test, y_test))
prediction2 = nn.predict(X_test)
train_pred2 = nn.predict(X_train)
print(mean_absolute_error(y_test, prediction2))
print(mean_squared_error(y_test, prediction2))
#print(mmy.inverse_transform(y_test))
#print(mmy.inverse_transform(prediction.reshape(-1,1)))
print('GPR')
gpr = GaussianProcessRegressor(alpha=0.05).fit(X_train, y_train.ravel())
print(gpr.score(X_test, y_test))
prediction3 = gpr.predict(X_test)
train_pred3 = gpr.predict(X_train)
print(mean_absolute_error(y_test, prediction3))
print(mean_squared_error(y_test, prediction3))
#print(mmy.inverse_transform(y_test))
#print(mmy.inverse_transform(prediction.reshape(-1,1)))
print('KNN')
knn = KNeighborsRegressor(n_neighbors=20).fit(X_train, y_train.ravel())
print(knn.score(X_test, y_test))
prediction4 = knn.predict(X_test)
train_pred4 = knn.predict(X_train)
print(mean_absolute_error(y_test, prediction4))
print(mean_squared_error(y_test, prediction4))
#print(mmy.inverse_transform(y_test))
#print(mmy.inverse_transform(prediction.reshape(-1,1)))
prediction = (prediction1+prediction2+prediction3+prediction4)/4.0
print(mean_absolute_error(y_test, prediction))
print(mean_squared_error(y_test, prediction))
train_preds = (train_pred1+train_pred2+train_pred3+train_pred4)/4.0
smoothen_data = np.concatenate([temp_train, train_preds.reshape(-1,1)], axis=1)
svr2 = SVR().fit(smoothen_data, y_train)
final = svr2.predict(np.concatenate([temp_test, prediction.reshape(-1,1)],axis=1))
print(mean_absolute_error(y_test, final))
print(mean_squared_error(y_test, final))

model1 = SVR(C=10).fit(x,y)
ans1 = model1.predict(x_test)
model2 = MLPRegressor(max_iter=1000000, hidden_layer_sizes=(200,200,)).fit(x,y)
ans2 = model2.predict(x_test)
model3 = GaussianProcessRegressor(alpha=0.05).fit(x,y)
ans3 = model3.predict(x_test)
model4 = KNeighborsRegressor(n_neighbors=20).fit(x,y)
ans4 = model4.predict(x_test)
print(len(temp2), ((ans1+ans2+ans3.T+ans4.T)/4.0).shape)
ans = np.concatenate([temp2, ((ans1+ans2+ans3.T+ans4.T)/4.0).reshape(-1,1)], axis=1)
ans = svr2.predict(ans)
ans = mmy.inverse_transform(ans.reshape(-1,1))
j=0
values = []
for i in range(len(xt_test)):
    if(xt_test[i][2]==0.0):
        values.append(0)
    else:
        values.append((ans[j]*x_teste[j][2]*x_teste[j][1])[0])
        j+=1
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
su = sum(values)
factor = 543.0/su

j=0

for i in range(len(xt_test)):
    if(xt_test[i][2]==0.0):
        print(test[i][0]+'    '+test[i][1]+'    '+'0/0')
    else:
        print(test[i][0]+'    '+test[i][1]+'    '+str(values[i]*factor)+'/'+str(x_teste[j][2]*x_teste[j][1]))
        j+=1

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
 
objects = [row[1] for row in test]
y_pos = np.arange(len(objects))
performance = [v*factor for v in values]
fig, ax = plt.subplots()
ax.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Ratio of Seats')
plt.title('Party')
for i, v in enumerate(performance):
    ax.text(v + 3, i-0.2 , str(int(round(v))), color='blue')
plt.tight_layout()
plt.show()
