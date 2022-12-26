import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('Carseats.csv')
df1 = df.copy()


#Problem 1 -- Data statistics
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
sns.histplot(df["Sales"],ax=axes[0,0])
sns.histplot(df["CompPrice"],ax=axes[0,1])
sns.histplot(df["Income"],ax=axes[0,2])
sns.histplot(df["Advertising"],ax=axes[0,3])
sns.histplot(df["Population"],ax=axes[1,0])
sns.histplot(df["Price"],ax=axes[1,1])
sns.countplot(df["ShelveLoc"],ax=axes[1,2])
sns.histplot(df["Age"],ax=axes[1,3])
sns.histplot(df["Education"],ax=axes[2,0])
sns.countplot(df["Urban"],ax=axes[2,1])
sns.countplot(df["US"],ax=axes[2,2])
plt.suptitle('Problem1: data statistics')
plt.title('statistics')
plt.show()
# relation between target variable and other features
sns.jointplot(x = df['CompPrice'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['Income'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['Advertising'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['Population'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['Price'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['Age'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['ShelveLoc'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['Education'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['Urban'],y = df['Sales'],kind = 'scatter')
sns.jointplot(x = df['US'],y = df['Sales'],kind = 'scatter')
plt.show()

#binary encode for the features 'Urban' and 'US'
df1.loc[df['Urban']=='Yes','Urban'] = 1
df1.loc[df['Urban']=='No','Urban'] = -1
df1.loc[df['US']=='Yes','US'] = 1
df1.loc[df['US']=='No','US'] = -1
#one hot encode for 'ShelveLoc'
sl = OneHotEncoder().fit_transform(np.array(df['ShelveLoc']).reshape(-1,1)).toarray()
df1['ShelveLoc1'] = sl[:,0]
df1['ShelveLoc2'] = sl[:,1]
df1['ShelveLoc3'] = sl[:,2]
df1 = df1.drop('ShelveLoc',axis=1)

# split data (first 300 rows as train, last 100 as test)
features = df1.iloc[:,1:]
targets = df1.Sales
X_train,X_test = np.array(features[:300]),np.array(features[300:])
Y_train, Y_test = np.array(targets[:300]),np.array(targets[300:])

# a function to calculate the sum squared error
def SSE(y,y_predict):
    sse = 0
    for i in range(len(y)):
        sse += (y[i]-y_predict[i])**2
    return sse

#Problem 2 -- Decision tree
# fit with dafault parameters
reg = tree.DecisionTreeRegressor(criterion="mse",random_state=12)
reg.fit(X_train,Y_train)
predict = reg.predict(X_test)
print('test loss of fit with default parameters(decision tree):'+ str(SSE(Y_test,predict)))
print('Regression of Sales of test data with default parameters(decision tree):')
print(predict)


# plot the train/test errors with respect to different maximum depths
train_losses = []
test_losses = []
max_depths = [i for i in range(1,21)]
for max_depth in max_depths:
    reg = tree.DecisionTreeRegressor(criterion="mse",max_depth=max_depth,random_state=12)
    reg.fit(X_train,Y_train)
    predict_train = reg.predict(X_train)
    predict_test = reg.predict(X_test)
    train_losses.append(SSE(predict_train,Y_train))
    test_losses.append(SSE(predict_test,Y_test))
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
md_train = pd.DataFrame({'max depth':max_depths,'train loss':train_losses})
sns.lineplot(x='max depth',y='train loss',data = md_train,ax=axes[0])
md_test = pd.DataFrame({'max depth':max_depths,'test loss':test_losses})
sns.lineplot(x='max depth',y='test loss',data = md_test,ax=axes[1])
plt.suptitle('Problem2: losses w.r.t. max depth')
plt.show()


# plot the train/test errors w.r.t. different least node sizes
train_losses = []
test_losses = []
min_samples_leaves = [i for i in range(1,11)]
for node_size in min_samples_leaves:
    reg = tree.DecisionTreeRegressor(criterion="mse",min_samples_leaf=node_size,random_state=12)
    reg.fit(X_train,Y_train)
    predict_train = reg.predict(X_train)
    predict_test = reg.predict(X_test)
    train_losses.append(SSE(predict_train,Y_train))
    test_losses.append(SSE(predict_test,Y_test))
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
ln_train = pd.DataFrame({'least node size':min_samples_leaves,'train loss':train_losses})
sns.lineplot(x='least node size',y='train loss',data = ln_train,ax=axes[0])
ln_test = pd.DataFrame({'least node size':min_samples_leaves,'test loss':test_losses})
sns.lineplot(x='least node size',y='test loss',data = ln_test,ax=axes[1])
plt.suptitle('Problem2: losses w.r.t. least node size')
plt.show()


#plotting the learned tree
reg = tree.DecisionTreeRegressor(criterion="mse")
reg.fit(X_train,Y_train)
tree.plot_tree(reg,max_depth=3,filled=True)
plt.title('Problem2:plot learned tree with max depth 3(easy to visualize)') # easy to visualize
plt.show()



#Problem3 : bagging
reg = BaggingRegressor(base_estimator=tree.DecisionTreeRegressor(),random_state=12)
reg.fit(X_train,Y_train)
predict = reg.predict(X_test)
print('test loss of fit with default parameters(bagging):'+str(SSE(predict,Y_test)))
print('Regression of Sales of test data with default parameters(bagging):')
print(predict)


# plot the train/test errors w.r.t. different depth
train_losses = []
test_losses = []
max_depths = [i for i in range(1,21)]
for max_depth in max_depths:
    reg = BaggingRegressor(tree.DecisionTreeRegressor(criterion="mse",max_depth=max_depth),random_state=12)
    reg.fit(X_train,Y_train)
    predict_train = reg.predict(X_train)
    predict_test = reg.predict(X_test)
    train_losses.append(SSE(predict_train,Y_train))
    test_losses.append(SSE(predict_test,Y_test))
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
md_train = pd.DataFrame({'depth':max_depths,'train loss':train_losses})
sns.lineplot(x='depth',y='train loss',data = md_train,ax=axes[0])
md_test = pd.DataFrame({'depth':max_depths,'test loss':test_losses})
sns.lineplot(x='depth',y='test loss',data = md_test,ax=axes[1])
plt.suptitle('Problem3: losses w.r.t. depth')
plt.show()


# plot the train/test errors w.r.t. different number of trees
train_losses = []
test_losses = []
num_of_learner = [i for i in range(1,21)]
for num in num_of_learner:
    reg = BaggingRegressor(tree.DecisionTreeRegressor(criterion="mse"),n_estimators=num,random_state=12)
    reg.fit(X_train,Y_train)
    predict_train = reg.predict(X_train)
    predict_test = reg.predict(X_test)
    train_losses.append(SSE(predict_train,Y_train))
    test_losses.append(SSE(predict_test,Y_test))
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
nl_train = pd.DataFrame({'number of trees':num_of_learner,'train loss':train_losses})
sns.lineplot(x='number of trees',y='train loss',data = nl_train,ax=axes[0])
nl_test = pd.DataFrame({'number of trees':num_of_learner,'test loss':test_losses})
sns.lineplot(x='number of trees',y='test loss',data = nl_test,ax=axes[1])
plt.suptitle('Problem3: losses w.r.t. number of trees')
plt.show()



#Problem 4: random forest
reg = RandomForestRegressor(random_state=12)
reg.fit(X_train,Y_train)
predict = reg.predict(X_test)
print('test loss of fit with default parameters(random forest):'+str(SSE(predict,Y_test)))
print('Regression of Sales of test data with default parameters(random forest):')
print(predict)


# plot the train/test errors w.r.t. different number of trees
train_losses = []
test_losses = []
num_of_learner = [10*i for i in range(1,11)]
for num in num_of_learner:
    reg = RandomForestRegressor(n_estimators=num,random_state=12)
    reg.fit(X_train,Y_train)
    predict_train = reg.predict(X_train)
    predict_test = reg.predict(X_test)
    train_losses.append(SSE(predict_train,Y_train))
    test_losses.append(SSE(predict_test,Y_test))
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
nl_train = pd.DataFrame({'number of trees':num_of_learner,'train loss':train_losses})
sns.lineplot(x='number of trees',y='train loss',data = nl_train,ax=axes[0])
nl_test = pd.DataFrame({'number of trees':num_of_learner,'test loss':test_losses})
sns.lineplot(x='number of trees',y='test loss',data = nl_test,ax=axes[1])
plt.suptitle('Problem4: losses w.r.t. number of trees')
plt.show()


# plot the train/test errors w.r.t. different values of m
train_losses = []
test_losses = []
max_features = [i for i in range(1,11)]
for m in max_features:
    reg = RandomForestRegressor(max_features=m,random_state=12)
    reg.fit(X_train,Y_train)
    predict_train = reg.predict(X_train)
    predict_test = reg.predict(X_test)
    train_losses.append(SSE(predict_train,Y_train))
    test_losses.append(SSE(predict_test,Y_test))
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
mf_train = pd.DataFrame({'value of m':max_features,'train loss':train_losses})
sns.lineplot(x='value of m',y='train loss',data = mf_train,ax=axes[0])
mf_test = pd.DataFrame({'value of m':max_features,'test loss':test_losses})
sns.lineplot(x='value of m',y='test loss',data = mf_test,ax=axes[1])
plt.suptitle('Problem4: losses w.r.t. values of m')
plt.show()



#Problem5
def get_bias(y,h_mean):
    return np.mean((y-h_mean)**2)

def get_var(estimators,h_mean,X_test):
    s = 0
    for estimator in estimators:
        h = estimator.predict(X_test)
        s += np.mean((h-h_mean)**2)
    return s/len(estimators)

num_of_trees = [10*i for i in range(1,11)]
bias2 = []
variances = []
for num in num_of_trees:
    reg = RandomForestRegressor(n_estimators=num,random_state=12)
    reg.fit(X_train,Y_train)
    predict = reg.predict(X_test)
    bias2.append(get_bias(Y_test,predict))
    variances.append(get_var(reg.estimators_,predict,X_test))
plt.plot(num_of_trees,bias2)
plt.xlabel('number of trees')
plt.ylabel('bias^2')
plt.title('Problem5: relationship between bias^2 and different number of trees')
plt.show()
plt.plot(num_of_trees,variances)
plt.xlabel('number of trees')
plt.ylabel('variance')
plt.title('Problem5: relationship between variance and different number of trees')
plt.show()