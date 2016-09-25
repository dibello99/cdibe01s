
# coding: utf-8

# In[ ]:

#Chris diBello CSC570R Boston Housing Assignment
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

bean = datasets.load_boston()
print bean.DESCR
#get dataset
def load_boston():
    scaler = StandardScaler()
    boston = datasets.load_boston()
    X=boston.data
    y=boston.target
    X = scaler.fit_transform(X)
    return train_test_split(X,y)

X_train, X_test, y_train, y_test = load_boston()

X_train.shape
coef = 3 * np.random.randn(5)
#coef = 5
#build linear regresion
clf = LinearRegression()
clf.fit(X_train, y_train)
#find R2 score
print 'Performance Measurement'
print 'Print R2 Score'
print r2_score(y_test, clf.predict(X_test))
#find MSE score
print 'Print MSE Score'
print mean_squared_error(y_test, clf.predict(X_test))

zip (y_test, clf.predict(X_test))
# Lasso


alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print 'Print Lasso'
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)
# ElasticNet

#plot enet and lasso graph
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print 'Print Enet'
print(enet)
print("r^2 on test data : %f" % r2_score_enet)
print 'enet.coef_'
print enet.coef_
print 'lasso.coef_'
print lasso.coef_
print 'coef'
print coef

get_ipython().magic(u'matplotlib inline')
plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(coef, '--', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()
#Optimize R2 score
print 'Optimization'
alpha = 0.3
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print 'Optimized R2 score'
print r2_score(y_test, y_pred_lasso)
#plot enet and lasso graph
get_ipython().magic(u'matplotlib inline')
plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(coef, '--', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()
#Optimize R2 score
print 'Optimization'
alpha = 0.01
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print 'Optimized R2 score'
print r2_score(y_test, y_pred_lasso)
#plot enet and lasso graph
get_ipython().magic(u'matplotlib inline')
plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(coef, '--', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()
print 'Optimization'
alpha = 0.3
lasso = Lasso(alpha=alpha)
#Optimize MSE score
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = mean_squared_error(y_test, y_pred_lasso)
print 'Optimized MSE score'
print mean_squared_error(y_test, y_pred_lasso)
#plot enet and lasso graph
get_ipython().magic(u'matplotlib inline')
plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(coef, '--', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()
print 'Optimization'
alpha = 0.01
lasso = Lasso(alpha=alpha)
#Optimize MSE score
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = mean_squared_error(y_test, y_pred_lasso)
print 'Optimized MSE score'
print mean_squared_error(y_test, y_pred_lasso)
#plot enet and lasso graph

get_ipython().magic(u'matplotlib inline')
plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(coef, '--', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()

