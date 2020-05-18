#%%
# =============================================================================
# #Import Libraries
# =============================================================================
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import warnings
import timeit

from keras.models                import Sequential
from keras.layers                import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras                       import optimizers
from sklearn.ensemble            import IsolationForest
from sklearn.linear_model        import LinearRegression,  LassoCV, Lasso, Ridge
from sklearn.tree                import DecisionTreeRegressor
from sklearn.ensemble            import RandomForestRegressor
from sklearn.metrics             import r2_score, mean_squared_error
from sklearn.model_selection     import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn                     import metrics
from sklearn.preprocessing       import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

#%%
# =============================================================================
# # Import Data
# =============================================================================
data = pd.read_excel('Malmo_Data.csv', encoding='utf-8')

#%%
# =============================================================================
# 
# =============================================================================
SalePrice = data['SalePrice']
price = data['SalePrice']
# data.drop(['SalePrice'], axis=1, inplace=True)
data.drop(['ContractDate'], axis=1, inplace=True)
data.drop(['MunicipalityCode'], axis=1, inplace=True)
data.drop(['HouseCategory'], axis=1, inplace=True)

#%%
# =============================================================================
# #Fill Dataset
# =============================================================================

fillwithMode = ['FullAdress', 'PostalTown', 'HouseType', 'HouseTenure']

for i in fillwithMode:
    data[i] = data[i].fillna(data[i].mode()[0])

data['RepoRate'].fillna((data['RepoRate'].mean()), inplace=True)
data['LendingRate'].fillna((data['LendingRate'].mean()), inplace=True)
data['DepositRate'].fillna((data['DepositRate'].mean()), inplace=True)
data['StreetNumber'].fillna(0, inplace=True)

#%%
# =============================================================================
# #Numerical Categorical 
# =============================================================================
dataTypes = data.dtypes

numericalData = dataTypes[(dataTypes == 'int64') | (dataTypes == 'float64')]
categoricalData = dataTypes[dataTypes == 'object']

#%%
# =============================================================================
# #One Hot
# =============================================================================
for i in categoricalData.index:
    labelEncoder = LabelEncoder()
    labelEncoder.fit(list(data[i].values))
    data[i] = labelEncoder.transform(list(data[i].values))
    
#%%
# =============================================================================
# ## Dummy Variables
# =============================================================================

print(data.shape)
data = pd.get_dummies(data)
print(data.shape)

#%%
# =============================================================================
# #%% outliers
# =============================================================================

data = data.join(price)
clf = IsolationForest(max_samples = 'auto', random_state = 42)
clf.fit(data)
y_noano = clf.predict(data)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

data = data.iloc[y_noano[y_noano['Top'] == 1].index.values]
data.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", data.shape[0])

SalePrice = data.SalePrice


data.drop(['SalePrice'], axis=1, inplace=True)

#%%
# =============================================================================
# save time before the training
# =============================================================================
start = timeit.default_timer()

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # #Predictions
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#%%
# =============================================================================
# #Multilinear Prediction
# =============================================================================

SalePrice = np.log10(SalePrice)
X_train, X_test, y_train, y_test = train_test_split(data, SalePrice, test_size=0.25, random_state=0)

linearRegressor = LinearRegression()
linearRegressor.fit(X_train, y_train)

linearPrediction = linearRegressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': linearPrediction})


# Note that for rmse, the lower that value is, the better the fit
linearRMSE = (np.sqrt(mean_squared_error(y_test, linearPrediction)))

# The closer towards 1, the better the fit
linearR2 = r2_score(y_test, linearPrediction)

a_list = list(range(1, len(df["Actual"]) + 1))


mdf = df
mdf  = mdf.sort_index(axis = 0)

#%%
# =============================================================================
# ## Lasso Prediction
# =============================================================================

SalePrice = np.log10(SalePrice)
X_train, X_test, y_train, y_test = train_test_split(data, SalePrice, test_size=0.25, random_state=0)

lassoModel = LassoCV(alphas=[1, 0.1, 0.01, 0.001, 0.0005], selection='random', max_iter=15000).fit(X_train, y_train)

lassoPrediction = lassoModel.predict(X_test)

lassoRMSE = (np.sqrt(mean_squared_error(y_test, lassoPrediction)))

# The closer towards 1, the better the fit
lassoR2 = r2_score(y_test, lassoPrediction)

ldf = pd.DataFrame({'Actual': y_test, 'Predicted': lassoPrediction})
ldf  = ldf.sort_index(axis = 0)

#%%
# =============================================================================
# # Ridge Prediction
# =============================================================================

SalePrice = np.log10(SalePrice)
X_train, X_test, y_train, y_test = train_test_split(data, SalePrice, test_size=0.25, random_state=0)
ridge = Ridge()
ridgeParameters = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False], "solver" : ["auto"]}

gridRidge = GridSearchCV(ridge, ridgeParameters, verbose=1, scoring="r2")
 
gridRidge.fit(X_train, y_train)

print("Best Ridge Model: " + str(gridRidge.best_estimator_))
print("Best Score: " + str(gridRidge.best_score_))

ridge = gridRidge.best_estimator_
ridge.fit(X_train, y_train)
ridgePrediction = ridge.predict(X_test)

ridgeRMSE = (np.sqrt(mean_squared_error(y_test, ridgePrediction)))

# The closer towards 1, the better the fit
ridgeR2 = r2_score(y_test, ridgePrediction)

ridf = pd.DataFrame({'Actual': y_test, 'Predicted': ridgePrediction})
ridf  = ridf.sort_index(axis = 0)

#%%
# =============================================================================
# # Random Forest Prediction
# =============================================================================

SalePrice = np.log10(SalePrice)
X_train, X_test, y_train, y_test = train_test_split(data, SalePrice, test_size=0.25, random_state=0)

MAXDEPTH = 60
randomForest = RandomForestRegressor(n_estimators=1200,   # No of trees in forest
                              criterion = "mse",       
                              max_features = "sqrt",   # no of features to consider for the best split
                              max_depth= MAXDEPTH,     #  maximum depth of the tree
                              min_samples_split= 2,    # minimum number of samples required to split an internal node
                              min_impurity_decrease=0, # Split node if impurity decreases greater than this value.
                              oob_score = True,        # whether to use out-of-bag samples to estimate error on unseen data.
                              n_jobs = -1,             #  No of jobs to run in parallel
                              random_state=0,
                              verbose = 10             # Controls verbosity of process
                              )

randomForest.fit(X_train, y_train)

randomForestPrediction = randomForest.predict(X_test)

randomForestRMSE = (np.sqrt(mean_squared_error(y_test, randomForestPrediction)))

# The closer towards 1, the better the fit
randomForestR2 = r2_score(y_test, randomForestPrediction)

rfdf = pd.DataFrame({'Actual': y_test, 'Predicted': randomForestPrediction})
# rfdf  = rfdf.sort_index(axis = 0)

#%%
# =============================================================================
# ### ANN Prediction
# =============================================================================

SalePrice = np.log10(SalePrice)

X_train, X_test, y_train, y_test = train_test_split(data, SalePrice, test_size=0.25, random_state=2)

# Model
model = Sequential()
model.add(Dense(11,  input_dim = 15 ,kernel_initializer='normal', activation='relu'))
model.add(Dense(7, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu'))
model.add(Dense(1,  kernel_initializer='normal'))
 
def rmse(y_true, y_pred):
 	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def r2(y_true, y_pred):
    SS_res =  backend.sum(backend.square(y_true - y_pred)) 
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + backend.epsilon()) )
 
# Compile model
model.compile(loss='mse', optimizer="rmsprop", metrics=["accuracy", rmse, r2])
history = model.fit(X_train, y_train, validation_split=0.25, epochs=150, batch_size=10, verbose=1)

ANNPrediction = model.predict(X_test)


adf = pd.DataFrame({'Actual': y_test, 'Predicted': ANNPrediction[:,0]})
adf  = adf.sort_index(axis = 0)

fig = plt.figure(figsize=(15,2))
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])

ax1.plot(adf["Actual"])
ax1.plot(adf["Predicted"])
ax1.set_title("ANN")
ax1.set_ylabel("2ere")
ax1.legend(["Actual","Predicted"],  loc="upper right")
ax1.grid()

#%%
# =============================================================================
# History
# =============================================================================
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%%
r = history.history['rmse']
# summarize history for loss
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('model rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['rmse', 'val_rmse'], loc='upper left')
plt.show()

#%%
#evaluate the keras model

loss, accuracy, rmse, r2 = model.evaluate(X_test, y_test)
print(rmse)
print(r2)

#%%
# =============================================================================
# End Time
# =============================================================================
stop = timeit.default_timer()

time = stop -start
print (time)

#%%
# =============================================================================
# Plot all graphs in one figure
# =============================================================================
fig = plt.figure(figsize=(15,2))

ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
ax3 = fig.add_axes([0.1, -0.3, 0.8, 0.4])
ax4 = fig.add_axes([0.1, -0.7, 0.8, 0.4])
ax5 = fig.add_axes([0.1, -1.1, 0.8, 0.4])

ax1.plot(mdf["Actual"])
ax1.plot(mdf["Predicted"])
ax1.set_title("Multiple Linear")
ax1.set_ylabel("Multilinear")
ax1.legend(["Actual","Predicted"],  loc="upper right")
ax1.grid()

ax2.plot(ldf["Actual"])
ax2.plot(ldf["Predicted"])
ax2.set_ylabel("Lasso")
ax2.grid()

ax3.plot(ridf["Actual"])
ax3.plot(ridf["Predicted"])
ax3.set_ylabel("Ridge")
ax3.grid()

ax4.plot(rfdf["Actual"])
ax4.plot(rfdf["Predicted"])
ax4.set_ylabel("Random Forest")
ax4.grid()

ax5.plot(adf["Actual"])
ax5.plot(adf["Predicted"])
ax5.set_ylabel("ANN")
ax5.grid()

#%%
# =============================================================================
# Setting up the correlation attributes
# =============================================================================
correlationMap = data.corr()["SalePrice"]

#%%
# =============================================================================
# ## remove features with low correlation rate
# =============================================================================
e = -1
for i in  correlationMap:
    print(i)
    e = e + 1
    if i > -0.000000000000001: 
        correlationMap = correlationMap.drop(correlationMap.index[e])
        e = e - 1

#%%
# =============================================================================
# # Example data
# =============================================================================   
correlationMap = correlationMap.drop(correlationMap.index[8])

labels = correlationMap.index.values
y_pos = np.arange(len(labels))

plt.rcParams["figure.dpi"] = 144

fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=2)

ax.barh(y_pos, correlationMap)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('SalePrice')
ax.set_ylabel('Featrues')
ax.set_title('Correlation')

plt.show()
#%%
# =============================================================================
# # Prepare Data
# =============================================================================
data.drop(['ContractDate'], axis=1, inplace=True)
data.drop(['MunicipalityCode'], axis=1, inplace=True)
data.drop(['HouseCategory'], axis=1, inplace=True)
#%%
# =============================================================================
# Setting up the correlation attributes
# =============================================================================
correlationMap = data.corr()
correlations = correlationMap["SalePrice"].sort_values(ascending=False)
features = correlations.index[0:10]
#%%
# =============================================================================
# Plot the heatmap
# =============================================================================
import seaborn as sns; sns.set()
uniform_data = data.corr()
plt.figure(figsize=(16,9))
plt.title("Heatmap Correlation of local dataset", fontsize = 25)
plt.ylabel("features", fontsize = 20)
ax = sns.heatmap(uniform_data, annot=True, fmt=".2f", )
