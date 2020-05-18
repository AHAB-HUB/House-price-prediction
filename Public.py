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

trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
allData = pd.concat((trainData, testData)).reset_index(drop=True)
allData.drop(['SalePrice'], axis=1, inplace=True)
###

id=trainData["Id"]
ID = testData["Id"]
SalePrice = trainData['SalePrice']

#%%
# =============================================================================
# # Setting up the correlation attributes
# =============================================================================

correlationMap = trainData.corr()["SalePrice"]
f, ax = plt.subplots(figsize=(30, 19))
sns.set(font_scale=1.50)
sns.heatmap(correlationMap, square=True,cmap='coolwarm')

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(30, 19))

#%%
# =============================================================================
# ## remove features with low correlation rate
# =============================================================================
e = -1
for i in  correlationMap:
    print(i)
    e = e + 1
    if i > 0: 
        correlationMap = correlationMap.drop(correlationMap.index[e])
        e = e - 1
        
#correlationMap = correlationMap.drop(correlationMap.index[27])

#%%
# Example data

plt.rcParams["figure.dpi"] = 144
labels = correlationMap.index.values
y_pos = np.arange(len(labels))

fig, ax = plt.subplots()

ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=1)

ax.barh(y_pos, correlationMap)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('SalePrice')
ax.set_ylabel('Featrues')
ax.set_title('Negative Correlation')

plt.show()

#%%
# =============================================================================
# # Correlations
# =============================================================================

correlations = correlationMap["SalePrice"].sort_values(ascending=False)
features = correlations.index[0:10]
###

#%%
# =============================================================================
# # Plotting important Correlated Features (HEAT MAP)
# =============================================================================
corr = trainData.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

rel_vars = corr[(corr.SalePrice < -0.024)]
rel_cols = list(rel_vars.index.values)
rel_cols.append('SalePrice')
corr2 = trainData[rel_cols].corr()

sns.heatmap(corr2, annot=True, annot_kws={'size':15}, fmt='.2f')

#%%
# =============================================================================
# # Dropping ID Column
# =============================================================================

trainData.drop(['Id'], axis=1, inplace=True)
testData.drop(['Id'], axis=1, inplace=True)
allData.drop(['Id'], axis=1, inplace=True)

#%%
# =============================================================================
# ### Removing SalePrice
# =============================================================================
SalePrice = trainData['SalePrice']
price = trainData['SalePrice']
trainData.drop(['SalePrice'], axis=1, inplace=True)
###

#%%
# =============================================================================
# # Checking NAN Values

# =============================================================================

trainNull = pd.isnull(trainData).sum()
testNull = pd.isnull(testData).sum()
allNull = pd.concat([trainNull, testNull], axis=1, keys=['Training Data', 'Testing Data'])

#%%
# =============================================================================
# Checking
# =============================================================================

manyNull = allNull[allNull.sum(axis=1) > 200]
fewNull = allNull[(allNull.sum(axis=1) > 0) & (allNull.sum(axis=1) < 200)]

#%%
# =============================================================================
# # Fill with None
# =============================================================================

fillWithNone = ["Alley","Electrical", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType"]
for i in fillWithNone:
    trainData[i].fillna("None", inplace=True)
    testData[i].fillna("None", inplace=True)
    allData[i].fillna("None", inplace=True)

#%%
# =============================================================================
# # Dropping Lot Frontage
# =============================================================================
trainData.drop("LotFrontage", axis=1, inplace=True)
testData.drop("LotFrontage", axis=1, inplace=True)
allData.drop("LotFrontage", axis=1, inplace=True)


#%%
# =============================================================================
## Mean Imputing 
# =============================================================================
cols = ['GarageYrBlt', 'MasVnrArea']
for i in cols:
    trainData[i].fillna(trainData[i].mean(), inplace=True)
    testData[i].fillna(testData[i].mean(), inplace=True)
    allData[i].fillna(allData[i].mean(), inplace=True)

#%%
# =============================================================================
# # Numerical & Categorical Values of Train and Test
# =============================================================================

trainTypes = trainData.dtypes
trainNumerical = trainTypes[(trainTypes == 'int64') | (trainTypes == 'float64')]
trainCategorical = trainTypes[trainTypes == 'object']

testTypes = testData.dtypes
testNumerical = testTypes[(testTypes  == 'int64') | (testTypes  == 'float64')]
testCategorical = testTypes[testTypes  == 'object']

allDataTypes = allData.dtypes
allDataCategorical = allDataTypes[allDataTypes == 'object']

#%%
# =============================================================================
# # Numerical Imputing
# =============================================================================

trainNumericalList = list(trainNumerical.index)
##testNumericalList = list(testNumerical.index)

imputeAllNumerical = trainNumericalList + testNumericalList

for i in imputeAllNumerical:
    trainData[i].fillna(trainData[i].mean(), inplace=True)
    ##testData[i].fillna(testData[i].mean(), inplace=True)
    ##allData[i].fillna(all
    #Data[i].mean(), inplace=True)


#%%
# =============================================================================
# # Categorical Imputing
# =============================================================================

trainCategoricalList = list(trainCategorical.index)
testCategoricalList = list(testCategorical.index)

fillCategorical = []

for i in trainCategoricalList:
    if i in list(fewNull.index):
        fillCategorical.append(i)

#%%
# =============================================================================
# # Most Common Categorical Values
# =============================================================================

def mostCatValues(catList):
    cateList = list(catList)
    return max(set(cateList), key=cateList.count)

mostCommon = []
for i in fillCategorical:
    mostCommon.append(mostCatValues(allData[i]))

#%%
# =============================================================================
# # Create Dictionary of most common
# =============================================================================

mostCommonDic = {fillCategorical[0]: [mostCommon[0]], fillCategorical[1]: [mostCommon[1]], fillCategorical[2]: [mostCommon[2]], fillCategorical[3]: [mostCommon[3]], fillCategorical[4]: [mostCommon[4]], fillCategorical[5]: [mostCommon[5]], fillCategorical[6]: [mostCommon[6]], fillCategorical[7]: [mostCommon[7]], fillCategorical[8]: [mostCommon[8]]}

#%%

# =============================================================================
# # Fill With most Common
# =============================================================================
mk = 0
for i in fillCategorical:
    trainData[i].fillna(mostCommon[mk], inplace=True)
    testData[i].fillna(mostCommon[mk], inplace=True)
    allData[i].fillna(mostCommon[mk], inplace=True)
    mk += 1


#%%
# =============================================================================
# # Create New Features
# =============================================================================

trainData['TotalSF'] = trainData['TotalBsmtSF'] + trainData['1stFlrSF'] + trainData['2ndFlrSF']
trainData['Total_Bathrooms'] = (trainData['FullBath'] + (0.5 * trainData['HalfBath']) +
                                trainData['BsmtFullBath'] + (0.5 * trainData['BsmtHalfBath']))

trainData['Total_porch_sf'] = (trainData['OpenPorchSF'] + trainData['3SsnPorch'] +
                              trainData['EnclosedPorch'] + trainData['ScreenPorch'] +
                              trainData['WoodDeckSF'])

testData['TotalSF'] = testData['TotalBsmtSF'] + testData['1stFlrSF'] + testData['2ndFlrSF']
testData['Total_Bathrooms'] = (testData['FullBath'] + (0.5 * testData['HalfBath']) +
                                testData['BsmtFullBath'] + (0.5 * testData['BsmtHalfBath']))

testData['Total_porch_sf'] = (testData['OpenPorchSF'] + testData['3SsnPorch'] +
                              testData['EnclosedPorch'] + testData['ScreenPorch'] +
                              testData['WoodDeckSF'])

allData['TotalSF'] = allData['TotalBsmtSF'] + allData['1stFlrSF'] + allData['2ndFlrSF']
allData['Total_Bathrooms'] = (allData['FullBath'] + (0.5 * allData['HalfBath']) +
                                allData['BsmtFullBath'] + (0.5 * allData['BsmtHalfBath']))

allData['Total_porch_sf'] = (allData['OpenPorchSF'] + allData['3SsnPorch'] +
                              allData['EnclosedPorch'] + allData['ScreenPorch'] +
                              allData['WoodDeckSF'])

trainData['haspool'] = trainData['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
trainData['has2ndfloor'] = trainData['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
trainData['hasgarage'] = trainData['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
trainData['hasbsmt'] = trainData['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
trainData['hasfireplace'] = trainData['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

testData['haspool'] = testData['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
testData['has2ndfloor'] = testData['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
testData['hasgarage'] = testData['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
testData['hasbsmt'] = testData['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
testData['hasfireplace'] = testData['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

allData['haspool'] = allData['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
allData['has2ndfloor'] = allData['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
allData['hasgarage'] = allData['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
allData['hasbsmt'] = allData['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
allData['hasfireplace'] = allData['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

removeCols = ['1stFlrSF',  'FullBath', 
              'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF', 
              '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 
              'PoolArea', '2ndFlrSF', 'GarageArea', 'TotalBsmtSF', 
              'Fireplaces']

# for i in removeCols:
#     trainData.drop([i], axis=1, inplace=True)
#     testData.drop([i], axis=1, inplace=True)
#     allData.drop([i], axis=1, inplace=True)


#%%
# =============================================================================
# # One Hot Encoding
# =============================================================================

trainingCatValues = list(trainCategorical.index)
testingCatValues = list(testCategorical.index)
totalCatValues = list(allDataCategorical.index)

for i in trainingCatValues:
    featureSet = set(trainData[i])
    for j in featureSet:
        featureList = list(featureSet)
        trainData.loc[trainData[i]==j, i] = featureList.index(j)

for i in testingCatValues:
    featureSet2 = set(testData[i])
    for j in featureSet2:
        featureList2 = list(featureSet2)
        testData.loc[testData[i]==j, i] = featureList2.index(j)

for i in totalCatValues:
    featureSet3 = set(allData[i])
    for j in featureSet3:
        featureList3 = list(featureSet3)
        allData.loc[allData[i]==j, i] = featureList3.index(j)

#%% 
# =============================================================================
# ##outliers
# =============================================================================

trainData = trainData.join(price)
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(trainData)
y_noano = clf.predict(trainData)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

trainData = trainData.iloc[y_noano[y_noano['Top'] == 1].index.values]
trainData.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", trainData.shape[0])

SalePrice = trainData.SalePrice

trainData.drop(['SalePrice'], axis=1, inplace=True)

#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # # Predictions
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#   #Multiple Linear Regression
# =============================================================================

# start = timeit.default_timer()

# SalePrice = np.log10(SalePrice)

# X_train, X_test, y_train, y_test = train_test_split(trainData, SalePrice, test_size=0.25, random_state=0)

# linearRegressor = LinearRegression()
# linearRegressor.fit(X_train, y_train)

# linearPrediction = linearRegressor.predict(X_test)

# df = pd.DataFrame({'Actual': y_test, 'Predicted': linearPrediction})

# # Note that for rmse, the lower that value is, the better the fit
# linearRMSE = (np.sqrt(mean_squared_error(y_test, linearPrediction)))

# # The closer towards 1, the better the fit
# linearR2 = r2_score(y_test, linearPrediction)

# mdf = df.sort_index(axis = 0) 

# stop = timeit.default_timer()
# time = stop -  start
# print (time)


#%%
# =============================================================================
# # # Ridge Regression
# =============================================================================

# start = timeit.default_timer()

# SalePrice = np.log10(SalePrice)
# X_train, X_test, y_train, y_test = train_test_split(trainData, SalePrice, test_size=0.25, random_state=0)
# ridge = Ridge()
# ridgeParameters = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False], "solver" : ["auto"]}

# gridRidge = GridSearchCV(ridge, ridgeParameters, verbose=1, scoring="r2")
 
# gridRidge.fit(X_train, y_train)

# print("Best Ridge Model: " + str(gridRidge.best_estimator_))
# print("Best Score: " + str(gridRidge.best_score_))

# ridge = gridRidge.best_estimator_
# ridge.fit(X_train, y_train)
# ridgePrediction = ridge.predict(X_test)

# ridgeRMSE = (np.sqrt(mean_squared_error(y_test, ridgePrediction)))

# # The closer towards 1, the better the fit
# ridgeR2 = r2_score(y_test, ridgePrediction)

# ridf = pd.DataFrame({'Actual': y_test, 'Predicted': ridgePrediction})
# ridf  = ridf.sort_index(axis = 0)


# stop = timeit.default_timer()

# time = stop -  start
# print (time)

#%%
# =============================================================================
# # Random Forest
# =============================================================================

# start = timeit.default_timer()
# SalePrice = np.log10(SalePrice)
# X_train, X_test, y_train, y_test = train_test_split(trainData, SalePrice, test_size=0.25, random_state=0)

# MAXDEPTH = 60
# randomForest = RandomForestRegressor(n_estimators=1200,   # No of trees in forest
#                               criterion = "mse",       
#                               max_features = "sqrt",   # no of features to consider for the best split
#                               max_depth= MAXDEPTH,     #  maximum depth of the tree
#                               min_samples_split= 2,    # minimum number of samples required to split an internal node
#                               min_impurity_decrease=0, # Split node if impurity decreases greater than this value.
#                               oob_score = True,        # whether to use out-of-bag samples to estimate error on unseen data.
#                               n_jobs = -1,             #  No of jobs to run in parallel
#                               random_state=0,
#                               verbose = 10             # Controls verbosity of process
#                               )

# randomForest.fit(X_train, y_train)

# randomForestPrediction = randomForest.predict(X_test)

# randomForestRMSE = (np.sqrt(mean_squared_error(y_test, randomForestPrediction)))

# # The closer towards 1, the better the fit
# randomForestR2 = r2_score(y_test, randomForestPrediction)
# rfdf = pd.DataFrame({'Actual': y_test, 'Predicted': randomForestPrediction})
# rfdf  = rfdf.sort_index(axis = 0)

# stop = timeit.default_timer()

# time = stop -  start
# print (time)

#%%
# =============================================================================
# # Lasso
# =============================================================================

# start = timeit.default_timer()

# SalePrice = np.log10(SalePrice)
# X_train, X_test, y_train, y_test = train_test_split(trainData, SalePrice, test_size=0.25, random_state=0)

# lassoModel = LassoCV(alphas=[1, 0.1, 0.01, 0.001, 0.0005], selection='random', max_iter=15000).fit(X_train, y_train)

# lassoPrediction = lassoModel.predict(X_test)

# lassoRMSE = (np.sqrt(mean_squared_error(y_test, lassoPrediction)))

# # The closer towards 1, the better the fit
# lassoR2 = r2_score(y_test, lassoPrediction)
# stop = timeit.default_timer()

# ldf = pd.DataFrame({'Actual': y_test, 'Predicted': lassoPrediction})
# ldf  = ldf.sort_index(axis = 0)

# time = stop -  start
# print (time)

#%%
# =============================================================================
# # #ANN
# =============================================================================

# prepro_y = MinMaxScaler()
# prepro_y.fit(SalePrice)

# SalePrice = np.log10(SalePrice)

# X_train, X_test, y_train, y_test = train_test_split(trainData, SalePrice, test_size=0.25, random_state=4)

# # Model
# model = Sequential()
# model.add(Dense(70,  input_dim = 86,kernel_initializer='normal', activation='relu'))
# model.add(Dense(60,   kernel_initializer='normal', activation='relu'))
# model.add(Dense(50,   kernel_initializer='normal', activation='relu'))
# model.add(Dense(25,   kernel_initializer='normal', activation='relu'))
# model.add(Dense(1,    kernel_initializer='normal'))
 
# def rmse(y_true, y_pred):
#  	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# def r2(y_true, y_pred):
#     SS_res =  backend.sum(backend.square(y_true - y_pred)) 
#     SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true))) 
#     return ( 1 - SS_res/(SS_tot + backend.epsilon()) )
 
# # Compile model
# model.compile(loss='mse', optimizer="rmsprop", metrics=["accuracy", rmse, r2])

# history = model.fit(X_train, y_train, validation_split=0.25, epochs=150, batch_size=10, verbose=1)

# ANNPrediction = model.predict(X_test)

# adf = pd.DataFrame({'Actual': y_test, 'Predicted': ANNPrediction[:,0]})
# adf  = adf.sort_index(axis = 0)


#%%
# =============================================================================
# # #list all data in history
# =============================================================================

#print(history.history.keys())
# summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # summarize history for loss
# plt.plot(history.history['rmse'])
# plt.plot(history.history['val_rmse'])
# plt.title('model rmse')
# plt.ylabel('rmse')
# plt.xlabel('epoch')
# plt.legend(['rmse', 'val_rmse'], loc='upper left')
# plt.show()

#%%
# =============================================================================
# #evaluate the keras model
# =============================================================================

# loss, accuracy, rmse, r2 = model.evaluate(X_test, y_test)
# print(rmse)
# print(r2)


#%%
# =============================================================================
# End time 
# =============================================================================
# stop = timeit.default_timer()

# time = stop -  start
# print (time)

#%%
# =============================================================================
# ##plotting all graphs into one figure
# =============================================================================

# fig = plt.figure(figsize=(15,2))

# ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
# ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
# ax3 = fig.add_axes([0.1, -0.3, 0.8, 0.4])
# ax4 = fig.add_axes([0.1, -0.7, 0.8, 0.4])
# ax5 = fig.add_axes([0.1, -1.1, 0.8, 0.4])

# ax1.plot(mdf["Actual"])
# ax1.plot(mdf["Predicted"])
# ax1.set_title("Multiple Linear")
# ax1.set_ylabel("Multilinear")
# ax1.legend(["Actual","Predicted"],  loc="upper right")
# ax1.grid()

# ax2.plot(ldf["Actual"])
# ax2.plot(ldf["Predicted"])
# ax2.set_ylabel("Lasso")
# ax2.grid()

# ax3.plot(ridf["Actual"])
# ax3.plot(ridf["Predicted"])
# ax3.set_ylabel("Ridge")
# ax3.grid()

# ax4.plot(rfdf["Actual"])
# ax4.plot(rfdf["Predicted"])
# ax4.set_ylabel("Random Forest")
# ax4.grid()

# ax5.plot(adf["Actual"])
# ax5.plot(adf["Predicted"])
# ax5.set_ylabel("ANN")
# ax5.grid()