# =============================================================================
# # Libraries
# =============================================================================

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import warnings
import timeit
import itertools

from sklearn.ensemble        import IsolationForest
from sklearn.linear_model    import LinearRegression, LassoCV, Lasso, Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn                 import metrics
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from keras.models            import Sequential
from keras.layers            import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras                  import optimizers, backend
from numpy.random import seed

warnings.filterwarnings("ignore")


#%%
# =============================================================================
# # Import Data
# =============================================================================
trainData = pd.read_csv('train.csv')

#%%
# =============================================================================
# # Get ID and SalePrice.  Drop ID and SalePrice from dataset
# =============================================================================

id=trainData["Id"]
trainData.drop(['Id'], axis=1, inplace=True)
SalePrice = trainData['SalePrice']
price = trainData['SalePrice']
trainData.drop(['SalePrice'], axis=1, inplace=True)

#%%
# =============================================================================
# # Filling the Dataset
# =============================================================================

fillWithNone = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
                "GarageType", "GarageFinish", "GarageQual", "GarageCond", 
                "BsmtExposure", "BsmtFinType2", "BsmtFinType1", "BsmtQual", 
                "BsmtCond", "MasVnrType", "MSSubClass"]

fillWithMode = ["SaleType", "Exterior1st", "Exterior2nd", "KitchenQual", 
                "Electrical", "MSZoning"]

fillWithZero = ["GarageArea", "GarageCars", "GarageYrBlt", "BsmtHalfBath", 
                "BsmtFullBath", "TotalBsmtSF", "BsmtUnfSF", "BsmtFinSF2", 
                "BsmtFinSF1", "MasVnrArea"]

for noneValue in fillWithNone:
    trainData[noneValue] = trainData[noneValue].fillna('None')

for modeValue in fillWithMode:
    trainData[modeValue] = trainData[modeValue].fillna(trainData[modeValue].mode()[0])

for zeroValue in fillWithZero:
    trainData[zeroValue] = trainData[zeroValue].fillna(0)


### Special Fill
trainData["Functional"] = trainData["Functional"].fillna("Typ")
trainData["LotFrontage"] = trainData.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

### Drop Utilities
trainData.drop(['Utilities'], axis=1, inplace=True)


#%%
# =============================================================================
# #Checking Missing data after filling Dataset
# =============================================================================

# trainNulls = (trainData.isnull().sum() / len(trainData)) * 100
# trainNulls = trainNulls.drop(trainNulls[trainNulls == 0].index).sort_values(ascending=False)
# missingData = pd.DataFrame({'Missing Ratio' :trainNulls})
# missingData.head()

#%%
# =============================================================================
# ## OneHot for categorical Values
# =============================================================================

from sklearn.preprocessing import LabelEncoder
trainDataTypes = trainData.dtypes
categoricalValues = trainDataTypes[trainDataTypes == 'object']

for catValue in categoricalValues.index:
    labelEncoder = LabelEncoder()
    labelEncoder.fit(list(trainData[catValue].values))
    trainData[catValue] = labelEncoder.transform(list(trainData[catValue].values))

#%%
# =============================================================================
# # Create New Features
# =============================================================================

# trainData['TotalSF'] = trainData['TotalBsmtSF'] + trainData['1stFlrSF'] + trainData['2ndFlrSF']

# trainData['Total_Bathrooms'] = (trainData['FullBath'] + (0.5 * trainData['HalfBath']) +
#                                 trainData['BsmtFullBath'] + (0.5 * trainData['BsmtHalfBath']))

# trainData['Total_porch_sf'] = (trainData['OpenPorchSF'] + trainData['3SsnPorch'] +
#                               trainData['EnclosedPorch'] + trainData['ScreenPorch'] +
#                               trainData['WoodDeckSF'])

# ######

# trainData['haspool'] = trainData['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
# trainData['has2ndfloor'] = trainData['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
# trainData['hasgarage'] = trainData['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
# trainData['hasbsmt'] = trainData['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
# trainData['hasfireplace'] = trainData['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# removalCat = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 
#               'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch', 
#               'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']

# for i in removalCat:
#     trainData.drop([i], axis=1, inplace=True)

#%%
# =============================================================================
# # Plotting New Feature
# =============================================================================

# plt.scatter(trainData.TotalSF, SalePrice, c = "blue", marker = "s")
# plt.title("New Feature")
# plt.xlabel("TotalSF")
# plt.ylabel("SalePrice")
# plt.show()

# plt.scatter(trainData.Total_Bathrooms, SalePrice, c = "blue", marker = "s")
# plt.title("New Feature")
# plt.xlabel("Total_Bathrooms")
# plt.ylabel("SalePrice")
# plt.show()

# plt.scatter(trainData.Total_porch_sf, SalePrice, c = "blue", marker = "s")
# plt.title("New Feature")
# plt.xlabel("Total_porch_sf")
# plt.ylabel("SalePrice")
# plt.show()

#%%
# =============================================================================
#  outliers
# # =============================================================================

trainData = trainData.join(SalePrice)
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(trainData)
y_noano = clf.predict(trainData)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

outliers = y_noano[y_noano['Top'] == -1]

# trainData = trainData.iloc[y_noano[y_noano['Top'] == 1].index.values]
# trainData.reset_index(drop = True, inplace = True)
# print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
# print("Number of rows without outliers:", trainData.shape[0])


##trainData = trainData.drop(trainData[(trainData['GrLivArea']>4000) & (trainData['SalePrice']<300000)].index)


# =============================================================================
# # Dummy Variables
# =============================================================================
#%%
trainData = pd.get_dummies(trainData)
print(trainData.shape)

# =============================================================================
# =============================================================================
# =============================================================================
# # # # Predictions
# =============================================================================
# =============================================================================
# =============================================================================
#%%
# =============================================================================
# ## Multilinear Prediction
# =============================================================================

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


#%%
# =============================================================================
# ## Lasso Prediction
# =============================================================================
# SalePrice = np.log10(SalePrice)
# X_train, X_test, y_train, y_test = train_test_split(trainData, SalePrice, test_size=0.25, random_state=0)

# lassoModel = LassoCV(alphas=[1, 0.1, 0.01, 0.001, 0.0005], selection='random', max_iter=15000).fit(X_train, y_train)

# lassoPrediction = lassoModel.predict(X_test)

# lassoRMSE = (np.sqrt(mean_squared_error(y_test, lassoPrediction)))

# # The closer towards 1, the better the fit
# lassoR2 = r2_score(y_test, lassoPrediction)
# ldf = pd.DataFrame({'Actual': y_test, 'Predicted': lassoPrediction})
# ldf  = ldf.sort_index(axis = 0)

# Ridge Prediction
#%%

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


#%%
# =============================================================================
# # Random Forest Prediction
# =============================================================================

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

#%%
# =============================================================================
# ### ANN Prediction
# =============================================================================

for i in range(99):
    trainData = trainData.drop(trainData.index[outliers.index[i] - i])


#%%
trainData.reset_index(drop = True, inplace = True)

SalePrice = trainData.SalePrice

trainData.drop(['SalePrice'], axis=1, inplace=True)

SalePrice = np.log10(SalePrice)

X_train, X_test, y_train, y_test = train_test_split(trainData, SalePrice, test_size=0.25, random_state= 0)

# Model
model = Sequential()
model.add(Dense(70,  input_dim = 78,kernel_initializer='normal', activation='relu'))
model.add(Dense(60,   kernel_initializer='normal', activation='relu'))
model.add(Dense(50,   kernel_initializer='normal', activation='relu'))
model.add(Dense(25,   kernel_initializer='normal', activation='relu'))
model.add(Dense(1,    kernel_initializer='normal'))

def rmse(y_true, y_pred):
 	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def r2(y_true, y_pred):
    SS_res =  backend.sum(backend.square(y_true - y_pred)) 
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + backend.epsilon()) )
 
# Compile model
model.compile(loss='mse', optimizer="adam", metrics=[rmse, r2])

history = model.fit(X_train, y_train, validation_split=0.25, epochs=150, batch_size=10, verbose=1)

ANNPrediction = model.predict(X_test)

adf = pd.DataFrame({'Actual': y_test, 'Predicted': ANNPrediction[:,0]})
adf  = adf.sort_index(axis = 0)

# =============================================================================
# #evaluate the keras model
# =============================================================================

loss, rmse, r2 = model.evaluate(X_test, y_test)
print(rmse)
print(r2)

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