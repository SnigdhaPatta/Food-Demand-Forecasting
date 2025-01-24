# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# %%
submission=pd.read_csv(r"/content/sample_submission_hSlSoT6.csv")
submission.head()

# %%
centres=pd.read_csv(r"/content/fulfilment_center_info.csv")
centres.head()

# %%
meal=pd.read_csv(r"/content/meal_info.csv")
meal.head()

# %%
train=pd.read_csv(r"/content/train.csv", on_bad_lines='warn')
train.head()

# %%
test=pd.read_csv(r"/content/test_QoiMO9B.csv.xls")
test.head()

# %%
combined=pd.concat((train,test),ignore_index=True)

combined=pd.merge(combined,centres,how="left",on="center_id")
combined=pd.merge(combined,meal,how="left",on="meal_id")

# %%
sns.displot(x="num_orders",data=combined)



# %%
sns.displot(np.log(combined.num_orders))
plt.show()

# %%
np.log(combined.num_orders).describe()

# %%
combined["log_target"]=np.log(combined.num_orders)

# %%
sns.displot(combined.base_price)
plt.show()

# %%


# %%
sns.displot(combined.checkout_price)
plt.show()

# %%
combined.head()

# %%
#Treatment of Prices
combined["price_diff"]=combined.base_price-combined.checkout_price

def price(x):
  if x<0:
    return("Discount")
  elif x>10:
    return("Taxes")
  else:
    return("Addnl_Chgs")

# %%
combined["price_cat"]=combined.price_diff.apply(price)

# %%
combined.price_cat.value_counts(normalize=True)

# %%
combined.loc[:, ["price_diff","num_orders"]].corr()

# %%
sns.kdeplot(combined.price_diff,color="blue")
plt.show()

# %%
sns.boxplot(x="price_cat", y="num_orders", data=combined)
plt.show()

# %%
combined.head()

# %%
#Social Media
combined["marketing"]=combined.emailer_for_promotion+combined.homepage_featured


# %%
sns.boxplot(x="marketing", y="num_orders", data=combined)
plt.show()

# %%
combined.head()

# %%
#Cuisine

combined.cuisine.value_counts()

# %%
sns.boxplot(x="cuisine", y="num_orders", data=combined)
plt.show()

# %%
#Category
combined.category.value_counts().plot(kind="bar")
plt.show()

#Top 5:- Beverages, Rice Bowl, Sandwich, Pizza, Starters

# %%
plt.figure(figsize=(15,10))
sns.boxplot(x="category", y="num_orders", data=combined)

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %%
# center_type, region code, city code

combined.center_type.value_counts

# %%
combined.center_type.value_counts().plot(kind="bar")

# %%
sns.boxplot(x="center_type", y="num_orders", data=combined)

# %%
sns.boxplot(x="region_code", y="num_orders", data=combined)

# %%
#top region-56, 71, 77

# %% [markdown]
# 

# %%


# %%
combined.city_code.nunique()

# %%
combined.city_code.value_counts()[:5].index


#top5 city codes-[590, 526, 638, 522, 517]

# %%
sns.scatterplot(x="city_code", y="num_orders", data=combined)

# %%
combined.head()

# %%
combined.op_area.describe()

# %%
import scipy.stats as stats
import math

def radii(x):
  return(np.sqrt(x/math.pi))

combined["radii"]=combined.op_area.apply(radii)


# %%
combined.loc[:, ["radii", "num_orders"]].corr()

# %%
combined.radii.describe()

# %%
#Week

combined.week.describe()

# %%
def year(x):
  if x<=52:
    return("Year1")
  elif x<=104:
    return("Year2")
  else:
    return("Year3")


# %%
combined["Year"]=combined.week.apply(year)

sns.boxplot(x="Year", y="num_orders", data=combined)
plt.show()

# %%
#Max week=155

sns.lineplot(x="week", y="num_orders", data=combined)

# %%
combined['sin_week'] = np.sin(2 * np.pi * combined['week'] / 52)
plt.figure(figsize=(10, 6))
sns.lineplot(data=combined, x='week', y='sin_week')
plt.title('Sin Transformation of Weeks')
plt.xlabel('Week')
plt.ylabel('Sin of Week')
plt.show()

# %%
combined['cos_week'] = np.cos(2 * np.pi * combined['week'] / 52)
plt.figure(figsize=(10, 6))
sns.lineplot(data=combined, x='week', y='cos_week')
plt.title('Cosine Transformation of Weeks')
plt.xlabel('Week')
plt.ylabel('Cosine of Week')
plt.show()

# %%
combined.head()

# %%
#Dropping 4 variables
combined.drop(["id", "week", "emailer_for_promotion", "homepage_featured"], axis=1, inplace=True)

# %%
combined.head()

# %%
combined.shape

# %%
combined.columns

# %% [markdown]
# #Feature Engineering
# 
# * center_id wise num_orders
# * meal_id wise num_orders
# * city_code wise num_orders
# * region_code wise num_orders
# * category wise num_orders
# 
# 

# %%
# Center wise
combined["magic1"]=combined.groupby("center_id")["num_orders"].transform("min")
combined["magic2"]=combined.groupby("center_id")["num_orders"].transform("mean")
combined["magic3"]=combined.groupby("center_id")["num_orders"].transform("max")
combined["magic4"]=combined.groupby("center_id")["num_orders"].transform("std")
combined["magic5"]=combined.groupby("center_id")["num_orders"].transform("median")

# %%
# Meal wise
combined["magic6"]=combined.groupby("meal_id")["num_orders"].transform("min")
combined["magic7"]=combined.groupby("meal_id")["num_orders"].transform("mean")
combined["magic8"]=combined.groupby("meal_id")["num_orders"].transform("max")
combined["magic9"]=combined.groupby("meal_id")["num_orders"].transform("std")
combined["magic10"]=combined.groupby("meal_id")["num_orders"].transform("median")

# %%
# City Code
combined["magic11"]=combined.groupby("city_code")["num_orders"].transform("min")
combined["magic12"]=combined.groupby("city_code")["num_orders"].transform("mean")
combined["magic13"]=combined.groupby("city_code")["num_orders"].transform("max")
combined["magic14"]=combined.groupby("city_code")["num_orders"].transform("std")
combined["magic15"]=combined.groupby("city_code")["num_orders"].transform("median")

# %%
# Region wise
combined["magic16"]=combined.groupby("region_code")["num_orders"].transform("min")
combined["magic17"]=combined.groupby("region_code")["num_orders"].transform("mean")
combined["magic18"]=combined.groupby("region_code")["num_orders"].transform("max")
combined["magic19"]=combined.groupby("region_code")["num_orders"].transform("std")
combined["magic20"]=combined.groupby("region_code")["num_orders"].transform("median")

# %%
# Category
combined["magic21"]=combined.groupby("category")["num_orders"].transform("min")
combined["magic22"]=combined.groupby("category")["num_orders"].transform("mean")
combined["magic23"]=combined.groupby("category")["num_orders"].transform("max")
combined["magic24"]=combined.groupby("category")["num_orders"].transform("std")
combined["magic25"]=combined.groupby("category")["num_orders"].transform("median")

# %% [markdown]
# # Variables with Checkout Price
# * category wise Checkout Price
# * center_id wise Checkout Price
# * meal_id wise Checkout Price
# * city_code wise Checkout Price
# * region_code wise Checkout Price
# 

# %%
#Category wise Checkout Price
combined["magic26"]=combined.groupby("category")["checkout_price"].transform("min")
combined["magic27"]=combined.groupby("category")["checkout_price"].transform("mean")
combined["magic28"]=combined.groupby("category")["checkout_price"].transform("max")
combined["magic29"]=combined.groupby("category")["checkout_price"].transform("std")
combined["magic30"]=combined.groupby("category")["checkout_price"].transform("median")

# %%
# center_id wise Checkout Price
combined["magic31"]=combined.groupby("center_id")["checkout_price"].transform("min")
combined["magic32"]=combined.groupby("center_id")["checkout_price"].transform("mean")
combined["magic33"]=combined.groupby("center_id")["checkout_price"].transform("max")
combined["magic34"]=combined.groupby("center_id")["checkout_price"].transform("std")
combined["magic35"]=combined.groupby("center_id")["checkout_price"].transform("median")

# %%
# meal_id wise Checkout Price
combined["magic36"]=combined.groupby("meal_id")["checkout_price"].transform("min")
combined["magic37"]=combined.groupby("meal_id")["checkout_price"].transform("max")
combined["magic38"]=combined.groupby("meal_id")["checkout_price"].transform("mean")
combined["magic39"]=combined.groupby("meal_id")["checkout_price"].transform("std")
combined["magic40"]=combined.groupby("meal_id")["checkout_price"].transform("median")

# %%
#city_code wise Checkout Price
combined["magic41"]=combined.groupby("city_code")["checkout_price"].transform("min")
combined["magic42"]=combined.groupby("city_code")["checkout_price"].transform("mean")
combined["magic43"]=combined.groupby("city_code")["checkout_price"].transform("max")
combined["magic44"]=combined.groupby("city_code")["checkout_price"].transform("std")
combined["magic45"]=combined.groupby("city_code")["checkout_price"].transform("median")

# %%
#region_code wise Checkout Price
combined["magic46"]=combined.groupby("region_code")["checkout_price"].transform("min")
combined["magic47"]=combined.groupby("region_code")["checkout_price"].transform("mean")
combined["magic48"]=combined.groupby("region_code")["checkout_price"].transform("max")
combined["magic49"]=combined.groupby("region_code")["checkout_price"].transform("std")
combined["magic50"]=combined.groupby("region_code")["checkout_price"].transform("median")

# %%
combined.shape

# %%


# %%
combined.columns

# %%
combined.shape

# %% [markdown]
# # Statistical Analysis

# %%
# Split the Data into Train and Test
newtrain=combined.loc[0:train.shape[0]-1,:]
newtest=combined.loc[train.shape[0]:, :]

train.shape, test. shape, newtrain.shape, newtest.shape

newtest.drop("num_orders", axis=1, inplace=True)

# %%
train.shape, test. shape, newtrain.shape, newtest.shape

# %%
pd.set_option('display.max_columns',100)
newtest.select_dtypes(include=np.number).columns

# %%
nums=['magic1', 'magic2', 'magic3', 'magic4',
       'magic5', 'magic6', 'magic7', 'magic8', 'magic9', 'magic10', 'magic11',
       'magic12', 'magic13', 'magic14', 'magic15', 'magic16', 'magic17',
       'magic18', 'magic19', 'magic20', 'magic21', 'magic22', 'magic23',
       'magic24', 'magic25', 'magic26', 'magic27', 'magic28', 'magic29',
       'magic30', 'magic31', 'magic32', 'magic33', 'magic34', 'magic35',
       'magic36', 'magic37', 'magic38', 'magic39', 'magic40', 'magic41',
       'magic42', 'magic43', 'magic44', 'magic45', 'magic46', 'magic47',
       'magic48', 'magic49', 'magic50']

# %%
#Ho=Predictor and Target are independent
#Ha=Ho is False

pvalue=[]

for i in nums:
  teststats,pval=stats.ttest_ind(newtrain[i],newtrain['num_orders'])
  pvalue.append(pval)

# %%
pvals=pd.DataFrame(pvalue, index=nums, columns=["PValue"])
significant=pvals.loc[pvals.PValue<0.05].index


# %%
pvals[pvals["PValue"]>0.05]

#these values are insignificant

# %%
pvals=pd.DataFrame(pvalue, index=nums, columns=["PValue"])
insignificant=pvals.loc[pvals.PValue>0.05].index
insignificant

# %%
#Price Difference Vs. Target
#Ho= Price difference has no relation with the number of orders.
#Ha= No of orders depend on Price Diff..

teststats, pval=stats.ttest_ind(newtrain.loc[:, "price_diff"], newtrain.num_orders)
print(teststats,pval)

# %%
#Radii an OP Area
#Ho=
#Ha=

teststats, pval=stats.ttest_ind(newtrain.loc[:, "op_area"],newtrain.num_orders)

print(teststats, pval)

teststats, pval=stats.ttest_ind(newtrain.loc[:, "radii"], newtrain.num_orders)

print(teststats, pval)


# %% [markdown]
# # Categorical analysis
# Cat Vars vs Num

# %%
newtrain.select_dtypes(include="object").columns

# %%
#Anova
#Ho: Predictors and Target are independent
#Ha: Ho is False

import statsmodels.formula.api as sfa
from statsmodels.stats.anova import anova_lm

model=sfa.ols("num_orders~center_type+category+price_cat+Year", data=newtrain).fit()
anova_table=anova_lm(model)
anova_table

# %%
#Drop the iNsignifiacant Features from the Train and Test

newtrain.drop(insignificant, axis=1, inplace=True)
newtest.drop(insignificant, axis=1, inplace=True)


# %%
newtrain.columns

# %%
newtrain.drop(['center_id', 'meal_id','city_code', 'region_code', 'category'], axis=1, inplace=True)
newtest.drop(['center_id', 'meal_id','city_code', 'region_code', 'category'], axis=1, inplace=True)

# %%
newtest.shape, newtrain.shape

# %%
newtrain.columns

# %% [markdown]
# # Modeling
# 
# * Split the data in X and y
# * K-Fold CV
# * Prediction
# * Submission

# %%
# Split the data in X and y
X=newtrain.drop(["num_orders", "log_target", "Year"], axis=1)
y=newtrain.num_orders
logy=newtrain.log_target

dummyx=pd.get_dummies(X, drop_first=True)


newtest.drop("log_target", axis=1, inplace=True)


# %%
newtest.drop("Year", axis=1, inplace=True)
dummytest=pd.get_dummies(newtest, drop_first=True)


# %%
# Model building on X and Y --> KFold CV

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error

kfold=KFold(n_splits=5, shuffle=True)

lr=LinearRegression()
rfr=RandomForestRegressor()
gbr=GradientBoostingRegressor()

# %%
pred=[]

for train_index, test_index in kfold.split(dummyx, y):
  xtrain=dummyx.iloc[train_index, :]
  xtest=dummyx.iloc[test_index, :]
  ytrain=y.loc[train_index]
  ytest=y.iloc[test_index]

  #modeling and prediction
  pred_lr=lr.fit(xtrain, ytrain).predict(xtest)

  #Cost Function
  cost=root_mean_squared_error(ytest,pred_lr)
  print(cost)

  #FINAL PREDICTION
  pred.append(lr.fit(xtrain, ytrain).predict(dummytest))



# %%
sumdummyx.columns

# %%
dummytest.columns

# %%
newtest.columns

# %%
newtrain.columns

# %%
submission=pd.DataFrame(pred).T.mean(axis=1)
submission

# %%
submission.to_csv("submission.csv", index=False)

# %%
#sample
sample=pd.read_csv("/content/sample_submission_hSlSoT6.csv")

sample["num_orders"]=np.abs(submission)

#Export it to csv
sample.to_csv("LRModel.csv", index=False)

# %%
sample.head()

# %%


# %%



