# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:46:29 2019

@author: PMishra
"""
#This is IEEE-CIS Fraud Detection : playground
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc,os,sys
import re

from sklearn import metrics, preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle

sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format

train_id = pd.read_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\train_identity.csv')
train_trn = pd.read_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\train_transaction.csv')
test_id = pd.read_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\test_identity.csv')
test_trn = pd.read_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\test_transaction.csv')

print(train_id.shape, test_id.shape)
print(train_trn.shape, test_trn.shape)

[c for c in train_trn.columns if c not in test_trn.columns] #['isFraud']

fc = train_trn['isFraud'].value_counts(normalize=True).to_frame()
fc.plot.bar()
fc.T

# Not all transactions have corresponding identity information.
#len([c for c in train_trn['TransactionID'] if c not in train_id['TransactionID'].values]) #446307

# Not all fraud transactions have corresponding identity information.
fraud_id = train_trn[train_trn['isFraud'] == 1]['TransactionID']
fraud_id_in_trn = [i for i in fraud_id if i in train_id['TransactionID'].values]
print(f'fraud data count:{len(fraud_id)}, and in trn:{len(fraud_id_in_trn)}')

#Identity data
#Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions. They're collected by Vesta’s fraud protection system and digital security partners. (The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)
#
#Categorical Features:
#
#DeviceType
#DeviceInfo
#id12 - id38

train_id_trn = pd.merge(train_id, train_trn[['isFraud','TransactionID']])
train_id_f0 = train_id_trn[train_id_trn['isFraud'] == 0]
train_id_f1 = train_id_trn[train_id_trn['isFraud'] == 1]
del train_id_trn
print(train_id_f0.shape, train_id_f1.shape)

def plotHistByFraud(col, bins=20, figsize=(8,3)):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=figsize)
        plt.hist([train_id_f0[col], train_id_f1[col]], bins=bins, density=True, color=['royalblue', 'orange'])
        
def plotCategoryRateBar(col, topN=np.nan, figsize=(8,3)):
    a, b = train_id_f0, train_id_f1
    if topN == topN: # isNotNan
        vals = b[col].value_counts(normalize=True).to_frame().iloc[:topN,0]
        subA = a.loc[a[col].isin(vals.index.values), col]
        df = pd.DataFrame({'normal':subA.value_counts(normalize=True), 'fraud':vals})
    else:
        df = pd.DataFrame({'normal':a[col].value_counts(normalize=True), 'fraud':b[col].value_counts(normalize=True)})
    df.sort_values('fraud', ascending=False).plot.bar(figsize=figsize)
    
    
plotHistByFraud('id_01')
plotHistByFraud('id_02')
plotHistByFraud('id_07')

numid_cols = [f'id_{str(i).zfill(2)}' for i in range(1,12)]
train_id_f1[['isFraud'] + numid_cols].head(10)
train_id_f0[['isFraud'] + numid_cols].head(10)

plotCategoryRateBar('id_15')
plotCategoryRateBar('id_16')

plotCategoryRateBar('id_17', 10)
plotCategoryRateBar('id_19', 20)
plotHistByFraud('id_19')
print('unique count:', train_id['id_19'].nunique())
plotCategoryRateBar('id_20', 20)
plotHistByFraud('id_20')
print('unique count:', train_id['id_20'].nunique())
plotCategoryRateBar('id_23')
plotCategoryRateBar('id_26', 15)
plotCategoryRateBar('id_28')
plotCategoryRateBar('id_29')
plotCategoryRateBar('id_31', 20)

train_id_f0['_id_31_ua'] = train_id_f0['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')
train_id_f1['_id_31_ua'] = train_id_f1['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')
plotCategoryRateBar('_id_31_ua', 10)

plotCategoryRateBar('id_32')
plotCategoryRateBar('id_33', 15)
plotCategoryRateBar('id_34')
plotCategoryRateBar('id_35')
plotCategoryRateBar('id_38')
plotCategoryRateBar('DeviceType')
plotCategoryRateBar('DeviceInfo', 10)

#Transaction data
#TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
#TransactionAMT: transaction payment amount in USD
#ProductCD: product code, the product for each transaction
#card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
#addr: address
#dist: distance
#P_ and (R__) emaildomain: purchaser and recipient email domain
#C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
#D1-D15: timedelta, such as days between previous transaction, etc.
#M1-M9: match, such as names on card and address, etc.
#Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

ccols = [f'C{i}' for i in range(1,15)]
dcols = [f'D{i}' for i in range(1,16)]
mcols = [f'M{i}' for i in range(1,10)]
vcols = [f'V{i}' for i in range(1,340)]

train_trn_f0 = train_trn[train_trn['isFraud'] == 0]
train_trn_f1 = train_trn[train_trn['isFraud'] == 1]
print(train_trn_f0.shape, train_trn_f1.shape)

def plotTrnHistByFraud(col, bins=20):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=(8,3))
        plt.hist([train_trn_f0[col], train_trn_f1[col]], bins=bins, density=True, color=['royalblue', 'orange'])

def plotTrnLogHistByFraud(col, bins=20):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=(8,3))
        plt.hist([np.log1p(train_trn_f0[col]), np.log1p(train_trn_f1[col])], bins=bins, density=True, color=['royalblue', 'orange'])
        
def plotTrnCategoryRateBar(col, topN=np.nan):
    a, b = train_trn_f0, train_trn_f1
    if topN == topN: # isNotNan
        vals = b[col].value_counts(normalize=True).to_frame().iloc[:topN,0]
        subA = a.loc[a[col].isin(vals.index.values), col]
        df = pd.DataFrame({'normal':subA.value_counts(normalize=True), 'fraud':vals})
    else:
        df = pd.DataFrame({'normal':a[col].value_counts(normalize=True), 'fraud':b[col].value_counts(normalize=True)})
    df.sort_values('fraud', ascending=False).plot.bar(figsize=(8,3))
    
train_trn_f0['TransactionDT'].nunique(), train_trn_f0['TransactionDT'].shape[0]
train_trn_f1['TransactionDT'].nunique(), train_trn_f1['TransactionDT'].shape[0]

plotTrnHistByFraud('TransactionAmt')
plotTrnLogHistByFraud('TransactionAmt')

amt_desc = pd.concat([train_trn_f0['TransactionAmt'].describe(), train_trn_f1['TransactionAmt'].describe()], axis=1)
amt_desc.columns = ['normal','fraud']
amt_desc

train_trn['_amount_max_card1'] = train_trn.groupby(['card1'])['TransactionAmt'].transform('max')
train_trn[['card1','_amount_max_card1']].drop_duplicates().sort_values(by='_amount_max_card1', ascending=False).head(10)

plotTrnCategoryRateBar('ProductCD')
cols = [f'card{n}' for n in range(1,7)]
train_trn[cols].isnull().sum()

pd.concat([train_trn[train_trn['card4']=='visa'][cols].head(),
        train_trn[train_trn['card4']=='mastercard'][cols].head()])
    
plotTrnCategoryRateBar('card1', 15)
plotTrnHistByFraud('card1', bins=30)

plotTrnCategoryRateBar('card2', 15)
plotTrnHistByFraud('card2', bins=30)

plotTrnCategoryRateBar('card3', 10)
plotTrnCategoryRateBar('card4')
plotTrnCategoryRateBar('card5', 10)
plotTrnCategoryRateBar('card6')

print(len(train_trn))
print(train_trn['card1'].nunique(), train_trn['card2'].nunique(), train_trn['card3'].nunique(), train_trn['card5'].nunique())

train_trn['card_n'] = (train_trn['card1'].astype(str) + '_' + train_trn['card2'].astype(str) \
       + '_' + train_trn['card3'].astype(str) + '_' + train_trn['card5'].astype(str))
print(train_trn['card_n'].nunique())

train_trn['card_n'].value_counts()

vc = train_trn['card_n'].value_counts()
vc[vc > 3000].plot.bar()

train_trn.groupby(['card_n'])['isFraud'].mean().sort_values(ascending=False)

cols = ['TransactionDT','TransactionAmt','isFraud'] + ccols
train_trn[train_trn['card1'] == 9500][cols].head(20)

cols = ['TransactionDT','TransactionAmt','isFraud'] + ccols
train_trn[train_trn['card1'] == 4774][cols].head(20)

train_trn[train_trn['card1'] == 14770][cols].head(20)

cols = ['TransactionDT','TransactionAmt','isFraud'] + dcols
train_trn[train_trn['card1'] == 9500][cols].head(20)

cols = ['TransactionDT','TransactionAmt','isFraud'] + dcols
train_trn[train_trn['card1'] == 4774][cols].head(20)

train_trn[train_trn['card1'] == 14770][cols].head(20)


plotTrnCategoryRateBar('addr1', 20)
plotTrnHistByFraud('addr1', bins=30)
plotTrnCategoryRateBar('addr2', 10)
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, ['addr1','addr2']].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, ['addr1','addr2']].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)
plotTrnCategoryRateBar('dist1', 20)
plotTrnLogHistByFraud('dist1', bins=30)

plotTrnCategoryRateBar('dist2', 20)
plotTrnLogHistByFraud('dist2', bins=30)
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, ['dist1','dist2']].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, ['dist1','dist2']].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)
plotTrnCategoryRateBar('P_emaildomain', 10)
plotTrnCategoryRateBar('R_emaildomain',10)

cols = ['TransactionDT','TransactionAmt','isFraud'] + ['addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain']
train_trn[train_trn['card1'] == 4774][cols].head(20)

train_trn[train_trn['card1'] == 9500][cols].head(20)

train_trn['P_emaildomain'].fillna('unknown',inplace=True)
train_trn['R_emaildomain'].fillna('unknown',inplace=True)

inf = pd.DataFrame([], columns=['P_emaildomain','R_emaildomain','Count','isFraud'])
for n in (train_trn['P_emaildomain'] + ' ' + train_trn['R_emaildomain']).unique():
    p, r = n.split()[0], n.split()[1]
    df = train_trn[(train_trn['P_emaildomain'] == p) & (train_trn['R_emaildomain'] == r)]
    inf = inf.append(pd.DataFrame([p, r, len(df), df['isFraud'].mean()], index=inf.columns).T)

inf.sort_values(by='isFraud', ascending=False).head(10)

plotTrnCategoryRateBar('C1',10)
plotTrnCategoryRateBar('C13',10)
train_trn[ccols].describe().loc[['count','mean','std','min','max']]
plt.figure(figsize=(10,5))

corr = train_trn[ccols].corr()
sns.heatmap(corr, annot=True, fmt='.2f')
plotTrnCategoryRateBar('D1',10)
plotTrnCategoryRateBar('D2',10)
plotTrnCategoryRateBar('D4',10)
plotTrnCategoryRateBar('D15',10)
train_trn[dcols].describe().loc[['count','mean','std','min','max']]
plt.figure(figsize=(10,5))

corr = train_trn[dcols].corr()
sns.heatmap(corr, annot=True, fmt='.2f')

fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, dcols].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, dcols].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)

plotTrnCategoryRateBar('M1')
plotTrnCategoryRateBar('M2')
plotTrnCategoryRateBar('M3')
plotTrnCategoryRateBar('M4')

plotTrnCategoryRateBar('M5')
plotTrnCategoryRateBar('M6')
plotTrnCategoryRateBar('M7')
plotTrnCategoryRateBar('M8')
plotTrnCategoryRateBar('M9')

fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, mcols].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, mcols].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)

for f in ['V1','V14','V41','V65','V88','V107','V305']:
    plotTrnCategoryRateBar(f)
train_trn[vcols].isnull().sum() / len(train_trn)
train_trn.loc[train_trn['V1'].isnull(), vcols].head(10)

train_trn.loc[train_trn['V1'].isnull() == False, vcols].head(10)
np.sort(train_trn[vcols].isnull().sum(axis=1).unique())
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, vcols].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, vcols].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)

train_trn[vcols].describe().T[['min','max']].T

vcols = [f'V{i}' for i in range(1,340)]

pca = PCA()
pca.fit(train_trn[vcols].fillna(-1))
plt.xlabel('components')
plt.plot(np.add.accumulate(pca.explained_variance_ratio_))
plt.show()

pca = PCA(n_components=0.99)
vcol_pca = pca.fit_transform(train_trn[vcols].fillna(-1))
print(vcol_pca.ndim)

#sc = preprocessing.StandardScaler()
sc = preprocessing.MinMaxScaler()

pca = PCA(n_components=2) #0.99
vcol_pca = pca.fit_transform(sc.fit_transform(train_trn[vcols].fillna(-1)))

fig, ax = plt.subplots(1, 2, figsize=(16, 3), sharey=True)
ax[0].scatter(x=vcol_pca[train_trn['isFraud'] == 1,0], y=vcol_pca[train_trn['isFraud'] == 1,1], alpha=0.5, c='r')
ax[1].scatter(x=vcol_pca[train_trn['isFraud'] == 0,0], y=vcol_pca[train_trn['isFraud'] == 0,1], alpha=0.1, c='b')
#plt.scatter(x=vcol_pca[:,0], y=vcol_pca[:,1], alpha=0.1, c=train_trn['isFraud'])

import matplotlib.cm as cm

km = KMeans(n_clusters=7, tol=1e-04, random_state=42)
y_km = km.fit_predict(vcol_pca)

plt.scatter(x=vcol_pca[:,0], y=vcol_pca[:,1], alpha=0.5, c=y_km, cmap=cm.seismic)

del train_trn_f0,train_trn_f1,train_id_f0,train_id_f1,y_km

print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],
                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])

#Feature Engineering

train_id = pd.read_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\train_identity.csv')
train_trn = pd.read_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\train_transaction.csv')
test_id = pd.read_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\test_identity.csv')
test_trn = pd.read_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\test_transaction.csv')

id_cols = list(train_id.columns.values)
trn_cols = list(train_trn.drop('isFraud', axis=1).columns.values)

X_train = pd.merge(train_trn[trn_cols + ['isFraud']], train_id[id_cols], how='left')
#X_train = reduce_mem_usage(X_train)
X_test = pd.merge(test_trn[trn_cols], test_id[id_cols], how='left')
#X_test = reduce_mem_usage(X_test)

X_train_id = X_train.pop('TransactionID')
X_test_id = X_test.pop('TransactionID')
del train_id,train_trn,test_id,test_trn

all_data = X_train.append(X_test, sort=False).reset_index(drop=True)

vcols = [f'V{i}' for i in range(1,340)]

sc = preprocessing.MinMaxScaler()

pca = PCA(n_components=2) #0.99
vcol_pca = pca.fit_transform(sc.fit_transform(all_data[vcols].fillna(-1)))

all_data['_vcol_pca0'] = vcol_pca[:,0]
all_data['_vcol_pca1'] = vcol_pca[:,1]
all_data['_vcol_nulls'] = all_data[vcols].isnull().sum(axis=1)

all_data.drop(vcols, axis=1, inplace=True)
all_data['card4'].fillna('unknown',inplace=True)
all_data['card6'].fillna('unknown',inplace=True)

all_data['P_emaildomain'].fillna('unknown',inplace=True)
all_data['R_emaildomain'].fillna('unknown',inplace=True)

import datetime

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
all_data['Date'] = all_data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
all_data['_weekday'] = all_data['Date'].dt.dayofweek
all_data['_hour'] = all_data['Date'].dt.hour
#all_data['_day'] = all_data['Date'].dt.day

all_data['_weekday'] = all_data['_weekday'].astype(str)
all_data['_hour'] = all_data['_hour'].astype(str)
all_data['_weekday__hour'] = all_data['_weekday'] + all_data['_hour']

all_data.drop(['TransactionDT','Date'], axis=1, inplace=True)

all_data['_id_31_ua'] = all_data['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')

all_data['_P_emaildomain__addr1'] = all_data['P_emaildomain'] + '__' + all_data['addr1'].astype(str)
all_data['_card1__card2'] = all_data['card1'].astype(str) + '__' + all_data['card2'].astype(str)
all_data['_card1__addr1'] = all_data['card1'].astype(str) + '__' + all_data['addr1'].astype(str)
all_data['_card2__addr1'] = all_data['card2'].astype(str) + '__' + all_data['addr1'].astype(str)
all_data['_card12__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)
all_data['_card_all__addr1'] = all_data['_card12__addr1'] + '__' + all_data['addr1'].astype(str)

all_data['_amount_decimal'] = ((all_data['TransactionAmt'] - all_data['TransactionAmt'].astype(int)) * 1000).astype(int)
all_data['_amount_decimal_len'] = all_data['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))
all_data['_amount_fraction'] = all_data['TransactionAmt'].apply(lambda x: float('0.'+re.sub('^[0-9]|\.|0+$', '', str(x))))
all_data[['TransactionAmt','_amount_decimal','_amount_decimal_len','_amount_fraction']].head(10)


cols = ['ProductCD','card1','card2','card5','card6','P_emaildomain','_card_all__addr1']
#,'card3','card4','addr1','dist2','R_emaildomain'

# amount mean&std
for f in cols:
    all_data[f'_amount_mean_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('mean')
    all_data[f'_amount_std_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('std')
    all_data[f'_amount_pct_{f}'] = (all_data['TransactionAmt'] - all_data[f'_amount_mean_{f}']) / all_data[f'_amount_std_{f}']

# freq encoding
for f in cols:
    vc = all_data[f].value_counts(dropna=False)
    all_data[f'_count_{f}'] = all_data[f].map(vc)
    
print('features:', all_data.shape[1])

cat_cols = [f'id_{i}' for i in range(12,39)]
for i in cat_cols:
    if i in all_data.columns:
        all_data[i] = all_data[i].astype(str)
        all_data[i].fillna('unknown', inplace=True)

enc_cols = []
for i, t in all_data.loc[:, all_data.columns != 'isFraud'].dtypes.iteritems():
    if t == object:
        enc_cols.append(i)
        #df = pd.concat([df, pd.get_dummies(df[i].astype(str), prefix=i)], axis=1)
        #df.drop(i, axis=1, inplace=True)
        all_data[i] = pd.factorize(all_data[i])[0]
        #all_data[i] = all_data[i].astype('category')
print(enc_cols)

X_train = all_data[all_data['isFraud'].notnull()]
X_test = all_data[all_data['isFraud'].isnull()].drop('isFraud', axis=1)
Y_train = X_train.pop('isFraud')
del all_data



import lightgbm as lgb

params={'learning_rate': 0.01,
        'objective': 'binary',
        'metric': 'auc',
        'num_threads': -1,
        'num_leaves': 256,
        'verbose': 1,
        'random_state': 42,
        'bagging_fraction': 1,
        'feature_fraction': 0.85
       }

oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])

clf = lgb.LGBMClassifier(**params, n_estimators=3000)
clf.fit(X_train, Y_train)
oof_preds = clf.predict_proba(X_train, num_iteration=clf.best_iteration_)[:,1]
sub_preds = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# Plot feature importance
feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 50:]
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10,12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

X_train.columns[np.argsort(-feature_importance)].values

submission = pd.DataFrame()
submission['TransactionID'] = X_test_id
submission['isFraud'] = sub_preds
submission.to_csv('C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\submission6.csv', index=False)

filename = 'C:\\Users\\PMishra\\Downloads\\ieee-fraud-detection\\model6.sav'
pickle.dump(clf, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)