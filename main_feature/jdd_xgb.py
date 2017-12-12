import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix
train=pd.read_csv('newtrain.csv')
test=pd.read_csv('newtest.csv')
trade=pd.concat([train,test])
dummies_city=pd.get_dummies(trade['city'],prefix='city')
dummies_log_from=pd.get_dummies(trade['log_from'],prefix='log_from')
dummies_result=pd.get_dummies(trade['result'],prefix='result')
dummies_type=pd.get_dummies(trade['type'],prefix='type')
X=pd.concat([trade[['timelong','is_scan','is_sec','is_risk','rowkey','devicerat','iprat']],dummies_city,dummies_log_from,dummies_result,dummies_type], axis=1)
X=X.reset_index(drop=True)
df_train=X[X.is_risk.notnull()]
df_test=X[X.is_risk.isnull()]
y=df_train['is_risk']
train1=df_train.drop(['is_risk','rowkey'],axis=1)
test1=df_test.drop(['is_risk','rowkey'],axis=1)
def fscore(preds, dtrain):
	label=dtrain.get_label()
	pred = [int(i>=0.48) for i in preds]
	confusion_matrixs = confusion_matrix(label, pred)
	recall =float(confusion_matrixs[0][0]) / float(confusion_matrixs[0][1]+confusion_matrixs[0][0])
	precision = float(confusion_matrixs[0][0]) / float(confusion_matrixs[1][0]+confusion_matrixs[0][0])
	F = 1.01*precision* recall/(0.01*precision+recall)
	return 'fscore',float(F)
params={
	'booster':'gbtree',
	'objective': 'binary:logistic',
	#'eval_metric':'auc',
	'subsample':0.8,
	'colsample_bytree':0.8,
	'min_child_weight':1,
	'gamma':0.1,
	'max_depth':10,
	'eta': 0.1,
	'seed':0
	}
xgbtrain = xgb.DMatrix(train1,y)
xgbtest = xgb.DMatrix(test1)
watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
num_rounds=60
model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15,feval=fscore,maximize=True)
ypred=model.predict(xgbtest)
y_pred=(ypred >= 0.5)*1
df_test['is_risk']=y_pred
df_test[['rowkey','is_risk']].to_csv('result1.csv',index=False)

