import pandas as pd
from sklearn import preprocessing
def remove_duplicate(df):
	df=df.drop_duplicates(keep='first')
	print("length of dataframe after preprocessing is " + str(len(df)))
	return df.to_json()

def remove_null_row(df):
	df=df.dropna(axis=0,how='any')
	print("length pf dataframe afer removing null rows is "+ str(len(df)))
	return df.to_json()


def normalise_df(df):
	val=df.values
	minmax=preprocessing.MinMaxScaler()
	scaled_val=minmax.fit_transform(val)
	df=pd.DataFrame(scaled_val)
	return df.to_json()


def avg_missing_data(df):
	print("no of missing values is "+str(df.isna().sum().sum()))
	df=df.fillna(value=df.mean())
	return df.to_json()






df=pd.read_csv("diabetes.csv")
print("length of dataframe before preprocesing" + str(len(df)))
res_remove_duplicate=remove_duplicate(df)


df1=pd.read_csv("diabetes.csv")
print("length of dataframe before preprocesing" + str(len(df1)))
res_null_row=remove_null_row(df)



df2=pd.read_csv("diabetes.csv")
print("length of dataframe before preprocesing" + str(len(df2)))
res_normalised_df=normalise_df(df2)
#print(res_normalised_df)


df3=pd.read_csv("diabetes.csv")
res=avg_missing_data(df3)