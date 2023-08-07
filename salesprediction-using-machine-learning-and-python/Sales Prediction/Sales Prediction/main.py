import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import  LinearRegression
from joblib import dump

df=pd.read_csv('sales_train.csv')
df=df.dropna()
df=df.drop(columns=["Item_Identifier",'Item_Visibility','Outlet_Identifier','Outlet_Size','Outlet_Establishment_Year'])
x=df.drop(columns=['Item_Outlet_Sales'])#independent
y=df["Item_Outlet_Sales"]#dependent

df.Item_Weight = df.Item_Weight.fillna(df.Item_Weight.mean())
df.Item_Outlet_Sales = df.Item_Outlet_Sales.fillna(df.Item_Outlet_Sales.mean())

#checking if there is any missing value
print(df.isnull().sum())
print("------------")

lf=LabelEncoder()
x['Item_Fat_Content']=lf.fit_transform(x['Item_Fat_Content'])

lt=LabelEncoder()
x['Item_Type']=lt.fit_transform(x['Item_Type'])


ll=LabelEncoder()
x['Outlet_Location_Type']=ll.fit_transform(x['Outlet_Location_Type'])

lg=LabelEncoder()
x['Outlet_Type']=lg.fit_transform(x['Outlet_Type'])

#scaling our independent data
sc=StandardScaler()
x=sc.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Random Forest Classifier Model

lc=LinearRegression()
lc.fit(x_train,y_train)
y_pred=lc.predict(x_test)

#checking accuracy
print("Linear Regression")
accuracy=r2_score(y_test, y_pred)
print(accuracy)

#saving models
dump(sc,"scaling.joblib")
dump(lc,"reg.joblib")
dump(lf,"fat.joblib")
dump(lt,"type.joblub")
dump(ll,"out.joblib")
dump(lg,"outtype.joblib")