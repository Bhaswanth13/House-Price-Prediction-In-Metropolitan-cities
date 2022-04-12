from xgboost import XGBRegressor #Trainer
import pandas as pd #read and write
import numpy as np  #Math operations
from geopy.geocoders import Nominatim #latitude and longitude
import warnings #for ignoring warnings if any, make them to exceptions
import pickle #storing model
warnings.simplefilter('ignore')
df1=pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Hyderabad.csv')

df1.Location=df1.Location.apply(lambda x : x.strip().lower()) #lowercase
location_states=df1.groupby('Location')['Location'].agg('count').sort_values(ascending=False) #location counts
other_locations=location_states[location_states<=10] #loacations of location_count less than 11
df1.Location=df1.Location.apply(lambda x: "other" if x in other_locations else x) #replacing loacation_names of location_count less than 11 by other
df1=df1[~(df1.Location=='other')] #removing other
df1=df1[~(df1.Area/df1['No. of Bedrooms']<300)] #outliers
df1['price_per_unit']=df1.Price/df1.Area #adding new_temp col

# function to remove outliers
def removing_extremes(df):
    res=pd.DataFrame()
    for key,subdf in df.groupby('Location'):
        m=np.mean(subdf.price_per_unit)
        sd=np.std(subdf.price_per_unit)
        reduced=subdf[(subdf.price_per_unit>=(m-sd)) & (subdf.price_per_unit<=(m+sd))]
        res=pd.concat([res,reduced],ignore_index=True)
    return res
df=removing_extremes(df1)


#generation of new columns latitudes and longitudes using feature_generation function
geolocator = Nominatim(user_agent="Ruch")

def feature_generation(df):
    d=dict()
    a=[]
    b=[]
    for i in df['Location']: 
        if(d.get(i)!=None):
            a.append(d[i][0])
            b.append(d[i][1])
            continue
        location = geolocator.geocode(i)
        if(location==None):
            d[i]=[-1,-1]
            a.append(d[i][0])
            b.append(d[i][1])
            continue
        d[i]=[location.latitude,location.longitude]
        a.append(d[i][0])
        b.append(d[i][1])
    df['Latitude']=a
    df['Longitude']=b
    return (df,d)
t=feature_generation(df)
df=t[0]  #new_dataframe
dic=t[1] #stroring loc co-ordinates in dictionary

df=df[~((df.Longitude==-1)&(df.Latitude==-1))] #unknown locations (better to leave this one)

dicdf=dict()
for i,j in dic.items():
    if(j!=[-1,-1]):
        dicdf[i]=j
dicdf=pd.DataFrame(data=dicdf) #converting dictionary to co-ordinates
dicdf.to_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Hyderabad_dic.csv',index=False)


X=df.drop(columns=['Location',"Price",'price_per_unit'],axis=1) #final attributes for training (dropping the columns loc,price,price_per_unit)
y=df.Price #price

##Training

mod= XGBRegressor(eta=0.3,max_depth=15,subsample=1) #model selection
mod.fit(X,y) #fitting the model

with open('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Hyderabad_model.pickle','wb') as f:
    pickle.dump(mod,f)  #storing model using pickle
    
print(mod.score(X,y))
