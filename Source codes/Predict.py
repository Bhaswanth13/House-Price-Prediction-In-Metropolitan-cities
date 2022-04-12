import pandas as pd
import numpy as np
import pickle

bangalore_dict=dict(pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Bangalore_dic.csv'))
chennai_dict=dict(pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Chennai_dic.csv'))
delhi_dict=dict(pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Delhi_dic.csv'))
hyderabad_dict=dict(pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Hyderabad_dic.csv'))
kolkata_dict=dict(pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Kolkata_dic.csv'))
mumbai_dict=dict(pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Mumbai_dic.csv'))


with open('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Bangalore_model.pickle','rb') as f:
    bangalore_model=pickle.load(f)
with open('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Chennai_model.pickle','rb') as f:
    chennai_model=pickle.load(f)
with open('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Delhi_model.pickle','rb') as f:
    delhi_model=pickle.load(f)
with open('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Hyderabad_model.pickle','rb') as f:
    hyderabad_model=pickle.load(f)
with open('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Kolkata_model.pickle','rb') as f:
    kolkata_model=pickle.load(f)
with open('C:\\Users\\BHASWANTH REDDY\\Desktop\\PROJECT1\\Mumbai_model.pickle','rb') as f:
    mumbai_model=pickle.load(f)

cols=['Area', 'No. of Bedrooms', 'Resale', 'MaintenanceStaff',
       'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack',
       'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom',
       'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security',
       'PowerBackup', 'CarParking', 'StaffQuarter', 'Cafeteria',
       'MultipurposeRoom', 'Hospital', 'WashingMachine', 'Gasconnection',
       'AC', 'Wifi', "Children'splayarea", 'LiftAvailable', 'BED',
       'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV', 'DiningTable',
       'Sofa', 'Wardrobe', 'Refrigerator', 'Latitude', 'Longitude']

def Predict(city,town,area,resale,cafeteria,staff,no_of_bedrooms,intercom,security,wardrobe,carparking):
    city=city.strip().lower()
    town=town.strip().lower()
    arr=[0 for i in range(40)]
    arr[0]=area
    arr[2]=resale
    arr[20]=cafeteria
    arr[3]=staff
    arr[1]=no_of_bedrooms
    arr[11]=intercom
    arr[16]=security
    arr[36]=wardrobe
    arr[18]=carparking
    arr[15]=1
    arr[22]=1
    mod=None
    if(city=='bangalore'):
        arr[38]=bangalore_dict[town][0]
        arr[39]=bangalore_dict[town][1]
        mod=bangalore_model
    elif(city=='chennai'):
        arr[38]=chennai_dict[town][0]
        arr[39]=chennai_dict[town][1]
        mod=chennai_model
    elif(city=='delhi'):
        arr[38]=delhi_dict[town][0]
        arr[39]=delhi_dict[town][1]
        mod=delhi_model
    elif(city=='hyderabad'):
        arr[38]=hyderabad_dict[town][0]
        arr[39]=hyderabad_dict[town][1]
        mod=hyderabad_model
    elif(city=='kolkata'):
        arr[38]=kolkata_dict[town][0]
        arr[39]=kolkata_dict[town][1]
        mod=kolkata_model
    else:
        arr[38]=mumbai_dict[town][0]
        arr[39]=mumbai_dict[town][1]
        mod=mumbai_model
    return mod.predict(pd.DataFrame(data=np.array([arr]),columns=cols))[0]

print(Predict('kolkata','barasat',1000,1,0,1,2,1,1,1,1)) 


        




        
