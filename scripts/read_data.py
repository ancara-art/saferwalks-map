import pandas as pd
from firebase import firebase
import os #handling paths and files in python

def read_firebase():
    #firebase_url = os.environ.get('FIREBASE_URL') #TODO nice to have for security.
    #Tutorial we followed for reading data in firebase from python: https://morioh.com/p/4dca3ded4cea
    fb = firebase.FirebaseApplication(r'https://safer-walks-default-rtdb.firebaseio.com/', None)
    parents_data = fb.get(r'/RegisteredParents', '')

    #Creating a dataframe from parents_data. 
    df_parents = pd.DataFrame()
    for i, parent in enumerate(parents_data.values()):
        df_parents = df_parents.append(pd.DataFrame(parent, index=[i]), ignore_index=True)

    #Splitting userLocation from df_parents, into latitude: user_lat and longitude: user_long.
    df_parents['user_lat'] = [float(s.split(',')[0]) for s in df_parents['userLocation']]
    df_parents['user_long'] = [float(s.split(',')[1]) for s in df_parents['userLocation']]
    return df_parents

def read_schools(file_name, path):
    #reading schoolsdata file which contains all schools in Sweden.
    schools_file = file_name
    df_schools = pd.read_csv(os.path.join(path, schools_file))
    return df_schools

def merge_data(df_parents, df_schools):
    #Merging df_schools into df_parents by schoolId(df_parents) and osm_id(df_schools)
    df = pd.merge(df_parents, df_schools, left_on='schoolId', right_on='osm_id')
    df.drop(columns=['osm_id'], inplace=True)
    return df