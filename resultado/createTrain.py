"""
Created on Sun Dec 20 15:29:55 2020
Creates training data
@author: Santiago L. Zuñiga, santiago.zuniga@ib.edu.ar
"""
import numpy as np
import ee
import pandas as pd
from tsaug import TimeWarp

pd.options.mode.chained_assignment = None  # to hide some pandas warnings


df = pd.read_csv('./data_train.csv') #read dataset
df = df[df.Cultivo != 'S/M']
labels = pd.read_csv('./Etiquetas.csv')

for i, label in enumerate(labels.Cultivo):
    df.loc[df['Cultivo'] == label, 'Cultivo'] = labels.CultivoId[i] 
    
df_1920 = df.loc[df['Campania'] == '19/20']
df_1920 = df_1920[[ 'Longitud', 'Latitud', 'Cultivo','GlobalId']]
df_1920 = df_1920.reset_index()

df_1819 = df.loc[df['Campania'] == '18/19']
df_1819 = df_1819[[ 'Longitud', 'Latitud', 'Cultivo','GlobalId']]
df_1819 = df_1819.reset_index()

# %% defs
def get_data(img): #function to map 
    date = ee.Date(img.get('system:time_start')).format('D')     
    value = img.reduceRegion(ee.Reducer.mean(), p_get,1)   
    B2 = value.get('B2')
    B3 = value.get( 'B3')
    B4 = value.get( 'B4')
    B8 = value.get( 'B8')
    B11 = value.get( 'B11')
    B12 = value.get( 'B12')
    QA60 = value.get( 'QA60')
    ft = ee.Feature(None, {'date': date, \
                          'B2': B2,
                          'B3': B3,
                          'B4': B4,
                          'B8': B8,
                          'B11': B11,
                          'B12': B12,
                          'QA60': QA60,
                          })
    return ft

def get_pd(ft): #convert to pandas dataset
    ft = ft.getInfo()['features']
    dum = pd.DataFrame(ft [:]).properties
    asd = []
    for i in range(len(ft)):
        asd.append(dum[i])
    asd = pd.DataFrame(asd)
    asd['date'] = asd['date'].apply(pd.to_numeric)
    return asd

def get_X(pdf): #removes cloudy samples and interpolates them
    for i in ['B2', 'B3', 'B4','B8', 'B11', 'B12']:
        pdf.loc[(pdf.QA60 != 0),i]=np.NaN  #replace cloudy days (qa60 bitmask not 0) with NaN
    pdf = pdf[['date','B2', 'B3', 'B4', 'B8', 'B11', 'B12']]

    for i in ['B2', 'B3', 'B4','B8', 'B11', 'B12']:
        dum = pdf[i]
        pdf[i] = dum.interpolate().fillna(method='bfill').values.ravel() #interpolates NaN
    
    pdf.date = pdf.date//5
    pdf = pdf.drop_duplicates(subset=['date'])  #removes duplicates from Sentintel 2 A/B
    pdf = pdf.drop('date',axis=1)
    pdf = pdf.iloc[:48]
    X = pdf.values
    return X

X = [] #create empty arrays
Y = []
ee.Initialize()

# %% get data 18/19
for l in range(df_1819.shape[0]):
    punto = df_1819.iloc[l] 
    p = ee.Geometry.Point(punto.Longitud,punto.Latitud).buffer(100)
    
    S2_collection = ee.ImageCollection("COPERNICUS/S2") \
                        .filterBounds(p) \
                        .filterDate('2018-09-01', '2019-04-30') \
                        .sort('system:id', opt_ascending=True) \
                                                
    p_get = p                 
    featCol = S2_collection.map(get_data)
    pdf = get_pd(featCol)
    X.append(get_X(pdf))
    Y.append(punto.Cultivo)
    
# %% get data 19/20
for l in range(df_1920.shape[0]):
    punto = df_1920.iloc[l] 
    p = ee.Geometry.Point(punto.Longitud,punto.Latitud).buffer(100)
    S2_collection = ee.ImageCollection("COPERNICUS/S2") \
                        .filterBounds(p) \
                        .filterDate('2019-09-01', '2020-04-30') \
                        .sort('system:id', opt_ascending=True) \
                                         
    p_get = p                 
    featCol = S2_collection.map(get_data)
    pdf = get_pd(featCol)
    X.append(get_X(pdf))
    Y.append(punto.Cultivo)

            
X = np.asarray(X)      
Y = np.asarray(Y)

np.save('X_train', X) #save
np.save('y_train', Y)   

# %% create augmented dataset
N = 10 #number of augmented samples per real sample
augmenter = (
      TimeWarp(n_speed_change = 2, max_speed_ratio=1.5) * N # perform 10 time warp augmentations
     )

y_aug = []
X_aug = augmenter.augment(X)

for i in range(X.shape[0]):
    for j in range(N):
        y_aug.append(Y[i]) # create new y array
y_aug = np.asarray(y_aug)

X2 = np.vstack([X, X_aug]) #join real and augmented arrays
y2 = np.hstack([Y, y_aug])

np.save('X_trainAug', X2) #save
np.save('y_trainAug', y2)

