#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:00:32 2019

@author: osama
"""
import matplotlib
matplotlib.use('agg')

import pickle, pandas as pd, numpy as np, time,os, matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from datetime import datetime, timedelta
from numpy.random import seed
from tensorflow import set_random_seed



# setting seeds ## 
seed(123)
set_random_seed(123)

path = os.getcwd()
# so that whole array gets printed
np.set_printoptions(threshold=np.nan)



def main():
    
    # importing file
    system_data = importing_file()
    
    # data preprocessing
    df_nn = preprocess (system_data)
    # Output
    y_train = df_nn[['Load']].values 
    # Input 
    x_train=df_nn[['no_of_month','day_of_month','day_of_week','time_index']].values
    x_train = x_train.reshape(x_train.shape[0] , x_train.shape[1])


    print("x_train: " + str(x_train) + "\n y_train: " + str(y_train))
    
    ## NN model ## 
    # initialization # 
    nn_model = Sequential()
    nn_model.add(Dense(300, activation="relu",input_shape=(4,))) # because we have 4 features
    nn_model.add(Dropout(0.3))
    nn_model.add(Dense(150, activation="relu"))
    nn_model.add(Dense(50, activation="relu"))
    nn_model.add(Dense(30, activation="relu"))
    nn_model.add(Dense(1))
        
    optimizer = optimizers.RMSprop(0.001)      
    nn_model.compile(loss='mse',optimizer=optimizer,metrics=['mae']) 
    
    start_time = time.time()
    
    ## training ##
    nn_model.fit(x_train,y_train,epochs=200,batch_size=20,verbose=1,shuffle=False)
    
    duration = round( time.time() - start_time,2)
    
    print ("Model fitting took : " + str(duration) + " time.")
    
    ## testing ##
    x_test = extracting_testing_data()
    
    ### making predictions for the next day
    predictions=nn_model.predict(x_test,batch_size=1)
    
    # plotting the predictions
    plt.figure(1)
    plt.xlabel('predictions')
    plt.plot(predictions)
    plt.savefig(path+'/predictions.png')
    plt.close()
    

def extracting_testing_data():
    
    some_random_date = datetime(2018,12,19)
    x_test = list()
    datetimes = next_25_hours_datetimes(some_random_date) 
    
    for i in range(len(datetimes)):
        x_test.append( [datetimes[i].month, datetimes[i].day,datetimes[i].weekday(),int( datetimes[i].minute/10 + datetimes[i].hour*6)] )
   
    # reshaping for NN
    x_test = np.asarray(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
    
    return x_test
    
    
def importing_file():
    try: 
        with open('system_data.pkl', 'rb') as f:
            system_data = pickle.load(f)
    except:
        raise Exception("File not present .. Exiting")
    return system_data    


##### Preprocessing functions ##### 
    
# returns 10 minutes sampled array of datetimes 24 hours into the future
def next_25_hours_datetimes(begin_time,next_hours = 25, freq = 10):
    start_time = to_previous_10th_minute (begin_time)
    datetimes = [start_time + timedelta(minutes=x) for x in range(0, next_hours*6*10,freq)]
    return datetimes

# returns start of previous 10th minutes for e.g. 12:44 will return 12:40
def to_previous_10th_minute(some_datetime):
    some_datetime = (datetime(some_datetime.year,some_datetime.month,some_datetime.day,some_datetime.hour,some_datetime.minute - some_datetime.minute%10))
    return some_datetime

    
def preprocess(data):
    df_data = pd.DataFrame(np.array(data).reshape(len(data),2), columns = ['Timestamp','Load'])
    df_data['no_of_month']=df_data.Timestamp.map(lambda x:x.month)
    df_data['day_of_month']=df_data.Timestamp.map(lambda x:x.day)
    df_data['day_of_week']=df_data.Timestamp.map(lambda x:x.weekday())
    df_data['time_index']=df_data.Timestamp.map(lambda x: int ((x.hour*60+x.minute)/10))
    
    df_data=df_data.drop(['Timestamp'],1)
    return df_data

if __name__ == "__main__":
    main()
