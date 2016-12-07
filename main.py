##
## CardioCare 
## Authors: Terence Chan, Begum Egilmez
##
## Last Modified: 12/06/16
##
import os 
import collections
import time
import datetime
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
import numpy as np
from math import pi
import pandas as pd

#IMPORTANT! Before using this code please modify phs_path variable to show the directory 
#where patient information is stored in your local.

#phs_path = r'C:\Users\Terence\Dropbox\Fall 2016\EECS 495 Personal Health Systems\Final Project\patient3\\'
phs_path = r'c:\Users\B\Dropbox\EECS495-PersonalHealth\project\INLIFErecordings\patient5\\'
data = collections.OrderedDict() # data is a dictionary of dictionaries
data['day1'] = collections.OrderedDict()
data['day2'] = collections.OrderedDict()

for day in os.listdir(phs_path):
    filepath = phs_path + day + '\\'
    raw_values = []
    for csv_file in os.listdir(filepath):

        if csv_file == 'Data_ActivityRawType.csv':
            data_key = 'activitytype'
        elif csv_file == 'Data_ActivityStepsPerMinute.csv':
            data_key = 'steps'
        elif csv_file == 'Data_HeartRate.csv':
            data_key = 'hr'
        else:
            continue

        if data_key == 'hr':              
            # Read file to a data frame
            hrFrame = pd.read_csv(filepath + csv_file)
            hrFrame.columns = ['timestamp', 'heart']

            # Convert timestamp to datetime
            hrFrame['dt'] = pd.to_datetime(hrFrame['timestamp'], unit = 'ms')
            hrFrame['hrOfDay'] = hrFrame['dt'].dt.hour
            # Just reserve a column for the day portion. 8 does not mean anything
            hrFrame['dayPortion'] = hrFrame['hrOfDay'] // 8
            # Actually set the day portion column
            hrFrame.loc[(hrFrame['hrOfDay'] >= 6) & (hrFrame['hrOfDay'] < 14), 'dayPortion'] = 0
            hrFrame.loc[(hrFrame['hrOfDay'] >= 14) & (hrFrame['hrOfDay'] < 20), 'dayPortion'] = 1
            hrFrame.loc[(hrFrame['hrOfDay'] >= 20) | (hrFrame['hrOfDay'] < 6), 'dayPortion'] = 2

            # Calculate the time difference between adjacent measurements in ms
            hrFrame['timestampDiff'] = hrFrame['timestamp'].shift(-1) -  hrFrame['timestamp']
            # Calculate the square root of the difference
            hrFrame['timestampDiffSquared'] = np.square(hrFrame['timestampDiff'])

            # Calculcate RMSSD for each portion of the day.
            morningRMSSD = np.sqrt(np.mean(hrFrame[(hrFrame['dayPortion'] == 0) & (hrFrame['timestampDiff'] < 1500)]['timestampDiffSquared']))
            afternoonRMSSD = np.sqrt(np.mean(hrFrame[(hrFrame['dayPortion'] == 1) & (hrFrame['timestampDiff'] < 1500)]['timestampDiffSquared']))
            nightRMSSD = np.sqrt(np.mean(hrFrame[(hrFrame['dayPortion'] == 2) & (hrFrame['timestampDiff'] < 1500)]['timestampDiffSquared']))

            # Calculcate PNN50 for each portion of the day
            morningPNN50 = float(len(hrFrame[(hrFrame['dayPortion'] == 0) & (hrFrame['timestampDiff'] < 1500)  
                                     & (hrFrame['timestampDiff'] > 50)])) / (float(len(hrFrame[(hrFrame['dayPortion'] == 0)  
                                                                                              & (hrFrame['timestampDiff'] < 1500)]))+1) #smoothing
            afternoonPNN50 = float(len(hrFrame[(hrFrame['dayPortion'] == 1) & (hrFrame['timestampDiff'] < 1500)  
                                     & (hrFrame['timestampDiff'] > 50)])) / (float(len(hrFrame[(hrFrame['dayPortion'] == 1)  
                                                                                              & (hrFrame['timestampDiff'] < 1500)]))+1)
            nightPNN50 = float(len(hrFrame[(hrFrame['dayPortion'] == 2) & (hrFrame['timestampDiff'] < 1500)  
                                     & (hrFrame['timestampDiff'] > 50)])) / (float(len(hrFrame[(hrFrame['dayPortion'] == 2)  
                                                                                              & (hrFrame['timestampDiff'] < 1500)]))+1)
            # Calculate the median heart rate for each portion of the day
            medianHr = hrFrame.groupby(by = ['dayPortion']).agg({'heart' : np.median})

            # Calculate the number of measurements where patient had > 180 heart rate for each day portion.
            greaterThan180 = hrFrame.groupby(by = ['dayPortion']).agg({'heart':lambda x: (x > 180).sum()})

        elif data_key == 'activitytype':
            # Read file to a data frame
            activityFrame = pd.read_csv(filepath + csv_file)
            activityFrame.columns = ['timestamp', 'activity']

            # Convert timestamp to datetime
            activityFrame['dt'] = pd.to_datetime(activityFrame['timestamp'], unit = 'ms')
            activityFrame['hrOfDay'] = activityFrame['dt'].dt.hour
            # Just reserve a column for the day portion. 8 does not mean anything
            activityFrame['dayPortion'] = activityFrame['hrOfDay'] // 8
            # Actually set the day portion column
            activityFrame.loc[(activityFrame['hrOfDay'] >= 6) & (activityFrame['hrOfDay'] < 14), 'dayPortion'] = 0
            activityFrame.loc[(activityFrame['hrOfDay'] >= 14) & (activityFrame['hrOfDay'] < 20), 'dayPortion'] = 1
            activityFrame.loc[(activityFrame['hrOfDay'] >= 20) | (activityFrame['hrOfDay'] < 6), 'dayPortion'] = 2

            silent = activityFrame.groupby(by = ['dayPortion']).agg({'activity':lambda x:(x==0).sum()})
            walking = activityFrame.groupby(by = ['dayPortion']).agg({'activity':lambda x:(x==1).sum()})
            running = activityFrame.groupby(by = ['dayPortion']).agg({'activity':lambda x:(x==2).sum()})
            non_wear = activityFrame.groupby(by = ['dayPortion']).agg({'activity':lambda x:(x==3).sum()})
            REM = activityFrame.groupby(by = ['dayPortion']).agg({'activity':lambda x:(x==4).sum()})
            NREM = activityFrame.groupby(by = ['dayPortion']).agg({'activity':lambda x:(x==5).sum()})
            charging = activityFrame.groupby(by = ['dayPortion']).agg({'activity':lambda x:(x==6).sum()})
            undefined = activityFrame.groupby(by = ['dayPortion']).agg({'activity':lambda x:(x>6).sum()})

            lSilent = silent['activity'].tolist()
            lWalking = walking['activity'].tolist()
            lRunning = running['activity'].tolist()
            lNonWear = non_wear['activity'].tolist()
            lREM = REM['activity'].tolist()
            lNREM = NREM['activity'].tolist()
            lCharging = charging['activity'].tolist()
            lUndefined = undefined['activity'].tolist()

            total_morning = lSilent[0] + lWalking[0] + lRunning[0] + lNonWear[0] + lREM[0] + lNREM[0] + lCharging[0] + lUndefined[0]
            total_afternoon = lSilent[1] + lWalking[1] + lRunning[1] + lNonWear[1] + lREM[1] + lNREM[1] + lCharging[1] + lUndefined[1]
            total_night = lSilent[2] + lWalking[2] + lRunning[2] + lNonWear[2] + lREM[2] + lNREM[2] + lCharging[2] + lUndefined[2]

            pSilent_m = (lSilent[0]/total_morning)*100
            pSilent_a = (lSilent[1]/total_afternoon)*100
            pSilent_n = (lSilent[2]/total_night)*100
            
            pWalking_m = (lWalking[0]/total_morning)*100
            pWalking_a = (lWalking[1]/total_afternoon)*100
            pWalking_n = (lWalking[2]/total_night)*100
            
            pRunning_m = (lRunning[0]/total_morning)*100
            pRunning_a = (lRunning[1]/total_afternoon)*100
            pRunning_n = (lRunning[2]/total_night)*100
            
            pNonWear_m = (lNonWear[0]/total_morning)*100
            pNonWear_a = (lNonWear[1]/total_afternoon)*100
            pNonWear_n = (lNonWear[2]/total_night)*100
            
            pREM_m = (lREM[0]/total_morning)*100
            pREM_a = (lREM[1]/total_afternoon)*100
            pREM_n = (lREM[2]/total_night)*100
            
            pNREM_m = (lNREM[0]/total_morning)*100
            pNREM_a = (lNREM[1]/total_afternoon)*100
            pNREM_n = (lNREM[2]/total_night)*100
            
            pCharging_m = (lCharging[0]/total_morning)*100
            pCharging_a = (lCharging[1]/total_afternoon)*100
            pCharging_n = (lCharging[2]/total_night)*100
            
            pUndefined_m = (lUndefined[0]/total_morning)*100
            pUndefined_a = (lUndefined[1]/total_afternoon)*100
            pUndefined_n = (lUndefined[2]/total_night)*100

            silent_allday = lSilent[0]+lSilent[1]+lSilent[2]
            walking_allday = lWalking[0]+lWalking[1]+lWalking[2]
            running_allday = lRunning[0]+ lRunning[1]+lRunning[2]
            nonwear_allday = lNonWear[0]+lNonWear[1]+lNonWear[2]
            REM_allday = lREM[0]+lREM[1]+lREM[2]
            NREM_allday = lNREM[0]+lNREM[1]+lNREM[2]
            charging_allday = lCharging[0]+lCharging[1]+lCharging[2]
            undefined_allday = lUndefined[0]+lUndefined[1]+lUndefined[2]

            total_allday = total_morning + total_night + total_afternoon
        
        ### read ###

        data[day][data_key + '_raw'] = np.empty([0,0], dtype = np.float64)
        data[day][data_key + '_time'] = np.empty([0,0])
        data[day][data_key] = np.empty([0,0], dtype = np.float64)

        start_counter = 0 
        with open(filepath + csv_file, 'r') as r:
            for line in r:
                temp = line.split(',')

                data[day][data_key + '_raw'] = np.append(data[day][data_key + '_raw'], float(temp[0])) # append raw UNIX times
                
                time_obj = datetime.datetime.fromtimestamp(float(temp[0])/1000)
                time_val = time_obj.day*24 + time_obj.hour + time_obj.minute/60 + time_obj.second/(60*60) + time_obj.microsecond/(3600000000)
                data[day][data_key + '_time'] = np.append(data[day][data_key + '_time'], time_val)
                if start_counter == 0:
                    raw_values.append(temp[0])
                    start_counter += 1
                data[day][data_key] = np.append(data[day][data_key], float(temp[1][:-1])) # append datapoint
                #line_count += 1 # for debugging
            r.close()
    
    time_unix_orig = min(raw_values)
    time_obj = datetime.datetime.fromtimestamp(float(time_unix_orig)/1000)
    daytime_orig = time_obj.day*24 + time_obj.hour + time_obj.minute/60 + time_obj.second/(60*60) + time_obj.microsecond/(3600000000)
    time_of_day = time_obj.hour + time_obj.minute/60 + time_obj.second/(60*60) + time_obj.microsecond/(3600000000)

    for csv_file in os.listdir(filepath):

        if csv_file == 'Data_ActivityRawType.csv':
            data_key = 'activitytype'
        elif csv_file == 'Data_ActivityStepsPerMinute.csv':
            data_key = 'steps'
        elif csv_file == 'Data_HeartRate.csv':
            data_key = 'hr'
        else:
            continue
        
        data[day][data_key + '_finaltime'] = data[day][data_key + '_time'] - daytime_orig
        data[day][data_key + '_daytime'] = data[day][data_key + '_finaltime'] + time_of_day # time of day in 24 hr format
        if data_key == 'activitytype':
            data_counter = 0
            for i in range(1, len(data[day][data_key])):
                if data[day][data_key][i] > 6: 
                    data[day][data_key][i] = data[day][data_key][i-1]
   
    ## For poincare plot
    data[day]['RRi'] = data[day]['hr'][:-1]
    data[day]['RRii'] = data[day]['hr'][1:]

    ## Interpolation -- append the times -> interpolate -> sort by time
    concat_time = np.append(data[day]['steps_daytime'],data[day]['hr_daytime'])
    concat_hr = np.interp(concat_time ,data[day]['hr_daytime'],data[day]['hr'])
    full_array = np.column_stack([concat_time, concat_hr])
    full_array = full_array[np.argsort(full_array[:,0])]
    data[day]['hr_interp'] = full_array[:,1]
    data[day]['hr_daytime_interp'] = full_array[:,0]

    concat_time = np.append(data[day]['hr_daytime'],data[day]['steps_daytime'])
    concat_steps = np.interp(concat_time ,data[day]['steps_daytime'],data[day]['steps'])
    full_array = np.column_stack([concat_time, concat_steps])
    full_array = full_array[np.argsort(full_array[:,0])]
    data[day]['steps_interp'] = full_array[:,1]
    data[day]['steps_daytime_interp'] = full_array[:,0]

    concat_time = np.append(data[day]['hr_daytime'],data[day]['activitytype_daytime'])
    concat_type = np.interp(concat_time ,data[day]['activitytype_daytime'],data[day]['activitytype'])
    full_array = np.column_stack([concat_time, concat_type])
    full_array = full_array[np.argsort(full_array[:,0])]
    data[day]['activitytype_interp'] = full_array[:,1]
    data[day]['activitytype_daytime_interp'] = full_array[:,0]

    data[day]['RRi_interp'] = data[day]['hr_interp'][:-1]
    data[day]['RRii_interp'] = data[day]['hr_interp'][1:]
    
    ## Get the periodogram
    ang_freqs = np.linspace(0.01*2*pi,0.5*2*pi, 10000)
    freqs = ang_freqs / (2*pi)
    hrs = np.array(data[day]['hr_interp'],dtype = np.float64)
    periodogram = lombscargle(np.array(data[day]['hr_daytime_interp']), (hrs - np.mean(hrs)), ang_freqs)
    data[day]['freqs'] = freqs
    data[day]['periodogram'] = periodogram
    plt.figure()
    plt.plot(data[day]['freqs'], data[day]['periodogram'],'.')
    plt.title('periodogram')
    plt.grid()
    
    # low frequency power = 0.04 - 0.15 Hz
    # high frequency power = 0.15 - 0.4 Hz
    spacing = data[day]['freqs'][1] - data[day]['freqs'][0]
    low_freqs = [x for x,y in zip(data[day]['periodogram'],data[day]['freqs']) if y >= 0.04 and y <= 0.15]
    high_freqs = [x for x,y in zip(data[day]['periodogram'],data[day]['freqs']) if y >= 0.15 and y <= 0.4]
    data[day]['low_power'] = np.trapz(low_freqs, None, dx = spacing)
    data[day]['high_power'] = np.trapz(high_freqs, None, dx = spacing)
    plt.show()

    # HR
    plt.figure()
    plt.plot(data[day]['hr_daytime_interp'], data[day]['hr_interp'])
    plt.title('HR')
    plt.grid()
    plt.show()

    # Poincare plot 
    plt.figure()
    plt.plot(data[day]['RRi'], data[day]['RRii'], '.')
    plt.xlabel('RR(i)')
    plt.ylabel('RR(i+1)')
    plt.title('Poincare Plot')
    plt.grid()
    plt.show()

    ## TODO: What keys you want to export, add them here! 
    export_keys =  ['RRi', 'RRii', 'RRi_interp', 'RRii_interp', 'freqs', 'periodogram', 'low_power', 'high_power']
    for csv_file in os.listdir(filepath):

        if csv_file == 'Data_ActivityRawType.csv':
            data_key = 'activitytype'
        elif csv_file == 'Data_ActivityStepsPerMinute.csv':
            data_key = 'steps'
        elif csv_file == 'Data_HeartRate.csv':
            data_key = 'hr'
        else:
            continue
        
        export_keys.extend([(data_key + '_daytime'), data_key, (data_key + '_interp'),  (data_key + '_daytime_interp')])
        
    plt.figure()
    plt.plot(data[day]['activitytype_daytime'], data[day]['activitytype'],'b') #,data[day]['activitytype_interp'],'r')
    plt.show()
    with open(filepath + day + '.csv', 'w') as w:
        for key in export_keys:
            data_list = [key]
            if type(data[day][key]) == np.float64:
                data_list.append(str(data[day][key]))
            else:
                data_list.extend(list(map(str, data[day][key])))
            w.write((','.join(data_list)) + '\n')

    
    with open(filepath + day + '.csv', 'a') as w:
        w.write('rMSSD,' + morningRMSSD.astype(str) + ',' + afternoonRMSSD.astype(str) + ','+ nightRMSSD.astype(str) + '\n')
        w.write('pNN50,' + str(morningPNN50) + ',' + str(afternoonPNN50) + ','+ str(nightPNN50) + '\n')
        w.write('Median')
        for heartRate in medianHr['heart'].tolist():
            w.write(','+ str(heartRate))
        w.write('\n')
        w.write('gt180')
        for heartRate in greaterThan180['heart'].tolist():
            w.write(','+ str(heartRate))
        w.write('\n')
        w.write('act_silent_percentage,' + str(pSilent_m) + ',' + str(pSilent_a) + ','+ str(pSilent_n) + '\n')
        w.write('act_walking_percentage,' + str(pWalking_m) + ',' + str(pWalking_a) + ','+ str(pWalking_n) + '\n')
        w.write('act_running_percentage,' + str(pRunning_m) + ',' + str(pRunning_a) + ','+ str(pRunning_n) + '\n')
        w.write('act_nonWear_percentage,' + str(pNonWear_m) + ',' + str(pNonWear_a) + ','+ str(pNonWear_n) + '\n')
        w.write('act_REM_percentage,' + str(pREM_m) + ',' + str(pREM_a) + ','+ str(pREM_n) + '\n')
        w.write('act_NREM_percentage,' + str(pNREM_m) + ',' + str(pNREM_a) + ','+ str(pNREM_n) + '\n')
        w.write('act_charging_percentage,' + str(pCharging_m) + ',' + str(pCharging_a) + ','+ str(pCharging_n) + '\n')
        w.write('act_undefined_percentage,' + str(pUndefined_m) + ',' + str(pUndefined_a) + ','+ str(pUndefined_n) + '\n')
        w.write('act_silent_percentage_allday,' + str((silent_allday/total_allday)*100) + '\n')
        w.write('act_walking_percentage_allday,' + str((walking_allday/total_allday)*100) + '\n')
        w.write('act_running_percentage_allday,' + str((running_allday/total_allday)*100) + '\n')
        w.write('act_nonWear_percentage_allday,' + str((nonwear_allday/total_allday)*100) + '\n')
        w.write('act_REM_percentage_allday,' + str((REM_allday/total_allday)*100) + '\n')
        w.write('act_NREM_percentage_allday,' + str((NREM_allday/total_allday)*100) + '\n')
        w.write('act_charging_percentage_allday,' + str((charging_allday/total_allday)*100) + '\n')
        w.write('act_undefined_percentage_allday,' + str((undefined_allday/total_allday)*100) + '\n')
    
    print('done!')
