import os 
import collections
import time
import datetime
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
import numpy as np
from math import pi
import pandas as pd

## TODO: Calcluate the DFA
## TODO: (optional) Fit the data / moving average for visualizing
## TODO: Calculate the entropy?
## TODO: (optional) Fit ellipse? -- seems more work than its worth since we're doing static

#phs_path = r'C:\Users\Terence\Dropbox\Fall 2016\EECS 495 Personal Health Systems\Final Project\patient3\\'
phs_path = r'c:\Users\B\Dropbox\EECS495-PersonalHealth\project\INLIFErecordings\patient3\\'
data = collections.OrderedDict() # data is a dictionary of dictionaries
data['day1'] = collections.OrderedDict()
data['day2'] = collections.OrderedDict()

for day in os.listdir(phs_path):
    filepath = phs_path + day + '\\'
    raw_values = []
    for csv_file in os.listdir(filepath):

        if csv_file == 'Data_ActivityRawType.csv':
            data_key = 'type'
        elif csv_file == 'Data_ActivityStepsPerMinute.csv':
            data_key = 'steps'
        elif csv_file == 'Data_HeartRate.csv':
            data_key = 'hr'
        else:
            continue

        ### BEGUM ###
        
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


        ### BEGUM ###


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
            data_key = 'type'
        elif csv_file == 'Data_ActivityStepsPerMinute.csv':
            data_key = 'steps'
        elif csv_file == 'Data_HeartRate.csv':
            data_key = 'hr'
        else:
            continue
        
        data[day][data_key + '_finaltime'] = data[day][data_key + '_time'] - daytime_orig
        data[day][data_key + '_daytime'] = data[day][data_key + '_finaltime'] + time_of_day # time of day in 24 hr format
        if data_key == 'type':
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
    data[day]['hr_time_interp'] = full_array[:,0]

    concat_time = np.append(data[day]['hr_daytime'],data[day]['steps_daytime'])
    concat_steps = np.interp(concat_time ,data[day]['steps_daytime'],data[day]['steps'])
    full_array = np.column_stack([concat_time, concat_steps])
    full_array = full_array[np.argsort(full_array[:,0])]
    data[day]['steps_interp'] = full_array[:,1]
    data[day]['steps_time_interp'] = full_array[:,0]

    concat_time = np.append(data[day]['hr_daytime'],data[day]['type_daytime'])
    concat_type = np.interp(concat_time ,data[day]['type_daytime'],data[day]['type'])
    full_array = np.column_stack([concat_time, concat_type])
    full_array = full_array[np.argsort(full_array[:,0])]
    data[day]['type_interp'] = full_array[:,1]
    data[day]['type_time_interp'] = full_array[:,0]

    data[day]['RRi_interp'] = data[day]['hr_interp'][:-1]
    data[day]['RRii_interp'] = data[day]['hr_interp'][1:]
    
    ## Get the periodogram
    ang_freqs = np.linspace(0.01*2*pi,0.5*2*pi, 10000)
    freqs = ang_freqs / (2*pi)
    hrs = np.array(data[day]['hr_interp'],dtype = np.float64)
    periodogram = lombscargle(np.array(data[day]['hr_time_interp']), (hrs - np.mean(hrs)), ang_freqs)
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
    plt.plot(data[day]['hr_time_interp'], data[day]['hr_interp'])
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
            data_key = 'type'
        elif csv_file == 'Data_ActivityStepsPerMinute.csv':
            data_key = 'steps'
        elif csv_file == 'Data_HeartRate.csv':
            data_key = 'hr'
        else:
            continue
        
        export_keys.extend([(data_key + '_daytime'), data_key, (data_key + '_interp'),  (data_key + '_time_interp')])
        
    with open(filepath + day + '.csv', 'w') as w:
        for key in export_keys:
            data_list = [key]
            if type(data[day][key]) == np.float64:
                data_list.append(str(data[day][key]))
            else:
                data_list.extend(list(map(str, data[day][key])))
            w.write((','.join(data_list)) + '\n')

    ### BEGUM ###
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
    ### BEGUM ###
    print('done!')

## HR
#plt.figure()
#plt.plot(data['hr_time'],data['hr'],'--*')
#plt.xlabel('Time (seconds')
#plt.ylabel('Heart Rate [bpm]')
#plt.title('HR vs time')
#plt.grid()

## Activity type
#plt.figure()
#plt.plot(data['type_time'],data['type'],'*')
#plt.xlabel('Time (seconds)')
#plt.ylabel('Type')
#plt.title('Type vs time')
#plt.grid()

## Steps per minute
#plt.figure()
#plt.plot(data['steps_time'],data['steps'],'*')
#plt.xlabel('Time (seconds')
#plt.ylabel('Steps Per Minute')
#plt.title('Steps per minute vs Time')
#plt.grid()

#plt.show()

## Shared axes of activity type and steps
#f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, sharey = False)
#ax1.plot(data['type_time'],data['type'],'--*')
#ax1.set_title('Activity and Steps')
#ax1.set_ylabel('Activity Type')
#ax1.grid()
#ax2.plot(data['steps_time'],data['steps'],'--*')
#ax2.set_ylabel('Steps per minute')
#ax2.grid()
#ax3.plot(data['hr_time'],data['hr'],'--*')
#ax3.set_ylabel('HR [bpm]')
#ax3.grid()
#f.subplots_adjust(hspace = 0.05)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible = False)

## Poincare plot 
#plt.figure()
#plt.plot(RRi, RRii, '.')
#plt.xlabel('RR(i)')
#plt.ylabel('RR(i+1)')
#plt.title('Poincare Plot')
#plt.grid()

## Lomb periodogram
#plt.figure()
#plt.plot(data['freqs'], data['periodogram'])
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Power')
#plt.grid()
  
## Get the run and walking ones
#run_steps = [int(x) for x, y in zip(data['steps'], data['type']) if y == '2']
#run_times = [int(x) for x, y in zip(data['steps_time'], data['type']) if y == '2']
#walk_steps = [int(x) for x, y in zip(data['steps'], data['type']) if y == '1']
#walk_times = [int(x) for x, y in zip(data['steps_time'], data['type']) if y == '1']
#plt.figure()
#ax = plt.plot(run_times, run_steps,'--b*', walk_times, walk_steps, '--r*')
#plt.legend(ax, ['Run', 'Walk'])


## Shared axes of activity type and steps
#f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = False)
#min_time = min([float(x) for x, y in zip(data['steps_time'], data['type']) if y == '2' or y == '1'])
#min_time = max([float(x) for x, y in zip(data['steps_time'], data['type']) if y == '2' or y == '1'])
#active_steps = [int(x) for x, y in zip(data['hr'], data['hr_time']) if y >= min_time and y <= max_time]
#active_times = [float(x) for x, y in zip(data['hr_time'], data['hr_time']) if y >= min_time and y <= max_time]

#ax1.plot(active_times, active_steps, '--*')
#ax1.set_title('HR during active session')
#ax1.set_ylabel('HR [bpm]')
#ax1.grid()
#ax2.plot(data['steps_time'],data['steps'],'--*')
#ax2.set_ylabel('Steps per minute')
#ax2.grid()
#f.subplots_adjust(hspace = 0.05)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible = False)

#unique_types = list(set(data['type']))
#for i in unique_types:
#    temp_time = [float(x) for x, y in zip(data['type_time'], data['type']) if y == str(i)]
#    temp_steps = [float(x) for x, y in zip(data['steps'], data['type']) if y == str(i)]
#    print('Mean steps for ' + str(i) + ' is ' + str(np.mean(np.asarray(temp_steps))))
#plt.show()