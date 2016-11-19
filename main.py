import os 
import collections
import time
import datetime
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
import numpy as np
from math import pi

## TODO: Interpolate the data 
## TODO: Calcluate the DFA
## TODO: (optional) Fit the data / moving average for visualizing
## TODO: Calculate the entropy?
## TODO: Write function to export data

phs_path = r'C:\Users\Terence\Dropbox\Fall 2016\EECS 495 Personal Health Systems\Final Project\patient1\day 1\\'
data = collections.OrderedDict()
data['time'] = []
raw_values = []
for csv_file in os.listdir(phs_path):
    if csv_file == 'Data_ActivityRawType.csv':
        data_key = 'type'
    elif csv_file == 'Data_ActivityStepsPerMinute.csv':
        data_key = 'steps'
    elif csv_file == 'Data_HeartRate.csv':
        data_key = 'hr'
    # intitalize lists
    data[data_key + '_raw'] = []
    data[data_key] = []
    file_counter = 0 
    with open(phs_path + csv_file,'r') as r:
        counter = 0
        #line_count = 0 # for debugging
        for line in r:
            temp = line.split(',')
            if file_counter == 0:
                raw_values.append(temp[0])
                file_counter += 1
            data[data_key + '_raw'].append(temp[0]) # append raw UNIX times
            data[data_key].append(temp[1][:-1]) # append datapoint
            #line_count += 1 # for debugging
        r.close()

time_unix_orig = min(raw_values)
print('Done reading in patient data')

# TODO: Do this in time of day
# Subtract relative to the time of first sample 
for csv_file in os.listdir(phs_path):
    if csv_file == 'Data_ActivityRawType.csv':
        data_key = 'type'
    elif csv_file == 'Data_ActivityStepsPerMinute.csv':
        data_key = 'steps'
    elif csv_file == 'Data_HeartRate.csv':
        data_key = 'hr'
    
    data[data_key + '_time'] = [] # initialize lists
    # Get the first time 
    time_obj = datetime.datetime.fromtimestamp(float(time_unix_orig)/1000)
    time_org_val = time_obj.day*24*60*60 + time_obj.hour*60*60 + time_obj.minute*60 + time_obj.second + time_obj.microsecond / (1000000)
    for i in data[data_key + '_raw']:
        # Subtract the original time to get relative difference
        time_obj = datetime.datetime.fromtimestamp(float(i)/1000)
        time_val = time_obj.day*24*60*60 + time_obj.hour*60*60 + time_obj.minute*60 + time_obj.second + time_obj.microsecond / (1000000)
        data[data_key + '_time'].append(time_val - float(time_org_val))
    
    # Get rid of the types > 6
    line_counter = 0
    if data_key == 'type':
        for i in data[data_key]:
            if line_counter == 0: # Assume first datapoint is correct
                line_counter += 1 
                continue
            else: 
                if int(data[data_key][line_counter]) > 6: # set datapoint > 6 to previous type
                    data[data_key][line_counter] = data[data_key][line_counter-1] 
                line_counter += 1

## For poincare plot
data['RRi'] = data['hr'][:-1]
data['RRii'] = data['hr'][1:]

## Get the periodogram
ang_freqs = np.linspace(0.01,0.5*2*pi, 250)
freqs = ang_freqs / (2*pi)
hrs = np.array(data['hr'],dtype = np.float64)
periodogram = lombscargle(np.array(data['hr_time']), hrs - np.mean(hrs), ang_freqs)
data['freqs'] = freqs
data['periodogram'] = periodogram

## HR
plt.figure()
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
