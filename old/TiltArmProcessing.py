import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.stats import linregress
from ptarg import phase_analysis, diff_phase_analysis, phase_analysis_full_scene

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#Tilt Logger Data Directory
dir = '/local-scratch/users/aplourde/field_data/'
files = glob.glob(dir + '/*/*inclinometer*.csv')

# Analysis Period
start_date = pd.to_datetime('2018-08-27')
end_date = pd.to_datetime('2020-07-15')

#global constants
arm_length = 1500 # mm
pivot_height = 115 # mm , taken from gruber, no value available in notes
frequency = 5.4049992e+09

# TODO individual par files!
wavelength = c/frequency
incidence_angle = 26.9
max_insar_disp = 2 * np.pi * (-wavelength / 4 / np.pi) / np.cos(np.radians(incidence_angle))

def importData(file):
    df = pd.read_csv(file, skiprows=14)

    # ensure consistent types
    df['Tilt 1'] = np.float64(df['Tilt 1'])
    df['Tilt 2'] = np.float64(df['Tilt 2'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    return df           

def convertAngles(angle_rads):
    # convert to radians
    #angle_rads = np.radians(angle_degrees)

    # vertical distance of arm end from horizontal
    dYL = np.sin(angle_rads) * arm_length # could use small angle approximation sinx = x

    # horizontal distance of actual arm end from horizontal arm end
    dX = (1 - np.cos(angle_rads)) * arm_length

    # vertical distance added by double pivot
    dYP = np.sqrt(pivot_height**2 - dX**2)

    # total vertical displacement
    dY = dYL + dYP

    return dY

def processData(file):
    data = importData(file)

    # calculate deflection
    data['VerticalDeflection_mm'] = convertAngles(data['Tilt 1'])

    # get rid of blank rows
    data = data.dropna(subset=['Tilt 1', 'VerticalDeflection_mm'])

    # ensure data is in order
    data = data.sort_values(by=['TIMESTAMP'])

    # truncate data to start/end date
    data = data.loc[data['TIMESTAMP'] >= start_date]
    data = data.loc[data['TIMESTAMP'] <= end_date]

    # throwaway errouneous data (sensor may malfuction below -40C)
    stdev = np.std(data['VerticalDeflection_mm'])
    mean = np.mean(data['VerticalDeflection_mm'])
    data = data.loc[data['VerticalDeflection_mm'] > mean - 4*stdev] 
    data = data.loc[data['VerticalDeflection_mm'] < mean + 4*stdev]
    data = data.reset_index()

    # calculate relative deflection
    data['VerticalDeflection_mm'] = data['VerticalDeflection_mm'] - data['VerticalDeflection_mm'][0]

    return data

def getDailyRanges():
    pass

def getDailyValues(df):

    df.index = df['TIMESTAMP']
    resampled = df.resample('D').median()

    out = pd.DataFrame(data = {'TIMESTAMP': resampled.index, 'VerticalDeflection_mm': resampled['VerticalDeflection_mm']})
    
    resampled = df.resample('D').mean()
    out['LoggerTemp_C'] = resampled['LOGGER TEMP']

    return out

def errorDueToTermalExpansion():
    pass


def getInSAR():
    insar_dates, insar_phase = phase_analysis_full_scene()
    timeseries = [date[1] for date in insar_dates]
    timeseries.insert(0, insar_dates[0][0])
    insar_phase.insert(0, 0)

    df = pd.DataFrame(data={'date': timeseries, 'insar': insar_phase})
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.loc[df['date'] >= start_date]
    df = df.loc[df['date'] < end_date]

    df['insar_disp_m'] = df['insar'] * (-wavelength/4/np.pi)/np.cos(np.radians(incidence_angle))
    df['insar_disp_mm'] = df['insar_disp_m'] * 1000

    return df

def combineData(tilt, insar):

    tilt = tilt.drop(columns=['TIMESTAMP'])
    tilt = tilt.reset_index()

    insar = insar.rename(columns={'date':'TIMESTAMP'})

    df = pd.merge(tilt, insar, on='TIMESTAMP', how='inner' )
    
    df = df.rename(columns={'VerticalDeflection_mm':'tilt_disp_mm'})

    df['wrapped'] = False
    df['insar_disp_mm_unwrap'] = df['insar_disp_mm']

    for i in range(len(df['tilt_disp_mm'])-1):
        relative_tilt_disp = df['tilt_disp_mm'][i+1] - df['tilt_disp_mm'][i]
        if np.abs(relative_tilt_disp)>np.abs(max_insar_disp*1000):
            df['wrapped'].iloc[i+1] = True
            print(relative_tilt_disp)
            modulo = np.round(np.abs(relative_tilt_disp)/np.abs(max_insar_disp*1000),0)
            #print(modulo)
            if relative_tilt_disp < 0:
                df['insar_disp_mm_unwrap'].iloc[i+1] = df['insar_disp_mm_unwrap'][i+1] - np.abs(max_insar_disp*1000)*modulo
            elif relative_tilt_disp > 0:
                df['insar_disp_mm_unwrap'].iloc[i+1] = df['insar_disp_mm_unwrap'][i+1] + np.abs(max_insar_disp*1000)*modulo
    #df['insar_disp_mm_unwrap'][2] = df['insar_disp_mm_unwrap'][2] + np.abs(max_insar_disp*1000)
    #df = df.drop(index = 5)
    df['insar_disp_mm_unwrap_cumsum'] = np.cumsum(df['insar_disp_mm_unwrap'])
    df['insar_disp_mm_cumsum'] = np.cumsum(df['insar_disp_mm'])

    df['tilt_disp_mm_rel'] = df['tilt_disp_mm'].diff()
    df['tilt_disp_mm_rel'][0] = 0
    if True:
        res = linregress(df['tilt_disp_mm_rel'],df['insar_disp_mm_unwrap'])
        print(df['tilt_disp_mm'].diff())
        x = np.linspace(-40, 40, 10)
        y = res[0]*x + res[1]

        #plt.plot([-40,40],[-40,40], '--', color = 'lightgray', label = 'y = x')    
        #print(list(df))
        #plt.scatter(df['tilt_disp_mm_rel'],df['insar_disp_mm_unwrap'])
        #plt.plot(x,y, 'k', label = 'y = {}x + {}'.format(round(res[0],2), round(res[1],0)))
        #plt.legend()
        #plt.xlabel('In-Situ Displacement (mm)')
        #plt.ylabel('InSAR Displacement (mm)')
        #plt.show()
    return df


def combineData_int(tilt, insar):
    tilt = tilt.drop(columns=['TIMESTAMP'])
    tilt = tilt.reset_index()

    insar = insar.rename(columns={'date': 'TIMESTAMP'})

    df = pd.merge(tilt, insar, on='TIMESTAMP', how='inner')

    df = df.rename(columns={'VerticalDeflection_mm': 'tilt_disp_mm'})

    df['wrapped'] = False
    df['insar_disp_mm_unwrap'] = df['insar_disp_mm']

    for i in range(len(df['tilt_disp_mm'])):
        d_tilt = df['tilt_disp_mm'][i]
        d_insar = df['insar_disp_mm'][i]

        while d_insar < d_tilt:
            d_insar += abs(max_insar_disp)*1000
        upper = d_insar - d_tilt
        lower = d_tilt - (d_insar - abs(max_insar_disp*1000))
        if upper < lower:
            df['insar_disp_mm_unwrap'][i] = d_insar
        else:
            df['insar_disp_mm_unwrap'][i] = d_insar - abs(max_insar_disp)*1000



    df['insar_disp_mm_unwrap_cumsum'] = df['insar_disp_mm_unwrap']
    df['insar_disp_mm_cumsum'] = np.cumsum(df['insar_disp_mm'])

    df['tilt_disp_mm_rel'] = df['tilt_disp_mm'].diff()
    df['tilt_disp_mm_rel'][0] = 0

    return df

if __name__ == "__main__":
    do_piecewise = False
    num_axes = 1

    # tilt logger data
    site1 = processData(glob.glob(os.path.join(tilt_dir, 'site_1/site_1_inclinometer_processed.csv'))[0])
    #site2 = processData(glob.glob(os.path.join(tilt_dir, "*" + site2_id + "*"))[0])
    #site3 = processData(glob.glob(os.path.join(tilt_dir, "*" + site3_id + "*"))[0])

    # process daily values
    site1_day = getDailyValues(site1)
    site2_day = getDailyValues(site2)
    site3_day = getDailyValues(site3)

    # insar data
    site1_insar = getInSAR()

    # map insar to tiltlogger
    site1_combined = combineData_int(site1_day, site1_insar)
    #site2_combined = combineData(site2_day, site2_insar)
    #site3_combined = combineData(site3_day, site3_insar)


    fig, axes = plt.subplots(num_axes, 1, sharex='col', sharey=False, gridspec_kw = {'wspace':0, 'hspace':0})

    fig.suptitle("Vertical Deflection\n")
    twin_ax = []
    if num_axes > 1:
        axes[0].plot(site1_day['TIMESTAMP'], site1_day['VerticalDeflection_mm'], color = 'red', label = "Site 1: Homogenous Region")
        #axes[0].plot(site1_combined['TIMESTAMP'], site1_combined['insar_disp_mm_cumsum'], '--k', label='Insar')
        #axes[0].plot(site1_combined['TIMESTAMP'], site1_combined['insar_disp_mm_unwrap_cumsum'], color='black', label='Insar "Unwrapped"')
        axes[1].plot(site2_day['TIMESTAMP'], site2_day['VerticalDeflection_mm'], color = 'blue', label = "Site 2: Low Ground")
        #axes[1].plot(site2_combined['TIMESTAMP'], site2_combined['insar_disp_mm_cumsum'], '--k', label='Insar')
        #axes[1].plot(site2_combined['TIMESTAMP'], site2_combined['insar_disp_mm_unwrap_cumsum'], color='black', label='Insar "Unwrapped"')
        axes[2].plot(site3_day['TIMESTAMP'], site3_day['VerticalDeflection_mm'], color = 'lime', label = "Site 3: High Ground")
        #axes[2].plot(site3_combined['TIMESTAMP'], site3_combined['insar_disp_mm_cumsum'], '--k', label='Insar')
        #axes[2].plot(site3_combined['TIMESTAMP'], site3_combined['insar_disp_mm_unwrap_cumsum'], color='black', label='Insar "Unwrapped"')
        axes[2].set_xlabel("Date")
        for i in range(len(axes)):
            axes[i].label_outer()
            axes[i].set_ylabel("Elevation change [mm]")
            twin_ax.append(axes[i].twinx())
            twin_ax[i].set_ylabel("Logger temperature [C]")
            axes[i].set_zorder(twin_ax[i].get_zorder()+1) # plot temp behind deflection
            axes[i].patch.set_visible(False)
            axes[i].legend()
            #twin_ax[i].plot([start_date, end_date], [0,0], '--', color = 'lightgrey')
            #axes[i].plot([start_date, end_date], [0,0], color = 'black')
        twin_ax[0].plot(site1_day['TIMESTAMP'], site1_day['LoggerTemp_C'], color = 'grey')
        twin_ax[1].plot(site2_day['TIMESTAMP'], site2_day['LoggerTemp_C'], color = 'grey')
        twin_ax[2].plot(site3_day['TIMESTAMP'], site3_day['LoggerTemp_C'], color = 'grey')
    else:
        
        axes.plot(site1_day['TIMESTAMP'], site1_day['VerticalDeflection_mm'], color = 'red', label = "Tilt Logger")
        axes.plot(site1_combined['TIMESTAMP'], site1_combined['insar_disp_mm_cumsum'], '--k', label='Insar')
        axes.plot(site1_combined['TIMESTAMP'], site1_combined['insar_disp_mm_unwrap_cumsum'], color='black', label='Insar "Unwrapped"')
        axes.scatter(site1_combined['TIMESTAMP'], site1_combined['insar_disp_mm_unwrap_cumsum'], color='black')
        #if do_piecewise:
        #    for i in range(len(site1_combined['TIMESTAMP'])-1):
        #        tilt = site1_combined['tilt_disp_mm'][i]
        #        insar = site1_combined['tilt_disp_mm'][i] + site1_combined['insar_disp_mm_unwrap'][i+1]
        #        axes.plot([site1_combined['TIMESTAMP'][i],site1_combined['TIMESTAMP'][i+1]],[tilt,insar], color = 'blue')
        #        axes.plot([site1_combined['TIMESTAMP'][i],site1_combined['TIMESTAMP'][i+1]],[site1_combined['tilt_disp_mm'][i],site1_combined['tilt_disp_mm'][i+1]], color = 'red')
        #    axes.plot([],[], color = "red", label = "tilt logger")
        #    axes.plot([],[], color = "blue", label = "InSAR Pieces")
        axes.label_outer()
        axes.set_ylabel("Elevation change [mm]")
        if False:
            twin_ax = axes.twinx()
            twin_ax.plot(site1_day['TIMESTAMP'], site1_day['LoggerTemp_C'], color = 'grey')
            twin_ax.set_ylabel("Logger temperature [C]")
            axes.set_zorder(twin_ax.get_zorder()+1) # plot temp behind deflection
            axes.patch.set_visible(False)
        axes.legend()
        axes.grid()



    plt.show()



    #if os.path.exists('processed_inclinometer.csv') and False:
    #    pass
    #else:
    #    tilt_df = pd.DataFrame(data={'site1':site1_day['VerticalDeflection_mm'],
    #                             'site2':site2_day['VerticalDeflection_mm'],
    #                             'site3':site3_day['VerticalDeflection_mm']})
    #    tilt_df.to_csv('processed_i