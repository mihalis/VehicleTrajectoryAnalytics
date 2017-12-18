import sys                      # system tools 

import numpy as np              # numerical computing with arrays 
import pandas as pd             # dataframe as in R

import matplotlib 
import matplotlib.pyplot as plt # ploting 
import matplotlib as mpl        # ploting 
import seaborn as sns           # ploting 

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import os                       # operating system io commands 
import itertools                # functonal programming tools 

import pdb 

#FIG_SIZE_X,FIG_SIZE_Y = 15, 10

from reportlab.pdfgen import canvas 

from reportlab.lib.pagesizes import letter, A4


class PdfReport(object):

    def __init__(self, outFileName, pagesize):
        
        self._outFileName = outFileName
        self._canvas = canvas.Canvas(outFileName, pagesize=pagesize)
        self._counter = 0

    def addChart(self, chart):
        
        self._counter += 1 

        figure = chart 
        figure.savefig("tmpFigure%d.png" % self._counter)
        #image = open("tmpFigure%d.png" %d, "rb").read()

        #im = Image.fromstring(image)
        #im = Image.open("tmpFigure%d.png" % self._counter)
        
        self._canvas.drawImage("tmpFigure%d.png" % self._counter, 0, 0)        

        os.remove("tmpFigure%d.png" % self._counter) 

        self._canvas.bookmarkPage(str(self._counter))
        self._canvas.addOutlineEntry(chart.getTitle(), str(self._counter), 0, 0)
        self._canvas.showPage()

    def addChart2(self, fileName, title):
        
        self._counter += 1 
        
        self._canvas.drawImage(fileName, 0, 0)        

        self._canvas.bookmarkPage(str(self._counter))
        self._canvas.addOutlineEntry(title, str(self._counter), 0, 0)
        self._canvas.showPage()
        
        
    def write(self):
        
        self._canvas.save()

#    def __del__(self):
#
#        for i in range(self._counter):
#            os.remove("tmpFigure%d.png" % (i + 1)) 


class VTAnalytics:


    @classmethod
    def describe(cls, var, numerical=True, decimals=1):
        
        count = int(var.shape[0])
        na    = int(var.isnull().sum())
        
        if numerical: 
            
            var2 = var.dropna().values 
            mean = np.mean(var2)
            std  = np.std(var2)
            
            min_ =  np.min(var2)   
            pct5 =  np.percentile(var2, 5) 
            pct25 = np.percentile(var2, 25)
            pct50 = np.percentile(var2, 50)
            pct75 = np.percentile(var2, 75) 
            pct95 = np.percentile(var2, 95)

            max_ = np.max(var2)

            values = [count,   na,    mean,   std,   min_,  pct5,   pct25,  pct50,   pct75,    pct95, max_]
            values = [np.round(v, decimals) for v in values]
            #values.insert(na, 0)
            #values.insert(count, 0)
            index = ['count', 'NA', 'mean', 'std', 'min', 'ptc5', 'pct25', 'pct50', 'ptc75', 'pct95', 'max']
            
            df = pd.DataFrame({'stats':values}, index=index)
            
            return df
        else:
            raise Exception("Not implemented yet")

    @classmethod 
    def readNGISIMData(cls, fName1, start_time=None, end_time=None): 
        
        """This function takes raw NGSIM data and prepares them for analysis
        """
        

        colNames = ['VehID', 'FrameID', 'TotalFrames', 'GlobalTime', 'locX', 'locY', 'globX', 
                    'globY', 'vehLength', 'vehWidth', 'vehClass', 'vehSpeed', 'vehAcceleration', 'lane', 
                   'precedingVeh', 'followingVeh', 'spacing', 'headway']
        
        ng = pd.read_csv(fName1, 
                         header=None, names=colNames, delim_whitespace=True)

        # define app specific variables 
        #ng['_spd']  = ng.vehSpeed * 3600 / 5280  # _spd in miles per hour 
        #ng['_acc']  = ng.vehAcceleration         # _acc is in feet per second squared 
        ng['_locy'] = ng.locY  #location along the Y axis in feet 
        ng['_vlen'] = ng.vehLength # vehicle length in feet  
        ng['_lane'] = ng.lane 

        #create a datetime variable 
        #result = [] 
        #for t in ng.GlobalTime.values: 
        #    t = np.datetime64(int(t),'ms') - np.timedelta64(7,'h')
        #    result.append(t)
        ng['_time'] = ng.GlobalTime / 1000 #convert the time into seconds 

        #create a new vehicle id because the one defined in the
        #raw data is wrong 
        newVehID = (ng.groupby(['VehID', 'TotalFrames', 'vehLength'])['locY'].mean()
                            .to_frame()
                            .reset_index()
                            .reset_index())
        newVehID.rename(columns={'index':'_vid'}, inplace=True)
        newVehID.drop(['locY'], axis=1, inplace=True)
        newVehID['_vid']  = newVehID['_vid'] + 1

        ng = ng.merge(newVehID, on=['VehID', 'TotalFrames', 'vehLength'])
        
        #limit the dataset based on the input times 
        if start_time: 
            ng  = ng[(ng._time.values >= start_time)]
        if end_time:
            ng = ng[(ng._time.values <= end_time)]
        
        #calculate dt and remove records with time step greater than 
        # 0.1 seconds 

        ng = ng.sort_values(['_vid', '_time']) 
        ng.index = np.arange(ng.shape[0])

        ng['_dt'] = ng._time.shift(-1) - ng._time
        ng.loc[ng._vid != ng._vid.shift(-1), '_dt'] = np.nan

        #there are some records with dt greater than the 0.1 don't know why 
        if ng._dt.unique().shape[0] > 2:
            print ("Error: the dataset has more than one timestep")
            #raise ValueError("The dataset has more than one timestep") 

        time_step = ng._dt.dropna().unique()[0]

        #calculate speed and acceleration 
        ng['_spd'] = (ng['_locy'].shift(-1) - ng['_locy']) / ng['_dt'] 
        ng['_acc'] = (ng['_spd'].shift(-1) - ng['_spd']) / ng['_dt']  #fpss 
        ng['_spd'] = ng['_spd'] * (3600 / 5280) # mph 

        ng = ng.dropna() 

        #define lane changes 

        ng['_leaveLane'] = 0
        ng.loc[ng._lane != ng._lane.shift(-1), '_leaveLane'] = 1
        ng.loc[ng._vid  != ng._vid.shift(-1), '_leaveLane'] = 0

        ng['_enterLane'] = 0
        ng.loc[ng._lane != ng._lane.shift(1), '_enterLane'] = 1
        ng.loc[ng._vid  != ng._vid.shift(1), '_enterLane'] = 0


        ng = ng.sort_values(['_lane', '_time', 'locY'], ascending=[True, True, True])
        ng.index = np.arange(ng.shape[0])

        #calculate Time to collision 
        ng['_gap'] = np.nan 
        ng['_gap'] = ng._locy.shift(-1) - ng._locy - ng._vlen.shift(-1) 
        ng.loc[ng._lane != ng._lane.shift(-1), '_gap']   =  np.nan 
        ng.loc[ng._time != ng._time.shift(-1), '_gap'] =  np.nan 

        ng['_ttc'] = 0 
        ng['_ttc'] = ng._gap / (ng._spd.shift(-1) - ng._spd) * (5280 / 3600)
        ng.loc[ng._time != ng._time.shift(-1), '_ttc'] = np.nan 
        ng.loc[ng._ttc < 0, '_ttc'] = np.nan
        ng.loc[ng._ttc == np.inf, '_ttc'] = np.nan


        #recalculate the preceding and following vehicle ids 
        ng['_prV'] = ng._vid.shift(-1)

        ng.loc[ng.lane != ng.lane.shift(-1), '_prV']   =  np.nan 
        ng.loc[ng._time != ng._time.shift(-1), '_prV'] =  np.nan 

        ng['_flV'] = ng._vid.shift(1)
        ng.loc[ng.lane != ng.lane.shift(1), '_flV'] =  np.nan 
        ng.loc[ng._time != ng._time.shift(1), '_flV'] =  np.nan 
        
        return VTAnalytics(ng)

    @classmethod
    def readModelData(cls, fName1, start_time=None, end_time=None):

        ai = pd.read_hdf(fName1, 'trajectories')

        ai['_vid']  = np.array(ai.oid, np.int)
        ai['_time'] = ai.time
        ai['_locy'] = ai.dist_along
        ai['_lane'] = 8 - ai['laneIndex']

        if start_time:
            ai = ai[ai._time.values >= start_time]

        if end_time:
            ai = ai[ai._time.values <= end_time]

        #calculate the time step 
        ai = ai.sort_values(['_vid', '_time']) 
        ai['_dt'] = ai._time.shift(-1) - ai._time
        ai.loc[ai._vid != ai._vid.shift(-1), '_dt'] = np.nan

        #calculate the lane change variables 
        ai['_leaveLane'] = 0
        ai.loc[ai._lane  != ai._lane.shift(-1), '_leaveLane'] = 1
        ai.loc[ai._vid   != ai._vid.shift(-1), '_leaveLane'] = 0

        ai['_enterLane'] = 0
        ai.loc[ai._lane  != ai._lane.shift(1), '_enterLane'] = 1
        ai.loc[ai._vid   != ai._vid.shift(1), '_enterLane'] = 0


        ai = ai.sort_values(['_lane', '_time', '_locy'], ascending=[True, True, True])
        ai.index = np.arange(ai.shape[0])

        ai['_prV']        = ai._vid.shift(-1)
        ai.loc[ai._lane  != ai._lane.shift(-1), '_prV'] =  np.nan 
        ai.loc[ai._time  != ai._time.shift(-1), '_prV'] =  np.nan 

        ai['_flV'] = ai._vid.shift(1)
        ai.loc[ai._lane != ai._lane.shift(1), '_flV'] =  np.nan 
        ai.loc[ai._time != ai._time.shift(1), '_flV'] =  np.nan 

        ai['_spd'] = ai.speed
        ai['_acc'] = ai.acceleration

        return VTAnalytics(ai)

    def __init__(self, df, space_bin=50, time_bin=10):

        assert '_spd' in df.columns
        assert '_acc' in df.columns 

        self.df = df
        self._calculateTimeStep() 

        self.space_bin = space_bin
        self.time_bin  = time_bin 
        self.numLanes = self.df._lane.max() 
        
        self.recalculateMacroVars(space_bin, time_bin) 

    def __add__(self, other):

        return VTAnalytics(pd.concat([self.df, other.df]))

    def _calculateTimeStep(self):

        self.df = self.df.sort_values(['_vid', '_time']) 
        self.df['_dt'] = self.df._time.shift(-1) - self.df._time
        #there are some records with dt greater than the 0.1 don't know why 
        self.df.loc[self.df._vid != self.df._vid.shift(-1), '_dt'] = np.nan

        if self.df._dt.unique().shape[0] > 2:
            print("ERROR: The dataset has more than one time step")
            #raise ValueError("The dataset has more than one timestep") 

        self.time_step = self.df._dt.dropna().unique()[0]

    def resample(self, timeStep):
        """returns a new dataset with resampled speed, acceleration, and location along 
        the corridor based on the given time step"""
        
        df = self.df.sort_values(['_vid', '_time'])

        result = [] 
        for VehId, group in df.groupby('_vid'):

            tmp2 = (group[['_time', '_acc', '_spd',  '_locy']]
                   .set_index('_time')
                   .resample('1s')
                   .mean()
                   .reset_index()
                   )

            tmp2['_vid'] = VehId
            result.append(tmp2)

        df1s = pd.concat(result)
        df1s = df1s.dropna()
        
        return df1s 

    def getTTCDistribution(self):

        hist, bins = np.histogram(self.df['_ttc'].dropna(), bins=np.arange(0, 16, 1))
        TTC_freq = pd.DataFrame(index=bins[:-1], data=hist, columns=['freq'])
        labels = ["%d <= TTC < %d" % (i, i+1) for i in bins[:-1]]
        TTC_freq.index = labels
        TTC_freq['ngRe_Percentage'] = TTC_freq['freq'] / self.df.shape[0] * 100

        return TTC_freq

    def calculateCorridorTravelTimes(self):
        
        self.df = self.df.sort_values(['_vid', '_time'])
        tmp1 = (self.df.groupby('_vid')[['_time', '_locy']]
                .first().reset_index()
                .rename(columns={'_time':'start', '_locy':'startDist'}))
        
        tmp2 = (self.df.groupby('_vid')[['_time', '_locy']]
                .last().reset_index()
                .rename(columns={'_time':'end', '_locy':'endDist'}))
        
        cor_times = pd.merge(tmp1, tmp2, on=['_vid'])
        
        cor_times['dur'] = (cor_times.end - cor_times.start)
        cor_times['dist'] = cor_times.endDist - cor_times.startDist 
        cor_times['_spd'] = (cor_times.dist / 5280.0) / (cor_times.dur / 3600.0)
        
        return cor_times 

    def plotSpeedDistribution(self, ax, lower=None, upper=None):

        if not lower:
            lower = np.floor(self.df._spd.min()) 
        if not upper:
            upper = np.ceil(self.df._spd.max()) 
        
        bins = np.arange(lower, upper)
        
        ax.hist(self.df._spd, bins=bins, color='blue', histtype='step', linewidth=2) 
        
        minor_locator = matplotlib.ticker.AutoMinorLocator(10)
        ax.xaxis.set_minor_locator(minor_locator)
        
        ax.set_title("Distribution of Speed", fontsize=18)
        ax.set_xlabel("Speed(mph)", fontsize=18)
        ax.set_ylabel("Frequency", fontsize=18)
        
        ax.grid(b=True, which='major', color='white', lw=2)
        ax.grid(b=True, which='minor', color='white', lw=0.5, alpha=1.0)
        
        freq, bins2 = np.histogram(self.df._spd, bins=bins)

        spdFreq = pd.DataFrame(index=['%d-%d' % (i, j) for (i,j) 
                              in zip(bins, bins2[1:])],
                       data=freq, columns=['Freq'])
        spdFreq.index.name='bin'
        return spdFreq

    def plotAccelerationDistribution(self, ax, lower=None, upper=None):
        
        if not lower:
            lower = np.floor(self.df._acc.min()) - 0.5
        if not upper:
            upper = np.ceil(self.df._acc.max()) + 0.5

        bins = np.arange(lower, upper, 1)
        
        ax.hist(self.df._acc, bins=bins, color='blue', histtype='step', linewidth=2) 
        
        ax.set_title("Distribution of Acceleration", fontsize=18)
        ax.set_xlabel("Acceleration(fpss)", fontsize=18)
        ax.set_ylabel("Frequency", fontsize=18)
        
        minor_locator = matplotlib.ticker.AutoMinorLocator(5)
        ax.xaxis.set_minor_locator(minor_locator)
        
        ax.grid(b=True, which='major', color='white', lw=2)
        ax.grid(b=True, which='minor', color='white', lw=0.5, alpha=1.0)

        freq, bins2 = np.histogram(self.df._acc, bins=bins)

        accFreq = pd.DataFrame(index=['%d-%d' % (i, j) for (i,j) 
                              in zip(bins, bins2[1:])],
                       data=freq, columns=['Freq'])
        accFreq.index.name='bin'
        return accFreq

    def getAccelerationJerk(self):
        """returns a new dataset with jerk values for a time step of 1 second"""
        
        self.df = self.df.sort_values(['_vid', '_time'])

        result = [] 
        for VehId, group in self.df.groupby('_vid'):

            group2 = group[['_acc', '_spd',  '_locy']].copy() 

            #create a datetime variable 
            new_time = [] 
            for t in group._time.values: 
                t = np.datetime64(int(t *1000) ,'ms')
                new_time.append(t)

            group2['_time'] = new_time 

            tmp2 = (group2.set_index('_time')
                   .resample('1s')
                   .mean()
                   .reset_index()
                   )

            tmp2['_vid'] = VehId

            tmp2['_jerk'] = tmp2._acc.shift(-1) - tmp2._acc
            result.append(tmp2)

        df1s = pd.concat(result)
        self.df1s = df1s.dropna()
        
        return self.df1s 

    def plotJerkDistribution(self, df, ax, lower=None, upper=None):
                
        if not lower:
            lower = np.floor(df._jerk.min()) 
        if not upper:
            upper = np.ceil(df._jerk.max()) 

        bins = np.arange(lower,
                         upper, 1)
        
        ax.hist(df._jerk, bins=bins, color='blue', 
                histtype='step', linewidth=2) 
        
        minor_locator = matplotlib.ticker.AutoMinorLocator(10)
        ax.xaxis.set_minor_locator(minor_locator)
        
        ax.set_title("Distribution of Jerk", fontsize=18)
        ax.set_xlabel("Jerk", fontsize=18)
        ax.set_ylabel("Frequency", fontsize=18)
        
        ax.grid(b=True, which='major', color='white', lw=2)
        ax.grid(b=True, which='minor', color='white', lw=0.5, alpha=1.0)
        
        hist, bins = np.histogram(df._jerk, 
                        bins=bins)
        jerk_freq = pd.DataFrame(index=bins[:-1], data=hist, columns=['freq'])
        labels = ["%.1f" % ((i+j)/2) for i,j in zip(bins[:], bins[1:])]
        jerk_freq.index = labels

        jerk_freq['Percentage'] = jerk_freq['freq'] / df.shape[0] * 100
        jerk_freq.index.name = 'bin center'

        return jerk_freq 

    def getARMSDistribution(self, speedbin=5):
        """Calculates ARMS for each speed bin"""
        result = [] 

        speedBins = np.array(self.df._spd, np.int) // speedbin * speedbin 
        for sbin, group in self.df.groupby(speedBins):

            arms = np.sqrt(np.sum(group._acc.dropna().values ** 2) / group.shape[0]) * 0.303

            result.append((sbin, arms, group.shape[0]))

        arms = pd.DataFrame(result, columns=['_spd', '_arms', 'n'])
        
        return arms 

    def plotARMS(self, df, ax):
    
        ax.plot(df._spd, df._arms, color='blue', lw=2)
            
        ax.set_ylabel('Acceleration Root Mean Squared', fontsize=18)
        ax.set_xlabel("Speed (mph)", fontsize=18)
        ax.set_title("ARMS", fontsize=18)

    def plotAllTrajectories(self, fig, ax, lane, start_time, end_time, start_dist, end_dist, point_size=0.5, title=""):

        """This function plots all the vehicle trajectories for a given lane, start, and end time and for 
        a particular section along the corridor identified by the start and end distance. 

        For each vehicle that is present in the provided temporal and spatial window the 
        function uses color-coded points to visualize successive vehicle positions. The 
        points are color coded by the vehicle speed using a single color for a 10 mph 
        interval.  

        Args:
           fig (matplotlib fibure):  The figure to draw the plot on
           ax  (matplotlib ax):      The axes to draw the plot
           lane (integer) : The lane number 
           start_time (datetime): the begining of the selected time window 
           end_time (datetime): the end of the selected time window
           start_dist (feet): the start distance along the corridor. 
           Vehicle positions before this threshold are not plotted. 
           end_dist (feet): the maximum linear poisition along the corridor to visualize. 
           Vehicle positions after this value will not be ploted.   

        Kwargs:
           point_size (float): The size of each point in pixels. The user can vary the size depending on canvas size and the number of points to visualized

        Returns:
           None

        Raises:
           AttributeError, KeyError

        In the example code below the variables t1 and t2 hold the time thresholds. 
        The dates correspond to valid times in the NGISM I-80 dataset.  
        A figure, and axes canvas are obtaind in the third library by calling 
        the appropate matplotlib function and by providing the selected figure size in inches.
        The fourth line applies the function that produces the image shown below. 
        All the trajectories in lane one  between t1, t2 and and between 0 and 
        1800 from the corridor start will be visualized.  
        
        >>> t1 = np.datetime64('2005-04-13 17:00:00')
        >>> t2 = np.datetime64('2005-04-13 17:30:00')
        >>> fig, ax = plt.subplots(figsize=(15,10), dpi=150)
        >>> plotTrajectories(fig, ax, lane=1, 
                             start_time=t1, end_time=t2, 
                             start_dist=0, end_dist=1800)
        

        .. figure::  _static/ngRe_traj_lane_2.png
           :align:   center

        The generated image can be saved by calling the savefig function. The image is 
        saved in the png format using 150 dots per inch (dpi) 
        
        >>> output_file = "trajectories.png"
        >>> fig.savefig(output_file, dpi=150)

        """
        
        #select the trajectories to plot based on inputs 
        tmp = self.df[self.df._lane == lane]
        tmp = tmp[tmp._time.values >= start_time]
        tmp = tmp[tmp._time.values <= end_time]
        tmp = tmp[tmp._locy >= start_dist]
        tmp = tmp[tmp._locy <= end_dist]
      
        if tmp.shape[0] == 0:
            raise ValueError("Nothing to plot")
        
        rect = 0.06,0.06,0.90,0.9
        ax = fig.add_axes(rect)
        fig.add_axes(ax)

        #define the coloring scheme for the points 
        bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        colors2 = ['#e31a1c','#fd8d3c', '#fecc5c','#ffffcc','#a1dab4','#41b6c4','#225ea8', '#000000']
        cmap = mpl.colors.ListedColormap(colors2) 
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        #define the color of the points 

        point_colors = [] 
        for spd in tmp._spd.values:

            if spd < 80:
                p_color = colors2[int(spd // 10)]
            else:
                p_color = colors2[-1]
        point_colors.append(p_color)

        #point_colors = tmp._spd.apply(lambda spd:colors2[int(spd // 10)]).values
        
        #x axis is the seconds from start time 
        #times = (tmp._time - start_time).dt.total_seconds().values
        times = tmp._time - start_time

        #y axis is the location along the corridor
        locations = tmp._locy.values 
        
        ax.scatter(times, locations, c=point_colors, s=point_size)
        ax.scatter(times[0],   locations[0], c=point_colors[0], s=point_size*8)
        ax.scatter(times[-1], locations[-1], c=point_colors[-1], s=point_size*8)
        
        
        ax.set_facecolor('white')
        ax.set_xlabel("Time in seconds from %s" % str(start_time), fontsize=18)
        ax.set_ylabel("Distance from Starting point %.1f (feet)" % start_dist,
                      fontsize=18)
        
        ax.set_xlim([times.min(), times.max()])
        ax.set_ylim([locations.min(), locations.max()])
        
        if title:
            ax.set_title(title)
        
        cmax = fig.add_axes([0.96, 0.1, 0.02, 0.8])
        mpl.colorbar.ColorbarBase(ax=cmax, cmap=cmap, norm=norm, boundaries=bounds) 

        ax.set_axisbelow(True)
        ax.grid(color='grey', lw=1, linestyle='dashed', alpha=0.2)
    
    def calculateSpaceMeanSpeedAndDensity(self, time_bin, space_bin):
        
        raise Exeption() 
        """time_bin in seconds 
        space_bin in feet """

        df['_spacebin'] = np.array(df._locy // space_bin, np.int)
        df['_timebin'] = pd.to_datetime(((ng._time.astype(np.int64) // 
                                         (time_bin * 1e9) ) * (time_bin * 1e9) ))
        
        df_hm = (df.groupby(['_lane', '_spacebin', '_timebin'])['_spd']
                          .aggregate([np.mean, np.size]))

        df_hm = (df_hm.reset_index()
                      .rename(columns={'size':'density', 'mean':'speed', '_lane':'_lane'}) ) 

        mi = pd.MultiIndex.from_product([sorted(df._lane.unique()), 
                                         sorted(df._spacebin.unique()), 
                                         df._timebin.unique()], 
                                         names=['_lane', '_spacebin', '_timebin'])
        mi = pd.DataFrame(index=mi)
        mi.reset_index(inplace=True)

        df_hm = df_hm.merge(mi, on=['_lane', '_spacebin', '_timebin'], how='right')
        df_hm = df_hm.set_index(['_lane', '_spacebin', '_timebin'])


        ### the simulation time step is 0.1 seconds this is why I am multiplying by 10 below 
        ##TODO change the 10 
        df_hm['density'] = df_hm['density'] * (5280 / space_bin) / (time_bin * 10)
        
        return df_hm 

    def plotMeanCorridorSpeedDistribution(self,df, ax):
        """
        '_vid', 'start', 'startDist', 'end', 'endDist', 'dur', 'dist', 'speed'], dtype='object'
        """
        
        bins = np.arange(np.floor(df._spd.min()),
                         np.ceil(df._spd.max()), 1)
        
        ax.hist(df._spd, bins=bins, color='blue', histtype='step', linewidth=2) 
        
        minor_locator = matplotlib.ticker.AutoMinorLocator(10)
        ax.xaxis.set_minor_locator(minor_locator)
        
        ax.set_title("Distribution of Average Vehicle Speed", fontsize=18)
        ax.set_xlabel("Speed(mph)", fontsize=18)
        ax.set_ylabel("Frequency", fontsize=18)
        
        ax.grid(b=True, which='major', color='white', lw=2)
        ax.grid(b=True, which='minor', color='white', lw=0.5, alpha=1.0)
        
        return np.histogram(df._spd, bins=bins)
    
    def recalculateMacroVars(self, space_bin, time_bin):

        """Recalculates the macroscopic fundamental variables
        time_bin in seconds 
        space_bin in feet 
        """

        self.df['_spacebin'] = np.array(self.df._locy // space_bin, np.int)
        self.df['_timebin'] = pd.to_datetime(((self.df._time.astype(np.int64) // 
                                         (time_bin * 1e9) ) * (time_bin * 1e9) ))

        self.space_bin = space_bin
        self.time_bin  = time_bin 
        
        df_hm = (self.df.groupby(['_lane', '_spacebin', '_timebin'])['_spd']
                          .aggregate([np.mean, np.size]))

        df_hm = (df_hm.reset_index()
                      .rename(columns={'size':'density', 'mean':'speed', '_lane':'_lane'}) ) 

        mi = pd.MultiIndex.from_product([sorted(self.df._lane.unique()), 
                                         sorted(self.df._spacebin.unique()), 
                                         self.df._timebin.unique()], 
                                         names=['_lane', '_spacebin', '_timebin'])
        mi = pd.DataFrame(index=mi)
        mi.reset_index(inplace=True)

        df_hm = df_hm.merge(mi, on=['_lane', '_spacebin', '_timebin'], how='right')
        df_hm = df_hm.set_index(['_lane', '_spacebin', '_timebin'])


        ### the simulation time step is 0.1 seconds this is why I am multiplying by 10 below 
        ##TODO change the 10 

        time_step_in_sec = self.time_step.astype(np.int64) / 1e9

        df_hm['density'] = df_hm['density'] * (5280 / space_bin) / (time_bin / time_step_in_sec)
        
        leaveLaneChangeRates = (self.df.groupby(['_lane', '_spacebin', '_timebin'])['_leaveLane']
                               .sum())
        enterLaneChangeRates = (self.df.groupby(['_lane', '_spacebin', '_timebin'])['_enterLane']
                               .sum())

        leaveLaneChangeRates  = leaveLaneChangeRates * (3600 / time_bin) * (5280 / self.space_bin) 
        enterLaneChangeRates  = enterLaneChangeRates * (3600 / time_bin) * (5280 / self.space_bin) 

        leaveLaneChangeRates = leaveLaneChangeRates.to_frame().rename(columns={"_leaveLane":'numVehiclesLeavingLane'})
        enterLaneChangeRates = enterLaneChangeRates.to_frame().rename(columns={"_enterLane":'numVehiclesEnteringLane'})       

        df_hm['numVehiclesLeavingLane']  = leaveLaneChangeRates['numVehiclesLeavingLane']
        df_hm['numVehiclesEnteringLane'] = enterLaneChangeRates['numVehiclesEnteringLane']

        self.df_macro = df_hm 

    def recalculateLaneChangeRates(self, space_bin, time_bin):

        """Recalculates the lane change rates 
        """
        self.df['_spacebin'] = np.array(self.df._locy // space_bin, np.int)
        self.df['_timebin'] = pd.to_datetime(((self.df._time.astype(np.int64) // 
                                         (time_bin * 1e9) ) * (time_bin * 1e9) ))

        self.space_bin = space_bin
        self.time_bin  = time_bin 

        time_step_in_sec = self.time_step.astype(np.int64) / 1e9

        laneChangeRates = (self.df.groupby(['_lane', '_spacebin', '_timebin'])['_leaveLane']
                           .sum()
                          )

        laneChangeRates = laneChangeRates * (3600 / time_bin) * (5280 / self.space_bin) 


        laneChangeRates = laneChangeRates.to_frame().reset_index()

        mi = pd.MultiIndex.from_product([sorted(self.df._lane.unique()), 
                                         sorted(self.df._spacebin.unique()), 
                                         sorted(self.df._timebin.unique())], 
                                         names=['_lane', '_spacebin', '_timebin'])

        mi = pd.DataFrame(index=mi)
        mi.reset_index(inplace=True)

        laneChangeRates = laneChangeRates.merge(mi, on=['_lane', '_spacebin', '_timebin'], how='right')

        #laneChangeRates = laneChangeRates.set_index(['_lane', '_spacebin', '_timebin']).unstack()

        self.df_lcr = laneChangeRates

        #self.recalculateMacroVars(space_bin, time_bin)

    def plotSelectedTrajectories(self, fig, ax, veh_ids):
        """This function plots selected vehicle trajectories defined by 
        a list of vehicle ids. Points in the plot are color-colded by speed. 
        """
    
        #define the coloring scheme for the points 
        bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        colors2 = ['#e31a1c','#fd8d3c', '#fecc5c','#ffffcc','#a1dab4','#41b6c4','#225ea8', '#000000']
        cmap = mpl.colors.ListedColormap(colors2) 
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        for veh_id in veh_ids:
            
            tmp = df[df._vid == veh_id]
            
            locations = tmp._locy.values 
            times = (tmp._time - start_time).dt.total_seconds().values
            
            #define the color of the points 
            point_colors = tmp._spd.apply(lambda spd:colors2[int(spd // 10)]).values
            
            #y axis is the location along the corridor
            locations = tmp._locy.values 
            
            ax.scatter(times, locations, c=point_colors, s=1)

    def plotSpeedVsDensity(self, fig, ax, speed_step=5, color='blue', max_density=250, max_speed=70, show_bin_info=True, plot_mean=True):

        points = ax.scatter(self.df_macro.density, self.df_macro.speed, 
             s=1, color=color, label=None)

        #set the limits of max density 
        ax.set_xlim([0, max_density])
        ax.set_ylim([0, max_speed])

        #plot the mean line 
        assert speed_step > 0 
        speed_groups = self.df_macro.density.values // speed_step * speed_step

        hm_mean = (self.df_macro.groupby(speed_groups)['speed'].aggregate([np.mean, np.size])
                      .reset_index()
                      .rename(columns={'index':'density', 'mean':'mean_speed'}))

        #the mean line is plotted only where there are more than 100 points
        if plot_mean:  
            tmp = hm_mean[hm_mean['size'] > 100]
            #tmp = hm_mean 
            mean_line = ax.plot(tmp.density, tmp.mean_speed, c=color, label="mean speed")

        ax.legend(fontsize=16)

        ax.set_ylabel("Speed (mph)", fontsize=18)
        ax.set_xlabel('Density (vpm)', fontsize=18)

        if show_bin_info:
            fig.text(0.85, 0.65, "SpaceBin:%dft\nTimeBin:%dsec" % (self.space_bin, self.time_bin), fontsize=18)

        ax.set_title("SpeedVsDensity")

        fig.tight_layout()

    def plotSpeedVsDensityByLane(self, fig, dot_color='blue', mean_color='white', plot_mean=True, max_density=250, max_speed=70, pad=1.0, w_pad=0.5, h_pad=1.0, alpha=0.5):

        numLanes = self.df._lane.max() 

        numRows = numLanes // 2 + 1
        gs = mpl.gridspec.GridSpec(numRows, 2)

        axes = [] 

        for laneNum in range(1, numLanes + 1):
            
            i = (laneNum - 1) // 2
            j = (laneNum - 1) % 2 
            
            ax = plt.subplot(gs[i,j])

            axes.append(ax)
            
            tmp = self.df_macro.loc[laneNum]
            
            ax.scatter(tmp.density, tmp.speed, s=1.5, color=dot_color, label=None, alpha=alpha)
            ax.set_xlim([0, max_density])
            ax.set_ylim([0, max_speed])
            ax.set_title("Lane %d" % laneNum, fontsize=16)
            
            tmp_mean = (tmp.groupby(tmp.density.values // 5 * 5)['speed']
                              .aggregate([np.mean, np.size])
                              .reset_index()
                              .rename(columns={'index':'density', 'mean':'mean_spd'}))
            
            #if you wish to plot the mean when you have 50 or more points
            #tmp_mean = tmp_mean[tmp_mean['size'] > 50]
            
            if plot_mean: 
                ax.plot(tmp_mean.density, tmp_mean.mean_spd, c=mean_color, label='mean speed')
            
            if j == 1:
                ax.set_yticklabels([])
                
            if j == 0:
                ax.set_ylabel("Speed (mph)", fontsize=18)
                
            ax.legend(fontsize=18)

        axes[-2].set_xlabel("Density (vpm)", fontsize=18)
        axes[-1].set_xlabel("Density (vpm)", fontsize=18)


        fig.suptitle("SpeedVsDensityByLane", fontsize=18)

        fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)

        fig.text(0.72, 0.13, "SpaceBin:%dft\nTimeBin:%dsec" % (self.space_bin, self.time_bin), fontsize=18)

    def getNumOfLaneChangesPerLane(self):
        """Returns a table with the number of lane changes
        """ 

        tmp = self.df.groupby('_lane')['_leaveLane'].sum().to_frame().reset_index()
        tmp.rename(columns={'_leaveLane':"NumOfVehiclesLeavingLane"}, inplace=True)
        tmp['NumOfVehiclesEnteringLane'] = self.df.groupby('_lane')['_enterLane'].sum().to_frame().reset_index()['_enterLane']
        
        return tmp 
 
    def plotLCR(self, fig, laneChangeType='leave', max_lcr=72000):
        """lane changes can be either enter or exit 
        """
        gs = mpl.gridspec.GridSpec(2, 5, width_ratios=[0.2, 1, 1, 1, 1])

        #set up teh colorscale 
        numColors = 10
        bounds = list(np.linspace(0, max_lcr, numColors))
        bounds.insert(0, -10)
        bounds[1] = 0.1

        colors = ['grey', '#ffffcc','#ffeda0','#fed976','#feb24c',
                 '#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
        cmap = mpl.colors.ListedColormap(colors) 
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        cb = None 
        ax = None

        for i in range(1, self.numLanes + 1):
            
            j = (i-1) % 4
            k = (i-1) // 4

            ax = plt.subplot(gs[k,j+1])
            
            if laneChangeType == 'leave':
                data = self.df_macro.loc[i]['numVehiclesLeavingLane'].unstack().values
            elif laneChangeType == 'enter': 
                data = self.df_macro.loc[i]['numVehiclesEnteringLane'].unstack().values
            else:
                raise ValueError("unknown lane change type")

            nan_mask = np.isnan(data)
            tmp = data.copy()
            tmp[nan_mask] = -1
          
            cb = ax.imshow(tmp, cmap=cmap, norm=norm, origin='lower', aspect='auto')
       
            ax.set_xticklabels(['', '17:00', '05', '10', '15', ''])

            ax.set_yticklabels([])
            
            ax.grid(True, color='black', ls='dashed', lw=0.3)
            
            ax.set_title("Lane %d" % i, fontsize=16)
            
        cmax = fig.add_axes([0.85, 0.15, 0.05, 0.3])

        fig.colorbar(cb, cax=cmax, norm=norm, boundaries=bounds, ticks=bounds)
        cmax.set_title("LCR colormap\n(lcvm)", fontsize=16)

        cm_labels = ["%d" % i for i in list(map(int, bounds))]
        cm_labels[0] = "" 
        cmax.set_yticklabels(cm_labels, fontsize=16)

        ax1 = plt.subplot(gs[0,0]) 
        ax2 = plt.subplot(gs[1,0])

        ytick_labels = [i for i in range(0,2000,200)]
        ax1.set_yticks(ytick_labels)
        ax1.set_yticklabels(ytick_labels, fontsize=16)
        ax1.set_xticklabels([])
        ax1.grid(False)
        ax1.set_ylabel("Length in feet along I80 corridor", fontsize=16)
        ax1.set_facecolor('white')

        ax2.set_yticks(ytick_labels)
        ax2.set_yticklabels(ytick_labels, fontsize=14)
        ax2.set_xticklabels([])
        ax2.grid(False)
        ax2.set_ylabel("Length in feet along I80 corridor", fontsize=16)
        ax2.set_facecolor('white')

        fig.text(0.85, 0.05, "SpaceBin:%dft\nTimeBin:%dsec" % 
                  (self.space_bin, self.time_bin), fontsize=16)

        fig.text(0.908, 0.15, "No data", fontsize=16)



class Directory:

    pass 

    def __init__(self):

        pass 


class Results:

    def __init__(self):

        pass 

    def execute(self):

        start_time = np.datetime64('2005-04-13 17:00:00')
        end_time   = np.datetime64('2005-04-13 17:30:00')

        print('reading the data')
        fileName = "./data/ngsimI80trajectories 0500-0530.txt"

        self._outputDir = Directory()

        self._outputDir.value = './results'

        #self.vt = VTAnalytics.readNGISIMData(fileName, 
        #    start_time=start_time, 
        #    end_time=end_time)


        self.vt = VTAnalytics.readModelData("./data/ai21.hdf",
                              start_time=start_time, end_time=end_time)

        self.data = self.vt.df 
        
        print('done reading the data')

        writer = pd.ExcelWriter(os.path.join(self._outputDir.value, 'tables.xlsx'))

        PAGE_SIZE = (1000, 792.0)
        pdfReport = PdfReport(os.path.join(self._outputDir.value, 'report.pdf'), PAGE_SIZE)

        fig, ax = plt.subplots(figsize=(10,8), dpi=100)
        self.vt.plotSpeedVsDensity(fig, ax, max_density=280)        
        fig.savefig(os.path.join(self._outputDir.value, "SpeedVsDensity.png"), dpi=100)
        pdfReport.addChart2(os.path.join(self._outputDir.value, "SpeedVsDensity.png"), 'SpeedVsDensity')


        fig, ax = plt.subplots(figsize=(10,8), dpi=100)
        self.vt.plotLCR(fig, laneChangeType='enter')
        fig.savefig(os.path.join(self._outputDir.value, "LCR.png"), dpi=100)
        pdfReport.addChart2(os.path.join(self._outputDir.value, "LCR.png"), "LaneChangeRate")


        fig, ax = plt.subplots(figsize=(10,8), dpi=100)
        self.vt.plotSpeedVsDensityByLane(fig, plot_mean=True, dot_color='blue', alpha=0.4)
        out_fileName = os.path.join(self._outputDir.value, "SpeedVsDensityByLane.png")
        fig.savefig(out_fileName,dpi=100)
        pdfReport.addChart2(out_fileName, "SpeedVsDensityByLane")


        fig, ax = plt.subplots(figsize=(10,8))
        spdDist = self.vt.plotSpeedDistribution(ax)
        out_fileName = os.path.join(self._outputDir.value, "SpeedDistribution.png")
        fig.savefig(out_fileName, dpi=100)
        pdfReport.addChart2(out_fileName, "SpeedDistribution")
        spdDist.to_excel(writer,'SpeedDistribution')

        fig, ax = plt.subplots(figsize=(10,8))
        accDist = self.vt.plotAccelerationDistribution(ax)
        out_fileName = os.path.join(self._outputDir.value, "AccelerationDistribution.png")
        fig.savefig(out_fileName, dpi=100)
        pdfReport.addChart2(out_fileName, "AccelerationDistribution")
        accDist.to_excel(writer,'AcclerationDistribution')

        fig, ax = plt.subplots(figsize=(10,8))
        jerk = self.vt.getAccelerationJerk()
        jerkDist = self.vt.plotJerkDistribution(jerk, ax)
        out_fileName = os.path.join(self._outputDir.value, "AccelerationJerkDistribution.png")
        fig.savefig(out_fileName ,dpi=100)
        pdfReport.addChart2(out_fileName, "AccelerationJerkDistribution")
        jerkDist.to_excel(writer,'JerkDistribution')


        armsDist = self.vt.getARMSDistribution()
        fig, ax = plt.subplots(figsize=(10,8))
        self.vt.plotARMS(armsDist, ax)
        out_fileName = os.path.join(self._outputDir.value, "AccelerationRootMeanSquareError.png")
        fig.savefig(out_fileName, dpi=100)
        pdfReport.addChart2(out_fileName, "AccelerationRootMeanSquareError")
        armsDist.to_excel(writer,'ARMS')

        writer.save()

        for i in range(1,8):
            fig, ax = plt.subplots(figsize=(10,8),dpi=100)
            self.vt.plotAllTrajectories(fig, ax, i, np.datetime64('2005-04-13 17:00:00'), 
                             np.datetime64('2005-04-13 17:10:00'), 0, 1000, point_size=0.5, title="All trajectories for lane %d" % i)
            out_fileName = os.path.join(self._outputDir.value, "All_trajectories_lane_%d.png" % i)
            fig.savefig(out_fileName, dpi=100)
            pdfReport.addChart2(out_fileName, "All trajectories for lane %d" % i)


        pdfReport.write()

        file_list = [os.path.join(self._outputDir.value, "graph%d.pdf" % i)
                          for i in range(1,5)]


        print('done')

if __name__ == "__main__":


    r = Results()

    r.execute() 


#http://matplotlib.org/sampledoc/index.html
#https://pythonhosted.org/an_example_pypi_project/sphinx.html#full-code-example
#https://pythonhosted.org/an_example_pypi_project/pkgcode.html
#http://www.sphinx-doc.org/en/stable/ext/autodoc.html

