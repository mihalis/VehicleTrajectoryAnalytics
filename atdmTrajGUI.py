from pyforms              import BaseWidget
from pyforms.Controls     import ControlText
from pyforms.Controls     import ControlButton
from pyforms.Controls     import ControlMatplotlib
from pyforms.Controls     import ControlDir 
from pyforms.Controls     import ControlFile 
from pyforms.Controls     import ControlCombo 

import pyforms 

import pandas as pd 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt # ploting 
import matplotlib as mpl        # ploting 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os 

from PyPDF2 import PdfFileReader, PdfFileWriter

from vehicleTrajectoryAnalytics import VTAnalytics

start_time = np.datetime64('2005-04-13 17:00:00')
end_time   = np.datetime64('2005-04-13 17:30:00')

def combinePdfFiles(listOfFileNames, outFileName):

    output = PdfFileWriter()

    for inputFile in listOfFileNames:

        inputStream = PdfFileReader(open(inputFile, "rb"))

        for i in range (0, inputStream.getNumPages() ):
            output.addPage(inputStream.getPage(i))

        # write the output file - change output folder as needed

    outputStream = open(outFileName, "wb")
    output.write(outputStream)
    outputStream.close()

class VehTrajAnalytics(BaseWidget):
    
    def __init__(self):

        super(VehTrajAnalytics, self).__init__('Vehicle Trajectory Analytics')

        #Definition of the forms fields
        self._firstname     = ControlText('First name', 'Default value')
        self._middlename    = ControlText('Middle name')
        self._lastname      = ControlText('Lastname name')
        self._fullname      = ControlText('Full name')
        self._button        = ControlButton('Press this button')
        self._button2       = ControlButton('Press this button2')

        self._outputDir     = ControlDir("Output Directory")
        self._inputFile     = ControlFile("Input Trajectory File")

        self._load          = ControlButton("Load")
        self._save          = ControlButton("Save")

        self._vehicleId     = ControlCombo("Vehicle id")
        self._time          = ControlCombo("Time(sec)")
        self._positionX     = ControlCombo("PositionX (feet)")
        self._positionY     = ControlCombo("PositionY (feet)")

        # Instrumented vehicle trajectories 
        self._radarRange    = ControlCombo("Radar range (feet)")
        self._radarAngle    = ControlCombo("Radar angle (degrees)")
        self._centerline    = ControlFile("CenterlineFile")

        #all traffic stream 
        self._lane          = ControlCombo("Lane")
        self._vLength       = ControlCombo("Vehicle Length")
        self._distanceAlong = ControlCombo("Distance Along Corridor")

        #Define the button action
        self._button.value = self.__buttonAction


        self.data = None 
        self.outFolder = None 

        self.formset = [ [('_inputFile', '_load'), '_vehicleId', "_time", 
                   '_positionX', "_positionY", "_distanceAlong"], "=", 
                          {
                            'All Vehicle Trajectories':['_lane', '_vLength'], 
                            'InstrumentedVehicleTrajectories': ['_radarAngle', '_radarRange']
                          },
                          '=',('_outputDir', '_save') ]


        self._load.value = self.__loadAction
        self._save.value = self.__saveAction 

    def __buttonAction(self):
        """Button action event"""
        self._fullname.value = self._firstname.value +" "+ self._middlename.value + \
        " "+ self._lastname.value

    def __loadAction(self):

        fileName = self._inputFile.value 

        if fileName.endswith("hdf"):
            self.data = pd.read_hdf(fileName, 'trajectories')
            self.vt = VTAnalytics.readModelData(fileName, 
                start_time=start_time, end_time=end_time)
        else:
            #self.data = pd.read_csv(fileName)
            self.vt = VTAnalytics.readNGISIMData(fileName, 
                start_time=start_time, 
                end_time=end_time)
            self.data = self.vt.df 

        columns = list(self.data.columns)

        self.__initializeBoxes2(columns)

    def __saveAction(self):


        fig, ax = plt.subplots(figsize=(10,8))
        self.vt.plotSpeedVsDensity(fig, ax, max_density=280)        
        fig.savefig(os.path.join(self._outputDir.value, "graph1.pdf"), dpi=100)

        fig, ax = plt.subplots(figsize=(10,8))
        self.vt.plotLCR(fig, laneChangeType='enter')
        fig.savefig(os.path.join(self._outputDir.value, "graph2.pdf"), dpi=100)

        fig, ax = plt.subplots(figsize=(10,8))
        self.vt.plotSpeedVsDensityByLane(fig, plot_mean=True, dot_color='blue', alpha=0.4)
        fig.savefig(os.path.join(self._outputDir.value, "graph3.pdf"),dpi=100)


        for i in range(1,8):
            fig, ax = plt.subplots(figsize=(10,8))
            self.vt.plotAllTrajectories(fig, ax, i, np.datetime64('2005-04-13 17:00:00'), 
                             np.datetime64('2005-04-13 17:10:00'), 0, 1000, point_size=0.5)
            fig.savefig(os.path.join(self._outputDir.value, "graph%d.pdf" % (3+i)), dpi=100)

        #file_list = [os.path.join(self._outputDir.value, "test.pdf"),
        # os.path.join(self._outputDir.value, "test2.pdf")]
        
        file_list = [os.path.join(self._outputDir.value, "graph%d.pdf" % i)
                          for i in range(1,5)]

        out_file = os.path.join(self._outputDir.value, "trajReport.pdf")

        combinePdfFiles(file_list, out_file)


    def __initializeBoxes2(self, columns):

        for i, col in enumerate(columns):

            self._vehicleId.add_item(col, i)
            self._time.add_item(col, i)
            self._positionX.add_item(col, i)
            self._positionY.add_item(col, i)
            self._lane.add_item(col, i)
            self._vLength.add_item(col, i)
            self._radarRange.add_item(col, i)
            self._radarAngle.add_item(col, i)
            self._distanceAlong.add_item(col, i)


        #self.__initializeBoxes()

    def __initializeBoxes(self):

        self._vehicleId.add_item("One",   '1')
        self._vehicleId.add_item("Two",   '2')
        self._vehicleId.add_item("Three", '3')

        self._time.add_item("One",   '1')
        self._time.add_item("Two",   '2')
        self._time.add_item("Three", '3')

        self._positionX.add_item("One",   '1')
        self._positionX.add_item("Two",   '2')
        self._positionX.add_item("Three", '3')

        self._positionY.add_item("One",   '1')
        self._positionY.add_item("Two",   '2')
        self._positionY.add_item("Three", '3')

        self._lane.add_item("One",   '1')
        self._lane.add_item("Two",   '2')
        self._lane.add_item("Three", '3')

        self._vLength.add_item("One",   '1')
        self._vLength.add_item("Two",   '2')
        self._vLength.add_item("Three", '3')


    def __saveResults(self):



        fig, ax =plt.subplots(figsize=(15,10))
        self.data.speed.hist(ax=ax, bins=np.arange(0,71,1), figsize=(15,10), 
                              width=0.8, color='grey', alpha=0.5)
        ax.set_title("Distribution of instantaneous speeds",fontsize=18)
        ax.set_xlabel("speed (mph)",fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(self.outFolder, 'test.png'))


#pip install -U git+https://github.com/UmSenhorQualquer/pyforms.git 
#pip install -U git+https://github.com/UmSenhorQualquer/pysettings.git
#pip install -U git+https://bitbucket.org/fchampalimaud/logging-bootstrap.gi‌​t


if __name__ == "__main__":   pyforms.start_app( VehTrajAnalytics )

