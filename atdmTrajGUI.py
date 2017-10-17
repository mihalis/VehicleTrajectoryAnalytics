from pyforms 			  import BaseWidget
from pyforms.Controls 	  import ControlText
from pyforms.Controls 	  import ControlButton
from pyforms.Controls 	  import ControlMatplotlib
from pyforms.Controls     import ControlDir 
from pyforms.Controls     import ControlFile 
from pyforms.Controls     import ControlCombo 

import pyforms 

import pandas as pd 


class SimpleExample1(BaseWidget):
	
	def __init__(self):

		super(SimpleExample1,self).__init__('Simple example 1')

		#Definition of the forms fields
		self._firstname 	= ControlText('First name', 'Default value')
		self._middlename 	= ControlText('Middle name')
		self._lastname 		= ControlText('Lastname name')
		self._fullname 		= ControlText('Full name')
		self._button 		= ControlButton('Press this button')
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
						  '=',('_outputDir') ]


		self._load.value = self.__loadAction
		#self._save.value = self.__saveAction 

	def __buttonAction(self):
		"""Button action event"""
		self._fullname.value = self._firstname.value +" "+ self._middlename.value + \
		" "+ self._lastname.value

	def __loadAction(self):

		fileName = self._inputFile.value 

		if fileName.endswith("hdf"):
			self.data = pd.read_hdf(fileName, 'trajectories')
		elif fileName.endswith("csv"):
			self.data = pd.read_csv(fileName)

		columns = list(self.data.columns)

		self.__initializeBoxes2(columns)

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


if __name__ == "__main__":   pyforms.start_app( SimpleExample1 )

