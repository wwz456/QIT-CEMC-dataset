# QIT-CEMC dataset
An open-source dataset about milling to monitoring tool wear status.

QIT-CEMC is an open-source tool wear dataset using the coated end milling cutter collected on a vertical machining centre for industrial big data and smart manufacturing experiments at Qilu Institute of Technology. The structure of the experiment setup is shown below:

https://github.com/wwz456/QIT-CEMC-dataset/blob/main/experiment%20setup.png

The machine conditions and some details are shown below.

https://github.com/wwz456/QIT-CEMC-dataset/blob/main/machine%20conditions.png

## Folder structure
The dataset consists of a CSV file that records the health indicators of tool wear and three folders, which are the vibration and sound signals folder, the force and torque signals folder, and the image folder.

https://github.com/wwz456/QIT-CEMC-dataset/blob/main/Structure%20of%20datasets%20folder.png

## Label structure
All labels are presented in the 'tool_wear.xls' file including indicators from the side edge and the end edge. The first column represents the cycle number of the experiment. All the labels indicating the tool wear status are recorded in the other columns. For the side edge, 3 indicators named the max VB, VB in the half cutting depth and, wear area value from the 4 cutting edges were recorded. For the end teeth, 2 indicators named the max VB and wear area value from the same 4 cutting edges were recorded. All columns are described below:

* Cycle: the machine cycle, represents a single process.
* Cutting edge: The tools used in this dataset have 4 cutting edges. Number 1 in the side teeth and the end teeth represent the same cutting edge.
* VBmax: the max value of flank wear
* VB in 1/2ap: the value of flank wear in the 1/2 cutting depth
* S: the wear area, mm^2
## Sensor data
The root folder name is 'QIT-CEMC Dataset' in which two folders are named 'Force and torque data' and 'Vibration and sound data'. There are 68 'txt' files in the 'Force' folder and each file has approximately 5,000,000 rows and 5 columns of data including time columns, force signals in the XYZ direction, and torque signals.
* Time: The order of the data points, and also the order of time, follows the data points sampled at a sampling frequency of 10k.
* Fx: Force data from channel x using a wireless rotating dynamometer.
* Fy: Data from the y channel
* Fz: Data from the z channel
* Mz: the torque acquired from a wireless rotating dynamometer.
There are 67 'csv' or 'exls' files in the 'Vibration and sound data' folder. Note: The data for the second cycle was lost because the acquisition software was disconnected.
* Column 1: time. the same as force data
* Column 2: Vibration data from the x channel
* Column 3: Vibration data from the y channel
* Column 4: Vibration data from the z channel
* Column 5: Sound data
## How to use
You can use these data to classify tool wear status. Download link:

https://www.qlit.edu.cn/datasets2/
or directly download the dataset from the Baidu disk link:

https://pan.baidu.com/s/11XeJoW_-3uVQEnPCI3Eaog?pwd=jm7f 
Access code: jm7f
If you have any questions, please contact us at 22010329@siswa.unimas.my

  




