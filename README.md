# QIT-CEMC dataset
An open-source dataset about milling to monitoring tool wear status.

QIT-CEMC is an open-source dataset with data collected on a vertical machining centre for industrial big data and smart manufacturing experiments at Qilu Institute of Technology.

## Folder structure
The dataset consists of a CSV file that records the health indicators of tool wear and three folders, which are the vibration and sound signals folder, the force and torque signals folder, and the image folder.

## Label structure
All labels are presented in the 'tool_wear.csv' file including indicators from the side edge and the end edge. The first column represents the cycle number of the experiment. All the labels indicating the tool wear status are recorded in the other columns. For the side edge, 3 indicators named the max VB, VB in the half cutting depth and, wear area value from the 4 cutting edges were recorded. For the end teeth, 2 indicators named the max VB and wear area value from the same 4 cutting edges were recorded. All columns are described below:

* Cycle: the machine cycle, represents a single process.
* Cutting edge: The tools used in this dataset have 4 cutting edges. Number 1 in the side teeth and the end teeth represent the same cutting edge.
* VBmax: the max value of flank wear
* VB in 1/2ap: the value of flank wear in the 1/2 cutting depth
* S: the wear area, mm^2





