from abaqusConstants import *
from odbAccess import *
#from assembly import *
import sys
import math
from sys import path

odbPath='D:/MIT_project/strings_with_air/infintie_elements_radius/odbs/infinite_elements_05m.odb'
odbName='infinite_elements_05m'
stepName='Step-1'
myOdb = session.openOdb(name=odbName,path=odbPath)
Steady_Step = myOdb.steps[stepName]
Coord_Array=Steady_Step.frames[0].fieldOutputs['POR'].bulkDataBlocks[3].instance.nodes #3 is outer air
Y_coord=[]
Z_coord=[]
node_num=[]
for k in range(len(Coord_Array)-1):
    if abs(Coord_Array[k].coordinates[0])==19: #check x-cut at 19
        node_num.append(k)
        Y_coord.append(Coord_Array[k][1])
        Z_coord.append(Coord_Array[k][2])
POR_mag={}
POR_phase={}
for frame_num in range(1,len(Steady_Step.frames)):
    frame=Steady_Step.frames[frame_num]
    print ("frame_num=",frame_num)
    print ("Frequency=",frame.frequency)
    POR_mag[frame.frequency]=[]
    POR_phase[frame.frequency]=[]
    POR_subset_data=frame.fieldOutputs['POR']
    POR_Data_r=POR_subset_data.bulkDataBlocks[3].data#1 is air
    POR_Data_i=POR_subset_data.bulkDataBlocks[3].conjugateData
    for k in range(len(node_num)):
        POR_mag[frame.frequency].append(POR_Data_r[node_num[k]][0]**2+POR_Data_i[node_num[k]][0]**2)#check if the list is indexed correctl
        POR_phase[frame.frequency].append(atan2(POR_Data_i[node_num[k]][0],POR_Data_r[node_num[k]][0]))
    POR_mag[frame.frequency]=[i**0.5 for i in POR_mag[frame.frequency]]
    Magnitude_of_Pressure = open("Pressure_magnitude_xplane_infelements_%d.txt" %frame.frequency, 'w')
    Magnitude_of_Pressure.write("Y-coordinate\tZ-coordinate\tMagnitude\tPhase\n")
    Magnitude_of_Pressure.close()
    for k in range(len(node_num)):
        Magnitude_of_Pressure=open("Pressure_magnitude_xplane_infelements_%d.txt" %frame.frequency, 'a')
        Magnitude_of_Pressure.write(str(Y_coord[k]))
        Magnitude_of_Pressure.write('\t')
        Magnitude_of_Pressure.write(str(Z_coord[k]))
        Magnitude_of_Pressure.write('\t')
        Magnitude_of_Pressure.write(str(POR_mag[frame.frequency][k]))
        Magnitude_of_Pressure.write('\t')
        Magnitude_of_Pressure.write(str(POR_phase[frame.frequency][k]))
        Magnitude_of_Pressure.write('\n')
        Magnitude_of_Pressure.close()
        