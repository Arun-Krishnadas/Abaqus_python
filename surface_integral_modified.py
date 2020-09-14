from abaqusConstants import *
from odbAccess import *
from assembly import *
import sys

from sys import path
path.append('C:\Users\Arun Krishnadas\Anaconda3\Lib\site-packages')

import timeit
import time
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import det
from numpy import dot

#**************************************************************************
#**************************************************************************
#**************************************************************************
def mat_mul(A,B):

    C=[[0.0 for i in range(3)] for j in range(3)]
    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
    C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]
    C[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]

    C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0]
    C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]
    C[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]

    C[2][0] = A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0]
    C[2][1] = A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1]
    C[2][2] = A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]

    return C
#**************************************************************************
#**************************************************************************
#**************************************************************************
def mat_vec(A,B):

    C=[0. for i in range(3)]
    C[0] = A[0][0]*B[0] + A[0][1]*B[1] + A[0][2]*B[2]
    C[1] = A[1][0]*B[0] + A[1][1]*B[1] + A[1][2]*B[2]
    C[2] = A[2][0]*B[0] + A[2][1]*B[1] + A[2][2]*B[2]

    return C
#**************************************************************************
#**************************************************************************
#**************************************************************************
def vec_mat(A,B):

    C=[0. for i in range(3)]
    C[0] = A[0]*B[0][0] + A[1]*B[1][0] + A[2]*B[2][0]
    C[1] = A[0]*B[0][1] + A[1]*B[1][1] + A[2]*B[2][1]
    C[2] = A[0]*B[0][2] + A[1]*B[1][2] + A[2]*B[2][2]

    return C
#**************************************************************************
#**************************************************************************
#**************************************************************************

#**************************************************************************
#**************************************************************************
#**************************************************************************
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%   Function TriGaussPoints provides the Gaussian points and weights    %
#%   for the Gaussian quadrature of order n for the standard triangles.  %
#%                                                                       %
#%   Input: n   - the order of the Gaussian quadrature (n<=12)           %
#%                                                                       %
#%   Output: xw - a n by 3 matrix:                                       %
#%              1st column gives the x-coordinates of points             %
#%              2nd column gives the y-coordinates of points             %
#%              3rd column gives the weights                             %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def TriGaussPoints(n):

    xw = np.zeros((n,3),dtype=float);

    if(n == 1):
        xw=np.array([[0.33333333333333,0.33333333333333,1.00000000000000]]);
    elif(n == 2):
        xw=np.array([[0.16666666666667,0.16666666666667,0.33333333333333],[0.16666666666667,0.66666666666667,0.33333333333333],[0.66666666666667,0.16666666666667,0.33333333333333]]);
    return xw
#**************************************************************************
#**************************************************************************
#**************************************************************************
def Translate_xytriangle(xyz):
    x1= xyz[0][0]; y1=xyz[0][1]; z1=xyz[0][2];
    xyz[0][0]=0; xyz[0][1]=0; xyz[0][2]=0;
    xyz[1][0]=xyz[1][0]-x1; xyz[1][1]=xyz[1][1]-y1; xyz[1][2]=xyz[1][2]-z1;
    xyz[2][0]=xyz[2][0]-x1; xyz[2][1]=xyz[2][1]-y1; xyz[2][2]=xyz[2][2]-z1;
    return ([x1,y1,z1],xyz)

def Rotate_Triangle_to_xy_plane(xyz):
    #Gram-Schmidt's orthonormalisation process
    [[a,b,c],xyz2]=Translate_xytriangle(xyz)
    v1=[xyz[1][0]-xyz[0][0],xyz[1][1]-xyz[0][1],xyz[1][2]-xyz[0][2]]
    v2=[xyz[2][0]-xyz[0][0],xyz[2][1]-xyz[0][1],xyz[2][2]-xyz[0][2]]
    
    w = np.cross(v1,v2);    w = w/norm(w,2);

    u=v1/norm(v1,2);
    v=v2-dot(u,v2)*u;
    v=v/norm(v,2);
    mat = np.array([ [u[0],u[1],u[2]],[v[0],v[1],v[2]],[w[0],w[1],w[2]] ])
    mat2=mat.T;
    xyz3=mat_mul(xyz2,mat2)
    
    inv_Pr_to_P_mat = inv(mat2);

    return ([a,b,c],xyz3,inv_Pr_to_P_mat)
#**************************************************************************
#**************************************************************************
#**************************************************************************
def I_surf(inten1,inten2,inten3,xyz,P):
    [x1,y1,z1]=[xyz[0][0],xyz[0][1],xyz[0][2]]
    [x2,y2,z2]=[xyz[1][0],xyz[1][1],xyz[1][2]]
    [x3,y3,z3]=[xyz[2][0],xyz[2][1],xyz[2][2]]
    [x0,y0,z0]=[P[0],P[1],P[2]]
    r1=((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)**0.5
    r2=((x0-x2)**2+(y0-y2)**2+(z0-z2)**2)**0.5
    r3=((x0-x3)**2+(y0-y3)**2+(z0-z3)**2)**0.5
    I=(inten1*r1+inten2*r2+inten3*r3)/(r1+r2+r3)
    return I
def int_f(N,x1,x2,x3,y1,y2,y3,inv_Pr_to_P_mat,a,b,c,xyz,inten1,inten2,inten3):
    #   This function evaluates \iint_K f(x,y) dxdy using 
    #   the Gaussian quadrature of order N where K is a 
    #   triangle with vertices (x1,y1), (x2,y2) and (x3,y3).


    # Get thequadrature points and weights: 
    #----------------------------------------------------------------------
    xw = TriGaussPoints(N);

    # Find number of Gauss points:
    #----------------------------------------------------------------------
    NP=int(xw.shape[1]); 

    # Evaluate the integral:
    #----------------------------------------------------------------------
    integral = 0.0;
    for j in range (0,NP):
        x_prime = x1*(1.0-xw[j][0]-xw[j][1])+x2*xw[j][0]+x3*xw[j][1];
        y_prime = y1*(1.0-xw[j][0]-xw[j][1])+y2*xw[j][0]+y3*xw[j][1];

        # Map (x_prime,y_prine) back to original (x,y,z)-coord system:

        P_orig=vec_mat([x_prime,y_prime,0],inv_Pr_to_P_mat)
        P_orig=P_orig+[a,b,c]

        #print"P_recover=",P_recover
        #print"P_original=",P_orig

        Integrand=I_surf(inten1,inten2,inten3,xyz,P_orig)
        #print "Integrand=",Integrand
        integral=integral+Integrand*xw[j][2];

    integral = 0.5*integral;

    return integral
#**************************************************************************
#**************************************************************************
#**************************************************************************
print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
odbPath='D:/MIT_project/python_scripting/to_delete_nonodbs/check_power_radiated.odb'
odbName='check_power_radiated'
stepName='Step-1'



print ("Opening the file...")
Power_target = open("Power_Spectrum.txt", 'w')
Freq_target = open("Frequency.txt", 'w')


Power_target.write("Power[Watts]:");Power_target.write("\n")
Power_target.write("----------------------");Power_target.write("\n");Power_target.close()

Freq_target.write("Frequency[Hertz]:");Freq_target.write("\n");
Freq_target.write("----------------------");Freq_target.write("\n");Freq_target.close()


print (odbPath)
myOdb = session.openOdb(name=odbName,path=odbPath)
Steady_Step = myOdb.steps[stepName]


Outer_Air_Surface = myOdb.rootAssembly.instances['PART-1-1'].surfaces['OUTER_AIR']
Outer_Air_ES = myOdb.rootAssembly.instances['PART-1-1'].elementSets['OUTER_AIR_ES']
#Outer_Air_NS = myOdb.rootAssembly.instances['PART-1-1'].nodeSets['OUTER_AIR_SURFACE_GEOMETRY_SET']
numElements = len(Outer_Air_Surface.elements); print ('numElements=',numElements)



max_x=1.0+1.0/(2.0*pi)
max_y=1.0+1.0/(2.0*pi)
max_z=1.0+1.0/(2.0*pi)


t_start_loops=time.clock() 
for frame in Steady_Step.frames:   
    Total_Power=0.
    Total_Area=0.

    Coord_Array      =frame.fieldOutputs['COORD']#.getSubset(region=Outer_Air_ES,elementType='AC3D4',position=NODAL)
    Inten_subset_data=frame.fieldOutputs['INTEN'].getSubset(region=Outer_Air_ES,elementType='AC3D4',position=ELEMENT_NODAL)

    for el in range(0,numElements-1): 
    #for el in range(0,1): 

        print '###########################'
        print 'el=',el
        element_face=str(Outer_Air_Surface.faces[el]); 
        print 'Element=',Outer_Air_Surface.elements[el].label
        node_list=Outer_Air_Surface.elements[el].connectivity;
   
        # Collect the face-data from each surface element in the database:
        #----------------------------------------------------------------------
        if(element_face=='FACE1'):
            #print 'FACE1'
            node1=node_list[1-1];   node2=node_list[2-1];       node3=node_list[3-1]
            Intensity_1=Inten_subset_data.values[el*4+0].data;  Coord_1=Coord_Array.values[node_list[1-1]-1].dataDouble
            Intensity_2=Inten_subset_data.values[el*4+1].data;  Coord_2=Coord_Array.values[node_list[2-1]-1].dataDouble
            Intensity_3=Inten_subset_data.values[el*4+2].data;  Coord_3=Coord_Array.values[node_list[3-1]-1].dataDouble
            #Intensity_1=Inten_subset_data.values[el*4+0].data;  Coord_1=Coord_Array.values[el*4+0].dataDouble
            #Intensity_2=Inten_subset_data.values[el*4+1].data;  Coord_2=Coord_Array.values[el*4+1].dataDouble
            #Intensity_3=Inten_subset_data.values[el*4+2].data;  Coord_3=Coord_Array.values[el*4+2].dataDouble
            pt1=Coord_1;            pt2=Coord_2;                pt3=Coord_3
            Inten1_vec=Intensity_1; Inten2_vec=Intensity_2;     Inten3_vec=Intensity_3
            Vec1=Coord_2-Coord_3
            Vec2=Coord_1-Coord_2
        elif(element_face=='FACE2'):
            #print 'FACE2'
            node1=node_list[1-1];   node2=node_list[2-1];       node3=node_list[4-1]
            Intensity_1=Inten_subset_data.values[el*4+0].data;  Coord_1=Coord_Array.values[node_list[1-1]-1].dataDouble
            Intensity_2=Inten_subset_data.values[el*4+1].data;  Coord_2=Coord_Array.values[node_list[2-1]-1].dataDouble
            Intensity_4=Inten_subset_data.values[el*4+3].data;  Coord_4=Coord_Array.values[node_list[4-1]-1].dataDouble
            #Intensity_1=Inten_subset_data.values[el*4+0].data;  Coord_1=Coord_Array.values[el*4+0].dataDouble
            #Intensity_2=Inten_subset_data.values[el*4+1].data;  Coord_2=Coord_Array.values[el*4+1].dataDouble
            #Intensity_4=Inten_subset_data.values[el*4+3].data;  Coord_4=Coord_Array.values[el*4+3].dataDouble
            pt1=Coord_1;            pt2=Coord_2;                pt3=Coord_4
            Inten1_vec=Intensity_1; Inten2_vec=Intensity_2;     Inten3_vec=Intensity_4;
            Vec1=Coord_2-Coord_1
            Vec2=Coord_4-Coord_2
        elif(element_face=='FACE3'):
            #print 'FACE3'
            node1=node_list[2-1];   node2=node_list[3-1];       node3=node_list[4-1]
            Intensity_2=Inten_subset_data.values[el*4+1].data;  Coord_2=Coord_Array.values[node_list[2-1]-1].dataDouble
            Intensity_3=Inten_subset_data.values[el*4+2].data;  Coord_3=Coord_Array.values[node_list[3-1]-1].dataDouble
            Intensity_4=Inten_subset_data.values[el*4+3].data;  Coord_4=Coord_Array.values[node_list[4-1]-1].dataDouble
            #Intensity_2=Inten_subset_data.values[el*4+1].data;  Coord_2=Coord_Array.values[el*4+1].dataDouble
            #Intensity_3=Inten_subset_data.values[el*4+2].data;  Coord_3=Coord_Array.values[el*4+2].dataDouble
            #Intensity_4=Inten_subset_data.values[el*4+3].data;  Coord_4=Coord_Array.values[el*4+3].dataDouble
            pt1=Coord_2;            pt2=Coord_3;                pt3=Coord_4
            Inten1_vec=Intensity_2; Inten2_vec=Intensity_3;     Inten3_vec=Intensity_4;
            Vec1=Coord_3-Coord_2
            Vec2=Coord_4-Coord_3
        elif(element_face=='FACE4'):
            #print 'FACE4'
            node1=node_list[1-1];   node2=node_list[3-1];       node3=node_list[4-1]
            Intensity_1=Inten_subset_data.values[el*4+0].data;  Coord_1=Coord_Array.values[node_list[1-1]-1].dataDouble
            Intensity_3=Inten_subset_data.values[el*4+2].data;  Coord_3=Coord_Array.values[node_list[3-1]-1].dataDouble
            Intensity_4=Inten_subset_data.values[el*4+3].data;  Coord_4=Coord_Array.values[node_list[4-1]-1].dataDouble
            #Intensity_1=Inten_subset_data.values[el*4+0].data;  Coord_1=Coord_Array.values[el*4+0].dataDouble
            #Intensity_3=Inten_subset_data.values[el*4+2].data;  Coord_3=Coord_Array.values[el*4+2].dataDouble
            #Intensity_4=Inten_subset_data.values[el*4+3].data;  Coord_4=Coord_Array.values[el*4+3].dataDouble
            pt1=Coord_1;            pt2=Coord_3;                pt3=Coord_4
            Inten1_vec=Intensity_1; Inten2_vec=Intensity_3;     Inten3_vec=Intensity_4;
            Vec1=Coord_4-Coord_1
            Vec2=Coord_3-Coord_4


        # Calculate the normal vector:
        #----------------------------------------------------------------------
        N1= (Vec1[1]*Vec2[2]-Vec2[1]*Vec1[2])
        N2=-(Vec1[0]*Vec2[2]-Vec2[0]*Vec1[2])
        N3= (Vec1[0]*Vec2[1]-Vec2[0]*Vec1[1])
        N=[N1,N2,N3]
        mag=sqrt(N[0]**2+N[1]**2+N[2]**2)
        normal=[N[0]/mag,N[1]/mag,N[2]/mag]
        #print 'normal=',normal


        ## Define the (x,y,z)_j:
        ##----------------------------------------------------------------------
        x1=pt1[0];y1=pt1[1];z1=pt1[2];
        x2=pt2[0];y2=pt2[1];z2=pt2[2];
        x3=pt3[0];y3=pt3[1];z3=pt3[2];


        # Calculate the area of the triangular face:
        #----------------------------------------------------------------------
        a1=x1-x2;   a2=y1-y2;   a3=z1-z2;
        b1=x1-x3;   b2=y1-y3;   b3=z1-z3;
        Area=0.5*np.sqrt(  (a2*b3-b2*a3)**2  +  (a1*b3-b1*a3)**2  +  (a1*b2-b1*a2)**2  )


        #Calculate the local surface normals:
        #----------------------------------------------------------------------
        #r1=sqrt(x1**2+y1**2+z1**2);  theta_1=atan2(y1,x1)-2*pi*min(np.sign(y1),0);        phi_1=acos(z1/r1);
        #r2=sqrt(x2**2+y2**2+z2**2);  theta_2=atan2(y2,x2)-2*pi*min(np.sign(y2),0);        phi_2=acos(z2/r2);
        #r3=sqrt(x3**2+y3**2+z3**2);  theta_3=atan2(y3,x3)-2*pi*min(np.sign(y3),0);        phi_3=acos(z3/r3);
        #n1=[sin(phi_1)*cos(theta_1),sin(phi_1)*sin(theta_1),cos(phi_1)];
        #n2=[sin(phi_2)*cos(theta_2),sin(phi_2)*sin(theta_2),cos(phi_2)];
        #n3=[sin(phi_3)*cos(theta_3),sin(phi_3)*sin(theta_3),cos(phi_3)];


        N1=[2*x1/(max_x**2),2*y1/(max_y**2),2*z1/(max_z**2)];   mag1=sqrt(N1[0]**2+N1[1]**2+N1[2]**2)
        N2=[2*x2/(max_x**2),2*y2/(max_y**2),2*z2/(max_z**2)];   mag2=sqrt(N2[0]**2+N2[1]**2+N2[2]**2)
        N3=[2*x3/(max_x**2),2*y3/(max_y**2),2*z3/(max_z**2)];   mag3=sqrt(N3[0]**2+N3[1]**2+N3[2]**2)

        n1=[N1[0]/mag1,N1[1]/mag1,N1[2]/mag1,]
        n2=[N2[0]/mag2,N2[1]/mag2,N2[2]/mag2,]
        n3=[N3[0]/mag3,N3[1]/mag3,N3[2]/mag3,]


        #print "(node1,node2,node3)=",node1,node2,node3    
        #print "Inten1_vec=",Inten1_vec
        #print "Inten2_vec=",Inten2_vec
        #print "Inten3_vec=",Inten3_vec
        #print"---------------"


        # Calculate I\dot \hat n:
        #----------------------------------------------------------------------
        Inten1=Inten1_vec[0]*n1[0]+Inten1_vec[1]*n1[1]+Inten1_vec[2]*n1[2]
        Inten2=Inten2_vec[0]*n2[0]+Inten2_vec[1]*n2[1]+Inten2_vec[2]*n2[2]
        Inten3=Inten3_vec[0]*n3[0]+Inten3_vec[1]*n3[1]+Inten3_vec[2]*n3[2]



        # Build the points and point matrix:
        #----------------------------------------------------------------------    
        xyz=[[0. for i in range(3)] for j in range(3)]
        xyz[0][0]=x1;  xyz[0][1]=y1;    xyz[0][2]=z1;       P1=[xyz[0][0],xyz[0][1],xyz[0][2]]
        xyz[1][0]=x2;  xyz[1][1]=y2;    xyz[1][2]=z2;       P2=[xyz[1][0],xyz[1][1],xyz[1][2]]
        xyz[2][0]=x3;  xyz[2][1]=y3;    xyz[2][2]=z3;       P3=[xyz[2][0],xyz[2][1],xyz[2][2]]

        # Rotate triangle to horizontal plane: (vertical normal vector)
        #----------------------------------------------------------------------    
        [a,b,c],xyz3,inv_Pr_to_P_mat=Rotate_Triangle_to_xy_plane(xyz)
       
        #----------------------------------------------------------------------
        Power=int_f(2,xyz3[0][0],xyz3[0][1],xyz3[1][0],xyz3[1][1],xyz3[2][0],xyz[2][1],inv_Pr_to_P_mat,a,b,c,xyz,Inten1,Inten2,Inten3)*Area*2.0

        #print "Power=",Power
        #print "Area=",Area
        Total_Area=Total_Area+Area    
        Total_Power=Total_Power+Power    

    Power_target = open("Power_Spectrum.txt", 'a')
    Power_target.write(str(Total_Power));Power_target.write("\n");Power_target.close()
    Freq_target = open("Frequency.txt", 'a')
    Freq_target.write(str(frame.frameValue));Freq_target.write("\n");Freq_target.close()


print"Loop Time=",time.clock()-t_start_loops
print "=============="    
print 'Total_Area=',Total_Area
print 'Total_Power=',Total_Power
