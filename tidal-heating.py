import matplotlib.pyplot as plt
import numpy as np
import math
#Defining initial conditions and constants:
p_i_io = np.array((4.234e8,0),dtype = "f")
v_i_io = np.array((0,17.263e3),dtype = "f")
p_i_europa = np.array((6.709e8,0),dtype = "f")
v_i_europa = np.array((0,13.74e3),dtype = "f")
p_i_gany = np.array((0,-1.07e9),dtype = "f")
v_i_gany = np.array((10.788e3,0),dtype = "f")
h = 1.5
G = 6.67e-11
M = 1.898e+27
n = 35000000 
m_io = 8.9319e22
m_europa = 4.8e+22
m_gany = 1.482e+23
pprime_io = p_i_io
vprime_io = v_i_io
pprime_europa = p_i_europa
vprime_europa = v_i_europa
pprime_gany = p_i_gany
vprime_gany = v_i_gany

#Functions that output the gradient in potential
def gradphijup(p):
    gradphijupiter = -G*M/(((p[0]**2+p[1]**2)**1.5))*p
    return gradphijupiter

def gradphiothers(p1,p2,p3,m2,m3):
    rel1 = p2-p1
    rel2 = p3-p1
    gradphi1 = G*m2/(((rel1[0]**2+rel1[1]**2)**1.5))*rel1
    gradphi2 = G*m3/(((rel2[0]**2+rel2[1]**2)**1.5))*rel2
    return gradphi1 + gradphi2

eccentricity = []
semimajoraxis = []

constant_new = 2.897*1e83

for i in range(1,n):
   
    rel = pprime_europa - pprime_io # points from io to europa
   
    vprime_io = vprime_io + h*gradphijup(pprime_io) + \
        h*gradphiothers(pprime_io,pprime_europa, pprime_gany, m_europa,m_gany)
    pprime_io = pprime_io + h*vprime_io
   
    vprime_europa = vprime_europa + h*gradphijup(pprime_europa) + \
        h*gradphiothers(pprime_europa,pprime_io,pprime_gany, m_io,m_gany)
    pprime_europa = pprime_europa + h*vprime_europa
   
    vprime_gany = vprime_gany + h*gradphijup(pprime_gany) + \
        h*gradphiothers(pprime_gany,pprime_io,pprime_europa, m_io,m_europa)
    pprime_gany = pprime_gany + h*vprime_gany
   
   
    E_io = 0.5*np.linalg.norm(vprime_io)**2-\
        (G*M/((pprime_io[0]**2+pprime_io[1]**2)**0.5))  
    J_io = np.linalg.norm(np.cross(pprime_io,vprime_io)) 
    a_io = abs(G*M/(2*E_io)) 
    e_io = (1+2*J_io**2*E_io/(G**2*M**2))**0.5
    dEdt = constant_new*e_io**2/(a_io**(15/2))
    dE = dEdt*h*1e+10
    T_io = (4*math.pi**2/(G*M)*a_io**3)**0.5
   
    E_europa = 0.5*np.linalg.norm(vprime_europa)**2-\
        (G*M/((pprime_europa[0]**2+pprime_europa[1]**2)**0.5))  
    J_europa = np.linalg.norm(np.cross(pprime_europa,vprime_europa)) 
    a_europa = abs(G*M/(2*E_europa)) 
    e_europa = (1+2*J_europa**2*E_europa/(G**2*M**2))**0.5
    T_europa = (4*math.pi**2/(G*M)*a_europa**3)**0.5
   
    E_gany = 0.5*np.linalg.norm(vprime_gany)**2-\
        (G*M/((pprime_gany[0]**2+pprime_gany[1]**2)**0.5))  
    J_gany = np.linalg.norm(np.cross(pprime_gany,vprime_gany)) 
    a_gany = abs(G*M/(2*E_gany)) #+/-
    e_gany = (1+2*J_gany**2*E_gany/(G**2*M**2))**0.5
    T_gany = (4*math.pi**2/(G*M)*a_gany**3)**0.5
   
    #Calculating the updated radial velocity, assuming angular momentum is completely
    #conserved as kinetic energy is removed
    r = np.linalg.norm(pprime_io)
    r_unit = [pprime_io[0]/r,pprime_io[1]/r]
    vtang_unit = [-pprime_io[1]/r,pprime_io[0]/r] #90 clockwise rotation between 
    #r unit and vrad unit
    vtang = np.dot(J_io/r,vtang_unit)
    vrad = vprime_io-vtang
    vrad_unit = np.dot(1/np.linalg.norm(vrad),vrad)
    if np.linalg.norm(vrad)**2-2*dE/m_io < 0 :
        vrad_new = vrad
    else:
        vrad_new = (np.linalg.norm(vrad)**2-2*dE/m_io)**0.5
        vrad_new = np.dot(vrad_new,vrad_unit)
    vnew = vrad_new+vtang
    E = 0.5*(J_io**2/r**2+np.linalg.norm(vrad_new)**2)-\
        (G*M/((pprime_io[0]**2+pprime_io[1]**2)**0.5))
    e = (1+2*J_io**2*E/(G**2*M**2))**0.5
    vprime = vnew
    eccentricity.append(e)
    if math.isnan(np.linalg.norm(vprime)) == True:
        print(i)