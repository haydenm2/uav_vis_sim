import numpy as np

def Rxa(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return R

def Rxp(theta):
    return Rxa(theta).transpose()

def Rya(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    return R

def Ryp(theta):
    return Rya(theta).transpose()

def Rza(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [ np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R

def Rzp(theta):
    return Rza(theta).transpose()

def Euler2Quaternion(phi, theta, psi):
    # From Small Unmanned Aircraft: Theory and Practice page 261
    e0 = np.cos(psi/2)*np.cos(theta/2)*np.cos(phi/2) + np.sin(psi/2)*np.sin(theta/2)*np.sin(phi/2)
    e1 = np.cos(psi/2)*np.cos(theta/2)*np.sin(phi/2) - np.sin(psi/2)*np.sin(theta/2)*np.cos(phi/2)
    e2 = np.cos(psi/2)*np.sin(theta/2)*np.cos(phi/2) + np.sin(psi/2)*np.cos(theta/2)*np.sin(phi/2)
    e3 = np.sin(psi/2)*np.cos(theta/2)*np.cos(phi/2) - np.cos(psi/2)*np.sin(theta/2)*np.sin(phi/2)
    e = np.array([e0, e1, e2, e3])
    return e

def Quaternion2Euler(e):
    # From Small Unmanned Aircraft: Theory and Practice page 260
    e0 = e.item(0)
    e1 = e.item(1)
    e2 = e.item(2)
    e3 = e.item(3)
    phi = np.arctan2(2*(e0*e1 + e2*e3), (e0**2 + e3**2 - e1**2 - e2**2))
    theta = np.arcsin(2*(e0*e2 - e1*e3))
    psi = np.arctan2(2*(e0*e3 + e1*e2), (e0**2 + e1**2 - e2**2 - e3**2))
    return [phi, theta, psi]

def RotationVehicle2Body(phi, theta, psi):
    R = Rxp(phi) @ Ryp(theta) @ Rzp(psi)
    return R

def RotationBody2Vehicle(phi, theta, psi):
    return RotationVehicle2Body(phi, theta, psi).transpose()

def Quaternion2Rotation(e):
    e0 = e.item(0)
    e1 = e.item(1)
    e2 = e.item(2)
    e3 = e.item(3)

    R = np.array([[e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2, 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)],
                  [2 * (e1 * e2 + e0 * e3), e0 ** 2 - e1 ** 2 + e2 ** 2 - e3 ** 2, 2 * (e2 * e3 - e0 * e1)],
                  [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2]
                  ])
    return R