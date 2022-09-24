from typing import List

import numpy as np
from scipy.optimize import least_squares


def DiscreteG2(commands:List[str], n:int):

    def getCircle(x:np.ndarray, y:np.ndarray) -> tuple:

        def residuals(parameters: np.ndarray) -> np.ndarray:
            """Measure of the distance between points (x, y) and the circle (x0, y0, R)

            Args:
                parameters (np.ndarray): [x0, y0, R]

            Returns:
                np.ndarray: vector of residuals
            """
            x0, y0, R = parameters
            return (x-x0)**2 + (y-y0)**2 - R**2

        x_m = np.mean(x)
        y_m = np.mean(y)
        r_m = np.mean(np.sqrt((x-x_m)**2 + (y-y_m)**2))

        result = least_squares(residuals, x0=[x_m, y_m, r_m])
        x0, y0, R = result.x

        return x0, y0, R


    def getTheta(x0:float, y0:float, R:float, x:float, y:float) -> float:

        x_prime = x - x0
        y_prime = y - y0

        if y_prime > 0:
            return np.arccos(x_prime/R)
        else:
            return -np.arccos(x_prime/R) % (2*np.pi)

    arc_commands = []
    b = 0
    for i, command in enumerate(commands):
        if command.split()[-1] == ';P1':
            i1 = i
            b = 1
        if b:
            arc_commands.append(command)
        if command.split()[-1] == ';P2':
            i2 = i
            break

    x = []
    y = []
    e = []
    for arc_command in arc_commands:
        for parameter in arc_command.split():
            if parameter[0] == 'X':
                x.append(float(parameter[1:]))
            elif parameter[0] == 'Y':
                y.append(float(parameter[1:]))
            elif parameter[0] == 'E':
                e.append(float(parameter[1:]))

    x0, y0, R = getCircle(x, y)

    theta1 = getTheta(x0, y0, R, x[0], y[0])
    theta2 = getTheta(x0, y0, R, x[-1], y[-1])

    theta = theta1 + (theta2 - theta1)*np.arange(1,n)/n
    e = e[0] + (e[-1] - e[0])*np.arange(1,n)/n

    x = R*np.cos(theta) + x0
    y = R*np.sin(theta) + y0

    new_commands = []
    for parameter in zip(x, y, e):
        new_commands.append(f'G1 X{parameter[0]:.3f} Y{parameter[0]:.3f} E{parameter[0]:.4f}')

    return commands[:i1+1] + new_commands + commands[i2:]


def SlowGratings(commands:List[str]):

    b = 0
    for i, command in enumerate(commands):
        if command == ';LAYER:1':
            b = 1
        if b:
            parameters = command.split()
            if ('G0' in parameters) and ('F6000' in parameters):
                parameters.remove('F6000')
                parameters.append('F1500')
                commands[i] = ' '.join(parameters)
        if command == ';LAYER:2':
            break
    
    return commands


def LowerSupport(commands:List[str]):

    b = 0
    for i, command in enumerate(commands):
        if command == ';LAYER:2':
            b = 1
        if b:
            parameters = command.split()
            if ('G0' in parameters) and ('Z' in command):
                for parameter in parameters:
                    if parameter[0] == 'Z':
                        z = float(parameter[1:]) - 0.4
                        parameters.remove(parameter)
                        parameters.append(f'Z{z:.3f}')
                commands[i] = ' '.join(parameters)

    return commands








x=[1, 1/np.sqrt(2), 0]
y=[0, 1/np.sqrt(2), 1]


