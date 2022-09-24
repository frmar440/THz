from typing import List

import numpy as np
from scipy.optimize import least_squares


FILE = "C:\\Users\\frede\\Documents\\ResearchA22\\CAD\\circular_v03.gcode"
n = 10
speed = 1500
height = 0.4

class GCODEModifier:


    def __init__(self, commands:List[str]):
        """Constructor

        Args:
            commands (List[str]): List of GCODE commands provided by the slicer
        """
        self.commands = commands


    def getCircle(self, x:np.ndarray, y:np.ndarray) -> tuple:
        """Get circle (x0, y0, R) from points (x, y) by solving
        a nonlinear least-squares problem

        Args:
            x (np.ndarray): Array of x positions
            y (np.ndarray): Array of y positions

        Returns:
            tuple: [x0, y0, R]
        """

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


    def getTheta(self, x0:float, y0:float, R:float, x:float, y:float) -> float:
        """Compute polar angle in range [0, 2pi] of point (x, y) on circle (x0, y0, R)

        Args:
            x0 (float): x position of center
            y0 (float): y position of center
            R (float): radius of circle
            x (float): x position of point
            y (float): y position of point

        Returns:
            float: polar angle in range [0, 2pi]
        """

        x_prime = x - x0
        y_prime = y - y0

        arg = x_prime/R

        if arg < -1:
            arg = -1
        elif arg > 1:
            arg = 1

        if y_prime > 0:
            return np.arccos(arg)
        else:
            return -np.arccos(arg) % (2*np.pi)


    def DiscreteG2(self, n:int, speed=1160) -> None:
        """Discrete arc command

        Args:
            n (int): number of discrete steps in printing
        """

        arc_commands = []
        b = False
        for i, command in enumerate(self.commands):
            if command.split()[-1] == ';P1':
                i1 = i
                b = True
            if b:
                arc_commands.append(command[:-2])
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

        x0, y0, R = self.getCircle(x, y)

        theta1 = self.getTheta(x0, y0, R, x[0], y[0])
        theta2 = self.getTheta(x0, y0, R, x[-1], y[-1])

        theta = theta1 + (theta2 - theta1)*np.arange(0,n+1)/n
        e = e[0] + (e[-1] - e[0])*np.arange(0,n+1)/n

        x = R*np.cos(theta) + x0
        y = R*np.sin(theta) + y0

        new_commands = []
        i = 0
        for parameter in zip(x, y, e):
            if i == 0:
                new_commands.append(f'G1 F{speed} X{parameter[0]:.3f} Y{parameter[1]:.3f} E{parameter[2]:.4f}\n')
                i += 1
            else:
                new_commands.append(f'G1 X{parameter[0]:.3f} Y{parameter[1]:.3f} E{parameter[2]:.4f}\n')

        self.commands = self.commands[:i1] + new_commands + self.commands[i2+1:]


    def SlowGratings(self, speed=1500) -> None:
        """Slow G0 commands in printing gratings

        Args:
            speed (int, optional): Printing speed (mm/min). Defaults to 1500.
        """

        b = False
        for i, command in enumerate(self.commands):
            if command == ';LAYER:1\n':
                b = True
            if b:
                parameters = command.split()
                if ('G0' in parameters) and ('F6000' in parameters):
                    parameters.remove('F6000')
                    parameters.append(f'F{speed}')
                    self.commands[i] = ' '.join(parameters) + '\n'
            if command == ';LAYER:2\n':
                break


    def LowerSupport(self, height=0.4) -> None:
        """Lower support printing height for layers 2+

        Args:
            height (float, optional): Lowering height. Defaults to 0.4.
        """
        b = False
        for i, command in enumerate(self.commands):
            if command == ';LAYER:2\n':
                b = True
            if b:
                parameters = command.split()
                if ('G0' in parameters) and ('Z' in command):
                    for parameter in parameters:
                        if parameter[0] == 'Z':
                            z = float(parameter[1:]) - height
                            parameters.remove(parameter)
                            parameters.append(f'Z{z:.3f}')
                    self.commands[i] = ' '.join(parameters) + '\n'


# load commands
with open(FILE, 'r') as f:
    commands = f.readlines()

# apply modifications
modifier = GCODEModifier(commands)
modifier.DiscreteG2(n=n)
modifier.SlowGratings(speed=speed)
modifier.LowerSupport(height=height)

name, ext = FILE.split('.')
# save modified commands
with open(f'{name}_mod.{ext}', 'w') as f:
    f.writelines(modifier.commands)
