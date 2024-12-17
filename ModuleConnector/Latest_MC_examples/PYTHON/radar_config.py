from __future__ import print_function, division
import sys
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pymoduleconnector
from pymoduleconnector import ModuleConnector
from pymoduleconnector import DataType
from pymoduleconnector.extras.x4_regmap_autogen import X4
from pymoduleconnector.extras.auto import auto
from pymoduleconnector.ids import *

from pymoduleconnector.moduleconnectorwrapper import PyXEP, PyX4M210, XTS_SM_STOP, XTS_SM_MANUAL
from xt_modules_print_info import *
from xt_modules_record_playback_messages import *

# User settings
# the following settings are default X4 configuration for X4M200/X4M300 sensors
x4_par_settings = {'downconversion': 1,  # 0: output rf data; 1: output baseband data
                   'dac_min': 949,
                   'dac_max': 1100,
                   'iterations': 16,
                   'tx_center_frequency': 3, #7.29GHz Low band: 3, 8.748GHz High band: 4
                   'tx_power': 2,
                   'pulses_per_step': 87,
                   'frame_area_offset': 0.18,
                   'frame_area': (0.5, 1),
                   'fps': 17,
                   }


def configure_x4(device_name, record=False, baseband=False, x4_settings=x4_par_settings) -> PyXEP:
    mc = pymoduleconnector.ModuleConnector(device_name)
    # Assume an X4M300/X4M200 module and try to enter XEP mode
    app: PyX4M210 = mc.get_x4m210()
    # Stop running application and set module in manual mode.
    try:
        app.set_sensor_mode(XTS_SM_STOP, 0)  # Make sure no profile is running.
    except RuntimeError:
        # Profile not running, OK
        pass
    try:
        app.set_sensor_mode(XTS_SM_MANUAL, 0)  # Manual mode.
    except RuntimeError:
        pass
    xep = mc.get_xep()

    print('Clearing buffer')
    while xep.peek_message_data_float():
        xep.read_message_data_float()
    print('Start recorder if recording is enabled')
    if record:
        start_recorder(mc)

    print('Set specific parameters')
    # Make sure that enable is set, X4 controller is programmed, ldos are enabled, and that the external oscillator has been enabled.
    xep.x4driver_init()
    # x4_settings['downconversion'] = int(baseband)
    for variable, value in x4_settings.items():
        try:
            # if 'output_control' in variable:
            #     variable = 'output_control'
            setter = getattr(xep, 'x4driver_set_' + variable)
        except AttributeError as e:
            print("X4 does not have a setter function for '%s'." % variable)
            raise e

        if isinstance(value, tuple):
            setter(*value)
        else:
            setter(value)

        print("Setting %s to %s" % (variable, value))
    print_x4_settings(xep)
    return xep


def plot_radar_raw_data_message(xep: PyXEP, baseband=False):
    def read_frame():
        """Gets frame data from module"""
        d = xep.read_message_data_float()  # wait until get data
        frame = np.array(d.data)
        # print('frame length:' + str(len(frame)))
        # Convert the resulting frame to a complex array if downconversion is enabled
        if baseband:
            n = len(frame)
            frame = frame[:n//2] + 1j*frame[n//2:]
        return frame

    def animate(i):
        frame = read_frame()
        if baseband:
            line_real.set_ydata(frame.real)  # update the real part
            line_imag.set_ydata(frame.imag)  # update the imaginary part
        else:
            line.set_ydata(frame)
        return line_real, line_imag if baseband else line,

    fig = plt.figure()
    fig.suptitle("Radar Raw Data")
    ax = fig.add_subplot(1, 1, 1)
    # keep graph in frame (FIT TO YOUR DATA), can be adjusted
    ax.set_ylim(-0.15, 0.15)
    frame = read_frame()
    if baseband:
        line_real, = ax.plot(frame.real, label='Real Part')
        line_imag, = ax.plot(frame.imag, label='Imaginary Part')
        ax.legend()
    else:
        line, = ax.plot(frame)

    ani = FuncAnimation(fig, animate, interval=1)
    try:
        plt.show()
    except:
        print('Messages output finish!')
    sys.exit(0)


def main():
    device_name = auto()[0]
    print_module_info(device_name)
    xep = configure_x4(device_name, False, True, x4_par_settings)
    plot_radar_raw_data_message(xep, True)


if __name__ == "__main__":
    main()


