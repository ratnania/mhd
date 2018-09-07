#!/usr/bin/env python3
'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import numpy as np
from scipy import fft
import OutputData

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class modes(object):
    '''
    Plots the time evolution of various phi fourier harmonics
    '''


    def __init__(self, o=None, nmodes=10, nt=-1, modesfrom=-1, modesto=-1, plotdirectory="."):
        '''
        Constructor
        '''
        
        self.o          = o
        if nt != -1:
            self.nt = nt
        else:
            self.nt         = self.o.ntime-1
        self.t          = self.o.get_scalar_t()[0][:self.nt+1]
        self.ht         = self.t[1] - self.t[0]
        self.length     = self.o.lx
        self.nx         = self.o.nx


        self.plotdirectory = plotdirectory
        self.filename_stem = "./test"

        if modesfrom == -1 or modesto == -1:
            self.modesfrom = 1
            self.modesto   = nmodes
        else:
            self.modesfrom = modesfrom
            self.modesto   = modesto
        
        self.k0 = 2.0*np.pi / self.length
      
        self.phiarray = np.ndarray((self.nt+1, self.nx))
        self.fftarray = np.ndarray((self.nt+1, self.nx), dtype=complex)
        for it in range(self.nt+1):
            self.phiarray[it] = self.o.get_field_x(it)[1][:-1]
            self.fftarray[it] = fft(self.phiarray[it])
        # scale to make value proportional to A, and indep. of nx
        self.fftnormarray = self.fftarray / self.nx
        # only using the positive frequency bins, so double non-zero bins
        self.fftnormarray[:,1:] *= 2.0
        # density coeff. differs from phi coeff. by factor k**2
#        self.fftnormarray[:,:] *= np.arange(self.fftnormarray.shape[1])**2 * self.k0**2


    def modes_plot(self, labelsize=None, labelmult=None, interactive_plots=False):
        fig = plt.figure()
        ax  = plt.subplot(111)

        for imode in range(self.modesfrom, self.modesto+1):
            ax.plot(self.t[:self.nt+1], np.abs(self.fftnormarray[:,imode]), label="n="+str(imode))
        ax.semilogy()
#        box = ax.get_position()
#        ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
#        ax.legend(loc='upper center', bbox_to_anchor=(0.05, -0.05), ncol=5)

        ax.legend(loc=4)
        plt.xlabel("$t$")
        plt.ylabel("$\widetilde{N}_{k=nk_0}$")

        if labelsize is not None:
            plt.gca().xaxis.label.set_size(labelsize)
            plt.gca().yaxis.label.set_size(labelsize)
        if labelmult is not None:
            for tk in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] + plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                tk.set_fontsize(tk.get_fontsize()*labelmult)
        plt.tight_layout()

#        if self.filename_stem.rfind('/') >= 0:
#            filename = self.filename_stem[self.filename_stem.rfind('/'):]
#        else:
        if interactive_plots:
            plt.show(block=False)
            dummy=input("press any key to continue")
        else:
            filename = self.filename_stem
            filename  = filename.replace('.hdf5', '_')
            filename += 'modes'
            fig.savefig(self.plotdirectory + "/" + filename + '.pdf')


    def modes_dump(self):
        nmodes = self.modesto - self.modesfrom + 1
        header = "#" + str(self.modesfrom) + "," + str(self.modesto)
        output = np.zeros((nmodes+1, self.t.size))
        output[0,:] = self.t[:]
        for imc, imode in enumerate(range(self.modesfrom, self.modesto+1)):
                output[imc+1,:] = np.abs(self.fftnormarray[:,imode])
        filename = self.filename_stem + 'modes'
        np.save(self.plotdirectory + "/" + filename + '.npy', output)
        


    def v0s_plot(self, modes=None, interactive_plots=False):
        angles2 = np.zeros((self.modesto+1, self.nt+1))
        xpos    = np.zeros((self.modesto+1, self.nt+1))
        vres    = np.zeros((self.modesto+1, self.nt+1))
        if modes is None:
            modes = range(self.modesfrom, self.modesto+1)
        for imode in modes:
            angles  = np.angle(self.fftnormarray[:,imode])
#            for it in range(1, self.grid.nt+1):
#                angles2[imode, it] = demodulo_angle(angles[it], angles2[imode, it-1])
#                angles2[imode, it] = angles[it]
            angles2[imode] = np.unwrap(angles)
            xpos[imode] = angles2[imode]/(imode*2*np.pi) * self.length
            vres[imode] = np.gradient(xpos[imode], self.ht)
#            xpos[imode] = angles2[imode]
        self.angles_test = angles2
        self.xpos_s = xpos

        fig = plt.figure()
        ax = plt.subplot(211)
        bx = plt.subplot(212, sharex=ax)
        for imode in modes:
            ax.plot(self.t[:self.nt+1], xpos[imode], label="n="+str(imode))
            bx.plot(self.t[:self.nt+1], vres[imode])
        ax.legend()
        if interactive_plots:
            plt.show(block=False)
            dummy=input("press any key to continue")
        else:
            filename = self.filename_stem
            filename  = filename.replace('.hdf5', '_')
            filename += 'v0s'
            fig.savefig(self.plotdirectory + "/" + filename + '.pdf')


    def mode_growth_rate(self, imode, lrangefrom, lrangeto, plotfit=False):
        lintrangefrom = np.rint(lrangefrom / self.ht).astype(int)
        lintrangeto   = np.rint(lrangeto   / self.ht).astype(int)
        self.mode_growth_fit = np.polyfit(
            self.t[:self.nt+1][lintrangefrom:lintrangeto],
            np.log(np.abs(self.fftnormarray[lintrangefrom:lintrangeto, imode])), 1)
        fit_fn = np.poly1d(self.mode_growth_fit)

        if plotfit:
            yl = plt.axes().get_ylim()
            plt.plot(self.t[:self.nt+1], np.exp(fit_fn(self.t[:self.nt+1])), 'k--')
            plt.axes().set_ylim(yl)

        print("mode:", imode, " growth_rate=", self.mode_growth_fit[0], ", k=", imode*self.k0)


    def mode_trap_value(self, imode, lintrangeto=None, lrangeto=None, plottrap=False):
        if lrangeto is None:
            lrangeto = self.t[:self.nt+1][lintrangeto]
        self.trapindex = find_maxima2(np.abs(self.fftnormarray[:, imode]))
        for itm in self.trapindex:
            if self.t[:self.nt+1][itm] > lrangeto:
                self.mode_value = np.abs(self.fftnormarray[itm, imode])
                self.time = self.t[:self.nt+1][itm]
                self.mode_n = imode
                break
            else:
                self.time = None
                self.mode_value = None
                self.mode_n = imode
        if plottrap:
            plt.plot([self.time], [self.mode_value], 'k+')

        print("mode:", imode, " trap_value=", self.mode_value, ", trap_time=", self.time, ", k=", imode*self.k0)
#        self.trapmax, self.ttrapmax = find_maxima(self.fftarray[lintrangefrom, lintrangeto, imode], xarr=self.t[lintrangefrom:lintrangeto])


    def mode_v0(self, imode, lintrangefrom=None, lintrangeto=None, lrangefrom=None, lrangeto=None):
        if lintrangefrom is None:
            if lrangefrom is None:
                lintrangefrom = 0
            else:
                lintrangefrom = np.rint(lrangefrom / self.ht).astype(int)
        if lintrangeto is None:
            if lrangeto is None:
                lintrangeto = self.nt
            else:
                lintrangeto = np.rint(lrangeto / self.ht).astype(int)
        angles       = np.angle(self.fftnormarray[:,imode])
        self.angles  = angles
        angles2      = np.zeros_like(angles)
        self.angles2 = angles2
        angles2[0]   = angles[0]

#        for it in range(1, self.nt+1):
#            angles2[it] = demodulo_angle(angles[it], angles2[it-1])
        angles2 = np.unwrap(angles)

        xpos             = angles2/(imode*2*np.pi) * self.length
        self.xpos = xpos
        self.mode_v0_fit = np.polyfit(
            self.t[:self.nt+1][lintrangefrom:lintrangeto],
            -xpos[lintrangefrom:lintrangeto], 1)
        print("mode:",imode, " v0=", self.mode_v0_fit[0])
        return self.mode_v0_fit[0]



    def __del__(self):
        pass
        

def find_maxima2(yarr):
    indexarr = []
    for it in range(0, yarr.size):
        itm = (it-1+yarr.size) % (yarr.size)
        itp = (it+1+yarr.size) % (yarr.size)
        if yarr[it] > yarr[itm] and yarr[it] > yarr[itp]:
            indexarr.append(it)
    return indexarr

def demodulo_angle(angle_new, angle_old):
    if angle_new > angle_old:
        sign = -1
    else:
        sign = +1
    if np.abs(angle_new - angle_old) > np.pi:
        angle_new = demodulo_angle(angle_new + sign*(2.0*np.pi), angle_old)
    return angle_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pic1dp Solver in 1D')
    parser.add_argument('directory', metavar='<dir>', type=str, help='data directory')
    parser.add_argument('-nmodes', metavar='i', type=int, default=10, help='plot the first i modes')
    parser.add_argument('-modesFrom', metavar='i', type=int, default=-1,
                        help='plot modes from')
    parser.add_argument('-modesTo', metavar='i', type=int, default=-1,
                        help='plot modes to')
    parser.add_argument('-readRange', metavar='i', type=int, default=0,
                        help='whether to try to read the range for fitting from file (1 or 0)')
    parser.add_argument('-writeRange', metavar='i', type=int, default=0,
                        help='whether to write the chosen range to file (0 or 1)')
#    parser.add_argument('-timeFrom', metavar='t', type=float, default=-1.0,
#                        help='plot modes from')
#    parser.add_argument('-timeTo', metavar='t', type=float, default=-1.0,
#                        help='plot modes to')

    parser.add_argument('-plotv0s', metavar='i', type=int, default=0,
                        help='whether to plot the v0 values')
    parser.add_argument('-fitGR', metavar='i', type=int, default=0,
                        help='whether to fit the growthrates of the modes (using from/to)')
    parser.add_argument('-nt', metavar='i', type=int, default=-1,
                        help='use only the first nt timepoints')
    parser.add_argument('-labelsize', metavar='i', type=float, default=None)
    parser.add_argument('-labelmult', metavar='i', type=float, default=None)
    parser.add_argument('-interactivePlots', metavar='i',  type=int, default='0')
    
    args = parser.parse_args()
    
    print
    print("Mode amplitudes from phi(x,t)" + args.directory)
    print
    o = OutputData.OutputData(args.directory)
    o.dirname = args.directory

    plotdir = args.directory
    if plotdir == "":
        plotdir = "."

    pot = modes(o=o, nmodes=args.nmodes, nt=args.nt, modesfrom=args.modesFrom,
                modesto=args.modesTo, plotdirectory=plotdir)

    if args.plotv0s == 1:
        pot.v0s_plot(interactive_plots=args.interactivePlots)
    else:
        pot.modes_plot(labelsize=args.labelsize, labelmult=args.labelmult,
                       interactive_plots=args.interactivePlots)
    pot.modes_dump()


    # put all processing into the class, and preferably reuse between other diagnostics
    if args.fitGR == 1:
        tfitfrom = None
        tfitto   = None
        range_filename = plotdir + '/modes.in'
        if args.readRange == 1:
            try:
                with open(range_filename, 'r') as f:
                    tfitfrom = float(f.readline())
                    tfitto   = float(f.readline())
            except FileNotFoundError:
                print("debug: FNFE", range_filename)

        if tfitfrom is None or tfitto is None:
            tfitfrom = float(input("pick range: from: "))
            tfitto   = float(input("pick range: to: "))
            if args.writeRange == 1:
                with open(range_filename, 'w') as f:
                    f.write(str(tfitfrom) + "\n")
                    f.write(str(tfitto) + "\n")

        for imode in range(pot.modesfrom, pot.modesto+1):
            pot.mode_growth_rate(imode, tfitfrom, tfitto, plotfit=True)
            pot.mode_trap_value(imode, lrangeto=tfitto, plottrap=True)
        plt.axvspan(tfitfrom, tfitto, facecolor='0.1', alpha=0.2)
        if args.interactivePlots:
            plt.show(block=False)
            dummy=input("any key to continue, 'N' not to save")
            if dummy.lower() == "n":
                print("not saving file")
            else:
                #TODO: make better
                filename = pot.filename_stem
                filename += 'modes_with_gr'
                plt.savefig(pot.plotdirectory + '/' + filename + '.pdf')
        else:
            filename = pot.filename_stem
            filename += 'modes_with_gr'
            plt.savefig(pot.plotdirectory + '/' + filename + '.pdf')
    else:
        for imode in range(pot.modesfrom, pot.modesto+1):
            itmax = np.argmax(np.abs(pot.fftnormarray[:,imode]))
            tmax  = pot.t[itmax]
            vmax  = np.abs(pot.fftnormarray[itmax, imode])
