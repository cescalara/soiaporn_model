import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from matplotlib import pyplot as plt
import h5py

from ..interfaces.integration import ExposureIntegralTable
from ..interfaces.stan import Direction
from ..interfaces import stan_utility
from ..utils import PlotStyle
from ..plotting import AllSkyMap


__all__ = ['Analysis']


class Analysis():
    """
    To manage the running of simulations and fits based on Data and Model objects.
    """

    def __init__(self, data, model, filename = None):
        """
        To manage the running of simulations and fits based on Data and Model objects.
        
        :param data: a Data object
        :param model: a Model object
        """

        self.data = data
        self.model = model
        self.filename = filename
        if self.filename:
            with h5py.File(self.filename, 'r+') as f:
                f.create_group('input')
                f.create_group('output')
            
        self.simulation_input = None
        self.fit_input = None
        
        self.simulation = None
        self.fit = None

        params = self.data.detector.params
        varpi = self.data.source.unit_vector
        self.tables = ExposureIntegralTable(varpi = varpi, params = params)
     
            
    def build_tables(self, num_points = 50, sim_only = False, fit_only = False):
        """
        Build the necessary integral tables.
        """

        if not fit_only:
            
            # kappa_true table for simulation
            kappa_true = self.model.kappa
                
            self.tables.build_for_sim(kappa_true)
    
        if not sim_only:

            # logarithmically spcaed array with 60% of points between KAPPA_MIN and 100
            kappa_first = np.logspace(np.log(1), np.log(10), int(num_points * 0.7), base = np.e)
            kappa_second = np.logspace(np.log(10), np.log(100), int(num_points * 0.2) + 1, base = np.e)
            kappa_third = np.logspace(np.log(100), np.log(1000), int(num_points * 0.1) + 1, base = np.e)
            kappa = np.concatenate((kappa_first, kappa_second[1:], kappa_third[1:]), axis = 0)
        
            # full table for fit
            self.tables.build_for_fit(kappa)
            
    def use_tables(self, input_filename, main_only = True):
        """
        Pass in names of integral tables that have already been made.
        Only the main table is read in by default, the simulation table 
        must be recalculated every time the simulation parameters are 
        changed.
        """

        if main_only:
            input_table = ExposureIntegralTable(input_filename = input_filename)
            self.tables.table = input_table.table
            self.tables.kappa = input_table.kappa
            
        else:
            self.tables = ExposureIntegralTable(input_filename = input_filename)

        
    def _get_zenith_angle(self, c_icrs, loc, time):
        """
        Calculate the zenith angle of a known point 
        in ICRS (equatorial coords) for a given 
        location and time.
        """
        c_altaz = c_icrs.transform_to(AltAz(obstime = time, location = loc))
        return (np.pi/2 - c_altaz.alt.rad)


    def _simulate_zenith_angles(self):
        """
        Simulate zenith angles for a set of arrival_directions.
        """

        start_time = 2004

        if len(self.arrival_direction.d.icrs) == 1:
            c_icrs = self.arrival_direction.d.icrs[0]
        else:
            c_icrs = self.arrival_direction.d.icrs 

        time = []
        zenith_angles = []
        stuck = []

        j = 0
        first = True
        for d in c_icrs:
            za = 99
            i = 0
            while (za > self.data.detector.threshold_zenith_angle.rad):
                dt = np.random.exponential(1 / self.N)
                if (first):
                    t = start_time + dt
                else:
                    t = time[-1] + dt
                tdy = Time(t, format = 'decimalyear')
                za = self._get_zenith_angle(d, self.data.detector.location, tdy)
        
                i += 1
                if (i > 100):
                    za = self.data.detector.threshold_zenith_angle.rad
                    stuck.append(1)
            time.append(t)
            first = False
            zenith_angles.append(za)
            j += 1
            #print(j , za)
            
        if (len(stuck) > 1):
            print('Warning: % of zenith angles stuck is', len(stuck)/len(zenith_angles) * 100)

        return zenith_angles

    
    def simulate(self, seed = None, Eth_sim = None):
        """
        Run a simulation.

        :param seed: seed for RNG
        """

        eps = self.tables.sim_table

        # handle selected sources
        if (self.data.source.N < len(eps)):
            eps = [eps[i] for i in self.data.source.selection]

        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        f = self.model.f
        F_T = self.model.F_T            
            
        # compile inputs from Model and Data
        self.simulation_input = {
                       'kappa_c' : self.data.detector.kappa_c, 
                       'N_A' : len(self.data.source.distance),
                       'varpi' : self.data.source.unit_vector, 
                       'D' : D,
                       'A' : self.data.detector.area,
                       'a_0' : self.data.detector.location.lat.rad,
                       'theta_m' : self.data.detector.threshold_zenith_angle.rad, 
                       'alpha_T' : alpha_T,
                       'eps' : eps}

        self.simulation_input['F_T'] = F_T
        self.simulation_input['f'] = f 
        self.simulation_input['kappa'] = self.model.kappa
        
        # run simulation
        print('running stan simulation...')
        self.simulation = self.model.simulation.sampling(data = self.simulation_input, iter = 1,
                                                         chains = 1, algorithm = "Fixed_param", seed = seed)

        print('done')

        # extract output
        print('extracting output...')
        self.Nex_sim = self.simulation.extract(['Nex_sim'])['Nex_sim']
        arrival_direction = self.simulation.extract(['event'])['event'][0]
        self.labels = (self.simulation.extract(['lambda'])['lambda'][0] - 1).astype(int)

        # convert to Direction object
        self.arrival_direction = Direction(arrival_direction)
        self.N = len(self.arrival_direction.unit_vector)
        print('done')

        
        # simulate the zenith angles
        print('simulating zenith angles...')
        self.zenith_angles = self._simulate_zenith_angles()
        print('done')
        
        eps_fit = self.tables.table
        kappa_grid = self.tables.kappa
    
        # handle selected sources
        if (self.data.source.N < len(eps_fit)):
            eps_fit = [eps_fit[i] for i in self.data.source.selection]
            
        # prepare fit inputs
        print('preparing fit inputs...')
        self.fit_input = {'N_A' : self.data.source.N, 
                          'varpi' :self.data.source.unit_vector,
                          'D' : D, 
                          'N' : self.N, 
                          'detected' : self.arrival_direction.unit_vector, 
                          'A' : np.tile(self.data.detector.area, self.N),
                          'kappa_c' : self.data.detector.kappa_c,
                          'alpha_T' : alpha_T, 
                          'Ngrid' : len(kappa_grid), 
                          'eps' : eps_fit, 
                          'kappa_grid' : kappa_grid,
                          'zenith_angle' : self.zenith_angles}      
            
        print('done')
        
        
    def save_simulation(self):
        """
        Write the simulated data to file.
        """
        if self.fit_input != None:
            
            with h5py.File(self.filename, 'r+') as f:

                # inputs
                sim_inputs = f['input'].create_group('simulation')
                for key, value in self.simulation_input.items():
                    sim_inputs.create_dataset(key, data = value)

                # outputs
                sim_outputs = f['output'].create_group('simulation')
                sim_outputs.create_dataset('Nex_sim', data = self.Nex_sim)                
                sim_fit_inputs = f['output/simulation'].create_group('fit_input')
                for key, value in self.fit_input.items():
                    sim_fit_inputs.create_dataset(key, data = value)
        else:
            print("Error: nothing to save!")

            
    def plot_simulation(self, cmap = None):
        """
        Plot the arrival directions on a skymap, 
        with a colour scale describing which source 
        the UHECR is from.
        """

        # plot style
        if cmap == None:
            style = PlotStyle()
        else:
            style = PlotStyle(cmap_name = cmap)
            
        # figure
        fig = plt.figure(figsize = (12, 6));
        ax = plt.gca()

        # skymap
        skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);

        self.data.source.plot(style, skymap)
        self.data.detector.draw_exposure_lim(skymap)
       
        Ns = self.data.source.N
        cmap = plt.cm.get_cmap('plasma', Ns + 2) 
        label = True

        try:
            self.lables = self.labels
        except:
            self.labels = np.ones(len(self.arrival_direction.lons))

        for lon, lat, lab in np.nditer([self.arrival_direction.lons, self.arrival_direction.lats, self.labels]):
            color = cmap(lab)
            if label:
                skymap.tissot(lon, lat, 4.0, npts = 30, facecolor = color,
                              alpha = 0.5, label = 'simulated data')
                label = False
            else:
                skymap.tissot(lon, lat, 4.0, npts = 30, facecolor = color, alpha = 0.5)

        # standard labels and background
        skymap.draw_standard_labels(style.cmap, style.textcolor)

        # legend
        plt.legend(bbox_to_anchor = (0.85, 0.85))
        leg = ax.get_legend()
        frame = leg.get_frame()
        frame.set_linewidth(0)
        frame.set_facecolor('None')
        for text in leg.get_texts():
            plt.setp(text, color = style.textcolor)    
        
    def use_simulation(self, input_filename):
        """
        Read in simulated data from a file to create fit_input.
        """

        self.simulation_input = {}
        self.fit_input = {}
        with h5py.File(input_filename, 'r') as f:
            
            sim_output = f['output/simulation']
            sim_input = f['input/simulation']
            for key in sim_input:
                self.simulation_input[key] = sim_input[key].value
                        
            sim_fit_input = sim_output['fit_input']
            for key in sim_fit_input:
                self.fit_input[key] = sim_fit_input[key].value
            self.arrival_direction = Direction(self.fit_input['detected'])
                
    def use_uhecr_data(self):
        """
        Build fit inputs from the UHECR dataset.
        """

        eps_fit = self.tables.table
        kappa_grid = self.tables.kappa
    
        # handle selected sources
        if (self.data.source.N < len(eps_fit)):
            eps_fit = [eps_fit[i] for i in self.data.source.selection]
                            
        print('preparing fit inputs...')
        self.fit_input = {'N_A' : self.data.source.N,
                          'varpi' : self.data.source.unit_vector,
                          'D' : self.data.source.distance,
                          'N' : self.data.uhecr.N,
                          'detected' : self.data.uhecr.unit_vector,
                          'A' : self.data.uhecr.A,
                          'kappa_c' : self.data.detector.kappa_c,
                          'alpha_T' : self.data.detector.alpha_T,
                          'Ngrid' : len(kappa_grid),
                          'eps' : eps_fit,
                          'kappa_grid' : kappa_grid,
                          'zenith_angle' : np.deg2rad(self.data.uhecr.incidence_angle)}
             
        print('done')
        
    def fit_model(self, iterations = 1000, chains = 4, seed = None, sample_file = None, warmup = None):
        """
        Fit a model.

        :param iterations: number of iterations
        :param chains: number of chains
        :param seed: seed for RNG
        """

        # fit
        self.fit = self.model.model.sampling(data = self.fit_input, iter = iterations, chains = chains, seed = seed,
                                             sample_file = sample_file, warmup = warmup)

        # Diagnositics
        self.fit_treedepth = stan_utility.check_treedepth(self.fit)
        self.fit_div = stan_utility.check_div(self.fit)
        self.fit_energy = stan_utility.check_energy(self.fit)
        self.n_eff = stan_utility.check_n_eff(self.fit)
        self.rhat = stan_utility.check_rhat(self.fit)
        
        self.chain = self.fit.extract(permuted = True)
        return self.fit

    def save_fit(self):

        if self.fit:

            with h5py.File(self.filename, 'r+') as f:

                fit_input = f['input'].create_group('fit')
                for key, value in self.fit_input.items():
                    fit_input.create_dataset(key, data = value)
                fit_input.create_dataset('params', data = self.data.detector.params)
                fit_input.create_dataset('theta_m', data = self.data.detector.threshold_zenith_angle.rad)
                fit_input.create_dataset('a_0', data = self.data.detector.location.lat.rad)
                
                fit_output = f['output'].create_group('fit')
                diagnostics = fit_output.create_group('diagnostics')
                diagnostics.create_dataset('treedepth', data = self.fit_treedepth)
                diagnostics.create_dataset('divergence', data = self.fit_div)
                diagnostics.create_dataset('energy', data = self.fit_energy)
                rhat = diagnostics.create_group('rhat')
                for key, value in self.rhat.items():
                    rhat.create_dataset(key, data = value)
                n_eff = diagnostics.create_group('n_eff')
                for key, value in self.n_eff.items():
                    n_eff.create_dataset(key, data = value)      
                samples = fit_output.create_group('samples')
                for key, value in self.chain.items():
                    samples.create_dataset(key, data = value)
                
        else:
            print('Error: no fit to save')
        
