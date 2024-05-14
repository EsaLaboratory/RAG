import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import picos as pc
import datetime
from datetime import timedelta
from typing import Callable, Any
import time

def timer(func:Callable[[Any], Any])->Callable[[Any], Any]:
    name=func.__name__
    
    def description(*args, **kwargs):
        arg_str=', '.join(repr(arg) for arg in args)
        start = time.time()
        resultat=func(*args, **kwargs)
        end = time.time()
        if kwargs is None:
            print(f"\nFunction {name}\nargs: {arg_str}\ndone in :{end - start}")
        else:
            key_word = ""
            for key in kwargs.keys():
                if len(repr(kwargs[key])) < 15:
                    key_word += ', ' + key + ": "+ repr(kwargs[key])
                else:
                    key_word += ', ' + key + ": "+ repr(type(kwargs[key]))
            print(f"\nFunction {name}\nargs {arg_str}\nkwargs {key_word}\ndone in :{end - start}")
        return resultat
    return description

class Market():
    """This class describes a basic electricity market.

    Attributes:
        prices: np.array = list of electricity prices.
        available_power: np.array = list of available power.
        dt: float = time delta.
        N: int = number of optimization point.
    """
    @timer
    def __init__(
        self, 
        prices : np.array, 
        day_price : float, 
        night_price : float, 
        available_power : np.array, 
        N : int
        ) -> None:
        """Initialize a Market object.

        Args:        
            prices: np.array = list of electricity prices.
            available_power: np.array = list of available power.
            dt: float = time delta.
            N: int = number of optimization point.
        """
        self.prices = prices
        if available_power is None:
            self.available_power = 100 + 5 * np.random.random(size=N)
        else:
            self.available_power = available_power
        self.N = N
        self.dt = 24 / N
        self.day_price = day_price
        self.night_price = night_price

class Model():
    """This class describes a basic user model. 
    
    The simulation starts at 7 am and ends at 7am the day after. 
    
    Attributes:
        EV: number of electric vehicules. 
        departure_time: leaving date of the vehicule (in the morning).
        arrival_time  : arrival date of the vehicule (in the end of the day).
        Tmin: minimum temparature in the house K.
        Tmax: maximum temparature in the house K.
        market: object containing electricity price evolution.
        EV_energy_goal: fully charged car J.
        V: volume of the house m^3.
        P: pressure of the room kPa.
        max_power: maximum usable power W.
        min_power: minimum usable power W.
        cv: heat capacity kJ/(kg.K).
        k: thermal conductivity W/(m·K).
        s: wall thickness m;
        R: ideal gas constant kJ/(kg.K).
        T0: initial temperature K.
        T_outside: outside temperature K.
        power_t: power consumption for house temperature W.
        power_e: power consumption for EV charging W.
        power: power consumption W.
        energy: electric vehicule energy J.
        temperature: house temperature K.
        cost: cost of the used energy.
    """
    @timer
    def __init__(
        self, 
        EV:  int, 
        date: datetime.datetime, 
        departure_time: int, 
        arrival_time: int, 
        Tmin: float, 
        Tmax: float, 
        market: Market, 
        weather_forecast: np.array,
        plot_path: str,
        ) -> None:
        """Initialization of Model object.
        
        Args:
        EV: number of electric vehicules. 
        departure_time: leaving date of the vehicule (in the morning).
        arrival_time  : arrival date of the vehicule (in the end of the day).
        Tmin: minimum temparature in the house K.
        Tmax: maximum temparature in the house K.
        market: object containing electricity price evolution.
        EV_energy_goal: fully charged car J.
        V: volume of the house m^3.
        surface: surface of walls
        P: pressure of the room kPa.
        max_power: maximum usable power W.
        min_power: minimum usable power W.
        cv: heat capacity kJ/(kg.K).
        Rh: heat transfer resistance m^2.K/W.
        R: specific ideal gas constant kPa.m^3/(kg.K).
        T0: initial temperature K.
        T_outside: outside temperature K.
        power_t: power consumption for house temperature W.
        power_e: power consumption for EV charging W.
        power: power consumption W.
        energy: electric vehicule energy J.
        temperature: house temperature K.
        cost: cost of the used energy.

        Returns:
            A Model instance able to optimize one user's consumption."""
        self.EV = EV
        self.date = date
        self.initial_time = date + timedelta(hours=departure_time)
        self.arrival_time = date + timedelta(hours=arrival_time)
        self.departure_time = date + timedelta(days=1, hours=departure_time) 
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.market = market
        self.T_outside = weather_forecast
        self.EV_energy_goal = 144000000 * self.EV / 3600000 # 40 kwh per car
        self.V = 50
        self.surface = 10
        self.P = 100
        self.max_power = 10
        self.min_power = 0
        self.cv = 0.717
        self.R = 0.287
        self.Rh = 1.4345
        self.T0 = (Tmax + Tmin) / 2
        self.power_t = pc.RealVariable("power_t", self.market.N)
        self.power_e = pc.RealVariable("power_e", self.market.N)
        self.power_h = np.random.random(self.market.N)
        self.power_h = 12 * self.power_h / np.sum(self.power_h)
        self.power = self.power_e + self.power_t + self.power_h
        self.energy = pc.RealVariable("energy", self.market.N+1)
        self.temperature = pc.RealVariable("temperature", self.market.N+1)
        self.cost = 0
        self.plot_path = plot_path
        if self.market.prices is None:
            prices = []
            for i in range(self.market.N):
                if self.day_tariff(i):
                    price = self.market.day_price
                else:
                    price = self.market.night_price
                prices.append([price])
            self.market.prices = np.array(prices)
        if weather_forecast is None:
            T1 = np.linspace(self.T0 - 5, self.T0 -10, self.market.N//2)
            T2 = np.linspace(self.T0 -10, self.T0 - 5, self.market.N//2)
            self.T_outside = np.concatenate((T1, T2))
            self.T_outside += np.random.random(self.market.N)

    def get_index(self, time: datetime.datetime) -> bool:
        """Return the closest index from datetime.
        
        Args:
            time: A datetime object of the simulation.
        
        Returns:
            An intenger between 0 and N and that encodes the time.
        
        Raises:
            Exception:time must be between initial and departure time.
        """
        if time >= self.initial_time and time <= self.departure_time:
            duration = ((time - self.initial_time) / self.market.dt)
            return duration.seconds// 3600
        else:
            raise Exception("time must be between initial and departure time") 

    def get_time(self, i: int) -> datetime.datetime:
        """Return time given an index
        
        Args:
            i: An integer between 0 and N.
        
        Returns:
            A datetime object that encodes the index.
        """
        return self.initial_time + timedelta(hours=(i * self.market.dt) % 24)
    
    def day_tariff(self, i: int) -> bool:
        """Returns a boolean refering to the electricity price of Economy 7.
        
        Args:
            i: An integer between 0 and N.
        
        Returns:
            A boolean that is true if the index refers to a time during the day.
        """
        current_time = self.get_time(i)
        if current_time.hour < 24 and current_time.hour > 7:
            return True
        else:
            return False

    def optimization(self) -> None:
        """ Optimization function

        We define the constraints and the function to optimize then we optmize.
        """
        cost = 0
        K = self.T0 * self.R * self.market.dt * 3600 / (self.P * self.V * self.cv)
        problem = pc.Problem()
        for i in range(self.market.N):
            # we can only charge the vehicule if it is in the house
            current_time = self.get_time(i)
            if current_time < self.arrival_time:
                problem += self.power_e[i] == 0
            # power and energy must be positive
            else:
                problem += self.power_e[i] >= 0
            problem += self.power_t[i] >= self.min_power
            # evolution rule for temperature and energy storage
            problem += self.energy[i + 1] == self.power_e[i] * self.market.dt + self.energy[i]
            problem += self.temperature[i + 1] == K * (self.power_t[i] - 4 * (self.surface * (self.temperature[i] - self.T_outside[i]))/(1000*self.Rh)) + self.temperature[i]
            # temperature must be between Tmin and Tmax
            problem += self.temperature[i + 1] <= self.Tmax, self.temperature[i + 1] >= self.Tmin
            # power must be between min_power and max_power
            problem += self.power[i] <= self.max_power
            # adding the energy used cost
            cost += (self.power_t[i] + self.power_e[i]) * self.market.prices[i]

        # At t0 the temperature must be equal to T0 and the car has no energy left at t0. At the end the car is full.
        problem += self.temperature[0] == self.T0, self.energy[0] == 0, self.energy[-1] >= self.EV_energy_goal, self.temperature[-1] == self.T0
        problem.minimize = cost
        problem.solve(solver="cvxopt")
        self.cost = problem.value
    
    def plot(self):
        """Display a graphical result of the optimization."""
        fig, ax = plt.subplots(4, 1)
        fig.set_size_inches((10, 5))
        fig.suptitle(f"Total cost {round(self.cost,0)} in p")
        # Shrink current axis by 20%
        for axe in ax:
            box = axe.get_position()
            axe.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        time = np.concatenate((np.arange(self.initial_time.hour, 24, self.market.dt).astype(int).astype(str), 
                               np.arange(0, self.departure_time.hour, self.market.dt).astype(int).astype(str)))

        # Plot price evolution
        ax[0].plot(self.market.prices)
        ax[0].set_xticks([])
        ax[0].set_ylabel(r"$Price \quad p$", fontsize=8)

        # Plot power consumption
        ax[1].plot(self.power.value)
        ax[1].plot(np.zeros(len(time)) + self.max_power, 'r--', label='Pmax')
        ax[1].plot(np.zeros(len(time)) + self.min_power, 'b--', label='Pmin')
        ax[1].legend(bbox_to_anchor=(1, 1))
        ax[1].set_xticks([])
        ax[1].set_ylabel(r"$Power \quad kW$", fontsize=8)

        # Plot EV energy
        ax[2].plot(100 *self.energy.value/self.EV_energy_goal)
        positions = [self.get_index(self.arrival_time), self.get_index(self.departure_time)]
        ax[2].set_xticks([]) # TODO add x ticks for arrival and departure time
        ax[2].set_yticks(ticks=np.linspace(0, 100, 3))
        ax[2].set_ylabel(r"$EV \quad energy \quad \%$", fontsize=8)

        # Plot temperature
        ax[3].plot(self.temperature.value - 273.15, label="House")
        ax[3].plot(np.zeros(len(time)) + self.Tmax - 273.15, 'r--', label='Tmax')
        ax[3].plot(np.zeros(len(time)) + self.Tmin - 273.15, 'b--', label='Tmin')
        ax[3].plot(self.T_outside - 273.15, label="Outside")
        ax[3].legend(bbox_to_anchor=(1, 1))
        ax[3].set_xticks(ticks=np.arange(0, len(time), self.market.dt), labels=time)
        ax[3].set_yticks(ticks=np.round(np.linspace(np.min(self.T_outside) - 273.15, self.Tmax - 270.15, 5), 0))
        ax[3].set_ylabel(r"$Temperature \quad °C$", fontsize=8)
        ax[3].set_xlabel(r"$time$", fontsize=8)
        plt.savefig(self.plot_path)

def tests(model : Model) -> None:
    """Test function of the class Model
    
    Raises:
        Exception: get_index({current_time}) != {index})
    """
    N = model.market.N
    for i in range(N):
        current_time = model.get_time(i)
        if i != model.get_index(current_time):
            raise Exception(f"get_index({current_time}) != {i}")