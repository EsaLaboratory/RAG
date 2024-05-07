import argparse
from datetime import datetime
# import os
# import sys
# sys.path.append(os.path.abspath("./rag/"))
from rag.user import *

def main():
    "Creation of the optim command"
    parser = argparse.ArgumentParser(
        description="Output an LLM's answer to a question on documents")
    parser.add_argument('--prices', 
                        metavar='prices',
                        type=np.array,
                        default=None,
                        help="Prices time series")
    parser.add_argument('--day_price', 
                        metavar='day_price',
                        type=float,
                        default=36.2,
                        help="Daily price")
    parser.add_argument('--night_price', 
                        metavar='night_price',
                        type=float,
                        default=15.8,
                        help="Night price")
    parser.add_argument('--N', 
                        metavar='N',
                        type=int,
                        default=24,
                        help="Number of iterations")
    parser.add_argument('--EV', 
                        metavar='EV',
                        type=int,
                        default=1,
                        help="Number of Electrical Vehicules")
    parser.add_argument('--date', 
                        metavar='date',
                        type=str,
                        default="30/04/24 00:00:00.0",
                        help="Date of simulation")
    parser.add_argument('--plot_path', 
                        metavar='plot_path',
                        type=str,
                        default="../img/optim",
                        help="Path for plot output")
    parser.add_argument('--departure_time', 
                        metavar='departure_time',
                        type=int,
                        default=8,
                        help="End of the simulation")
    parser.add_argument('--arrival_time', 
                        metavar='arrival_time',
                        type=int,
                        default=18,
                        help="Start of the simulation")
    parser.add_argument('--Tmin', 
                        metavar='Tmin',
                        type=float,
                        default=18 + 273.15,
                        help="Minimal temperature Kelvin")
    parser.add_argument('--Tmax', 
                        metavar='Tmax',
                        type=float,
                        default=20 + 273.15,
                        help="Maximal temperature Kelvin")
    parser.add_argument('--weather_forecast', 
                        metavar='weather_forecast',
                        type=np.array,
                        default=None,
                        help="Maximal temperature Kelvin")

    args = parser.parse_args()

    prices = args.prices
    day_price = args.day_price
    night_price = args.night_price
    N = args.N
    EV = args.EV
    date = args.date
    departure_time = args.departure_time
    arrival_time = args.arrival_time
    Tmin = args.Tmin
    Tmax = args.Tmax
    weather_forecast = args.weather_forecast
    plot_path = args.plot_path

    market_kwargs = {
        'prices' : prices,
        'day_price' : day_price,
        'night_price' : night_price,
        'available_power' : None,
        'N' : N
    }
    market = Market(**market_kwargs)
    
    model_kwargs = {
        'EV' : EV, 
        'date' : datetime.datetime.strptime(date, "%d/%m/%y %H:%M:%S.%f"),
        'departure_time' : departure_time,
        'arrival_time' : arrival_time,
        'Tmin' : Tmin,
        'Tmax' : Tmax,
        'market' : market,
        'weather_forecast' : weather_forecast,
        'plot_path' : plot_path,
    }
    user = Model(**model_kwargs)
    tests(user)
    user.optimization()
    user.plot()

if __name__ == '__main__':
    main()