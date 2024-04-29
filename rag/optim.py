import argparse
import datetime
from rag.user import *

def main():
    "Creation of the optim command"
    parser = argparse.ArgumentParser(
        description="Output an LLM's answer to a question on documents")
    parser.add_argument('--reranker', 
                        metavar='reranker',
                        type=str,
                        help="Computes interactions between query document")
    parser.add_argument('--question', 
                        metavar='question', 
                        type=str,
                        help="question on data for llm")
    parser.add_argument('--model_path', 
                        metavar='model_path', 
                        type=str,
                        help="path of the local model (optional)")
    parser.add_argument('--tokenizer_path', 
                        metavar='tokenizer_path', 
                        type=str,
                        help="path of the local tokenizer (optional)")
    parser.add_argument('--save_path', 
                        metavar='save_path', 
                        type=str,
                        help="save path for llm and tokenizer (optional)")
    parser.add_argument('--embedding_model_name', 
                        metavar='embedding_model_name', 
                        type=str,
                        help="Name of embedding model (optional)")
    parser.add_argument('--multiprocess', 
                        metavar='multiprocess', 
                        type=str,
                        help="Options loading embbeding (optional)")
    parser.add_argument('--model_kwargs', 
                        metavar='model_kwargs', 
                        type=str,
                        help="Embeding kwargs, format json (optional)")
    parser.add_argument('--encode_kwargs',
                        metavar='encode_kwargs', 
                        type=str,
                        help="Embeding kwargs, format json (optional)")
    parser.add_argument('--faiss_path', 
                        metavar='faiss_path', 
                        type=str,
                        help="Path for local faiss object")
    parser.add_argument('--tokenizer_path', 
                        metavar='tokenizer_path', 
                        type=str,
                        help="Path for local tokenizer (optional)")
    parser.add_argument('--reranker_name', 
                        metavar='reranker_name', 
                        type=str,
                        help="Name of the reranker (optional)")    

    args = parser.parse_args()

    prices = args.prices
    if args.day_price is not None:
        day_price = args.day_price
    else:
        raise Exception("Provide day price with --day_price")
    if args.night_price is not None:
        night_price = args.night_price
    else:
        raise Exception("Provide night price with --night_price")
    N = args.N
    EV = args.EV
    date = args.date
    departure_time = args.departure_time
    arrival_time = args.arrival_time
    Tmin = args.t_min
    Tmax = args.t_max
    if args.weather_forecast is not None:
        weather_forecast = args.weather_forecast
    else:
        raise Exception("Provide weather forecast with --weather_forecast")

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
        'date' : datetime.strptime(date, format_data="%d/%m/%y %H:%M:%S.%f"),
        'departure_time' : departure_time,
        'arrival_time' : arrival_time,
        'Tmin' : Tmin,
        'Tmax' : Tmax,
        'market' : market,
        'weather_forecast' : weather_forecast,

    }
    user = Model(**model_kwargs)
    tests(user)
    user.optimization()
    user.plot()

if __name__ == '__main__':
    main()