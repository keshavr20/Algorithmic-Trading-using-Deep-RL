import argparse
from tabulate import tabulate
from tradingSimulator import TradingSimulator


if(__name__ == '__main__'):

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='TCS', type=str, help="Name of the stock (market)")
    args = parser.parse_args()
    
    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock

    # Training and testing of the trading strategy specified for the stock (market) specified
    simulator.simulateNewStrategy(strategy, stock, saveStrategy=False)
    

