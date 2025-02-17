import numpy as np

from tabulate import tabulate
from matplotlib import pyplot as plt


class PerformanceEstimator:
    

    def __init__(self, tradingData):

        self.data = tradingData


    def computePnL(self):
       
        # Compute the PnL
        self.PnL = self.data["Money"][-1] - self.data["Money"][0]
        return self.PnL
    

    def computeAnnualizedReturn(self):
        
        # Compute the cumulative return over the entire trading horizon
        cumulativeReturn = self.data['Returns'].cumsum()
        cumulativeReturn = cumulativeReturn[-1]
        
        # Compute the time elapsed (in days)
        start = self.data.index[0].to_pydatetime()
        end = self.data.index[-1].to_pydatetime()     
        timeElapsed = end - start
        timeElapsed = timeElapsed.days

        # Compute the Annualized Return
        if(cumulativeReturn > -1):
            self.annualizedReturn = 100 * (((1 + cumulativeReturn) ** (365/timeElapsed)) - 1)
        else:
            self.annualizedReturn = -100
        return self.annualizedReturn
    
    
    def computeAnnualizedVolatility(self):
        
        # Compute the Annualized Volatility (252 trading days in 1 trading year)
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['Returns'].std()
        return self.annualizedVolatily
    
    
    def computeSharpeRatio(self, riskFreeRate=0):
        # Compute the expected return
        expectedReturn = self.data['Returns'].mean()
        
        # Compute the returns volatility
        volatility = self.data['Returns'].std()
        
        # Compute the Sharpe Ratio (252 trading days in 1 year)
        if expectedReturn != 0 and volatility != 0:
            self.sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sharpeRatio = 0
        return self.sharpeRatio
    
    
    def computeSortinoRatio(self, riskFreeRate=0):
        # Compute the expected return
        expectedReturn = np.mean(self.data['Returns'])
        
        # Compute the negative returns volatility
        negativeReturns = [returns for returns in self.data['Returns'] if returns < 0]
        volatility = np.std(negativeReturns)
        
        # Compute the Sortino Ratio (252 trading days in 1 year)
        if expectedReturn != 0 and volatility != 0:
            self.sortinoRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sortinoRatio = 0
        return self.sortinoRatio
    
    
    def computeMaxDrawdown(self, plotting=False):

        # Compute both the Maximum Drawdown and Maximum Drawdown Duration
        capital = self.data['Money'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through != 0:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through])/capital[peak]
            self.maxDDD = through - peak
        else:
            self.maxDD = 0
            self.maxDDD = 0
            return self.maxDD, self.maxDDD

        # Plotting of the Maximum Drawdown if required
        if plotting:
            plt.figure(figsize=(10, 4))
            plt.plot(self.data['Money'], lw=2, color='Blue')
            plt.plot([self.data.iloc[[peak]].index, self.data.iloc[[through]].index],
                     [capital[peak], capital[through]], 'o', color='Red', markersize=5)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.savefig(''.join(['Figures/', 'MaximumDrawDown', '.png']))
            #plt.show()

        # Return of the results
        return self.maxDD, self.maxDDD
    

    def computeProfitability(self):
        # Initialization of some variables
        good = 0
        bad = 0
        profit = 0
        loss = 0
        index = next((i for i in range(len(self.data.index)) if self.data['Action'][i] != 0), None)
        if index == None:
            self.profitability = 0
            self.averageProfitLossRatio = 0
            return self.profitability, self.averageProfitLossRatio
        money = self.data['Money'][index]

        # Monitor the success of each trade over the entire trading horizon
        for i in range(index+1, len(self.data.index)):
            if(self.data['Action'][i] != 0):
                delta = self.data['Money'][i] - money
                money = self.data['Money'][i]
                if(delta >= 0):
                    good += 1
                    profit += delta
                else:
                    bad += 1
                    loss -= delta

        # Special case of the termination trade
        delta = self.data['Money'][-1] - money
        if(delta >= 0):
            good += 1
            profit += delta
        else:
            bad += 1
            loss -= delta

        # Compute the Profitability
        self.profitability = 100 * good/(good + bad)
         
        # Compute the ratio average Profit/Loss  
        if(good != 0):
            profit /= good
        if(bad != 0):
            loss /= bad
        if(loss != 0):
            self.averageProfitLossRatio = profit/loss
        else:
            self.averageProfitLossRatio = float('Inf')

        return self.profitability, self.averageProfitLossRatio
        

    def computeSkewness(self):
        
        # Compute the Skewness of the returns
        self.skewness = self.data["Returns"].skew()
        return self.skewness
        
    
    def computePerformance(self):
    
        # Compute the entire set of performance indicators
        self.computePnL()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeProfitability()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        self.computeSkewness()

        # Generate the performance table
        self.performanceTable = [["Profit & Loss (P&L)", "{0:.0f}".format(self.PnL)], 
                                 ["Annualized Return", "{0:.2f}".format(self.annualizedReturn) + '%'],
                                 ["Annualized Volatility", "{0:.2f}".format(self.annualizedVolatily) + '%'],
                                 ["Sharpe Ratio", "{0:.3f}".format(self.sharpeRatio)],
                                 ["Sortino Ratio", "{0:.3f}".format(self.sortinoRatio)],
                                 ["Maximum Drawdown", "{0:.2f}".format(self.maxDD) + '%'],
                                 ["Maximum Drawdown Duration", "{0:.0f}".format(self.maxDDD) + ' days'],
                                 ["Profitability", "{0:.2f}".format(self.profitability) + '%'],
                                 ["Ratio Average Profit/Loss", "{0:.3f}".format(self.averageProfitLossRatio)],
                                 ["Skewness", "{0:.3f}".format(self.skewness)]]
        
        return self.performanceTable


    def displayPerformance(self, name):
        
        # Generation of the performance table
        self.computePerformance()
        
        # Display the table in the console (Tabulate for the beauty of the print operation)
        headers = ["Performance Indicator", name]
        tabulation = tabulate(self.performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)
    