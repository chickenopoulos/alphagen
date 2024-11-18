import pandas as pd
import numpy as np
import empyrical as ep
import talib

class Signal:
    def __init__(self, close=None, high=None, low=None, open=None, volume=None):
        self.close = close if close is not None else pd.DataFrame()
        self.high = high if high is not None else pd.DataFrame() 
        self.low = low if low is not None else pd.DataFrame()
        self.open = open if open is not None else pd.DataFrame()
        self.volume = volume if volume is not None else pd.DataFrame()

    def RSI(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.RSI(self.close[asset], window)
        return pd.DataFrame(d)

    def BBAND_UPPER(self, window, std_mul):
        mean = self.close.rolling(window).mean().shift()
        std = self.close.rolling(window).std().shift()
        return mean + std_mul * std

    def BBAND_LOWER(self, window, std_mul):
        mean = self.close.rolling(window).mean().shift()
        std = self.close.rolling(window).std().shift()
        return mean - std_mul * std

    def SMA(self, window):
        return self.close.rolling(window).mean().shift()

    def ROLL_HIGH(self, window):
        return self.close.rolling(window).max().shift()

    def ROLL_LOW(self, window):
        return self.close.rolling(window).max().shift()

    def ROC(self, window):
        return self.close.ffill().pct_change(window) * 100

    # EMPYRICAL

    def SHARPE(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = ep.roll_sharpe_ratio(self.close[asset].ffill().pct_change(), window)
        d = pd.DataFrame(d)
        nan_dates = list(set(self.close.index) - set(pd.DataFrame(d).index))
        nan_row = pd.DataFrame([[np.nan]*len(self.close.columns)] * len(nan_dates), columns=self.close.columns)
        nan_row.index = nan_dates
        nan_row.sort_index(inplace=True)
        d = pd.concat([nan_row, d])
        d.index.name = 'date'
        return d
        
    def MAXDD(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = ep.roll_max_drawdown(self.close[asset].ffill().pct_change(), window)
        d = pd.DataFrame(d)
        nan_dates = list(set(self.close.index) - set(pd.DataFrame(d).index))
        nan_row = pd.DataFrame([[np.nan]*len(self.close.columns)] * len(nan_dates), columns=self.close.columns)
        nan_row.index = nan_dates
        nan_row.sort_index(inplace=True)
        d = pd.concat([nan_row, d])
        d.index.name = 'date'
        return d
        
    def AVOL(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = ep.roll_annual_volatility(self.close[asset].ffill().pct_change(), window)
        d = pd.DataFrame(d)
        nan_dates = list(set(self.close.index) - set(pd.DataFrame(d).index))
        nan_row = pd.DataFrame([[np.nan]*len(self.close.columns)] * len(nan_dates), columns=self.close.columns)
        nan_row.index = nan_dates
        nan_row.sort_index(inplace=True)
        d = pd.concat([nan_row, d])
        d.index.name = 'date'
        return d
        
    def SORTINO(self, window):
        d = {}   
        for asset in self.close.columns:
            d[asset] = ep.roll_sortino_ratio(self.close[asset].ffill().pct_change(), window)
        d = pd.DataFrame(d)
        nan_dates = list(set(self.close.index) - set(pd.DataFrame(d).index))
        nan_row = pd.DataFrame([[np.nan]*len(self.close.columns)] * len(nan_dates), columns=self.close.columns)
        nan_row.index = nan_dates
        nan_row.sort_index(inplace=True)
        d = pd.concat([nan_row, d])
        d.index.name = 'date'
        return d
        
    # TALIB

    def ADOSC(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.ADOSC(self.high[asset], self.low[asset], self.close[asset], self.volume[asset]) / 10_000
        return pd.DataFrame(d)

    def ADX(self, window=14):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.ADX(self.high[asset], self.low[asset], self.close[asset], window)
        return pd.DataFrame(d)

    def APO(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.APO(self.close[asset])
        return pd.DataFrame(d)

    def AROONOSC(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.AROONOSC(self.high[asset], self.low[asset])
        return pd.DataFrame(d)

    def BOP(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.BOP(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CCI(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CCI(self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDL2CROWS(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDL2CROWS(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDL3BLACKCROWS(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDL3BLACKCROWS(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDL3INSIDE(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDL3INSIDE(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDL3OUTSIDE(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDL3OUTSIDE(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDL3LINESTRIKE(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDL3LINESTRIKE(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)
        
    def CDL3STARSINSOUTH(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDL3STARSINSOUTH(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDL3WHITESOLDIERS(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDL3WHITESOLDIERS(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLABANDONEDBABY(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLABANDONEDBABY(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLADVANCEBLOCK(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLADVANCEBLOCK(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLBELTHOLD(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLBELTHOLD(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLBREAKAWAY(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLBREAKAWAY(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLCLOSINGMARUBOZU(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLCLOSINGMARUBOZU(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLCONCEALBABYSWALL(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLCONCEALBABYSWALL(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLCOUNTERATTACK(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLCOUNTERATTACK(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLDARKCLOUDCOVER(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLDARKCLOUDCOVER(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLDOJI(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLDOJI(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLDOJISTAR(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLDOJISTAR(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLDRAGONFLYDOJI(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLDRAGONFLYDOJI(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLENGULFING(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLENGULFING(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLEVENINGDOJISTAR(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLEVENINGDOJISTAR(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLEVENINGSTAR(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLEVENINGSTAR(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLGAPSIDESIDEWHITE(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLGAPSIDESIDEWHITE(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLGRAVESTONEDOJI(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLGRAVESTONEDOJI(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLHAMMER(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLHAMMER(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLHANGINGMAN(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLHANGINGMAN(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLHARAMI(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLHARAMI(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLHARAMICROSS(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLHARAMICROSS(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLHIGHWAVE(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLHIGHWAVE(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLHIKKAKE(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLHIKKAKE(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLHIKKAKEMOD(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLHIKKAKEMOD(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLHOMINGPIGEON(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLHOMINGPIGEON(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLIDENTICAL3CROWS(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLIDENTICAL3CROWS(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLINNECK(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLINNECK(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLINVERTEDHAMMER(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLINVERTEDHAMMER(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLKICKING(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLKICKING(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLKICKINGBYLENGTH(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLKICKINGBYLENGTH(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLLADDERBOTTOM(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLLADDERBOTTOM(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLLONGLEGGEDDOJI(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLLONGLEGGEDDOJI(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLLONGLINE(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLLONGLINE(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLMARUBOZU(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLMARUBOZU(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLMATCHINGLOW(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLMATCHINGLOW(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLMATHOLD(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLMATHOLD(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLMORNINGDOJISTAR(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLMORNINGDOJISTAR(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLMORNINGSTAR(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLMORNINGSTAR(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLONNECK(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLONNECK(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLPIERCING(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLPIERCING(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLRICKSHAWMAN(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLRICKSHAWMAN(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLRISEFALL3METHODS(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLRISEFALL3METHODS(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLSEPARATINGLINES(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLSEPARATINGLINES(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLSHOOTINGSTAR(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLSHOOTINGSTAR(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLSHORTLINE(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLSHORTLINE(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLSPINNINGTOP(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLSPINNINGTOP(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLSTALLEDPATTERN(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLSTALLEDPATTERN(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLSTICKSANDWICH(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLSTICKSANDWICH(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLTAKURI(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLTAKURI(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLTASUKIGAP(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLTASUKIGAP(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLTHRUSTING(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLTHRUSTING(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLTRISTAR(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLTRISTAR(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLUNIQUE3RIVER(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLUNIQUE3RIVER(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLUPSIDEGAP2CROWS(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLUPSIDEGAP2CROWS(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CDLXSIDEGAP3METHODS(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CDLXSIDEGAP3METHODS(self.open[asset], self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def CMO(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.CMO(self.close[asset])
        return pd.DataFrame(d)

    def DEMA(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.DEMA(self.close[asset], window)
        return pd.DataFrame(d)

    def DX(self):
        d = {}
        for asset in self.close.columns:
            plus_di = talib.PLUS_DI(self.high[asset], self.low[asset], self.close[asset], timeperiod=14)
            minus_di = talib.MINUS_DI(self.high[asset], self.low[asset], self.close[asset], timeperiod=14)
            dx = talib.DX(self.high[asset], self.low[asset], self.close[asset], timeperiod=14)
            
            # Define the DX threshold for trend strength
            dx_threshold = 25
            
            # Initialize a series with zeros (no signal)
            signals = pd.Series(0, index=self.close.index)
            
            # Set to 1 where a buy signal is true
            signals[(plus_di > minus_di) & (dx > dx_threshold)] = 1
            
            # Set to -1 where a sell signal is true
            signals[(minus_di > plus_di) & (dx > dx_threshold)] = -1
            
            d[asset] = signals

        return pd.DataFrame(d)

    def EMA(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.EMA(self.close[asset], window)
        return pd.DataFrame(d)

    def KAMA(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.KAMA(self.close[asset], window)
        return pd.DataFrame(d)

    def LINEARREG(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.LINEARREG(self.close[asset], window)
        return pd.DataFrame(d)

    def LINEARREG_SLOPE(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.LINEARREG_SLOPE(self.close[asset], window)
        return pd.DataFrame(d)

    def LINEARREG_ANGLE(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.LINEARREG_ANGLE(self.close[asset], window)
        return pd.DataFrame(d)

    def MACD(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            line, signal, hist = talib.MACD(self.close[asset])
            macd_df = pd.DataFrame({'line': line, 'signal': signal})
            signals = pd.Series(0, index=macd_df.index)
            signals[macd_df['signal'] > macd_df['line']] = 1  # Buy signal
            signals[macd_df['signal'] < macd_df['line']] = -1  # Sell signal
            d[asset] = signals.values
        return pd.DataFrame(d, index=self.close.index, columns=self.close.columns)   

    def MFI(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.MFI(self.high[asset], self.low[asset], self.close[asset], self.volume[asset], window)
        return pd.DataFrame(d)

    def MOM(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.MOM(self.close[asset], window)
        return pd.DataFrame(d)

    def PPO(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.PPO(self.close[asset])
        return pd.DataFrame(d)

    def SAR(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.SAR(self.high[asset], self.low[asset])
        return pd.DataFrame(d)

    def STDDEV(self, window):
        d = {}
        for asset in self.close.columns:
            i = talib.STDDEV(self.close[asset], window)
            ma = i.rolling(window).mean().shift()
            signal = pd.Series(0, index=self.close.index)
            signal[(i < ma)] = 1
            signal[(i > ma)] = -1
            d[asset] = signal
        return pd.DataFrame(d)

    def STOCH(self, lower, upper):
        d = {}
        for asset in self.close.columns:
            k, d = talib.STOCH(self.high[asset], self.low[asset], self.close[asset])
            # Entry signal (1) when the %K line crosses above the %D line and both are below 20 (oversold)
            # Exit signal (-1) when the %K line crosses below the %D line and both are above 80 (overbought)
            signal = pd.Series(0, index=self.close.index)
            signal[(k.shift(1) < d.shift(1)) & (k > d) & (k < lower) & (d < lower)] = 1
            signal[(k.shift(1) > d.shift(1)) & (k < d) & (k > upper) & (d > upper)] = -1
            d[asset] = signal.values
        return pd.DataFrame(d, index=self.close.index, columns=self.close.columns)

    def STOCHRSI(self, lower, upper):
        d = {}
        for asset in self.close.columns:
            k, d = talib.STOCHRSI(self.close[asset])
            # Entry signal (1) when the %K line crosses above the %D line and both are below 20 (oversold)
            # Exit signal (-1) when the %K line crosses below the %D line and both are above 80 (overbought)
            signal = pd.Series(0, index=self.close.index)
            signal[(k.shift(1) < d.shift(1)) & (k > d) & (k < lower) & (d < lower)] = 1
            signal[(k.shift(1) > d.shift(1)) & (k < d) & (k > upper) & (d > upper)] = -1
            d[asset] = signal.values
        return pd.DataFrame(d, index=self.close.index, columns=self.close.columns)

    def T3(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.T3(self.close[asset], window)
        return pd.DataFrame(d)

    def TEMA(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.TEMA(self.close[asset], window)
        return pd.DataFrame(d)

    def WMA(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.WMA(self.close[asset], window)
        return pd.DataFrame(d)

    def TRIMA(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.TRIMA(self.close[asset], window)
        return pd.DataFrame(d)

    def TRIX(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.TRIX(self.close[asset], window)
        return pd.DataFrame(d)

    def ULTOSC(self, no_input=None):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.ULTOSC(self.high[asset], self.low[asset], self.close[asset])
        return pd.DataFrame(d)

    def WILLR(self, window):
        d = {}
        for asset in self.close.columns:
            d[asset] = talib.WILLR(self.high[asset], self.low[asset], self.close[asset], window).abs()
        return pd.DataFrame(d)

    def ATR(self, window):
        d = {}
        for asset in self.close.columns:
            i = talib.ATR(self.high[asset], self.low[asset], self.close[asset])
            ma = i.rolling(window).mean().shift()
            signal = pd.Series(0, index=self.close.index)
            signal[(i < ma)] = 1
            signal[(i > ma)] = -1
            d[asset] = signal
        return pd.DataFrame(d)

    # BUILD ALPHA

    def WEEKDAY(self, no_input=None):
        return pd.Series([x.weekday() for x in self.close.index], index=self.close.index)    

    def WEEK(self, no_input=None):
        return pd.Series([x.week for x in self.close.index], index=self.close.index)    

    def MONTH(self, no_input=None):
        return pd.Series([x.month for x in self.close.index], index=self.close.index)    

    def ODDDAY(self, no_input=None):
        is_odd_day = self.close.index.day % 2 == 1
        return pd.Series(is_odd_day, index=self.close.index).astype(int).replace(0, -1)

    def CONSECUTIVE_HIGHS(self):
        d = {}
        for asset in self.close.columns:
            higher = self.close[asset].pct_change().diff() > 0
            groups = (~higher).cumsum()
            d[asset] = higher.groupby(groups).cumsum()
        return pd.DataFrame(d)

    def CONSECUTIVE_LOWS(self):
        d = {}
        for asset in self.close.columns:
            lower = self.close[asset].pct_change().diff() < 0
            groups = (~lower).cumsum()
            d[asset] = lower.groupby(groups).cumsum()
        return pd.DataFrame(d)

    def BARPATH(self, path):
        d = {}
        for asset in self.close.columns:
            c = self.close[asset].copy()
            if path == 1:
                _barpath = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) * (c.shift(2) < c.shift(3))
            elif path == 2:
                _barpath = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) * (c.shift(2) > c.shift(3))
            elif path == 3:
                _barpath = (c < c.shift(1)) & (c.shift(1) > c.shift(2)) * (c.shift(2) < c.shift(3))
            elif path == 4:
                _barpath = (c < c.shift(1)) & (c.shift(1) > c.shift(2)) * (c.shift(2) > c.shift(3))
            elif path == 5:
                _barpath = (c > c.shift(1)) & (c.shift(1) < c.shift(2)) * (c.shift(2) < c.shift(3))
            elif path == 6:
                _barpath = (c > c.shift(1)) & (c.shift(1) < c.shift(2)) * (c.shift(2) > c.shift(3))
            elif path == 7:
                _barpath = (c > c.shift(1)) & (c.shift(1) > c.shift(2)) * (c.shift(2) < c.shift(3))
            elif path == 8:
                _barpath = (c > c.shift(1)) & (c.shift(1) > c.shift(2)) * (c.shift(2) > c.shift(3))

            d[asset] = _barpath

        return pd.DataFrame(d)

    def IBR(self):
        d = {}
        for asset in self.close.columns:
            d[asset] = ((self.close[asset] - self.low[asset]) / (self.high[asset] - self.low[asset])).shift().round(2)
        return pd.DataFrame(d)

    def VOL_BREAKOUT(self, mul):
        d = {}
        for asset in self.close.columns:
            d[asset] = (self.high[asset] > self.close[asset].shift() + mul * talib.ATR(self.high[asset], self.low[asset], self.close[asset])).shift()
        return pd.DataFrame(d)

    def keltner_channel(self, high, l, c, window, mul):
        # Calculate the EMA of closing prices
        ema = self.close.ewm(span=window, adjust=False).mean().shift()

        # Calculate the True Range
        high_low = self.high - self.low
        high_close = (self.high - self.close.shift()).abs()
        low_close = (self.low - self.close.shift()).abs()
        true_ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = true_ranges.max(axis=1)

        # Calculate the Average True Range (ATR)
        atr = true_range.rolling(14).mean().shift()

        # Calculate the Upper and Lower Channels
        upper_channel = ema + mul * atr
        lower_channel = ema - mul * atr

        return ema, upper_channel, lower_channel

    def KELTNER_UPPER(self, window, mul):
        d = {}
        for asset in self.close.columns:
            ema, kc_upper, kc_lower = self.keltner_channel(self.high[asset], self.low[asset], self.close[asset], window, mul) 
            d[asset] = kc_upper
        return pd.DataFrame(d)
        
    def KELTNER_LOWER(self, window, mul):
        d = {}
        for asset in self.close.columns:
            ema, kc_upper, kc_lower = self.keltner_channel(self.high[asset], self.low[asset], self.close[asset], window, mul) 
            d[asset] = kc_lower
        return pd.DataFrame(d)

    def KAUF_EFF_RATIO(self, lookback):
        d = {}
        for asset in self.close.columns:
            direction = self.close[asset].diff(lookback).abs().shift()
            volatility = self.close[asset].diff().abs().rolling(lookback).sum().shift()
            kauf_eff_ratio = direction / volatility
            d[asset] = kauf_eff_ratio
        return pd.DataFrame(d)

    def VO(self, period):
        d = {}
        for asset in self.close.columns:
            Rng = self.high[asset] - self.low[asset]
            Med = (self.high[asset] + self.low[asset]) / 2.0
            volUnit = Rng.rolling(period).mean()
            relative = Med.rolling(period).mean()
            vo = (self._open[asset] - relative) / (volUnit * (1.0/period))
            d[asset] = vo.shift()
        return pd.DataFrame(d)

    def VH(self, period):
        d = {}
        for asset in self.close.columns:
            Rng = self.high[asset] - self.low[asset]
            Med = (self.high[asset] + self.low[asset]) / 2.0
            volUnit = Rng.rolling(period).mean()
            relative = Med.rolling(period).mean()
            vh = (self.high[asset] - relative) / (volUnit * (1.0/period))
            d[asset] = vh.shift()
        return pd.DataFrame(d)

    def VL(self, period):
        d = {}
        for asset in self.close.columns:
            Rng = self.high[asset] - self.low[asset]
            Med = (self.high[asset] + self.low[asset]) / 2.0
            volUnit = Rng.rolling(period).mean()
            relative = Med.rolling(period).mean()
            vl = (self.low[asset] - relative) / (volUnit * (1.0/period))
            d[asset] = vl.shift()
        return pd.DataFrame(d)

    def VC(self, period):
        d = {}
        for asset in self.close.columns:
            Rng = self.high[asset] - self.low[asset]
            Med = (self.high[asset] + self.low[asset]) / 2.0
            volUnit = Rng.rolling(period).mean()
            relative = Med.rolling(period).mean()
            vc = (self.close[asset] - relative) / (volUnit * (1.0/period))
            d[asset] = vc.shift()
        return pd.DataFrame(d)

    def WINS_LAST(self, n):
        def count_positives(n):
            return (n > 0).sum()
        
        d = {}
        for asset in self.close.columns:
            roll_pos_count = self.close[asset].pct_change().rolling(n).apply(count_positives, raw=True)
            signal = pd.Series(0, index=self.close.index)
            signal[(roll_pos_count > n/2)] = 1
            signal[(roll_pos_count < n/2)] = -1
            d[asset] = signal.shift()
        return pd.DataFrame(d)

    # ΜOVING AVERAGES

    def WMA(self, period):
        d = {}
        for asset in self.close.columns:
            weights = np.arange(1, period + 1)  # Weighting factors
            wma = self.close[asset].rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            d[asset] = wma.shift()
        return pd.DataFrame(d)

    def SMOOTHED_MA(self, period):
        smma = self.close.ewm(alpha=1/period, adjust=False).mean()
        return smma.shift()

    def VAMA(self, period):
        d = {}
        for asset in self.close.columns:
            # Calculate Volume Weighted Price
            vwp = self.close[asset] * self.volume[asset]
            
            # Calculate the sum of Volume Weighted Price over the period
            sum_vwp = vwp.rolling(window=period).sum()

            # Calculate the sum of volume over the period
            sum_volume = self.volume[asset].rolling(window=period).sum()

            # Calculate VAMA as the ratio of sum_vwp to sum_volume
            vama = sum_vwp / sum_volume
            d[asset] = vama.shift()

        return pd.DataFrame(d)

    def VWAP(self):
        d = {}
        for asset in self.close.columns:
            typical_price = (self.high[asset] + self.low[asset] + self.close[asset]) / 3
            vwap = (typical_price * self.volume[asset]).cumsum() / self.volume[asset].cumsum()
            d[asset] = vwap.shift()
        return pd.DataFrame(d)

    def HMA(self, period):
        d = {}
        for asset in self.close.columns:
            # Calculate the weighted moving average (WMA) with half the period
            half_length_wma = self.close[asset].rolling(window=int(period/2)).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)

            # Calculate the WMA for the full period
            full_length_wma = self.close[asset].rolling(window=period).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)

            # Double the first WMA and subtract the second
            hma_intermediate = 2 * half_length_wma - full_length_wma

            # Calculate the HMA
            hma = hma_intermediate.rolling(window=int(np.sqrt(period))).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)

            d[asset] = hma.shift()

        return pd.DataFrame(d)

    def TMA(self, period):
        # First Simple Moving Average (SMA) with the specified period
        first_sma = self.close.rolling(window=period).mean()

        # Second Simple Moving Average applied on the first SMA
        tma = first_sma.rolling(window=period).mean()
        
        return tma.shift()

    def ZLEMA(self, period):
        d = {}
        for asset in self.close.columns:
            lag = (period - 1) / 2
            lag_adjustment = self.close[asset] - self.close[asset].shift(int(lag))
            zlema = (lag_adjustment + self.close[asset]).ewm(span=period, adjust=False).mean()
            d[asset] = zlema.shift()
        return pd.DataFrame(d)

    def JMA(self, period, smoothing=3):
        # JMA-like indicator, produced by ChatGPT
        
        # Exponential Moving Average with dynamic smoothing
        ema = self.close.ewm(span=period, adjust=False).mean()
        
        # Adjusting EMA to make it smoother and more responsive
        jma_like = ema.ewm(span=smoothing).mean()
        
        return jma_like.shift()
