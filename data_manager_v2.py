import pandas as pd


def load_data(fpath):
    integrated_data = pd.read_csv(fpath, thousands=',', converters={'date': lambda x: str(x)})
    chart_data = integrated_data[['date', 'open', 'high', 'low', 'close', 'volume']]
    training_data = integrated_data[['date', 'per', 'pbr', 'roe', 
       'ratio_price_ma5', 'ratio_price_ma20', 'ratio_price_ma60', 'ratio_price_ma120', 
       'ratio_volume_ma5', 'ratio_volume_ma20', 'ratio_volume_ma60', 'ratio_volume_ma120', 
       'ratio_market_kospi_ma5', 'ratio_market_kospi_ma20', 'ratio_market_kospi_ma60', 'ratio_market_kospi_ma120', 
       'ratio_market_kosdaq_ma5', 'ratio_market_kosdaq_ma20', 'ratio_market_kosdaq_ma60', 'ratio_market_kosdaq_ma120', 
       'ratio_bond_k3y_ma5', 'ratio_bond_k3y_ma20', 'ratio_bond_k3y_ma60', 'ratio_bond_k3y_ma120']]
    training_data = integrated_data[['date', 'per', 'pbr', 'roe', 
       'ratio_price_ma5', 'ratio_price_ma20', 'ratio_price_ma60', 'ratio_price_ma120', 
       'ratio_volume_ma5', 'ratio_volume_ma20', 'ratio_volume_ma60', 'ratio_volume_ma120']]
    return chart_data, training_data
