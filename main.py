import logging
import pandas as pd
import pandas_datareader.data as web
import numpy as np

import matplotlib.pyplot as plt

import config as cfg


def clean_colnames(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure that all column names are lowercase and don't contain spaces
    """
    clean_names = {x: x.lower().replace(" ", "_") for x in data.columns}
    return data.rename(columns=clean_names)


def get_symbols_from_nasdaq() -> pd.DataFrame:
    """
    Load symbols of all stocks traded on NASDAQ 
    Text file provided by NASDAQ, available on their FTP server
    Updated daily 
    """
    # load text file from NASDAQs FTP server
    nasdaq_symbols_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    all_stock_symbols = pd.read_csv(nasdaq_symbols_url, sep="|")

    # adjust column names
    clean_stock_symbols = clean_colnames(all_stock_symbols)
    return clean_stock_symbols


def load_etf_data(symbols: list) -> dict:
    etfs = {}
    for symbol in symbols:
        logging.info(f"Load data for {symbol}")
        try:
            data = web.DataReader(
                name=symbol,
                data_source="av-daily-adjusted",
                start=start_time,
                end=end_time,
                api_key=cfg.av_key,
            )
            etfs[symbol] = clean_colnames(data)
        except Exception as e:
            logging.warning(
                f"Error while loading data for {symbol} from Alpha Vantage. Error: {e}",
                exc_info=True,
            )
    return etfs


def calc_portfolio_return(weights: pd.Series, mean_rets: pd.Series) -> float:
    """
    Total Portfolio Return given the respective weights
    """
    return (weights * mean_rets).sum() * 252


def calc_portfolio_std(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """
    Portfolio Variance given the respective covariance Matrix
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)


def calc_sharpe_ratio(port_return: float, port_std: float) -> float:
    """
    Sharpe ratio assuming risk-free rate is zero
    """
    return port_return / port_std


if __name__ == '__main__':

    # inputs
    start_time = pd.Timestamp(2019, 1, 1)
    end_time = pd.to_datetime("now") + pd.DateOffset(days=0, normalize=True)
    size_universe = 10

    # get etf universe (all ETFs traded on NASDAQ)
    stock_symbols = get_symbols_from_nasdaq()
    etf_universe = stock_symbols.query('etf == "Y"')
    etf_symbols = etf_universe["symbol"].tolist()

    # load data
    etf_dict = load_etf_data(symbols=etf_symbols[:5])

    # extract adjusted close and volume data
    etfs = pd.DataFrame()
    for symbol, data in etf_dict.items():
        
        temp_data = data[['adjusted_close', 'volume']]
        temp_data['symbol'] = symbol
        etfs = etfs.append(temp_data)

    # extract etf with largest traded volume
    largest_vol_symbols = etfs.groupby('symbol')['volume'].mean().nlargest(10).index
    largest_etfs = etfs[etfs['symbol'].isin(largest_vol_symbols)]

    # change from long to wide format 
    largest_etfs_wide = largest_etfs.pivot(values='adjusted_close', columns='symbol')
    
    # calc percentage returns 
    # todo: add log return
    pct_rets = largest_etfs_wide.pct_change()

    # calc total returns and covariance
    mean_rets = pct_rets.mean()
    cov = pct_rets.cov()
    
    k = 10000

    results = []
    for _ in range(k):
        rand_nums = np.random.randint(10, size=len(mean_rets))
        rand_weights = rand_nums / rand_nums.sum()
        
        pf_ret = calc_portfolio_return(weights=rand_weights, mean_rets=mean_rets)
        pf_std = calc_portfolio_std(weights=rand_weights, cov_matrix=cov)
        results.append({'ret': pf_ret, 'std': pf_std})

    results_df = pd.DataFrame(results)

    # plot the results
    results_df.plot.scatter(x='std', y='ret')
    plt.show()
    
    # # keep 50 ETFs with largest average volume during last month
    # volumes_mavg: pd.DataFrame = volumes.rolling(window=21).mean()
    # largest_volumes: pd.Series = volumes_mavg.iloc[-1].nlargest(size_universe)

    # # price universe
    # price_universe: pd.DataFrame = prices[largest_volumes.index]

    # rets = price_universe.pct_change()
    # mean_rets = rets.mean()
    # cov_matrix = rets.cov()

    # print(price_universe)
