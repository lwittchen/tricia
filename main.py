import logging
import pandas as pd
import pandas_datareader.data as web
import config as cfg
import pandas as pd


def clean_colnames(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure that all column names are lowercase and don't include spaces
    """
    clean_names = {x: x.lower().replace(' ', '_') for x in data.columns}
    return data.rename(columns=clean_names) 


def get_symbol_info_from_nasdaq() -> pd.DataFrame:
    """
    Load symbols of all stocks traded on NASDAQ 
    Data provided by NASDAQ, available on their FTP server
    Updated daily 
    """
    NASDAQ_SYMBOLS_URL = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt'
    # load text file from NASDAQs FTP server
    all_stocks_symbols = pd.read_csv(NASDAQ_SYMBOLS_URL, sep='|')
    # adjust column names
    return clean_colnames(all_stocks_symbols)

# inputs
start_time = pd.Timestamp(2019, 1, 1)
end_time = pd.to_datetime("now") + pd.DateOffset(days=0, normalize=True)


# get etf universe (all ETFs traded on NASDAQ)
nasdaq_all_stock_symbols: pd.DataFrame = get_symbol_info_from_nasdaq()
nasdaq_all_etf_symbols: pd.DataFrame = nasdaq_all_stock_symbols.query('etf == "Y"')

print('All ETFs:')
print(nasdaq_all_etf_symbols.info())

etf_universe: list = nasdaq_all_etf_symbols['symbol'].tolist()

# load data
etfs = {}
for symbol in etf_universe[:5]:
    logging.info(f'Load data for {symbol}')
    try:
        data: pd.DataFrame = web.DataReader(
            name=symbol,
            data_source="av-daily-adjusted",
            start=start_time,
            end=end_time,
            api_key=cfg.av_key,
        )
        etfs[symbol] = clean_colnames(data)        
    except Exception as e:
        logging.warning(f'Error while loading data for {symbol} from Alpha Vantage. Error: {e}', exc_info=True)

# extract adjusted close and volume data
lst_prices = []
lst_volumes = []
for symbol, data in etfs.items():
    temp_close = data['adjusted_close']
    temp_close.name = symbol
    lst_prices.append(temp_close)

    temp_volume = data['volume']
    temp_volume.name = symbol
    lst_volumes.append(temp_volume)

prices: pd.DataFrame = pd.concat(lst_prices, axis=1)
volumes: pd.DataFrame = pd.concat(lst_volumes, axis=1)

# keep 50 ETFs with largest average volume during last month
volumes_mavg: pd.DataFrame = volumes.rolling(window=21).mean()
largest_volumes: pd.Series = volumes_mavg.iloc[-1].nlargest(50)

# price universe
price_universe: pd.DataFrame = prices[largest_volumes.index]

print(price_universe)