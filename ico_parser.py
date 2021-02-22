"""Module for extracting data from ICOs tables and training Neural Networks."""

import pandas as pd
from datetime import datetime, timedelta
from typing import Callable, Dict, List
from exchange_addresses import ADRESS_LIST
import pytz
import requests
import json
import time

def _sum_dict_values(d1: dict, 
                     d2: dict,
                     lambda_sum: Callable=lambda x, y: x + y) -> dict:
    """Sum que values of correspondent key of two dictionaries.

    Parameters
    ----------
    d1 : dict
        Main dictionary to use as reference.

    d2 : dict
        Other dictionary to get key values to sum to d1 values.

    lambda_sum : Callable (default=lambda x, y: x + y)
        Lambda function to use to add values from dicts.


    Returns
    -------
    res : dict
        Output dictionary with added values from d1 and d2.
    """
    res = d1.copy()
    for key, val in d2.items():
        try:
            res[key] = lambda_sum(res[key], val)
        except KeyError:
            res[key] = val
    return res


def _check_if_holder(
    contract_adress : str,
    list_exchance: List[str] = ADRESS_LIST,
    api_key: str ='NYBDRYT4RGH7I7PGTBKYVBVVZMQ15B4B34',
) -> bool:
    """Check if contract adress is related to a physical holder.

    Parameters
    ----------
    contract_adress : str
        String with contract adress to check for holder.

    list_exchance : List[str] (default=ADRESS_LIST)
        List of Exchange Adresses to check for.

    api_key : str ='NYBDRYT4RGH7I7PGTBKYVBVVZMQ15B4B34'
        Etherscan API key to use for requests.

    Returns
    -------
    bool
        True for holder and False if Exchange.

    """
    if contract_adress in list_exchance:
        return False
    payload = {
        'module': 'proxy',
        'action': 'eth_getCode',
        'address': contract_adress,
        'tag': 'latest',
        'apikey': api_key,
    }
    request = requests.get('https://api.etherscan.io/api', params=payload)
    result = request.json().get('result')
    time.sleep(0.5)
    if not result or result == '0x':
        return True
    elif result != '0x':
        return False


def _get_bigbuyer(dict_cumsum_percentage: Dict) -> Dict:
    """Get biggest holder from a dictionary with daily ICO information.
    
    Parameters
    ----------
    dict_cumsum_percentage : Dict
        Dictionary with daily information about holders.

    Returns
    -------
    dict_percentage_holders : Dict
        Dictionary with biggest holder and respective percentage of coins
        broken by day.

    """

    list_sorted_days = sorted(dict_cumsum_percentage.keys())
    dict_percentage_holders = {}
    dict_holder_status = {}
    counter_api_calls = 0
    for day in list_sorted_days:
        # print(day)
        dict_current_day = dict_cumsum_percentage.get(day)
        # print(dict_current_day)
        if len(dict_current_day) == 0:
            dict_percentage_holders[day] = 0
        else:
            found_holder = False
            while not found_holder:
                if len(dict_current_day) == 0:
                    dict_percentage_holders[day] = [None, 0]
                    break
                else:
                    max_key = max(dict_current_day, key=dict_current_day.get)
                    status_for_key = dict_holder_status.get(max_key)
                    if status_for_key:
                        found_holder = True
                        dict_percentage_holders[day] = [
                            max_key,
                            dict_current_day.get(max_key),
                        ]

                    elif status_for_key == False:
                        del dict_current_day[max_key]
                    elif status_for_key == None:
                        counter_api_calls += 1
                        dict_holder_status[max_key] = _check_if_holder(max_key)

                        if dict_holder_status.get(max_key):
                            # print(max_key)
                            found_holder = True
                            # print(max_key, dict_current_day.get(max_key))
                            dict_percentage_holders[day] = [
                                max_key,
                                dict_current_day.get(max_key),
                            ]
                        else:
                            del dict_current_day[max_key]
    print('Number of API calls', counter_api_calls)
    print(f'List adresses checked: {dict_holder_status.keys()}')
    return dict_percentage_holders


def _filter_df_for_training_days(df: pd.DataFrame, 
                                 date_col: str,
                                 ico_start_date: datetime.date, 
                                 ico_end_date: datetime.date
                                 ) -> pd.DataFrame:
    """
    Slice a dataframe based on start and end date.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to filter.

    date_col : str
        Column date to use for filtering.
    
    ico_start_date : datetime.date
        Start date.
    
    ico_end_date : datetime.date
        End date.

    Returns
    -------
    pd.DataFrame
        A slice dataframe from start to end date.
    
    """
    if ico_start_date:
        return df.loc[
            (df[date_col] >= ico_start_date) & (df[date_col] < ico_end_date)
        ]
    else:
        print('First define ICO start date.')


def _set_dataframe_max_date(df: pd.DataFrame, 
                            date_col: str,
                            max_date: datetime.date) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to filter for max date.

    date_col : str
        Name of column with date information.

    max_date : datetime.date
        Maximum date to use as reference for filtering.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    
    """
    df_max_date = df.copy()
    df_max_date[date_col] = pd.to_datetime(df_max_date[date_col]).dt.date
    return df_max_date[df_max_date[date_col] <= max_date]


class ICOParser:
    def __init__(
        self,
        path_to_csv: str,
        date_column: str='BLOCK_TIMESTAMP',
        value_column: str='VALUE',
        ico_start_date: str=None,
        dateformat: str='%Y-%m-%d',
        fraud_flag: int=None,
        len_time_series: int=60,
        rolling_window_days: int=3,
    ):
        """Class for parsing data coming from ICO.

        Parameters
        ----------
        path_to_csv : str
            Local path to ICO CSV file.

        date_column : str (default='BLOCK_TIMESTAMP')
            Date column name.

        value_column : str (default='VALUE')
            Name of column to carry values.

        ico_start_date : str (default=None)
            Start date to consider for ICO.

        dateformat : str (default='%Y-%m-%d')
            Date format for use when parsing date column.

        fraud_flag : int (default=None)
            1 for fraud and 0 for reliable ICO.

        len_time_series : int (default=60)
            Size of the series to extract from data.

        rolling_window_days : int (default=3)
            Size of rolling windows in days to calculate moving average.

        Attributes
        ----------
        len_time_series : int
            Stored len_time_series parameter.

        fraud_flag : int
            Stored fraud_flag parameter.

        df : pd.DataFrame
            Dataframe loaded from path_to_csv parameter.

        date_column : str
            Stored date_column parameter.

        value_column : str
            Stored value_column parameter.

        df_resample_day : pd.DataFrame
            df parameter daily resampled summing value_column.

        ico_start_date : str
            Stored ico_start_date parameter.

        ico_end_date: str
            Stored ico_end_date parameter.

        rolling_window_days : int
            Stored rolling_window_days parameter.

        df_newuser : pd.DataFrame
            Dataframe filtered for new users.

        df_newuser_resample : pd.DataFrame
            Dataframe filtered for new users resampled on a daily basis.

        dict_balance : dict
            Dictionary storing information about balance for each analysed day.

        dict_cumsum_balance : dict
            Dictionary storing information about balance summing values.

        dict_percentage_holders : dict
            Dictionary storing information about percentage of stocks for each
            holder.

        dict_daily_newholders : dict
            Dictionary storing information about new holders.

        dict_perc_bigbuyer : dict
            Dictionary storing information about biggest percentage of a holder
            for each day.

        dict_newuser_ratio : dict
            Dictionary storing information about ratio between newusers and the
            rest of the holders daily.

        array_daily_transactions : Sequence
            Array with information about daily transactions.

        array_perc_newholders : Sequence
            Array with information about percentage of new holders.

        array_bigbuyer : Sequence
            Array with information about daily biggest buyers.

        array_newuser : Sequence
            Array with information about daily new users.

        array_gaslimit : Sequence
            Array with information about daily gas limit.

        array_daily_transactions_ma : Sequence
            Array with information about daily transactions calculated using
            moving average.

        array_perc_newholders_ma : Sequence
            Array with information about daily percentage of new holders
            calculated using moving average.

        array_bigbuyer_ma : Sequence
            Array with information about biggest buyers calculated using moving
            average.

        array_newuser_ma : Sequence
            Array with information about daily new users calculated using
            moving average.

        array_gaslimit_ma : Sequence
            Array with information about daily gas limit calculated using
            moving average.

        array_autocorrelation_transactions : Sequence
            Array with information about daily transactions autocorrelation.
        """
        # Process start_date and end_date
        ico_start_date = (
            datetime.strptime(ico_start_date, dateformat)
            .replace(tzinfo=pytz.UTC)
            .date()
        )
        ico_end_date = ico_start_date + timedelta(days=len_time_series)

        # Slice df for defined start and end date
        df = pd.read_csv(path_to_csv)
        df.sort_values(by=date_column, inplace=True)
        df['transactions'] = 1

        df[date_column] = pd.to_datetime(df[date_column]).dt.date

        df = df.loc[df[date_column] <= ico_end_date]
        df_for_resample = df.copy()
        df_for_resample[date_column] = pd.to_datetime(
            df_for_resample[date_column]
        )

        self.len_time_series = len_time_series
        self.fraud_flag = fraud_flag
        self.df = df.copy()
        self.date_column = date_column
        self.value_column = value_column
        self.df_resample_day = df_for_resample.resample(
            'D', on=date_column
        ).sum()
        self.ico_start_date = ico_start_date
        self.ico_end_date = ico_end_date
        self.rolling_window_days = rolling_window_days
        self.df_newuser = None
        self.df_newuser_resample = None
        self.dict_balance = None
        self.dict_cumsum_balance = None
        self.dict_percentage_holders = None
        self.dict_daily_newholders = None
        self.dict_perc_bigbuyer = None
        self.dict_newuser_ratio = None
        self.array_daily_transactions = None
        self.array_perc_newholders = None
        self.array_bigbuyer = None
        self.array_newuser = None
        self.array_gaslimit = None
        self.array_daily_transactions_ma = None
        self.array_perc_newholders_ma = None
        self.array_bigbuyer_ma = None
        self.array_newuser_ma = None
        self.array_gaslimit_ma = None
        self.array_autocorrelation_transactions = None

        ## To do:
        self.df_newuser_resample_day = None

    def define_ico_start_date(self):
        """Filter the dataset based on delta parameter in days."""
        change_series = self.df_resample_day['transactions'].pct_change()
        if self.ico_start_date:
            self.ico_end_date = self.ico_start_date + timedelta(
                days=self.len_time_series
            )
        else:
            for index, value in change_series.iteritems():

                if value > 50:

                    if index - timedelta(days=5) in change_series.index:
                        self.ico_start_date = index - timedelta(days=5)
                        self.ico_end_date = self.ico_start_date + timedelta(
                            days=self.len_time_series
                        )
                    else:
                        self.ico_start_date = change_series.index.min()
                        self.ico_end_date = self.ico_start_date + timedelta(
                            days=self.len_time_series
                        )
                    print(self.ico_start_date)
                    break

    def get_array_autocorrelation_transactions(self, nlags=60):
        """Get array of autocorrelation for transaction series.

        Parameters
        ----------

        nlags : int (default=60)
            Size of series to use as reference.

        """
        df_resample_func = self.df_resample_day.reset_index()
        df_resample_func['BLOCK_TIMESTAMP'] = df_resample_func[
            'BLOCK_TIMESTAMP'
        ].dt.date
        array_transactions_from_start = df_resample_func.loc[
            df_resample_func[self.date_column] >= self.ico_start_date
        ]
        self.array_autocorrelation_transactions = sm.tsa.acf(
            array_transactions_from_start.transactions, nlags=nlags
        )

    def filter_df_for_training_days(self, df):
        """Filter input dataframe based on ICO start and end parameter.

        Parameters
        ----------

        df : pd.DataFrame
            Input dataframe to perform filtering.

        """
        if self.ico_start_date:
            return self.df_resample_day.loc[
                (self.df_resample_day.index >= self.ico_start_date)
                & (self.df_resample_day.index < self.ico_end_date)
            ]
        else:
            print('First define ICO start date.')

    def get_newuser_dataframe(self):
        """Get dataframe with transactions of new users (nonce 0 or 1)."""
        df_nonce_01 = self.df[self.df.NONCE.isin([1, 0])]
        list_newuser = list(df_nonce_01.FROM_ADDRESS_BLOCKCHAIN.unique())
        self.df_newuser = self.df[
            self.df.FROM_ADDRESS_BLOCKCHAIN.isin(list_newuser)
        ].reset_index()
        self.df_newuser[self.date_column] = pd.to_datetime(
            self.df_newuser[self.date_column]
        )
        self.df_newuser_resample = self.df_newuser.resample(
            'D', on=self.date_column
        ).sum()

    def get_array_daily_transactions(self):
        """Get series for number of daily transactions."""
        df_resample_func = self.df_resample_day.reset_index()
        df_resample_func['BLOCK_TIMESTAMP'] = df_resample_func[
            'BLOCK_TIMESTAMP'
        ].dt.date
        df_resample_func_filtered = df_resample_func.loc[
            df_resample_func[self.date_column] <= self.ico_end_date
        ]
        list_cumsum = df_resample_func_filtered.transactions.cumsum().to_list()
        list_cumsum_ma = (
            df_resample_func_filtered.transactions.cumsum()
            .rolling(window=self.rolling_window_days)
            .mean()
            .round(4)
            .to_list()
        )
        self.array_daily_transactions = [
            round(val / list_cumsum[-1], 4) for val in list_cumsum
        ][-self.len_time_series :]
        self.array_daily_transactions_ma = [
            round(val / list_cumsum[-1], 4) for val in list_cumsum_ma
        ][-self.len_time_series :]

    def get_balance(self):
        """Process dataframe to extract daily balance for each individual."""
        # Define start date and days of activity
        value_column = self.value_column
        print(self.ico_start_date, self.ico_end_date)
        dataframe = self.df

        dataframe.set_index(self.date_column, inplace=True)
        dataframe[value_column] = dataframe[value_column].astype(float)
        start_date = dataframe.index.min()
        print(start_date)
        days_activity = (dataframe.index.max() - start_date).days
        print(days_activity)
        dict_balance = {}
        for delta in range(days_activity + 1):
            current_date = dataframe.index.min() + timedelta(delta)
            df_current_date = dataframe.loc[dataframe.index == current_date]
            dict_user_balance = {}
            for user in set(
                list(df_current_date.FROM_ADDRESS.unique())
                + list(df_current_date.TO_ADDRESS.unique())
            ):
                to_adress_value = df_current_date.loc[
                    df_current_date.TO_ADDRESS == user
                ].VALUE.sum()
                from_adress_value = df_current_date.loc[
                    df_current_date.FROM_ADDRESS == user
                ].VALUE.sum()
                dict_user_balance[user] = to_adress_value - from_adress_value
            dict_user_balance_sorted = {
                k: v
                for k, v in sorted(
                    dict_user_balance.items(), key=lambda item: item[1]
                )
            }

            dict_balance[str(current_date)] = dict_user_balance_sorted

        self.dict_balance = dict_balance

    def get_cumsum_balance(self):
        """Calculate cumulative sum for dict_balance."""
        dict_balance = self.dict_balance.copy()
        dict_cumsum_balance = {}
        list_sorted_days = sorted(dict_balance.keys())

        for index, day in enumerate(list_sorted_days):
            if index - 1 < 0:
                dict_cumsum_balance[day] = dict_balance.get(day)
            else:
                dict_current_cumsum = _sum_dict_values(
                    dict_cumsum_balance.get(list_sorted_days[index - 1]),
                    dict_balance.get(day),
                )
                dict_cumsum_balance[day] = dict_current_cumsum
        self.dict_cumsum_balance = dict_cumsum_balance

    def get_cumsum_daily_percentage(self):
        dict_cumsum_balance = self.dict_cumsum_balance.copy()
        list_sorted_days = sorted(dict_cumsum_balance.keys())
        dict_percentage_holders = {}
        for day in list_sorted_days:
            total_value = sum(
                [
                    val
                    for val in list(dict_cumsum_balance.get(day).values())
                    if val > 0
                ]
            )
            dict_current_day = dict_cumsum_balance.get(day)
            dict_daily_percentage = {}
            for user in dict_current_day.keys():
                if dict_current_day.get(user) > 0:
                    dict_daily_percentage[user] = (
                        dict_current_day.get(user) / total_value
                    )
            dict_percentage_holders[day] = dict_daily_percentage
        self.dict_percentage_holders = dict_percentage_holders

    def get_bigbuyer_dict(self):
        """Filter dictionary of holders for the biggest ones each day."""
        self.dict_perc_bigbuyer = _get_bigbuyer(
            self.dict_percentage_holders
        )

    def get_bigbuyer_array(self):
        """Extract the series biggest holders."""
        series_bigbuyer_array = pd.Series(
            [
                round(value[1], 4)
                for key, value in self.dict_perc_bigbuyer.items()
            ]
        )
        self.array_bigbuyer = series_bigbuyer_array.round(
            4
        ).tolist()[-self.len_time_series :]
        self.array_bigbuyer_ma = (
            series_bigbuyer_array.rolling(
                window=self.rolling_window_days
            )
            .mean()
            .round(4)
            .tolist()[-self.len_time_series :]
        )

    def get_newuser_ratio_dict(self):
        """Extract dictionary of percentage of daily new users."""
        df_ratio = self.df_newuser_resample / self.df_resample_day
        df_ratio.index = df_ratio.index.astype(str)
        df_ratio.fillna(0, inplace=True)
        self.dict_newuser_ratio = df_ratio.transactions.to_dict()

    def get_newuser_array(self):
        """Extract the series new users."""
        series_newuser = pd.Series(
            [round(val, 4) for val in list(self.dict_newuser_ratio.values())]
        )
        self.array_newuser = series_newuser.round(4).tolist()[
            -self.len_time_series :
        ]
        self.array_newuser_ma = (
            series_newuser.rolling(window=self.rolling_window_days)
            .mean()
            .round(4)
            .tolist()[-self.len_time_series :]
        )

    def get_gaslimit_array(self):
    	"""Extract gas limit series."""
        if not self.df_newuser_resample.empty:
            self.df_newuser_resample['GAS_RATIO'] = (
                self.df_newuser_resample['RECEIPT_GAS_USED']
                / self.df_newuser_resample['GAS']
            )
            self.df_newuser_resample.fillna(0, inplace=True)
            series_gaslimit = self.df_newuser_resample.GAS_RATIO
            self.array_gaslimit = series_gaslimit.round(4).tolist()[
                -self.len_time_series :
            ]
            self.array_gaslimit_ma = (
                series_gaslimit.rolling(window=self.rolling_window_days)
                .mean()
                .round(4)
                .tolist()[-self.len_time_series :]
            )
        else:
            print(
                'self.df_newuser_resample does not exist.\nPlease run self.get_newuser_dataframe().'
            )

    def get_daily_number_of_new_holder(self, max_date=None):
        """Extract dictionary of daily new holders."""
        dict_cumsum = self.dict_cumsum_balance.copy()
        dict_result = {}
        list_sorted_days = sorted(dict_cumsum.keys())
        if not max_date:
            max_users = len(dict_cumsum.get(max(list_sorted_days)))
        else:
            max_users = len(dict_cumsum.get(max_date))

        for day in list_sorted_days:
            total_users = len(dict_cumsum.get(day))
            dict_result[day] = {
                'total_users': total_users,
                'percentage': total_users / max_users,
            }
        self.dict_daily_newholders = dict_result

    def get_array_perc_newholders(self):
    	"""Extract series for newholders."""
        series_perc_newholders = pd.Series(
            [
                round(value.get('percentage'), 4)
                for key, value in self.dict_daily_newholders.items()
            ]
        )
        self.array_perc_newholders = series_perc_newholders.round(
            4
        ).tolist()[-self.len_time_series :]
        self.array_perc_newholders_ma = (
            series_perc_newholders.rolling(window=self.rolling_window_days)
            .mean()
            .round(4)
            .tolist()[-self.len_time_series :]
        )

    def pipeline(self):
        """Call ICOParses instance methods in sequence."""
        print('Running method: get_newuser_dataframe ... ')
        self.get_newuser_dataframe()
        print('Running method: get_balance ... ')
        self.get_balance()
        print('Running method: get_cumsum_balance ... ')
        self.get_cumsum_balance()
        print('Running method: get_cumsum_daily_percentage ... ')
        self.get_cumsum_daily_percentage()
        print('Running method: get_daily_number_of_new_holder ... ')
        self.get_daily_number_of_new_holder()
        print('Running method: get_array_daily_transactions ... ')
        self.get_array_daily_transactions()
        print('Running method: get_array_perc_newholders ... ')
        self.get_array_perc_newholders()
        print('Running method: get_bigbuyer_dict ... ')
        self.get_bigbuyer_dict()
        print('Running method: get_bigbuyer_array ... ')
        self.get_bigbuyer_array()
        print('Running method: get_newuser_ratio_dict ... ')
        self.get_newuser_ratio_dict()
        print('Running method: get_newuser_array ... ')
        self.get_newuser_array()
        print('Running method: get_gaslimit_array ... ')
        self.get_gaslimit_array()

    def pipeline_2_arrays(self):
        """Call ICOParses instance methods in sequence (version 2)."""
        print('Running method: define_ico_start_date ... ')
        self.define_ico_start_date()
        print('Running method: get_newuser_dataframe ... ')
        self.get_newuser_dataframe()
        print('Running method: get_balance ... ')
        self.get_balance()
        print('Running method: get_cumsum_balance ... ')
        self.get_cumsum_balance()
        print('Running method: get_cumsum_daily_percentage ... ')
        self.get_cumsum_daily_percentage()
        print('Running method: get_daily_number_of_new_holder ... ')
        self.get_daily_number_of_new_holder()
        print('Running method: get_array_daily_transactions ... ')
        self.get_array_daily_transactions()
        print('Running method: get_array_perc_newholders ... ')
        self.get_array_perc_newholders()
        print('Running method: get_newuser_ratio_dict ... ')
        self.get_newuser_ratio_dict()
        print('Running method: get_newuser_array ... ')
        self.get_newuser_array()
        print('Running method: get_gaslimit_array ... ')
        self.get_gaslimit_array()



