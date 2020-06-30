import datetime
from datetime import datetime as dt
from datetime import timedelta as td
import os
import re as reg
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.offline import plot
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import pandas as pd



#Constant
EPOCH_TIME = datetime.datetime.utcfromtimestamp(0)




#Functions
def access_csv_url(url):
    """This python module downloads the csv from a url passed
    and return the dataframe else raises a FileNotFoundError exception"""
    df = pd.read_csv(url)
    if df.empty is False:
        return df
    else:
        raise FileNotFoundError

def get_company_name(company_df):
    """lvl1 @2 : Following line loads the initial data of list of companies"""
    company_symbol = ''
    while (company_df.empty is False and company_symbol == ''):
        surety_value = input("Are you sure of company symbol you want to search? (Y/N/E(Exit)) : ")
        if surety_value.upper() == 'Y' or surety_value.upper() == 'N':
            if surety_value.upper() == 'N':
                search_dict = company_search('', company_df)
                if len(search_dict) == 0:
                    print("\n No related results found, Give it another Try!!")
                    continue
                elif len(search_dict) > 0:
                    if len(search_dict) > 10:
                        print("Showing Top 10 results for your search which gave ", len(search_dict), " results")
                    else:
                        print("found ", str(len(search_dict)), "results")

                    print(" \t Symbol \t Name")
                    print("\t _________", "\t", "_________")
                    for index, key in enumerate(search_dict.keys()):
                        if index+1 == 11:
                            break
                        else:
                            print("\t", key, "\t\t", search_dict[key])
                    surety_value = input("Have you found your symbol yet ? Y/N : ")
                    if surety_value.upper() == 'N' or surety_value.upper() == 'Y':
                        if surety_value.upper() == 'Y':
                            company_symbol = input("Enter the final symbol : ")
                            search_dict = company_search(company_symbol, company_df)
                            if len(search_dict) > 1:
                                print("Your search resulted into multiple results, please reselect your company!")
                                company_symbol = '' #resetting the value so that value can be input again
                            elif len(search_dict) == 0:
                                print("Your search yielded no results")
                                company_symbol = ''
                        else:
                            continue
                    else:
                        print("please choose only Y or N or y or n or E or e")
                        continue
            elif surety_value.upper() == 'Y':
                company_symbol = input("Enter the final symbol : ")
                search_dict = company_search(company_symbol, company_df)
                if len(search_dict) > 1:
                    print("Your search resulted into multiple results, please reselect your company!")
                    company_symbol = '' #resetting the value so that value can be input again
                elif len(search_dict) == 0:
                    print("Your search yielded no results")
                    company_symbol = ''
        elif surety_value.upper() == 'E':
            company_symbol = ''
            break
        else:
            print("please choose only Y or N or y or n")
            continue
    return company_symbol.upper()

def file_exists(filename, filepath):
    file_tree = [file for file in os.listdir(filepath) if os.path.isfile(file)]
    if filename not in file_tree:
        return False
    else:
        return True

def update_company_data(company_symbol):
    """If a file does not exit then data will be downloaded from the website and save in the file with that company name"""
    file_name = (str(company_symbol)+'.csv')
    existing_file = file_exists(file_name, '.')
    end_date = dt.date(dt.utcnow() - td(seconds=14400))
    if existing_file is False:
        alpha_url = f"http://quotes.wsj.com/{company_symbol}/historical-prices/download?MOD_VIEW=page%20&num_rows=7500&range_days=7500&startDate=11/01/1970%20&endDate={end_date}" #mm/dd/yyyy
        company_data = pd.read_csv(alpha_url)
        company_data.columns = [col_name.lstrip() for col_name in company_data.columns]
        company_data['Date'] = pd.to_datetime(company_data['Date'], format='%m/%d/%y')
        company_data['Date'] = company_data['Date'].dt.date
        company_data = company_data.sort_values(by='Date')
        if not company_data.empty:
            company_data.to_csv(f"{company_symbol}.csv")
    else:
        """if the file exists, read the last line and update the data until todays date"""
        company_data = pd.read_csv(file_name, index_col=0)
        company_data['Date'] = company_data['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
        row = company_data.sort_values('Date').tail(1)
        date = row['Date']
        date = str(date).split()
        date = datetime.datetime.strptime(date[1], '%Y-%m-%d').date() + td(days=1)
        if end_date != date:
            remaining_df = pd.read_csv('http://quotes.wsj.com/'+company_symbol+'/historical-prices/download?MOD_VIEW=page%20&num_rows=7500&range_days=7500&startDate='+str(date.month)+'/'+str(date.day)+'/'+str(date.year)+'%20&endDate='+str(end_date.month)+'/'+str(end_date.day)+'/'+str(end_date.year))
            remaining_df.columns = company_data.columns
            remaining_df['Date'] = pd.to_datetime(remaining_df['Date'], format='%m/%d/%y')
            remaining_df['Date'] = remaining_df['Date'].dt.date
            company_data = company_data.append(remaining_df, sort=False)
            company_data.columns = [col_name.lstrip() for col_name in company_data.columns]
            company_data['Date'] = pd.to_datetime(company_data['Date'], format='%Y-%m-%d')
            company_data['Date'] = company_data['Date'].dt.date
            company_data = company_data.sort_values(by='Date')
            company_data.reset_index(inplace=True, drop=True)
            company_data.to_csv(str(company_symbol)+'.csv')
    return company_data



def print_menu(company_symbol):
    """This prints the main user menu with dynamic company name"""
    print("/" + "-" * 56 + "\\")
    #print(f"\t\t   USER MENU: {company_symbol}")
    print(f"\t      Stock Analysis MENU of {company_symbol}\t\t\t\t ")
    print("|" + "-" * 56 + "|")
    print("| 1.  Current Data\t\t\t\t\t |")
    print("| 2.  Summary Statistic \t\t\t\t |")
    print("| 3.  Raw time-series \t\t\t\t\t |")
    print("| 4.  Linear trend line \t\t\t\t |")
    print("| 5.  Moving Averages \t\t\t\t\t |")
    print("| 6.  Predict close price for a day \t\t\t |")
    print("| 7.  Enhance Prediction \t\t\t\t |")
    print("| 8.  Predict close price for N-future days\t\t |")
    print("| 9.  Compare 2 companies using candlestick chart\t |")
    print("| 10. Analyse with new start and end date\t\t |")
    print("| 11. Search New Company \t\t\t\t |")
    print("| 12. Exit \t\t\t\t\t\t |")
    print("\\" + "-" * 56 + "/")

def date_validation(start_date, end_date):
    try:
        #Check for format of start date, It should be in format of YYYY-MM-DD
        datetime.datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        #If any errors, raise an exception
        print("Incorrect Start Date Format")
        return '1'
    try:
        #Check for format of start date, It should be in format of YYYY-MM-DD
        datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        #If any errors, raise an exception
        print("Incorrect End Date Format")
        return '2'
    try:
        #Start Date cannot be later than today
        if datetime.datetime.strptime(start_date, "%Y-%m-%d") >= dt.today():
            raise ValueError
    except ValueError:
        #If any errors, raise an exception
        print("Start Date cannot be greater than today's date")
        return '3'
    try:
        #End date cannot be greater than today
        if datetime.datetime.strptime(end_date, "%Y-%m-%d") >= dt.today():
            raise ValueError
    except ValueError:
        #If any errors, raise an exception
        print("End Date cannot be greater than today's date")
        return '4'
    try:
        #Start date can not greater than end date
        if datetime.datetime.strptime(start_date, "%Y-%m-%d") >= datetime.datetime.strptime(end_date, "%Y-%m-%d"):
            raise ValueError
    except ValueError:
        print("Start Date should be less than End date")
        return '5'

def period_validation(start_date, end_date, period):
    try:
        #Period should be greater than 0 and less than days between start date and end date
        if period < (end_date - start_date).days and period > 0:
            return False
        else:
            raise ValueError
    except ValueError:
        print('Incorrect value of Window')
        return '1'

def current_data(company_symbol):
    """This API gives statistical data of the last working business day"""
    last_stats_url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=" + company_symbol + "&apikey=T11CBFXU1UTRD2KG&datatype=csv"
    last_stats = pd.read_csv(last_stats_url)
    last_stat_transposed = last_stats.T
    print(last_stat_transposed)

def norm_values_plot(norm_values, start_date, end_date, company_symbol):
    """Plotting of normalised values"""
    fig = go.Figure(data=[go.Scatter(x=norm_values['Date'], y=norm_values['NormValues'], name="Normalised Prices")])
    fig.update_layout(title=f"Normalised Prices of {company_symbol}", xaxis_title=f"Date range from {start_date} to {end_date}",
                      yaxis_title="Normalised Prices", showlegend=True, font=dict(size=18))
    plot(fig, filename=f'{company_symbol}_normalised_graph.html')

def raw_time_series(required_data, company_symbol, start_date, end_date):
    """Plotting Closing Values of the company symbol selected"""
    required_data.Date = pd.to_datetime(required_data['Date'])
    fig = go.Figure(data=go.Scatter(x=required_data['Date'], y=required_data['Close'], name='Closing price'),
                    layout=go.Layout(title=go.layout.Title(text="Linear Raw Time Series", font=dict(size=18)),
                                     yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=f"Closing price of {company_symbol}", font=dict(size=18))),
                                     xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=f"Date range from {start_date} to {end_date}", font=dict(size=18)))))
    fig.update_layout(showlegend=True)
    plot(fig, filename=f'{company_symbol}_raw_time_graph.html')

def linear_trend_line(required_data, start_date, end_date, company_symbol):
    """Plotting linear trend line which indicates if the closing price at the end date
       is higher or lower from the closing price during the start date"""
    required_data.Date = pd.to_datetime(required_data['Date'])
    required_data.reset_index(inplace=True, drop=True)
    plt.plot(required_data['Close'], label='Closing Price')
    plt.ylabel(f'Closing Price of {company_symbol}')
    plt.xlabel(f'Date range from {start_date} to {end_date}')
    plt.title(f'Linear Trend line of {company_symbol}', loc='center')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    z = np.polyfit(required_data.index, required_data.Close, 1)
    p = np.poly1d(z)
    plb.plot(required_data.index, p(required_data.index), 'm--', label='Linear Trend Line')
    plb.legend(loc='upper left')
    plb.plot()
    required_data[['ds', 'y']] = required_data[['Date', 'Close']]
    df = required_data[['ds', 'y']]
    m = Prophet(weekly_seasonality=False, yearly_seasonality=False, daily_seasonality=False, n_changepoints=15)
    m.fit(df)
    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)
    m.plot_components(forecast)
    plt.show()

#Create a Subplot showing moving average graphs
#https://plot.ly/python/subplots/
#https://www.learndatasci.com/tutorials/python-finance-part-3-moving-average-trading-strategy/
def moving_average_all(dataframe, window, start_date, end_date, company_symbol):
    """Calculate Simple Moving Average"""
    """System takes above data and uses rolling property"""
    """of dataframe and use plotly over timeseries index to plot the graph"""
    subset_df = dataframe
    subset_df = subset_df.sort_values(by='Date')
    subset_df.drop_duplicates(inplace=True)
    subset_df.set_index('Date', inplace=True)
    window_df = subset_df.rolling(window=window).mean()
    roller = (window + 1)
    series = np.arange(1, roller)
    WMASeriesData = pd.DataFrame()
    WMASeriesData['Close'] = subset_df['Close'].rolling(window).apply(lambda close: np.dot(close, series)/series.sum(), raw=True)
    fig = make_subplots(rows=3, cols=1)
    #Plotting of Weighted Moving Averages
    fig.add_trace(go.Scatter(x=WMASeriesData.index, y=WMASeriesData['Close'], name="Weighted MA"), row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Weighted Moving Average", row=1, col=1)
    #Plotting of Linear Moving Averages
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df['Close'], name="Linear MA"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Linear Moving Average", row=2, col=1)
    #Plotting of Moving Averages Convergence and Divergence
    EWMA12 = subset_df.Close.ewm(span=12, adjust=False).mean()
    EWMA26 = subset_df.Close.ewm(span=26, adjust=False).mean()
    MAConvDiv = EWMA26-EWMA12
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=MAConvDiv, name="MACD"), row=3, col=1)
    fig.update_xaxes(title_text=f"Date range from {start_date} to {end_date}", row=3, col=1)
    fig.update_yaxes(title_text="MACD Values", row=3, col=1)
    #Title for Main plot
    fig.update_layout(title=f"Moving Averages of {company_symbol}", font=dict(size=12))
    #Plotting of subplots in one plot
    plot(fig, filename=f'{company_symbol}_moving_average_graph.html')

def future_date_validation(start_date):
    try:
        #Date should be in format of YYYY-MM-DD format
        datetime.datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        print("Incorrect Date Format")
        return '1'

def get_hypertuning_params(X_train_scaled, y_train_scaled):
    """This function finds the best value for hyper parameter that the predictive model use
    Hyper parameter changes according to dataset, and user is advised to use this approach for accurate
    predictions
    """
    model_2 = GradientBoostingRegressor()
    params_gradient = {'max_depth':[25, 50, 75, 100, 125, 150, 200, 250],
                       'max_leaf_nodes':[25, 50, 75, 100, 125, 150, 200, 250],
                       'n_estimators':[25, 50, 75, 100, 125, 150, 200, 250]}
    grid_gradient1 = GridSearchCV(estimator=model_2, param_grid=params_gradient, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
    grid_gradient1.fit(np.array(X_train_scaled).reshape(-1, 1), y_train_scaled.ravel())
    best_params1 = grid_gradient1.best_params_
    print('Best hyper parameter for gradient:', best_params1)
    model_3 = RandomForestRegressor()
    grid_gradient2 = GridSearchCV(estimator=model_3, param_grid=params_gradient, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
    grid_gradient2.fit(np.array(X_train_scaled).reshape(-1, 1), y_train_scaled.ravel())
    best_params2 = grid_gradient2.best_params_
    print('Best hyper parameter for random forest:', best_params2)
    return best_params1, best_params2

def predict_close_values(dataframe, input_date, option_selected, historical_records):
    """This function helps to find the closing price of a company for a date in future/Past
    The function works on 3 models : Linear, Gradient , RandomForest Regressor
    """
    date_valid_indicator = ''
    dataframe = dataframe.sort_values(by='Date')
    if input('Do you want to define test frame? Press \'Y\' for YES or any other key for NO: ').upper() == 'Y':
        start_date = input("Please enter Start Date in YYYY-MM-DD format for testing the model: ")
        end_date = input("Please enter End Date in YYYY-MM-DD format for testing the model: ")
        date_valid_indicator = date_validation(start_date, end_date)
       # In case of any validation errors:
        if not date_valid_indicator:
            start_date = pd.to_datetime(start_date, format='%Y-%m-%d').date()
            end_date = pd.to_datetime(end_date, format='%Y-%m-%d').date()
            required_data_to_test = historical_records[(historical_records['Date'] >= start_date) & (historical_records['Date'] <= end_date)]
            X_train = dataframe['Date']
            X_test = required_data_to_test['Date']
            y_train = dataframe['Close']
            y_test = required_data_to_test['Close']
    else:
        X_train, X_test, y_train, y_test = train_test_split(dataframe['Date'], dataframe['Close'], test_size=0.3)
        
    model_1 = LinearRegression()
    #https://stackoverflow.com/questions/44290635/python-how-to-convert-datetime-data-using-toordinal-considering-the-time
    #https://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
    #Since dates can't be used as training data directly as regressors accept only numerical data
    #hence conversion of timestamp to number of days was imminent.
    if not date_valid_indicator:
        X_train = X_train.apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
        X_train = X_train.apply(days_to_milliseconds)
        X_train = np.array(X_train).reshape(-1, 1)
        X_test = X_test.apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
        X_test = X_test.apply(days_to_milliseconds)
        X_test = np.array(X_test).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        y_train = np.array(y_train).reshape(-1, 1)
        scale = StandardScaler()
        X_train_scaled = scale.fit_transform(X_train)
        x_train_mean = scale.mean_
        x_train_std = scale.scale_
        y_train_scaled = scale.fit_transform(y_train)
        X_test_scaled = scale.fit_transform(X_test)
        y_test_scaled = scale.fit_transform(y_test)
        model_1.fit(np.array(X_train_scaled).reshape(-1, 1), np.array(y_train_scaled).reshape(-1, 1))
        y_pred = model_1.predict(X_test_scaled)
        if option_selected == '7':
            #CAUTION!!!! THIS WILL TAKE TOO MUCH TIME
            model1_params, model2_params = get_hypertuning_params(X_train_scaled, y_train_scaled)
            max_depth_gradient = model1_params.get('max_depth')
            max_leaf_gradient = model1_params.get('max_leaf_nodes')
            n_estimators_gradient = model1_params.get('n_estimators')
            max_depth_forest = model2_params.get('max_depth')
            max_leaf_forest = model2_params.get('max_leaf_nodes')
            n_estimators_forest = model2_params.get('n_estimators')
        elif option_selected == '6':
        #We are running the model on default parameters
            max_depth_gradient = 100
            max_leaf_gradient = 100
            n_estimators_gradient = 100
            max_depth_forest = 100
            max_leaf_forest = 100
            n_estimators_forest = 100
        model_2 = GradientBoostingRegressor(random_state=1, max_depth=max_depth_gradient, max_leaf_nodes=max_leaf_gradient, n_estimators=n_estimators_gradient)
        model_3 = RandomForestRegressor(random_state=1, n_estimators=n_estimators_forest, max_depth=max_depth_forest, max_leaf_nodes=max_leaf_forest)
    #   https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
    #   The idea to have multiple regressors was taken from Datacamp course as well as above link
        ereg = VotingRegressor(estimators=[('model_1', model_1), ('model_2', model_2), ('model_3', model_3)])
        ereg.fit(np.array(X_train_scaled).reshape(-1, 1), np.array(y_train_scaled).ravel())
        y_pred = ereg.predict(X_test_scaled)
        rms_2 = sqrt(mean_squared_error(y_pred, y_test_scaled.ravel()))
        r_squared = r2_score(np.array(y_test_scaled).ravel(), y_pred)
        print("R squared values of the model is : ", r_squared)
        print("Ensemble RMSE values is :", rms_2)
        user_date = pd.to_datetime(input_date, format='%Y-%m-%d')
        x_input = datetime.datetime.strftime(user_date, '%Y-%m-%d')
        x_input = days_to_milliseconds(x_input)
        #print(x_input)
        x_input_scaled = ((x_input - x_train_mean) / x_train_std)
        particular_date_pred = ereg.predict(np.array(x_input_scaled).reshape(-1, 1))
        print("Prediction for", input_date, ":", scale.inverse_transform(particular_date_pred))

###https://towardsdatascience.com/time-series-forecasting-with-prophet-54f2ac5e722e
def predict_n_days(required_data, start_date, end_date, company_symbol):
    number_of_days = input('How many days in the future you want to predict? : ')
    required_data[['ds', 'y']] = required_data[['Date', 'Close']]
    df = required_data[['ds', 'y']]
    m = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=True, n_changepoints=15)
    m.fit(df)
    future = m.make_future_dataframe(periods=int(number_of_days))
    forecast = m.predict(future)
    fig = plot_plotly(m, forecast, xlabel=f'Trained model from {start_date} to {end_date} and predicted {number_of_days} days in the future', ylabel=f'Closing price of {company_symbol} ', figsize=(1800,800))
    fig.update_layout(showlegend=True)
    plot(fig, filename=f'{company_symbol}_predict_{number_of_days}_days_graph.html')


def compare_two_companies(company_symbol, company_symbol_to_compare, start_date, end_date, required_data, required_data_to_compare):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=required_data['Date'],
                                 open=required_data['Open'],
                                 high=required_data['High'],
                                 low=required_data['Low'],
                                 close=required_data['Close'],
                                 name=f'{company_symbol}'))
    fig.add_trace(go.Candlestick(x=required_data_to_compare['Date'],
                                 open=required_data_to_compare['Open'],
                                 high=required_data_to_compare['High'],
                                 low=required_data_to_compare['Low'],
                                 close=required_data_to_compare['Close'],
                                 increasing={'line': {'color': '#7b3025'}}, decreasing={'line': {'color': '#e9b595'}},
                                 name=f'{company_symbol_to_compare}'))
    fig.update_layout(
        title=f'Compare share prices of {company_symbol} and {company_symbol_to_compare}',
        yaxis_title='Closing price',
        font=dict(size=18),
        xaxis=go.layout.XAxis(
            autorange=True,
            title=f'Date range from {start_date} to {end_date}',
            type="date"),
        yaxis=go.layout.YAxis(
            anchor="x",
            autorange=True,
            domain=[0, 1],
            linecolor="#673ab7",
            mirror=True,
            title='Closing Price',
            showline=True,
            side="left",
            tickfont={"color": "#000000"},
            tickmode="auto",
            ticks="",
            titlefont={"color": "#000000"},
            type="linear",
            zeroline=False),
        yaxis2=go.layout.YAxis(
            anchor="x",
            autorange=True,
            domain=[0, 1],
            linecolor="#E91E63",
            mirror=True,
            showline=True,
            side="left",
            tickfont={"color": "#000000"},
            tickmode="auto",
            ticks="",
            titlefont={"color": "#000000"},
            type="linear",
            zeroline=False))
    fig.update_layout(
        dragmode="zoom",
        hovermode="x",
        legend=dict(traceorder="normal"),
        height=800,
        template="plotly",
        margin=dict(
            t=100,
            b=100))
    plot(fig, filename=f'{company_symbol}_compared_with_{company_symbol_to_compare}_graph.html')


def company_search(user_input='', df=''):
    """This module accepts user input, finds the probable list of companies
    and return a dictionary of symbols and description"""
    dict_company = {}
    if user_input == '':
        user_input = input("Enter the approx company name to be searched for : ")
        regex_comp_symb = str('.*'+user_input+'.*').upper()
        if not df.empty:
            for index,row in df.iterrows():
                if reg.search(regex_comp_symb, str(row['Name']).upper()):
                    key = str(row['Symbol'].upper())
                    value = str(row['Name'])
                    dict_company.update({key:value})
    else:
        if not df.empty:
            for index,row in df.iterrows():
                if user_input.upper() == row['Symbol']:
                    key = str(row['Symbol'].upper())
                    value = str(row['Name'])
                    dict_company.update({key:value})

    return dict_company

def days_to_milliseconds(date):
    dtime = datetime.datetime.strptime(date, '%Y-%m-%d')
    return (dtime - EPOCH_TIME).total_seconds() * 1000.0
