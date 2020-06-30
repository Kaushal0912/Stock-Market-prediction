# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:31:04 2019

@author: chait
"""
import getpass as gp
import os
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import KBC_trading_functions as pk



#functions

def analyse_company(company_df):
    search_new_company = ''
    company_symbol = pk.get_company_name(company_df)
    if company_symbol:
        start_date = ''
        end_date = ''
        historical_records = pk.update_company_data(company_symbol)
        if not historical_records.empty:
            loop = True
            search_new_company = ''
            options = ["2", "3", "4", "5", "6", "7", "8", "9"]
            while loop:          ## While loop which will keep going until loop = False
                pk.print_menu(company_symbol)
                choice = input("Enter your choice [1-12]: ")
                if choice in options and not start_date and not end_date:
                    start_date = input("Please enter Start Date in YYYY-MM-DD format: ")
                    end_date = input("Please enter End Date in YYYY-MM-DD format: ")
                    date_valid_indicator = pk.date_validation(start_date, end_date)
                   # In case of any validation errors:
                    if not date_valid_indicator:
                       #Converting String inputs of dates to YYYY-MM-DD format
                        start_date = pd.to_datetime(start_date, format='%Y-%m-%d').date()
                        end_date = pd.to_datetime(end_date, format='%Y-%m-%d').date()
                        required_data = historical_records[(historical_records['Date'] >= start_date) &
                                                           (historical_records['Date'] <= end_date)]
                    else:
                       # In case of errors in dates, initialise and start the while loop again
                        start_date = ''
                        end_date = ''
                        continue
                if choice == "1":
                    pk.current_data(company_symbol)
                    input('Press ENTER to see the MENU...')
                elif choice == "2":
                    # Displaying Central tendency measures
                    print(required_data['Close'].describe())
                   # Displaying of Variance
                    print("Variance =", required_data.Close.var())
                   # Coefficient of variance
                    print("Coefficient of Variance =", required_data.Close.std() /
                          required_data.Close.mean() * 100)
                    norm_values = required_data.sort_values(by=['Date'])
                    norm_close = norm_values['Close'].iloc[0]
                    norm_values.set_index('Date')
                   # Calculating normalised values = (( Closing value - closing value on start date )  / closing value on start date )) * 100
                    norm_values['NormValues'] = required_data['Close'].apply(lambda x: (((x - norm_close)/norm_close)*100))
                    print("Normalised Prices: Please look at the graph")
                   #Plotting Normalised values
                    pk.norm_values_plot(norm_values, start_date, end_date, company_symbol)
                    input('Press ENTER to see the MENU...')
                    ## Call get reporting data functions here
                elif choice == "3":
                    # plot of Linear Raw time Series
                    pk.raw_time_series(required_data, company_symbol, start_date, end_date)
                elif choice == "4":
                    print("Linear Trend Line")
                    pk.linear_trend_line(required_data, start_date, end_date, company_symbol)
                elif choice == "5":
                    difference_in_dates = (end_date - start_date).days
                    try:
                        #period for Moving Averages graphs
                        period = int(input(f"Please enter window period in days less than {difference_in_dates}: "))
                    except ValueError:
                        print('Please enter only positive integer value for window')
                        input('Press ENTER to see the MENU...')
                        continue
                    #Validation of period
                    windowind = pk.period_validation(start_date, end_date, period)
                    if not windowind:
                        pk.moving_average_all(required_data, period, start_date, end_date, company_symbol)
                    else:
                        input('Press ENTER to see the MENU...')
                        continue
                elif choice == "6":
                    future_date = input("Enter date of prediction in YYYY-MM-DD format : ")
                    #Date Validation
                    date_valid_indicator = pk.future_date_validation(future_date)
                    #If no errors, then convert string input date to YYYY-MM-DD format
                    if not date_valid_indicator:
                        future_date = pd.to_datetime(future_date, format='%Y-%m-%d').date()
                        pk.predict_close_values(required_data, future_date, choice, historical_records)
                    else:
                        #If there is any error clear the value of future_Date and start while loop again
                        future_date = ''
                        continue
                elif choice == "7":
                    future_date = input("Enter date of prediction in YYYY-MM-DD format : ")
                    #Date Validation
                    date_valid_indicator = pk.future_date_validation(future_date)
                    #If no errors, then convert string input date to YYYY-MM-DD format
                    if not date_valid_indicator:
                        future_date = pd.to_datetime(future_date, format='%Y-%m-%d').date()
                        pk.predict_close_values(required_data, future_date, choice, historical_records)
                    else:
                        #If there is any error  clear the value of future_Date and start while loop again
                        future_date = ''
                        continue
                elif choice == "8":
                    pk.predict_n_days(required_data, start_date, end_date, company_symbol)
                elif choice == "9":
                    company_symbol_to_compare = pk.get_company_name(company_df)
                    if company_symbol_to_compare:
                        if company_symbol_to_compare != company_symbol:
                            historical_records_to_compare = pk.update_company_data(company_symbol_to_compare)
                            if not historical_records_to_compare.empty:
                                required_data_to_compare = historical_records_to_compare[(historical_records_to_compare['Date'] >= start_date) &
                                                                                         (historical_records_to_compare['Date'] <= end_date)]
                                pk.compare_two_companies(company_symbol, company_symbol_to_compare, start_date, end_date, required_data, required_data_to_compare)
                            else:
                                print(f"No data found for {company_symbol_to_compare}")
                                input('Press ENTER to see the MENU...')
                        else:
                            print('Cannot compare between 2 same compaines')
                            input('Press ENTER to see the MENU...')
                    ## You can add your code or functions here
                elif choice == "10":
                    start_date = input("Please enter Start Date in YYYY-MM-DD format: ")
                    end_date = input("Please enter End Date in YYYY-MM-DD format: ")
                    date_valid_indicator = pk.date_validation(start_date, end_date)
                    if not date_valid_indicator:
                        start_date = pd.to_datetime(start_date, format='%Y-%m-%d').date()
                        end_date = pd.to_datetime(end_date, format='%Y-%m-%d').date()
                        required_data = historical_records[(historical_records['Date'] >= start_date) &
                                                           (historical_records['Date'] <= end_date)]
                    else:
                        start_date = ''
                        end_date = ''
                        continue
                elif choice == "11":
                    ## Search new company
                    search_new_company = True # This will make the while loop to end as value of loop is set to False
                    loop = False
                elif choice == "12":
                    ## Exit
                    search_new_company = False
                    loop = False # This will make the while loop to end as not value of loop is set to False
                else:
                    # Any integer inputs other than values 1-5 we print an error message
                    print("Wrong option selection.")
        else:
            print(f"No data found for {company_symbol}")
            search_new_company = True
    return search_new_company



### User welcome message and start of the code
print("_" * 70)
print("Welcome "+gp.getuser()+" to KBC home page !! \n")
print("If you are asked a question,please check the options and then answer appropriately \n")
print("Enjoy the Analysis \n")
print("_" * 70)

File_name_company = (dt.date(dt.utcnow() - td(seconds=14400)))
FILE_TREE = [file for file in os.listdir('.') if os.path.isfile(file)]

if (str(File_name_company) + '.csv') not in FILE_TREE:
    COMPANY_NASDAQ_URL = 'http://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download'
    try:
        company_df = pk.access_csv_url(COMPANY_NASDAQ_URL)
    except FileNotFoundError:
        print('Download did not work, please check with network team if link is accessible')
    if not company_df.empty:
        try:
            company_df.to_csv(str(File_name_company) + '.csv')
        except IOError:
            print('File could not be created!!!, please check if you have write access')
else:
    company_df = pd.read_csv(str(File_name_company) + '.csv')

if company_df.empty is False:
    print("Data is loaded")
else:
    print("Data is not loaded,")

search_new_company = True

while search_new_company:
    search_new_company = analyse_company(company_df)

print("Thank you for using KBC trading... Have a great day!!")
