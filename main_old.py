"""
Perishable Products Inventory Management using Monte Carlo Simulation
Instructor: John Weible, jweible@illinois.edu

TODO:
- change x axis of plots
- change simlaution to monthly?
- improve code
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean


def mod_pert_random(low, likely, high, confidence=4, samples=1):
    """Produce random numbers according to the 'Modified PERT' distribution.
    :param low: The lowest value expected as possible.
    :param likely: The 'most likely' value, statistically, the mode.
    :param high: The highest value expected as possible.
    :param confidence: This is typically called 'lambda' in literature
                        about the Modified PERT distribution. The value
                        4 here matches the standard PERT curve. Higher
                        values indicate higher confidence in the mode.
                        Currently allows values 1-18
    :param samples: Number of samples
    I got this function from Professor Weible's class examples :
    https://github.com/iSchool-597PR/Examples_2021Fall/blob/main/week_07/Probability_Distributions.ipynb
    """
    # Check for reasonable confidence levels to allow:
    if confidence < 1 or confidence > 18:
        raise ValueError('confidence value must be in range 1-18.')

    mean1 = (low + confidence * likely + high) / (confidence + 2)

    a = (mean1 - low) / (high - low) * (confidence + 2)
    b = ((confidence + 1) * high - low - confidence * likely) / (high - low)

    beta = np.random.beta(a, b, samples)
    beta = beta * (high - low) + low
    beta = [int(x) for x in beta]
    return beta


def initial_stock():
    """
    Generates inventory dataframe, storage capacity for each item and initializes expiry days for all items
    :return: Inventory Dataframe, Expiry dictionary and storage
    >>> initial_stock()
    (    A  B
    0   3  5
    1   3  5
    2   3  5
    3   3  5
    4   3  5
    .. .. ..
    95  3  5
    96  3  5
    97  3  5
    98  3  5
    99  3  5
    <BLANKLINE>
    [100 rows x 2 columns], {'A': 3, 'B': 5}, 100)
    """
    expiry_days = {'A': 3, 'B': 5}        # Initializes expiry days for both the products
    storage = 100                         # Maximum storage capacity for each item
    df_stock = pd.DataFrame({'A': [expiry_days['A']] * storage, 'B': [expiry_days['B']] * storage})  # Inventory Dataframe
    return df_stock, expiry_days, storage


def demand_items(storage):
    """
    Generates daily demand using PERT for each item and stores it in a list corresponding to each item key in dictionary.
    :param storage: maximum storage capacity for each item
    :return: a dictionary of lists of daily demand for each item
    """
    days_to_run = 28           # Program checks Inventory for a month i.e. 28days
    stock_demand = {'A': mod_pert_random(0.15 * storage, 0.18 * storage, 0.20 * storage, samples=days_to_run),             # Generates random demand for 28 days for both items
                    'B': mod_pert_random(0.10 * storage, 0.12 * storage, 0.15 * storage, samples=days_to_run)}
    return stock_demand


def defective(items):
    """
    Generates a random integer which represents the number of defective items from the total restocked items.
    :param items: total number of items to be restocked, out of which we check the defective items
    :return: an integer number of defective items
    """
    percent_of_defective = np.random.choice(list(range(5, 11)))                # Selects a random % value between 5 to 10
    defective_items = (items * percent_of_defective) // 100                     # Number of defective items during restock (5-10%)
    return defective_items


def item_df(df):
    """
    Makes a list of dataframes for each item
    :param df: Whole dataframe with each column as each item and number of rows represent the number of items in stock
    :return: list of dataframes for each item
    >>> df1 = pd.DataFrame({'A': [2] * 10, 'B': [5] * 10})
    >>> item_df(df1)
    [   A
    0  2
    1  2
    2  2
    3  2
    4  2
    5  2
    6  2
    7  2
    8  2
    9  2,    B
    0  5
    1  5
    2  5
    3  5
    4  5
    5  5
    6  5
    7  5
    8  5
    9  5]
    """
    l1 = list(df.columns)                     # List of items
    df_list = []
    for i in l1:
        df_list.append(df[[i]].copy())         # Generates a list of individual Dataframes of items
    return df_list


def financials(sold, missed, weekly_wastage, key):
    """
    Calculates total loss, total missed profit and total profit for each item
    :param sold: Number of sold items
    :param missed: Number of missed orders due to no stock
    :param weekly_wastage: Number of waste items because they were defective or expired
    :param key: Name of item (A or B)
    :return: List of loss and missed profit and total profit
    >>> financials(5,10,7,'A')
    [140, 40, 20]
    >>> financials(3,8,4,'B')
    [60, 24, 9]
    """
    # Initialized cost and profit for both items
    cost_a = 20
    cost_b = 15
    profit_a = 4
    profit_b = 3

    # Calculates total loss, missed profit and total profit for both items
    if key == 'A':
        total_loss = (cost_a * weekly_wastage)
        missed_profit = (missed * profit_a)
        sold_profit = (sold * profit_a)
    else:
        total_loss = (cost_b * weekly_wastage)
        missed_profit = (missed * profit_b)
        sold_profit = (sold * profit_b)

    return [total_loss, missed_profit, sold_profit]


def restocking(scenario, storage, df, weekly_demand):
    """
    Generates the total number of items to be restocked in the inventory on the basis of type of scenario
    :param scenario: type of scenario (1 or 2)
    :param storage: storage capacity for each item
    :param df: Dataframe of item (current stock)
    :param weekly_demand: Previous week's demand (integer)
    :return: an integer number of the items to be restocked
    >>> df_A = pd.DataFrame([3]*10)
    >>> restocking(1,20,df_A,5)
    10
    >>> df_B = pd.DataFrame([2]*10)
    >>> restocking(2,20,df_B,5)
    5
    """
    if scenario == 1:                # Check is scenario 1
        items_to_restock = storage - df.shape[0]          # Restock upto its maximum storage
        return items_to_restock

    if scenario != 1:                # Check if scenario 2 or 3
        items_to_restock = int(weekly_demand)             # Restock number of items equal to previous cumulative demand

        if items_to_restock > (storage - df.shape[0]):          # if previous cumulative demand greater than the vacant space
            items_to_restock = storage - df.shape[0]            # then items to restock equal to vacant space i.e. upto maximum storage

        return items_to_restock


def update_inventory(a, expiry, storage, scenario):
    """
    Updates the inventory on the basis of demand.
    Drops sold and expired items each day and at the end of each week, calls the restocking function to restock the inventory.
    :param a: Dataframe of inventory with all items
    :param expiry: Dictionary of expiry days for all items
    :param storage: storage capacity for each item
    :param scenario: type of scenario (1 or 2)
    :return: list of Dictionaries for loss and missed profit. Each dictionary again contains a list for each item.
    >>> df1 = pd.DataFrame({'A':[3]*10,'B':[2]*10})
    >>> expiry_days = {'A':3, 'B':8}
    >>> update_inventory(df1, expiry_days, 10, 1) # doctest: +ELLIPSIS
    [{'A': ...
    >>> update_inventory(df1,expiry_days,10,2) # doctest: +ELLIPSIS
    [{'A': ...
    """
    demand = demand_items(storage)
    df_list = item_df(a)
    wastage_dict = {'A': [], 'B': []}
    loss_dict = {'A': [], 'B': []}
    missed_profit_dict = {'A': [], 'B': []}
    sold_profit_dict = {'A': [], 'B': []}
    previous_demand_list = []

    for df, k in zip(df_list, demand):  # iterating over dataframe and demand of an item
        week = 1
        day = 1
        weekly_expired_items = 0
        missed = 0
        item_demand_before_expiry = 0
        sold = 0

        for i in demand[k]:  # demand for each day

            if day <= expiry[k]:
                item_demand_before_expiry += i          # add items' demand before it gets expired

            if i <= df.shape[0]:                        # check if demand is less than the available stock
                sold += i                               # add sold items
                df.drop(df.index[:i], axis=0, inplace=True)  # sold items hence drop them from inventory dataframe
                df.reset_index(inplace=True, drop=True)

            else:
                missed += i                             # add demand to missed orders because of not enough stock to fulfill it

            df[k] = df[k] - 1  # end of day hence reduce expiry days remaining by 1

            weekly_expired_items += df[df[k] < 0].count()[0]  # check expired items and store them

            df.drop(df[df[k] < 0].index, inplace=True)         # Throw expired items i.e. drop those from the dataframe
            df.reset_index(inplace=True, drop=True)

            if (day == 7 and scenario != 3) or (df.empty is True and scenario == 3):       # Check if end of week for scenario 1 and 2 or if no stock in inventory for scenario 3
                previous_demand_list.append(item_demand_before_expiry)
                items_to_restock = restocking(scenario, storage, df, mean(previous_demand_list))            # Call restocking function

                df2 = pd.DataFrame(list([expiry[k]] * items_to_restock), columns=list(k))             # Add rows for restocked items
                df = df.append(df2, ignore_index=True)

                weekly_defective = defective(items_to_restock)                  # check for defective products in restocked items
                weekly_wastage = weekly_defective + weekly_expired_items        # Total waste products are expired and defective
                weekly_financials = financials(sold, missed, weekly_wastage, k)         # Call financials to calculate total loss, total profit and missed orders

                # append all calculations to their respective dictionaries
                wastage_dict[k].append(weekly_wastage)
                loss_dict[k].append(weekly_financials[0])
                missed_profit_dict[k].append(weekly_financials[1])
                sold_profit_dict[k].append(weekly_financials[2])

                day = 0
                weekly_expired_items = 0
                week += 1
                item_demand_before_expiry = 0
                missed = 0

            day = day + 1

    return [loss_dict, missed_profit_dict, sold_profit_dict]


def cumulative_avg(l1, new):
    """
    Calculates cumulative average for the given list and the new element
    :param l1: List of elements
    :param new: new element
    :return: cumulative average of the list elements and the new element
    >>> list1 = [1,2,3,4]
    >>> cumulative_avg(list1,5)
    3.0
    """
    temp = sum(l1)
    length = len(l1)
    cum_avg = (temp + new) / (length + 1)           # calculates cumulative average of all elements in list and the new element
    return cum_avg


def mc_simulation():
    """
    Runs the program multiple times as specified by the user in number of simulations.
    Also plots graphs to represent the aggregate statistics after all simulations.
    :return: None
    """
    loss_simulation_dict = {1: {'A': [], 'B': []}, 2: {'A': [], 'B': []}, 3: {'A': [], 'B': []}}
    missed_profit_simulation_dict = {1: {'A': [], 'B': []}, 2: {'A': [], 'B': []}, 3: {'A': [], 'B': []}}
    sold_profit_simulation_dict = {1: {'A': [], 'B': []}, 2: {'A': [], 'B': []}, 3: {'A': [], 'B': []}}
    simulations = None
    how_to_restock = None

    while True:
        try:
            simulations = int(input('Enter number of simulations\n'))           # takes number of simulations as user input
            if simulations < 2:                                                 # We need at least 2 values for a line plot
                print('Enter a number greater than 1\n')
                continue
        except ValueError:
            print('Enter a valid simulation\n')
            continue
        else:
            break

    flag = 0
    while True:

        while True:
            try:
                how_to_restock = int(input("1. Press '1' to restock weekly to full capacity\n2. Press '2' to restock weekly based on demand\n3. Press '3' to restock dynamically based on demand\n4. Press '4' for comparison\n5. Press '5' to exit\n"))
                if how_to_restock not in [1, 2, 3, 4, 5]:           # Available options
                    raise ValueError
            except ValueError:
                print('Enter a valid choice\n')
                continue
            else:
                break

        if how_to_restock == 5:
            break

        if how_to_restock < 4:
            flag += 1
            for j in range(simulations):            # Run whole program for the number of times user asked
                i, e, s = initial_stock()           # Initialize stock
                u1 = update_inventory(i, e, s, how_to_restock)          # Update inventory i.e. sell, check expired
                # updates dictionaries of missed profit, sold items and total loss for each simulation
                for k in loss_simulation_dict[how_to_restock]:
                    sold_profit_simulation_dict[how_to_restock][k].append(sum(u1[2][k]))

                    if len(loss_simulation_dict[how_to_restock][k]) == 0:
                        loss_simulation_dict[how_to_restock][k].append(sum(u1[0][k]))           # loss value for first simulation
                        missed_profit_simulation_dict[how_to_restock][k].append(sum(u1[1][k]))      # missed profit value for first simulation
                    else:
                        loss_cum_avg = cumulative_avg(loss_simulation_dict[how_to_restock][k], sum(u1[0][k]))
                        loss_simulation_dict[how_to_restock][k].append(loss_cum_avg)        # stores loss statistics for each simulation

                        missed_cum_avg = cumulative_avg(missed_profit_simulation_dict[how_to_restock][k], sum(u1[1][k]))
                        missed_profit_simulation_dict[how_to_restock][k].append(missed_cum_avg)        # stores missed profit statistics for each simulation

            # plot aggregate statistics for after all simulations
            for k in loss_simulation_dict[how_to_restock]:
                i, e, s = initial_stock()
                lb = str(k) + '(' + str(e[k]) + ' days expiry)'
                plt.figure(1, figsize=(8, 5))
                plt.tight_layout(pad=2)
                st = 'Scenario ' + str(how_to_restock)
                plt.suptitle(st)

                plt.subplot(121)
                plt.title('Loss (Monthly)')
                plt.xlabel('Number of simulations')
                plt.ylabel('Loss (in $)')
                plt.plot(loss_simulation_dict[how_to_restock][k], label=lb)
                plt.legend()

                plt.subplot(122)
                plt.title('Missed Profits (Monthly)')
                plt.xlabel('Number of simulations')
                plt.ylabel('Profit (in $)')
                plt.plot(missed_profit_simulation_dict[how_to_restock][k], label=lb)
                plt.legend()

            plt.show()

        if how_to_restock == 4:
            if flag != 3:
                print('Please run all simulations first\n')
            else:
                for k in loss_simulation_dict:
                    for j in loss_simulation_dict[k]:
                        l2 = 'Scenario_' + str(k) + '_Item_' + j
                        plt.figure(2, figsize=(8, 5))
                        plt.tight_layout(pad=2)
                        st = 'Scenario 1    vs    Scenario 2    vs    Scenario 3'
                        plt.suptitle(st)

                        plt.subplot(121)
                        plt.title('Loss (Monthly)')
                        plt.xlabel('Number of simulations')
                        plt.ylabel('Loss (in $)')
                        plt.plot(loss_simulation_dict[k][j], label=l2)
                        plt.legend()

                        plt.subplot(122)
                        plt.title('Missed Profits (Monthly)')
                        plt.xlabel('Number of simulations')
                        plt.ylabel('Profit (in $)')
                        plt.plot(missed_profit_simulation_dict[k][j], label=l2)
                        plt.legend()

                plt.figure(3, figsize=(8, 5))
                plt.title('Average Monthly Profit (Including all items)')
                plt.ylabel('Profit (in $)')
                s1_avg = mean(sold_profit_simulation_dict[1]['A']) + mean(sold_profit_simulation_dict[1]['B'])      # average profits for all simulations
                s2_avg = mean(sold_profit_simulation_dict[2]['A']) + mean(sold_profit_simulation_dict[2]['B'])
                s3_avg = mean(sold_profit_simulation_dict[3]['A']) + mean(sold_profit_simulation_dict[3]['B'])
                plt.bar(['Scenario 1', 'Scenario 2', 'Scenario 3'], [s1_avg, s2_avg, s3_avg])
                plt.show()


if __name__ == '__main__':
    mc_simulation()          # runs simulations


