import pandas as pd
import numpy as np


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


# Intiialialize the product below from the user input
def add_product():
    # Product()
    pass



class Product:
    def __init__(self, name, price, cost, expiry_days,
                 demand_upper_bound_frac, demand_likely_frac, demand_lower_bound_frac,
                 storage_capacity, days_to_simulate=28):  #, storage_cost_per_day, restock_option):
        # input parameters
        self.name = name
        self.price = price
        self.cost = cost
        self.expiry_days = expiry_days
        self.demand_upper_bound = demand_upper_bound_frac
        self.demand_likely = demand_likely_frac
        self.demand_lower_bound = demand_lower_bound_frac
        self.storage_capacity = storage_capacity
        self.days_to_simulate = days_to_simulate

    def build_inventory(self):
        inventory_series = pd.Series([self.expiry_days]*self.days_to_simulate
        return inventory_series

    def simulate_demand(self):
        # mean_demand_bound = round( ((self.demand_upper_bound + self.demand_lower_bound)/2),2)
        sim_daily_demand = (mod_pert_random(self.demand_lower_bound * self.storage_capacity,
                                            self.demand_likely * self.storage_capacity,
                                            self.demand_upper_bound * self.storage_capacity,
                                            samples=self.days_to_simulate))

        # self._daily_demand = daily_demand
        return sim_daily_demand

    def calc_weekly_defective(self, n_items):
        percent_defective = np.random.choice(list(range(5, 11)))  # Selects a random % value between 5 to 10
        defective_items = (n_items * percent_defective) // 100  # Number of defective items during restock (5-10%)
        return defective_items

    def financials(self, sold, missed, weekly_wastage):
        # Calculates total loss, missed profit and total profit for both items

        total_loss = (self.cost * weekly_wastage)
        missed_profit = missed * (self.price - self.cost)
        sold_profit = sold * (self.price - self.cost)

        return [total_loss, missed_profit, sold_profit]

    def restock_quantity(self, scenario, inventory_series, weekly_demand):
        n_items = len(inventory_series)
        if scenario == 1:  # Check is scenario 1
            items_to_restock = self.storage_capacity - n_items  # Restock upto its maximum storage
            return items_to_restock

        if scenario != 1:  # Check if scenario 2 or 3
            items_to_restock = int(weekly_demand)  # Restock number of items equal to previous cumulative demand

            if items_to_restock > (self.storage_capacity - n_items):  # if previous cumulative demand greater than the vacant space
                items_to_restock = self.storage_capacity - n_items  # then items to restock equal to vacant space i.e. upto maximum storage

            return items_to_restock













def update_inventory(product, scenario):
    """
    Updates the inventory on the basis of demand.
    Drops sold and expired items each day and at the end of each week, calls the restocking function to restock the inventory.
    :param product: Dataframe of inventory with all items
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
    demand = product.simulate_demand()
    product_inventory = product.build_inventory()

    weekly_wastage = []
    weekly_loss = []
    weekly_missed_profit = []
    weekly_sold_profit = []
    previous_demand_list = []

    # for df, k in zip(df_list, demand):  # iterating over dataframe and demand of an item

    week = 1
    day = 1
    weekly_expired_items = 0
    missed = 0
    item_demand_before_expiry = 0
    sold = 0

    for i in demand:  # demand for each day

        if day <= product.expiry_days:
            item_demand_before_expiry += i          # add items' demand before it gets expired

        if i <= len(product_inventory):                        # check if demand is less than the available stock
            sold += i                               # add sold items

            product_inventory.drop(product_inventory.index[:i], axis=0, inplace=True)  # sold items hence drop them from inventory dataframe
            product_inventory.reset_index(inplace=True, drop=True)

        else:
            missed += i                             # add demand to missed orders because of not enough stock to fulfill it

        product_inventory = product_inventory - 1  # end of day hence reduce expiry days remaining by 1

        weekly_expired_items += product_inventory[product_inventory < 0].count()  # check expired items and store them

        product_inventory.drop(product_inventory[product_inventory < 0].index, inplace=True)         # Throw expired items i.e. drop those from the dataframe
        product_inventory.reset_index(inplace=True, drop=True)

        if (day == 7 and scenario != 3) or (product_inventory.empty is True and scenario == 3):       # Check if end of week for scenario 1 and 2 or if no stock in inventory for scenario 3
            previous_demand_list.append(item_demand_before_expiry)
            items_to_restock = product.restock_quantity(scenario, product_inventory, mean(previous_demand_list))            # Call restocking function

            new_stock = pd.Series(list([product.expiry_days] * items_to_restock))             # Add rows for restocked items
            product_inventory = product_inventory.append(new_stock, ignore_index=True)

            weekly_defective = product.calc_weekly_defective(items_to_restock)                  # check for defective products in restocked items
            weekly_wastage = weekly_defective + weekly_expired_items        # Total waste products are expired and defective
            weekly_financials = product.financials(sold, missed, weekly_wastage)         # Call financials to calculate total loss, total profit and missed orders

            # append all calculations to their respective dictionaries
            weekly_wastage.append(weekly_wastage)
            weekly_loss.append(weekly_financials[0])
            weekly_missed_profit.append(weekly_financials[1])
            weekly_sold_profit.append(weekly_financials[2])

            day = 0
            weekly_expired_items = 0
            week += 1
            item_demand_before_expiry = 0
            missed = 0

        day = day + 1

    return [weekly_loss, weekly_missed_profit, weekly_sold_profit]

