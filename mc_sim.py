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


# Intiialialize the product below
def add_product():
    # print("Add the Product Specifics")
    product_name = input("Add Name")
    product_price = float(input("Add Product's Price"))
    product_cost = float(input("Add Product's Cost"))
    product_expiry_days = int(input("Add Product's Days before Expiry"))
    product_storage_capacity = int(input("Add Product's Storage Capacity"))
    product_demand_low = float(input("Add Product's Daily Demand's Lower Bound Fraction"))
    product_demand_likely = float(input("Add Product's Daily Demand's Likely Fraction"))
    product_demand_high = float(input("Add Product's Daily Demand's Upper Bound Fraction"))
    product_days_to_simulate = int(input("Add the number of days to simulate"))

    return Product(product_name, product_price, product_cost, product_expiry_days,
                   product_demand_high, product_demand_likely, product_demand_low,
                   product_storage_capacity, product_days_to_simulate)




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
        inventory_series = pd.Series([self.expiry_days]*self.days_to_simulate)
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













def update_inventory(product_list, scenario):
    """
    Updates the inventory on the basis of demand.
    Drops sold and expired items each day and at the end of each week, calls the restocking function to restock the inventory
    :param product_list: Dataframe of inventory with all items
    :param scenario: type of scenario (1 or 2)
    :return: list of Dictionaries for loss and missed profit. Each dictionary again contains a list for each item.
    >>> df1 = pd.DataFrame({'A':[3]*10,'B':[2]*10})
    >>> expiry_days = {'A':3, 'B':8}
    >>> update_inventory(df1, expiry_days, 10, 1) # doctest: +ELLIPSIS
    [{'A': ...
    >>> update_inventory(df1,expiry_days,10,2) # doctest: +ELLIPSIS
    [{'A': ...
    """


    wastage_dict = {product.name: [] for product in product_list}
    loss_dict = {product.name: [] for product in product_list}
    missed_profit_dict = {product.name: [] for product in product_list}
    sold_profit_dict = {product.name: [] for product in product_list}
    previous_demand_list = []

    # for df, k in zip(df_list, demand):  # iterating over dataframe and demand of an item
    for product in product_list:

        demand = product.simulate_demand()
        product_inventory = product.build_inventory()

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
                items_to_restock = product.restock_quantity(scenario, product_inventory, np.mean(previous_demand_list))            # Call restocking function

                new_stock = pd.Series(list([product.expiry_days] * items_to_restock))             # Add rows for restocked items
                product_inventory = product_inventory.append(new_stock, ignore_index=True)

                weekly_defective = product.calc_weekly_defective(items_to_restock)                  # check for defective products in restocked items
                weekly_wastage = weekly_defective + weekly_expired_items        # Total waste products are expired and defective
                weekly_financials = product.financials(sold, missed, weekly_wastage)         # Call financials to calculate total loss, total profit and missed orders

                # append all calculations to their respective dictionaries
                wastage_dict[product.name].append(weekly_wastage)
                loss_dict[product.name].append(weekly_financials[0])
                missed_profit_dict[product.name].append(weekly_financials[1])
                sold_profit_dict[product.name].append(weekly_financials[2])

                day = 0
                weekly_expired_items = 0
                week += 1
                item_demand_before_expiry = 0
                missed = 0

            day = day + 1

    return [loss_dict, missed_profit_dict, sold_profit_dict]






















def mc_simulation():
    """
    Runs the program multiple times as specified by the user in number of simulations.
    Also plots graphs to represent the aggregate statistics after all simulations.
    :return: None
    """
    print("Add the first product specifics")
    products = list()
    products.append(add_product())
    while True:
        add_new = int(input("Type 1 to add another product or 0 to proceed to simulation"))
        if add_new == 0:
            break
        print("Add the product specifics")
        products.append(add_product)

    loss_simulation_dict = {1: {product.name: [] for product in products},
                            2: {product.name: [] for product in products},
                            3: {product.name: [] for product in products}}

    missed_profit_simulation_dict = {1: {product.name: [] for product in products},
                                     2: {product.name: [] for product in products},
                                     3: {product.name: [] for product in products}}

    sold_profit_simulation_dict = {1: {product.name: [] for product in products},
                                   2: {product.name: [] for product in products},
                                   3: {product.name: [] for product in products}}

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
                # i, e, s = initial_stock()           # Initialize stock
                u1 = update_inventory(products, how_to_restock)          # Update inventory i.e. sell, check expired
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




mc_simulation()