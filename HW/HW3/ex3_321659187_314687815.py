import pandas as pd
import math
from itertools import permutations

FORD = 'ford'
BMW = 'bmw'
KIA = 'kia'
VOLKSWAGEN = 'vw'
FERRARI = 'ferrari'

B = [FORD, BMW, KIA, VOLKSWAGEN, FERRARI]


########## Part A ###############


def generate_permutations(years):
    mappings = []
    for p in permutations(B):
        mapping = {car_type: year for car_type, year in zip(p, years)}
        mappings.append(mapping)
    return mappings


def find_single_bundle(data, mapping, bundle):
    iteration_bundle = []
    value = 0

    for brand in B:
        brand_data = data[(data.brand == brand) & (data.year == mapping[brand])]
        brand_data = brand_data.sort_values('value', ascending=False)
        for _, row in brand_data.iterrows():
            if row.id not in bundle:
                iteration_bundle.append(row.id)
                value += row.value
                break
    return iteration_bundle, value


def opt_bnd(data, k, years):
    # returns the optimal bundle of cars for that k and list of years and their total value.
    optimal_bundle = []
    optimal_value = 0

    mappings = generate_permutations(years)
    for i in range(k):
        bundles = [find_single_bundle(data, mapping, optimal_bundle) for mapping in mappings]
        max_bundle, max_value = max(bundles, key=lambda x: x[1] if len(x[0]) == 5 else 0)
        optimal_bundle.extend(max_bundle)
        optimal_value += max_value

    return {"cost": optimal_value, "bundle": optimal_bundle}


def calculate_payment(data, k, years, optimal_value, id):
    # sw stands for social welfare
    sw_with_id = optimal_value - data[data.id == id].value.iloc[0]
    data_without_id = data[data.id != id]
    sw_without_id = opt_bnd(data_without_id, k, years)['cost']
    return sw_with_id - sw_without_id


def comb_vcg(data, k, years):
    # runs the VCG procurement auction
    payments = {}

    optimal_bundle_dict = opt_bnd(data, k, years)
    optimal_bundle = optimal_bundle_dict['bundle']
    optimal_value = optimal_bundle_dict['cost']
    for id in optimal_bundle:
        print(id)
        payment = calculate_payment(data, k, years, optimal_value, id)
        payments[id] = payment
    return payments


########## Part B ###############
def extract_data(brand, year, size, data):
    #extract the specific data for that type
    return []




class Type:
    cars_num = 0
    buyers_num = 0

    def __init__(self, brand, year, size, data):
        self.data = extract_data(brand, year, size, data)

    def avg_buy(self):
        # runs a procurement vcg auction for buying cars_num cars on the given self.data.
        # returns the average price paid for a winning car.
        return 0

    def cdf(self, x):
        # return F(x) for the histogram self.data
        return 1


    def os_cdf(self, r, n, x):
        #The r out of n order statistic CDF
        return 1

    def exp_rev(self):
        # returns the expected revenue in future auction for cars_num items and buyers_num buyers

        return 0

    def exp_rev_median(self, n):
        
        return 0

    ########## Part C ###############

    def reserve_price(self):
        # returns your suggestion for a reserve price based on the self_data histogram.
        return 0

