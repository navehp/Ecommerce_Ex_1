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
        brand_data = brand_data.sort_values('value', ascending=True)
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
        min_bundle, min_value = min(bundles, key=lambda x: x[1] if len(x[0]) == 5 else 0)
        optimal_bundle.extend(min_bundle)
        optimal_value += min_value

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
    #  extract the specific data for that type
    filtered_data = data[(data.brand == brand) & (data.year == year) & (data.engine_size == size)]
    return filtered_data.value.tolist()




class Type:
    cars_num = 0
    buyers_num = 0

    def __init__(self, brand, year, size, data):
        self.data = extract_data(brand, year, size, data)

    def avg_buy(self):
        # runs a procurement vcg auction for buying cars_num cars on the given self.data.
        # returns the average price paid for a winning car.
        next_car = sorted(self.data)[self.cars_num]
        return next_car

    def cdf(self, x):
        # return F(x) for the histogram self.data
        sorted_data = sorted(self.data)

        cars_count = 0
        previous_bid = 0
        cur_bid = 0
        for bid in sorted_data:
            cur_bid = bid
            if x <= bid:
                break
            cars_count += 1
            previous_bid = bid

        cdf = cars_count / len(sorted_data) + (x - cur_bid) / (cur_bid - previous_bid) * sorted_data.count(cur_bid) / len(
            sorted_data)

        return cdf

    def os_cdf(self, r, n, x):
        #The r out of n order statistic CDF

        cdf = 0
        x_cdf = self.cdf(x)

        for j in range(r, n + 1):
            cdf += math.comb(n, j) * x_cdf ** j * (1 - x_cdf) ** (n-j)

        return cdf

    def exp_rev(self):
        # returns the expected revenue in future auction for cars_num items and buyers_num buyers

        order_statistics_expected_values = []

        for r in range(1, self.buyers_num + 2):
            expected_value = 0
            x = 0
            while True:
                os_cdf = self.os_cdf(r, self.cars_num, x)
                expected_value += 1 - os_cdf
                if os_cdf == 1:
                    break
            order_statistics_expected_values.append(expected_value)

        expected_revenue = 0
        for r in range(1, self.buyers_num + 1):
            expected_revenue += r * (order_statistics_expected_values[r] - order_statistics_expected_values[r - 1])
        return expected_revenue

    def exp_rev_median(self, n):
        reserve price
        return 0

    ########## Part C ###############

    def reserve_price(self):
        # returns your suggestion for a reserve price based on the self_data histogram.
        return 0

