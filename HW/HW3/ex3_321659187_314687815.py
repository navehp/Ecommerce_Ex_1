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


def proc_vcg(data, k, years):  # TODO check if we need to call this proc_vcg
    # runs the VCG procurement auction
    payments = {}

    optimal_bundle_dict = opt_bnd(data, k, years)
    optimal_bundle = optimal_bundle_dict['bundle']
    optimal_value = optimal_bundle_dict['cost']
    for id in optimal_bundle:
        payment = calculate_payment(data, k, years, optimal_value, id)
        payments[id] = payment
    return payments


########## Part B ###############
def extract_data(brand, year, size, data):
    #  extract the specific data for that type
    if isinstance(data, list):
        return data
    filtered_data = data[(data.brand == brand) & (data.year == year) & (data.engine_size == size)]
    return filtered_data.value.tolist()




class Type:
    cars_num = 0
    buyers_num = 0

    def __init__(self, brand, year, size, data):
        self.brand = brand
        self.year = year
        self.size = size
        self.data = extract_data(brand, year, size, data)

    def avg_buy(self):
        # runs a procurement vcg auction for buying cars_num cars on the given self.data.
        # returns the average price paid for a winning car.
        next_car = sorted(self.data)[self.cars_num]
        return next_car

    def cdf(self, x):
        # return F(x) for the histogram self.data
        sorted_data = sorted(self.data)

        if x < sorted_data[0]:
            return 0.0
        if sorted_data[-1] <= x:
            return 1.0

        cars_count = 0
        previous_bid = 0
        cur_bid = 0
        for bid in sorted_data:
            cur_bid = bid
            if x < bid:
                break
            cars_count += 1
            previous_bid = bid

        cdf = cars_count / len(sorted_data) + (x - previous_bid) / (cur_bid - previous_bid) * sorted_data.count(cur_bid) / len(
            sorted_data)

        return cdf

    def os_cdf(self, r, n, x):
        #The r out of n order statistic CDF

        cdf = 0
        x_cdf = self.cdf(x)

        for j in range(r, n + 1):
            cdf += self.comb(n, j) * x_cdf ** j * (1 - x_cdf) ** (n-j)

        return cdf


    def exp_rev(self):
        # returns the expected revenue in future auction for cars_num items and buyers_num buyers
        r = self.buyers_num - self.cars_num
        n = self.buyers_num
        expected_revenue = self.order_statistic_expected_value(r, n) * self.cars_num
        return expected_revenue

    def order_statistic_expected_value(self, r, n):
        expected_value = 0
        x = 0

        if r < 0:
            return 0

        while True:
            os_cdf = self.os_cdf(r, n, x)
            expected_value += 1 - os_cdf
            if os_cdf == 1:
                break
            else:
                x += 1
        return expected_value

    def exp_rev_median(self, n):
        reserve_price = self.median(self.data)
        reserve_price_cdf = self.cdf(reserve_price)

        self.buyers_num = n

        median_data = [i for i in self.data if i >= reserve_price]
        median_type = Type(self.brand, self.year, self.size, median_data)

        result = 0

        if n == 2:
            # both above reserve
            result += (1 - reserve_price_cdf) ** 2 * median_type.order_statistic_expected_value(1, 2)
            # one above reserve, another below
            result += 2 * reserve_price_cdf * (1 - reserve_price_cdf) * reserve_price

        elif n == 3:
            # all above reserve
            result += (1 - reserve_price_cdf) ** 3 * median_type.order_statistic_expected_value(2, 3)
            # one below reserve
            result += 3 * (1 - reserve_price_cdf) ** 2 * reserve_price_cdf * median_type.order_statistic_expected_value(1, 2)
            # two below reserve
            result += 3 * (1 - reserve_price_cdf) * reserve_price_cdf ** 2 * reserve_price
        else:
            raise Exception("Aval... Aval... ")
        return result

    ########## Part C ###############

    def reserve_price_expected_revenue(self, reserve_price):
        reserve_price_cdf = self.cdf(reserve_price)
        reserve_price_data = [i for i in self.data if i >= reserve_price]
        reserve_price_type = Type(self.brand, self.year, self.size, reserve_price_data)

        expected_revenue = 0
        for k in range(1, self.buyers_num + 1):
            prob = self.comb(self.buyers_num, k) * ((1 - reserve_price_cdf) ** k) * (reserve_price_cdf ** (self.buyers_num - k))
            if k <= self.cars_num:
                expected_revenue += prob * k * reserve_price
            else:
                expected_revenue += prob * self.cars_num * \
                                    reserve_price_type.order_statistic_expected_value(r=k - self.cars_num, n=k)

        return expected_revenue

    def reserve_price(self):
        # returns your suggestion for a reserve price based on the self_data histogram.
        min_bound = int(self.order_statistic_expected_value(r=self.buyers_num - self.cars_num, n=self.buyers_num))
        max_bound = int(self.order_statistic_expected_value(r=self.buyers_num, n=self.buyers_num))
        print(min_bound, max_bound)

        best_reserve_price = 0
        best_revenue = float('-inf')

        for reserve_price in range(min_bound, max_bound, 100):
            print(reserve_price)
            expected_revenue = self.reserve_price_expected_revenue(reserve_price)
            if expected_revenue > best_revenue:
                best_revenue = expected_revenue
                best_reserve_price = reserve_price

        return best_reserve_price

    @staticmethod
    def median(values):

        n = len(values)
        values = sorted(values)

        if n % 2 == 0:
            return (values[n // 2 - 1] + values[n // 2]) / 2

        return values[math.ceil(n / 2) - 1]

    @staticmethod
    def comb(n, k):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
