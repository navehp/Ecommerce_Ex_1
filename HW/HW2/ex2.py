import abc
import pickle
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.sparse
from sklearn import linear_model

from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import time
from pickle import load, dump


class Recommender(abc.ABC):

    def __init__(self, ratings: pd.DataFrame, *args):
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        r = true_ratings['rating']
        p = true_ratings.apply(lambda x: self.predict(int(x['user']), int(x['item']), x['timestamp']), axis=1)
        return np.sqrt(((r - p) ** 2).sum() / len(r))


class BaselineRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.R_hat = ratings.rating.mean()
        self.b_u = ratings.groupby('user').rating.mean() - self.R_hat
        self.b_i = ratings.groupby('item').rating.mean() - self.R_hat

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        prediction = self.R_hat + self.b_u[user] + self.b_i[item]
        prediction = max(0.5, prediction)
        prediction = min(5, prediction)
        return prediction


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.R_hat = ratings.rating.mean()
        self.centered_ratings = ratings.copy()
        self.centered_ratings.rating = self.centered_ratings.rating - self.R_hat
        self.b_u = ratings.groupby('user').rating.mean() - self.R_hat
        self.b_i = ratings.groupby('item').rating.mean() - self.R_hat

        self.users = self.centered_ratings.user.apply(int).unique()
        self.user_items = ratings.groupby('user')['item'].apply(set).to_dict()
        self.user_ratings = {user: {} for user in self.users}

        for index, row in self.centered_ratings.iterrows():
            self.user_ratings[int(row['user'])][int(row['item'])] = row['rating']

        self.corr_matrix = {}
        for user1 in self.users:
            for user2 in self.users:
                if user1 > user2:
                    continue
                elif user1 == user2:
                    self.corr_matrix[(user1, user2)] = 1
                else:
                    similarity = self.user_similarity(user1, user2)
                    self.corr_matrix[(user1, user2)] = similarity
                    self.corr_matrix[(user2, user1)] = similarity

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        user_correlations = [(user2, self.corr_matrix[(user, user2)]) for user2 in self.users
                             if item in self.user_items[user2] and user2 != user]
        top_correlations_indices = sorted(user_correlations, key=lambda x: abs(x[1]))[-3:]

        neighbors_arg = 0
        correlations_sum = 0
        for neighbor, correlation in top_correlations_indices:
            rating = self.user_ratings[neighbor][item]
            neighbors_arg += correlation * float(rating)
            correlations_sum += np.abs(correlation)

        if correlations_sum == 0:
            prediction = self.R_hat + self.b_u[user] + self.b_i[item]
        else:
            prediction = self.R_hat + self.b_u[user] + self.b_i[item] + neighbors_arg / correlations_sum
        prediction = max(0.5, prediction)
        prediction = min(5, prediction)
        return prediction


    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        user1, user2 = int(user1), int(user2)
        intersection_items = list(self.user_items[user1].intersection(self.user_items[user2]))
        user_1_ratings = self.user_ratings[user1]
        user_2_ratings = self.user_ratings[user2]
        user1_vector = np.array([user_1_ratings[item] for item in intersection_items])
        user2_vector = np.array([user_2_ratings[item] for item in intersection_items])
        nominator = np.dot(user1_vector, user2_vector)
        denominator = np.linalg.norm(user1_vector) * np.linalg.norm(user2_vector)
        if denominator == 0:
            return 0
        else:
            return nominator / denominator


class LSRecommender(Recommender):

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.b_d_lambda = lambda timestamp: int(6 <= datetime.fromtimestamp(timestamp).hour < 18)
        self.b_n_lambda = lambda timestamp: int(1 - self.b_d_lambda(timestamp))
        self.b_w_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() in [4, 5])
        day = ratings.timestamp.apply(self.b_d_lambda)
        night = ratings.timestamp.apply(self.b_n_lambda)
        weekend = ratings.timestamp.apply(self.b_w_lambda)

        self.X = pd.get_dummies(ratings.user, prefix='user')
        self.X = pd.concat([self.X, pd.get_dummies(ratings.item, prefix='item')], axis=1)
        self.X['b_d'] = day
        self.X['b_n'] = night
        self.X['b_w'] = weekend
        self.X['bias'] = pd.Series(np.ones(len(ratings)))

        columns_list = list(self.X.columns)
        self.column2ind = {col: columns_list.index(col) for col in columns_list}

        self.R_hat = ratings['rating'].mean()
        self.y = ratings.rating - self.R_hat

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        user_vector = np.zeros(len(self.X.columns))
        user_vector[self.column2ind[f'user_{float(user)}']] = 1
        user_vector[self.column2ind[f'item_{float(item)}']] = 1
        user_vector[self.column2ind[f'b_d']] = self.b_d_lambda(timestamp)
        user_vector[self.column2ind[f'b_n']] = self.b_n_lambda(timestamp)
        user_vector[self.column2ind[f'b_w']] = self.b_w_lambda(timestamp)
        user_vector[self.column2ind[f'bias']] = 1

        prediction = (self.beta.T @ user_vector) + self.R_hat

        prediction = max(0.5, prediction)
        prediction = min(5.0, prediction)

        return prediction


    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """

        self.beta = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

        return (self.X, self.beta, self.y)


class CompetitionRecommender(Recommender):

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.ratings = ratings
        self.b_t1_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).hour < 3)
        self.b_t2_lambda = lambda timestamp: int(3 <= datetime.fromtimestamp(timestamp).hour < 6)
        self.b_t3_lambda = lambda timestamp: int(6 <= datetime.fromtimestamp(timestamp).hour < 9)
        self.b_t4_lambda = lambda timestamp: int(9 <= datetime.fromtimestamp(timestamp).hour < 12)
        self.b_t5_lambda = lambda timestamp: int(12 <= datetime.fromtimestamp(timestamp).hour < 15)
        self.b_t6_lambda = lambda timestamp: int(15 <= datetime.fromtimestamp(timestamp).hour < 18)
        self.b_t7_lambda = lambda timestamp: int(18 <= datetime.fromtimestamp(timestamp).hour < 21)
        self.b_t8_lambda = lambda timestamp: int(21 <= datetime.fromtimestamp(timestamp).hour)

        self.b_d1_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() == 1)
        self.b_d2_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() == 2)
        self.b_d3_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() == 3)
        self.b_d4_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() == 4)
        self.b_d5_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() == 5)
        self.b_d6_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() == 6)
        self.b_d7_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() == 7)


        self.b_m1_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 1)
        self.b_m2_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 2)
        self.b_m3_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 3)
        self.b_m4_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 4)
        self.b_m5_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 5)
        self.b_m6_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 6)
        self.b_m7_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 7)
        self.b_m8_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 8)
        self.b_m9_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 9)
        self.b_m10_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 10)
        self.b_m11_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 11)
        self.b_m12_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).month == 12)


        self.ratings['t1'] = ratings.timestamp.apply(self.b_t1_lambda)
        self.ratings['t2'] = ratings.timestamp.apply(self.b_t2_lambda)
        self.ratings['t3'] = ratings.timestamp.apply(self.b_t3_lambda)
        self.ratings['t4'] = ratings.timestamp.apply(self.b_t4_lambda)
        self.ratings['t5'] = ratings.timestamp.apply(self.b_t5_lambda)
        self.ratings['t6'] = ratings.timestamp.apply(self.b_t6_lambda)
        self.ratings['t7'] = ratings.timestamp.apply(self.b_t7_lambda)
        self.ratings['t8'] = ratings.timestamp.apply(self.b_t8_lambda)

        self.ratings['d1'] = ratings.timestamp.apply(self.b_d1_lambda)
        self.ratings['d2'] = ratings.timestamp.apply(self.b_d2_lambda)
        self.ratings['d3'] = ratings.timestamp.apply(self.b_d3_lambda)
        self.ratings['d4'] = ratings.timestamp.apply(self.b_d4_lambda)
        self.ratings['d5'] = ratings.timestamp.apply(self.b_d5_lambda)
        self.ratings['d6'] = ratings.timestamp.apply(self.b_d6_lambda)
        self.ratings['d7'] = ratings.timestamp.apply(self.b_d7_lambda)

        self.ratings['m1'] = ratings.timestamp.apply(self.b_m1_lambda)
        self.ratings['m2'] = ratings.timestamp.apply(self.b_m2_lambda)
        self.ratings['m3'] = ratings.timestamp.apply(self.b_m3_lambda)
        self.ratings['m4'] = ratings.timestamp.apply(self.b_m4_lambda)
        self.ratings['m5'] = ratings.timestamp.apply(self.b_m5_lambda)
        self.ratings['m6'] = ratings.timestamp.apply(self.b_m6_lambda)
        self.ratings['m7'] = ratings.timestamp.apply(self.b_m7_lambda)
        self.ratings['m8'] = ratings.timestamp.apply(self.b_m8_lambda)
        self.ratings['m9'] = ratings.timestamp.apply(self.b_m9_lambda)
        self.ratings['m10'] = ratings.timestamp.apply(self.b_m10_lambda)
        self.ratings['m11'] = ratings.timestamp.apply(self.b_m11_lambda)
        self.ratings['m12'] = ratings.timestamp.apply(self.b_m12_lambda)


        self.ratings['bias'] = pd.Series(np.ones(len(ratings)))

        self.items = self.ratings.item.unique()
        self.users = self.ratings.user.unique()

        self.user_indices = {user: i for i, user in enumerate(self.users)}
        self.item_indices = {item: len(self.user_indices) + i for i, item in enumerate(self.items)}

        self.bias_index = len(self.user_indices) + len(self.item_indices)

        self.t1_index = len(self.user_indices) + len(self.item_indices) + 1
        self.t2_index = len(self.user_indices) + len(self.item_indices) + 2
        self.t3_index = len(self.user_indices) + len(self.item_indices) + 3
        self.t4_index = len(self.user_indices) + len(self.item_indices) + 4
        self.t5_index = len(self.user_indices) + len(self.item_indices) + 5
        self.t6_index = len(self.user_indices) + len(self.item_indices) + 6
        self.t7_index = len(self.user_indices) + len(self.item_indices) + 7
        self.t8_index = len(self.user_indices) + len(self.item_indices) + 8

        self.d1_index = len(self.user_indices) + len(self.item_indices) + 1 + 8
        self.d2_index = len(self.user_indices) + len(self.item_indices) + 2 + 8
        self.d3_index = len(self.user_indices) + len(self.item_indices) + 3 + 8
        self.d4_index = len(self.user_indices) + len(self.item_indices) + 4 + 8
        self.d5_index = len(self.user_indices) + len(self.item_indices) + 5 + 8
        self.d6_index = len(self.user_indices) + len(self.item_indices) + 6 + 8
        self.d7_index = len(self.user_indices) + len(self.item_indices) + 7 + 8

        self.m1_index = len(self.user_indices) + len(self.item_indices) + 1 + 15
        self.m2_index = len(self.user_indices) + len(self.item_indices) + 2 + 15
        self.m3_index = len(self.user_indices) + len(self.item_indices) + 3 + 15
        self.m4_index = len(self.user_indices) + len(self.item_indices) + 4 + 15
        self.m5_index = len(self.user_indices) + len(self.item_indices) + 5 + 15
        self.m6_index = len(self.user_indices) + len(self.item_indices) + 6 + 15
        self.m7_index = len(self.user_indices) + len(self.item_indices) + 7 + 15
        self.m8_index = len(self.user_indices) + len(self.item_indices) + 8 + 15
        self.m9_index = len(self.user_indices) + len(self.item_indices) + 9 + 15
        self.m10_index = len(self.user_indices) + len(self.item_indices) + 10 + 15
        self.m11_index = len(self.user_indices) + len(self.item_indices) + 11 + 15
        self.m12_index = len(self.user_indices) + len(self.item_indices) + 12 + 15

        self.R_hat = ratings['rating'].mean()
        self.y = ratings.rating - self.R_hat

        # self.create_sparse_matrix()
        # with open('sparse_matrix.pickle', 'wb') as f:
        #     pickle.dump(self.sparse_ratings, f)
        with open('sparse_matrix.pickle', 'rb') as f:
            self.sparse_ratings = pickle.load(f)

        print("Processed Data")
        start = time.time()
        self.model = linear_model.RidgeCV(cv=5).fit(self.sparse_ratings, self.y)
        print(f"Fitted Model, time: {time.time() - start}")
        print('Parameters: ', self.model.get_params(), self.model.alpha_ )


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        user_vector = np.zeros(len(self.users) + len(self.items) + 11)

        if user in self.user_indices:
            user_vector[self.user_indices[user]] = 1
        if item in self.item_indices:
            user_vector[self.item_indices[item]] = 1

        user_vector[self.bias_index] = 1

        user_vector[self.t1_index] = self.b_t1_lambda(timestamp)
        user_vector[self.t2_index] = self.b_t2_lambda(timestamp)
        user_vector[self.t3_index] = self.b_t3_lambda(timestamp)
        user_vector[self.t4_index] = self.b_t4_lambda(timestamp)
        user_vector[self.t5_index] = self.b_t5_lambda(timestamp)
        user_vector[self.t6_index] = self.b_t6_lambda(timestamp)
        user_vector[self.t7_index] = self.b_t7_lambda(timestamp)
        user_vector[self.t8_index] = self.b_t8_lambda(timestamp)

        user_vector[self.d1_index] = self.b_d1_lambda(timestamp)
        user_vector[self.d2_index] = self.b_d2_lambda(timestamp)
        user_vector[self.d3_index] = self.b_d3_lambda(timestamp)
        user_vector[self.d4_index] = self.b_d4_lambda(timestamp)
        user_vector[self.d5_index] = self.b_d5_lambda(timestamp)
        user_vector[self.d6_index] = self.b_d6_lambda(timestamp)
        user_vector[self.d7_index] = self.b_d7_lambda(timestamp)

        user_vector[self.m1_index] = self.b_m1_lambda(timestamp)
        user_vector[self.m2_index] = self.b_m2_lambda(timestamp)
        user_vector[self.m3_index] = self.b_m3_lambda(timestamp)
        user_vector[self.m4_index] = self.b_m4_lambda(timestamp)
        user_vector[self.m5_index] = self.b_m5_lambda(timestamp)
        user_vector[self.m6_index] = self.b_m6_lambda(timestamp)
        user_vector[self.m7_index] = self.b_m7_lambda(timestamp)
        user_vector[self.m8_index] = self.b_m8_lambda(timestamp)
        user_vector[self.m9_index] = self.b_m9_lambda(timestamp)
        user_vector[self.m10_index] = self.b_m10_lambda(timestamp)
        user_vector[self.m11_index] = self.b_m11_lambda(timestamp)
        user_vector[self.m12_index] = self.b_m12_lambda(timestamp)


        prediction = self.model.predict(user_vector.reshape(1, -1)) + self.R_hat

        prediction = max(0.5, prediction)
        prediction = min(5.0, prediction)
        return prediction


    def create_sparse_matrix(self):
        rows = []
        cols = []
        data = []

        for i, row in tqdm(self.ratings.iterrows(), "Data to sparse matrix"):
            rows.extend([i] * 12)
            cols.extend([
                self.user_indices[row['user']],
                self.item_indices[row['item']],
                self.t1_index,
                self.t2_index,
                self.t3_index,
                self.t4_index,
                self.t5_index,
                self.t6_index,
                self.t7_index,
                self.t8_index,
                self.weekend_index,
                self.bias_index
            ])
            data.extend([1, 1,
                         1 if row['t1'] else 0,
                         1 if row['t2'] else 0,
                         1 if row['t3'] else 0,
                         1 if row['t4'] else 0,
                         1 if row['t5'] else 0,
                         1 if row['t6'] else 0,
                         1 if row['t7'] else 0,
                         1 if row['t8'] else 0,
                         1 if row['weekend'] else 0,
                         1])

        self.sparse_ratings = scipy.sparse.coo_matrix((data, (rows, cols)),
                                                      shape=(len(self.ratings), len(self.user_indices) + len(self.item_indices) + 11))

        print("loaded data to sparse matrix successfully")

if __name__ == '__main__':
    pass