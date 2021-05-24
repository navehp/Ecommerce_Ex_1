import abc
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.sparse
import sklearn.neural_network

from sklearn.neural_network import MLPRegressor
from tqdm import tqdm


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

# TODO inherit from abstract class and delete rmse

class CompetitionRecommender():

    def __init__(self, ratings):
        self.initialize_predictor(ratings)

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        r = true_ratings['rating']
        p = true_ratings.apply(lambda x: self.predict(int(x['user']), int(x['item']), x['timestamp']), axis=1)
        return np.sqrt(((r - p) ** 2).sum() / len(r))

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.ratings = ratings
        self.b_d_lambda = lambda timestamp: int(6 <= datetime.fromtimestamp(timestamp).hour < 18)
        self.b_n_lambda = lambda timestamp: int(1 - self.b_d_lambda(timestamp))
        self.b_w_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() in [4, 5])
        self.ratings['day'] = ratings.timestamp.apply(self.b_d_lambda)
        self.ratings['night'] = ratings.timestamp.apply(self.b_n_lambda)
        self.ratings['weekend'] = ratings.timestamp.apply(self.b_w_lambda)
        self.ratings['bias'] = pd.Series(np.ones(len(ratings)))

        self.items = self.ratings.item.unique()
        self.users = self.ratings.user.unique()

        self.user_indices = {user: i for i, user in enumerate(self.users)}
        self.item_indices = {item: len(self.user_indices) + i for i, item in enumerate(self.items)}
        self.day_index = len(self.user_indices) + len(self.item_indices)
        self.night_index = len(self.user_indices) + len(self.item_indices) + 1
        self.weekend_index = len(self.user_indices) + len(self.item_indices) + 2
        self.bias_index = len(self.user_indices) + len(self.item_indices) + 3

        self.R_hat = ratings['rating'].mean()
        self.y = ratings.rating - self.R_hat

        self.create_sparse_matrix()
        self.model = sklearn.neural_network.MLPRegressor(max_iter=50).fit(self.sparse_ratings, self.y)
        print("Fitted Model")


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        user_vector = np.zeros(len(self.users) + len(self.items) + 4)

        user_vector[self.user_indices[user]] = 1
        user_vector[self.item_indices[item]] = 1
        user_vector[self.day_index] = self.b_d_lambda(timestamp)
        user_vector[self.night_index] = self.b_n_lambda(timestamp)
        user_vector[self.weekend_index] = self.b_w_lambda(timestamp)
        user_vector[self.bias_index] = 1

        prediction = self.model.predict(user_vector.reshape(1, -1)) + self.R_hat

        prediction = max(0.5, prediction)
        prediction = min(5.0, prediction)
        return prediction


    def create_sparse_matrix(self):
        rows = []
        cols = []
        data = []

        for i, row in tqdm(self.ratings.iterrows(), "Data to sparse matrix"):
            rows.extend([i] * 5)
            cols.extend([
                self.user_indices[row['user']],
                self.item_indices[row['item']],
                self.day_index if row['day'] else self.night_index,
                self.weekend_index,
                self.bias_index
            ])
            data.extend([1, 1, 1, 1 if row['weekend'] else 0, 1])

        self.sparse_ratings = scipy.sparse.coo_matrix((data, (rows, cols)),
                                                      shape=(len(self.ratings), len(self.user_indices) + len(self.item_indices) + 4))

        print("loaded data to sparse matrix successfully")

if __name__ == '__main__':
    pass