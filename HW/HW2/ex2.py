import abc
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
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
        p = self.R_hat + self.b_u[user] + self.b_i[item]
        if p < 0.5 or 5 < p:
            print(1, p)
        return self.R_hat + self.b_u[user] + self.b_i[item]


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.R_hat = ratings.rating.mean()
        self.centered_ratings = ratings.copy()
        self.centered_ratings.rating = self.centered_ratings.rating - self.R_hat
        self.b_u = ratings.groupby('user').rating.mean() - self.R_hat
        self.b_i = ratings.groupby('item').rating.mean() - self.R_hat

        ratings_matrix = pd.pivot_table(self.centered_ratings, values='rating', index='user', columns=['item']).fillna(0)
        self.user_corrs = cosine_similarity(ratings_matrix, ratings_matrix)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        user_correlations = self.user_corrs[user]
        top_correlations_indices = user_correlations.argsort()[-4:-1]
        neighbors_arg = 0
        correlations_sum = 0
        for neighbor in top_correlations_indices:
            correlation = user_correlations[neighbor]
            neighbor_ratings = self.centered_ratings[self.centered_ratings.user == neighbor]
            rating = neighbor_ratings[neighbor_ratings.item == item].rating
            if len(rating) > 0:
                neighbors_arg += correlation * float(rating)
            correlations_sum += np.abs(correlation)
        p = self.R_hat + self.b_u[user] + self.b_i[item] + neighbors_arg / correlations_sum
        if p < 0.5 or 5 < p:
            print(p)
        return self.R_hat + self.b_u[user] + self.b_i[item] + neighbors_arg / correlations_sum

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        return self.user_corrs[user1][user2]


class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """

        user_vec = np.zeros((self.U + self.I + 3, 1))

        user_vec[user] = 1
        user_vec[self.U + item] = 1
        user_vec[-3] = self.b_d_lambda(timestamp)
        user_vec[-2] = self.b_n_lambda(timestamp)
        user_vec[-1] = self.b_w_lambda(timestamp)

        prediction = (self.beta.T @ user_vec)[0][0]

        # assert 0.5 <= prediction <= 5

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
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass


# if __name__ == '__main__':
#     from datetime import datetime
#
#     datetimeexamples = [835355664, 835355532, 1260759205, 949949538]
#
#
#
#     for datetimeexample in datetimeexamples:
#         print(datetime.fromtimestamp(datetimeexample))
#         print(b_d_lambda(datetimeexample))
#         print(b_n_lambda(datetimeexample))
#         print(b_w_lambda(datetimeexample))
