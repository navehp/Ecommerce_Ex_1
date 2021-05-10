import abc
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime

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

        return np.sqrt(np.average([columns.loc['rating'] - self.predict(int(columns.loc['user']), int(columns.loc['item']), int(columns.loc['timestamp'])) for _, columns in true_ratings.iterrows()]))




class BaselineRecommender(Recommender):
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


class NeighborhoodRecommender(Recommender):
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

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        pass


class LSRecommender(Recommender):

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.b_d_lambda = lambda timestamp: int(6 <= datetime.fromtimestamp(timestamp).hour < 18)
        self.b_n_lambda = lambda timestamp: int(1 - self.b_d_lambda(timestamp))
        self.b_w_lambda = lambda timestamp: int(datetime.fromtimestamp(timestamp).weekday() in [4, 5])

        self.U = ratings['user'].nunique()
        self.I = ratings['item'].nunique()

        self.X = np.zeros((len(ratings.index), self.U+self.I+3))
        self.R = np.zeros((len(ratings.index), 1))


        for index, columns in ratings.iterrows():

            user_ind = int(columns.loc['user'])
            item_ind = int(columns.loc['item'])
            timestamp = columns.loc['timestamp']

            self.X[index, user_ind] = 1
            self.X[index, self.U + item_ind] = 1
            self.X[index, -3] = self.b_d_lambda(timestamp)
            self.X[index, -2] = self.b_n_lambda(timestamp)
            self.X[index, -1] = self.b_w_lambda(timestamp)

            self.R[index, 0] = columns.loc['rating']


        self.y = self.R - ratings['rating'].mean()



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
