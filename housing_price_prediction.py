from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from sklearn.utils import shuffle

# enable/disable useful plots of housing data
enable_plots = True


class Columns(Enum):
    median_house_value = "median_house_value"
    ocean_proximity = "ocean_proximity"
    households = "households"
    population = "population"
    total_bedrooms = "total_bedrooms"
    total_rooms = "total_rooms"
    longitude = "longitude"
    latitude = "latitude"
    bedroom_ratio = "bedroom_ratio"
    household_rooms = "household_rooms"


def main():
    x_train, x_test, y_train, y_test = get_train_test()
    skewed_columns = [Columns.total_rooms, Columns.total_bedrooms, Columns.population, Columns.households]

    # build train data
    train_data = x_train.join(y_train)
    train_data = gen_features(train_data, skewed_columns)

    # build test data
    test_data = x_test.join(y_test)
    test_data = gen_features(test_data, skewed_columns)

    # reassign x_train y_train with new features in train_data
    x_train, y_train = get_x_y(train_data)
    x_test, y_test = get_x_y(test_data)

    if enable_plots:
        view_plots(train_data)

    # train/test Random Forest Regressor model
    train_test_rfr(x_train, y_train, x_test, y_test)


def get_train_test():
    """
    retrieves the housing data set and splits into train/test x/y dataframes
    converts "ocean_proximity" column to one hot encoded columns
    there are only a few entries where ocean_proximity=ISLAND
    we are shuffling the data here to ensure ISLAND entries
    exist in both train and test sets
    """
    data = shuffle(pd.read_csv("housing.csv"))
    data = data.dropna()

    # one hot encode ocean proximity column
    ocean_prox = pd.get_dummies(data[Columns.ocean_proximity.value]).astype(int)
    data = data.join(ocean_prox)
    data = data.drop([Columns.ocean_proximity.value], axis=1)

    x = data.drop([Columns.median_house_value.value], axis=1)
    y = data[Columns.median_house_value.value]
    return train_test_split(x, y, test_size=0.2)


def gen_features(df, skewed_columns):
    """
    1. converts skewed columns to their log values. this gives them a normal distribution
    2. adds additional feature columns to the dataframe.
    """
    # convert skewed columns
    for column in skewed_columns:
        df[column.value] = np.log(df[column.value] + 1)

    # add additional feature columns
    df[Columns.bedroom_ratio.value] = df[Columns.total_bedrooms.value] / df[Columns.total_rooms.value]
    df[Columns.household_rooms.value] = df[Columns.total_rooms.value] / df[Columns.households.value]

    return df


def view_plots(train_data):
    """
    generates 2 plots of the train data
    1. scatterplot of geographic position with hue determined by median house value
    2. heatmap of correlation between dataframe columns
    """
    plt.figure(figsize=(15, 8))

    # view spatial distribution of house value
    sns.scatterplot(x=Columns.latitude.value, y=Columns.longitude.value, data=train_data,
                    hue=Columns.median_house_value.value,
                    palette="coolwarm")
    plt.show()

    # view correlation between features
    sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
    plt.show()


def get_x_y(df):
    """
    splits dataframe into x and y
    """
    return df.drop([Columns.median_house_value.value], axis=1), df[Columns.median_house_value.value]


def train_test_rfr(x_train, y_train, x_test, y_test):
    """
    trains and tests on a random forest regressor model
    """
    forest = RandomForestRegressor()

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 8]
    }
    grid_search = GridSearchCV(forest, param_grid, cv=5,
                               scoring="neg_mean_squared_error",
                               return_train_score=True)
    grid_search.fit(x_train, y_train)
    best_forest = grid_search.best_estimator_
    score = best_forest.score(x_test, y_test)
    print(f"Best Forest scored: {score * 100:.2f}")
    print(best_forest)


if __name__ == "__main__":
    main()
