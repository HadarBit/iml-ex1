

import pandas as pd
import numpy as np
import datetime as dt
import os
from sklearn.model_selection import train_test_split
from typing import NoReturn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from linear_regression import LinearRegression
from sklearn import linear_model



def splitting_into_train_test(X, y,test_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size,random_state = 42)
    return X_train,X_test, y_train, y_test
    

def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    #removing rows with no ID 
    index_to_drop  = X[X.isna().any(axis=1)].index
    X = X.drop(index_to_drop)
    y = y.drop(index_to_drop)

    index_to_drop  =y[y.isna()].index
    X = X.drop(index_to_drop)
    y = y.drop(index_to_drop)

    # removing rows that the hours renovate before the hours was built
    f_was_renovate  = X["yr_renovated"] != 0 
    wrong_val  = X["yr_renovated"] < X["yr_built"]
    index_to_drop = X[f_was_renovate&wrong_val].index
    X = X.drop(index_to_drop)
    y = y.drop(index_to_drop)

    # removing rows that the sale date was before the hous was built 
    index_to_drop = X[pd.to_datetime(X['date']).dt.year <X["yr_built"]].index
    X = X.drop(index_to_drop)
    y = y.drop(index_to_drop)

    # X["was_renovated"] = 0 
    # X.loc[X["yr_renovated"]!=0, "was_renovated"] = 1
    # X["years_since_built"] = dt.date.today().year - X['yr_built']
    # X["years_since_renovate "] = dt.date.today().year - X['was_renovated']
    # X["years_since_sale_date"] = np.floor((pd.to_datetime(dt.date.today())- pd.to_datetime(X['date'])).dt.days/ 365)

    # del X["yr_built"]
    # del X["yr_renovated"]
    del X["date"]
    return X, y 


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # X["was_renovated"] = 0 
    # X.loc[X["yr_renovated"]!=0, "was_renovated"] = 1
    # X["years_since_built"] = dt.date.today().year - X['yr_built']
    # X["years_since_renovate "] = dt.date.today().year - X['was_renovated']
    # X["years_since_sale_date"] = np.floor((pd.to_datetime(dt.date.today())- pd.to_datetime(X['date'])).dt.days/ 365)

    # del X["yr_built"]
    # del X["yr_renovated"]
    del X["date"]
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    
    for feature in X.columns:
        # Calculate Pearson correlation manually
        # print(X[feature].values)
        covariance = np.cov(X[feature].values,y.values)[0, 1]
        std_feature = np.std(X[feature].values)
        std_y = np.std(y.values)
        correlation = covariance / (std_feature * std_y)
        
        # Create scatter plot
        plt.figure()
        plt.scatter(X[feature], y, alpha=0.5)
        plt.title(f'{feature} vs Response\nPearson Correlation: {correlation:.2f}')
        plt.xlabel(feature)
        plt.ylabel('Response')
        
        # Save plot to specified path
        filename = f"{feature}_vs_response.png"
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath)
        plt.close()

if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df, df["price"]
    del X["price"]
    # Question 2 - split train test
    np.random.seed(30)
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state = 30)
    # Question 3 - preprocessing of housing prices train dataset
    X_train,y_train = preprocess_train(X_train,y_train)
    # print(X_train["id"].isna().value_counts())
    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train,y_train,"./vis") 
    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)
    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # percentages = np.arange(10, 101, 1)
    # mean_losses = []
    # std_losses = []
    # all_data= []
    # for p in percentages:
    #     # graph_df["percentage"] = p
    #     print(p)
    #     losses = []
    #     for i in range(10):
    #         if p == 100:
    #             train_sample_rate = 1
    #         else:
    #             train_sample_rate = p/100
    #         s_X_train, s_X_test, s_y_train, s_y_test = train_test_split(X_train, y_train, train_size=train_sample_rate,random_state=30)
    #         sklearn_model = linear_model.LinearRegression(fit_intercept = True)
    #         model = LinearRegression(include_intercept=True)
    #         model.fit(s_X_train, s_y_train)
    #         sklearn_model.fit(s_X_train,s_y_train)
    #         losses.append(sklearn_model.score(X_test,y_test))
    #     #print("finish_lop")
    #     # print(losses)
    #     mean_losses.append(np.mean(losses))
    #     std_losses.append(np.std(losses))
    #     all_data.append([p,np.mean(losses),np.std(losses)])

    # graph_df = pd.DataFrame(all_data,columns= ["percentage","mean","std"])
    # graph_df.to_csv("graph_result.csv")
    # mean_losses = np.array(mean_losses)
    # std_losses = np.array(std_losses)

    # print(mean_losses,std_losses)
    # # print(graph_df)
    # # Plotting average loss as a function of training size with error ribbon
    # plt.figure(figsize=(10, 6))
    # print(graph_df.astype(int))
    # plt.plot(graph_df["percentage"], graph_df["mean"])
    # # plt.fill_between(percentages, mean_losses - 2*std_losses, mean_losses + 2*std_losses, alpha=0.2, color='blue', label='Mean ± 2*STD')
    # plt.xlabel('Percentage of Training Data')
    # plt.ylabel('Mean Squared Error Loss')
    # plt.title('Loss as Function of Training Data Size')
    # plt.legend()
    # plt.show()


    # X_test, y_test = preprocess_data(X_test, y_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    ps = list(range(10, 101))
    results = np.zeros((len(ps), 10))
    for i, p in enumerate(ps):
        for j in range(results.shape[1]):
            _X = X_train.sample(frac=p / 100.0)
            _y = y_train.loc[_X.index]
            model = LinearRegression(include_intercept=True)
            results[i, j] = model.fit(_X, _y).loss(X_test, y_test)

    m, s = results.mean(axis=1), results.std(axis=1)
    fig = go.Figure([go.Scatter(x=ps, y=m-2*s, fill=None, mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=ps, y=m+2*s, fill='tonexty', mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=ps, y=m, mode="markers+lines", marker=dict(color="black"))],
                    layout=go.Layout(title="Test MSE as Function Of Training Size",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set"),
                                     showlegend=False))
    fig.write_image("mse.over.training.percentage.png")
