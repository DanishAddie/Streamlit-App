import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting, hierarchical_portfolio, HRPOpt
import yfinance as yfin

yfin.pdr_override()  # Fix DataReader

def plot_cum_returns(data, title):
    daily_cum_returns = 1 + data.dropna().pct_change()
    daily_cum_returns = daily_cum_returns.cumprod() * 100
    fig = px.line(daily_cum_returns, title=title)
    return fig

def plot_efficient_frontier_and_max_sharpe(mu, S):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_max_sharpe = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    # Find the max sharpe portfolio
    ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
    # Output
    ax.legend()
    return fig
def plot_efficient_frontier_and_min_volatility(mu, S):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_min_volatility = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    # Find the max sharpe portfolio
    ef_min_volatility.min_volatility()
    ret_tangent, std_tangent, _ = ef_min_volatility.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Min Volatility")
    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
    # Output
    ax.legend()
    return fig
def plot_efficient_frontier_and_efficient_risk(mu, S):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_efficient_risk = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    # Find the max sharpe portfolio
    ef_efficient_risk.efficient_risk(100)
    ret_tangent, std_tangent, _ = ef_efficient_risk.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Maximized Return")
    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
    # Output
    ax.legend()
    return fig
def plot_efficient_frontier_and_target_volatility(mu, S, max_volatility):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_efficient_risk = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    # Find the max sharpe portfolio
    ef_efficient_risk.efficient_risk(max_volatility/100)
    ret_tangent, std_tangent, _ = ef_efficient_risk.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Maximized Return")
    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
    # Output
    ax.legend()
    return fig
def plot_efficient_frontier_and_hierarchical_portfolio(mu,S):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    hrp = HRPOpt(mu, S)
    hrp.optimize()
    fig = plotting.plot_dendrogram(hrp)
    return fig


def plot_random_3d_max_sharpe(mu, S, tickers):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_max_sharpe = EfficientFrontier(mu, S)
    ef_max_sharpe.max_sharpe(risk_free_rate=0.02)

    # Find weight of each stock in optimized portfolio
    weights = ef_max_sharpe.clean_weights()  # Ordered_Dict of weights for each stock in optimized portfolio

    # Find the max sharpe portfolio
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()  # Return and Variability of max_sharpe portfolio
    max_sharpe = ret_tangent / std_tangent

    # Generate random portfolios
    n_samples = 49  # 49 + 1 (max sharpe) = 50
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)  # ndArray of arrays of random portfolio allocation
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds  # array of sharpe for random portfolios

    # Create array x, y and z and add weights to them
    x = []
    y = []
    z = []
    size = []
    for i in range(n_samples):
        x.append(round(w[i][0], 4))
        y.append(round(w[i][1], 4))
        z.append(round(w[i][2], 4))
        size.append(abs(round(sharpes[i], 4)))
    x.append(round(weights[tickers[0]], 4))
    y.append(round(weights[tickers[1]], 4))
    z.append(round(weights[tickers[2]], 4))
    size.append(round(max_sharpe))

    data = {
        tickers[0]: x,
        tickers[1]: y,
        tickers[2]: z
    }
    df = pd.DataFrame(data)

    fig = px.scatter_3d(data_frame=df, x=tickers[0], y=tickers[1], z=tickers[2], size=size)

    return fig
def plot_random_3d_min_volatility(mu, S, tickers):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_min_volatility = EfficientFrontier(mu, S)
    ef_min_volatility.min_volatility()

    # Find weight of each stock in optimized portfolio
    weights = ef_min_volatility.clean_weights()  # Ordered_Dict of weights for each stock in optimized portfolio

    # Find the max sharpe portfolio
    ret_tangent, std_tangent, _ = ef_min_volatility.portfolio_performance()  # Return and Variability of max_sharpe portfolio
    max_sharpe = ret_tangent / std_tangent

    # Generate random portfolios
    n_samples = 49  # 49 + 1 (max sharpe) = 50
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)  # ndArray of arrays of random portfolio allocation
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds  # array of sharpe for random portfolios

    # Create array x, y and z and add weights to them
    x = []
    y = []
    z = []
    size = []
    for i in range(n_samples):
        x.append(round(w[i][0], 4))
        y.append(round(w[i][1], 4))
        z.append(round(w[i][2], 4))
        size.append(abs(round(sharpes[i], 4)))
    x.append(round(weights[tickers[0]], 4))
    y.append(round(weights[tickers[1]], 4))
    z.append(round(weights[tickers[2]], 4))
    size.append(round(max_sharpe))

    data = {
        tickers[0]: x,
        tickers[1]: y,
        tickers[2]: z
    }
    df = pd.DataFrame(data)

    fig = px.scatter_3d(data_frame=df, x=tickers[0], y=tickers[1], z=tickers[2], size=size)

    return fig
def plot_random_3d_efficient_risk(mu, S, tickers):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_efficient_risk = EfficientFrontier(mu, S)
    ef_efficient_risk.efficient_risk(100)

    # Find weight of each stock in optimized portfolio
    weights = ef_efficient_risk.clean_weights()  # Ordered_Dict of weights for each stock in optimized portfolio

    # Find the max sharpe portfolio
    ret_tangent, std_tangent, _ = ef_efficient_risk.portfolio_performance()  # Return and Variability of max_sharpe portfolio
    max_sharpe = ret_tangent / std_tangent

    # Generate random portfolios
    n_samples = 49  # 49 + 1 (max sharpe) = 50
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)  # ndArray of arrays of random portfolio allocation
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds  # array of sharpe for random portfolios

    # Create array x, y and z and add weights to them
    x = []
    y = []
    z = []
    size = []
    for i in range(n_samples):
        x.append(round(w[i][0], 4))
        y.append(round(w[i][1], 4))
        z.append(round(w[i][2], 4))
        size.append(abs(round(sharpes[i], 4)))
    x.append(round(weights[tickers[0]], 4))
    y.append(round(weights[tickers[1]], 4))
    z.append(round(weights[tickers[2]], 4))
    size.append(round(max_sharpe))

    data = {
        tickers[0]: x,
        tickers[1]: y,
        tickers[2]: z
    }
    df = pd.DataFrame(data)

    fig = px.scatter_3d(data_frame=df, x=tickers[0], y=tickers[1], z=tickers[2], size=size)

    return fig
def plot_random_3d_target_volatility(mu, S, tickers, max_volatility):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_efficient_risk = EfficientFrontier(mu, S)
    ef_efficient_risk.efficient_risk(max_volatility/100)

    # Find weight of each stock in optimized portfolio
    weights = ef_efficient_risk.clean_weights()  # Ordered_Dict of weights for each stock in optimized portfolio

    # Find the max sharpe portfolio
    ret_tangent, std_tangent, _ = ef_efficient_risk.portfolio_performance()  # Return and Variability of max_sharpe portfolio
    max_sharpe = ret_tangent / std_tangent

    # Generate random portfolios
    n_samples = 49  # 49 + 1 (max sharpe) = 50
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)  # ndArray of arrays of random portfolio allocation
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds  # array of sharpe for random portfolios

    # Create array x, y and z and add weights to them
    x = []
    y = []
    z = []
    size = []
    for i in range(n_samples):
        x.append(round(w[i][0], 4))
        y.append(round(w[i][1], 4))
        z.append(round(w[i][2], 4))
        size.append(abs(round(sharpes[i], 4)))
    x.append(round(weights[tickers[0]], 4))
    y.append(round(weights[tickers[1]], 4))
    z.append(round(weights[tickers[2]], 4))
    size.append(round(max_sharpe))

    data = {
        tickers[0]: x,
        tickers[1]: y,
        tickers[2]: z
    }
    df = pd.DataFrame(data)

    fig = px.scatter_3d(data_frame=df, x=tickers[0], y=tickers[1], z=tickers[2], size=size)

    return fig
def plot_random_3d_hierarchical_portfolio(stocks_return, mu, S, tickers):
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    ef_hierarchical_portfolio = HRPOpt(stocks_return, S)
    ef_hierarchical_portfolio.optimize()

    # Find weight of each stock in optimized portfolio
    weights = ef_hierarchical_portfolio.clean_weights()  # Ordered_Dict of weights for each stock in optimized portfolio

    # Find the max sharpe portfolio
    ret_tangent, std_tangent, _ = ef_hierarchical_portfolio.portfolio_performance()  # Return and Variability of max_sharpe portfolio
    max_sharpe = ret_tangent / std_tangent

    # Generate random portfolios
    n_samples = 49  # 49 + 1 (max sharpe) = 50
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)  # ndArray of arrays of random portfolio allocation
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds  # array of sharpe for random portfolios

    # Create array x, y and z and add weights to them
    x = []
    y = []
    z = []
    size = []
    for i in range(n_samples):
        x.append(round(w[i][0], 4))
        y.append(round(w[i][1], 4))
        z.append(round(w[i][2], 4))
        size.append(abs(round(sharpes[i], 4)))
    x.append(round(weights[tickers[0]], 4))
    y.append(round(weights[tickers[1]], 4))
    z.append(round(weights[tickers[2]], 4))
    size.append(round(max_sharpe))

    data = {
        tickers[0]: x,
        tickers[1]: y,
        tickers[2]: z
    }
    df = pd.DataFrame(data)

    fig = px.scatter_3d(data_frame=df, x=tickers[0], y=tickers[1], z=tickers[2], size=size)

    return fig


st.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")
st.header("Stock Portfolio Optimizer")


col_date_1, col_date_2 = st.columns(2)

with col_date_1:
    start_date = st.date_input("Start Date", datetime(2013, 1, 1))

with col_date_2:
    end_date = st.date_input("End Date")  # it defaults to current date

tickers_string = st.text_input(
    'Enter all stock tickers to be included in portfolio separated by commas WITHOUT spaces, e.g. "AMZN,GOOGL,AAPL"',
    '').upper()
tickers = tickers_string.split(',')

option = st.selectbox('How would you like to optimize your portfolio?', ('Maximum Sharpe Ratio', 'Minimum Volatility (Risk)', 'Maximum Return', 'Set Maximum Volatility (Risk)', 'Hierarchical Risk Parity'))
# ^GSPC, NDX, QQQ

if option == "Set Maximum Volatility (Risk)":
    max_volatility = st.slider('Maximum Volatility (Risk) of the portfolio (in %)')

comparison_selectbox = st.selectbox('Compare Portfolio to:', ('S&P500', 'NASDAQ 100', 'Invesco QQQ Trust'))
if comparison_selectbox == 'S&P500':
    tickers.append('^GSPC')
elif comparison_selectbox == 'NASDAQ 100':
    tickers.append('NDX')
elif comparison_selectbox == "Invesco QQQ Trust":
    tickers.append('QQQ')

start = st.button('Generate Portfolio')

if start:
    try:
        # Get Stock Prices using pandas_datareader Library
        all_df = pdr.get_data_yahoo(tickers, start=start_date, end=end_date)['Adj Close']

        # Initialize a new dataframe for stocks in the portfolio
        stocks_df = pd.DataFrame()
        for i in range(len(tickers) - 1):
            stocks_df[tickers[i]] = all_df.loc[:,tickers[i]]

        # Initialize a new dataframe for comparison
        comparison_df = pd.DataFrame()
        comparison_df[comparison_selectbox] = all_df.loc[:,tickers[-1]]

        # Plot Individual Stock Prices
        fig_price = px.line(stocks_df, title='Price of Individual Stocks')
        # Plot Individual Stock Prices and Comparison
        fig_price_and_comparison = px.line(all_df, title='Price of Individual Stocks and Comparison')
        # Plot Individual Cumulative Returns
        fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
        # Calculate and Plot Correlation Matrix between Stocks
        corr_df = stocks_df.corr().round(2)
        fig_corr = px.imshow(corr_df, text_auto=True, title='Correlation between Stocks')

        # Calculate expected returns and sample covariance matrix for portfolio optimization later
        stocks_return = expected_returns.returns_from_prices(stocks_df)
        S = risk_models.sample_cov(stocks_df)
        mu = expected_returns.mean_historical_return(stocks_df)

        if option == "Maximum Sharpe Ratio":
            # Plot efficient frontier curve by Maximizing Sharpe
            fig_max_sharpe = plot_efficient_frontier_and_max_sharpe(mu, S)
            fig_efficient_frontier_max_sharpe = BytesIO()
            fig_max_sharpe.savefig(fig_efficient_frontier_max_sharpe, format="png")

            # Get optimized weights by Maximizing Sharpe
            ef_max_sharpe = EfficientFrontier(mu, S)
            ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
            weights_max_sharpe = ef_max_sharpe.clean_weights()
            expected_annual_return_max_sharpe, annual_volatility_max_sharpe, _ = ef_max_sharpe.portfolio_performance()
            sharpe_ratio_max_sharpe = expected_annual_return_max_sharpe / annual_volatility_max_sharpe
            weights_df_max_sharpe = pd.DataFrame.from_dict(weights_max_sharpe, orient='index')
            weights_df_max_sharpe.columns = ['Weights']

            # Calculate returns of portfolio with optimized weights by Maximizing Sharpe
            stocks_df['Optimized Portfolio (Max Sharpe)'] = 0
            for ticker, weight in weights_max_sharpe.items():
                stocks_df['Optimized Portfolio (Max Sharpe)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for max sharpe
            fig_cum_returns_optimized_max_sharpe = plot_cum_returns(stocks_df['Optimized Portfolio (Max Sharpe)'],
                                                                    'Cumulative Returns of Optimized Portfolio Starting with $100')

            # Calculate returns of optimized portfolio and comparison
            comparison_df['Optimized Portfolio (Max Sharpe)'] = 0
            for ticker, weight in weights_max_sharpe.items():
                comparison_df['Optimized Portfolio (Max Sharpe)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for max sharpe
            fig_cum_returns_optimized_max_sharpe_and_comparison = plot_cum_returns(comparison_df,
                                                                    'Cumulative Returns of Optimized Portfolio and Comparison Starting with $100')

            # 3d plot random portfolio for max sharpe
            max_sharpe_3d_plot = plot_random_3d_max_sharpe(mu, S, tickers)

            # Display on Streamlit
            with st.container():
                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized Max Sharpe Portfolio Weights")
                    st.dataframe(data=weights_df_max_sharpe, width=None, height=None, use_container_width=True)

                with contents_right:
                    st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))
                    st.plotly_chart(fig_cum_returns_optimized_max_sharpe)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized Max Sharpe Portfolio Performance")
                    st.image(fig_efficient_frontier_max_sharpe)

                with contents_right:
                    st.subheader("Random Portfolio Against Optimized Portfolio")
                    st.plotly_chart(max_sharpe_3d_plot)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader(
                        'Expected annual return: {}%'.format((expected_annual_return_max_sharpe * 100).round(2)))
                    st.subheader('Annual volatility: {}%'.format((annual_volatility_max_sharpe * 100).round(2)))
                    st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio_max_sharpe.round(2)))

                with contents_right:
                    st.plotly_chart(fig_corr)  # fig_corr is not a plotly chart

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price)

                with contents_right:
                    st.plotly_chart(fig_cum_returns)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price_and_comparison)

                with contents_right:
                    st.plotly_chart(fig_cum_returns_optimized_max_sharpe_and_comparison)

        elif option == "Minimum Volatility (Risk)":
            # Plot efficient frontier curve by Minimize Volatility
            fig_min_volatility = plot_efficient_frontier_and_min_volatility(mu, S)
            fig_efficient_frontier_min_volatility = BytesIO()
            fig_min_volatility.savefig(fig_efficient_frontier_min_volatility, format="png")

            # Get optimized weights by Minimizing Volatility
            ef_min_volatility = EfficientFrontier(mu, S)
            ef_min_volatility.min_volatility()
            weights_min_volatility = ef_min_volatility.clean_weights()
            expected_annual_return_min_volatility, annual_volatility_min_volatility, _ = ef_min_volatility.portfolio_performance()
            sharpe_ratio_min_volatility = expected_annual_return_min_volatility / annual_volatility_min_volatility
            weights_df_min_volatility = pd.DataFrame.from_dict(weights_min_volatility, orient='index')
            weights_df_min_volatility.columns = ['Weights']

            # Calculate returns of portfolio with optimized weights by Minimizing Volatility
            stocks_df['Optimized Portfolio (Min Volatility)'] = 0
            for ticker, weight in weights_min_volatility.items():
                stocks_df['Optimized Portfolio (Min Volatility)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for min volatility
            fig_cum_returns_optimized_min_volatility = plot_cum_returns(stocks_df['Optimized Portfolio (Min Volatility)'],
                                                                        'Cumulative Returns of Optimized Portfolio Starting with $100')

            # Calculate returns of optimized portfolio and comparison
            comparison_df['Optimized Portfolio (Min Volatility)'] = 0
            for ticker, weight in weights_min_volatility.items():
                comparison_df['Optimized Portfolio (Min Volatility)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for min volatility
            fig_cum_returns_optimized_min_volatility_and_comparison = plot_cum_returns(comparison_df,
                                                                                   'Cumulative Returns of Optimized Portfolio and Comparison Starting with $100')

            # 3d plot random portfolio for min volatility
            min_volatility_3d_plot = plot_random_3d_min_volatility(mu, S, tickers)

            # Display on Streamlit
            with st.container():
                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized Min Volatility Portfolio Weights")
                    st.dataframe(data=weights_df_min_volatility, width=None, height=None, use_container_width=True)

                with contents_right:
                    st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))
                    st.plotly_chart(fig_cum_returns_optimized_min_volatility)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized Min Volatility Portfolio Performance")
                    st.image(fig_efficient_frontier_min_volatility)

                with contents_right:
                    st.subheader("Random Portfolio Against Optimized Portfolio")
                    st.plotly_chart(min_volatility_3d_plot)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader(
                        'Expected annual return: {}%'.format((expected_annual_return_min_volatility * 100).round(2)))
                    st.subheader('Annual volatility: {}%'.format((annual_volatility_min_volatility * 100).round(2)))
                    st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio_min_volatility.round(2)))

                with contents_right:
                    st.plotly_chart(fig_corr)  # fig_corr is not a plotly chart

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price)

                with contents_right:
                    st.plotly_chart(fig_cum_returns)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price_and_comparison)

                with contents_right:
                    st.plotly_chart(fig_cum_returns_optimized_min_volatility_and_comparison)

        elif option == "Maximum Return":
            # Plot efficient frontier curve by Maximizing Return
            fig_efficient_risk = plot_efficient_frontier_and_efficient_risk(mu, S)
            fig_efficient_frontier_efficient_risk = BytesIO()
            fig_efficient_risk.savefig(fig_efficient_frontier_efficient_risk, format="png")

            # Get optimized weights by Maximizing Return
            ef_efficient_risk = EfficientFrontier(mu, S)
            ef_efficient_risk.efficient_risk(100)
            weights_efficient_risk = ef_efficient_risk.clean_weights()
            expected_annual_return_efficient_risk, annual_volatility_efficient_risk, _ = ef_efficient_risk.portfolio_performance()
            sharpe_ratio_efficient_risk = expected_annual_return_efficient_risk / annual_volatility_efficient_risk
            weights_df_efficient_risk = pd.DataFrame.from_dict(weights_efficient_risk, orient='index')
            weights_df_efficient_risk.columns = ['Weights']

            # Calculate returns of portfolio with optimized weights by Maximizing Return
            stocks_df['Optimized Portfolio (Max Return)'] = 0
            for ticker, weight in weights_efficient_risk.items():
                stocks_df['Optimized Portfolio (Max Return)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for max return
            fig_cum_returns_optimized_efficient_risk = plot_cum_returns(stocks_df['Optimized Portfolio (Max Return)'],
                                                                        'Cumulative Returns of Optimized Portfolio Starting with $100')

            # Calculate returns of optimized portfolio and comparison
            comparison_df['Optimized Portfolio (Max Return)'] = 0
            for ticker, weight in weights_efficient_risk.items():
                comparison_df['Optimized Portfolio (Max Return)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio and comparison for max return
            fig_cum_returns_optimized_efficient_risk_and_comparison = plot_cum_returns(comparison_df,
                                                                                   'Cumulative Returns of Optimized Portfolio and Comparison Starting with $100')

            # 3d plot random portfolio for max return
            efficient_risk_3d_plot = plot_random_3d_efficient_risk(mu, S, tickers)

            # Display on Streamlit
            with st.container():
                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized Max Return Portfolio Weights")
                    st.dataframe(data=weights_df_efficient_risk, width=None, height=None, use_container_width=True)

                with contents_right:
                    st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))
                    st.plotly_chart(fig_cum_returns_optimized_efficient_risk)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized Max Return Portfolio Performance")
                    st.image(fig_efficient_frontier_efficient_risk)

                with contents_right:
                    st.subheader("Random Portfolio Against Optimized Portfolio")
                    st.plotly_chart(efficient_risk_3d_plot)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader(
                        'Expected annual return: {}%'.format((expected_annual_return_efficient_risk * 100).round(2)))
                    st.subheader('Annual volatility: {}%'.format((annual_volatility_efficient_risk * 100).round(2)))
                    st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio_efficient_risk.round(2)))

                with contents_right:
                    st.plotly_chart(fig_corr)  # fig_corr is not a plotly chart

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price)

                with contents_right:
                    st.plotly_chart(fig_cum_returns)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price_and_comparison)

                with contents_right:
                    st.plotly_chart(fig_cum_returns_optimized_efficient_risk_and_comparison)

        elif option == "Set Maximum Volatility (Risk)":
            temp = EfficientFrontier(mu, S)
            temp.min_volatility()
            _, minimum_volatility, _ = temp.portfolio_performance()

            # Plot efficient frontier curve by Maximizing Return
            fig_efficient_risk = plot_efficient_frontier_and_target_volatility(mu, S, max_volatility)
            fig_efficient_frontier_efficient_risk = BytesIO()
            fig_efficient_risk.savefig(fig_efficient_frontier_efficient_risk, format="png")

            # Get optimized weights by Maximizing Return
            ef_efficient_risk = EfficientFrontier(mu, S)
            ef_efficient_risk.efficient_risk(max_volatility/100)
            weights_efficient_risk = ef_efficient_risk.clean_weights()
            expected_annual_return_efficient_risk, annual_volatility_efficient_risk, _ = ef_efficient_risk.portfolio_performance()
            sharpe_ratio_efficient_risk = expected_annual_return_efficient_risk / annual_volatility_efficient_risk
            weights_df_efficient_risk = pd.DataFrame.from_dict(weights_efficient_risk, orient='index')
            weights_df_efficient_risk.columns = ['Weights']

            # Calculate returns of portfolio with optimized weights by Maximizing Return
            stocks_df['Optimized Portfolio (Target Volatility)'] = 0
            for ticker, weight in weights_efficient_risk.items():
                stocks_df['Optimized Portfolio (Target Volatility)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for Target Volatility
            fig_cum_returns_optimized_efficient_risk = plot_cum_returns(stocks_df['Optimized Portfolio (Target Volatility)'],
                                                                        'Cumulative Returns of Optimized Portfolio Starting with $100')

            # Calculate returns of optimized portfolio and comparison
            comparison_df['Optimized Portfolio (Target Volatility)'] = 0
            for ticker, weight in weights_efficient_risk.items():
                comparison_df['Optimized Portfolio (Target Volatility)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for Target Volatility
            fig_cum_returns_optimized_efficient_risk_and_comparison = plot_cum_returns(comparison_df,
                                                                                       'Cumulative Returns of Optimized Portfolio and Comparison Starting with $100')

            # 3d plot random portfolio for Target Volatility
            efficient_risk_3d_plot = plot_random_3d_target_volatility(mu, S, tickers, max_volatility)

            # Display on Streamlit
            with st.container():
                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized Target Volatility Portfolio Weights")
                    st.dataframe(data=weights_df_efficient_risk, width=None, height=None, use_container_width=True)

                with contents_right:
                    st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))
                    st.plotly_chart(fig_cum_returns_optimized_efficient_risk)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized Target Volatility Portfolio Performance")
                    st.image(fig_efficient_frontier_efficient_risk)

                with contents_right:
                    st.subheader("Random Portfolio Against Optimized Portfolio")
                    st.plotly_chart(efficient_risk_3d_plot)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader(
                        'Expected annual return: {}%'.format((expected_annual_return_efficient_risk * 100).round(2)))
                    st.subheader('Annual volatility: {}%'.format((annual_volatility_efficient_risk * 100).round(2)))
                    st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio_efficient_risk.round(2)))

                with contents_right:
                    st.plotly_chart(fig_corr)  # fig_corr is not a plotly chart

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price)

                with contents_right:
                    st.plotly_chart(fig_cum_returns)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price_and_comparison)

                with contents_right:
                    st.plotly_chart(fig_cum_returns_optimized_efficient_risk_and_comparison)

        elif option == "Hierarchical Risk Parity":
            # Plot efficient frontier curve by Hierarchical Risk Parity
            fig_hierarchical_portfolio = plot_efficient_frontier_and_hierarchical_portfolio(stocks_return, S)
            fig_dendrogram = BytesIO()
            fig_hierarchical_portfolio.figure.savefig(fig_dendrogram, format="png")

            # Get optimized weights by Maximizing Sharpe
            ef_hierarchical_portfolio = HRPOpt(stocks_return, S)
            ef_hierarchical_portfolio.optimize()
            weights_hierarchical_portfolio = ef_hierarchical_portfolio.clean_weights()
            expected_annual_return_hierarchical_portfolio, annual_volatility_hierarchical_portfolio, _ = ef_hierarchical_portfolio.portfolio_performance()
            sharpe_ratio_hierarchical_portfolio = expected_annual_return_hierarchical_portfolio / annual_volatility_hierarchical_portfolio
            weights_df_hierarchical_portfolio = pd.DataFrame.from_dict(weights_hierarchical_portfolio, orient='index')
            weights_df_hierarchical_portfolio.columns = ['Weights']

            # Calculate returns of portfolio with optimized weights by Maximizing Sharpe
            stocks_df['Optimized Portfolio HRP'] = 0
            for ticker, weight in weights_hierarchical_portfolio.items():
                stocks_df['Optimized Portfolio HRP'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for max sharpe
            fig_cum_returns_optimized_hierarchical_portfolio = plot_cum_returns(stocks_df['Optimized Portfolio HRP'],
                                                                    'Cumulative Returns of Optimized Portfolio Starting with $100')

            # Calculate returns of optimized portfolio and comparison
            comparison_df['Optimized Portfolio (HRP)'] = 0
            for ticker, weight in weights_hierarchical_portfolio.items():
                comparison_df['Optimized Portfolio (HRP)'] += stocks_df[ticker] * weight

            # Plot Cumulative Returns of Optimized Portfolio for Target Volatility
            fig_cum_returns_optimized_hierarchical_portfolio_and_comparison = plot_cum_returns(comparison_df,
                                                                                       'Cumulative Returns of Optimized Portfolio and Comparison Starting with $100')

            # 3d plot random portfolio for max sharpe
            hierarchical_portfolio_3d_plot = plot_random_3d_hierarchical_portfolio(stocks_return,mu, S, tickers)

            # Display on Streamlit
            with st.container():
                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Optimized HRP Portfolio Weights")
                    st.dataframe(data=weights_df_hierarchical_portfolio, width=None, height=None, use_container_width=True)

                with contents_right:
                    st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))
                    st.plotly_chart(fig_cum_returns_optimized_hierarchical_portfolio)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader("Hierarchical Risk Parity Cluster Dendrogram")
                    st.image(fig_dendrogram)

                with contents_right:
                    st.subheader("Random Portfolio Against Optimized Portfolio")
                    st.plotly_chart(hierarchical_portfolio_3d_plot)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.subheader(
                        'Expected annual return: {}%'.format((expected_annual_return_hierarchical_portfolio * 100).round(2)))
                    st.subheader('Annual volatility: {}%'.format((annual_volatility_hierarchical_portfolio * 100).round(2)))
                    st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio_hierarchical_portfolio.round(2)))

                with contents_right:
                    st.plotly_chart(fig_corr)  # fig_corr is not a plotly chart

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price)

                with contents_right:
                    st.plotly_chart(fig_cum_returns)

                st.write("")  # Empty
                empty_left, contents_left, empty_middle, contents_right, empty_right = st.columns([0.1, 2, 0.1, 2, 0.1])

                with contents_left:
                    st.plotly_chart(fig_price_and_comparison)

                with contents_right:
                    st.plotly_chart(fig_cum_returns_optimized_hierarchical_portfolio_and_comparison)

    except:
        if tickers_string != "":
            if len(tickers) == 1 and tickers[0] != "":
                st.write(
                    'Enter AT LEAST 2 correct stock tickers to be included in portfolio separated by commas WITHOUT '
                    'spaces, ''e.g. "AMZN,GOOGL,AAPL" and click Generate Portfolio.')
            elif option == "Set Maximum Volatility (Risk)":
                if max_volatility > minimum_volatility:
                    st.write('The minimum volatility is {}. Please use a higher target volatility'.format(
                        (minimum_volatility * 100).round()))
                else:
                    st.write('Enter correct stock tickers to be included in portfolio separated by commas WITHOUT spaces, '
                        'e.g. "AMZN,GOOGL,AAPL" and click Generate Portfolio.')
            else:
                st.write('Enter correct stock tickers to be included in portfolio separated by commas WITHOUT spaces, '
                         'e.g. "AMZN,GOOGL,AAPL" and click Generate Portfolio.')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
