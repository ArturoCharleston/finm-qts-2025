# My functions
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.regression.rolling import RollingOLS


def calculate_return_metrics(df, adj=12, adjusted = True, quantile = 0.05):
    """
    Calculate return metrics for a given dataset (DataFrame or Series).

    Args:
        data: pandas DataFrame or pandas Series
        adj: int, default 12

    Returns:
        results_df: pandas DataFrame
    """

    results_df = pd.DataFrame(index=df.columns)
    if adjusted == True:
        results_df['Annualized Return'] = df.mean() * adj
        results_df['Annualized Volatility'] = df.std() * np.sqrt(adj)
    else:
        results_df['Annualized Return'] = df.mean()
        results_df['Annualized Volatility'] = df.std()

    # This works if you are calculating excess returns
    results_df['Sharpe Ratio'] = results_df['Annualized Return'] / results_df['Annualized Volatility']

    # Include skewness
    results_df['Skewness'] = df.skew()
    # Include Value at Risk
    results_df[f"VaR ({quantile})"] = df.quantile(quantile, axis=0)

    wealth_index = 1000 * (1 + df).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    results_df["Max Drawdown"] = drawdowns.min()

    # Include Kurtosis
    results_df["Excess Kurtosis"] = df.kurtosis()
    
    # Handling Sortino Ratio: avoid dividing by zero
    downside_std = df[df < 0].std()
    results_df['Annualized Sortino Ratio'] = results_df['Annualized Return'] / (downside_std * np.sqrt(adj)) if not downside_std.empty else np.nan

    return results_df



def calc_risk_metrics(data, var=0.05):
    """
    Calculate risk metrics for a DataFrame of assets.

    Args:
        data (pd.DataFrame): DataFrame of asset returns.
        var (float, optional): VaR level. Defaults to 0.05.

    Returns:
        Union[dict, DataFrame]: Dict or DataFrame of risk metrics.
    """
    summary = dict()
    summary["Skewness"] = data.skew()
    summary["Excess Kurtosis"] = data.kurtosis()
    summary[f"VaR ({var})"] = data.quantile(var, axis=0)
    summary[f"CVaR ({var})"] = data[data <= data.quantile(var, axis=0)].mean()
    summary["Min"] = data.min()
    summary["Max"] = data.max()
    summary['VaR per Vol'] = summary[f"VaR ({var})"]/data.std()

    wealth_index = 1000 * (1 + data).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    summary["Max Drawdown"] = drawdowns.min()
    summary['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
    summary["MDD Bottom"] = drawdowns.idxmin()

    recovery_date = []
    peak_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][: drawdowns[col].idxmin()].max()
        peak_date.append(previous_peaks[col][:drawdowns[col].idxmin()].idxmax())
        recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin() :]]).T
        recovery_date.append(
            recovery_wealth[recovery_wealth[col] >= prev_max].index.min()
        )
    summary["Recovery"] = ["-" if pd.isnull(i) else i for i in recovery_date]
    summary['MDD Peak'] = peak_date

    summary["Duration (days)"] = [
        (i - j).days if isinstance(i, pd.Timestamp) and isinstance(j, pd.Timestamp) else "-"
        for i, j in zip(summary["Recovery"], summary["MDD Bottom"])
    ]

    return pd.DataFrame(summary, index=data.columns)

# You can use this for Linear Factor Pricing Models as you already included for every variable.
def calc_performance_stats_regressions(df, market, risk_free_rate=0, adj=12, intercept=True, save_residuals=False, save_predicted=False):
    # Prepare the DataFrame for results
    performance = pd.DataFrame(columns=['Alpha'] + [f'Beta_{col}' for col in market.columns] + ['Treynor Ratio', 'Information Ratio', 'Tracking Error'])
    residuals = pd.DataFrame(index=df.index) if save_residuals else None
    predicted = pd.DataFrame(index=df.index) if save_predicted else None

    # Define a function to apply regression analysis
    def calculate_stats(series):
        if intercept:
            X = sm.add_constant(market)  # Add constant for intercept
        else:
            X = market
            
        model = sm.OLS(series, X, missing='drop').fit()  # Fit the model

        alpha = (model.params.iloc[0] if intercept else 0) * adj
        betas = model.params.iloc[1:] if intercept else model.params

        # Calculate performance metrics
        treynor_ratio = adj * (series.mean() - risk_free_rate) / betas.iloc[0] if betas.iloc[0] != 0 else np.nan
        tracking_error = (model.resid.std()) * np.sqrt(adj)
        information_ratio = (alpha / tracking_error) if tracking_error != 0 else np.nan
        sortino_ratio = np.sqrt(adj) * series.mean() / series[series < 0].std()
        r_squared = model.rsquared 

        if save_residuals:
            residuals[series.name] = model.resid

        if save_predicted:
            predicted[series.name] = model.predict(X)

        return pd.Series({
            'Alpha': alpha,
            **{f'Beta_{col}': beta for col, beta in zip(market.columns, betas)},
            'Sortino Ratio': sortino_ratio,
            'Treynor Ratio': treynor_ratio,
            'Information Ratio': information_ratio,
            'Tracking Error': tracking_error,
            'R-Squared': r_squared
        })

    # Apply the regression calculation to all numerical columns in the DataFrame
    performance = df.select_dtypes(include=np.number).apply(calculate_stats, axis=0).T

    # Return only the specified output
    if save_residuals:
        return residuals
    if save_predicted:
        return predicted
    return performance
    

def cs_performance(portfolio_df, factors_df, risk_free_rate=0, adj=12, intercept=True, save_residuals=False, stats = False, predicted_premiums = False):
    # Step 1: Calculate performance stats for each portfolio using calc_performance_stats_regressions
    ts_stats = calc_performance_stats_regressions(
        df=portfolio_df, 
        market=factors_df, 
        risk_free_rate=risk_free_rate, 
        adj=adj, 
        intercept=intercept, 
        save_residuals=save_residuals
    )
    
    # Filter betas and calculate the mean return for each portfolio
    betas = ts_stats.filter(like='Beta')
    portfolio_means = pd.DataFrame(portfolio_df.mean() * adj, columns=['Mean Return'])
    
    # Step 2: Prepare the regression model based on the intercept parameter
    if intercept:
        X = sm.add_constant(betas)
        column_names = ['Alpha'] + [f'Beta_{col}' for col in betas.columns] + ['R-Squared']
    else:
        X = betas
        column_names = [f'Beta_{col}' for col in betas.columns] + ['R-Squared']
    
    # Fit the OLS regression
    ols_model = sm.OLS(portfolio_means, X).fit()
    alpha = ols_model.params.get('const', None)  # Handle the absence of an intercept
    params = ols_model.params.drop('const', errors='ignore')  # Avoid errors if 'const' doesn't exist
    
    # Step 3: Compile the results DataFrame
    results_df = pd.DataFrame(columns=column_names)
    results_values = ([alpha] if alpha is not None else []) + list(params) + [ols_model.rsquared]
    results_df.loc['Portfolio Mean Return'] = results_values

    if stats:
        # Time Series (TS) MAE and Annualized MAE
        ts_mae = ts_stats['Alpha'].abs().mean()
        ts_annualized_mae = (ts_stats['Alpha'] * adj).abs().mean()
        ts_r_squared = ts_stats['R-Squared'].mean()
        
        # Cross Sectional (CS) MAE and Annualized MAE
        cs_mae = ols_model.resid.abs().mean()
        cs_annualized_mae = cs_mae * adj
        cs_r_squared = ols_model.rsquared
        
        # Factor Premia for TS (mean of factors, annualized) and CS (regression coefficients)
        ts_premia = factors_df.mean().values
        ts_annualized_premia = (factors_df.mean() * adj).values
        cs_premia = params.values
        cs_annualized_premia = (params * adj).values

        # Compile TS and CS statistics into a single DataFrame with TS before CS
        stats_df = pd.DataFrame({
            'MAE': [ts_mae, cs_mae],
            'Annualized MAE': [ts_annualized_mae, cs_annualized_mae],
            'R-Squared': [ts_r_squared, cs_r_squared],
            **{f'{col} Premia': [ts_premia[i], cs_premia[i]] for i, col in enumerate(factors_df.columns)},
            **{f'{col} Annualized Premia': [ts_annualized_premia[i], cs_annualized_premia[i]] for i, col in enumerate(factors_df.columns)}
        }, index=['TS', 'CS'])
        
        return stats_df
    

    if predicted_premiums:
        # Ensure factors_df has the same columns as betas
        ts_premia = factors_df.mean() * adj
        # Rename the indices of ts_premia to match the columns of betas
        ts_premia.index = betas.columns
        # Time Series (TS) Predicted Premiums
        ts_predicted_premium = (ts_premia * betas).sum(axis=1)

        # Cross Sectional (CS) Predicted Premiums
        cs_predicted_premium_no_intercept = betas.dot(params)
        if intercept:
            cs_predicted_premium_with_intercept = alpha + cs_predicted_premium_no_intercept
        else:
            cs_predicted_premium_with_intercept = cs_predicted_premium_no_intercept  # No intercept
        
        # Compile Predicted Premiums into DataFrame
        predicted_premium_df = pd.DataFrame({
            'TS Predicted Premium': ts_predicted_premium,
            'CS Predicted Premium (No Intercept)': cs_predicted_premium_no_intercept
        }, index=portfolio_df.columns)
        if intercept:
            predicted_premium_df['CS Predicted Premium (With Intercept)'] = cs_predicted_premium_with_intercept
        
        return predicted_premium_df
    
    return results_df


# NOTE: USE A TIME SERIES FOR ASSET, a dataframe for signals. No need to shift, that is done in the function. But shift if doing one regression

def oos_forecast(signals, asset, t = 60, rolling = False, roll_exp = False, intercept = True):
    
    '''
    Computes an out-of-sample forecast based on expanding regression periods
    
    signals: DataFrame containing the signals (regressors) to be used in each regression
    asset: DataFrame containing the values (returns) of the asset being predicted
    t: The minimum number of periods
    rolling: False if expanding, else enter an integer window
    roll_exp: If using rolling, indicate whether to use expanding up to the minimum periods 
    intercept: Boolean indicating the inclusion of an intercept in the regressions
    '''
    
    n = len(signals)
    
    # Convert asset to a Series if it's a DataFrame with a single column
    if isinstance(asset, pd.DataFrame):
        if asset.shape[1] == 1:
            asset = asset.iloc[:, 0]
        else:
            raise ValueError('Asset DataFrame must have exactly one column.')
    elif not isinstance(asset, pd.Series):
        raise TypeError('Asset must be a pandas Series or a single-column DataFrame.')
    
    
    if intercept:
        signals = sm.add_constant(signals)
    
    if t > n:
        
        raise ValueError('Min. periods (t) greater than number of data points')
    
    output = pd.DataFrame(index = signals.index, columns = ['Actual','Predicted','Null'])
    
    # If expanding
    if not rolling:
        
        for i in range(t,n):

            y = asset.iloc[:i]
            x = signals.iloc[:i].shift()

            if intercept:
                null_pred = y.mean()

            else:
                null_pred = 0

            model = sm.OLS(y,x,missing='drop').fit()

            pred_x = signals.iloc[[i - 1]]
            pred = model.predict(pred_x)[0]

            output.iloc[i]['Actual'] = asset.iloc[i]
            output.iloc[i]['Predicted'] = pred
            output.iloc[i]['Null'] = null_pred
    
    # If rolling
    else:
        
        if rolling > n:
            
            raise ValueError('Rolling window greater than number of data points')
        
        y = asset
        x = signals.shift()
        
        if intercept:
            
            if roll_exp:
                null_pred = y.rolling(window = rolling, min_periods = 0).mean().shift()
            else:
                null_pred = y.rolling(window = rolling).mean().shift()

        else:
            null_pred = 0
        
        # When expanding == True, there is a minimum number of observations
        # Keep ^ in mind
        model = RollingOLS(y,x,window = rolling, expanding = roll_exp).fit()

        output['Actual'] = asset
        output['Predicted'] = (model.params * signals).dropna().sum(axis=1).shift()
        output['Null'] = null_pred
        
        
    return output


def oos_r_squared(data):
    
    '''
    Computes the out-of-sample r squared
    data: DataFrame containing actual, model-predicted, and null-predicted values
    '''
    
    model_error = data['Actual'] - data['Predicted']
    null_error = data['Actual'] - data['Null']
    
    r2_oos = 1 - (model_error ** 2).sum() / (null_error ** 2).sum()
    
    return r2_oos


def probability_greater_than_z(X, Z = 0, adj = 12, years=5):
    """
    Calculate the probability that a random variable X is greater than Z, 
    with optional period adjustment and annualization.

    Parameters:
    X (pd.Series or np.ndarray): The variable for which to calculate the probability.
    Z (float): The threshold value.
    period (int): The time period for calculating probability (e.g., 60 for 60 days, 12 for annual).
    annualize (bool): Whether to annualize the mean and standard deviation.

    Returns:
    float: Probability of X being greater than Z.
    """
    from scipy.stats import norm
    # Calculate the mean and standard deviation
    mu = X.mean()*adj
    sigma = X.std()*np.sqrt(adj)
    
    # Calculate the probability
    prob = norm.cdf(Z, mu, sigma / np.sqrt(years))
    return prob

# Define the function
def prob_larger_than(mu, sigma, h, threshold):
    from scipy.stats import norm
    z = np.sqrt(h) * (threshold - mu) / sigma
    return 1 - norm.cdf(z)

# NOTE: Use this according to your needs (+- , or 1- depending on < or >). 
def prob(mu, sigma, h):
    return norm.cdf(np.sqrt(h)*mu/sigma)


def max_min_corr(correlation_matrix):
    # Print the max and minimum correlations
    # Calculate the pair with the highest correlation and minimum correlation
    # Unstack the matrix to get pairwise correlations
    pairwise_correlations = correlation_matrix.unstack()

    # Remove self-correlations
    pairwise_correlations = pairwise_correlations[pairwise_correlations != 1]

    # Find the maximum and minimum correlation pairs
    max_correlation_pair = pairwise_correlations.idxmax()
    min_correlation_pair = pairwise_correlations.idxmin()

    # Get the maximum and minimum correlation values
    max_correlation_value = pairwise_correlations.max()
    min_correlation_value = pairwise_correlations.min()

    print(f"Highest Correlation Pair: {max_correlation_pair}, Value: {max_correlation_value}")
    print(f"Minimum Correlation Pair: {min_correlation_pair}, Value: {min_correlation_value}")


def calc_regression_summary(y, x, include_constant=True, adj_factor=12):
    """
    Perform OLS regression and return a summary of the results and some metrics.

    Args:
        y (pd.Series): Target variable.
        x (pd.DataFrame): Predictor variables.
        include_constant (bool, optional): Whether to include a constant term in the regression. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing the regression summary including Alpha, R-squared, Betas, Information Ratio, Tracking Error, Correlation, and Mean Fitted.
    """
    import statsmodels.api as sm

    if include_constant:
        x = sm.add_constant(x)
    
    # Fit the OLS model
    ols = sm.OLS(y, x, missing='drop').fit()
    
    # Extract results
    intercept = adj_factor*ols.params.iloc[0] if include_constant else 0
    betas = ols.params.iloc[1:] if include_constant else ols.params
    r2 = ols.rsquared
    tracking_error = ols.resid.std() * np.sqrt(adj_factor)
    information_ratio = (intercept / tracking_error) * np.sqrt(adj_factor) if tracking_error != 0 else np.nan
    correlation = y.corr(ols.fittedvalues)
    mean_fitted = ols.fittedvalues.mean() * adj_factor
   
    # Create a DataFrame for all the requested information
    summary_df = pd.DataFrame({
        'Value': [intercept, r2] + betas.tolist() + [information_ratio, tracking_error, correlation, mean_fitted]
    }, index=['Alpha', 'R-squared'] + [f"{col} Beta" for col in betas.index.tolist()] + ['Information Ratio', 'Tracking Error', 'Correlation', 'Mean fitted'])
    
    return summary_df




def calc_rolling_regression(X, Y, window_size=60):
    """
    Perform rolling window regression to replicate a target series using specified factors.

    Args:
        X (pd.DataFrame): DataFrame containing the factor series.
        Y (pd.Series): Series containing the target series to be replicated.
        window_size (int, optional): Size of the rolling window. Defaults to 60.

    Returns:
        pd.DataFrame: DataFrame containing the actual and replicated values.
    """
    from collections import defaultdict
    import statsmodels.api as sm
    import pandas as pd
    import numpy as np

    summary = defaultdict(list)

    for idx in range(window_size, len(Y)):
        X_window = X.iloc[idx-window_size:idx]
        Y_window = Y.iloc[idx-window_size:idx]

        oos_Y = Y.iloc[idx]
        oos_X = X.iloc[idx]

        reg = sm.OLS(Y_window, sm.add_constant(X_window), missing='drop').fit()

        # Save parameters
        coeffs = reg.params

        # Predict the out-of-sample value
        y_pred = coeffs.iloc[0] + (coeffs.iloc[1:] @ oos_X)  # Intercept + (Beta * X)

        summary['Replicated'].append(y_pred)
        summary['Actual'].append(oos_Y)

    summary = pd.DataFrame(summary, index=Y.index[window_size:])

    # Calculate OOS R-Squared
    oos_rsquared = (
        1 - (summary["Actual"] - summary["Replicated"]).var() / summary["Actual"].var()
    )

    return summary, oos_rsquared






# # Define starting index with respect to start_date
# min_periods = df.index.get_loc(start_date) - 1


# def calc_historic_var(excess_returns, m_window=252, window=60, theta=0.94, initial_vol=0.20 / np.sqrt(252), quantile = 0.05, mu = 0, zscore = None, normal_aprox = True):
#     """
#     Calculate expanding, rolling, and EWMA volatility for a DataFrame of returns and compare them to excess returns.

#     Args:
#         excess_returns (pd.DataFrame): DataFrame containing the returns and excess returns.
#         m_window (int, optional): Window size for rolling volatility. Defaults to 252.
#         window (int, optional): Window size for rolling volatility. Defaults to 60.
#         theta (float, optional): Smoothing parameter for EWMA. Defaults to 0.94.
#         initial_vol (float, optional): Initial volatility for EWMA. Defaults to 0.20 / sqrt(252).

#     Returns:
#         pd.DataFrame: DataFrame containing the calculated volatilities and their comparison to excess returns.
#     """
#     from scipy.stats import norm
#     if zscore is None:
#         zscore = norm.ppf(quantile)
#     else:
#         zscore = zscore

#     expanding_window = np.sqrt((excess_returns**2).expanding(window).mean().shift())
#     expanding_window = expanding_window[window:]
#     rolling_window = np.sqrt((excess_returns**2).rolling(m_window).mean().shift())
#     rolling_window = rolling_window[window:]

#     def ewma_vol(returns, theta=0.94, initial_vol=0.20/np.sqrt(252)):
#         sigma2_t_ewma = initial_vol ** 2
        
#         ewma_variances = [sigma2_t_ewma]
        
#         for r in returns:
#             sigma2_t_ewma = theta * sigma2_t_ewma + (1 - theta) * r ** 2
#             ewma_variances.append(sigma2_t_ewma)
        
#         ewma_volatilities = np.sqrt(ewma_variances)
        
#         return ewma_volatilities

#     ewma = ewma_vol(excess_returns, theta, initial_vol)
#     ewma = ewma[window+1:]

#     std = pd.DataFrame({'expanding_window': expanding_window, 'rolling_window': rolling_window, 'EWMA': ewma})

#     if normal_aprox:
#         VaR = mu + zscore * std
#         CVaR = mu - norm.pdf(zscore)/quantile * std

#     # Calculate frequency table
#     hit_ratio = std.apply(lambda x: excess_returns.loc[x.index] < x).mean().to_frame('Hit Ratio')

#     return std, hit_ratio


# NOTE: For rolling window, use it as time series; for normal calculation, use dataframe. 
# Also note that you might have to adjust one shift to make it fit the one period ahead.

def calc_var(df, q=0.05, normal_aprox=True, alt_zscore=None, mean=None, window=None):
    """
    Calculate the VaR and CVaR for a DataFrame of returns, either for the entire dataset or rolling.

    Args:
        df (pd.DataFrame): DataFrame of returns.
        q (float, optional): Quantile for VaR and CVaR. Defaults to 0.05.
        normal_aprox (bool, optional): Whether to use normal approximation for VaR/CVaR. Defaults to True.
        alt_zscore (float, optional): Alternative z-score value. Defaults to None.
        mean (float, optional): Mean value for returns. Defaults to None.
        window (int, optional): Rolling window size. If None, it calculates without rolling. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the VaR and CVaR values.
    """
    from scipy.stats import norm

    if alt_zscore is None:
        zscore = norm.ppf(q)
    else:
        zscore = alt_zscore

    if mean is None:
        mu = df.mean() if window is None else df.rolling(window).mean()
    else:
        mu = mean

    if normal_aprox:
        if window is None:
            VaR = mu + zscore * df.std()
            CVaR = mu - norm.pdf(zscore) / q * df.std()
        else:
            rolling_std = df.rolling(window).std()
            VaR = mu + zscore * rolling_std
            CVaR = mu - norm.pdf(zscore) / q * rolling_std
    else:
        if window is None:
            VaR = df.quantile(q)
            CVaR = df[df <= df.quantile(q)].mean()
        else:
            VaR = df.rolling(window).quantile(q)
            CVaR = df.rolling(window).apply(lambda x: x[x <= x.quantile(q)].mean(), raw=False)

    return pd.DataFrame({'VaR': VaR, 'CVaR': CVaR})

def expanding_var(return_series: pd.Series, percentile = 0.05, min_periods = 60) -> pd.Series:
    return return_series.expanding(min_periods=min_periods).quantile(percentile)

def historical_rolling_var(return_series: pd.Series, percentile = 0.05, window = 60) -> pd.Series:
    return return_series.rolling(window=window).quantile(percentile)

def calc_var_rolling(df, q=0.05, normal_approx=True, alt_zscore=None, mean=None, rolling=52):
    """
    Calculate the rolling VaR and CVaR for a DataFrame of returns.

    Args:
        df (pd.TimeSeries). Provide a time series not dataframe
        q (float, optional): Quantile for VaR and CVaR. Defaults to 0.05.
        normal_approx (bool, optional): Whether to use normal approximation for VaR/CVaR. Defaults to True.
        alt_zscore (float, optional): Alternative z-score value. Defaults to None.
        mean (float, optional): Mean value for returns. Defaults to None.
        rolling (int): Rolling window size.

    Returns:
        pd.DataFrame: DataFrame containing the rolling VaR and CVaR values.
    """
    from scipy.stats import norm

    if alt_zscore is None:
        zscore = norm.ppf(q)
    else:
        zscore = alt_zscore

    if mean is None:
        rolling_mean = df.rolling(rolling).mean().shift(1)
    else:
        rolling_mean = mean

    if normal_approx:
        rolling_std = df.rolling(rolling).std().shift(1)
        VaR = rolling_mean + zscore * rolling_std
        CVaR = rolling_mean - norm.pdf(zscore) / q * rolling_std
    else:
        VaR = df.rolling(rolling).quantile(q).shift(1)
        CVaR = df.rolling(rolling).apply(lambda x: x[x <= x.quantile(q)].mean(), raw=False)

    return pd.DataFrame({'VaR': VaR, 'CVaR': CVaR})


# NOTE: For hit ratio, use a time series
def calc_hit_ratio(df, var, q=0.05):
    """
    Calculate the hit ratio and hit ratio error based on returns and VaR.

    Args:
        df (pd.Series): Series of returns.
        var (pd.Series): Series of VaR values.
        q (float, optional): Quantile used for VaR calculation. Defaults to 0.05.

    Returns:
        pd.DataFrame: DataFrame containing the hit ratio and hit ratio error.
    """
    # Align the indices of df and var
    aligned_df, aligned_var = df.align(var, join='inner')

    # Calculate the hit ratio
    hit_ratio = (aligned_df < aligned_var).mean()

    # Calculate the hit ratio error
    hit_ratio_error = abs(hit_ratio / q - 1)

    # Create a DataFrame to return both metrics
    results = pd.DataFrame({
        'Hit Ratio': [hit_ratio],
        'Hit Ratio Error': [hit_ratio_error]
    })

    return results

    

def find_tangency_weights(df, regularization_cov=1, reg_diag = 1, adj_factor=12, expected_returns = None, add_stats = True, portfolio_performance = True):

    if regularization_cov == 1:
        cov_inv = np.linalg.inv(df.cov() * adj_factor)
    else:
        cov = df.cov()
        covdiag = np.diag(np.diag(cov))
        covsum = regularization_cov * (cov + covdiag * reg_diag)
        cov_inv = np.linalg.inv(covsum * adj_factor)

    if expected_returns is not None:
        mu = expected_returns * adj_factor # Remember to use not annualized returns
    else:
        mu = df.mean() * adj_factor
    
    ones = np.ones(df.columns.shape)
    scale = ones @ cov_inv @ mu
    sigmu = cov_inv @ mu
    weights = pd.DataFrame((1/scale) * sigmu, index=df.columns, columns=['Weights'])

    if add_stats:
        mean_returns = df.mean() * adj_factor
        volatility_returns = df.std() * np.sqrt(adj_factor)

        sharpes = mean_returns / volatility_returns

        # Combine the results into a single DataFrame
        annual_stats = pd.DataFrame({
            'Mean Annual Return': mean_returns,
            'Annual Volatility': volatility_returns,
            'Sharpe Ratio': sharpes
        })

        # Combine weights and annual_stats into a single DataFrame
        results = weights.join(annual_stats)
    else:
        results = weights

    if portfolio_performance:
        port_performance = calculate_return_metrics(df @ weights, adj=adj_factor, adjusted=True)
        port_performance.index = ['Tangency Portfolio']

    if portfolio_performance:
        return results, port_performance
    else:
        return results
        
# TO REESCALE (REMEMBER TO CLOSE THE DENOMINATOR PARENTHESIS)
# rescale these to hit the target mean
#wts *= TARG_MEAN / (retsx.mean()@wts)

# TO CALCULATE THE RETURNS ON THE PORFOLIO Multiply the values by their weights; each value. 
# multi_asset_etf@weights_df; df @ weights


def find_gmv_weights(df, adj_factor=12):
    
    # Step 1: Create a vector of ones for portfolio weights
    ones = np.ones(df.columns.shape)
    # Step 2: Calculate the inverse of the covariance matrix, adjusted by the adj_factor
    cov_inv = np.linalg.inv(df.cov() * adj_factor)
    # Step 3: Scaling factor for the weights
    scale = ones @ cov_inv @ ones
    # Step 4: Calculate GMV weights using the scaling factor and inverse covariance matrix
    gmv_tot = (1/scale) * cov_inv @ ones
    # Step 5: Create a DataFrame to store the GMV weights, indexed by the columns of df
    gmv_wts = pd.DataFrame(index=df.columns, data=gmv_tot, columns=['GMV Weights'])

    return gmv_wts


def mv_portfolio(tot_returns, target_ret, adj_factor = 12, return_delta = False):

    gmv_weights = find_tangency_weights(tot_returns, regularization_cov=1, reg_diag = 1, adj_factor=adj_factor, expected_returns = None, add_stats = False, portfolio_performance = False)
    tangency_weights = find_gmv_weights(tot_returns, adj_factor)

    # Obtain tangency weights using find_tangency_weights
    tangency_portfolio = (tot_returns @ tangency_weights)

    gmv_portfolio = (tot_returns @ gmv_weights)

    r_v = gmv_portfolio.mean().iloc[0]
    r_t = tangency_portfolio.mean().iloc[0]
    
    # Calculate delta: The weight factor between tangency and GMV portfolios
    delta = (target_ret - r_v) / (r_t - r_v)
    
    # Compute MV weights by combining tangency and GMV weightsxww
    mv_weights = delta * tangency_weights + (1 - delta) * gmv_weights

    # Ensure delta is within [0, 1]
    if not (0 <= delta <= 1):
        raise ValueError("Target return is out of achievable range with given portfolios.")
    
    gmv_weights = gmv_weights.reindex(tangency_weights.index)

    # Calculate the MV weights
    mv_weights = pd.DataFrame(
        index=tangency_weights.index,
        data=delta * tangency_weights.values + (1 - delta) * gmv_weights.values,
        columns=['MV Weights']
    )
    
    # Join all three DataFrames
    MV = mv_weights.join([gmv_weights.rename(columns={'GMV Weights': 'GMV Weights'}), 
                      tangency_weights.rename(columns={'Weights': 'Tangency Weights'})])

    # Return the final DataFrame
    if return_delta:
        return delta
    else:
        return MV


def calc_ewma_volatility(excess_returns: pd.Series, theta = 0.94, initial_vol = .005):
    var_t0 = initial_vol ** 2
    ewma_var = [var_t0]
    for i in range(len(excess_returns.index)):
        new_ewma_var = ewma_var[-1] * theta + (excess_returns.iloc[i] ** 2) * (1 - theta)
        ewma_var.append(new_ewma_var)
    ewma_var.pop(0) # Remove var_t0
    ewma_vol = [np.sqrt(v) for v in ewma_var]
    return pd.Series(ewma_vol, index=excess_returns.index)




############################################################################################################
# Midterm 2
############################################################################################################

# TO CALCULATE H TEST FOR CAPM

# T = portfolios_er_1981.shape[0]
# SR = np.sqrt(12)*(factors_1981[['Mkt-RF']].mean() / factors_1981[['Mkt-RF']].std())
# residuals = calc_performance_stats_regressions(portfolios_er_1981, factors_1981[['Mkt-RF']], adj=12, intercept=False, save_residuals=True)
# sigma = residuals.cov()
# sigma_inv = pd.DataFrame(np.linalg.inv(sigma), index=sigma.index, columns=sigma.columns)
# alpha = performance_capm['Alpha']

# H = T * (1 + SR**2)**(-1) * (alpha @ sigma_inv @ alpha)

# print('H = {:.2f}'.format(H.item()))
# pvalue = 1 - stats.chi2.cdf(H, df=25)
# print('p-value = {:.4f}'.format(pvalue.item()))


# CREATE A FUNCTION TO CALCULATE BOTH THE Time series and cross section tests
# This should create the different statistics, coefficients, R squared, Alpha, MAE. 


# def calc_var_cvar_summary(
#     returns: Union[pd.Series, pd.DataFrame],
#     quantile: Union[None, float] = .05,
#     window: Union[None, str] = None,
#     return_hit_ratio: bool = False,
#     filter_first_hit_ratio_date: Union[None, str, datetime.date] = None,
#     return_stats: Union[str, list] = ['Returns', 'VaR', 'CVaR', 'Vol'],
#     full_time_sample: bool = False,
#     z_score: float = None,
#     shift: int = 1,
#     normal_vol_formula: bool = False,
#     ewma_theta : float = .94,
#     ewma_initial_vol : float = .2 / np.sqrt(252),
#     garch_p: int = 1,
#     garch_q: int = 1,
#     keep_columns: Union[list, str] = None,
#     drop_columns: Union[list, str] = None,
#     keep_indexes: Union[list, str] = None,
#     drop_indexes: Union[list, str] = None,
#     drop_before_keep: bool = False,
# ):
#     """
#     Calculates a summary of VaR (Value at Risk) and CVaR (Conditional VaR) for the provided returns.

#     Parameters:
#     returns (pd.Series or pd.DataFrame): Time series of returns.
#     quantile (float or None, default=0.05): Quantile to calculate the VaR and CVaR.
#     window (str or None, default=None): Window size for rolling calculations.
#     return_hit_ratio (bool, default=False): If True, returns the hit ratio for the VaR.
#     return_stats (str or list, default=['Returns', 'VaR', 'CVaR', 'Vol']): Statistics to return in the summary.
#     full_time_sample (bool, default=False): If True, calculates using the full time sample.
#     z_score (float, default=None): Z-score for parametric VaR calculation.
#     shift (int, default=1): Period shift for VaR/CVaR calculations.
#     normal_vol_formula (bool, default=False): If True, uses the normal volatility formula.
#     keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
#     drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
#     keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
#     drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
#     drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.

#     Returns:
#     pd.DataFrame: Summary of VaR and CVaR statistics.
#     """
#     if window is None:
#         print('Using "window" of 60 periods, since none was specified')
#         window = 60
#     if isinstance(returns, pd.DataFrame):
#         returns_series = returns.iloc[:, 0]
#         returns_series.index = returns.index
#         returns = returns_series.copy()

#     summary = pd.DataFrame({})

#     # Returns
#     summary[f'Returns'] = returns

#     # VaR
#     summary[f'Expanding {window:.0f} Historical VaR ({quantile:.2%})'] = returns.expanding(min_periods=window).quantile(quantile)
#     summary[f'Rolling {window:.0f} Historical VaR ({quantile:.2%})'] = returns.rolling(window=window).quantile(quantile)
#     if normal_vol_formula:
#         summary[f'Expanding {window:.0f} Volatility'] = returns.expanding(window).std()
#         summary[f'Rolling {window:.0f} Volatility'] = returns.rolling(window).std()
#     else:
#         summary[f'Expanding {window:.0f} Volatility'] = np.sqrt((returns ** 2).expanding(window).mean())
#         summary[f'Rolling {window:.0f} Volatility'] = np.sqrt((returns ** 2).rolling(window).mean())
#     summary[f'EWMA {ewma_theta:.2f} Volatility'] = calc_ewma_volatility(returns, theta=ewma_theta, initial_vol=ewma_initial_vol)
#     summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility'] = calc_garch_volatility(returns, p=garch_p, q=garch_q)

#     z_score = norm.ppf(quantile) if z_score is None else z_score
#     summary[f'Expanding {window:.0f} Parametric VaR ({quantile:.2%})'] = summary[f'Expanding {window:.0f} Volatility'] * z_score
#     summary[f'Rolling {window:.0f} Parametric VaR ({quantile:.2%})'] = summary[f'Rolling {window:.0f} Volatility'] * z_score
#     summary[f'EWMA {ewma_theta:.2f} Parametric VaR ({quantile:.2%})'] = summary[f'EWMA {ewma_theta:.2f} Volatility'] * z_score
#     summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametric VaR ({quantile:.2%})'] = summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility'] * z_score

#     if return_hit_ratio:
#         shift_stats = [
#             f'Expanding {window:.0f} Historical VaR ({quantile:.2%})',
#             f'Rolling {window:.0f} Historical VaR ({quantile:.2%})',
#             f'Expanding {window:.0f} Parametric VaR ({quantile:.2%})',
#             f'Rolling {window:.0f} Parametric VaR ({quantile:.2%})',
#             f'EWMA {ewma_theta:.2f} Parametric VaR ({quantile:.2%})',
#             f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametric VaR ({quantile:.2%})'
#         ]
#         summary_shift = summary.copy()
#         summary_shift[shift_stats] = summary_shift[shift_stats].shift()
#         if filter_first_hit_ratio_date:
#             if isinstance(filter_first_hit_ratio_date, (datetime.date, datetime.datetime)):
#                 filter_first_hit_ratio_date = filter_first_hit_ratio_date.strftime("%Y-%m-%d")
#             summary_shift = summary_shift.loc[filter_first_hit_ratio_date:]
#         summary_shift = summary_shift.dropna(axis=0)
#         summary_shift[shift_stats] = summary_shift[shift_stats].apply(lambda x: (x - summary_shift['Returns']) > 0)
#         hit_ratio = pd.DataFrame(summary_shift[shift_stats].mean(), columns=['Hit Ratio'])
#         hit_ratio['Hit Ratio Error'] = (hit_ratio['Hit Ratio'] - quantile) / quantile
#         hit_ratio['Hit Ratio Absolute Error'] = abs(hit_ratio['Hit Ratio Error'])
#         hit_ratio = hit_ratio.sort_values('Hit Ratio Absolute Error')
#         return filter_columns_and_indexes(
#             hit_ratio,
#             keep_columns=keep_columns,
#             drop_columns=drop_columns,
#             keep_indexes=keep_indexes,
#             drop_indexes=drop_indexes,
#             drop_before_keep=drop_before_keep
#         )

#     # CVaR
#     summary[f'Expanding {window:.0f} Historical CVaR ({quantile:.2%})'] = returns.expanding(window).apply(lambda x: x[x < x.quantile(quantile)].mean())
#     summary[f'Rolling {window:.0f} Historical CVaR ({quantile:.2%})'] = returns.rolling(window).apply(lambda x: x[x < x.quantile(quantile)].mean())
#     summary[f'Expanding {window:.0f} Parametrical CVaR ({quantile:.2%})'] = - norm.pdf(z_score) / quantile * summary[f'Expanding {window:.0f} Volatility']
#     summary[f'Rolling {window:.0f} Parametrical CVaR ({quantile:.2%})'] = - norm.pdf(z_score) / quantile * summary[f'Rolling {window:.0f} Volatility']
#     summary[f'EWMA {ewma_theta:.2f} Parametrical CVaR ({quantile:.2%})'] = - norm.pdf(z_score) / quantile * summary[f'EWMA {ewma_theta:.2f} Volatility']
#     summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametrical CVaR ({quantile:.2%})'] = - norm.pdf(z_score) / quantile * summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility']

#     if shift > 0:
#         shift_columns = [c for c in summary.columns if not bool(re.search("returns", c))]
#         summary[shift_columns] = summary[shift_columns].shift(shift)
#         print(f'VaR and CVaR are given shifted by {shift:0f} period(s).')
#     else:
#         print('VaR and CVaR are given in-sample.')

#     if full_time_sample:
#         summary = summary.loc[:, lambda df: [c for c in df.columns if bool(re.search('expanding', c.lower()))]]
#     return_stats = [return_stats.lower()] if isinstance(return_stats, str) else [s.lower() for s in return_stats]
#     return_stats = list(map(lambda x: 'volatility' if x == 'vol' else x, return_stats))
#     if return_stats == ['all'] or set(return_stats) == set(['returns', 'var', 'cvar', 'volatility']):
#         return filter_columns_and_indexes(
#             summary,
#             keep_columns=keep_columns,
#             drop_columns=drop_columns,
#             keep_indexes=keep_indexes,
#             drop_indexes=drop_indexes,
#             drop_before_keep=drop_before_keep
#         )
#     return filter_columns_and_indexes(
#         summary.loc[:, lambda df: df.columns.map(lambda c: bool(re.search(r"\b" + r"\b|\b".join(return_stats) + r"\b", c.lower())))],
#         keep_columns=keep_columns,
#         drop_columns=drop_columns,
#         keep_indexes=keep_indexes,
#         drop_indexes=drop_indexes,
#         drop_before_keep=drop_before_keep
#     )



# DEPRECATED