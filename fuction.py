# %%
import pandas as pd
import numpy as np


def ts_sue(df,days):
    yoy=(df-df.shift(days))
    sue= (yoy-yoy.rolling(days).mean())/(yoy.rolling(days).std())


def bin_Neutralization(Y: pd.DataFrame, *X_list: pd.DataFrame):
    def get_rsid(X: np.ndarray, Y: np.ndarray):
        def get_beta(X: np.ndarray, Y: np.ndarray):
            # 计算回归系数
            coefficients = np.linalg.pinv(X.T @ X) @ X.T @ Y
            return coefficients

        coefficients = get_beta(X, Y)
        predicted_Y = X @ coefficients
        rsid = Y - predicted_Y
        return rsid

    Y_stack = Y.stack()
    Y_stack.index.names = ['date', 'order_book_id']
    X = pd.concat({i: X_list[i] for i in range(len(X_list))}, axis=1).stack(dropna=False)
    X.index.names = ['date', 'order_book_id']
    
    neutralized_df = (
        pd.concat([Y_stack, X], axis=1)
        .dropna()
        .groupby('date')
        .apply(lambda data: pd.Series(get_rsid(data.iloc[:, 1:].values, data.iloc[:, 0].values), 
                                    index=data.index.get_level_values('order_book_id')))
    )
    neutralized_df.index.names = [None, 'order_book_id']
    return neutralized_df.unstack().reindex_like(Y)


###########################################

def ts_min_max_ratio(df,day):
    return (df.rolling(day).min()/df.rolling(day).max())-1

def ts_sharpe(df,day):
    return (df.rolling(day).mean()/df.rolling(day).std())

def ts_max_min_ratio(df,day):
    return (df.rolling(day).max()/df.rolling(day).min())-1

def cs_reciprocal(df):
    return 1/df.replace(0, np.nan)


##############################################


def resample_to_period(df, period, index_data):
    df = df.copy()  # 避免修改原始数据
    df['datetime'] = df.index  # 备份索引
    df = df.groupby(df.index.to_period(period)).apply(lambda x: x.iloc[-1])  # 取周期内最后一行
    df.index = df['datetime']  # 恢复原索引
    df = df.drop(columns=['datetime'])  # 删除临时列
    df = df.reindex(index_data.index, method='ffill')  # 用 `index_data` 的索引重采样并前向填充
    return df


def cs_clip(data):
    return data.clip(upper=3)


def cs_cdf(data):
    from scipy.stats import norm

    def row_cdf(row):
        std = row.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.5, index=row.index)  # 若無變異，整列給 0.5
        z = (row - row.mean()) / std
        return pd.Series(norm.cdf(z), index=row.index)

    result = data.apply(row_cdf, axis=1)
    result.index = data.index  # 確保 index 沒變
    result.columns = data.columns  # 確保 columns 沒變
    return result

# 絕對值
def cs_abs(x):
    return x.abs() #if isinstance(x, pd.DataFrame) else abs(x)

# 加法
def bin_add(x, y):
    return x + y

# 除法，避免除以0
def bin_divide(x, y):
    return x / (y.replace(0, np.nan)) #if isinstance(y, pd.DataFrame) else x / y if y != 0 else np.nan

# 乘法
def bin_mul(x, y):
    return x * y

# 次方
def bin_power(x, y):
    return x ** y

# 相反數
def cs_reversed_val(x):
    return -1*x

# 符號函數：正為1，負為-1，0為0
def cs_sign(x):
    return np.sign(x)

def bin_max(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.maximum(x.values, y.values), index=x.index, columns=x.columns)

def bin_min(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.minimum(x.values, y.values), index=x.index, columns=x.columns)

def bin_sign_power(x,y):
    return np.sign(x**y)

def cs_sqrt(x):
    return x**0.5

def bin_sub(x,y):
    return x-y

# ts_arg_max
def ts_arg_max(x, d):
    return x.rolling(window=d, min_periods=1).apply(lambda s: (len(s)-1) - np.argmax(s.values[::-1]))

# ts_arg_min
def ts_arg_min(x, d):
    return x.rolling(window=d, min_periods=1).apply(lambda s: (len(s)-1) - np.argmin(s.values[::-1]))

# ts_av_diff
def ts_av_diff(x, d):
    return x - x.rolling(window=d, min_periods=1).mean()

# ts_corr
def tsbin_corr(x, y, d):
    return x.rolling(window=d).corr(y)

# ts_count_nans
def ts_count_nans(x, d):
    return x.rolling(window=d).apply(lambda s: s.isna().sum())

# ts_covariance
def tsbin_covariance(y, x, d):
    return y.rolling(window=d).cov(x)

# ts_delay
def ts_delay(x, d):
    return x.shift(d)

# ts_delta
def ts_delta(x, d):
    return x - x.shift(d)

# ts_mean
def ts_mean(x, d):
    return x.rolling(window=d, min_periods=1).mean()

# ts_product
def ts_product(x, d):
    return x.rolling(window=d, min_periods=1).apply(np.prod)

# ts_rank
def ts_rank(x, d ):
    return x.rolling(window=d, min_periods=1).rank(ascending=True,pct=True)

# ts_scale
def ts_scale(x, d, constant=0):
    min_ = x.rolling(window=d, min_periods=1).min()
    max_ = x.rolling(window=d, min_periods=1).max()
    return (x - min_) / (max_ - min_ + 1e-9) + constant

# ts_std_dev
def ts_std_dev(x, d):
    return x.rolling(window=d, min_periods=1).std()

# ts_sum
def ts_sum(x, d):
    return x.rolling(window=d, min_periods=1).sum()

# ts_zscore
def ts_zscore(x, d):
    mean = x.rolling(window=d, min_periods=1).mean()
    std = x.rolling(window=d, min_periods=1).std()
    return (x - mean) / (std + 1e-9)


def cs_rank(x):
    return x.rank(axis=1, pct=True)


def cs_zscore(x):
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return x.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


#####有夠慢

# 距上一次改变的天数
def days_from_last_change(x):
    out = x.copy() * 0
    for col in x.columns:
        series = x[col]
        last_value = None
        counter = 0
        temp = []
        for val in series:
            if pd.isna(val):
                counter += 1
            elif val != last_value:
                counter = 0
                last_value = val
            else:
                counter += 1
            temp.append(counter)
        out[col] = temp
    return out

# hump：限制变化幅度（平滑处理）
def hump(x, hump=0.01):
    return x.clip(lower=-hump, upper=hump)

# kth_element：回溯第k大
def kth_element(x, d, k):
    return x.rolling(window=d, min_periods=1).apply(lambda s: np.sort(s.dropna())[-k] if len(s.dropna()) >= k else np.nan)

# last_diff_value：返回与当前值不同的最后一个值
def last_diff_value(x, d):
    result = x.copy()
    for col in x.columns:
        result[col] = x[col].rolling(window=d+1).apply(lambda s: next((val for val in reversed(s[:-1]) if val != s.iloc[-1]), np.nan), raw=False)
    return result

def ts_decay_linear(x, d, dense=False):
    weights = np.arange(1, d + 1)

    def decay(s):
        if not dense:
            s = s.fillna(0)
        if s.isna().all():
            return np.nan
        w = weights[-len(s):]  
        return np.dot(s, w) / w.sum()

    return x.rolling(window=d, min_periods=1).apply(decay, raw=False)

def cs_scale(x, scale=1, longscale=1, shortscale=1):
    result = x.copy()
    long_mask = x > 0
    short_mask = x < 0
    result[long_mask] = result[long_mask] / result[long_mask].sum(axis=1).replace(0, np.nan) * longscale
    result[short_mask] = result[short_mask] / result[short_mask].abs().sum(axis=1).replace(0, np.nan) * shortscale
    return result * scale


def cs_winsorize(x, std=4):
    mean = x.mean(axis=1)
    std_dev = x.std(axis=1)
    lower = mean - std * std_dev
    upper = mean + std * std_dev
    return x.clip(lower=lower, upper=upper, axis=0)

# ts_quantile
def ts_quantile(x, d, driver="gaussian"):
    from scipy.stats import norm, uniform, cauchy
    dist_map = {
        "gaussian": norm,
        "uniform": uniform,
        "cauchy": cauchy
    }
    dist = dist_map.get(driver, norm)
    return x.rolling(window=d, min_periods=1).apply(lambda s: dist.ppf((s.rank().iloc[-1] - 1) / (len(s.dropna()) - 1)) if len(s.dropna()) > 1 else np.nan)