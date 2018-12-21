from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np


def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end',
            'Is_quarter_start']
    if time:
        attr = attr + ['Hour']
    for n in attr:
        df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop:
        df.drop(fldname, axis=1, inplace=True)


def preprocess(df):
    str_dtype, int_dtype, float_dtype = [], [], []

    for col in df.columns:
        if is_string_dtype(df[col]):
            str_dtype.append(col)
        elif is_numeric_dtype(df[col]):
            if str(df[col].dtypes)[:3] == 'int':
                int_dtype.append(col)
            else:
                float_dtype.append(col)

    for col in int_dtype:
        for num in [8, 16, 32, 64]:
            if np.mean(
                    df[col] == df[col].astype('int' + str(num))) == 1:
                df[col] = df[col].astype('int' + str(num))
                break

    for col in float_dtype:
        for num in [16, 32, 64]:
            if np.mean(df[col] == df[col].astype('float' + str(num))
                       ) == 1:
                if np.mean(df[col] == df[col].astype('int' + str(num))
                           ) == 1:
                    df[col] = df[col].astype('int' + str(num))
                else:
                    df[col] = df[col].astype('float' + str(num))
                break

    return df


def preprocess(df):
    str_dtype, int_dtype, float_dtype = [], [], []

    for col in df.columns:
        if is_string_dtype(df[col]):
            str_dtype.append(col)
        elif is_numeric_dtype(df[col]):
            if str(df[col].dtypes)[:3] == 'int':
                int_dtype.append(col)
            else:
                float_dtype.append(col)

    for col in int_dtype:
        for num in [8, 16, 32, 64]:
            if np.mean(df[col] == df[col].astype('int' + str(num))) == 1:
                df[col] = df[col].astype('int' + str(num))
                break

    for col in float_dtype:
        for num in [16, 32, 64]:
            if np.mean(df[col] == df[col].astype('float' + str(num))) == 1:
                if np.mean(df[col] == df[col].astype('int' + str(num))) == 1:
                    df[col] = df[col].astype('int' + str(num))
                else:
                    df[col] = df[col].astype('float' + str(num))
                break

    return df


def reset_index(df):
    df.reset_index(inplace=True)
    df.drop('index', 1, inplace=True)
    return df


def get_elapsed(df, fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    res = []

    for v, d in zip(df[fld].values, df['date'].values):
        if v != 0:
            last_date = d
        res.append(((d - last_date).astype('timedelta64[D]') / day1))

    df[pre + '_' + fld] = res
    return df
