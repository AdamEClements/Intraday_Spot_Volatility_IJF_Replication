import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb


def create_lstm_dataset(obs, timesteps):
    X = np.zeros((len(obs) - timesteps,timesteps))
    y = np.zeros((len(obs) - timesteps,1))
    for i in np.arange(0, len(obs) - timesteps, 1):
        X[i] = obs[i:i+timesteps]
        y[i] = obs[i+timesteps]
    return np.array(X), np.array(y)

# Read in data
temp_dats = pd.read_csv('IBM_ok.csv')
print(temp_dats.columns)
ok5 = temp_dats['ok5_all'].to_numpy()

# No. of periods/day and ndays
npers = 78
ndays = len(ok5)/npers

# Choose AR lags in form of days
nlags = 22*npers

# Initial window and no. of forecasts
initwin = 5700
nfore = int(ndays - initwin)

i = 1

ok5_tmp = ok5[i * npers:(initwin + i) * npers]

X, y = create_lstm_dataset(ok5_tmp, nlags)
gbm_model = lgb.LGBMRegressor()
gbm_fit = gbm_model.fit(X, y)
gbm_pred = gbm_fit.predict(X)

# Reshape forecasts and write output
y = y.flatten()
tmp_out = {'y': y, 'fit': gbm_pred}
out_df = pd.DataFrame(tmp_out)
out_df.to_csv('IBM_gbm_resid.csv', index=False)