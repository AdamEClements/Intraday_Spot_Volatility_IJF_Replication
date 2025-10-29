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

# Take logs
ok5[ok5 == 0] = np.mean(ok5)
ok5 = np.log(ok5)

# No. of periods/day and ndays
npers = 78
ndays = len(ok5)/npers

# Choose AR lags in form of days
nlags = 22*npers

# Initial window and no. of forecasts
initwin = 1000
nfore = int(ndays - initwin)

# Store mean forecasts and target
fore_gbm = np.zeros((nfore,npers))
target = np.zeros((nfore,npers))

# Grid for training at
train_grid = np.arange(0, nfore, 250)

# Forecasting loop
for i in np.arange(0, nfore, 1):
    ok5_tmp = ok5[i * npers:(initwin + i) * npers]
    target[i] = ok5[(initwin + i) * npers:(initwin + i + 1) * npers]

    # Create and train model only for times in train_grid
    if np.sum(train_grid == i) > 0:
        X, y = create_lstm_dataset(ok5_tmp, nlags)
        gbm_model = lgb.LGBMRegressor()
        gbm_fit = gbm_model.fit(X, y)
        fits = gbm_fit.predict(X[-100*npers:, ])
        evar = np.var(y[-100*npers:] - fits)



    # Generate forecasts
    tmp_input = np.concatenate((ok5_tmp[-nlags:], target[i]))
    tmp_input = np.reshape(tmp_input, (1,tmp_input.shape[0]))
    for j in np.arange(0, npers, 1):
        pred_input = np.reshape(tmp_input[0, j:j + nlags],(1,nlags))
        fore_gbm[i, j] = gbm_fit.predict(pred_input + 0.5*evar)
        fore_gbm[i, j] = np.exp(fore_gbm[i, j])

    print(i)

target = np.exp(target)

# Reshape forecasts and write output
fore_lstm_out = np.reshape(fore_gbm, (npers*nfore), order="C")
target_out = np.reshape(target, (npers*nfore), order="C")
tmp_out = {'target': target_out, 'forecast': fore_lstm_out}
out_df = pd.DataFrame(tmp_out)
out_df.to_csv('IBM_gbm_fore_log.csv', index=False)
