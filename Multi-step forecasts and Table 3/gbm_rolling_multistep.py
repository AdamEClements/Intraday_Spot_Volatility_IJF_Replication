import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb


def create_lstm_dataset_multi(obs, timesteps, win, npers1, hor):

    X = np.zeros((win*npers1,timesteps))
    y = np.zeros((win*npers1,1))

    cntr = 0
    for i in np.arange(0, win, 1):

        for j in np.arange(0, npers1, 1):

            X[cntr] = obs[i*(npers1+1) + j:i*(npers1+1) + j + timesteps]
            y[cntr] = np.mean(obs[i*(npers1+1) + j + timesteps:i*(npers1+1) + j + timesteps + hor])
            cntr = cntr + 1

    return np.array(X), np.array(y)

# Read in data
temp_dats = pd.read_csv('IBM_ok.csv')
print(temp_dats.columns)
ok5 = temp_dats['ok5_all'].to_numpy()

# Forecast horizon
forwin = 2

# No. of periods/day and ndays
npers = 78
ndays = len(ok5)/npers
npers1 = npers - forwin + 1

# Choose AR lags in form of days
nlags = 22*npers

# Initial window and no. of forecasts
initwin = 1000
nfore = int(ndays - initwin)

# Store mean forecasts and target
fore_gbm = np.zeros((nfore,npers1))
target = np.zeros((nfore,npers1))

# Grid for training at
train_grid = np.arange(0, nfore, 250)

# Forecasting loop
for i in np.arange(0, nfore, 1):
    ok5_tmp = ok5[i * npers:(initwin + i) * npers]

    target_tmp = ok5[(initwin + i) * npers:(initwin + i + 1) * npers]
    for j in np.arange(0, npers1, 1):
        target[i,j] = np.mean(target_tmp[j:(j+forwin)])

    # Create and train model only for times in train_grid
    if np.sum(train_grid == i) > 0:
        X, y = create_lstm_dataset_multi(ok5_tmp, nlags, initwin-22, npers1, forwin)
        gbm_model = lgb.LGBMRegressor()
        gbm_fit = gbm_model.fit(X, y)


    # Generate forecasts
    tmp_input = np.concatenate((ok5_tmp[-nlags:], target_tmp[0:npers-forwin]))
    tmp_input = np.reshape(tmp_input, (1,tmp_input.shape[0]))
    for j in np.arange(0, npers1, 1):
        pred_input = np.reshape(tmp_input[0, j:j + nlags],(1,nlags))
        fore_gbm[i, j] = gbm_fit.predict(pred_input)

    print(i)

# Reshape forecasts and write output
fore_lstm_out = np.reshape(fore_gbm, (npers1*nfore), order="C")
target_out = np.reshape(target, (npers1*nfore), order="C")
tmp_out = {'target': target_out, 'forecast': fore_lstm_out}
out_df = pd.DataFrame(tmp_out)
out_df.to_csv('IBM_2_gbm_fore.csv', index=False)
