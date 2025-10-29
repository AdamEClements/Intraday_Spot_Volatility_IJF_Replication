clc
clear all

% Regression forecasts
formatSpec = '%f %f %f';
temp_regfore = readtable('IBM_fore.csv','Format',formatSpec,'ReadVariableNames',true);
target = temp_regfore.target1;
target(target == 0) = mean(target); 
fore_reg = temp_regfore.miHAR;

% Diurnal forecasts
formatSpec = '%f %f';
temp_difore = readtable('IBM_multireg_di_fore.csv','Format',formatSpec,'ReadVariableNames',true);
fore_di = temp_difore.fore_di_multireg;

% GBM forecasts
formatSpec = '%f %f';
temp_gbmfore = readtable('IBM_gbm_fore.csv','Format',formatSpec,'ReadVariableNames',true);
fore_gbm = temp_gbmfore.forecast;

% LSTM forecasts
formatSpec = '%f';
temp_lstmfore = readtable('IBM_lstm_fore.csv','Format',formatSpec,'ReadVariableNames',true);
fore_lstm = temp_lstmfore.fore_LSTM1;

% All forecasts
forecasts = [fore_reg fore_gbm fore_lstm fore_di];
forecasts_names = {'miHAR';'liGBM';'LSTM';'miHAR**'};

% MSE Loss and ratios
mse_fores = mean((forecasts - repmat(target,1,size(forecasts,2))).^2);
mse_loss = (forecasts - repmat(target,1,size(forecasts,2))).^2;

display(forecasts_names')
mse_ratios = mse_fores(1)./mse_fores

% QLIKE Loss and ratios
qlike = mean(repmat(target,1,size(forecasts,2))./forecasts - log(repmat(target,1,size(forecasts,2))./forecasts) - 1);
qlike_loss = repmat(target,1,size(forecasts,2))./forecasts - log(repmat(target,1,size(forecasts,2))./forecasts) - 1;

display(forecasts_names')
qlike_ratios = qlike(1)./qlike

% MSE MCS
[inR,pvalsR,exR,inSQ,pvalsSQ,exSQ] = mcs(mse_loss,0.1,3000,30);

[[exR; inR] pvalsR  ]
table(forecasts_names([exR; inR]), pvalsR)

% QLIKE MCS
[inR,pvalsR,exR,inSQ,pvalsSQ,exSQ] = mcs(qlike_loss,0.1,3000,30);

[[exR; inR] pvalsR  ]
table(forecasts_names([exR; inR]), pvalsR)

