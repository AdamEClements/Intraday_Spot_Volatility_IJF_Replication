clc
clear all

% Regression forecasts
formatSpec = '%f %f %f';
temp_regfore = readtable('IBM_fore.csv','Format',formatSpec,'ReadVariableNames',true);
target = temp_regfore.target1;
target(target == 0) = mean(target); 
fore_reg = temp_regfore.miHAR;
fore_sHAR = temp_regfore.diHAR;
fore_sHAR(fore_sHAR <= 0) = mean(fore_sHAR);

% GBM forecasts
formatSpec = '%f %f';
temp_gbmfore = readtable('IBM_gbm_fore.csv','Format',formatSpec,'ReadVariableNames',true);
fore_gbm = temp_gbmfore.forecast;

% LSTM forecasts
formatSpec = '%f';
temp_lstmfore = readtable('IBM_lstm_fore.csv','Format',formatSpec,'ReadVariableNames',true);
fore_lstm = temp_lstmfore.fore_LSTM1;

% All forecasts
forecasts = [fore_reg fore_gbm fore_lstm fore_sHAR];
forecasts_names = {'miHAR';'liGBM';'LSTM';'diHAR'};

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

% Compare forecasts in different intervals
nfore = length(fore_reg)/78;
fore_reg1 = reshape(fore_reg,78,nfore);
fore_gbm1 = reshape(fore_gbm,78,nfore);
target1 = reshape(target,78,nfore);
fore_sHAR1 = reshape(fore_sHAR,78,nfore);
mse1_reg = mean(reshape(mse_loss(:,1),78,nfore),2);
mse1_gbm = mean(reshape(mse_loss(:,2),78,nfore),2);
qlike1_reg = mean(reshape(qlike_loss(:,1),78,nfore),2);
qlike1_gbm = mean(reshape(qlike_loss(:,2),78,nfore),2);

mse_out = mse1_reg./mse1_gbm;
qlike_out = qlike1_reg./qlike1_gbm;

tabout = table(mse_out, qlike_out);
writetable(tabout, 'loss_ratios.csv')

