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

% DI
formatSpec = '%f %f';
temp_DI = readtable('IBM_multireg_fore_simple_1_0.csv','Format',formatSpec,'ReadVariableNames',true);
fore_DI = temp_DI.fore_multireg;

% OI
formatSpec = '%f %f';
temp_OI = readtable('IBM_multireg_fore_simple_0_1.csv','Format',formatSpec,'ReadVariableNames',true);
fore_OI = temp_OI.fore_multireg;

% OI
formatSpec = '%f %f';
temp_I = readtable('IBM_multireg_fore_simple_0_0.csv','Format',formatSpec,'ReadVariableNames',true);
fore_I = temp_I.fore_multireg;

% All forecasts
forecasts = [fore_reg fore_DI fore_OI fore_I];
forecasts_names = {'miHAR';'miHAR^DI';'miHAR^OI';'miHAR^I'};

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

