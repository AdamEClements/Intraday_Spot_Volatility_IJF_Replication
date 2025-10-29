clc
clear all

load IBM_5minvol_Win.mat

% Data series
data_tmp = ok5_all;
ndays = length(udts5);

% No. of periods/day and ndays
npers = 78;

% Daily strucutre
lags = [1; 5; 22];
nlags = length(lags);
maxlags = max(lags);

% Intraday strucutre
i_lags = [1; 5; 20];
i_nlags = length(i_lags);
i_maxlags = max(i_lags);

% Initial window and no. of forecasts
initwin = 5700;
nfore = ndays - initwin;

ii = 1;
    
data_tmp_tmp = data_tmp((ii - 1)*npers + 1:(initwin + ii -1)*npers);

on_ret_tmp = on_ret(ii:ii + initwin - 1);
    
ndays = length(data_tmp_tmp)/npers - maxlags;

% Order: HAR, overnight + intraday
npars = nlags + 1 + i_nlags;  
   
% Store full y & x for all periods
y_all = zeros(ndays*npers,1);
x_all = zeros(ndays*npers,npars);
    
% Training data
% Variables for first intraday period
i = 1;
index_tmp = (maxlags*npers+1:npers:initwin*npers)' + i - 1;
y = data_tmp_tmp(index_tmp);

x = [];     % HAR
for k = 1:nlags
    xtmp = zeros(length(y),1);
    for kk = 1:lags(k)
        xtmp = xtmp + data_tmp_tmp(index_tmp - kk*npers);
    end
    xtmp = xtmp./lags(k);
    x = [x xtmp];         
end
x = [x abs(on_ret_tmp(maxlags+1:end))];     % O/night

                                  % Intraday
for k = 1:i_nlags
    ixtmp = zeros(length(y),1);
    for kk = 1:i_lags(k)
        ixtmp = ixtmp + data_tmp_tmp(index_tmp - kk);
    end
    ixtmp = ixtmp./i_lags(k);
    x = [x ixtmp];         
end

% Store full y & x for all periods
y_all((0:1:ndays-1)'*npers + i) = y;
x_all((0:1:ndays-1)'*npers + i, :) = x;

% Variables for remaining intraday periods
for i = 2:npers

    index_tmp = (maxlags*npers+1:npers:initwin*npers)' + i - 1;
    y = data_tmp_tmp(index_tmp);
    
    % Set up for for the daily HAR regressors
    x = [];
    for k = 1:nlags
        xtmp = zeros(length(y),1);
        for kk = 1:lags(k)
            xtmp = xtmp + data_tmp_tmp(index_tmp - kk*npers);
        end
        xtmp = xtmp./lags(k);
        x = [x xtmp];         
    end

    % Include overnight
    x = [x abs(on_ret_tmp(maxlags+1:end))];

    % Set up for intra daily HAR regressors    
    for k = 1:i_nlags
        ixtmp = zeros(length(y),1);
        for kk = 1:i_lags(k)
            ixtmp = ixtmp + data_tmp_tmp(index_tmp - kk);
        end
        ixtmp = ixtmp./i_lags(k);
        x = [x ixtmp];         
    end

    % Store full y & x for all periods
    y_all((0:1:ndays-1)'*npers + i) = y;
    x_all((0:1:ndays-1)'*npers + i, :) = x;
    
    
end     % end of all intraday periods - have all y and x variables
      
% Standardise data
inobs = length(y_all);
x_mu = mean(x_all);
x_std = std(x_all);
x_zall = (x_all - repmat(x_mu,inobs,1))./repmat(x_std,inobs,1);
y_mu = mean(y_all);
y_std = std(y_all);
y_zall = (y_all - repmat(y_mu,inobs,1))./repmat(y_std,inobs,1);

% Split
cut = floor(0.8*inobs);

x_train = x_zall(1:cut, :);
x_val = x_zall(cut+1:end, :);
y_train = y_zall(1:cut);
y_val = y_zall(cut+1:end);

% inputs as cells
for k = 1:length(y_train)
    x_train_c{k,1} = x_train(k, :)';
end

for k = 1:length(y_val)
    x_val_c{k,1} = x_val(k, :)';
end    

% ===== LSTM ====

layers = [
sequenceInputLayer(size(x_all,2))
lstmLayer(8, 'OutputMode', 'sequence') % First LSTM layer
lstmLayer(8, 'OutputMode', 'last')    % Second LSTM layer    
fullyConnectedLayer(1)
regressionLayer];

options = trainingOptions('adam', ...
'MaxEpochs',100, ...
'MiniBatchSize',1024, ...
'ValidationData',{x_val_c, y_val}, ...
'InitialLearnRate',0.001, ...
'Shuffle', 'every-epoch', ...
'LearnRateSchedule','piecewise', ...
'LearnRateDropFactor',0.1, ...
'ValidationPatience', 10, ... % Early stopping rounds
'SequencePaddingDirection', 'left', ......
'Verbose',1);   
    
% Train 
net = trainNetwork(x_train_c,y_train,layers,options);
 
% Predict
yfit = zeros(length(x_zall),1);
for i = 1:length(yfit)

    yfit(i) = predict(net,x_zall(i,:)');

end
yfit = yfit.*y_std + y_mu;

tout = table(y_all, yfit);
writetable(tout,'IBM_LSTM_resid.csv','WriteVariableNames',true)

aa = 0;
