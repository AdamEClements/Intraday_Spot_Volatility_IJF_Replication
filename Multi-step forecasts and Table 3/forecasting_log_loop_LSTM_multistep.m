clc
clear all

load IBM_5minvol_Win.mat
fileout = 'IBM_2_lstm_fore_log_test.csv';

% Data series - take logs
ok5_all(ok5_all == 0) = mean(ok5_all);
data_tmp = log(ok5_all);
on_ret(on_ret == 0) = 1e-4;

% Forecast horizon
forwin = 2;

% No. of periods/day and ndays
npers = 78;
ndays = length(udts5);

% Daily strucutre
lags = [1; 5; 22];
nlags = length(lags);
maxlags = max(lags);

% Intraday strucutre
i_lags = [1; 5; 20];
i_nlags = length(i_lags);
i_maxlags = max(i_lags);

% Initial window and no. of forecasts
initwin = 1000;
nfore = ndays - initwin;

% Store mean forecasts
fore_LSTM = zeros(nfore,npers - forwin + 1);     % npers indep. regs.

% Store target
target = zeros(nfore,npers - forwin + 1);
target_out = zeros(nfore,npers - forwin + 1);
npers1 = npers - forwin + 1;

% Grid to train LSTM
train_grid = (1:250:nfore)';

% Ensembles
nensem = 1;

% Forecasting loop
for ii = 1:nfore
    
    data_tmp_tmp = data_tmp((ii - 1)*npers + 1:(initwin + ii -1)*npers);
    on_ret_tmp = on_ret(ii:ii + initwin - 1);

    target_tmp = data_tmp((initwin + ii - 1)*npers+1:(initwin + ii)*npers)';
    
    for j = 1:npers1

        target_out(ii,j) = mean(exp(target_tmp(j:j+forwin-1)));
        target(ii,j) = mean(target_tmp(j:j+forwin-1));

    end

    ndays = length(data_tmp_tmp)/npers - maxlags;

    % Order: HAR, overnight + intraday
    npars = nlags + 1 + i_nlags;  
       
    % Store full y & x for all periods
    y_all = zeros(ndays*npers1,1);
    x_all = zeros(ndays*npers1,npars);
    
    % Training data
    % Variables for first intraday period
    i = 1;
    index_tmp = (maxlags*npers+1:npers:initwin*npers)' + i - 1;
    y = zeros(length(index_tmp),1);
    for j = 1:length(index_tmp)
    
        y(j) = mean(data_tmp_tmp(index_tmp(j):index_tmp(j) + forwin -1));
    
    end

    x = [];     % HAR
    for k = 1:nlags
        xtmp = zeros(length(y),1);
        for kk = 1:lags(k)
            xtmp = xtmp + data_tmp_tmp(index_tmp - kk*npers);
        end
        xtmp = xtmp./lags(k);
        x = [x xtmp];         
    end
    x = [x log(abs(on_ret_tmp(maxlags+1:end)))];     % O/night

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
    y_all((0:1:ndays-1)'*npers1 + i) = y;
    x_all((0:1:ndays-1)'*npers1 + i, :) = x;
    
    % Variables for remaining intraday periods
    for i = 2:npers - forwin + 1
   
        index_tmp = (maxlags*npers+1:npers:initwin*npers)' + i - 1;
        y = zeros(length(index_tmp),1);
        for j = 1:length(index_tmp)
        
            y(j) = mean(data_tmp_tmp(index_tmp(j):index_tmp(j) + forwin -1));
        
        end
        
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
        x = [x log(abs(on_ret_tmp(maxlags+1:end)))];
    
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
        y_all((0:1:ndays-1)'*npers1 + i) = y;
        x_all((0:1:ndays-1)'*npers1 + i, :) = x;
        
        
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
    
    % Train (only at grid points)
    if sum(train_grid == ii) > 0
        for iensem = 1:nensem
            net{iensem} = trainNetwork(x_train_c,y_train,layers,options);
        end
		
		% In-sample error variance    
        for k = 1:length(x_zall)
            x_zall_c{k,1} = x_zall(k, :)';
        end
        yhat = predict(net{1},x_zall_c);
        yhat = yhat*y_std + y_mu;
        res_var = var(yhat - y_zall);
    end

    % Forecasting inputs

    % First period
    x_tmp = [];
    for k = 1:nlags
        xtmp = 0;
        for kk = 1:lags(k)                    
            xtmp = xtmp + data_tmp_tmp((initwin - kk)*npers + 1);
        end
        xtmp = xtmp./lags(k);
        x_tmp = [x_tmp; xtmp];         
    end

    % Include overnight
    x_tmp = [x_tmp; log(abs(on_ret(initwin+ii)))];

    % Set up for intra daily
    for k = 1:i_nlags
        ixtmp = 0;
        for kk = 1:i_lags(k)
            ixtmp = ixtmp + data_tmp_tmp(initwin*npers - kk + 1);
        end
        ixtmp = ixtmp./i_lags(k);
        x_tmp = [x_tmp; ixtmp];         
    end
       
    % Predict and rescale
    fore_tmp = 0;
    x_tmp = (x_tmp - x_mu')./x_std';
    for iensem = 1:nensem
        fore_tmp = fore_tmp + predict(net{iensem},x_tmp);
    end
    fore_tmp = fore_tmp/nensem;
    fore_tmp = fore_tmp*y_std + y_mu;
	fore_tmp = exp(fore_tmp + 0.5*res_var);
    fore_LSTM(ii,1) = fore_tmp;

    % Remaining periods
    for j = 2:npers - forwin + 1

        % Set up for daily
        x_tmp = [];
        for k = 1:nlags
            xtmp = 0;
            for kk = 1:lags(k)        
                xtmp = xtmp + data_tmp_tmp((initwin - kk)*npers + j);
            end
            xtmp = xtmp./lags(k);
            x_tmp = [x_tmp; xtmp];         
        end
           
        % Include overnight
        x_tmp = [x_tmp; log(abs(on_ret(initwin+ii)))];
    
    
        % Set up for intra daily
        data_tmp1 = [data_tmp_tmp; target(ii,:)'];
            
        for k = 1:i_nlags
            ixtmp = 0;
            for kk = 1:i_lags(k)
                % + j - 1 gives starting point previous to fore time 
                % - kk + 1 how many lags to go back
                ixtmp = ixtmp + data_tmp1(initwin*npers + j - 1 - kk + 1);
            end
            ixtmp = ixtmp./i_lags(k);
            x_tmp = [x_tmp; ixtmp];         
        end
        
        x_tmp = (x_tmp - x_mu')./x_std';
        
        fore_tmp = 0;
        for iensem = 1:nensem
            fore_tmp = fore_tmp + predict(net{iensem},x_tmp);
        end
        fore_tmp = fore_tmp/nensem;
        fore_tmp = fore_tmp*y_std + y_mu;
		fore_tmp = exp(fore_tmp + 0.5*res_var);
        fore_LSTM(ii,j) = fore_tmp;

        
    end
    

    
    ii
end

 
tmp_grid = (1:1:nfore*npers1)';

fore_LSTM1 = reshape(fore_LSTM',nfore*npers1,1);
target1 = reshape(target_out',nfore*npers1,1);

tout = table(fore_LSTM1);
writetable(tout,fileout,'WriteVariableNames',true)

aa = 0;
