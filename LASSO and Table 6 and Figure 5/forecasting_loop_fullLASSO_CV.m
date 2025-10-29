% Choosing lambda using hv-cross validation
% Racine JoE 

% Turn back on parallel

clc
clear all

parpool('local',4);
poolobj = gcp;

load IBM_5minvol_Win.mat
fileout = 'IBM_multireg_fore_LASSO_CV.csv';

% Data series
data_tmp = ok5_all;

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

% Parameters for CV
del = 0.95;                     % Delta coefficient
p = i_maxlags + maxlags + 2;    % No of predictors
h = 50;                         % Around CV window
n = initwin - maxlags;          % Nobs in each interval regression
n_delta = floor(n^del);         % Size of estimation window
v = (n - n_delta - 2*h - 1)/2;  % Width of one side of CV window
nlam = 100;                      % No of lamba points used in gridsearch
nocv = 5;                       % No of CV at each lambda gridpoint

% Store mean forecasts
fore_regs = zeros(nfore,npers);     % npers indep. regs.
fore_seasHAR = zeros(nfore,npers);
fore_regs_lasso = zeros(nfore,npers);

% Store target
target = zeros(nfore,npers);

% Store LASSO
beta_full = zeros(npers, i_maxlags + maxlags + 2, nfore);

% Forecasting loop
% PAR here
parfor i = 1:nfore
    
    data_tmp_tmp = data_tmp((i - 1)*npers + 1:(initwin + i -1)*npers);
    on_ret_tmp = on_ret(i:i + initwin - 1);
    
    target(i,:) = data_tmp((initwin + i - 1)*npers+1:(initwin + i)*npers)';

    % Multi eq - LASSO
    [out1, beta_t]= lasso_fore_func(data_tmp_tmp, on_ret_tmp, npers, lags, nlags, maxlags, ...  
                    i_lags, i_nlags, i_maxlags, length(data_tmp_tmp)/npers, on_ret(i + initwin), target(i,:), h, v, nlam, nocv, n);
    fore_regs_lasso(i,:) = out1;
    beta_full(:,:,i) = beta_t;  
    
    i
end

delete(poolobj)

tmp_grid = (1:1:nfore*npers)';

fore_regs_lasso1 = reshape(fore_regs_lasso',nfore*npers,1);
neg_fore = tmp_grid(fore_regs_lasso1 <= 0);
fore_regs_lasso1(neg_fore) = fore_regs_lasso1(neg_fore-1);
fore_multireg_lasso_CV = fore_regs_lasso1;

target1 = reshape(target',nfore*npers,1);
tout = table(target1, fore_multireg_lasso_CV);
writetable(tout, fileout,'WriteVariableNames',true)

nfores = size(beta_full,3);

coeffs_daily = zeros(nfores, 78);
coeffs_onight = zeros(nfores, 78);
coeffs_intra = zeros(nfores, 78);

for i = 1:nfores

    for j = 1:78
    
        coeffs_daily(i,j) = sum(abs(beta_full(j, 2:23, i)') > 1e-10);
        coeffs_onight(i,j) = sum(abs(beta_full(j, 24, i)) > 1e-10);
        coeffs_intra(i,j) = sum(abs(beta_full(j, 25:end, i)') ~= 0);

    end


end

tout = table(coeffs_daily);
writetable(tout, 'component_1.csv','WriteVariableNames',true)

tout = table(coeffs_onight);
writetable(tout, 'component_2.csv','WriteVariableNames',true)

tout = table(coeffs_intra);
writetable(tout, 'component_3.csv','WriteVariableNames',true)


aa = 0;


%==============================================================================
function[fore_out, beta_hat_new] = lasso_fore_func(data_tmp, on_ret, npers, lags, nlags, maxlags,...
                        i_lags, i_nlags, i_maxlags, ndays, on_ret_t1, final_day, h, v, nlam, nocv, n);

% ==== Estimate ====

% Order: intercepts, HAR, overnight + intraday
npars = 1 + maxlags + 1 + i_maxlags;

beta_hat_new = zeros(npers,npars);
 
% Store fitted values
yhat = zeros(npers*ndays,1);

% Do regression for first intraday period
i = 1;
index_tmp = (maxlags*npers+1:npers:ndays*npers)' + i - 1;
y = data_tmp(index_tmp);

% Set up for for the daily HAR regressors
x = [];
xtmp = zeros(length(y),maxlags);
for kk = 1:maxlags
    xtmp(:,kk) = data_tmp(index_tmp - kk*npers);
end
x = [x xtmp];         

% Include overnight
x = [x abs(on_ret(maxlags+1:end))];
% Set up for for the intra daily HAR regressors
x_i = [];
ixtmp = zeros(length(y),i_maxlags);
for kk = 1:i_maxlags
    ixtmp(:,kk) = (index_tmp - kk);
end
x = [x ixtmp];     

% Get grid of lambdas
[bout, fit_las] = lasso(x,y, 'Intercept',true, 'NumLambda', nlam);
lam_grid = fit_las.Lambda;

% Store CV MSE loss
cv_loss = zeros(nlam,nocv);

% Double loop for get loss in each CV window at each lambda
i_out = randi([h+v+1, n-h-v], nocv, 1);
for ii = 1:nlam

    for jj = 1:nocv

        % Choose CV point
        % Choose estimation and CV windows        
        y_cv = y(i_out(jj) - v : i_out(jj) + v);
        x_cv = x(i_out(jj) - v : i_out(jj) + v, :);
        mask = true(n, 1);
        rowsToRemove = (i_out(jj) - v - h : 1 : i_out(jj) + v + h)';
        mask(rowsToRemove) = false;
        y_est = y(mask);
        x_est = x(mask,:);
        
        % Estimate at labmda grid point
        [beta_las, fit_las] = lasso(x_est,y_est, 'Intercept',true, 'Lambda', lam_grid(ii));

        % Get CV loss
        yhat_cv = x_cv*beta_las;
        cv_loss(ii,jj) = mean((y_cv - yhat_cv).^2);

    end

end

cv_loss_mu = mean(cv_loss,2);
lam_opt = lam_grid(cv_loss_mu == min(cv_loss_mu));

[beta_las, fit_las] = lasso(x,y, 'Intercept',true, 'Lambda', lam_opt);
beta_tmp = [fit_las.Intercept; beta_las];
beta_hat_new(i,:) = beta_tmp';

for i = 2:npers
   
    index_tmp = (maxlags*npers+1:npers:npers*ndays)' + i - 1;
    y = data_tmp(index_tmp);
    
    % Set up for for the daily HAR regressors
    x = [];
    xtmp = zeros(length(y),maxlags);
    for kk = 1:maxlags
        xtmp(:,kk) = data_tmp(index_tmp - kk*npers);
    end
    x = [x xtmp];

    % Include overnight
    x = [x abs(on_ret(maxlags+1:end))];
    % Set up for for the intra daily HAR regressors
    x_i = [];
    ixtmp = zeros(length(y),i_maxlags);
    for kk = 1:i_maxlags
        ixtmp(:,kk) = (index_tmp - kk);
    end
    x = [x ixtmp];
    
    % Get grid of lambdas
    [bout, fit_las] = lasso(x,y, 'Intercept',true, 'NumLambda', nlam);
    lam_grid = fit_las.Lambda;
    
    % Store CV MSE loss
    cv_loss = zeros(nlam,nocv);
    
    % Double loop for get loss in each CV window at each lambda
    for ii = 1:nlam
    
        for jj = 1:nocv
    
            % Choose CV point
            % Choose estimation and CV windows
            i_out = randi([h+v+1, n-h-v], 1, 1);
            y_cv = y(i_out - v : i_out + v);
            x_cv = x(i_out - v : i_out + v, :);
            mask = true(n, 1);
            rowsToRemove = (i_out - v - h : 1 : i_out + v + h)';
            mask(rowsToRemove) = false;
            y_est = y(mask);
            x_est = x(mask,:);
            
            % Estimate at labmda grid point
            [beta_las, fit_las] = lasso(x_est,y_est, 'Intercept',true, 'Lambda', lam_grid(ii));
    
            % Get CV loss
            yhat_cv = x_cv*beta_las;
            cv_loss(ii,jj) = mean((y_cv - yhat_cv).^2);
    
        end
    
    end

    cv_loss_mu = mean(cv_loss,2);
    lam_opt = lam_grid(cv_loss_mu == min(cv_loss_mu));
    
    [beta_las, fit_las] = lasso(x,y, 'Intercept',true, 'Lambda', lam_opt);
    beta_tmp = [fit_las.Intercept; beta_las];
    beta_hat_new(i,:) = beta_tmp';
                      
end % end of estimation

% Swtich off overnight==============
% beta_hat_new(:,5) = 0;
%==================================

% ==== Forecast ====
fore_out = zeros(npers,1);

% First period
x_tmp = 1; 

xtmp = zeros(maxlags,1);
for kk = 1:maxlags        
    xtmp(kk) = data_tmp((ndays - kk)*npers + 1);
end
x_tmp = [x_tmp; xtmp];         


% Include overnight
x_tmp = [x_tmp; abs(on_ret_t1)];

% Set up for intra daily
ixtmp = zeros(i_maxlags,1);
for kk = 1:i_maxlags
    ixtmp(kk) = data_tmp(ndays*npers - kk + 1);
end
x_tmp = [x_tmp; ixtmp];         


fore_out(1) = beta_hat_new(1,:)*x_tmp;

% Remaining periods
for j = 2:npers
   
    x_tmp = 1; 

    % Set up for daily
    xtmp = zeros(maxlags,1);
    for kk = 1:maxlags        
        xtmp(kk) = data_tmp((ndays - kk)*npers + j);
    end
    x_tmp = [x_tmp; xtmp];      
       
    % Include overnight
    x_tmp = [x_tmp; abs(on_ret_t1)];


    % Set up for intra daily
    data_tmp1 = [data_tmp; final_day'];

    ixtmp = zeros(i_maxlags,1);
    for kk = 1:i_maxlags
        % + j - 1 gives starting point previous to fore time 
        % - kk + 1 how many lags to go back
        ixtmp(kk) = data_tmp1(ndays*npers + j - 1 - kk + 1);
    end
    x_tmp = [x_tmp; ixtmp];         

    
    fore_out(j) = beta_hat_new(j,:)*x_tmp;
    
end


end  % end of forecasting function for each day


