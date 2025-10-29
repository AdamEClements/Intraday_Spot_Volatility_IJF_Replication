clc
clear all

% parpool('local',4);
% poolobj = gcp;

load IBM_5minvol_Win.mat
fileout = 'IBM_multireg_fore_fix.csv';

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

% Store mean forecasts
fore_regs = zeros(nfore,npers);     % npers indep. regs.
fore_seasHAR = zeros(nfore,npers);
fore_regs_lasso = zeros(nfore,npers);
fore_regs_fixed = zeros(nfore,npers);

% Store target
target = zeros(nfore,npers);

% Forecasting loop
% PAR here
for i = 1:nfore
    
    data_tmp_tmp = data_tmp((i - 1)*npers + 1:(initwin + i -1)*npers);
    on_ret_tmp = on_ret(i:i + initwin - 1);
    
    target(i,:) = data_tmp((initwin + i - 1)*npers+1:(initwin + i)*npers)';
        
    % Multi eq
    [out1]= fore_func(data_tmp_tmp, on_ret_tmp, npers, lags, nlags, maxlags, ...  
                    i_lags, i_nlags, i_maxlags, length(data_tmp_tmp)/npers, on_ret(i + initwin), target(i,:));
         
    fore_regs(i,:) = out1;

    % Multi eq - fixed
    [out1]= fix_fore_func(data_tmp_tmp, on_ret_tmp, npers, lags, nlags, maxlags, ...  
                    i_lags, i_nlags, i_maxlags, length(data_tmp_tmp)/npers, on_ret(i + initwin), target(i,:));
         
    fore_regs_fixed(i,:) = out1;    
    
    i
end

%delete(poolobj)

tmp_grid = (1:1:nfore*npers)';

fore_regs_fixed1 = reshape(fore_regs_fixed',nfore*npers,1);
neg_fore = tmp_grid(fore_regs_fixed1 <= 0);
fore_regs_fixed1(neg_fore) = fore_regs_fixed1(neg_fore-1);
fore_multireg_fixed = fore_regs_fixed1;

target1 = reshape(target',nfore*npers,1);

tout = table(target1, fore_multireg_fixed);
writetable(tout,fileout,'WriteVariableNames',true)

aa = 0;

%==============================================================================
function[fore_out] = fore_func(data_tmp, on_ret, npers, lags, nlags, maxlags,...
                        i_lags, i_nlags, i_maxlags, ndays, on_ret_t1, final_day);

% ==== Estimate ====

% Order: intercepts, HAR, overnight + intraday
npars = 1 + nlags + 1 + i_nlags;  

beta_hat_new = zeros(npers,npars);
 
% Store fitted values
yhat = zeros(npers*ndays,1);

% Do regression for first intraday period
i = 1;
index_tmp = (maxlags*npers+1:npers:ndays*npers)' + i - 1;
y = data_tmp(index_tmp);

% Set up for for the daily HAR regressors
x = [];
for k = 1:nlags
    xtmp = zeros(length(y),1);
    for kk = 1:lags(k)
        xtmp = xtmp + data_tmp(index_tmp - kk*npers);
    end
    xtmp = xtmp./lags(k);
    x = [x xtmp];         
end

% Include overnight
x = [x abs(on_ret(maxlags+1:end))];

% Set up for for the intra daily HAR regressors
x_i = [];
for k = 1:i_nlags
    ixtmp = zeros(length(y),1);
    for kk = 1:i_lags(k)
        ixtmp = ixtmp + data_tmp(index_tmp - kk);
    end
    ixtmp = ixtmp./i_lags(k);
    x = [x ixtmp];         
end

% Estimate
mdl = fitlm(x,y,'linear');
coeff = mdl.Coefficients.Estimate;
beta_hat_new(i,:) = coeff';
yhat(index_tmp,:) = mdl.Fitted';

for i = 2:npers
   
    index_tmp = (maxlags*npers+1:npers:npers*ndays)' + i - 1;
    y = data_tmp(index_tmp);
    
    % Set up for for the daily HAR regressors
    x = [];
    for k = 1:nlags
        xtmp = zeros(length(y),1);
        for kk = 1:lags(k)
            xtmp = xtmp + data_tmp(index_tmp - kk*npers);
        end
        xtmp = xtmp./lags(k);
        x = [x xtmp];         
    end

    % Include overnight
    x = [x abs(on_ret(maxlags+1:end))];

    % Set up for intra daily HAR regressors    
    for k = 1:i_nlags
        ixtmp = zeros(length(y),1);
        for kk = 1:i_lags(k)
            ixtmp = ixtmp + data_tmp(index_tmp - kk);
        end
        ixtmp = ixtmp./i_lags(k);
        x = [x ixtmp];         
    end
    
    % Estimate
    mdl = fitlm(x,y,'linear');
    coeff = mdl.Coefficients.Estimate;
    beta_hat_new(i,:) = coeff';
    yhat(index_tmp,:) = mdl.Fitted;
                      
end % end of estimation

% Swtich off overnight==============
% beta_hat_new(:,5) = 0;
%==================================

% ==== Forecast ====
fore_out = zeros(npers,1);

% First period
x_tmp = 1; 

for k = 1:nlags
    xtmp = 0;
    for kk = 1:lags(k)        
        xtmp = xtmp + data_tmp((ndays - kk)*npers + 1);
    end
    xtmp = xtmp./lags(k);
    x_tmp = [x_tmp; xtmp];         
end

% Include overnight
x_tmp = [x_tmp; abs(on_ret_t1)];

% Set up for intra daily
for k = 1:i_nlags
    ixtmp = 0;
    for kk = 1:i_lags(k)
        ixtmp = ixtmp + data_tmp(ndays*npers - kk + 1);
    end
    ixtmp = ixtmp./i_lags(k);
    x_tmp = [x_tmp; ixtmp];         
end

fore_out(1) = beta_hat_new(1,:)*x_tmp;

% Remaining periods
for j = 2:npers
   
    x_tmp = 1; 

    % Set up for daily
    for k = 1:nlags
        xtmp = 0;
        for kk = 1:lags(k)        
            xtmp = xtmp + data_tmp((ndays - kk)*npers + j);
        end
        xtmp = xtmp./lags(k);
        x_tmp = [x_tmp; xtmp];         
    end
       
    % Include overnight
    x_tmp = [x_tmp; abs(on_ret_t1)];


    % Set up for intra daily
    data_tmp1 = [data_tmp; final_day'];
    for k = 1:i_nlags
        ixtmp = 0;
        for kk = 1:i_lags(k)
            % + j - 1 gives starting point previous to fore time 
            % - kk + 1 how many lags to go back
            ixtmp = ixtmp + data_tmp1(ndays*npers + j - 1 - kk + 1);
        end
        ixtmp = ixtmp./i_lags(k);
        x_tmp = [x_tmp; ixtmp];         
    end
    
    fore_out(j) = beta_hat_new(j,:)*x_tmp;
    
end


end  % end of forecasting function for each day

%==============================================================================
function[fore_out] = fix_fore_func(data_tmp, on_ret, npers, lags, nlags, maxlags,...
                        i_lags, i_nlags, i_maxlags, ndays, on_ret_t1, final_day);

% ==== Estimate ====

% Order: intercepts, HAR, overnight + intraday
npars = 1 + nlags + 1 + i_nlags;  

beta_hat_new = zeros(npers,npars);
 
% Store fitted values
yhat = zeros(npers*ndays,1);

% Do regression for first intraday period
i = 1;
index_tmp = (maxlags*npers+1:npers:ndays*npers)' + i - 1;
y = data_tmp(index_tmp);

% Set up for for the daily HAR regressors
x = [];
for k = 1:nlags
    xtmp = zeros(length(y),1);
    for kk = 1:lags(k)
        xtmp = xtmp + data_tmp(index_tmp - kk*npers);
    end
    xtmp = xtmp./lags(k);
    x = [x xtmp];         
end

% Include overnight
x = [x abs(on_ret(maxlags+1:end))];

% Set up for for the intra daily HAR regressors
x_i = [];
for k = 1:i_nlags
    ixtmp = zeros(length(y),1);
    for kk = 1:i_lags(k)
        ixtmp = ixtmp + data_tmp(index_tmp - kk);
    end
    ixtmp = ixtmp./i_lags(k);
    x = [x ixtmp];         
end

% Estimate
mdl = fitlm(x,y,'linear');
coeff = mdl.Coefficients.Estimate;
beta_hat_new(i,:) = coeff';
yhat(index_tmp,:) = mdl.Fitted';

for i = 2:npers
   
    index_tmp = (maxlags*npers+1:npers:npers*ndays)' + i - 1;
    y = data_tmp(index_tmp);
    
    % Set up for for the daily HAR regressors
    x = [];
    for k = 1:nlags
        xtmp = zeros(length(y),1);
        for kk = 1:lags(k)
            xtmp = xtmp + data_tmp(index_tmp - kk*npers);
        end
        xtmp = xtmp./lags(k);
        x = [x xtmp];         
    end

    % Include overnight
    x = [x abs(on_ret(maxlags+1:end))];

    % Set up for intra daily HAR regressors    
    for k = 1:i_nlags
        ixtmp = zeros(length(y),1);
        for kk = 1:i_lags(k)
            ixtmp = ixtmp + data_tmp(index_tmp - kk);
        end
        ixtmp = ixtmp./i_lags(k);
        x = [x ixtmp];         
    end
    
    % Estimate
    if i <= 15
        mdl = fitlm(x,y,'linear');
        coeff = mdl.Coefficients.Estimate;
        beta_hat_new(i,:) = coeff';
        yhat(index_tmp,:) = mdl.Fitted;
    else
        beta_hat_new(i,:) = beta_hat_new(i-1,:);
    end
                      
end % end of estimation

% Swtich off overnight==============
% beta_hat_new(:,5) = 0;
%==================================

% ==== Forecast ====
fore_out = zeros(npers,1);

% First period
x_tmp = 1; 

for k = 1:nlags
    xtmp = 0;
    for kk = 1:lags(k)        
        xtmp = xtmp + data_tmp((ndays - kk)*npers + 1);
    end
    xtmp = xtmp./lags(k);
    x_tmp = [x_tmp; xtmp];         
end

% Include overnight
x_tmp = [x_tmp; abs(on_ret_t1)];

% Set up for intra daily
for k = 1:i_nlags
    ixtmp = 0;
    for kk = 1:i_lags(k)
        ixtmp = ixtmp + data_tmp(ndays*npers - kk + 1);
    end
    ixtmp = ixtmp./i_lags(k);
    x_tmp = [x_tmp; ixtmp];         
end

fore_out(1) = beta_hat_new(1,:)*x_tmp;

% Remaining periods
for j = 2:npers
   
    x_tmp = 1; 

    % Set up for daily
    for k = 1:nlags
        xtmp = 0;
        for kk = 1:lags(k)        
            xtmp = xtmp + data_tmp((ndays - kk)*npers + j);
        end
        xtmp = xtmp./lags(k);
        x_tmp = [x_tmp; xtmp];         
    end
       
    % Include overnight
    x_tmp = [x_tmp; abs(on_ret_t1)];


    % Set up for intra daily
    data_tmp1 = [data_tmp; final_day'];
    for k = 1:i_nlags
        ixtmp = 0;
        for kk = 1:i_lags(k)
            % + j - 1 gives starting point previous to fore time 
            % - kk + 1 how many lags to go back
            ixtmp = ixtmp + data_tmp1(ndays*npers + j - 1 - kk + 1);
        end
        ixtmp = ixtmp./i_lags(k);
        x_tmp = [x_tmp; ixtmp];         
    end
    
    fore_out(j) = beta_hat_new(j,:)*x_tmp;
    
end


end  % end of forecasting function for each day


%==============================================================================
function[fore_out] = lasso_fore_func(data_tmp, on_ret, npers, lags, nlags, maxlags,...
                        i_lags, i_nlags, i_maxlags, ndays, on_ret_t1, final_day);

% ==== Estimate ====

% Order: intercepts, HAR, overnight + intraday
npars = 1 + nlags + 1 + i_nlags;  

beta_hat_new = zeros(npers,npars);
 
% Store fitted values
yhat = zeros(npers*ndays,1);

% Do regression for first intraday period
i = 1;
index_tmp = (maxlags*npers+1:npers:ndays*npers)' + i - 1;
y = data_tmp(index_tmp);

% Set up for for the daily HAR regressors
x = [];
for k = 1:nlags
    xtmp = zeros(length(y),1);
    for kk = 1:lags(k)
        xtmp = xtmp + data_tmp(index_tmp - kk*npers);
    end
    xtmp = xtmp./lags(k);
    x = [x xtmp];         
end

% Include overnight
x = [x abs(on_ret(maxlags+1:end))];

% Set up for for the intra daily HAR regressors
x_i = [];
for k = 1:i_nlags
    ixtmp = zeros(length(y),1);
    for kk = 1:i_lags(k)
        ixtmp = ixtmp + data_tmp(index_tmp - kk);
    end
    ixtmp = ixtmp./i_lags(k);
    x = [x ixtmp];         
end

% Estimate
mdl = fitlm(x,y,'linear');
coeff = mdl.Coefficients.Estimate;
beta_hat_new(i,:) = coeff';
yhat(index_tmp,:) = mdl.Fitted';

% [beta_las, fit_las] = lasso(x,y, 'Intercept',true);
% fit = fit_las.MSE;
% [minmse, minind] = min(fit);
% beta_tmp = [fit_las.Intercept(minind); beta_las(:, minind)];
% beta_hat_new(i,:) = beta_tmp';

for i = 2:npers
   
    index_tmp = (maxlags*npers+1:npers:npers*ndays)' + i - 1;
    y = data_tmp(index_tmp);
    
    % Set up for for the daily HAR regressors
    x = [];
    for k = 1:nlags
        xtmp = zeros(length(y),1);
        for kk = 1:lags(k)
            xtmp = xtmp + data_tmp(index_tmp - kk*npers);
        end
        xtmp = xtmp./lags(k);
        x = [x xtmp];         
    end

    % Include overnight
    x = [x abs(on_ret(maxlags+1:end))];

    % Set up for intra daily HAR regressors    
    for k = 1:i_nlags
        ixtmp = zeros(length(y),1);
        for kk = 1:i_lags(k)
            ixtmp = ixtmp + data_tmp(index_tmp - kk);
        end
        ixtmp = ixtmp./i_lags(k);
        x = [x ixtmp];         
    end
    
    % Estimate
    mdl = fitlm(x,y,'linear');
    coeff = mdl.Coefficients.Estimate;
    beta_hat_new(i,:) = coeff';
    yhat(index_tmp,:) = mdl.Fitted;
    % [beta_las, fit_las] = lasso(x,y, 'Intercept',true);
    % fit = fit_las.MSE;
    % [minmse, minind] = min(fit);
    % beta_tmp = [fit_las.Intercept(minind); beta_las(:, minind)];
    % beta_hat_new(i,:) = beta_tmp';
                      
end % end of estimation

% Swtich off overnight==============
% beta_hat_new(:,5) = 0;
%==================================

% ==== Forecast ====
fore_out = zeros(npers,1);

% First period
x_tmp = 1; 

for k = 1:nlags
    xtmp = 0;
    for kk = 1:lags(k)        
        xtmp = xtmp + data_tmp((ndays - kk)*npers + 1);
    end
    xtmp = xtmp./lags(k);
    x_tmp = [x_tmp; xtmp];         
end

% Include overnight
x_tmp = [x_tmp; abs(on_ret_t1)];

% Set up for intra daily
for k = 1:i_nlags
    ixtmp = 0;
    for kk = 1:i_lags(k)
        ixtmp = ixtmp + data_tmp(ndays*npers - kk + 1);
    end
    ixtmp = ixtmp./i_lags(k);
    x_tmp = [x_tmp; ixtmp];         
end

fore_out(1) = beta_hat_new(1,:)*x_tmp;

% Remaining periods
for j = 2:npers
   
    x_tmp = 1; 

    % Set up for daily
    for k = 1:nlags
        xtmp = 0;
        for kk = 1:lags(k)        
            xtmp = xtmp + data_tmp((ndays - kk)*npers + j);
        end
        xtmp = xtmp./lags(k);
        x_tmp = [x_tmp; xtmp];         
    end
       
    % Include overnight
    x_tmp = [x_tmp; abs(on_ret_t1)];


    % Set up for intra daily
    data_tmp1 = [data_tmp; final_day'];
    for k = 1:i_nlags
        ixtmp = 0;
        for kk = 1:i_lags(k)
            % + j - 1 gives starting point previous to fore time 
            % - kk + 1 how many lags to go back
            ixtmp = ixtmp + data_tmp1(ndays*npers + j - 1 - kk + 1);
        end
        ixtmp = ixtmp./i_lags(k);
        x_tmp = [x_tmp; ixtmp];         
    end
    
    fore_out(j) = beta_hat_new(j,:)*x_tmp;
    
end


end  % end of forecasting function for each day


%==============================================================================
function[fore_out] = seasHAR_fore_func(data_tmp, npers, ndays, final_day)
                  
% ==== Estimate ====

data_tmp1 = reshape(data_tmp,npers,ndays)';

% Daily strucutre
lags = [1; 5; 22];
maxlags = max(lags);

%=== Dep var====
y = data_tmp(maxlags*npers + 1:end);

%====Get indep vars now===

% Previous intrady interval
oki1 = data_tmp(maxlags*npers:end - 1);

% no fo full days after taking out daily lags
ndays1 = ndays - maxlags; 

% daily HAR lags
okd = zeros(ndays1*npers,1);
okw = zeros(ndays1*npers,1);
okm = zeros(ndays1*npers,1);

for i = 1:ndays1
    
    okd((i-1)*npers + 1:i*npers) = mean(data_tmp1(maxlags + i -1,:));
    tmpw = data_tmp1(maxlags + i - lags(2):maxlags + i - 1,:);
    okw((i-1)*npers + 1:i*npers) = mean(reshape(tmpw,lags(2)*npers,1));
    tmpm = data_tmp1(maxlags + i - lags(3):maxlags + i - 1,:);
    okm((i-1)*npers + 1:i*npers) = mean(reshape(tmpm,lags(3)*npers,1));
    
end

% Get the diurnal pattern
muvol = repmat(mean(data_tmp1)',ndays1,1);

% Do the regression
mdl = fitlm([muvol oki1 okd okw okm],y,'linear');
coeff = mdl.Coefficients.Estimate;
beta_hat = coeff;

% end of estimation

% ==== Forecast ====
fore_out = zeros(npers,1);

% First period
okd = mean(data_tmp1(end,:));
tmpw = data_tmp1(end - lags(2) + 1:end,:);
okw = mean(reshape(tmpw,lags(2)*npers,1));
tmpm = data_tmp1(end - lags(3) + 1:end,:);
okm = mean(reshape(tmpm,lags(3)*npers,1));
    
x_tmp = [1 muvol(1) data_tmp(end) okd okw okm];

fore_out(1) = x_tmp*beta_hat;

% Remaining periods
for j = 2:npers
   
    x_tmp = [1 muvol(j) final_day(j-1) okd okw okm];
    
    fore_out(j) = x_tmp*beta_hat;
    
end


end  % end of forecasting function for each day


