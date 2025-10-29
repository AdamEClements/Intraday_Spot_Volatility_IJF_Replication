clc
clear all

load('IBM_5minvol_Win.mat');

data_tmp = ok5_all;

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


% Order: intercepts, HAR, overnight + intraday
npars = 1 + nlags + 1 + i_nlags;  

% Store coeffs
beta_hat_new = zeros(npers,npars);
beta_hatlasso_new = zeros(npers,npars);
se_new = zeros(npers,npars);
var_robust = zeros(npers,npars);
covar_robust = zeros(npars*npers, npars);
covar_ols = zeros(npars*npers, npars);
 
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

mdl = fitlm(x,y,'linear');
coeffcov = mdl.CoefficientCovariance;
coeff = mdl.Coefficients.Estimate;
beta_hat_new(i,1:end) = coeff';
se = mdl.Coefficients.SE;
se_new(i,1:end) = se';
covar_ols((i-1)*npars+1:i*npars, :) = coeffcov;

[EstCoeffCov,se,coeff_hac] = hac(x,y, Display="off");

var_robust(i,:) = diag(EstCoeffCov);
covar_robust((i-1)*npars+1:i*npars, :) = EstCoeffCov;

yhat(index_tmp,:) = mdl.Fitted';

for i = 2:npers
   
    index_tmp = (maxlags*npers+1:npers:npers*ndays)' + i - 1;
    y = data_tmp(index_tmp);
    
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
    
    mdl = fitlm(x,y,'linear');
    coeffcov = mdl.CoefficientCovariance;
    coeff = mdl.Coefficients.Estimate;
    beta_hat_new(i,:) = coeff';
    se = mdl.Coefficients.SE;
    se_new(i,1:end) = se';
    covar_ols((i-1)*npars+1:i*npars, :) = coeffcov;

    [EstCoeffCov,se,coeff_hac] = hac(x,y, Display="off");
    var_robust(i,:) = diag(EstCoeffCov);
    covar_robust((i-1)*npars+1:i*npars, :) = EstCoeffCov;

    yhat(index_tmp,:) = mdl.Fitted;
       
end

yact = data_tmp;
resid = yact - yhat; 
resid = resid(npers*maxlags+1:end);
yact = yact(npers*maxlags+1:end);
yhat = yhat(npers*maxlags+1:end);

% Write out resdiuals
tout = table(yact, yhat, resid);
writetable(tout,'IBM_miHAR_resids.csv','WriteVariableNames',true)

% OLS Std Errs
% tout_se = table(se_new);
% writetable(tout_se,'IBM_miHAR_se.csv','WriteVariableNames',true)

% OLS coeffs
betas = beta_hat_new;
tout = table(betas);
writetable(tout,'IBM_OLS_estimates.csv','WriteVariableNames',true)


% Robust Cov matrix (scale to write to file)
covar_robust = covar_robust.*1e6;
tout = table(covar_robust);
writetable(tout,'IBM_robust_scaled_covars.csv','WriteVariableNames',true)
