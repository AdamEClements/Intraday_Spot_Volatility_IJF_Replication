clc
clear all

load('IBM_5minvol_Win.mat');

data_tmp = ok5_all;

npers = 78;
ndays = length(udts5);

data_tmp1 = reshape(data_tmp,npers,ndays)';

% Daily strucutre
lags = [1; 5; 22];
nlags = length(lags);
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
beta_hat = coeff';

yhat = mdl.Fitted;
resid = y - yhat; 

tout = table(y, yhat, resid);
writetable(tout,'IBM_diHAR_resids.csv','WriteVariableNames',true)


