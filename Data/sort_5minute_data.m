%==========================================================================
%
% Script to process raw 5 minute OHLC data
%
%==========================================================================

clc
clear all

% Read in raw 5 minute OHLC data
formatSpec = '%s %s %s %s %s %f %f %f %f';
temp_dat = readtable('IBM.N5MinuteOHLC.csv','Format',formatSpec,'ReadVariableNames',true);

% Convert dates/times to Matlab numbers
dts_tms = datenum(temp_dat.Date_Time,'yyyy-mm-ddTHH:MM:SS.FFF');
dts_text5 = datestr(dts_tms,'dd/mm/yyyy');
dts5 = datenum(dts_text5,'dd/mm/yyyy');
tms_text5 = datestr(dts_tms,'HH:MM:SS');
tms5 = datenum(tms_text5,'HH:MM:SS');

% Extract OHLC prices
open5 = temp_dat.Open;
high5 = temp_dat.High;
low5 = temp_dat.Low;
close5 = temp_dat.Last;

% Remove observations outside 9:30 and 15:55
tmstmp = (tms5 >= datenum('9:30:00','HH:MM:SS'))...
     & (tms5 <= datenum('15:55:00','HH:MM:SS'));
 
 dts_text5 = dts_text5(tmstmp,:);
 tms_text5 = tms_text5(tmstmp,:);
 dts5 = dts5(tmstmp);
 tms5 = tms5(tmstmp);
 open5 = open5(tmstmp);
 high5 = high5(tmstmp);
 low5 = low5(tmstmp);
 close5 = close5(tmstmp); 
 
  
% Check unique days and count 5 mins in each day
udts5 = unique(dts5);
mincount = zeros(length(udts5),1);
for i = 1:length(udts5);
   
    mincount(i) = sum(dts5 == udts5(i));
    
end

% Remove days with less than 78 intervals
udts5 = udts5(mincount == 78);
mincount = mincount(mincount == 78);

% Store variables
on_ret = zeros(length(udts5),1);
rv = zeros(length(udts5),1);
rets = zeros(length(udts5),1);

rets5 = zeros(length(udts5),78);
ok5 = zeros(length(udts5),78);
range5 = zeros(length(udts5),78);



for i = 1:length(udts5)
    
    tmp_op = open5(dts5 == udts5(i));
    tmp_cl = close5(dts5 == udts5(i));
    tmp_h = high5(dts5 == udts5(i));
    tmp_l = low5(dts5 == udts5(i));
            
    if i > 1
        on_ret(i) = log(tmp_op(1)/yest_close);
    end
    
    % Store today close for next overnight return
    yest_close = tmp_cl(end);
    
    % Get 5 minute returns and range
    r_t = log(tmp_cl./tmp_op);    
    if sum(abs(r_t) > 0.025) > 0
        r_t(abs(r_t) > 0.025) = 0; % Just to remove effect of any errors in prices
        tmp_cut = find(abs(r_t) > 0.025);
    end
    rets5(i,:) = r_t';
    range5(i,:) = log(tmp_h)' - log(tmp_l)';
    
    
    % Replace volatility if a bad return is set to zero
    if sum(abs(r_t) > 0.025) > 0
        ok5(i,:) = 0.811.*range5(i,:) - 0.369*abs(rets5(i,:));
        ok5(i,tmp_cut) = 0;
    else
        ok5(i,:) = 0.811.*range5(i,:) - 0.369*abs(rets5(i,:));                
    end
    
    % Daily RV and returns
    rv(i) = sum(r_t.^2);
    rets(i) = sum(r_t);
               
end

% Winsorize - following Zhang et. al. (2023)
ok5_all = reshape(ok5',length(udts5)*78,1); 
top = prctile(ok5_all,99.95);
bot = prctile(ok5_all,0.05);
ok5_all(ok5_all >= top) = top;
ok5_all(ok5_all <= bot) = bot;

save("IBM_5minvol_Win.mat",'ok5','ok5_all','on_ret', 'range5', 'rets', 'rets5','rv','udts5')



