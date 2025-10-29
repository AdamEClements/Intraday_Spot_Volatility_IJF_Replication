setwd("_____")


# FiT/Act/res + acf plots
tmp = read.csv("IBM_miHAR_resids.csv", header = TRUE,stringsAsFactors = FALSE)
nobs = nrow(tmp)
y_act = tmp$yact
mi_fit = tmp$yhat
mi_resids = tmp$resid

tmp = read.csv("IBM_diHAR_resids.csv", header = TRUE,stringsAsFactors = FALSE)
di_fit = tmp$yhat
di_resids = tmp$resid

tmp = read.csv("IBM_gbm_resid.csv", header = TRUE,stringsAsFactors = FALSE)
gbm_y = tmp$y
gbm_fit = tmp$fit
gbm_resids  = gbm_y - gbm_fit

tmp = read.csv("IBM_LSTM_resid.csv", header = TRUE,stringsAsFactors = FALSE)
lstm_y = tmp$y_all
lstm_fit = tmp$yfit
lstm_resids  = lstm_y - lstm_fit

maxlag = 780
y_act_acf = acf(y_act, maxlag, type = c("correlation"), plot = FALSE)
mi_resids_acf = acf(mi_resids, maxlag, type = c("correlation"), plot = FALSE)
di_resids_acf = acf(di_resids, maxlag, type = c("correlation"), plot = FALSE)
gbm_resids_acf = acf(gbm_resids, maxlag, type = c("correlation"), plot = FALSE)
lstm_resids_acf = acf(lstm_resids, maxlag, type = c("correlation"), plot = FALSE)

# Dataframe with ACFs
df_acf <- data.frame(y_act_acf$acf[2:(maxlag+1)], mi_resids_acf$acf[2:(maxlag+1)], di_resids_acf$acf[2:(maxlag+1)], gbm_resids_acf$acf[2:(maxlag+1)], lstm_resids_acf$acf[2:(maxlag+1)])
colnames(df_acf) <- c('y', 'mi', 'di', 'gbm', 'lstm')
write.csv(df_acf, file = "SACFs.csv", row.names = FALSE)

