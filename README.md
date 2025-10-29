# Intraday_Spot_Volatility_IJF_Replication
For replicating the IBM results for 'Modeling and Forecasting Intraday Spot Volatility'

Each section below relates to sets of results in the paper. There is a folder in the repository (with the same name as each section) that contains the relevant data, code and output.

## Data
Raw data file: IBM.N5MinuteOHLC.csv

Use sort_5minute_data.m (used Matlab 2024b) to filter the raw 5-minute data

Output for the 5-minute OK volatility is written to IBM_5minvol_Win.mat to use in subsequent Matlab code

5-minute OK volatility is laos stored in IBM_ok.csv

## In-sample and Figure 1
