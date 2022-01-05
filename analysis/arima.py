# %%
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# %%
data_path = '../data/SILSO data/SN_m_tot_V2.0.csv'
data = pd.read_csv(data_path, sep=';', names=['year', 'month', 'year_frac', 'ssn_total', 'ssn_stdev', 'nobs', 'marker'])

plt.figure(figsize=(30, 8))
plt.scatter(range(len(data)), data['ssn_total'])

# %% ARIMA model 
from statsmodels.tsa.arima_model import ARIMA

y = np.array(data['ssn_total'])
model = ARIMA(y, order=(1,0,1)) #ARMA(1,1) model
model_fit = model.fit(disp = 0)

print(model_fit.summary())# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2, figsize=(30, 8))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()# Actual vs Fitted

cut_t = 30
predictions = model_fit.predict()
plot = pd.DataFrame({'Date':data.loc[cut_t:, 'year_frac'],'Actual':abs(y[cut_t:]),"Predicted": predictions[cut_t:]})
plot.plot(x='Date',y=['Actual','Predicted'],title = 'ARMA(1,1) Sunspots Prediction',legend = True, figsize=(30,8))
RMSE = np.sqrt(np.mean(residuals**2))  

# %%
forecast = model_fit.forecast(steps=12*3)

# %%
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(forecast[0])
ax.fill_between(range(len(forecast[0])), *zip(*forecast[2]), alpha=0.2)

# %%
