# Summary of relevant papers and other resources 
## Online resources
### Wikipedia 
- https://en.wikipedia.org/wiki/Solar_cycle_25
- https://en.wikipedia.org/wiki/Sunspot

### US National Oceanic and Atmospheric Administration
- background: https://www.swpc.noaa.gov/phenomena/sunspotssolar-cycle
- https://www.swpc.noaa.gov/products/solar-cycle-progression
- Website with interactive plots of sunspot and radio flux predictions
- "Data" section which has .json files with sunspot numbers and predictions
    - Provides monthly indices and daily observations 
    - Also links to Belgian [datasource](https://wwwbis.sidc.be/silso/datafiles) (SILSO)
    - **TODO** we should check if the data lines up with other sources
- "Impacts" section has nice write-ups about impact of sunspots on: 
    - Space Weather and GPS Systems
    - Electric Power Transmission
    - HF Radio Communications
    - Satellite Drag
- The solar cycle prediction on the page is the "official" forecast by NOAA, NASA and Space Environmental Service (SOS)
    - I couldn't find details about the prediction model on the website, and might need to dig deeper
    - [This page](https://www.nasa.gov/feature/goddard/2020/what-will-solar-cycle-25-look-like-sun-prediction-model) seems to suggest that the predictions come from a "polar magnetic field model". 
        - "This uses measurements of the magnetic field at the Sun’s north and south poles. The idea is that the magnetic field at the Sun’s poles acts like a seed for the next cycle. If it’s strong during solar minimum, the next solar cycle will be strong; if it’s diminished, the next cycle should be too."

### Miscellaneous 
- Blog post with predictive models using ARIMA and LSTMs: https://towardsdatascience.com/modelling-the-number-of-sunspots-with-time-series-analysis-39ce7d88cff3
- Tutorial on fitting gaussian process kernel using TensorFlow https://peterroelants.github.io/posts/gaussian-process-kernel-fitting/
- Different kernels https://www.cs.toronto.edu/~duvenaud/cookbook/

## Papers
### Modelling and Prediction of Sunspot Cycles [2001]
- This is a Doctoral thesis submitted at MIT to the department of mathematics in 2001 
- Focuses on predicting magnitudes of next cycle maximum and minimum, time from minimum to maximum (rise time) and maximum to next cycle minimum (fall time)
- Proposes a parsimonious regression model for maximum and rise time 
- Benchmark models are described in the "Simulations" section and include
    - autoregressive (AR)
    - subset AR (SAR)
    - threshold AR transformed by squaring (TTAR)
    - bi-linear model 
    - adaptive spline threshold regression (ASTAR)
    - a neural network model (CNAR)
    - fundamental model proposed by solar physicists
- Uses MLL, AIC, BIC to select between models. The latter two won't be relevant to us if we use GPs since they need a parameter count. 
- Background and introduction section has some nice stuff we could use
    - Increased solar activity resulted in the Skylab station (launched in 1973, weighting over 100 tons) to fall into the earth's atmosphere in 1979. It disintegrated and fell into the Indian Ocean and Western Australia. 
    - A solar storm on July 14, 2000 likely caused a Japanese scientific satellite to spin out of control and end up with solar its solar panels not facing the sun. 
    - At least two official sunspot numbers are reported: Belgian and US. 
    - Previous attempts at modelling the cycles, such as using Fourier analysis, have struggled due to the inconsistent periodicity. For example, the "Maunder minimum" was a period of very low activity between 1653-1715
- Hathaway, Wilson and Reichmann's models
    - $f(t) = a(t - t_0)^3 / (e^((t - t_0)^2 / b^2) - c)$
    - Goes through some suggest parameterisations, see page 54 or the reference
    - This might be the "fundamental solar physicists" model 
- Chapter 11 (pg 107), "Comparison of the models", has some observed undesirable properties of models, which we might also want to look out for in ours 

### Prediction of the Maximum Amplitude and Timing of Sunspot Cycle 24 [2009]
- Describes a "precursor" technique
    - "The high level of geomagnetic activity occurs not only at sunspot maximum but also in the following two to four years, thereby supporting the idea of the ‘extended solar cycle’ where a solar cycle really begins some years before solar minimum and where two solar cycles co-exist on the Sun for a number of years."
    - Uses annual geomagnetic data (*aa* indices) to predict the amplitude of the maximum annual mean sunspot number of solar cycle 24
    - Method based on "Prediction of the Amplitude of Sunspot Cycle 23, R. Jain, 1997"
    - Data seems to be from: https://www.ngdc.noaa.gov/geomag/data.shtml. See "2. Data" section on pg226
- The technique doesn't seem to be interesting, but using geomagnetic data to enhance sunspot prediction model might be worth exploring if we have the time, and if the data is amenable 

### Prediction of sunspot number amplitude and solar cycle length for cycles 24 and 25 [2011]
- Model uses spectral analysis and regression interactive method of monthly and annual averages of sunspot numbers to predict solar activity 
- Uses data after 1850 since it claims this is more reliable 
- Model is of the form: 
    - $f(t) = \sum_(i=1)^N r_i sin(\frac{2\pi}{T_i} + \phi_i)$
- To improve prediction, they also use a filtering procedure using wavelet analysis, decomposing the sunspot number into 10 spectral bands using an orthonormal Meyer wavelet
- Table 1 has a list of predictions for the cycle 24 peak for 18 other methodologies. Some predictions come with error bars. The predictions vary quite a lot (from 74 to 180). We could use this metric (maximum in cycle) to compare against other techniques, if we cannot replicate the more granular fits for each model - i.e. if we cannot get monthly numbers from each model. 

### Prediction of sunspot and plage coverage for Solar Cycle 25 [2021]
- Presents here a new approach that uses more than 100 years of measured fractional areas of the visible solar disk covered by sunspots and plages and an empirical relationship for each of these two indices of solar activity in even-odd cycles.
- Model produces annual predictions. 

### Predicting Sunspot Numbers for Solar Cycles 25 and 26 [2021]
- Uses a modified logistic function, TMLP model 

### Sunspot Prediction using Neural Networks
- Seems like a very old paper. Uses a fully connected network

### Sunspot cycle prediction using Warped Gaussian process regression [2019]
- A "warped" GP is just a regular GP, but with the prediction mapped through a non-linear, monotonic function, to avoid predicting negative values. For example, tanh function, Box-Cox transformation, knots and splines. 
- Uses the RBF kernel in GP. 

### Forecasting peak smooth sunspot number of solar cycle 25 A method based on even-odd pair of solar cycle
- The sun has a magnetic cycle period of around 22 years, which is around double that of a solar cycle. This observation leads them to group pairs of solar cycles 
- They argue that the area under the curve of SSN is proportional to the peak SSN (corr 0.91). So they estimate the sum of each pair's peaks and their total length. 

### Long-Term Sunspot Number Prediction based on EMD Analysis and AR Model [2008]
- EMD: Empirical Mode Decomposition; aka Hilbert-Huang Transformation 


## Notes and ideas
- Use proxy sunspot data in model: aa index, 10.7cm solar flux, phage (bright spots)
- Use periodic kernel to estimate / confirm the length of average solar cycle 