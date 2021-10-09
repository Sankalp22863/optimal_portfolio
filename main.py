# pip install streamlit pystan = 2.19.9.1 fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


# Load Packages
import numpy as np
import pandas as pd
from pandas_datareader import data
from datetime import date

import matplotlib.pyplot as plt



START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Portfolio Optimization')

# Create a page dropdown 
page = st.sidebar.selectbox("Choose your page", ["View Stocks.", "Future Stock price prediction.", "Portfolio Reccomendations."]) 

stocks = ('OFSS.NS', 'ZEEL.NS', 'IDBI.NS', 'BHARTIARTL.NS')

translate = {"OFSS.NS":'Oracle', "ZEEL.NS" :  'Zee', "IDBI.NS" : 'IDBI', "BHARTIARTL.NS" : 'Airtel'}

@st.cache
def load_data(ticker):
	data = yf.download(ticker, START, TODAY)
	data.reset_index(inplace=True)
	return data
	


if page == "View Stocks.":

	selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
	data = load_data(selected_stock)
	# n_years = st.sidebar.slider('Years of prediction:', 1, 4)
	# period = n_years * 365
	"The Stock Prices of ", translate[selected_stock], "are as follows : "

	if selected_stock == 'OFSS.NS':
		st.image('./Oracle-logo.png')
	elif selected_stock == 'ZEEL.NS':
		st.image('./Zee-logo.png')
	elif selected_stock == 'IDBI.NS':
		st.image('./IDBI-logo.png')
	else:
		st.image('./Airtel-logo.png')
		
	data_load_state = st.text('Loading data...')
	data = load_data(selected_stock)
	data_load_state.text('Loading data... done!')

	st.subheader('Stock Prices')
	st.write(data.tail())

	# Plot raw data
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
		fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
		
	plot_raw_data()


if page == "Future Stock price prediction.":

	selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
	data = load_data(selected_stock)

	n_years = st.sidebar.slider('Years of prediction:', 1, 4)
	period = n_years * 365

	# Predict forecast with Prophet.
	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
	df_train.dropna()

	m = Prophet()
	m.fit(df_train)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)

	# Show and plot forecast
	st.subheader('Forecast data')
	st.write(forecast.tail())
		
	st.write(f'Forecast plot for {n_years} years')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

	st.write("Forecast components")
	fig2 = m.plot_components(forecast)
	st.write(fig2)

if page == "Portfolio Reccomendations.":

	t = {"Oracle":'OFSS.NS', "Zee" :  'ZEEL.NS', "IDBI" : 'IDBI.NS', "Airtel" : 'BHARTIARTL.NS'}
	n_years = st.sidebar.slider('The Peirod of investment :', 1, 4)
	period = n_years * 365
	amount = st.sidebar.text_input('Enter the Amount you want to invest(Rs./anum).')

	metrics = st.sidebar.multiselect("What Stocks would you like to invest in?", ("Oracle", "Zee", "IDBI", "Airtel"))

	"So the Optimal portfolio for the Target Investment of ", amount*n_years, "is being calculated."

	data_load_state = st.text('Predicting the Data...')
	portfolio = pd.DataFrame([])
	for i in metrics:
		data = load_data(t[i])
		df_train = data[['Date','Close']]
		df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
		m = Prophet()
		m.fit(df_train)
		future = m.make_future_dataframe(periods=period)
		forecast = m.predict(future)
		portfolio["Date"] = forecast['ds']
		portfolio[i] = forecast['trend']

		
	if metrics != []:
		data_load_state.text('Predicting data... done!')
		portfolio.set_index('Date', inplace=True)
		portfolio.index = pd.to_datetime(portfolio.index, errors='coerce')
		# st.write(portfolio.head())

		# Finding the Covariance Matrix.
		cov_matrix = portfolio.pct_change().apply(lambda x: np.log(1+x)).cov()

		st.write("Covariance Matrix (x10^-6) :")
		st.write(cov_matrix*10**6)

		# Finding the Corellation Matirx.
		corr_matrix = portfolio.pct_change().apply(lambda x: np.log(1+x)).corr()


		st.write("Corellation Matrix :")
		st.write(corr_matrix)


		# Yearly returns for individual companies
		ind_er = portfolio.resample('Y').last().pct_change().mean()

		# # Portfolio returns
		# w = [0.1, 0.2, 0.5, 0.2]
		# port_er = (w*ind_er).sum()

		# Volatility is given by the annual standard deviation. We multiply by 250 because there are 250 trading days/year.
		ann_sd = portfolio.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

		assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
		assets.columns = ['Returns', 'Volatility']

		p_ret = [] # Define an empty array for portfolio returns
		p_vol = [] # Define an empty array for portfolio volatility
		p_weights = [] # Define an empty array for asset weights

		num_assets = len(portfolio.columns)
		num_portfolios = 10000

		for _ in range(num_portfolios):
			weights = np.random.random(num_assets)
			weights = weights/np.sum(weights)
			p_weights.append(weights)
			returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
											# weights 
			p_ret.append(returns)
			var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
			sd = np.sqrt(var) # Daily standard deviation
			ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
			p_vol.append(ann_sd)

		data = {'Returns':p_ret, 'Volatility':p_vol}

		for counter, symbol in enumerate(portfolio.columns.tolist()):
			#print(counter, symbol)
			data[symbol+' weight'] = [w[counter] for w in p_weights]

		portfolios  = pd.DataFrame(data)

		st.write(portfolios.head())

		# Plot efficient frontier
		
		# portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

		# Plot raw data
		def plot_data():
			# Plotting optimal portfolio

			fig = plt.figure(figsize=(10, 10))
			ax = fig.gca()
			ax.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3)
			ax.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
			ax.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
			st.pyplot(fig)

		min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
		# idxmin() gives us the minimum value in the column specified.  
		"Minimum Volume Portfolio :"                             
		min_vol_port*100
		

		# Finding the optimal portfolio
		rf = 0.01 # risk factor
		optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
		"Optimal Risky Portfolio : "
		optimal_risky_port*100


		"So the Money investment portfolio will be as follows : "
		"For Minimum Volume Portfolio :"
		"Investing total of ", amount

		for i in metrics:
			i, ": Rs. ", float(min_vol_port[str(i) + ' weight'])*int(amount), " per Year"

		"For Optimal Risky Portfolio :"
		"Investing total of ", amount

		for i in metrics:
			i, ": Rs. ", float(optimal_risky_port[str(i) + ' weight'])*int(amount), " per Year"
		

		plot_data()



	else:
		data_load_state.text('Please Select at least one stock.')

