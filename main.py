# pip install streamlit pystan = 2.19.9.1 fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# import tkinter as tk
# from tkinter import ttk

from streamlit_player import st_player


# Load Packages
import numpy as np
import pandas as pd
from pandas_datareader import data
from datetime import date

import matplotlib.pyplot as plt

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()


@st.cache
def pred(metrics):
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


st.set_page_config(
    page_title="Portfolio Reccomendation App",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto"
)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Portfolio Optimization')
# st.write("[![Star](<https://img.shields.io/github/stars/><username>/<repo>.svg?logo=github&style=social)](<https://gitHub.com/><Sanklp22863>/<Optimal Portfolio>)")

# Create a page dropdown 
page = st.sidebar.selectbox("Choose your page", ["View Stocks.", "Future Stock price prediction.", "Portfolio Reccomendations."]) 

stocks = ('Oracle', 'Zee', 'IDBI', 'Airtel', 'HDFC Bank', 'Reliance', 'Adani Ports', 'ONGC', 'ITC', 'SBI')

t = {"Oracle":'OFSS.NS', "Zee" :  'ZEEL.NS', "IDBI" : 'IDBI.NS', "Airtel" : 'BHARTIARTL.NS', 'HDFC Bank' : 'HDFCBANK.NS', 'Reliance' : 'RELIANCE.NS', 'Adani Ports' : 'ADANIPORTS.NS', 
"ONGC" : 'ONGC.NS', 'ITC' : 'ITC.NS', 'SBI' : 'SBIN.NS'}

@st.cache
def load_data(ticker):
	data = yf.download(ticker, START, TODAY)
	data.reset_index(inplace=True)
	return data
	


if page == "View Stocks.":

	selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)

	# data = load_data(t[selected_stock])
	# n_years = st.sidebar.slider('Years of prediction:', 1, 4)
	# period = n_years * 365
	

	img_path = './' + selected_stock + '-logo.png'

	col1, col2, col3 = st.columns(3)

	try:
		col2.image(img_path)
	except:
		col2.image('./placeholder.gif')

	"The Stock Prices of ", selected_stock, "are as follows : "
		
	# data_load_state = st.text('Loading data...')
	data = load_data(t[selected_stock])
	# data_load_state.text('Loading data... done!')

	col1, col2, col3, col4 = st.columns(4)

	col1.metric(label = "Open", value = "Rs. " + str(round(data.iloc[-1]["Open"], 2)), delta = str(round(data.iloc[-1]["Open"] - data.iloc[-2]["Open"], 2)))
	col2.metric(label = "High", value = "Rs. " + str(round(data.iloc[-1]["High"], 2)), delta = str(round(data.iloc[-1]["High"] - data.iloc[-2]["High"], 2)))
	col3.metric(label = "Low", value = "Rs. " + str(round(data.iloc[-1]["Low"], 2)), delta = str(round(data.iloc[-1]["Low"] - data.iloc[-2]["Low"], 2)))
	col4.metric(label = "Close", value = "Rs. " + str(round(data.iloc[-1]["Close"], 2)), delta = str(round(data.iloc[-1]["Close"] - data.iloc[-2]["Close"], 2)))

	st.subheader('Stock Prices')
	st.write(data.tail())

	def plot_stock(df, name):
		fig = go.Figure()
		config = {'responsive': True}
		fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
		st.plotly_chart(fig)

	# Plot raw data
	def plot_raw_data():
		fig = go.Figure()
		
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
		fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
		
	# plot_raw_data()
	plot_stock(data, selected_stock)


if page == "Future Stock price prediction.":

	selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
	data = load_data(t[selected_stock])

	n_years = st.sidebar.slider('Years of prediction:', 1, 4)
	period = n_years * 365

	# Predict forecast with Prophet.
	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

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
	amount = 0
	metrics = []
	# popupmsg("Mutual Funds are subject to market risk.")

	# Embed a music from SoundCloud
	# st_player("https://soundcloud.com/imaginedragons/demons")

	n_years = st.sidebar.slider('The Period of investment :', 1, 4)
	period = n_years * 365
	amount = st.sidebar.text_input('Enter the Amount you want to invest(Rs./anum).')
	no_of_stocks = st.sidebar.number_input('Enter the No. of stocks you want to invest.' , min_value=1, max_value=10, value=4)

	try:
		if len(metrics) < int(no_of_stocks):
			metrics = st.sidebar.multiselect("What Stocks would you like to invest in?", stocks)
	except:
		pass


	data_load_state = st.text('Select appropriate options.')
	
	
	try:
		if int(amount) > 0 and len(metrics) >= int(no_of_stocks):
			"Meanwhile get some advice about the investments."
			# Embed a youtube video
			st_player("https://www.youtube.com/watch?v=jnJI6vp5b7o;&autoplay=1")
	except:
		pass

	try:
		if len(metrics) == int(no_of_stocks):
			portfolio = pd.DataFrame([])
			data_load_state.text('Predicting data...')
			pred(metrics)
	

		if True:	
			try:
				"So the Optimal portfolio for the Target Investment of ", int(amount)*int(n_years), "is being calculated."
			except:
				"Please Enter an amount you want to invest per Anum."

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
				# Finding the optimal portfolio
				rf = 0.01 # risk factor
				optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
				col1, col2 = st.columns(2)
				col1.write("Minimum Volume Portfolio :")
				col2.write("Optimal Risky Portfolio : ")
				col1.dataframe(min_vol_port*100)
				col2.dataframe(optimal_risky_port*100)


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
				data_load_state.text('Please Select at stocks.')

	except:
		pass
