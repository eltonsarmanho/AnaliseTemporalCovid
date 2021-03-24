'''
Created on 4 de abr de 2020

@author: elton
'''
# importing the required libraries
import pandas as pd

# Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
import ipywidgets as widgets
from IPython import display


# Manipulating the default plot size
plt.rcParams['figure.figsize'] = 10, 12

# Disable warnings
import warnings

warnings.filterwarnings('ignore')
import plotly
import plotly.io as pio

import plotly.graph_objects as go

plotly.io.renderers.default = 'colab'
import plotly.express as px

from plotly.subplots import make_subplots
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.dates import DateFormatter
from datetime import date, timedelta
from datetime import datetime

plt.rcParams["savefig.dpi"] = 300
# params = {'legend.fontsize': 14,
#          'legend.handlelength': 2}
params = {'legend.fontsize': 14,
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)


# matplotlib.use('TkAgg')
class PreditionCovid(object):

    def TraningSet(self):
        dataframe = pd.read_csv("/home/elton/Documents/Corona Virus/CovidModel/covidBR.csv", sep=';')
        # Run simulation
        return dataframe;

    def __init__(self):
        self.path_image = "/home/elton/Documents/Corona Virus/Corona Figuras/{0}";

        # url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        url = "/home/elton/Documents/Corona Virus/CovidModel/dataset/owid-covid-data.csv"
        data = pd.read_csv(url, sep=',', parse_dates=['date'])
        #

        list_date = []
        for d in data['date']:
            list_date.append(d - timedelta(days=1))
        data['date'] = list_date;

        df_confirmed = data[['location', 'date', 'total_cases', 'new_cases']]
        df_deaths = data[['location', 'date', 'total_deaths', 'new_deaths', 'total_cases']]

        df_confirmed = df_confirmed[df_confirmed['location'] == 'Brazil']
        df_deaths = df_deaths[df_deaths['location'] == 'Brazil']

        filter = df_confirmed['total_cases'] >= 100
        df_confirmed = df_confirmed[filter]
        df_deaths = df_deaths[filter]
        df_confirmed = df_confirmed[(df_confirmed['date'] <= '2020-05-15')]
        df_deaths = df_deaths[(df_deaths['date'] <= '2020-05-15')]

        # df_confirmed = df_confirmed[(df_confirmed['date'] >= '2020-01-20') & (df_confirmed['date'] <= '2020-05-15')]
        # df_deaths = df_deaths[(df_deaths['date'] >= '2020-01-20') & (df_deaths['date'] >= '2020-05-15')]

        # self.plotAcumulado(df_confirmed,df_deaths)

        confirmed = df_confirmed.groupby('date').sum()['total_cases'].reset_index()
        confirmed.columns = ['ds', 'y']
        confirmed['ds'] = pd.to_datetime(confirmed['ds'])

        ##Limitar a partir do Caso numero 100
        self.runForest(confirmed)
        pass;

    def runForecast_with_setTraning(self, confirmed):
        traningSet = self.TraningSet()
        # create a series with the cummulative number of cases
        # traningSet = traningSet[ (traningSet['Days']  <= "2020-06-01")]
        # y = traningSet['Infected']
        # x =  pd.to_datetime(traningSet['Days'])
        y = confirmed['y']
        x = confirmed['ds']
        dataframe = pd.DataFrame(columns=['y', 'ds']);
        dataframe['y'] = y;
        dataframe['ds'] = x;
        ############################################################
        m = Prophet(interval_width=0.95, growth='linear')
        m.fit(dataframe)

        ###########################################################
        # datelist = pd.date_range(start="2020-05-23",end="2020-05-30")
        # teste = pd.DataFrame(columns=['ds' ]);
        # teste['ds'] = datelist

        future = m.make_future_dataframe(periods=7)
        future.tail()

        # predicting the future with date, and upper and lower limit of y value
        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        # forecast = m.predict(teste)

        ###########################################################
        dataframe = pd.DataFrame(columns=['ds', 'y', 'yhat', 'trend']);
        # dataframe['ds'] = confirmed['ds']
        # dataframe['y'] = confirmed['y']
        dataframe['yhat'] = forecast['yhat']
        dataframe['trend'] = forecast['trend']

        ###########################################################
        # plt.plot(confirmed['ds'],confirmed['y'],'*',color='g')
        plt.plot(forecast['ds'], forecast['yhat'], '*')

        myFmt = DateFormatter("%d/%m/%Y")
        plt.gca().xaxis.set_major_formatter(myFmt)
        plt.ylabel('Casos', fontsize=18)
        plt.xlabel('Período', fontsize=16)
        # plt.legend(['Casos Acumulados Observados','Casos Acumulados Estimados','Intervalos de Incerteza'])
        plt.legend(['Dado Observado', 'Dado Estimado'])
        plt.title("Dados Modelo SEIR para Treinamento no cenário do COVID-19")
        # plt.xticks(rotation='vertical')
        plt.show()

    def runForest(self, df):
        df.columns = ['ds', 'y']

        limite_saturacao = 1000 * 1000  # Brasil
        print("limite_saturacao: %s " % (limite_saturacao))
        df['cap'] = limite_saturacao
        m = Prophet(interval_width=0.95, growth='logistic')
        m.fit(df)

        future = m.make_future_dataframe(periods=8)
        future.tail()
        future['cap'] = limite_saturacao

        # predicting the future with date, and upper and lower limit of y value
        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        print(forecast[['ds', 'yhat']].tail(10))
        dt_forecast_plot = m.plot(forecast)
        myFmt = DateFormatter("%d/%m/%Y")
        plt.gca().xaxis.set_major_formatter(myFmt)
        plt.ylabel('Casos', fontsize=30)
        plt.xlabel('Período', fontsize=30)
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=27)

        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(100000))
        # plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(100000))

        plt.legend(['Casos Acumulados', 'Casos Acumulados Estimados', 'Ponto de Saturação', 'Intervalos de Incerteza'])
        figure = plt.gcf()
        figure.set_size_inches(20, 18)
        url = self.path_image.format("Figura_2_1.svg")
        plt.savefig(url, dpi=300)
        # plt.legend(loc=2, prop={'size': 14})
        plt.show()

        # self.runValidation(m)

        return forecast, m;

    def runValidation(self, model):
        df_cv = cross_validation(model, initial='50 days', period='2 days', horizon='14 days')

        df_p = performance_metrics(df_cv)

        fig = plot_cross_validation_metric(df_cv, metric='mape')
        plt.ylabel('Mean Absolute Percentage Error (MAPE)', fontsize=16)
        plt.xlabel('Forecast Horizon (Dias)', fontsize=16)
        plt.legend(['Erro Percentual Absoluto', 'MAPE'])
        figure = plt.gcf()
        figure.set_size_inches(12, 9)
        url = self.path_image.format("Figura__3.svg")
        plt.savefig(url, dpi=300)
        plt.show()

    def plotAcumulado(self, df_confirmed, df_deaths):
        # df_confirmed = df_confirmed[df_confirmed['date'] >= '2020-02-26']
        # df_deaths = df_deaths[df_deaths['date'] >= '2020-02-26']

        parametro_1 = 'total_cases';
        parametro_2 = 'total_deaths';
        confirmed = df_confirmed.groupby('date').sum()[parametro_1].reset_index()
        deaths = df_deaths.groupby('date').sum()[parametro_2].reset_index()

        fig = go.Figure()

        fig = px.bar(confirmed, x="date", y=parametro_1, color=parametro_1, orientation='v', height=600,
                     title='', color_discrete_sequence=px.colors.cyclical.IceFire)
        fig.update_layout(showlegend=True, font=dict(
            family="sans-serif",
            size=18,
            color="black"), xaxis=dict(title='Período'),

                          xaxis_tickfont_size=14, yaxis_tickfont_size=14, xaxis_tickformat='%d/%m/%Y',
                          yaxis=dict(title='Quantidade'))

        url = self.path_image.format("Figura_0.svg")
        fig.write_image(url, width=1000, height=800)
        fig.show(renderer='browser')


if __name__ == '__main__':
    run = PreditionCovid()




