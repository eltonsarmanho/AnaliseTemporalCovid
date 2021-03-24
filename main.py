'''
Created on 4 de abr de 2020

@author: elton
'''
# importing the required libraries
from IPython import display
from datetime import datetime, timedelta
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot
from fbprophet.plot import plot_cross_validation_metric
from folium import plugins
import folium
import matplotlib
from matplotlib.dates import DateFormatter
import plotly
from plotly.subplots import make_subplots
import warnings

import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.express as px
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns


# Visualisation libraries
# Manipulating the default plot size
plt.rcParams['figure.figsize'] = 10, 12

# Disable warnings
warnings.filterwarnings('ignore')

plotly.io.renderers.default = 'colab'
plt.rcParams["savefig.dpi"] = 300
#params = {'legend.fontsize': 14,
#          'legend.handlelength': 2}
params = {'legend.fontsize': 14,
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
#matplotlib.use('TkAgg')
class ForescastCovid(object):


    def __init__(self):
        self.path_image = "/home/elton/Documents/Corona Virus/Corona Figuras/{0}";
    def forescasting(self):
        url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        #url = "/home/elton/Documents/CoronaVirus/CovidModel/dataset/owid-covid-data.csv"

        #df = pd.read_csv("/home/elton/Documents/Corona Virus/CovidModel/covidBR.csv",sep=';',parse_dates=['data'])
        df = pd.read_csv(url,sep=',',parse_dates=['date'])

        df_confirmed = df[['location','date','total_cases','new_cases']]
        df_deaths = df[['location','date','total_deaths','new_deaths','total_cases']]

        df_confirmed = df_confirmed[df_confirmed['location'] == 'Brazil']
        df_deaths = df_deaths[df_deaths['location'] == 'Brazil']

        filter = (df_confirmed['date'] > '2020-03-14') & (df_confirmed['date'] <= '2020-05-15')
        filter = (df_confirmed['date'] > '2020-03-14')
        print(filter.count())
        df_confirmed = df_confirmed[filter]
        print(df_confirmed)
        df_deaths = df_deaths[filter]
        #print(pio.renderers)
        parametro_1 = 'total_cases';
        parametro_2 = 'total_deaths';
        confirmed = df_confirmed.groupby('date').sum()[parametro_1].reset_index()
        deaths = df_deaths.groupby('date').sum()[parametro_2].reset_index()

        layout = go.Layout(height=900,width=1800,autosize=False)
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(x=confirmed['date'], y=confirmed[parametro_1], mode='lines+markers',
                                 name='Confirmados',line=dict(color='royalblue', width=2)))
        fig.add_trace(go.Scatter(x=deaths['date'], y=deaths[parametro_2], mode='lines+markers',
                                  name='Óbitos', line=dict(color='firebrick', width=2)))


        forecast, model = self.runForest(confirmed)
        values  = np.array(forecast['yhat'],dtype=int)
        values =  list(map(lambda x: 0 if x < 0 else x, values))

        fig.add_trace(go.Scatter(x=forecast['ds'], y= values,opacity=0.5,mode='lines', name='Previsão',
                                 line=dict(color='royalblue', width=2,dash='dash')));

        #fig.add_annotation(x='2020-04-13', y=22367, text="dict Text")
        fig.add_annotation(x='2020-05-04', y=37559,xref="x",yref="y",text="20/04/2020<br>Estimado: 110.896<br>Observado: 107.780",showarrow=True,
                           font=dict(family="Courier New, monospace",size=16,color="#ffffff"),
                           align="center", arrowhead=4,arrowsize=3,arrowwidth=4,arrowcolor="#636363",
                           ax=30,ay=-20,bordercolor="#c7c7c7", borderwidth=3,borderpad=4,bgcolor="#ff7f0e",opacity=0.65)

        #29.349
        fig.add_annotation(x='2020-05-07', y=51285,xref="x",yref="y",text="22/04/2020<br>Estimado: 135.245<br>Observado: 135.106",showarrow=True,
                           font=dict(family="Courier New, monospace",size=16,color="#ffffff"),
                           align="center", arrowhead=3,arrowsize=2,arrowwidth=2,arrowcolor="#636363",
                           ax=-40,ay=30,bordercolor="#c7c7c7", borderwidth=2,borderpad=4,bgcolor="green",opacity=0.65)
        #33113
        fig.add_annotation(x='2020-05-10', y=65000,xref="x",yref="y",text="24/04/2020<br>Estimado: 162.479<br>Observado: 162.699",showarrow=True,
                           font=dict(family="Courier New, monospace",size=16,color="#ffffff"),
                           align="center", arrowhead=2,arrowsize=1,arrowwidth=2,arrowcolor="#636363",
                           ax=-40,ay=30,bordercolor="#c7c7c7", borderwidth=2,borderpad=4,bgcolor="olive",opacity=0.65)



        fig.update_layout( showlegend=True,
                          title='Evolução do N° de Casos e Óbitos (Acumulado) no Brasil', font=dict(
            family="sans-serif",
            size=18,
            color="black"),xaxis=dict(title='Período'),
                          xaxis_tickfont_size=16,yaxis_tickfont_size=16,xaxis_tickformat = '%d/%m/%Y',yaxis=dict(title='Quantidade'))

        #fig.show(renderer='browser')
        #fig.show(renderer='png')
        #fig.write_image("/home/elton/Pictures/Corona Figuras/Figura 5.svg")
        #############################################################

    def plotAcumulado(self):
        url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        url = "/home/elton/Documents/CoronaVirus/CovidModel/dataset/owid-covid-data.csv"
        data = pd.read_csv(url,sep=',',parse_dates=['date'])

        list_date = []
        for d in data['date'] :
            list_date.append(d - timedelta(days=1))
        data['date'] =list_date;

        df_confirmed = data[['location','date','total_cases','new_cases']]
        df_deaths = data[['location','date','total_deaths','new_deaths','total_cases']]

        df_confirmed = df_confirmed[df_confirmed['location'] == 'Brazil']
        df_deaths = df_deaths[df_deaths['location'] == 'Brazil']


        #df_confirmed = df_confirmed[df_confirmed['data'] <= '2020-04-19']
        #df_deaths = df_deaths[df_deaths['data'] <= '2020-04-19']

        #filter = df_confirmed['total_cases'] >= 100
        #df_confirmed = df_confirmed[filter]
        #df_deaths = df_deaths[filter]

        df_confirmed = df_confirmed[df_confirmed['date'] >= '2020-02-26']
        df_deaths = df_deaths[df_deaths['date'] >= '2020-02-26']

        parametro_1 = 'total_cases';
        parametro_2 = 'total_deaths';
        confirmed = df_confirmed.groupby('date').sum()[parametro_1].reset_index()
        deaths = df_deaths.groupby('date').sum()[parametro_2].reset_index()

        fig = go.Figure()

        fig = px.bar(confirmed, x="date", y=parametro_1, color=parametro_1, orientation='v', height=600,
             title='', color_discrete_sequence = px.colors.cyclical.IceFire)
        fig.update_layout( showlegend=True, font=dict(
            family="sans-serif",
            size=18,
            color="black"),xaxis=dict(title='Período'),

                          xaxis_tickfont_size=14,yaxis_tickfont_size=14,xaxis_tickformat = '%d/%m/%Y',yaxis=dict(title='Quantidade'))


        url = self.path_image.format("Figura_1.svg")
        print(url)
        fig.write_image(url,width=1000, height=800)
        fig.show(renderer='browser')

    def runForest(self,df):

        df.columns = ['ds','y']
        max= np.max(df['y'])
        df['ds'] = pd.to_datetime(df['ds'])


        limite_saturacao = max+425*1000#Brasil
        #limite_saturacao = max+20*1000#Brasil
        print("limite_saturacao: %s " % (limite_saturacao))
        df['cap']  = limite_saturacao
        m = Prophet(interval_width=0.95,mcmc_samples=300,growth='logistic')
        m.fit(df)


        print("M (off set): %s" % m.params['m'][599])
        print("K: %s "% m.params['k'][599])
        print("limite_saturacao: %s " % (limite_saturacao))

        #print(len(m.params['k']))
        future = m.make_future_dataframe(periods=60)
        future.tail()
        future['cap'] = limite_saturacao

        #predicting the future with date, and upper and lower limit of y value
        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        fig = plot(m,forecast,figsize=(10,10))

        ax = fig.gca()
        myFmt = DateFormatter("%d/%m/%Y")
        plt.gca().xaxis.set_major_formatter(myFmt)

        plt.ylabel('Casos', fontsize=18)
        plt.xlabel('Período', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(['Casos Acumulados','Casos Acumulados Estimados','Ponto de Saturação','Intervalos de Incerteza'])
        figure = plt.gcf()
        #figure.set_size_inches(30, 30)
        plt.savefig("/home/elton/Pictures/Corona Figuras/Figura 2.svg",dpi=300)
        plt.legend(loc=2, prop={'size': 15})

        plt.show()



        #self.runValidation(m)


        return  forecast,m;

    def runValidation(self,model):
        #df_cv = cross_validation(model, initial='60 days', period='1 days', horizon = '14 days')
        df_cv = cross_validation(model, initial='42 days', period='3 days', horizon = '14 days')


        fig = plot_cross_validation_metric(df_cv, metric='mape')
        plt.ylabel('Erro  Médio  Absoluto Percentual (MAPE)',fontsize=18)
        plt.xlabel('Horizonte (Dias)',fontsize=18)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(['Erro Percentual Absoluto','MAPE'])
        figure = plt.gcf()
        plt.savefig("/home/elton/Pictures/Corona Figuras/Figura 4.svg",dpi=300)
        plt.show()



if __name__ == '__main__':
    run =  ForescastCovid()
    #run.plotAcumulado()
    run.forescasting()



