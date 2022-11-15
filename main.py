# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:25:11 2022

@author: Naresh Sampara (PhD)
"""
# Import the libraries 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Load the Data 
WholeNHSDataAE=pd.read_csv('WholeNHSDataAE.csv')
WholeNHSDataAE.set_index(WholeNHSDataAE['DATE'])

# Functions
from statsmodels.tsa.stattools import adfuller
def StationaryOrNot(Data,Window_Size,Graph_Title):
    # Rolling data 
    RolMean=Data.rolling(window=Window_Size).mean()
    RolSTD=Data.rolling(window=Window_Size).std()
    
    # Augumented Dickey-fuller test
    ADF_result=adfuller(Data,autolag='AIC') 
    ADF_Stats=pd.Series(ADF_result[0:4],index=['Test stats','p-value','#Lags used','Number of Observation used'])
    for key, values in ADF_result[4].items():
        ADF_Stats['Critical value (%s)'%key]=values
    

        
    return RolMean,RolSTD,ADF_result, ADF_Stats

# Descrptive Analysis 
x1=WholeNHSDataAE['DATE'] # dates
y1=WholeNHSDataAE['AE_Total_Attend'] # total attendance to the A&E
y2=WholeNHSDataAE['Total Emergency Admissions'] # Patients admitted in the Emergency
y3=WholeNHSDataAE['4to12hrs_Admission'] #  waiting time is 4 to 12 hrs 
y4=WholeNHSDataAE['More12hrAdmission'] # waiting time is more than 12hrs

fig_1=make_subplots(rows=2,cols=2,
                    subplot_titles=('Total patient attended to A&E',
                                    'Emergency Admissions',
                                    'Waiting time: 4 to 12 hrs',
                                    'Waiting time : more than 12 hrs'))
fig_1.update_layout(autosize=True)

fig_1.add_trace(go.Scatter(x=x1,y=y1),row=1,col=1)
fig_1.add_trace(go.Scatter(x=x1,y=y2),row=1,col=2)
fig_1.add_trace(go.Scatter(x=x1,y=y3),row=2,col=1)
fig_1.add_trace(go.Scatter(x=x1,y=y4),row=2,col=2)
fig_1.update(layout_showlegend=False)
fig_1['layout']['xaxis3']['title']='Date'
fig_1['layout']['xaxis4']['title']='Date'
fig_1['layout']['yaxis']['title']='Number of Patients'
fig_1['layout']['yaxis3']['title']='Number of Patients'
# Admission to the hospital
fig_2=go.Figure()

#fig_2.add_trace(go.Scatter(x=x1,y=y1,name='A&E total attendance'))
fig_2.add_trace(go.Scatter(x=x1,y=WholeNHSDataAE['AE_Attend_Type1'],name='A&E Type 1  attendance'))
fig_2.add_trace(go.Scatter(x=x1,y=WholeNHSDataAE['AE_Attend_Type2'],name='A&E Type 2 attendance'))
fig_2.add_trace(go.Scatter(x=x1,y=WholeNHSDataAE['AE_Attend_Type3'],name='A&E Type 3 attendance'))

fig_2.update_layout(xaxis_title='Date',
                    yaxis_title='Number of Patients')

# Emergency Admission
fig_3=go.Figure()
fig_3.add_trace(go.Scatter(x=x1,y=y2-(y3+y4),name='Waiting time less than 4hrs'))
fig_3.add_trace(go.Scatter(x=x1,y=y3,name='Waiting time 4hrs to 12hr'))
fig_3.add_trace(go.Scatter(x=x1,y=y4,name='Waiting time more than 12hrs'))
fig_3['layout']['xaxis']['title']='Date'
fig_3['layout']['yaxis']['title']='Number of Emergency Admissions'


# build dashboard
add_sidebar=st.sidebar.selectbox('Whole or individual NHS A&E Data',('NHS A&E Data','Individual Hospital Data'))

# Whole Data analysis
if add_sidebar=='NHS A&E Data':
    st.title('NHS UK A&E Data')
    
    # Descriptive ANalaysis of the A&E Data
    st.write('The data, including monthly A&E attendeances and emergency' 
             'admission, has obtained from NHS England site')
       
    st.plotly_chart(fig_1,use_container_width=True)
    
    # Time series analysis of each Data
    DataType=st.selectbox('A&E Attendance or Emergency Admission',('A&E Attenedance','Emergency Admission'))
    
    if DataType=='A&E Attenedance':
        # Descriptive Analysis. 
        
        
        # Attendance data 
        st.plotly_chart(fig_2)
        
        # Time Series Analysis 
        y=WholeNHSDataAE['AE_Total_Attend']
        y.index=pd.to_datetime(WholeNHSDataAE['DATE'],format='%Y-%m-%d')
        #y=y.sort_index(inplace= True)
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition=seasonal_decompose(y)
        RolMean=y.rolling(window=12).mean()

        trend=decomposition.trend
        seasonal=decomposition.seasonal
        residual=decomposition.resid
        
        fig_5=make_subplots(rows=4,cols=1,
                            subplot_titles=('Total patient attended to A&E',
                                            'Trend',
                                            'Seasonality',
                                            'Residual'))
        fig_5.add_trace(go.Scatter(x=x1,y=y),row=1,col=1)
        fig_5.add_trace(go.Scatter(x=x1,y=RolMean),row=1,col=1)
        fig_5.add_trace(go.Scatter(x=x1,y=trend),row=2,col=1)
        fig_5.add_trace(go.Scatter(x=x1,y=seasonal),row=3,col=1)
        fig_5.add_trace(go.Scatter(x=x1,y=residual),row=4,col=1)
        fig_5.update_layout(autosize=False, width=700, height=1000,showlegend=False)
        
        st.plotly_chart(fig_5)
        
        # Fitting the model
        y=np.log(y)
        y_train=y[y.index < pd.to_datetime('2019-02-01',format='%Y-%m-%d')]
        y_test=y[y.index > pd.to_datetime('2019-02-01',format='%Y-%m-%d')]

        # Stationarity check: ADF augument dickey-fuller test
        RolMean,RolSTD,ADF_result, ADF_Stats=StationaryOrNot(y_train, 12, 'Original Data')

        # changing the Data
        y_train_diff=y_train.diff().dropna()
        RolMean,RolSTD,ADF_result, ADF_Stats=StationaryOrNot(y_train_diff, 12, 'Original Data')
        d=1
        # trend, Seasonality and random 
        
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        SARIMAXmodel=SARIMAX(y_train,order=(4,1,2), seasonal_order=(2,2,2,12))
        SARIMAXmodel=SARIMAXmodel.fit()

        y_pred=SARIMAXmodel.get_forecast(len(y_test))
        y_pred_df=y_pred.conf_int(alpha=0.05)
        y_pred_df['Prediction']=SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
        y_pred_df.index = y_test.index
        y_pred_out = y_pred_df["Prediction"] 
        
        fig_4=go.Figure()
        fig_4.add_trace(go.Scatter(x=x1,y=np.exp(y),name='A&E attendance'))
        fig_4.add_trace(go.Scatter(x=y_pred_out.index,y=np.exp(y_pred_out),name='Prediction of A&E attendance'))
        fig_4.add_trace(go.Scatter(x=y_test.index,y=np.exp(y_test),line_color='black',name='A&E attendance after covid'))
        fig_4.add_vline(x="2019-03-01",line_width=2,line_color='black',
                        line_dash='dash')
        fig_4.add_vrect(x0="2019-03-01", x1="2022-10-01", 
                        annotation_text="COVID", annotation_position="top left",
                        opacity=0.05, line_width=1,annotation=dict(font_size=16,
                                                                   font_family="Times New Roman"))
        fig_4['layout']['xaxis']['title']='Date'
        fig_4['layout']['yaxis']['title']='Number of Patients'
        fig_4.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,
                                        xanchor="right",x=1))
        
        st.plotly_chart(fig_4)
    
    if DataType=='Emergency Admission':
        # Emergency admission
        st.plotly_chart(fig_3)
        
        # Time Series Analysis 
        y=WholeNHSDataAE['Total Emergency Admissions']
        y.index=pd.to_datetime(WholeNHSDataAE['DATE'],format='%Y-%m-%d')
        #y=y.sort_index(inplace= True)
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition=seasonal_decompose(y)

        trend=decomposition.trend
        seasonal=decomposition.seasonal
        residual=decomposition.resid
        
        fig_6=make_subplots(rows=4,cols=1,
                            subplot_titles=('Total patient attended to A&E',
                                            'Trend',
                                            'Seasonality',
                                            'Residual'))
        fig_6.add_trace(go.Scatter(x=x1,y=y),row=1,col=1)
        fig_6.add_trace(go.Scatter(x=x1,y=trend),row=2,col=1)
        fig_6.add_trace(go.Scatter(x=x1,y=seasonal),row=3,col=1)
        fig_6.add_trace(go.Scatter(x=x1,y=residual),row=4,col=1)
        fig_6.update_layout(autosize=False, width=700, height=1000,showlegend=False)
        
        st.plotly_chart(fig_6)
        
        # Fitting the model
        y=np.log(y)
        y_train=y[y.index < pd.to_datetime('2019-02-01',format='%Y-%m-%d')]
        y_test=y[y.index > pd.to_datetime('2019-02-01',format='%Y-%m-%d')]

        # Stationarity check: ADF augument dickey-fuller test
        RolMean,RolSTD,ADF_result, ADF_Stats=StationaryOrNot(y_train, 12, 'Original Data')

        # changing the Data
        y_train_diff=y_train.diff().dropna()
        RolMean,RolSTD,ADF_result, ADF_Stats=StationaryOrNot(y_train_diff, 12, 'Original Data')
        d=1
        # trend, Seasonality and random 
        
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        SARIMAXmodel=SARIMAX(y_train,order=(4,1,2), seasonal_order=(2,2,2,12))
        SARIMAXmodel=SARIMAXmodel.fit()

        y_pred=SARIMAXmodel.get_forecast(len(y_test))
        y_pred_df=y_pred.conf_int(alpha=0.05)
        y_pred_df['Prediction']=SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
        y_pred_df.index = y_test.index
        y_pred_out = y_pred_df["Prediction"] 
        
        fig_7=go.Figure()
        fig_7.add_trace(go.Scatter(x=x1,y=np.exp(y),name='Emergency Admission'))
        fig_7.add_trace(go.Scatter(x=y_pred_out.index,y=np.exp(y_pred_out),name='Prediction of Emergency Admission'))
        fig_7.add_trace(go.Scatter(x=y_test.index,y=np.exp(y_test),line_color='black',name='Emergency admission after covid'))
        fig_7.add_vline(x="2019-03-01",line_width=2,line_color='black',
                        line_dash='dash')
        fig_7.add_vrect(x0="2019-03-01", x1="2022-10-01", 
                        annotation_text="COVID", annotation_position="top left",
                        opacity=0.05, line_width=1,annotation=dict(font_size=16,
                                                                   font_family="Times New Roman"))
        fig_7['layout']['xaxis']['title']='Date'
        fig_7['layout']['yaxis']['title']='Number of Patients'
        fig_7.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,
                                        xanchor="right",x=1))
        
        st.plotly_chart(fig_7)
        
if add_sidebar=='Individual Hospital Data':
    Data=pd.read_csv('NHS_A&E_Indvidual_Hosp_Data.csv')
    #seperate most visited hospitals
    Hosp=Data.groupby(by=['Org_Name']).sum()
    HospitalList=Hosp.loc[Hosp['AE_atte_Type1']>300000,'AE_atte_Type1'].sort_values(ascending=False).index
    st.title('Hospital A&E Data Analysis')
    Hospitals=st.selectbox('List of Hospitals',(HospitalList),index=13)
    
    st.write('The hospital data the ')
    
    Org_Data=Data.loc[Data['Org_Name']==Hospitals,:]
    Org_Data=Org_Data.groupby(by=['Date'],dropna=False).sum()
    
    Org_Data['Total A&E Attendance']=Org_Data[['AE_atte_Type1', 'AE_atte_Type2', 'AE_atte_Other']].sum(axis=1)
    Org_Data['Total Emergency Admision']=Org_Data[['Emg_Admis_AE_Type1', 'Emg_Admis_AE_Type2', 'Emg_Admis_AE_Other']].sum(axis=1)

    X1=Org_Data.index
    Y1=Org_Data['Total A&E Attendance']
    Y2=Org_Data['Total Emergency Admision']
    Y3=Org_Data['4to12hs_to_admis']
    Y4=Org_Data['More12hs_to_admis']
    
    fig_8=make_subplots(rows=2,cols=2,
                        subplot_titles=('Total patient attended to A&E',
                                        'Emergency Admissions',
                                        'Waiting time: 4 to 12 hrs',
                                        'Waiting time : more than 12 hrs'))
    fig_8.update_layout(autosize=True)
    fig_8.update(layout_showlegend=False)
    fig_8.add_trace(go.Scatter(x=X1,y=Y1),row=1,col=1)
    fig_8.add_trace(go.Scatter(x=X1,y=Y2),row=1,col=2)
    fig_8.add_trace(go.Scatter(x=X1,y=Y3),row=2,col=1)
    fig_8.add_trace(go.Scatter(x=X1,y=Y4),row=2,col=2)
    
    st.plotly_chart(fig_8)
