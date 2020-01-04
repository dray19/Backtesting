import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import plotly as py
import plotly.graph_objs as go
#pip install plotly==4.4.1
from IPython.display import clear_output
from plotly.graph_objs import *
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

class total_return:
    def __init__(self, df):
        self.df = df
    
    def moving_avg(self):
        self.df['SP_6m'] = round(self.df.SP_price.rolling(window = 6).mean(),2)
        self.df['HS_6m'] = round(self.df.HS_price.rolling(window = 6).mean(),2)
        self.df['Gold_6m'] = round(self.df.Gold_price.rolling(window = 6).mean(),2)
        self.df['FTSE_6m'] = round(self.df.FTSE_price.rolling(window = 6).mean(),2)
        self.df['DJ_6m'] = round(self.df.DJ_price.rolling(window = 6).mean(),2)
        self.df['AGG_6m'] = round(self.df.AGG_price.rolling(window = 6).mean(),2)
        
        self.df['SP_1y'] = round(self.df.SP_price.rolling(window = 12).mean(),2)
        self.df['HS_1y'] = round(self.df.HS_price.rolling(window = 12).mean(),2)
        self.df['Gold_1y'] = round(self.df.Gold_price.rolling(window = 12).mean(),2)
        self.df['FTSE_1y'] = round(self.df.FTSE_price.rolling(window = 12).mean(),2)
        self.df['DJ_1y'] = round(self.df.DJ_price.rolling(window = 12).mean(),2)
        self.df['AGG_1y'] = round(self.df.AGG_price.rolling(window = 12).mean(),2)   
        
        
    def select_dates(self, start, end):
        self.start2 = start
        self.end2 = end
        self.df4 = self.df[(self.df['Date'] >= start) & (self.df['Date'] <= end)]
        return(self.df4)
    
    def inf_rate(self):
        df5 = self.df4[['CPI_PCH','Date']]
        df5['Inflation_Rate'] = df5['CPI_PCH'].cumsum().shift().fillna(0)
        df6 = df5[['Date', 'Inflation_Rate']]
        self.df4.drop('CPI_PCH', inplace = True, axis = 1)
        return(self.df4.merge(df6, on = 'Date'))
    
    def plot_avg(self,df,price,avg_6m,avg_1y, index):
        trace1 = go.Scatter(x = df.Date,y = price,mode = "lines",name = index + " Price",marker = dict(color = 'black'))
        trace2 = go.Scatter(x = df.Date,y = avg_6m,mode = "lines",name = index + " 6 month Average",marker = dict(color = 'red'))
        trace3 = go.Scatter(x = df.Date,y = avg_1y,mode = "lines",name = index +  " 1 Year Average",marker = dict(color = 'blue'))
        data = [trace1, trace2, trace3]
        layout = dict(title = index + ' Price vs Moving Average',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis= dict(title= 'Dollars',ticklen= 5,zeroline= False)
                 )
        fig = dict(data = data, layout = layout)
        plotly.offline.iplot(fig)
    
    def plot_moving_avg(self, df, index):
        if index == 'SP500':
            price = df[df.columns[1]]
            avg_6m = df[df.columns[14]]
            avg_1y = df[df.columns[20]]
            self.plot_avg(df, price, avg_6m ,avg_1y, index)
        if index == 'DJ':
            price = df[df.columns[9]]
            avg_6m = df[df.columns[18]]
            avg_1y = df[df.columns[24]]
            self.plot_avg(df, price, avg_6m ,avg_1y, index)
        if index == 'HS':
            price = df[df.columns[3]]
            avg_6m = df[df.columns[15]]
            avg_1y = df[df.columns[21]]
            self.plot_avg(df, price, avg_6m ,avg_1y,index)
        if index == 'FTSE':
            price = df[df.columns[7]]
            avg_6m = df[df.columns[17]]
            avg_1y = df[df.columns[23]]
            self.plot_avg(df, price, avg_6m ,avg_1y,index)
        if index == 'GOLD':
            price = df[df.columns[5]]
            avg_6m = df[df.columns[16]]
            avg_1y = df[df.columns[22]]
            self.plot_avg(df, price, avg_6m ,avg_1y,index)
        if index == 'AGG':
            price = df[df.columns[11]]
            avg_6m = df[df.columns[19]]
            avg_1y = df[df.columns[25]]
            self.plot_avg(df, price, avg_6m ,avg_1y,index)
            
    def money_value(self, total,w_sp, w_dj, w_hs, w_ft, w_agg, w_gold):
        self.sp500_start = round((total * self.sp500_w),2)
        self.dj_start = round((total * self.dj_w),2)
        self.hs_start = round((total * self.hs_w),2)
        self.ft_start = round((total * self.ft_w),2)
        self.agg_start = round((total * self.agg_w),2)
        self.gold_start = round((total * self.gold_w),2)
        total_w = sum([self.sp500_w,self.dj_w,self.hs_w,self.ft_w,self.agg_w,self.gold_w])
        total_start = sum([ self.sp500_start,self.dj_start,self.hs_start,self.agg_start,self.ft_start,
                          self.gold_start])
        data_matrix = [
                ['S&P500', '${:,.2f}'.format(self.sp500_start),self.sp500_w ],
               ['DJ', '${:,.2f}'.format(self.dj_start),self.dj_w ],
               ['HS', '${:,.2f}'.format(self.hs_start), self.hs_w],
               ['FTSE', '${:,.2f}'.format(self.ft_start), self.ft_w],
               ['AGG', '${:,.2f}'.format(self.agg_start), self.agg_w],
               ['Gold', '${:,.2f}'.format(self.gold_start), self.gold_w],
                      ['Total', '${:,.2f}'.format(total_start), total_w]]
        st_data = pd.DataFrame(data_matrix, columns = ['Index', 'Cash', 'Percent'])
        print(st_data.to_string(index = False))
        print('==' * 10)
    
    def start(self, total,sp500, dj, hs, ftse, agg, gold):
        data_mat = [[self.start2, self.end2]]
        data2 = pd.DataFrame(data_mat, columns = ['Start Date', 'End Date'])  
        print(data2.to_string(index = False))

        print('=='*10)
        print('Starting Point')
        self.sp500_w = sp500
        self.dj_w = dj
        self.hs_w = hs
        self.ft_w = ftse
        self.agg_w = agg
        self.gold_w = gold
        total = total

        self.money_value(total,self.sp500_w, self.dj_w, self.hs_w, self.ft_w,self.agg_w ,self.gold_w)
        us_w = sum([self.dj_w, self.sp500_w])
        us_m = sum([self.dj_start, self.sp500_start])
        inter_w = sum([self.hs_w, self.ft_w])
        inter_m = sum([self.hs_start, self.ft_start])
        
        
        mat1 = [['US', us_w, '${:,.2f}'.format(us_m)],
               ['INTER', inter_w, '${:,.2f}'.format(inter_m)],
               ['Bonds',self.agg_w,'${:,.2f}'.format(self.agg_start)],
               ['Gold',self.gold_w,'${:,.2f}'.format(self.gold_start)]]
        data3 = pd.DataFrame(mat1, columns = ['Sector', 'Percent', 'Cash'])
        print(data3.to_string(index = False))

        
    def start_value(self):
        self.sp500_value = self.sp500_start
        self.hs_value = self.hs_start
        self.gold_value = self.gold_start
        self.ft_value = self.ft_start
        self.dj_value = self.dj_start
        self.agg_value = self.agg_start
        self.total_start = sum([self.sp500_value,self.hs_value, self.gold_value, self.ft_value, self.dj_value, self.agg_value])
        self.df_null = pd.DataFrame()
        self.df_null = self.df_null.append({'Date':self.df4.iloc[0][0], "return_sp":self.sp500_start, "return_dj" :self.dj_start,
                                     "return_hs" :self.hs_start,"return_ftse" :self.ft_start,
                                     "return_agg" : self.agg_start,"return_gold" :self.gold_start,
                                     "diff_10_2":self.df4.iloc[0][13], 'total':self.total_start, 
                                            'Inflation': round(self.total_start,2),
                                           'inflation_adjusted_return':round(self.total_start,2)}, ignore_index = True)
    def allo(self, total):
        self.sp500_w = float(input('S&P 500:'))
        self.dj_w = float(input('DJ:'))
        self.hs_w = float(input('HS:'))
        self.ft_w = float(input('FTSE:'))
        self.agg_w = float(input('AGG:'))
        self.gold_w = float(input('Gold:'))
        self.sp500_value = round((total * self.sp500_w),2)
        self.dj_value = round((total * self.dj_w),2)
        self.hs_value = round((total * self.hs_w),2)
        self.ft_value = round((total * self.ft_w),2)
        self.agg_value = round((total * self.agg_w),2)
        self.gold_value = round((total * self.gold_w),2)
        total2 = sum([self.sp500_value,self.hs_value, self.gold_value, 
                                      self.ft_value, self.dj_value, self.agg_value])
        
        self.df_null = self.df_null.append({'Date':'allocation', 
                                        "return_sp":self.sp500_value, "return_dj" :self.dj_value,
                                         "return_hs" :self.hs_value,"return_ftse" :self.ft_value,
                                         "return_agg" :self.agg_value,"return_gold" :self.gold_value,
                                         "diff_10_2":'NA', 'total':total2,'Inflation':'NA' ,
                                           'inflation_adjusted_return': 'NA'}, ignore_index = True)
        
        
        us_w = sum([self.dj_w, self.sp500_w])
        us_m = sum([self.dj_start, self.sp500_value])
        inter_w = sum([self.hs_w, self.ft_w])
        inter_m = sum([self.hs_start, self.ft_value])
        mat40 = [['US', us_w, '${:,.2f}'.format(us_m)],
               ['INTER', inter_w, '${:,.2f}'.format(inter_m)],
               ['Bonds',self.agg_w,'${:,.2f}'.format(self.agg_value)],
               ['Gold',self.gold_w,'${:,.2f}'.format(self.gold_value)]]
        data40 = pd.DataFrame(mat40, columns = ['Sector', 'Percent', 'Cash'])
        print(data40.to_string(index = False))
        print('--'*40)
        print(' ')
        time.sleep(0.5)
        
        
    def reall(self, total):
        self.sp500_value = round((total * self.sp500_w),2)
        self.dj_value = round((total * self.dj_w),2)
        self.hs_value = round((total * self.hs_w),2)
        self.ft_value = round((total * self.ft_w),2)
        self.agg_value = round((total * self.agg_w),2)
        self.gold_value = round((total * self.gold_w),2)
        print('Reallocation')
        mat30 = [['S&P500', '${:,.2f}'.format(self.sp500_value),self.sp500_w ],
                        ['DJ', '${:,.2f}'.format(self.dj_value),self.dj_w ],
                        ['HS', '${:,.2f}'.format(self.hs_value), self.hs_w],
                        ['FTSE', '${:,.2f}'.format(self.ft_value), self.ft_w],
                        ['AGG', '${:,.2f}'.format(self.agg_value), self.agg_w],
                       ['Gold', '${:,.2f}'.format(self.gold_value), self.gold_w]]
        data30 = pd.DataFrame(mat30, columns = ['Index', 'Cash', 'Percent'])
        print(data30.to_string(index = False))
        
        print(' ')
        print('--' * 40)
        total2 = sum([self.sp500_value,self.hs_value, self.gold_value, 
                                      self.ft_value, self.dj_value, self.agg_value])
        self.df_null = self.df_null.append({'Date':'reallocate', 
                                        "return_sp":self.sp500_value, "return_dj" :self.dj_value,
                                         "return_hs" :self.hs_value,"return_ftse" :self.ft_value,
                                         "return_agg" :self.agg_value,"return_gold" :self.gold_value,
                                         "diff_10_2":'NA', 'total':total2,'Inflation': 'NA' ,
                                           'inflation_adjusted_return': 'NA' }, ignore_index = True)
        time.sleep(0.5)
        
    def final_return(self, t, ia , ir):
        print('Final Total')
        mat20 = [['Total Cash', '${:,.2f}'.format(round(t,2))],
                ['Inflation Adjusted Return','${:,.2f}'.format(ia) ],
                ['Inflation on Starting Value','${:,.2f}'.format(ir)]]
        
        data20 = pd.DataFrame(mat20, columns = ['Return', 'Cash'])
        print(data20.to_string(index = False))
        
    def raw_returns(self,df):
        
        self.start_value()
        
        count = 0
        total_count = 0
        month_count = 1
        diff_list = list()
        df = df[1:]
        for i in range(len(df['Date'])):
            dt = df.iloc[i][0]
            sp500 = df.iloc[i][2]
            hs = df.iloc[i][4]
            gold = df.iloc[i][6]
            ft = df.iloc[i][8]
            dj = df.iloc[i][10]
            agg = df.iloc[i][12]
            diff = df.iloc[i][13]
            inf = df.iloc[i][26]
            
            diff_list.append(diff)
        
            
            sp500_cash = round(self.sp500_value * (1 + sp500) ,2)
            self.sp500_value = sp500_cash

            hs_cash = round(self.hs_value * (1 + hs) ,2)
            self.hs_value = hs_cash

            gold_cash = round(self.gold_value * (1 + hs) ,2)
            self.gold_value = gold_cash

            ft_cash = round(self.ft_value * (1 + hs) ,2)
            self.ft_value = ft_cash

            dj_cash = round(self.dj_value * (1 + hs) ,2)
            self.dj_value = dj_cash

            agg_cash = round(self.agg_value * (1 + hs) ,2)
            self.agg_value = agg_cash

            self.time_total = sum([agg_cash, dj_cash, ft_cash, gold_cash, hs_cash, sp500_cash]) 
            self.inf_rate = round(self.total_start * (1 + inf),2)
            self.inf_adj = round(self.time_total / (1 + inf),2)
            self.df_null = self.df_null.append({'Date':dt, "return_sp":self.sp500_value, "return_dj" :self.dj_value,
                                     "return_hs" :self.hs_value,"return_ftse" :self.ft_value,
                                     "return_agg" :self.agg_value,"return_gold" :self.gold_value,
                                     "diff_10_2":diff,'total':self.time_total, 'Inflation': self.inf_rate,
                                               'inflation_adjusted_return':self.inf_adj}, ignore_index = True)
            
            print('Date:', dt)
            print('Month #:', month_count)
            print('10 minus 2:', diff)
            print(' ')
            
            mat2 = [['SP500', '${:,.2f}'.format(self.sp500_value)],
                   ['DJ', '${:,.2f}'.format(self.dj_value)],
                   ['HS','${:,.2f}'.format(self.hs_value)],
                   ['FTSE','${:,.2f}'.format(self.ft_value)],
                   ['Gold', '${:,.2f}'.format(self.gold_value)],
                   ['AGG', '${:,.2f}'.format(self.agg_value)],
                   ['Total cash','${:,.2f}'.format(self.time_total)],
                   ['Inflation Adjusted','${:,.2f}'.format(self.inf_adj)],
                   ['Inflation','${:,.2f}'.format(self.inf_rate)]]
            
            data4 = pd.DataFrame(mat2, columns = ['Index', 'Cash'])
            print(data4.to_string(index = False))
            print('==' * 10)
            
            month_count += 1
            count += 1
            total_count += 1
            sub2 = len(df) - total_count
            time.sleep(1)
            
            
            if count == 6:
                print('--' * 40)
                print(' ')
                print('6 MONTH CHECK')
                print('Date:',dt)
                print('10 - 2:', diff_list)
                print(' ')
                mat4 = [['Total Cash', '${:,.2f}'.format(round(self.time_total,2))],
                       ['Inflation Adjusted Return','${:,.2f}'.format(self.inf_adj) ],
                       ['Inflation on Starting Value','${:,.2f}'.format(self.inf_rate)]]
                data5 = pd.DataFrame(mat4, columns = ['Return', 'Cash'])
                print(data5.to_string(index = False))
                
                diff_list.clear()
                print('==' * 10)
                
                count = 0
                total = self.time_total
                agg_pct = round(agg_cash/total,3)
                sp_pct = round(sp500_cash/total,3)
                dj_pct = round(dj_cash/total,3)
                ft_pct = round(ft_cash/total,3)
                hs_pct = round(hs_cash/total,3)
                gold_pct = round(gold_cash/total,3)
                
                mat5 = [['S&P500', '${:,.2f}'.format(sp500_cash),sp_pct ],
                            ['DJ', '${:,.2f}'.format(dj_cash),dj_pct],
                            ['HS', '${:,.2f}'.format(hs_cash), hs_pct],
                            ['FTSE', '${:,.2f}'.format(ft_cash), ft_pct],
                            ['AGG', '${:,.2f}'.format(agg_cash), agg_pct],
                            ['Gold', '${:,.2f}'.format(gold_cash), gold_pct]]
                data6 = pd.DataFrame(mat5, columns = ['Index', 'Cash', 'Percent'])
                print(data6.to_string(index = False))

                print('==' * 10)
                
                us_w = round(sum([dj_pct, sp_pct]),3)
                us_m = round(sum([dj_cash,    sp500_cash]),2)
                inter_w = round(sum([hs_pct, ft_pct]),3)
                inter_m = round(sum([hs_cash, ft_cash]))
        
        
                mat8 = [['US', us_w, '${:,.2f}'.format(us_m)],
                       ['INTER', inter_w, '${:,.2f}'.format(inter_m)],
                       ['Bonds',agg_pct,'${:,.2f}'.format(agg_cash)],
                       ['Gold',gold_pct,'${:,.2f}'.format(gold_cash)]]
                data7 = pd.DataFrame(mat8, columns = ['Sector', 'Percent', 'Cash'])
                print(data7.to_string(index = False))
                
                df55 = df[:i]
                plt.figure(figsize=(15,7))
                sns.set(style="darkgrid")
                sns.lineplot(data=df55, x='Date',y='diff_10_2', color = 'red',marker="o")
                plt.axhline(y=0, c='black',linewidth=3)
                plt.ylim(-2, 3)
                plt.title('10 minus 2', fontsize = 20)
                plt.xlabel('Date',fontsize = 15)
                plt.ylabel('Percent Difference',fontsize = 15)
                plt.tick_params(labelsize=15)
                
                plt.show()
                
                
                
                
                print('--' * 40)
                sub = len(df) - total_count
                
                if sub != 0:
                    ask2 = str(input('Would you like to try a new allocation:'))
                    if ask2 == 'yes':
                        print(' ')
                        self.allo(total)
                    else:
                        ask = str(input('Would you like to reallocate:'))
                        if ask == 'yes':
                            print(' ')
                            self.reall(total)
                ask3 = str(input('Would you like to clear output:'))
                if ask3 == 'yes':
                    clear_output()
                if ask3 == 'no':
                    print('--' * 40)
                time.sleep(0.5)
            if sub2 == 0:
                print('==' *40)
                self.final_return(self.time_total,self.inf_adj,self.inf_rate)
                        
    def data_return(self):
        self.df_null = self.df_null[['Date', 'return_sp', 'return_dj', 'return_ftse', 'return_hs', 
                                    'return_agg', 'return_gold', 'diff_10_2', 'total', 
                                    'inflation_adjusted_return', 'Inflation']]
        return(self.df_null)
    
    
    
    def plot_return(self):
        df11 = self.df_null[(self.df_null['Date'] != 'reallocate') & (self.df_null['Date'] != 'allocation')]
        df11['Date'] = pd.to_datetime(df11['Date'], format="%Y-%m-%d")
        df12 = df11.drop(['diff_10_2', 'total','inflation_adjusted_return', 'Inflation'], axis = 1)
        df12.columns = ['Date', 'SP500', 'DJ', 'FTSE', 'HS', 'AGG', 'Gold']
        trace1 = Scatter(x = df12.Date,y = df12.SP500,mode = "lines+markers",name = "SP500", marker = dict(color = 'blue'))
        trace2 = Scatter(x = df12.Date,y = df12.DJ,visible=False,mode = "lines+markers",name = "DJ",marker = dict(color = 'orange'))
        trace3 = Scatter(x = df12.Date,y = df12.FTSE,visible=False,mode = "lines+markers",name = "FTSE", marker = dict(color = 'purple'))
        trace4 = Scatter(x = df12.Date,y = df12.HS, visible=False,mode = "lines+markers",name = "HS", marker = dict(color = 'green'))
        trace5 = Scatter(x = df12.Date, y = df12.AGG,visible=False,mode = "lines+markers",name = "AGG",marker = dict(color = 'red'))
        trace6 = Scatter(x = df12.Date,y = df12.Gold,visible=False,mode = "lines+markers",name = "Gold",marker = dict(color = 'gold'))
        data = Data([trace1, trace2, trace3, trace4, trace5, trace6])
        layout = Layout(title='Asset Total Return',xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                          yaxis= dict(title= 'Dollars',ticklen= 5,zeroline= False),
                        updatemenus=list([
                dict(x=-0.05,y=1,yanchor='top',
                buttons=list([
                        dict(args=['visible', [True, False,False,False,False,False]],label='SP500',method='restyle'),
                        dict(args=['visible', [False, True,False,False,False,False]],label='DJ',method='restyle'),
                        dict(args=['visible', [False, False, True,False,False,False]],label='HS',method='restyle'),
                        dict(args=['visible', [False, False,False, True, False,False]],label='FTSE',method='restyle'),
                        dict(args=['visible', [False, False,False,False, True,False]],label='AGG',method='restyle'),
                        dict(args=['visible', [False, False,False,False, False,True]],label='Gold',method='restyle'),
                        dict(args=['visible', [True, True,True,True, True,True]],label='All',method='restyle'),
                    ]),
                    )
            ]),
        ) 
        fig = Figure(data=data, layout=layout)
        py.offline.iplot(fig)
        
    def plot_adj_return(self):
        df11 = self.df_null[(self.df_null['Date'] != 'reallocate') & (self.df_null['Date'] != 'allocation')]
        df11['Date'] = pd.to_datetime(df11['Date'], format="%Y-%m-%d")
            
        trace1 = go.Scatter(x = df11.Date,y = df11.inflation_adjusted_return,mode = "lines+markers",name = "Inflation Adjusted Return",marker = dict(color = 'red'))
        trace2 = go.Scatter(x = df11.Date,y = df11.Inflation,mode = "lines+markers",name = "Inflation",marker = dict(color = 'green'))
        data = [trace1, trace2]
        layout = dict(title = 'Inflation Adjusted Return vs. Inflation ',
                xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                yaxis= dict(title= 'Dollars',ticklen= 5,zeroline= False)
                )
        fig = dict(data = data, layout = layout)
        py.offline.iplot(fig)
               
    def all_one(self, df):
        self.df22 = pd.DataFrame()
        self.df22 = self.df22.append({'Date':self.df4.iloc[0][0], "sp_all":self.total_start, "dj_all" :self.total_start,
                                     "hs_all" :self.total_start,"ftse_all" :self.total_start,
                                     "agg_all" :self.total_start,"gold_all" :self.total_start,
                                      'Inflation': self.total_start}, ignore_index = True)
        sp500_value = self.total_start
        dj_value = self.total_start
        hs_value = self.total_start
        ft_value = self.total_start
        hs_value = self.total_start
        agg_value = self.total_start
        gold_value = self.total_start
        df = df[1:]
        for i in range(len(df['Date'])):
            dt = df.iloc[i][0]
            sp500 = df.iloc[i][2]
            hs = df.iloc[i][4]
            gold = df.iloc[i][6]
            ft = df.iloc[i][8]
            dj = df.iloc[i][10]
            agg = df.iloc[i][12]
            inf = df.iloc[i][26]

            sp500_cash = round(sp500_value * (1 + sp500) ,2)
            sp500_value = sp500_cash
            sp_adj = round(sp500_value / (1 + inf),2)


            hs_cash = round(hs_value * (1 + hs) ,2)
            hs_value = hs_cash
            hs_adj = round(hs_value / (1 + inf),2)

            gold_cash = round(gold_value * (1 + gold) ,2)
            gold_value = gold_cash
            gold_adj = round(gold_value / (1 + inf),2)

            ft_cash = round(ft_value * (1 + ft) ,2)
            ft_value = ft_cash 
            ft_adj = round(ft_value / (1 + inf),2)

            dj_cash = round(dj_value * (1 + dj) ,2)
            dj_value = dj_cash 
            dj_adj = round(dj_value / (1 + inf),2)

            agg_cash = round(agg_value * (1 + agg) ,2)
            agg_value = agg_cash
            agg_adj = round(agg_value / (1 + inf),2)


            inf_rate = round(self.total_start * (1 + inf),2)
            self.df22 = self.df22.append({'Date':dt, "sp_all":sp_adj, "dj_all" :dj_adj,
                                         "hs_all" :hs_adj,"ftse_all" :ft_adj,
                                         "agg_all" :agg_adj,"gold_all" :gold_adj,
                                          'Inflation': inf_rate}, ignore_index = True)
        return(self.df22)
    
    def plot_all(self):
        df11 = self.df_null[(self.df_null['Date'] != 'reallocate') & (self.df_null['Date'] != 'allocation')]
        df11['Date'] = pd.to_datetime(df11['Date'], format="%Y-%m-%d")
        df12 = df11.drop(['return_sp','return_dj','return_ftse', 'return_hs', 
                          'return_agg', 'return_gold','diff_10_2','total','Inflation'], axis = 1)
        df13 = self.df22
        df_all = df12.merge(df13, on = 'Date')
        df_all.rename({'sp_all':'SP500', 'dj_all':'DJ','hs_all':'HS','ftse_all':'FTSE', 'agg_all':'AGG',
                      'gold_all':'Gold'}, inplace = True, axis = 1)
        trace1 = go.Scatter(x = df_all.Date,y = df_all.SP500,mode = "lines",name = "SP500",
                                marker = dict(color = 'purple'))
        trace2 = go.Scatter(x = df_all.Date,y = df_all.DJ,mode = "lines",name = "DJ",
                                marker = dict(color = 'orange'))
        trace3 = go.Scatter(x = df_all.Date,y = df_all.FTSE,mode = "lines",name = "FTSE",
                                marker = dict(color = 'lightblue'))
        trace4 = go.Scatter(x = df_all.Date,y = df_all.HS, mode = "lines",name = "HS",
                                marker = dict(color = 'lightgreen'))
        trace5 = go.Scatter(x = df_all.Date, y = df_all.AGG,mode = "lines",name = "AGG",
                                marker = dict(color = 'pink'))
        trace6 = go.Scatter(x = df_all.Date,y = df_all.Gold,mode = "lines",name = "Gold",
                                marker = dict(color = 'gold'))
        trace7 = go.Scatter(x = df_all.Date,y = df_all.inflation_adjusted_return,mode = "lines",
                                name = "Inflation Adjusted Return",marker = dict(color = 'red',size = 30 ))
        trace8 = go.Scatter(x = df_all.Date,y = df_all.Inflation,mode = "lines",name = "Inflation",
                                marker = dict(color = 'black',size = 30))
        data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8]
        layout = dict(title = 'Retrun if all of Money was in one Asset',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis= dict(title= 'Dollars',ticklen= 5,zeroline= False)
                 )
        fig = dict(data = data, layout = layout)
        py.offline.iplot(fig)