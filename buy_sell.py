from statistics import mean,stdev
import configparser
import pandas as pd
import pytz
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
from oandapyV20.endpoints.trades import TradeDetails, TradeClose
from oandapyV20.exceptions import V20Error
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import copy
import time
import re
import math
from tabulate import tabulate

config = configparser.ConfigParser()
config.read('./data/account.txt')
account_id = config['oanda']['account_id']  # ID
api_key = config['oanda']['api_key']        #トークンパス
ex_pair = config['oanda']['pair']           #対象通貨
lot = config['oanda']['lot']                #lot数 
asi = config['oanda']['asi']                #取得した時間足
bar = config['oanda']['bar']                #予測先
lerndata = config['oanda']['lern']          #学習するデータ数
data_num = config['oanda']['data']

limit_path = 'data/limit.txt'#limit path
config_lim = configparser.ConfigParser()
config_lim.read(limit_path)
lim_up = float(config_lim[ex_pair+'_'+asi+'_'+bar+'_'+lerndata+'_'+data_num]['limit_up'])        #利確位置
lim_down = float(config_lim[ex_pair+'_'+asi+'_'+bar+'_'+lerndata+'_'+data_num]['limit_down'])      #利確位置
#modelパス
loa_path = 'FXmodel'+'_'+ex_pair+'_'+asi+'_'+bar+'_'+lerndata+'_'+data_num+'.pth'

bar = int(bar)
lerndata = int(lerndata)
api = oandapyV20.API(access_token=api_key, environment="live")

def get_mdata(ex_pair, api, asi, bar):
    params1 = {"instruments": ex_pair}
    psnow = pricing.PricingInfo(accountID=account_id, params=params1)
    try:
        now = api.request(psnow) #現在の価格を取得
    except V20Error:
        api = oandapyV20.API(access_token=api_key, environment="live")
        now = api.request(psnow) #現在の価格を取得
    end = now['time']
    params = {"count":35 + lerndata,"granularity":asi,"to":end}
    r = instruments.InstrumentsCandles(instrument=ex_pair, params=params,)
    try:
        apires = api.request(r)
    except V20Error:
        api = oandapyV20.API(access_token=api_key, environment="live")
        apires = api.request(r)
    res = r.response['candles']
    end = res[0]['time']
    data = []
    price = []
    #形を成形
    for raw in res:
        data.append([raw['time'], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])
    #DataFrameに変換
    df = pd.DataFrame(data)
    df.columns = ['date', 'open', 'high', 'low', 'close']
    
    #時間を全て日本時間に変更する。
    for i in df['date']:
        i = pd.Timestamp(i).tz_convert('Asia/Tokyo')

    df.iloc[:, 0] = df.iloc[:, 0].astype('datetime64[ns]')
    df.iloc[:, 1:] = df.iloc[:, 1:].astype('float')
    return df

#MACDの値を返す関数を定義する。（引数はロウソク足の終値）
def macd_data(data):
    #短期と長期の指数平滑移動平均、MACDのリスト、signalのリスト
    Lema, Sema, MACD, signal = [], [], [], []
    num = 0
    num2 = 0
    d = []
    for m in data:
        d.append(m)

    for i in range(26, len(d)):
        if i == 26 and num == 0:
            Lema.append(mean(d[i-26:i]))
            Sema.append(mean(d[i-12:i]))
            MACD.append(Sema[num]-Lema[num])
            num+=1
        elif i != 26:
            Lema.append(Lema[num-1]*(1-(2/27))+d[i-1]*2/27)
            Sema.append(Sema[num-1]*(1-(2/13))+d[i-1]*2/13)
            MACD.append(Sema[num]-Lema[num])
            num+=1
    for i in range(9, len(MACD)):
        if i == 9 and num2 == 0:
            signal.append(mean(MACD[i-9:i]))
            num2+=1
        elif i != 9:
            signal.append(signal[num2-1]*(1-(2/10))+MACD[i-1]*2/10)
            num2+=1
    #MACDとsignalの値を返す
    return MACD, signal
#MACDのシグナルを計算
def macd_signal(MACD, signal):
    MACD_signal = []
    MACD = MACD[9:]
    for i in range(len(MACD)):
        MACD_signal.append(MACD[i] - signal[i])
    return MACD_signal
#dfを成形
def make_df(df, df_all, bar, lerndata):
    bar = int(bar)
    lerndata = int(lerndata)
    #データ作成
    for i in range(lerndata-1):
        if i == 0:
            df_shift = df
        df_shift = df_shift.shift(-1)
        df = pd.concat([df, df_shift], axis=1)
        
    #結合
    sh = df.shape
    df.columns = range(sh[1])
    df = df.dropna(how='any')
    print(df)
    return df

#買い
def buy_signal(now_price, account_id, api, lot, ex_pair, times, lim_up, lim_down, trade_id, flag):
    lot = str(lot)
    profit = now_price + lim_up
    loss = now_price + (lim_down*3)
    profit = str(round(profit, 3))
    loss = str(round(loss, 3))
    #売りポジション決済
    if flag["sell_signal"] == 1:
        #print(trade_id)
        for i in range(len(trade_id)-1):
            data_clo = None
            ticket = TradeClose(accountID=account_id, tradeID=trade_id[i], data=data_clo)
            try:
                api.request(ticket)
            except V20Error:
                pass
        trade_id = []
        flag["sell_signal"] = 0
    data = {
         "order": {
            "instrument": ex_pair,
            "units": "+"+lot,
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {
                "timeInForce": "GTC",
                "price": profit
            },
            "stopLossOnFill": {
                "timeInForce": "GTC",
                "price": loss
            }
        }
    }
    ticket = orders.OrderCreate(account_id, data=data)
    
    try:
        res = api.request(ticket)
        trade_id.append(res['orderCreateTransaction']['id']) #tradeID
        times = 0
        headers = ["profit", "now_price", "loss", "trade"]
        table = [(profit, now_price, loss, "buy")]
        result=tabulate(table, headers)
        print(result)
        flag["buy_signal"] = 1
    except V20Error:
        api = oandapyV20.API(access_token=api_key, environment="live")
        res = api.request(ticket)
        trade_id.append(res['orderCreateTransaction']['id']) #tradeID
        times = 0
        headers = ["profit", "now_price", "loss", "trade"]
        table = [(profit, now_price, loss, "buy")]
        result=tabulate(table, headers)
        print(result)
        flag["buy_signal"] = 1
    
    return trade_id, flag, times
#売り
def sell_signal(now_price, account_id, api, lot, ex_pair, times, lim_up, lim_down, trade_id, flag):
    lot = str(lot)
    profit = now_price + lim_down
    loss = now_price +(lim_up*3)
    profit = str(round(profit, 3))
    loss = str(round(loss, 3))
    #買いポジション決済
    if flag["buy_signal"] == 1:
        #print(trade_id)
        for i in range(len(trade_id)-1):
            data_clo = None
            ticket = TradeClose(accountID=account_id, tradeID=trade_id[i], data=data_clo)
            try:
                api.request(ticket)
            except V20Error:
                pass
        trade_id = []
        flag["buy_signal"] = 0
    data = {
         "order": {
             "instrument": ex_pair,
             "units": "-"+lot,
             "type": "MARKET",
             "positionFill": "DEFAULT",
             "takeProfitOnFill": {
                "timeInForce": "GTC",
                "price":profit
            },
            "stopLossOnFill": {
                "timeInForce": "GTC",
                "price": loss
            }
        }
    }
    ticket = orders.OrderCreate(account_id, data=data)
    
    try:
        res = api.request(ticket)
        trade_id.append(res['orderCreateTransaction']['id']) #tradeID
        times = 0
        headers = ["profit", "now_price", "loss", "trade"]
        table = [(profit, now_price, loss, "sell")]
        result=tabulate(table, headers)
        print(result)
        flag["sell_signal"] = 1
    except V20Error:
        api = oandapyV20.API(access_token=api_key, environment="live")
        res = api.request(ticket)
        trade_id.append(res['orderCreateTransaction']['id']) #tradeID
        times = 0
        headers = ["profit", "now_price", "loss", "trade"]
        table = [(profit, now_price, loss, "sell")]
        result=tabulate(table, headers)
        print(result)
        flag["sell_signal"] = 1
    return trade_id, flag, times


#学習
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=600):
        #print(input_dim, output_dim)
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.d1 = nn.Dropout(p=0.1)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.d2 = nn.Dropout(p=0.1)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Dropout(p=0.1)
        self.a3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_dim, hidden_dim)
        self.d4 = nn.Dropout(p=0.1)
        self.a4 = nn.ReLU()
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.d5 = nn.Dropout(p=0.1)
        self.a5 = nn.ReLU()
        self.l6 = nn.Linear(hidden_dim, output_dim)
        
        self.layers = [self.l1, self.d1, self.a1, self.l2, self.d2, self.a2, self.l3, self.d3, self.a3, self.l4, self.d4, self.a4, self.l5, self.d5, self.a5, self.l6]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#メイン
trade_id = []
flag = {"buy_signal" : 0,
        "sell_signal" : 0
	}
model = Model(int(lerndata), 5).to(device)
model.load_state_dict(torch.load(loa_path))
model.eval()
if 'S' in asi :
    mini = int(re.sub(r"\D", "", asi))/5
if 'M' in asi:
    mini = (int(re.sub(r"\D", "", asi))*60)/5
if 'H' in asi:
    mini = (int(re.sub(r"\D", "", asi))*360)/5
if 'D' in asi:
    mini = (int(re.sub(r"\D", "", asi))*8640)/5
times = (bar * mini)
#print(times)
old_price = 0
while True:
    df_all = get_mdata(ex_pair, api, asi, bar)
    #print(df_all)
    now_price = df_all.iloc[len(df_all)-1, 4]
    #print(now_price)
    if old_price == 0:
        old_price = now_price
    if times == (bar * mini):
        df_all = get_mdata(ex_pair, api, asi, bar)
        #print(df_all)
        old_price = df_all.iloc[len(df_all)-1, 4]
        df_clo = df_all['close']
        MACD, signal = macd_data(df_clo)
        MACD_signal = macd_signal(MACD, signal)
        df_mac = pd.Series(MACD_signal)
        df = make_df(df_mac, df_all, bar, lerndata)
        x = df.iloc[0, :]
        x = torch.tensor(x.astype(np.float32)) #astype() : ndarray要素のデータ型を変更しtensor化
        x = x.to(device)
        model.eval() #ネットワークを推論モードに
        pre = model(x)
        #print(pre)
        pre = torch.argmax(pre)
        #print(pre)
        
        if pre == 0:
            trade_id, flag, times = buy_signal(now_price, account_id, api, lot, ex_pair, times, lim_up, lim_down, trade_id, flag)
        elif pre == 3:
            trade_id, flag, times = sell_signal(now_price, account_id, api, lot, ex_pair, times, lim_up, lim_down, trade_id, flag)
        
        
    else:
        times += 1
    time.sleep(5)


