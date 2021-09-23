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
lim_up = float(config['oanda']['limit_up'])        #利確位置
lim_down = float(config['oanda']['limit_down'])      #利確位置

#modelパス
loa_path = 'FXmodel'+'_'+ex_pair+'_'+asi+'_'+bar+'.pth'
bar = int(bar)
api = oandapyV20.API(access_token=api_key, environment="live")

def get_mdata(ex_pair, api, asi, bar):
    params1 = {"instruments": ex_pair}
    psnow = pricing.PricingInfo(accountID=account_id, params=params1)
    try:
        now = api.request(psnow) #現在の価格を取得
    except V20Error:
        now = api.request(psnow) #現在の価格を取得
    end = now['time']
    params = {"count":35 + bar,"granularity":asi,"to":end}
    r = instruments.InstrumentsCandles(instrument=ex_pair, params=params,)
    try:
        apires = api.request(r)
    except V20Error:
        apires = api.request(r)
    res = r.response['candles']
    end = res[0]['time']
    n = 0
    res1 = res

    data = []
    price = []
    #形を成形
    for raw in res1:
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
def make_df(df, df_all, bar):
    bar = int(bar)
    #label作成
    df_al = df_all[['close']]
    df_al2 = df_al.shift(-bar)
    sh = df_al2.shape
    df_al2.columns = range(sh[1]) 
    df_al3 = (df_al.iloc[:, 0] - df_al2.iloc[:, 0])
    df_al3 = df_al3.shift(-35)
    df_al3 = df_al3.dropna(how='any')
    df_label = df_al3
    #上、下、そのままに三分割
    idx_up = df_al3.index[df_al3 > 0]
    idx_down = df_al3.index[df_al3 < 0]
    idx_no = df_al3.index[df_al3 == 0]
    df_al3 = pd.DataFrame(df_al3, columns = ['diff'])
    df_al3 = df_al3.assign(label=0)
    df_al3.iloc[idx_up, 1] = 1
    df_al3.iloc[idx_down, 1] = 2
    df_up = df_al3[df_al3.iloc[:, 1] == 1]
    df_down = df_al3[df_al3.iloc[:, 1] == 2]
    #さらに分割して５つに分ける
    df_up = df_up.sort_values('diff', ascending=False)
    df_down = df_down.sort_values('diff', ascending=False)
    len_up = math.floor(len(df_up)/2)
    len_down = math.floor(len(df_down)/2)
    df_1 = df_up[len_up:]
    df_2 = df_up[:len_up-1]
    df_3 = df_down[len_down+1:]
    df_4 = df_down[:len_down]
    idx_1 = list(df_1.index)
    idx_2 = list(df_2.index)
    idx_3 = list(df_3.index)
    idx_4 = list(df_4.index)
    df_label[idx_1] = 1
    df_label[idx_2] = 2
    df_label[idx_3] = 3
    df_label[idx_4] = 4
    df_label[idx_no] = 5

    #MACDデータ作成
    for i in range(bar-1):
        if i == 0:
            df_shift = df
        df_shift = df_shift.shift(-1)
        df = pd.concat([df, df_shift], axis=1)
        
    #結合
    sh = df.shape
    df.columns = range(sh[1])
    df = df.dropna(how='any') 
    df = pd.concat([df_label, df], axis=1)
    sh = df.shape
    df.columns = range(sh[1])
    return df
#dfから特徴量とラベルのみ抽出
def extract_x_y(df: pd.DataFrame):
    x = df.iloc[:, 1:] #特徴量
    y = df.iloc[:, 0] #ラベル
    return x, y

#買い
def buy_signal(now_price, account_id, api, lot, ex_pair, times, lim_up, lim_down, unit):
    lot = str(lot)
    profit = now_price + lim_up
    loss = now_price + (lim_down*2)
    profit = str(round(profit, 3))
    loss = str(round(loss, 3))
    #売りポジション決済
    if unit < 0:
        data = {"shortUnits":"ALL"}
        ticket = positions.PositionClose(accountID=account_id, instrument=ex_pair, data=data)
        try:
            api.request(ticket)
        except V20Error:
            print(sell, close, error)
            api.request(ticket)
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
    except V20Error:
        print(buy, error)
        res = api.request(ticket)
    now_price= res['orderFillTransaction']['price']  # 約定レート
    unit = int(res['orderFillTransaction']['units']) #lot数
    times = 0
    headers = ["profit", "now_price", "loss", "trade"]
    table = [(profit, now_price, loss, "buy")]
    result=tabulate(table, headers)
    print(result)
    return unit, times
#売り
def sell_signal(now_price, account_id, api, lot, ex_pair, times, lim_up, lim_down, unit):
    lot = str(lot)
    profit = now_price + lim_down
    loss = now_price +(lim_up*2)
    profit = str(round(profit, 3))
    loss = str(round(loss, 3))
    #買いポジション決済
    if unit > 0:
        data = {"longUnits":"ALL"}
        ticket = positions.PositionClose(accountID=account_id, instrument=ex_pair, data=data)
        try:
            api.request(ticket)
        except V20Error:
            print(buy, close, error)
            api.request(ticket)
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
    except V20Error:
        print(sell, error)
        res = api.request(ticket)
    now_price= res['orderFillTransaction']['price']  # 約定レート
    unit = int(res['orderFillTransaction']['units']) #lot数
    times = 0
    headers = ["profit", "now_price", "loss", "trade"]
    table = [(profit, now_price, loss, "sell")]
    result=tabulate(table, headers)
    print(result)
    return unit, times


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=500):
        #print(input_dim, output_dim)
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.d1 = nn.Dropout(p=0.2)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.d2 = nn.Dropout(p=0.2)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Dropout(p=0.2)
        self.a3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_dim, hidden_dim)
        self.d4 = nn.Dropout(p=0.2)
        self.a4 = nn.ReLU()
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.d5 = nn.Dropout(p=0.2)
        self.a5 = nn.ReLU()
        self.l6 = nn.Linear(hidden_dim, output_dim)
        """
        self.layers = [self.l1, self.d1, self.a1,
                        self.l2, self.d2, self.a2,
                        self.l3, self.d3, self.a3,
                        self.l4]
        self.layers = [self.l1, self.a1,
                        self.l2, self.a2,
                        self.l4]
        """
        self.layers = [self.l1, self.a1, self.l2, self.a2, self.l3, self.a3, self.l4, self.a4, self.l5, self.a5, self.l6]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_data(x, y): #ラベルと特徴量を結合してリストにする
    x = x.astype(np.float32) #astype() : ndarray要素のデータ型を変更
    y = y.astype(np.int64)
    data = []
    for xx, yy in zip(x, y):
        data.append((xx, yy))
    return data

#メイン
unit = 0
model = Model(int(bar), 5).to(device)
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
    now_price = df_all.iloc[54, 4]
    #print(now_price)
    if old_price == 0:
        old_price = now_price
    if times == (bar * mini):
        df_all = get_mdata(ex_pair, api, asi, bar)
        #print(df_all)
        old_price = df_all.iloc[54, 4]
        df_clo = df_all['close']
        MACD, signal = macd_data(df_clo)
        MACD_signal = macd_signal(MACD, signal)
        df_mac = pd.Series(MACD_signal)
        df = make_df(df_mac, df_all, bar)  
        x, y = extract_x_y(df)
        data = make_data(x.to_numpy(), y.to_numpy())
        #データローダの設定
        vali_dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for (x, t) in vali_dataloader:
            x, t = x.to(device), t.to(device)
            model.eval() #ネットワークを推論モードに
            pre = model(x) 
            #print(pre)
            pre = torch.argmax(pre)
            #print(pre)

        if pre == 0:
            unit, times = buy_signal(now_price, account_id, api, lot, ex_pair, times, lim_up, lim_down, unit)
        elif pre == 3:
            unit, times = sell_signal(now_price, account_id, api, lot, ex_pair, times, lim_up, lim_down, unit)
        
    else:
        times += 1
    time.sleep(5)


