from statistics import mean,stdev
import configparser
import pandas as pd
import pytz
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
from datetime import datetime, timedelta

config = configparser.ConfigParser()
config.read('./data/account.txt')
account_id = config['oanda']['account_id']  # ID
api_key = config['oanda']['api_key']        #トークンパス
ex_pair = config['oanda']['pair']           #対象通貨
lot = config['oanda']['lot']                #lot数 
asi = config['oanda']['asi']                #取得した時間足
get_date = config['oanda']['date']          #どの期間まで取得したいか指定

api = oandapyV20.API(access_token=api_key, environment="live")

params1 = {"instruments": ex_pair}
psnow = pricing.PricingInfo(accountID=account_id, params=params1)
now = api.request(psnow) #現在の価格を取得

end = now['time']

i=0
while(end > get_date):
    params = {"count":5000,"granularity":asi,"to":end}
    r = instruments.InstrumentsCandles(instrument=ex_pair, params=params,)
    apires = api.request(r)
    res = r.response['candles']
    end = res[0]['time']
    n = 0
    if i == 0 : res1 = res
    else :
        for j in range(n,len(res1)):res.append(res1[j])
        if end < get_date:
            for j in range(5000):
                if res[j]['time'] > get_date:
                    end = res[j-1]['time']
                    n = j
                    break
    res1 = res[n:]
    i+=1

data = []
#少し形を成形
for raw in res1:
    data.append([raw['time'], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])

#DataFrameに変換して、CSVファイルに保存
df = pd.DataFrame(data)
df.columns = ['date', 'open', 'high', 'low', 'close']

#時間を全て日本時間に変更する。
for i in df['date']:
    i = pd.Timestamp(i).tz_convert('Asia/Tokyo')

df = df.set_index('date')
df.index = df.index.astype('datetime64[ns]')
df.to_csv('./data/'+ex_pair+'_'+asi+'.csv', encoding='UTF8')
