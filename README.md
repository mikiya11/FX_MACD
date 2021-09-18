# FX_MACD
Automatic trading of FX using machine learning  
This trading uses the technical indicator MACD

python 3.8.5  
pytorch 1.7.1  
oandapyv20 0.6.3

**How to use**   
1.install pytorch,pandas,numpy,oandapyv20  
2.Input id and token for data/account.txt  
You can change exchange pair,lot,chart bar,learning date...etc by data/account.txt   *1  
3.Run make_model.py  
  ・print `best loss updated`  
4.Run buy_sell.py  
  ・print `b`  `s`  `number`  *2    
 

If you want to trade other than USD_JPY and GBP_JPY, change data/account.txt and run getdata.py  

*1  
|account_id|api_key|pair|lot|asi|date|epoch|bar|
|:---|:---|:---|:---|:---|:---|:---|:---|
|account id|API token|exchange pair|lot|chart bar time|start date|number times training|prediction bar|

  
*2  
b=buy  
s=sell  
number=total profit and loss 