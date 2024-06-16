# BootCamo AI
[![PyPI](https://img.shields.io/pypi/v/danger-python)](https://pypi.org/project/danger-python/)
![Python versions](https://img.shields.io/pypi/pyversions/danger-python)
[![Build Status](https://travis-ci.org/danger/python.svg?branch=master)](https://travis-ci.org/danger/python)
# python
Write your Dangerfiles in Python.
### Requirements
:warning: `danger-python` is currently work in progress. Breaking changes may occur.

<div align="center">
  <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/Capture.PNG" width="1000"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">Hamed Aghapanah </font></b>
    <sup>
      <a href="https://HamedAghapanah.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">HamedAghapanah platform</font></b>
    <sup>
      <a href="https://HamedAghapanah.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>  </div>

  </div>

 # تمارین هفته‌ی اول
 
 لینک ارسال پاسخ :
 
 https://forms.gle/drkwhMqnV7dvjBrt8
      
   <b><font size="5">
1_ از کاربر یک عدد (صحیح)  بگیرد و به تعداد آن،  عدد دریافت نماید. سپس بیشترین، کمترین و میانگین آنها را برگرداند.

2_ از کاربر 10 عدد  (صحیح)  دریافت کرده و سپس  فاکتوریل هر کدام را به همراه شیوه محاسبه آن پرینت کند.
3_ یک لیست از 20 عدد (صحیح) را از کاربر دریافت کرده و  سپس تعداد اعداد زوج در آن لیست را بشمارید و چاپ کنید.
4- ایجاد یک برنامه ساده که قادر به تبدیل دما بین واحدهای سانتی‌گراد، فارنهایت و کلوین باشد؟
5- در ورودی دما به همراه تایپ ورودی و خروجی بگیرد و خروجی بدهد.
مثلا برای تبدیل دمای 100 درجه سانتیگراد به فارنهایت ورودی زیر را بگیرد و خروجی بدهد:
100,    C,F


  # تمارین هفته‌ی دوم و سوم
  
   لینک ارسال پاسخ :
   
https://docs.google.com/forms/d/e/1FAIpQLScT4zcqblgRqSl87XusKsNwJ_BXMyF7UtypdPMLvWG0ERLeOw/viewform?usp=sf_link


1) با استفاده از مجموعه داده بوستون، یک مدل رگرسیون خطی ایجاد کنید که قیمت مسکن را پیش‌بینی کند. سپس معیارهای دقت مدل را ارزیابی کنید.(حداقل ۲ روش)
2) از مجموعه داده کالیفرنیا، یک مدل رگرسیون چندجمله‌ای ایجاد کنید که میانگین تعداد اتاق‌ها را بر اساس ویژگی‌های دیگر مسکن پیش‌بینی کند. سپس دقت مدل را ارزیابی کنید.(حداقل ۲ 3روش)
3) با استفاده از مجموعه داده Iris، یک مدل طبقه‌بندی ساده ایجاد کنید که گونه گل را بر اساس ویژگی‌هایش تشخیص دهد. سپس دقت مدل را ارزیابی کنید.(حداقل ۲ روش)
4) با استفاده از مجموعه داده Breast Cancer، یک مدل طبقه‌بندی SVM ایجاد کنید که توانایی تشخیص بین سرطان خوش‌خیم و سرطان بدخیم را بر اساس ویژگی‌های موجود داشته باشد. سپس دقت مدل را ارزیابی کنید.(حداقل ۲ روش)

 # تمارین هفته‌ی چهارم و پنجم
 
 لینک ارسال پاسخ :
 


## Question 1:
Read a dataset from the internet and implement an LSTM network that can perform predictions.

Sugestation Solution:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, usecols=[1])
data = data.values.astype('float32')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
```
### نتایج پیشنهادی با LSTM

 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/LSTM_train_vs_validation_loss.png" width="300"/>
 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/LSTM_true_vs_predictions.png" width="300"/>
 
##  Question 2:
Use an RNN network to perform the same task as in Question 1.

Sugestation Solution:
```python
 from tensorflow.keras.layers import SimpleRNN
```



### نتایج پیشنهادی با RNN

 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/RNN_train_vs_validation_loss_rnn.png" width="300"/> 
 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/RNN_true_vs_predictions_rnn.png" width="300"/>
 

##  Question 3:
Implement a network that combines both RNN and LSTM to achieve the best prediction results.

Sugestation Solution:
```python
# Create and fit the combined RNN + LSTM network
model_combined = Sequential()
model_combined.add(SimpleRNN(50, return_sequences=True, input_shape=(look_back, 1)))
model_combined.add(LSTM(50))
model_combined.add(Dense(1))
model_combined.compile(loss='mean_squared_error', optimizer='adam')
model_combined.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=2)

# Make predictions
train_predict_combined = model_combined.predict(X_train)
test_predict_combined = model_combined.predict(X_test)

# Invert predictions
train_predict_combined = scaler.inverse_transform(train_predict_combined)
test_predict_combined = scaler.inverse_transform(test_predict_combined)

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(data))
plt.plot(np.arange(look_back, look_back + len(train_predict_combined)), train_predict_combined)
plt.plot(np.arange(len(train_predict_combined) + (look_back * 2) + 1, len(data) - 1), test_predict_combined)
plt.show()

```

### نتایج پیشنهادی با RNN + LSTM



 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/results/all_train_vs_validation_loss_advanced_with_bidirectional_attention.png" width="300"/>
 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/results/all_true_vs_predictions_advanced_with_bidirectional_attention.png" width="300"/>
 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/best_advanced_model_with_bidirectional_attention.png" width="300"/>

 
 

<div style="text-align: Right; color: red; font-size: 30px;">
 
 
 # لیست حاضران
<div style="text-align: Right ">

1) حامد آقاپناه   10 خرداد
  ( پس از بارگذاری عکس و برنامه به اسم خودتان، آنرا در زیر اسمتان نمایش دهید )
  
2)  محمد کریمی   11 خرداد
3) محد امین رشیدی  11 خرداد   Jonas Raschidie >= Mohammad Amin Rashidi
4) سپیده شفیعی 11 خرداد
5) محمد محمدی 11 خرداد


            نمایش جواب پروژه تحلیل داده های EXAMPLE1 => تحلیل با شبکه LSTM
 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/mohammad_mohammadi_example1_code_and_answer/example1_1_CO.png" width="500"/> 
      
 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/mohammad_mohammadi_example1_code_and_answer/example1_2_CO.png" width="500"/>

<img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/mohammad_mohammadi_example1_code_and_answer/example1_3_CO.png" width="500"/>

<img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/mohammad_mohammadi_example1_code_and_answer/example1_1_NOx.png" width="500"/>

<img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/mohammad_mohammadi_example1_code_and_answer/example1_2_NOx.png" width="500"/>

 <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/mohammad_mohammadi_example1_code_and_answer/example1_3_NOx.png" width="500"/>



7) جعفر آقاجاني  11 خرداد
8) حمید رضا قربانی 11 خرداد
   NOx
   <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/Ghorbani-NOx.png" width="1000"/>

   Co
   <img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/Ghorbani-Co.png" width="1000"/>

   
9) انسیه باقری ۱۱ خرداد
10) حمیدرضا حسن زاده 11 خرداد
11) بشری ربانی 11 خرداد
12) علیرضا مهدی 11 خرداد
<img src="https://github.com/Hamed-Aghapanah/BOOTCAMP_AI/blob/main/Alireza%20Mahdi.png" width="1000"/>
13) زهزه سورانی ۱۱ خرداد
14) فاطمه فولادی ۱۱ خرداد
15) محسن فیاضی 12 خرداد
16) محمد جواد کویری منش 16 خرداد
17) مهربد خشوعی ۱۸ خرداد
18) قاسمیان

  </div>
  </div>


  </div>
  </div>
  </div>
