from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
# https://github.com/FinanceData/FinanceDataReader
import matplotlib.pyplot as plt
import seaborn as sns

sdate = datetime(2023,1,1) 
edate = datetime(2024,5,21)

df = fdr.DataReader('KS11', '2022-01-01', '2022-12-31') # 2022-01-01 ~ 2022-12-31
print(f"use : {fdr.__version__}, period :{sdate} - {edate}")
print(f" response : \n {df.head(10)}")

sns.set_style('whitegrid')
df['Comp'].plot()
plt.show()