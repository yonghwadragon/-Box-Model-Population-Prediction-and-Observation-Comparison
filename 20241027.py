import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc # 한글
import matplotlib.ticker as ticker
import pandas as pd

# 한글 폰트 설정 (예: 나눔고딕)
font_path = 'C:/Windows/Fonts/NanumGothic.ttf'  # 폰트 경로 설정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# ======================================
# data 읽기
# ======================================
file = "BoxBodelData.csv"
df = pd.read_csv(file, encoding='euc-kr')

# ======================================
# 초기값 설정
# ======================================

# 초기 인구수 설정  (DB의 서울의 출생아수, 사망자수, 전입, 전출이 반영하여 나온 2012년 인구수에서 출생아수를 빼고 사망자수를 더하고 전입을 빼고 전출을 더하여 나온 수. 즉, 2012년의 서울의 출생아수, 사망자수, 전입, 전출을 고려하지 않은 인구수.)
C0 = 10246565

# Qin: 서울 전입
Qin = df['Qin']

# Qout: 서울 전출
Qout = df['Qout']

# P: 서울 출생아수 (명)
P = df['출생아수(명)']

# D: 서울 사망자수 (명)
D = df['사망자수(명)']

time = len(df)  # 총 시간 (year) 지금은 11년
dt = 1          # 시간 간격 (year)

# ======================================
# 1D Box model
# ======================================
dC = np.zeros(time)
dC[0] = C0  # 초기 인구 설정

for t in range(1, time):
    dC[t] = dC[t-1] + (Qin[t-1] - Qout[t-1] + (P[t-1] - D[t-1])) * dt

# ======================================
# visulization
# ======================================

# 추가 그래프 계산(DB의 서울의 출생아수, 사망자수, 전입, 전출이 반영하여 나온 특정년도 인구수에서 출생아수를 빼고 사망자수를 더하고 전입을 빼고 전출을 더하여 나온 수. 즉, 각 년도별 서울의 출생아수, 사망자수, 전입, 전출을 고려하지 않은 인구수.)
new_dC = df['남자인구수 (명)'] + df['여자인구수 (명)'] - df['출생아수(명)']+df["사망자수(명)"]-df['Qin']+df['Qout']

plt.figure(figsize=(10,6))
years = range(2012, 2012 + time)  # 2012년부터 시작하는 x축 설정
plt.plot(years, dC, color='k',marker='o', linestyle='-', label='모델 인구수')

# 추가된 그래프 (빨간색 선)
plt.plot(years, new_dC, color='r', marker='x', linestyle='--', label='실제 인구수') 

plt.title("1D BOX MODEL 및 실제 지표 비교")
plt.xlabel("시간 (년)")
plt.ylabel("인구수 (명)")

# y축 천 단위 콤마 표시
plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.grid(True)

plt.xticks(years)  # 모든 연도에 대해 눈금 설정

plt.legend()
plt.tight_layout()
plt.show()