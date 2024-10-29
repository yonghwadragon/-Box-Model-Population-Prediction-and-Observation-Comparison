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

# 초기 인구수 설정: C0 = 2012년의 인구수 남+여자인구수-출생아수+사망자수-전입+전출이므로
# 초기 인구수 설정: C0 = 남성 인구 + 여성 인구 - 출생아수 + 사망자수 - 전입 + 전출
C0 = 5041336 + 5153982 - 93914 + 41514 - 1555281 + 1658928

# Qin: 서울 전입
Qin = df['Qin']

# Qout: 서울 전출
Qout = df['Qout']

# P: 서울 출생아수 (명)
P = df['출생아수(명)']

# D: 서울 사망자수 (명)
D = df['사망자수(명)']

# 일반혼인율 (남편)
marriage_rate_husband = df['일반혼인율(남편)']

# 일반혼인율 (아내)
marriage_rate_wife = df['일반혼인율(아내)']

time = len(df)  # 총 시간 (year) 지금은 11년
dt = 1          # 시간 간격 (year) (6개월) (3개월) (45일) (22일 12시간) 이거 1년 이하는 의미 없음

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

plt.figure(figsize=(14,8))
years = range(2012, 2012 + time)  # 2012년부터 시작하는 x축 설정

# 인구수 (모델 인구수)
plt.plot(years, dC, color='k',marker='o', linestyle='-', label='모델 인구수')

# 전입과 전출
plt.plot(years, Qin, color='blue', marker='^', linestyle='--', label='전입(Qin)')
plt.plot(years, Qout, color='red', marker='v', linestyle='--', label='전출(Qout)')

# 출생아수와 사망자수
plt.plot(years, P, color='green', marker='s', linestyle='-.', label='출생아수(P)')
plt.plot(years, D, color='purple', marker='D', linestyle='-.', label='사망자수(D)')

# 일반혼인율 (남편과 아내)
plt.plot(years, marriage_rate_husband, color='orange', marker='o', linestyle=':', label='일반혼인율(남편)')
plt.plot(years, marriage_rate_wife, color='brown', marker='x', linestyle=':', label='일반혼인율(아내)')


plt.title("1D BOX MODEL 및 보조 지표")
plt.xlabel("시간 (년)")
plt.ylabel("인구수 (명)")

# y축 천 단위 콤마 표시
plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.grid(True)

plt.xticks(years)  # 모든 연도에 대해 눈금 설정

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()