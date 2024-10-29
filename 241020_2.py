import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc  # 한글
import matplotlib.ticker as ticker
import pandas as pd

# 한글 폰트 설정 (예: 나눔고딕)
font_path = 'C:/Windows/Fonts/NanumGothic.ttf'  # 폰트 경로 설정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# ======================================
# 데이터 읽기
# ======================================
file = "BoxBodelData.csv"
df = pd.read_csv(file, encoding='euc-kr')

# ======================================
# 초기값 설정
# ======================================

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
dt = 1          # 시간 간격 (year)

# ======================================
# 1D Box model
# ======================================
dC = np.zeros(time)
dC[0] = C0  # 초기 인구 설정

for t in range(1, time):
    dC[t] = dC[t-1] + (Qin[t-1] - Qout[t-1] + (P[t-1] - D[t-1])) * dt

# ======================================
# 시각화
# ======================================

fig, axs = plt.subplots(3, 2, figsize=(18, 15))
years = range(2012, 2012 + time)  # 2012년부터 시작하는 x축 설정

# 첫 번째 서브플롯: 모델 인구수
axs[0, 0].plot(years, dC, color='k', marker='o', linestyle='-')
axs[0, 0].set_title("모델 인구수")
axs[0, 0].set_xlabel("시간 (년)")
axs[0, 0].set_ylabel("인구수 (명)")
axs[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
axs[0, 0].grid(True)

# 두 번째 서브플롯: 전입(Qin)과 전출(Qout)
axs[0, 1].plot(years, Qin, color='blue', marker='^', linestyle='--', label='전입(Qin)')
axs[0, 1].plot(years, Qout, color='red', marker='v', linestyle='--', label='전출(Qout)')
axs[0, 1].set_title("전입 및 전출")
axs[0, 1].set_xlabel("시간 (년)")
axs[0, 1].set_ylabel("인구수 (명)")
axs[0, 1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
axs[0, 1].grid(True)
axs[0, 1].legend()

# 세 번째 서브플롯: 출생아수(P)과 사망자수(D)
axs[1, 0].plot(years, P, color='green', marker='s', linestyle='-.', label='출생아수(P)')
axs[1, 0].plot(years, D, color='purple', marker='D', linestyle='-.', label='사망자수(D)')
axs[1, 0].set_title("출생아수 및 사망자수")
axs[1, 0].set_xlabel("시간 (년)")
axs[1, 0].set_ylabel("인구수 (명)")
axs[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
axs[1, 0].grid(True)
axs[1, 0].legend()

# 네 번째 서브플롯: 일반혼인율 (남편)
axs[1, 1].plot(years, marriage_rate_husband, color='orange', marker='o', linestyle=':', label='일반혼인율(남편)')
axs[1, 1].set_title("일반혼인율 (남편)")
axs[1, 1].set_xlabel("시간 (년)")
axs[1, 1].set_ylabel("혼인율 (%)")
axs[1, 1].grid(True)
axs[1, 1].legend()

# 다섯 번째 서브플롯: 일반혼인율 (아내)
axs[2, 0].plot(years, marriage_rate_wife, color='brown', marker='x', linestyle=':', label='일반혼인율(아내)')
axs[2, 0].set_title("일반혼인율 (아내)")
axs[2, 0].set_xlabel("시간 (년)")
axs[2, 0].set_ylabel("혼인율 (%)")
axs[2, 0].grid(True)
axs[2, 0].legend()

# 여섯 번째 서브플롯: 빈 공간 (추가 지표를 위한 공간)
axs[2, 1].axis('off')  # 빈 서브플롯 비활성화

plt.tight_layout()
plt.show()
