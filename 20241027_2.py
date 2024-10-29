import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc  # 한글 폰트 설정
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
# 2023년 데이터 추가
# ======================================
df_2023 = pd.DataFrame({
    'Qin': [1206963],              # 2023년 서울 전입 인구수
    'Qout': [1238213],             # 2023년 서울 전출 인구수
    '출생아수(명)': [39456],       # 2023년 출생아수
    '사망자수(명)': [51446],       # 2023년 사망자수
    '남자인구수 (명)': [4540031],  # 2023년 남자인구수
    '여자인구수 (명)': [4846003]   # 2023년 여자인구수
})

df = pd.concat([df, df_2023], ignore_index=True)

# ======================================
# 초기값 설정
# ======================================

# 초기 인구수 설정
C0 = 10246565

# 데이터 업데이트
Qin = df['Qin']
Qout = df['Qout']
P = df['출생아수(명)']
D = df['사망자수(명)']

time = len(df)
dt = 1

# ======================================
# 1D 박스 모델
# ======================================
dC = np.zeros(time)
dC[0] = C0

for t in range(1, time):
    dC[t] = dC[t-1] + (Qin[t-1] - Qout[t-1] + P[t-1] - D[t-1]) * dt

# ======================================
# 실제 인구수 계산
# ======================================
new_dC = df['남자인구수 (명)'] + df['여자인구수 (명)'] - df['출생아수(명)'] + df['사망자수(명)'] - df['Qin'] + df['Qout']

# ======================================
# 오차율 계산
# ======================================
실제값 = new_dC.iloc[-1]
예측값 = dC[-1]
오차율 = (예측값 - 실제값) / 실제값 * 100
print(f"2023년 오차율: {오차율:.2f}%")

# ======================================
# 2024년 데이터 예측 및 추가
# ======================================
# 최근 3년 평균으로 2024년 데이터 추정
avg_Qin = Qin[-3:].mean()
avg_Qout = Qout[-3:].mean()
avg_P = P[-3:].mean()
avg_D = D[-3:].mean()

# 2024년 데이터 추가
df_2024 = pd.DataFrame({
    'Qin': [avg_Qin],
    'Qout': [avg_Qout],
    '출생아수(명)': [avg_P],
    '사망자수(명)': [avg_D]
})

df = pd.concat([df, df_2024], ignore_index=True)

# 데이터 업데이트
Qin = df['Qin']
Qout = df['Qout']
P = df['출생아수(명)']
D = df['사망자수(명)']

time = len(df)

# ======================================
# 모델 재실행
# ======================================
dC = np.zeros(time)
dC[0] = C0

for t in range(1, time):
    dC[t] = dC[t-1] + (Qin[t-1] - Qout[t-1] + P[t-1] - D[t-1]) * dt

# 2024년 인구수 예측 결과 출력
예측_2024_인구수 = dC[-1]
print(f"2024년 예측 인구수: {예측_2024_인구수:.0f}명")

# ======================================
# 시각화
# ======================================
plt.figure(figsize=(10,6))
years = range(2012, 2012 + time)
plt.plot(years, dC, color='k', marker='o', linestyle='-', label='모델 인구수')

# 실제 인구수는 2023년까지 있으므로, 그에 맞게 슬라이싱
plt.plot(years[:len(new_dC)], new_dC, color='r', marker='x', linestyle='--', label='실제 인구수')

plt.title("1D BOX MODEL 및 실제 지표 비교")
plt.xlabel("시간 (년)")
plt.ylabel("인구수 (명)")

# y축 천 단위 콤마 표시
plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.grid(True)
plt.xticks(years)
plt.legend()
plt.tight_layout()
plt.show()