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
    '출생아수(명)': [39456],        # 2023년 출생아수
    '사망자수(명)': [51446],        # 2023년 사망자수
    '남자인구수 (명)': [4540031],   # 2023년 남자인구수
    '여자인구수 (명)': [4846003]    # 2023년 여자인구수
})

df = pd.concat([df, df_2023], ignore_index=True)

# ======================================
# 필요한 기간의 데이터 선택 (2018~2023년)
# ======================================
# 연도 열 추가
df['연도'] = np.arange(2012, 2012 + len(df))

# 2018년부터 2023년까지의 데이터 선택
df_selected = df[(df['연도'] >= 2018) & (df['연도'] <= 2023)].reset_index(drop=True)

# ======================================
# 초기값 설정
# ======================================
# 초기 인구수 설정 (2018년 데이터 사용)
C0 = df_selected['남자인구수 (명)'][0] + df_selected['여자인구수 (명)'][0] - df_selected['출생아수(명)'][0] + df_selected['사망자수(명)'][0] - df_selected['Qin'][0] + df_selected['Qout'][0]

# 데이터 업데이트
Qin = df_selected['Qin'].values
Qout = df_selected['Qout'].values
P = df_selected['출생아수(명)'].values
D = df_selected['사망자수(명)'].values

time = len(df_selected)
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
new_dC = df_selected['남자인구수 (명)'] + df_selected['여자인구수 (명)'] - df_selected['출생아수(명)'] + df_selected['사망자수(명)'] - df_selected['Qin'] + df_selected['Qout']

# ======================================
# 오차 및 평가 지표 계산
# ======================================
# 실제 값과 예측 값의 차이 계산
차이 = dC - new_dC.values  # 모델 예측값 - 실제값

# 각 년도의 오차율 계산
오차율_연도별 = (차이 / new_dC.values) * 100

# 연도 리스트 생성
연도들 = df_selected['연도'].values

# 각 년도의 오차율 출력
print("연도별 오차율:")
for year, error_rate in zip(연도들, 오차율_연도별):
    print(f"{year}년 오차율: {error_rate:.2f}%")

# 평균 절대 오차 (MAE)
MAE = np.mean(np.abs(차이))
print(f"\n평균 절대 오차 (MAE): {MAE:.2f}명")

# 평균 제곱근 오차 (RMSE)
RMSE = np.sqrt(np.mean(차이 ** 2))
print(f"평균 제곱근 오차 (RMSE): {RMSE:.2f}명")

# 평균 절대 백분율 오차 (MAPE)
MAPE = np.mean(np.abs(오차율_연도별))
print(f"평균 절대 백분율 오차 (MAPE): {MAPE:.2f}%")

# 마지막 해의 오차율 (2023년)
실제값_2023 = new_dC.iloc[-1]
예측값_2023 = dC[-1]
오차율_2023 = (예측값_2023 - 실제값_2023) / 실제값_2023 * 100
오차_2023 = 예측값_2023 - 실제값_2023
print(f"\n2023년 오차: {오차_2023:.0f}명")
print(f"2023년 오차율: {오차율_2023:.2f}%")

# ======================================
# 시각화
# ======================================
plt.figure(figsize=(10,6))
years = df_selected['연도'].values
plt.plot(years, dC, color='k', marker='o', linestyle='-', label='모델 인구수')

# 실제 인구수
plt.plot(years, new_dC, color='r', marker='x', linestyle='--', label='실제 인구수')

plt.title("1D BOX MODEL 및 실제 지표 비교 (2018~2023년)")
plt.xlabel("시간 (년)")
plt.ylabel("인구수 (명)")

# y축 천 단위 콤마 표시
plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.grid(True)
plt.xticks(years)
plt.legend()
plt.tight_layout()
plt.show()