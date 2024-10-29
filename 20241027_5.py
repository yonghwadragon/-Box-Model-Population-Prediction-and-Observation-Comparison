import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc  # 한글 폰트 설정
import matplotlib.ticker as ticker
import pandas as pd
from sklearn.linear_model import LinearRegression

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
# 초기값 설정
# ======================================

# 초기 인구수 설정
C0 = 10246565

# 데이터 업데이트
Qin = df['Qin'].values
Qout = df['Qout'].values
P = df['출생아수(명)'].values
D = df['사망자수(명)'].values

time = len(df)
dt = 1

# ======================================
# 변수들의 추세 분석 및 미래 값 예측
# ======================================
# 연도 설정
years = np.arange(2012, 2012 + time).reshape(-1, 1)

# 각 변수에 대해 선형 회귀 모델을 피팅하고 미래 값 예측
def predict_future_values(y_values, years, future_years):
    model = LinearRegression()
    model.fit(years, y_values)
    future_years = np.array(future_years).reshape(-1, 1)
    return model.predict(future_years)

# 미래 예측을 위한 연도들
future_years = [2012 + time, 2012 + time + 1]  # 2024, 2025년

# Qin 예측
predicted_Qin = predict_future_values(Qin, years, future_years)

# Qout 예측
predicted_Qout = predict_future_values(Qout, years, future_years)

# P 예측 (출생아수)
predicted_P = predict_future_values(P, years, future_years)

# D 예측 (사망자수)
predicted_D = predict_future_values(D, years, future_years)

# ======================================
# 예측된 값들을 데이터프레임에 추가
# ======================================
df_future = pd.DataFrame({
    'Qin': predicted_Qin,
    'Qout': predicted_Qout,
    '출생아수(명)': predicted_P,
    '사망자수(명)': predicted_D
})

df = pd.concat([df, df_future], ignore_index=True)

# 데이터 업데이트
Qin = df['Qin'].values
Qout = df['Qout'].values
P = df['출생아수(명)'].values
D = df['사망자수(명)'].values

time = len(df)

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
# new_dC는 실제 데이터만 있으므로, 미래 예측 부분은 제외
actual_length = len(df) - len(future_years)
new_dC = df['남자인구수 (명)'][:actual_length] + df['여자인구수 (명)'][:actual_length] - df['출생아수(명)'][:actual_length] + df['사망자수(명)'][:actual_length] - df['Qin'][:actual_length] + df['Qout'][:actual_length]

# ======================================
# 오차 및 평가 지표 계산
# ======================================
# 실제 값과 예측 값의 차이 계산
차이 = dC[:actual_length] - new_dC.values  # 모델 예측값 - 실제값

# 각 년도의 오차율 계산
오차율_연도별 = (차이 / new_dC.values) * 100

# 연도 리스트 생성
연도들 = np.arange(2012, 2012 + actual_length)

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
예측값_2023 = dC[actual_length - 1]
오차율_2023 = (예측값_2023 - 실제값_2023) / 실제값_2023 * 100
오차_2023 = 예측값_2023 - 실제값_2023
print(f"\n2023년 오차: {오차_2023:.0f}명")
print(f"2023년 오차율: {오차율_2023:.2f}%")

# ======================================
# 2024년, 2025년 인구수 예측 결과 출력
# ======================================
예측_2024_인구수 = dC[-2]
예측_2025_인구수 = dC[-1]
print(f"\n2024년 예측 인구수: {예측_2024_인구수:.0f}명")
print(f"2025년 예측 인구수: {예측_2025_인구수:.0f}명")

# ======================================
# 시각화
# ======================================
plt.figure(figsize=(10,6))
years_total = np.arange(2012, 2012 + time)
plt.plot(years_total, dC, color='k', marker='o', linestyle='-', label='모델 인구수')

# 실제 인구수는 2023년까지
years_actual = np.arange(2012, 2012 + actual_length)
plt.plot(years_actual, new_dC, color='r', marker='x', linestyle='--', label='실제 인구수')

plt.title("1D BOX MODEL 및 실제 지표 비교")
plt.xlabel("시간 (년)")
plt.ylabel("인구수 (명)")

# y축 천 단위 콤마 표시
plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.grid(True)
plt.xticks(years_total)
plt.legend()
plt.tight_layout()
plt.show()