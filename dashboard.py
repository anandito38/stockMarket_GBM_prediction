import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Mengunduh data saham ARCI
data_saham = yf.download("ARCI.JK", start="2021-01-01", end="2022-12-31")
data_saham.head()

# Mengecek missing values
missing_values = data_saham.isnull().sum()
print("Missing values:\n", missing_values)

# Menginterpolasi missing values
data_saham_interpolated = data_saham.interpolate(method='linear')

# Mengecek kembali missing values setelah interpolasi
missing_values_after = data_saham_interpolated.isnull().sum()
print("Missing values after interpolation:\n", missing_values_after)

# Menambahkan kolom log return
data_saham_interpolated['LogReturn'] = np.log(data_saham_interpolated['Adj Close'] / data_saham_interpolated['Adj Close'].shift(1))
data_saham_interpolated.dropna(inplace=True)

print(data_saham_interpolated.head())

# Moving Average
data_saham_interpolated['MA50'] = data_saham_interpolated['Adj Close'].rolling(window=50).mean()
data_saham_interpolated['MA200'] = data_saham_interpolated['Adj Close'].rolling(window=200).mean()

# Relative Strength Index (RSI)
def compute_rsi(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data_saham_interpolated['RSI'] = compute_rsi(data_saham_interpolated['Adj Close'], 14)
print(data_saham_interpolated.head())

# Tangani NaN dalam indikator teknikal
data_saham_interpolated['MA50'] = data_saham_interpolated['MA50'].fillna(data_saham_interpolated['MA50'].mean())
data_saham_interpolated['MA200'] = data_saham_interpolated['MA200'].fillna(data_saham_interpolated['MA200'].mean())
data_saham_interpolated['RSI'] = data_saham_interpolated['RSI'].fillna(data_saham_interpolated['RSI'].mean())
print(data_saham_interpolated.head())
# Memastikan tidak ada NaN
print(data_saham_interpolated.isna().sum())
data = data_saham_interpolated.copy()

# Pembagian fitur dan target sebelum splitting
features = data[['MA50', 'MA200', 'RSI']]
target = data['LogReturn']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print(f"Jumlah data pelatihan: {X_train.shape[0]}")
print(f"Jumlah data pengujian: {X_test.shape[0]}")

# Melatih model Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Melakukan prediksi pada set pengujian
y_pred = model.predict(X_test)

# Menghitung metrik evaluasi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R2: {r2}')

new_model = RandomForestRegressor(
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=100
)
new_model.fit(X_train, y_train)

# Melakukan prediksi pada set pengujian
y_pred = new_model.predict(X_test)

# Menghitung metrik evaluasi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R2: {r2}')

def gbm_simulate(S0, mu, sigma, n_days, n_simulations):
    dt = 1 / 252  # Asumsi 252 hari perdagangan dalam setahun
    simulations = []

    for _ in range(n_simulations):
        prices = [S0]
        for _ in range(n_days):
            epsilon = np.random.normal()
            S_t = prices[-1] * np.exp((mu * dt) + (sigma * epsilon * np.sqrt(dt)))
            prices.append(S_t)
        simulations.append(prices)

    return np.array(simulations)

# Menggunakan model machine learning untuk memprediksi drift
predicted_drift = new_model.predict(features)
# Menghitung volatilitas
sigma = data['LogReturn'].std()
# Parameter untuk simulasi
S0 = data['Adj Close'].iloc[-1]  # Harga penutupan terakhir
n_days = 252  # Jangka waktu simulasi (1 tahun)
n_simulations = 1000  # Jumlah simulasi

# Melakukan simulasi GBM
simulations = gbm_simulate(S0, predicted_drift[-1], sigma, n_days, n_simulations)
# Menampilkan beberapa hasil simulasi
print(simulations.shape)
# Menghitung statistik dari hasil simulasi
simulations_end = simulations[:, -1]

mean_price = np.mean(simulations_end)
median_price = np.median(simulations_end)
percentile_5 = np.percentile(simulations_end, 5)
percentile_95 = np.percentile(simulations_end, 95)
print(f"Mean Price: {mean_price}")
print(f"Median Price: {median_price}")
print(f"5th Percentile Price: {percentile_5}")
print(f"95th Percentile Price: {percentile_95}")

# Membuat plot distribusi harga simulasi
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(simulations_end, bins=50, alpha=0.7, color='blue')
ax1.axvline(mean_price, color='red', linestyle='dashed', linewidth=2, label='Mean')
ax1.axvline(median_price, color='green', linestyle='dashed', linewidth=2, label='Median')
ax1.axvline(percentile_5, color='orange', linestyle='dashed', linewidth=2, label='5th Percentile')
ax1.axvline(percentile_95, color='orange', linestyle='dashed', linewidth=2, label='95th Percentile')
ax1.set_title('Distribusi Harga Simulasi di Masa Depan')
ax1.set_xlabel('Harga Saham')
ax1.set_ylabel('Frekuensi')
ax1.legend()

# Memvisualisasikan beberapa jalur simulasi
fig2, ax2 = plt.subplots(figsize=(14, 7))
for i in range(3):  # Menampilkan 3 jalur simulasi
    ax2.plot(simulations[i], lw=1)
ax2.set_title('Simulasi Pergerakan Harga Saham')
ax2.set_xlabel('Hari')
ax2.set_ylabel('Harga Saham')


# Streamlit layout
with st.sidebar:
    selected = option_menu(
        "Menu", ["Home", "Distribusi Harga Simulasi", "Simulasi Pergerakan Harga Saham"]
    )

if selected == "Home":
    st.title('Penjelasan Model')
    st.write("""Model yang digunakan adalah Random Forest Regressor, yang bertujuan untuk memprediksi log return dari harga saham ARCI. Model ini dibuat untuk menganalisis
    pola historis harga saham dan menggunakan pola tersebut untuk memprediksi pergerakan harga di masa depan, membantu dalam pengambilan keputusan investasi. Parameter yang
    digunakan dalam model ini meliputi Moving Averages (MA50 dan MA200) dan Relative Strength Index (RSI), yang dihitung dari data harga saham historis.
    Data dibagi menjadi set pelatihan dan pengujian dengan rasio 80:20. Setelah model awal dilatih dan dievaluasi menggunakan Mean Squared Error (MSE) dan R-Squared (R2),
    model dioptimalkan lebih lanjut dengan pengaturan hyperparameter seperti max_depth, max_features, min_samples_leaf, min_samples_split, dan n_estimators untuk meningkatkan
    akurasi prediksi. Selain itu, dilakukan simulasi Geometric Brownian Motion (GBM) untuk memprediksi pergerakan harga saham di masa depan, menggunakan
    drift yang diprediksi oleh model dan volatilitas yang dihitung dari data historis. Hasil simulasi ini memberikan gambaran kemungkinan harga saham
    di masa depan, yang dapat digunakan sebagai alat bantu dalam strategi investasi.
        """)
    st.subheader("Nama Anggota Kelompok")
    nama_list = ["ATILLA FEJRIL (1301210495)", "NABILA AURELLIA (1301213017)",  "ANANDITO SATRIA ASYRAF (1301213026)",
                "NICHOLAS DAVI ERLANGGA PUTRA SUSILO (1301213136)", "MUHAMMAD RIVADITYA AZZAKA (1301210511)"]
    for nama in nama_list:
        st.markdown(nama)
elif selected == "Distribusi Harga Simulasi":
    st.title('Distribusi Harga Simulasi di Masa Depan')
    st.pyplot(fig1)
elif selected == "Simulasi Pergerakan Harga Saham":
    st.title('Simulasi Pergerakan Harga Saham')
    st.pyplot(fig2)