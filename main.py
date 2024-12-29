import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('data/weather_forecast.csv')

def load_and_train_model(k=3, train=80):
    X = data[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
    y = data['Rain']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-(train/100), random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model, scaler, X_train, X_test, y_train, y_test

if "model" not in st.session_state:
    model, scaler, X_train, X_test, y_train, y_test = load_and_train_model(k=3)
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

st.set_page_config(
    page_title="Prediksi Cuaca",
    page_icon="☁️",
)

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Visualisasi Data", "Setting Model", "Prediksi Cuaca"])

if page == "Beranda":
    st.title("Prediksi dan Analisis Cuaca ☁️🌧️")
    if "model" not in st.session_state:
        st.write("Melatih model untuk pertama kali...")
    else:
        st.success("Model siap dipakai.")
    st.markdown("""
    Selamat datang di aplikasi prediksi cuaca menggunakan model **K-Nearest Neighbors (KNN)**!
    Aplikasi ini memungkinkan Anda memprediksi apakah akan terjadi hujan atau tidak berdasarkan beberapa parameter cuaca.
    """)

    st.subheader("Deskripsi Dataset")
    st.write(f"Jumlah data: {len(data)}")
    st.write(f"Fitur yang digunakan: Temperature, Humidity, Wind Speed, Cloud Cover, Pressure")
    st.write(data.describe())  



elif page == "Visualisasi Data":
    st.title("Visualisasi Data 📈📊")
    tab1, tab2 = st.tabs(["Distribusi Suhu", "Distribusi dan hubungan Cuaca"])
    with tab1:
        st.subheader("Distribusi Suhu")
        fig, ax = plt.subplots()
        sns.histplot(data['Temperature'], kde=True, ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Distribusi rain dan no rain")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Rain', ax=ax)
        st.pyplot(fig)
        st.write("")
        st.subheader("Hubungan Suhu dan Kelembapan")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Temperature', y='Humidity', hue='Rain', ax=ax)
        st.pyplot(fig)
        st.write("")
        st.subheader("Hubungan Suhu dan Kecepatan Angin")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Temperature', y='Wind_Speed', hue='Rain', ax=ax)
        st.pyplot(fig)



elif page == "Setting Model":
    st.title("Setting Training model ⚙️")
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred, output_dict=True)

    jumlahK = st.number_input("Masukan Jumlah k untuk KNN", 1, 10, 3)
    persenTrain = st.slider("Masukan % Train", 10, 100, 80)
    retrain = st.button("Ganti Sekarang")

    accuracy_placeholder = st.empty()
    classification_placeholder = st.empty()
    accuracy_placeholder.write(f"**Accuracy:** {accuracy * 100:.2f}%")
    classification_placeholder.write("**Classification Report:**")
    classification_df = pd.DataFrame(classification).transpose()
    classification_placeholder.dataframe(classification_df)
    if retrain:
        model, scaler, X_train, X_test, y_train, y_test = load_and_train_model(jumlahK, persenTrain)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_pred = st.session_state.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred, output_dict=True)

        accuracy_placeholder.write(f"**Accuracy:** {accuracy * 100:.2f}%")
        classification_placeholder.write("**Classification Report:**")
        classification_df = pd.DataFrame(classification).transpose()
        classification_placeholder.dataframe(classification_df)



elif page == "Prediksi Cuaca":
    st.title("Hitung Prediksi Cuaca ✏️🌧️")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Masukan suhu C°", 10, 35, 23)
        humidity = st.number_input("Masukan Kelembapan", 30, 100, 89)
        wind_Speed = st.number_input("Masukan kecepatan Angin", 0, 20, 7)
        cloud_Cover = st.number_input("Masukan penutupan Awan", 1, 100, 50)
        pressure = st.slider("Masukan Tekanan udara", 980, 1050, 1032)
    with col2:
        st.write("Faktor yang mungkin mempengaruhi:")
        st.write("Jika suhu > 30°C, maka lebih sedikit hujan.")
        st.write("Jika kelembapan ≤ 70%, maka lebih sedikit hujan.")
        st.write("Jika kecepatan angin ≤ 5 m/s, maka lebih sedikit hujan.")
        st.write("Jika penutupan awan ≤ 60%, maka lebih sedikit hujan.")
        st.write("Jika tekanan ≥ 1000 hPa, maka lebih sedikit hujan.")
    hitung = st.button("Prediksi Sekarang")
    if hitung:
        data_baru = pd.DataFrame([[temperature, humidity, wind_Speed, cloud_Cover, pressure]], columns=['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure'])
        scaled_data_baru = st.session_state.scaler.transform(data_baru)
        prediksi = st.session_state.model.predict(scaled_data_baru)
        st.success(f"Cuaca diprediksi akan : {'Hujan (Rain)' if prediksi[0] == 'rain' else 'Tidak Hujan (No Rain)'}")

