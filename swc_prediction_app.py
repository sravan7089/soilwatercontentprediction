# ######################################################################################################################################

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Conv1D, Flatten
# from tensorflow.keras.optimizers import Adam, RMSprop
# from tensorflow.keras.callbacks import EarlyStopping
# import joblib
# import matplotlib.pyplot as plt

# # Streamlit UI setup
# st.title('Soil Water Content Prediction with Deep Learning Models')
# st.sidebar.header('Upload Your Dataset')

# # File uploader
# uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# if uploaded_file is not None:
#     # Load dataset
#     data = pd.read_csv(uploaded_file)
#     st.write("### Preview of Uploaded Data:", data.head())
    
#     # Handle missing values
#     data = data.fillna(data.mean())

#     # Define features and target
#     features = [col for col in data.columns if 'moisture' in col]
#     target = 'moisture0'

#     # Feature Scaling
#     scaler = MinMaxScaler()
#     data[features] = scaler.fit_transform(data[features])

#     # Train-test split
#     X = data[features]
#     y = data[target]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Random Forest Model
#     rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
#     rf_model.fit(X_train, y_train)
#     rf_predictions = rf_model.predict(X_test)
#     rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

#     # ANN Model
#     ann_model = Sequential([
#         Input(shape=(X_train.shape[1],)),
#         Dense(256, activation='relu'),
#         Dropout(0.4),
#         BatchNormalization(),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         BatchNormalization(),
#         Dense(64, activation='relu'),
#         Dense(1)
#     ])
#     ann_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
#     ann_history = ann_model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10)])
#     ann_predictions = ann_model.predict(X_test)
#     ann_rmse = np.sqrt(mean_squared_error(y_test, ann_predictions))

#     # LSTM Model
#     X_train_lstm = np.expand_dims(X_train, axis=1)
#     X_test_lstm = np.expand_dims(X_test, axis=1)
#     lstm_model = Sequential([
#         Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
#         LSTM(128, activation='tanh', return_sequences=True),
#         Dropout(0.4),
#         LSTM(64, activation='tanh'),
#         Dropout(0.3),
#         Dense(1)
#     ])
#     lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
#     lstm_history = lstm_model.fit(X_train_lstm, y_train, epochs=200, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10)])
#     lstm_predictions = lstm_model.predict(X_test_lstm)
#     lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))

#     # CNN Model
#     X_train_cnn = np.expand_dims(X_train, axis=2)
#     X_test_cnn = np.expand_dims(X_test, axis=2)
#     cnn_model = Sequential([
#         Input(shape=(X_train_cnn.shape[1], 1)),
#         Conv1D(filters=128, kernel_size=3, activation='relu'),
#         Dropout(0.4),
#         Conv1D(filters=64, kernel_size=3, activation='relu'),
#         Dropout(0.3),
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dense(1)
#     ])
#     cnn_model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
#     cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=200, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10)])
#     cnn_predictions = cnn_model.predict(X_test_cnn)
#     cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_predictions))

#     # Compare model performances
#     results = {'Random Forest': rf_rmse, 'ANN': ann_rmse, 'LSTM': lstm_rmse, 'CNN': cnn_rmse}
#     best_model = min(results, key=results.get)

#     st.write("### Model Performance Comparison:")
#     for model, rmse in results.items():
#         st.write(f"{model}: RMSE = {rmse:.4f}")

#     st.write(f"### Best Performing Model: {best_model} with RMSE = {results[best_model]:.4f}")

#     # Save models
#     joblib.dump(rf_model, 'random_forest_model.pkl')
#     ann_model.save('ann_model.h5')
#     lstm_model.save('lstm_model.h5')
#     cnn_model.save('cnn_model.h5')

#     # Plot loss curves
#     st.write("### Model Training Loss Curves")
#     fig, ax = plt.subplots()
#     ax.plot(ann_history.history['loss'], label='ANN Loss')
#     ax.plot(lstm_history.history['loss'], label='LSTM Loss')
#     ax.plot(cnn_history.history['loss'], label='CNN Loss')
#     ax.legend()
#     st.pyplot(fig)


# ###########################################################################################################################




import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

# Streamlit UI setup
st.title('🌱 Soil Water Content Prediction with Deep Learning Models')
st.sidebar.header('📂 Upload Your Dataset')

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### 📄 Preview of Uploaded Data:", data.head())

    # Handle missing values
    data = data.fillna(data.mean(numeric_only=True))

    # Define features and target
    features = [col for col in data.columns if 'moisture' in col]
    target = 'moisture0'

    # Feature Scaling
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    # Train-test split
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

    # ANN Model
    ann_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    ann_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    ann_history = ann_model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10)])
    ann_predictions = ann_model.predict(X_test)
    ann_rmse = np.sqrt(mean_squared_error(y_test, ann_predictions))

    # LSTM Model
    X_train_lstm = np.expand_dims(X_train, axis=1)
    X_test_lstm = np.expand_dims(X_test, axis=1)
    lstm_model = Sequential([
        Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        LSTM(128, activation='tanh', return_sequences=True),
        Dropout(0.4),
        LSTM(64, activation='tanh'),
        Dropout(0.3),
        Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    lstm_history = lstm_model.fit(X_train_lstm, y_train, epochs=200, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10)])
    lstm_predictions = lstm_model.predict(X_test_lstm)
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))

    # CNN Model
    X_train_cnn = np.expand_dims(X_train, axis=2)
    X_test_cnn = np.expand_dims(X_test, axis=2)
    cnn_model = Sequential([
        Input(shape=(X_train_cnn.shape[1], 1)),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        Dropout(0.4),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    cnn_model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=200, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10)])
    cnn_predictions = cnn_model.predict(X_test_cnn)
    cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_predictions))

    # Compare model performances
    results = {'Random Forest': rf_rmse, 'ANN': ann_rmse, 'LSTM': lstm_rmse, 'CNN': cnn_rmse}
    best_model = min(results, key=results.get)

    st.write("### 📊 Model Performance Comparison:")
    for model, rmse in results.items():
        st.write(f"{model}: RMSE = {rmse:.4f}")

    st.success(f"### ✅ Best Performing Model: **{best_model}** with RMSE = {results[best_model]:.4f}")

    # Save models
    joblib.dump(rf_model, 'random_forest_model.pkl')
    ann_model.save('ann_model.h5')
    lstm_model.save('lstm_model.h5')
    cnn_model.save('cnn_model.h5')

    # Plot loss curves
    st.write("### 📉 Model Training Loss Curves")
    fig, ax = plt.subplots()
    ax.plot(ann_history.history['loss'], label='ANN Loss')
    ax.plot(lstm_history.history['loss'], label='LSTM Loss')
    ax.plot(cnn_history.history['loss'], label='CNN Loss')
    ax.legend()
    st.pyplot(fig)

    # ==============================
    # 🤖 Simple Chatbot Assistant
    # ==============================
    st.write("### 💬 Ask our assistant about the models or data:")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about the app, models, or dataset..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # Basic rule-based response
        if "model" in prompt.lower():
            response = f"This app uses Random Forest, ANN, LSTM, and CNN models. The best in this run was **{best_model}**."
        elif "features" in prompt.lower() or "columns" in prompt.lower():
            response = f"The features used for training are: {', '.join(features)}"
        elif "target" in prompt.lower():
            response = f"The target variable for prediction is: **{target}**"
        elif "rmse" in prompt.lower():
            response = "\n".join([f"{k}: {v:.4f}" for k, v in results.items()])
        else:
            response = "I'm your assistant for this app. You can ask about models, features, predictions, or errors."

        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)








