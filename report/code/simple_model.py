basic_model = Sequential()
basic_model.add(LSTM(500, input_shape=(X_train_vals.shape[1], X_train_vals.shape[2])))
basic_model.add(Dense(prediction_size))
basic_model.compile(loss='mae', optimizer='adam')