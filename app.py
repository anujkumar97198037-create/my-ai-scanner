import streamlit as st
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.set_page_config(page_title="My AI Scanner", page_icon="📈")

st.title("🤖 Market AI Scanner")
st.write("Ye AI market scan karke trend batata hai.")

ticker = st.text_input("Stock Symbol daalein (e.g. RELIANCE.NS, SBIN.NS, ^NSEI):", "RELIANCE.NS")

if st.button('Scan Now'):
    with st.spinner('AI analysis kar raha hai...'):
        data = yf.download(ticker, period="1y", interval="1d")
        if not data.empty:
            # Indicators
            data.ta.macd(append=True)
            data.ta.bbands(append=True)
            data.ta.rsi(append=True)
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data.dropna(inplace=True)

            # AI Training
            features = ['Close', 'RSI_14', 'MACD_12_26_9', 'BBL_5_2.0', 'BBU_5_2.0']
            X = data[features]
            y = data['Target']
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X[:-1], y[:-1])

            # Result
            prob = model.predict_proba(X.tail(1))[0]
            pred = "UPAR JAYEGA 📈" if prob[1] > prob[0] else "GIRNE WALA HAI 📉"
            
            st.subheader(f"Result: {pred}")
            st.metric("Confidence", f"{round(max(prob)*100, 2)}%")
            st.line_chart(data['Close'].tail(30))
        else:
            st.error("Data nahi mila! Symbol check karein.")
