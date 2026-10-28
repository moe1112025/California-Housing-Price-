"""California Housing dataset regression demonstration.

Author: Moe Htet Ar Kar (Phoe Cho)

The source data is from the 1990 U.S. Census and is not a current appraisal tool.
"""

import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title = "California Housing Analytics", page_icon = "🏡", layout = "wide")
st.markdown("<style>" + """.stApp{background:linear-gradient(145deg,#101a24 0%,#192a32 55%,#101a22 100%);color:#edf7f4;} [data-testid='stSidebar']{background:#192b30;} 
           [data-testid='stMetric']{background:#20373a;border:1px solid #436b62;padding:14px;border-radius:12px;} .stButton>button{background:#79c7ad;color:#122620;border:0;border-radius:9px;
           font-weight:700;} h1,h2,h3{color:#e7fff5;}""" + "</style>", unsafe_allow_html = True)

st.title("🏡 California Housing Analytics")
st.caption("Census block-group features · Linear Regression")
st.warning("The dataset is based on the 1990 U.S. Census. Estimates are educational examples, not current market valuations or appraisals.")


@st.cache_resource
def train_housing_model():

    """Fetch the scikit-learn dataset and train a baseline regression model."""
    housing = fetch_california_housing(as_frame = True)
    frame = housing.frame
    features = frame.drop(columns = ["MedHouseVal"])
    target = frame["MedHouseVal"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size = 0.2, random_state = 42
    )
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    predicted = regressor.predict(x_test)
    mse = mean_squared_error(y_test, predicted)
    r2 = r2_score(y_test, predicted)
    return regressor, float(mse), float(r2), list(features.columns), len(frame)


try:
    model, mse, r2, feature_names, row_count = train_housing_model()
except Exception as exc:
    st.error("Could not download or train from the California Housing dataset.")
    st.info("Check your internet connection. Scikit-learn downloads this dataset on the first run and caches it locally.")
    st.exception(exc)
    st.stop()

st.sidebar.header("District characteristics")
median_income = st.sidebar.slider("Median income (in $10,000)", 0.5, 15.0, 3.5)
house_age = st.sidebar.slider("Median house age (years)", 1.0, 52.0, 28.0)
average_rooms = st.sidebar.slider("Average rooms per household", 1.0, 10.0, 5.0)
average_bedrooms = st.sidebar.slider("Average bedrooms per household", 0.5, 3.0, 1.0)
population = st.sidebar.slider("Block-group population", 100, 10000, 1200)
average_occupancy = st.sidebar.slider("Average household occupancy", 1.0, 10.0, 3.0)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 37.88)
longitude = st.sidebar.slider("Longitude", -124.3, -114.3, -122.23)

input_data = pd.DataFrame(
    [[median_income, house_age, average_rooms, average_bedrooms, population, average_occupancy, latitude, longitude]],
    columns = feature_names,
)

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Input summary")
    st.dataframe(input_data, use_container_width = True, hide_index = True)
    st.metric("Training dataset rows", f"{row_count:,}")

with right:

    st.subheader("Valuation report")
    if st.button("Estimate median house value", use_container_width = True):

        # The dataset's target is in $100,000 units, so convert to USD.
        predicted_value_usd = float(model.predict(input_data[feature_names])[0]) * 100000
        st.metric("Estimated median house value", f"${predicted_value_usd:,.0f}")
        st.caption("Estimate from historical census data; not a present-day property appraisal.")
    else:
        st.info("Adjust the district features and run the estimate to see the model output.")

st.subheader("Held-out regression metrics")
metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Mean squared error", f"{mse:.4f}")
metric_col2.metric("R² score", f"{r2:.4f}")
st.caption("MSE is measured in squared $100,000 target units because evaluation uses the source target scale.")

st.divider()
st.caption("Educational housing regression project · Historical data · Not for lending or appraisal decisions")
