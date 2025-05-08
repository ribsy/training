
import sqlite3
import hashlib
import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageFont
import os
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
import re
import matplotlib.font_manager as fm


from scipy.stats import beta
from scipy.stats import gamma


st.set_page_config(
        page_title="FORECASTER TRAINING",
        page_icon=":shark:",
        #layout="wide",
        initial_sidebar_state="expanded",  # or "collapsed"
        # Apply the theme from the config file
        #theme="dark"
    )

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()

    #CREATE USER TABLE
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    ''')
    conn.commit()

    #CREATE PROBS TABLE
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS probs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unlikely NUMERIC NOT NULL,
            probable NUMERIC NOT NULL,
            likely NUMERIC NOT NULL,
            highly_likely NUMERIC NOT NULL
        )
    ''')
    conn.commit()

    #CREATE FORECAST TABLE
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            low_cow NUMERIC NOT NULL,
            high_cow NUMERIC NOT NULL,
            low_elph NUMERIC NOT NULL,
            high_elph NUMERIC NOT NULL,
            score REAL NOT NULL,
            money_value REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Hash the password for secure storage
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Add a new user to the database
def add_user_to_db(username, hashed_password, role='user'):
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                       (username, hashed_password, role))
        conn.commit()
        st.success("Account created successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose another.")
    conn.close()

# Function to write slider values to the database
def write_slider_values(slider1, slider2, slider3, slider4):
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO probs (unlikely, likely, probable, highly_likely) VALUES (?, ?, ?, ?)",
                   (slider1, slider2, slider3, slider4))
    conn.commit()
    conn.close()

def write_elephant_values(username, low_cow, high_cow, low_elph, high_elph, score, money_value):
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO forecasts (username, low_cow, high_cow, low_elph, high_elph, score, money_value) VALUES (?, ?, ?, ?, ?, ?,?)",
                   (username, low_cow, high_cow, low_elph, high_elph, score, money_value))
    conn.commit()
    conn.close()

# Verify user credentials and retrieve user role
def verify_user(username, password):
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?",
                   (username, hashed_password))
    result = cursor.fetchone()
    conn.close()
    return (result is not None, result[0] if result else None)

# Sign-up function for new users
def signup():
    new_username = st.text_input("Create Username")
    new_password = st.text_input("Create Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            hashed_password = hash_password(new_password)
            add_user_to_db(new_username, hashed_password)
            st.session_state['role'] = 'user'
            st.session_state['username'] = new_username


# Login function for registered users
def login():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        is_valid, role = verify_user(username, password)
        if is_valid:
            st.success(f"Welcome back, {username}!")
            st.session_state['role'] = role
            st.session_state['username'] = username
        else:
            st.error("Invalid username or password")

#def prob_chart_back(prob_val, name_val):
#    fig, ax = plt.subplots(figsize=(8, 2))
#    ax.hist(prob_val, bins=100, edgecolor='black')  # Adjust bins as needed
#    #ax.set_xlim(0, 100)
#    ax.set_xlabel(name_val)
#    ax.set_ylabel(name_val)
#    ax.set_title('Histogram of Probability Words')
#    st.pyplot(fig)

def prob_chart(prob_val, name_val):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, 100)

    # Set black background
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Histogram with red bars and white edges
    ax.hist(prob_val, color='red')

    # Set white axis labels and ticks

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel(name_val, color='white')
    ax.set_ylabel(name_val, color='white')

    # Set white title
    ax.set_title('Histogram of Probability Words', color='white')

    st.pyplot(fig)


# Member dashboard for regular users
def probability_words():

    if 'username' in st.session_state:
        st.title("Probability Words")
        st.write(f"What do these words mean as probabilities, {st.session_state['username']}?")
    else:
        st.warning("Please log in to access the dashboard.")

    # Create four sliders
    slider1 = st.slider("Unlikely", 0, 100, 0, key='slider_1')  # Default value is 0
    slider2 = st.slider("Probable", 0, 100, 0, key='slider_2')  # Default value is 0
    slider3 = st.slider("Likely", 0, 100, 0, key='slider_3')  # Default value is 0
    slider4 = st.slider("Highly Likely", 0, 100, 0, key='slider_4')  # Default value is 0

    st.session_state['slider1'] = slider1
    st.session_state['slider2'] = slider2
    st.session_state['slider3'] = slider3
    st.session_state['slider4'] = slider4

    #st.write(st.session_state['slider_values


    if 'show_group_results' not in st.session_state:
        st.session_state['show_group_results'] = False
        st.session_state['rerun_group_results'] = False

    # Write slider values to the database when a button is clicked
    if st.button("Save Slider Values"):
        write_slider_values(st.session_state['slider1'], st.session_state['slider2'], st.session_state['slider3'], st.session_state['slider4'])
        st.success("Slider values saved to database!")
        st.session_state['show_group_results'] = True  # Set the session state variable to True

    #st.write("")
    #st.write(f"ONLY AFTER submitting your results – click on the 'Show Group Results' button :sunglasses:")
    #st.write()

    if st.session_state['show_group_results'] or st.session_state['rerun_group_results']:
        if st.button("Refresh Group Results"):
            st.session_state['rerun_group_results'] = True

        conn = sqlite3.connect('train.db')
        probs_data = conn.execute('SELECT unlikely, probable, likely, highly_likely FROM probs').fetchall()
        conn.close()

        unlikely = [row[0] for row in probs_data]
        probable = [row[1] for row in probs_data]
        likely = [row[2] for row in probs_data]
        highly_likely = [row[3] for row in probs_data]

        prob_chart(unlikely, "Unlikely")
        prob_chart(probable, "Probable")
        prob_chart(likely, "Likely")
        prob_chart(highly_likely, "Highly")

def markdown_box(bold_title, text_vals):
    """
    Displays a text box with a bold title and text values using markdown.
    """
    html_code = f"""
        <style>
        .boxed-text {{
            border: 1px solid #000;
            padding: 10px;
            border-radius: 5px; /* Optional: Add rounded corners */
        }}
        </style>
        <div class="boxed-text">
            <B>{bold_title}</B><br>
            {text_vals}
        </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)



def forecast_elephant():
    st.title("FORECASTING LARGE MAMMALS")

    st.markdown(
      """
      <style>
      img {
          border: 3px solid darkred; /* Adjust border width and color as needed */
          border-radius: 10px; /* Optional: Add rounded corners */
      }
      </style>
      """,
      unsafe_allow_html=True,
    )
    st.image("./data/elephant_forecast.png")

    st.markdown(
        """
        <style>
        .boxed-text {
            border: 1px solid #000;
            padding: 10px;
            border-radius: 5px; /* Optional: Add rounded corners */
        }
        </style>
        <div class="boxed-text">
            <B>THE ART OF DECOMPOSITIONAL FORECASTING</B><br>
            How do your predict things you know almost nothing about? <B>DECOMPOSITION!</B> In this example, you will decompose various
            large mammals into smaller creatures...<b>cows</b> and <b>kirks</b> aka humans to be exact. See the instructions for each section below.<br><br>
            <B>PROGRESSIVE GAMBELING</B><br>
            To help calibrate you (turn you into a bookie) money will be provided.  Your goal is to grow it as much as possible relative to your peers.
            <B>INITIAL FUNDS: $100</B>
        </div>
       """,
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("Play A New Game"):
        st.session_state.money_value = 100  # Reset money_value
        # Reset other session variables used in the function:
        st.session_state.high_cow_lbs = 0
        st.session_state.low_cow_lbs = 0
        st.session_state.mean_cow_lbs = 0
        st.session_state.high_elph_lbs = 0
        st.session_state.low_elph_lbs = 0
        st.session_state.score = 0
        st.rerun()  # Rerun the script to reflect the changes

    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize money_value in session state
    if "money_value" not in st.session_state:
        st.session_state.money_value = 100

    # Display current money value
    #st.write(f"Initial Money Value: ${round(st.session_state.money_value,2)}")



    # Add the dropdown menu
    mammal_options = ["Rhinoceros", "Hippopotamus","Elephant"]
    selected_mammal = st.selectbox("Select Target Mammal", mammal_options)

    st.divider()

    st.markdown(
        """
        <style>
        .boxed-text {
            border: 1px solid #000;
            padding: 10px;
            border-radius: 5px; /* Optional: Add rounded corners */
        }
        </style>
        <div class="boxed-text">
            <B>USING PEOPLE TO PREDICT THE WEIGHT OF A COW</B><br>
            How many 200 lbs males fit into a cow? Use you imagination. Can you see how many fit? What is your upper bound? What is your lower bound?
            For your upper value, there should be an 90% probability that the value is at or below the upper bound.
            For your lower value, there should be an 90% probability that the value is at or above the lower bound.
            This is your 80% prediction range.<br><br>
            <B>PROGRESSIVE GAMBELING</B><br>
            You will make forecasts about three progressively larger mammals. Your goal is to beat your opponents!
            <B>INITIAL FUNDS: $100</B>
        </div>
       """,
        unsafe_allow_html=True,
    )

    # Row 1
    col1 = st.columns(1)  # Create two columns in the first row
    with col1[0]:

        # Create the slider with a range
        cow_range = st.slider(
            " ",
            min_value=0,
            max_value=100,
            value=(0, 5),
            key="cow_slider"
        )

        low_cow = cow_range[0]
        high_cow = cow_range[1]

        low_cow_lbs = low_cow * 200
        high_cow_lbs = high_cow * 200
        mean_cow_lbs = (high_cow_lbs + low_cow_lbs) / 2

        st.session_state['high_cow_lbs'] = high_cow_lbs
        st.session_state['low_cow_lbs'] = low_cow_lbs
        st.session_state['mean_cow_lbs'] = mean_cow_lbs

        # Display the selected range
        #st.write(f"Lower: {low_cow} - {high_cow} Kirks")
        st.write(f"Lower Bound Cow Weight: {low_cow_lbs} lbs")
        st.write(f"Upper Bound Cow Weight: {high_cow_lbs} lbs")
        st.write(f"Unit Cow Weight: {mean_cow_lbs} lbs")

    st.divider()

    st.markdown(
        """
        <style>
        .boxed-text {
            border: 1px solid #000;
            padding: 10px;
            border-radius: 5px; /* Optional: Add rounded corners */
        }
        </style>
        <div class="boxed-text">
            <B>USING COWS TO PREDICT THE WEIGHT OF AN ELEPHANTS (OR OTHER BIGGIES)</B><br>
            How many cows fit into an large mammal? Use your imagination again. You will use the average cow weight you predictive above to do this.
            First determine what your upper bound count of cows is? Next, what is your lower bound?
            For your upper value, there should be an 90% probability that the value is at or below the upper bound.
            For your lower value, there should be an 90% probability that the value is at or above the lower bound.
            This is your 80% prediction range.
        </div>
       """,
        unsafe_allow_html=True,
    )

    # Row 2
    col2 = st.columns(1)  # Create two columns in the second row
    with col2[0]:
        # Create the slider with a range
        elph_range = st.slider(
            "",
            min_value=0,
            max_value=100,
            value=(0, 5),
            key="elph_slider"
        )

        low_elph = elph_range[0]
        high_elph = elph_range[1]

        # Display the selected range
        #st.write(f"Selected range: {low_elph} - {high_elph}")

    col3 = st.columns(1)
    with col3[0]:
        #if st.button("Calculate Elephant"):
        high_elph_lbs = high_elph * st.session_state['mean_cow_lbs']
        low_elph_lbs = low_elph * st.session_state['mean_cow_lbs']
        st.session_state['high_elph_lbs'] = high_elph_lbs
        st.session_state['low_elph_lbs'] = low_elph_lbs

        if 'high_elph_lbs' in st.session_state:

            if selected_mammal == "Elephant":
                low_val = 4000
                high_val = 14000
            elif selected_mammal == "Rhinoceros":
                low_val = 4400
                high_val = 8000
            elif selected_mammal == "Hippopotamus":
                low_val = 2200
                high_val = 9900
            elif selected_mammal == "T-Rex":
                low_val = 11000
                high_val = 16000
            else:
                low_val = 4400
                high_val = 8000

            st.write(f"Lower Bound Mammal Weight: {st.session_state['low_elph_lbs']} lbs")
            st.write(f"Upper Bound Mammal Weight: {st.session_state['high_elph_lbs']} lbs")
            score = modified_brier_score(st.session_state['low_elph_lbs'], st.session_state['high_elph_lbs'], low_val, high_val)
            score = round(score,2)
            st.session_state['score'] = score
            #st.write(f"Modified Brier-like Score: {score}")

    if st.button("Submit Large Mammal Forecast"):
         # Subtract score from money_value

         if st.session_state.score < 1:
            st.session_state.money_value += (1 - st.session_state.score) * 100
         elif st.session_state.score < 10:
            st.session_state.money_value += st.session_state.score + 50
         else:
            st.session_state.money_value -= st.session_state.score

         # Display updated money value
         st.write(f"Updated Money Value: ${round(st.session_state.money_value,2)}")

         st.write(f"Forecasting Score: {st.session_state['score']}")
         write_elephant_values(st.session_state['username'], st.session_state['low_cow_lbs'], st.session_state['high_cow_lbs'],
                               st.session_state['low_elph_lbs'], st.session_state['high_elph_lbs'],
                               st.session_state['score'], st.session_state['money_value'])
         if st.success("Mammal Forecast Saved!"):
            st.session_state['elephant_saved'] = True  # Set the session state variable to True

    if 'elephant_saved' in st.session_state:
        st.markdown(
            """
            <style>
            .boxed-text {
                border: 1px solid #000;
                padding: 10px;
                border-radius: 5px; /* Optional: Add rounded corners */
            }
            </style>
            <div class="boxed-text">
                <B>SCORING FORECASTS</B><br>
                The lower your score is the better. A score of 0 means you are exactly right in term of the mammals weight range.
                The futher off you are from that range, the higher the score is.  A score below 100 is incredibly good.
                And anything below 1,000 is not bad.
            </div>
          """,
            unsafe_allow_html=True,
        )

        st.write("\n")
        if st.button("View All Mammal Forecasts and Scores"):
            view_forecasts()


#Modified Brier Score For Elephant Forecasts
def modified_brier_score(forecast_lower, forecast_upper, actual_lower, actual_upper):
    # Calculate the squared differences between the forecast and actual range boundaries
    # Normalize like Brier
    lower_diff = (forecast_lower - actual_lower) ** 2/(forecast_upper - forecast_lower)
    upper_diff = (forecast_upper - actual_upper) ** 2/(forecast_upper - forecast_lower)

    # Calculate the average of the squared differences (Brier-like score)
    score = round(((lower_diff + upper_diff) / 2)/100,2)
    return score

# Display forecast data
def view_forecasts():
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, low_cow, high_cow, low_elph, high_elph, score, money_value FROM forecasts")
    fc_data = cursor.fetchall()
    conn.close()

    # Get column names from the cursor description
    column_names = [description[0] for description in cursor.description]

    # Create a pandas DataFrame with column names
    df = pd.DataFrame(fc_data, columns=column_names)

    # Display the DataFrame using st.dataframe()
    st.dataframe(df)

def calculate_hdi(dist, mass=0.89, size=1000):
    """
    Calculates the approximate Highest Density Interval (HDI).

    Args:
        dist: A scipy.stats distribution object.
        mass (float): The probability mass to include in the HDI.
        size (int): The size of the x_vals array.

    Returns:
        A tuple containing the lower and upper bounds of the HDI.
    """
    x_vals = np.linspace(0, 1, size)
    pdf = dist.pdf(x_vals)
    pdf = pdf / pdf.sum()
    idx = np.argsort(pdf)[::-1]
    mass_cum = 0
    indices = []
    for i in idx:
        mass_cum += pdf[i]
        indices.append(i)
        if mass_cum >= mass:
            break
    return x_vals[np.sort(indices)[[0, -1]]]

def transfer_values():
    st.session_state.mone_open = st.session_state.total_open
    st.session_state.mone_fixed = st.session_state.total_fixed
    st.session_state.mtwo_open = 0
    st.session_state.mthree_open = 0
    st.session_state.mfour_open = 0
    st.session_state.mtwo_fixed = 0
    st.session_state.mthree_fixed = 0
    st.session_state.mfour_fixed = 0
    st.session_state.total_open = st.session_state.mone_open
    st.session_state.total_fixed = st.session_state.mone_fixed
    #st.experimental_rerun()

def append_risks_and_calculate_rates():
       # Get current risk values from session state
       observed_risks = [st.session_state[key] for key in ["mone_open", "mtwo_open", "mthree_open", "mfour_open"]]
       departed_risks = [st.session_state[key] for key in ["mone_fixed", "mtwo_fixed", "mthree_fixed", "mfour_fixed"]]

       # Append to session state lists
       st.session_state.observed_risks.extend(observed_risks)
       st.session_state.departed_risks.extend(departed_risks)

       # Calculate rates of change (example using simple differences)
       obs_rate_change = st.session_state.observed_risks[-1] - st.session_state.observed_risks[-5] if len(st.session_state.observed_risks) >= 5 else 0
       dpt_rate_change = st.session_state.departed_risks[-1] - st.session_state.departed_risks[-5] if len(st.session_state.departed_risks) >= 5 else 0

       # Calculate average departed/observed ratio and trend
       avg_ratio_current = np.mean(np.array(st.session_state.departed_risks) / np.array(st.session_state.observed_risks)) if st.session_state.observed_risks and all(x != 0 for x in st.session_state.observed_risks) else 0
       avg_ratio_previous = np.mean(np.array(st.session_state.departed_risks[:-4]) / np.array(st.session_state.observed_risks[:-4])) if len(st.session_state.observed_risks) >= 5 and all(x != 0 for x in st.session_state.observed_risks[:-4]) else 0
       ratio_trend = "Increased" if avg_ratio_current > avg_ratio_previous else "Decreased" if avg_ratio_current < avg_ratio_previous else "Unchanged"

       # Display results
       st.write(f"Observed Risks Rate Change: {obs_rate_change}")
       st.write(f"Departed Risks Rate Change: {dpt_rate_change}")
       st.write(f"Average Departed/Observed Ratio Trend: {ratio_trend}")

def burn_trend_graph(risk_list,id_val, title_val):
       # Create a Pandas DataFrame
       df = pd.DataFrame({'Observed Risks': risk_list, 'Time Period': range(1, len(risk_list) + 1)})

       # Calculate average trend line
       average_risk = np.mean(risk_list)
       df['Average Trend'] = average_risk

       # Calculate average rate for every 4 risks
       average_rates = []
       for i in range(0, len(risk_list), 4):
          average_rates.extend([np.mean(risk_list[i:i + 4])] * 4)  # Extend for 4 periods

       df['Average Rate'] = average_rates[:len(risk_list)]  # Truncate if necessary

       #st.write(df)


       # Gamma-Poisson parameters
       alpha_prior = 0.5  # Adjust as needed
       beta_prior = 0.001 # Adjust as needed

       # Calculate posterior parameters
       alpha_posterior = alpha_prior + sum(risk_list)
       beta_posterior = beta_prior + len(risk_list)

       # Calculate credible interval for the average risk (rate)
       lower_bound = gamma.ppf(0.025, alpha_posterior, scale=1/beta_posterior)
       upper_bound = gamma.ppf(0.975, alpha_posterior, scale=1/beta_posterior)

       # Add credible interval to DataFrame
       df['Lower Bound'] = lower_bound
       df['Upper Bound'] = upper_bound

       # Create the time series graph using Plotly Express
       fig = px.line(df, x='Time Period', y=['Observed Risks', 'Average Rate', 'Average Trend'],
              title=title_val)


       # Add credible interval as a filled area
       fig.add_trace(go.Scatter(
           x=df['Time Period'],
           y=df['Upper Bound'],
           mode='lines',
           line=dict(width=0),
           fillcolor='rgba(0,100,80,0.2)',
           fill='tonexty',
           name='Credible Interval'
       ))
       fig.add_trace(go.Scatter(
           x=df['Time Period'],
           y=df['Lower Bound'],
           mode='lines',
           line=dict(width=0),
           fillcolor='rgba(0,100,80,0.2)',
           fill='tonexty',
           name='Credible Interval'
       ))

       # Customize the graph
       fig.update_layout(
           xaxis_title='Time Period',
           yaxis_title='Observed Risks',
           legend_title_text='Series',
           showlegend=True
       )

       # Display the graph in Streamlit
       st.plotly_chart(fig, key=id_val)


def burn_ratio_trend_graph(observed_risks, departed_risks, id_val, title_val, sla_value):
    """
    Plots the trend of departed risks over observed risks with a beta distribution average,
    a cumulative credible interval that is a smooth curve following the average, and an SLA line.
    Includes the SLA line in the legend, projects the average ratio four time periods out with uncertainty,
    and gives more weight to the last two ratios for the trend calculation.
    The projected ratio with uncertainty is bounded to stay within the range of 0 to 1.
    The trend line is also capped at 1 and is always visible on the graph.

    Args:
        observed_risks: A list of observed risks over time.
        departed_risks: A list of departed risks over time.
        id_val: A unique identifier for the graph.
        title_val: The title of the graph.
        sla_value: The value of the SLA to be plotted as a horizontal line.
    """
    # Calculate cumulative observed and departed risks
    cumulative_observed = np.cumsum(observed_risks)
    cumulative_departed = np.cumsum(departed_risks)

    # Calculate cumulative ratios and average trend
    cumulative_ratios = cumulative_departed / cumulative_observed
    cumulative_ratios = np.nan_to_num(cumulative_ratios, nan=0)  # Handle potential NaN values
    average_ratio = np.mean(cumulative_ratios)

    # Create a Pandas DataFrame
    df = pd.DataFrame({'Ratio': cumulative_ratios,
                       'Time Period': range(1, len(cumulative_ratios) + 1)})
    df['Average Trend'] = average_ratio

    # Calculate credible interval for each time period
    lower_bounds = []
    upper_bounds = []
    for i in range(len(cumulative_ratios)):
        # Calculate cumulative totals up to current time period
        alpha_up_to_i = cumulative_departed[i] + 1
        beta_up_to_i = cumulative_observed[i] - cumulative_departed[i] + 1
        lower_bound = beta.ppf(0.025, alpha_up_to_i, beta_up_to_i)  # Use updated parameters
        upper_bound = beta.ppf(0.975, alpha_up_to_i, beta_up_to_i)  # Use updated parameters
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    df['Lower Bound'] = lower_bounds
    df['Upper Bound'] = upper_bounds

    # Project average ratio four time periods out with uncertainty
    last_two_ratios = cumulative_ratios[-2:]  # Focus on the last two ratios
    slope = (last_two_ratios[-1] - last_two_ratios[0])  # More weight on recent trend
    projected_ratios = [cumulative_ratios[-1] + i * slope for i in range(1, 5)]
    projected_time_periods = [df['Time Period'].max() + i for i in range(1, 5)]

    # Add uncertainty (e.g., using standard deviation of recent ratios)
    std_dev = np.std(cumulative_ratios[-5:])  # Adjust the window as needed
    uncertainty = [std_dev * i for i in range(1, 5)]

    # Bound the uncertainty and projected ratios so they stay within [0, 1]
    projected_ratios = [min(1, r) for r in projected_ratios]  # Cap projected ratios at 1
    lower_bounds_proj = [max(0, r - u) for r, u in zip(projected_ratios, uncertainty)]
    upper_bounds_proj = [min(1, r + u) for r, u in zip(projected_ratios, uncertainty)]

    # Add projected data to DataFrame for plotting
    projected_df = pd.DataFrame({
        'Ratio': projected_ratios,
        'Time Period': projected_time_periods,
        'Lower Bound': lower_bounds_proj,
        'Upper Bound': upper_bounds_proj
    })
    # Ensure the average trend is at or below 1
    projected_df['Average Trend'] = min(1, np.mean(projected_ratios))  

    df = pd.concat([df, projected_df])

    # Create the time series graph using Plotly Express with smoothing
    fig = px.line(df, x='Time Period', y=['Ratio'], title=title_val)
    fig.update_traces(mode='lines+markers', line=dict(shape='spline', smoothing=1.3))

    # Add SLA line as a trace for legend
    fig.add_trace(
        go.Scatter(
            x=[df['Time Period'].min(), df['Time Period'].max()],
            y=[sla_value, sla_value],
            mode="lines",
            line=dict(color="Beige", width=2, dash="dash"),
            name="SLA"
        )
    )

    # Add credible interval as a filled area with smoothing and fill='toself'
    fig.add_trace(go.Scatter(
        x=df['Time Period'],
        y=df['Lower Bound'],
        mode='lines',
        line=dict(width=0, shape='spline', smoothing=1.3),
        fillcolor='rgba(0,100,80,0.2)',
        fill=None,  # No fill for lower bound
        name='Credible Interval',
        showlegend=False  # Hide this trace from the legend
    ))
    fig.add_trace(go.Scatter(
        x=df['Time Period'],
        y=df['Upper Bound'],
        mode='lines',
        line=dict(width=0, shape='spline', smoothing=1.3),
        fillcolor='rgba(0,100,80,0.2)',
        fill='tonexty',  # Fill to previous trace for credible interval area
        name='Credible Interval'  # Use the same name for both upper and lower bounds
    ))

    # Find the index of the 'Ratio' trace and move it to the beginning if needed
    try:
        ratio_index = fig.data.index(next((trace for trace in fig.data if trace.name == 'Ratio'), None))
        if ratio_index is not None and ratio_index != 0:
            ratio_trace = fig.data[ratio_index]
            new_data = [ratio_trace] + list(fig.data[:ratio_index]) + list(fig.data[ratio_index + 1:])
            fig.data = new_data
    except ValueError:
        pass

    # Plot projected values with uncertainty
    fig.add_trace(
        go.Scatter(
            x=projected_df['Time Period'],
            y=projected_df['Ratio'],
            mode='lines+markers',
            line=dict(color='red', dash='dot'),
            name='Projected Ratio'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projected_df['Time Period'],
            y=projected_df['Lower Bound'],
            mode='lines',
            line=dict(color='red', width=0),
            fillcolor='rgba(255,0,0,0.2)',
            fill='tonexty',  # Fill to previous trace for uncertainty area
            name='Projection Uncertainty',
            showlegend=False  # Hide this trace from the legend
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projected_df['Time Period'],
            y=projected_df['Upper Bound'],
            mode='lines',
            line=dict(color='red', width=0),
            fillcolor='rgba(255,0,0,0.2)',
            fill='tonexty',  # Fill to previous trace for uncertainty area
            name='Projection Uncertainty'  # Use the same name for both upper and lower bounds
        )
    )

    # Customize the graph layout
    fig.update_layout(
        yaxis=dict(range=[0, 1.1]),  # Set y-axis range to ensure visibility of trend line at 1
        xaxis=dict(range=[1, df['Time Period'].max()]),  # Start x-axis at 1
        xaxis_title='Time Period',
        yaxis_title='Departed/Observed Ratio',
        legend_title_text='Series',
        showlegend=True
    )

    # Display the graph in Streamlit
    st.plotly_chart(fig, key=id_val)

def play_burndown():

    # Initialize Session State for Lists
    if "observed_risks" not in st.session_state:
        st.session_state.observed_risks = []
    if "departed_risks" not in st.session_state:
        st.session_state.departed_risks = []

    st.title("BURNDOWN")

    a = "You use burndown to measure the rate of risk elimination against a target SLA. "
    b = "Underneath the hood it uses a first seen time stamps and a closed or eliminated time stamp. "
    c = "Burndowns are really useful when you are focusing a critical class of risk and you want "
    d = "to know if you are consistently keeping up with your SLA...over time."
    burn_msg = a + b + c + d
    markdown_box("THE HITS AND MISSES METRIC!", burn_msg)

    st.write("\n")

    col_sla_1, col_sla_2 = st.columns(2)
    with col_sla_1:
        st.markdown("######")
        st.markdown("##### SLA")
    with col_sla_2:
        sla_val = st.number_input("",value=.5, key='sla_val', step=0.05)
        st.session_state['sla'] = sla_val

    st.divider()

    #with st.form(key='multi_row_form'):
      # Header Row
    col_header1, col_header3, col_header4, col_header5, col_header6, col_header7 = st.columns(6)
    with col_header1:
        st.markdown("##### RISK")
    #with col_header2:
    #    st.markdown("##### PRIOR")
    with col_header3:
        st.markdown("##### T1")
    with col_header4:
          st.markdown("##### T2")
    with col_header5:
        st.markdown("##### T3")
    with col_header6:
        st.markdown("##### T4")
    with col_header7:
        st.markdown("##### TOTAL")

    # ROW ONE
    col1, col3, col4, col5, col6, col7 = st.columns(6)
    with col1:
        st.markdown("######")
        st.markdown("OPENED")
    #with col2:
    #    prior_open = st.number_input("", value=1, step=1, key = "prior_open", disabled=True)
    with col3:
        mone_open = st.number_input("", step=1, key = "mone_open")
    with col4:
        mtwo_open = st.number_input("", step=1, key = "mtwo_open")
    with col5:
        mthree_open = st.number_input("", step=1, key = "mthree_open")
    with col6:
        mfour_open = st.number_input("", step=1, key = "mfour_open")
    with col7:
        # Calculate total_open
        total_open = mone_open + mtwo_open + mthree_open + mfour_open
        st.session_state['total_open'] = total_open

        #st.session_state['total_open_input'] = total_open - 1. # Address Prior
        st.number_input("", value=None, step=1, key = "total_open", disabled = True)

    # ROW TWO
    col8, col10, col11, col12, col13, col14 = st.columns(6)
    with col8:
        st.markdown("######")
        st.markdown("FIXED")
    #with col9:
    #    prior_fixed = st.number_input("", value=1, step=1, key = "prior_fixed", disabled=True)
    with col10:
        mone_fixed = st.number_input("", step=1, key = "mone_fixed")
    with col11:
        mtwo_fixed = st.number_input("", step=1, key = "mtwo_fixed")
    with col12:
        mthree_fixed = st.number_input("", step=1, key = "mthree_fixed")
    with col13:
        mfour_fixed = st.number_input("",  step=1, key = "mfour_fixed")
    with col14:
        total_fixed = mone_fixed + mtwo_fixed + mthree_fixed + mfour_fixed
        st.session_state['total_fixed'] = total_fixed
        #st.session_state['total_fixed_input'] = total_fixed - 1. # Address Prior
        st.number_input("", value=None, step=1, key = "total_fixed", disabled=True)

    #st.write(st.session_state['total_fixed'])
    #st.write(st.session_state['total_open'])

    #total_open = prior_open + mone_open + mtwo_open + mthree_open + mfour_open
    #total_fixed = prior_fixed + mone_fixed + mtwo_fixed + mthree_fixed + mfour_fixed

    if 'show_graph' not in st.session_state:
        st.session_state['show_graph'] = False

    st.divider()

    if st.button("Calculate Burndown Rate"):
         st.session_state['show_graph'] = True
    else:
        if st.session_state['total_open'] == 0:
            st.session_state['show_graph'] = True

    st.divider()


    # if st.button("Calculate Arrival, Departure, and Removal Rates"):

    #     # Gamma-Poisson rate estimation
    #     observed_risks = [st.session_state[key] for key in ["mone_open", "mtwo_open", "mthree_open", "mfour_open"]]
    #     departed_risks = [st.session_state[key] for key in ["mone_fixed", "mtwo_fixed", "mthree_fixed", "mfour_fixed"]]

    #     # Prior parameters (adjust as needed)
    #     alpha_prior = .5  # Shape parameter
    #     beta_prior = 0   # Rate parameter

    #     # Calculate posterior parameters
    #     alpha_obs_posterior = alpha_prior + sum(observed_risks)
    #     beta_obs_posterior = beta_prior + len(observed_risks)

    #     alpha_dpt_posterior = alpha_prior + sum(departed_risks)
    #     beta_dpt_posterior = beta_prior + len(departed_risks)

    #     # Estimate rate (mean of posterior Gamma distribution)
    #     estimated_obs_rate = alpha_obs_posterior / beta_obs_posterior

    #     estimated_dpt_rate = alpha_dpt_posterior / beta_dpt_posterior

    #     # Calculate 95% credible interval
    #     lower_obs_bound = gamma.ppf(0.025, alpha_obs_posterior, scale=1/beta_obs_posterior)
    #     upper_obs_bound = gamma.ppf(0.975, alpha_obs_posterior, scale=1/beta_obs_posterior)

    #     lower_dpt_bound = gamma.ppf(0.025, alpha_dpt_posterior, scale=1/beta_dpt_posterior)
    #     upper_dpt_bound = gamma.ppf(0.975, alpha_dpt_posterior, scale=1/beta_dpt_posterior)

    #     st.write(f"Estimated Arrival Rate: {estimated_obs_rate:.2f}")
    #     st.write(f"95% Arrival Credible Interval: ({lower_obs_bound:.2f}, {upper_obs_bound:.2f})")

    #     st.write(f"Estimated Departure Rate: {estimated_dpt_rate:.2f}")
    #     st.write(f"95% Departure Credible Interval: ({lower_dpt_bound:.2f}, {upper_dpt_bound:.2f})")

    #     st.write(f"AVERAGE RISK REMOVAL RATE: ({(estimated_dpt_rate/estimated_obs_rate)*100:.2f})%")

    # st.divider()

    # if st.button("Aggregate Values", on_click=transfer_values):
    #     pass  # No need for other code here

    # st.divider()

    if st.button("Append Risks And Calculate Trends"):
      # Get current risk values from session state
       observed_risks = [st.session_state[key] for key in ["mone_open", "mtwo_open", "mthree_open", "mfour_open"]]
       departed_risks = [st.session_state[key] for key in ["mone_fixed", "mtwo_fixed", "mthree_fixed", "mfour_fixed"]]

       # Append to session state lists
       st.session_state.observed_risks.extend(observed_risks)
       st.session_state.departed_risks.extend(departed_risks)

       # Calculate rates of change (example using simple differences)
       obs_rate_change = st.session_state.observed_risks[-1] - st.session_state.observed_risks[-5] if len(st.session_state.observed_risks) >= 5 else 0
       dpt_rate_change = st.session_state.departed_risks[-1] - st.session_state.departed_risks[-5] if len(st.session_state.departed_risks) >= 5 else 0

       # Calculate average departed/observed ratio and trend
       avg_ratio_current = np.mean(np.array(st.session_state.departed_risks) / np.array(st.session_state.observed_risks)) if st.session_state.observed_risks and all(x != 0 for x in st.session_state.observed_risks) else 0
       avg_ratio_previous = np.mean(np.array(st.session_state.departed_risks[:-4]) / np.array(st.session_state.observed_risks[:-4])) if len(st.session_state.observed_risks) >= 5 and all(x != 0 for x in st.session_state.observed_risks[:-4]) else 0
       ratio_trend = "Increased" if avg_ratio_current > avg_ratio_previous else "Decreased" if avg_ratio_current < avg_ratio_previous else "Unchanged"

       # Display results
       st.write(f"Observed Risks Rate Change: {obs_rate_change}")
       st.write(f"Departed Risks Rate Change: {dpt_rate_change}")
       st.write(f"Average Departed/Observed Ratio Trend: {ratio_trend}")

       burn_ratio_trend_graph(st.session_state.observed_risks, st.session_state.departed_risks, "burn_trend", "Cummulative Risk Burndown Trend With Uncertainty and SLA", st.session_state['sla'])

       burn_trend_graph(risk_list=st.session_state.observed_risks, id_val="test_arrive",
                        title_val="Risk Arrivals Over Time with Average Trend and Credible Interval")

       burn_trend_graph(risk_list=st.session_state.departed_risks, id_val="test_depart",
                        title_val="Risk Departures Over Time with Average Trend and Credible Interval")

    st.divider()

    if st.button("Calculate Trends"):

       # Calculate rates of change (example using simple differences)
       obs_rate_change = st.session_state.observed_risks[-1] - st.session_state.observed_risks[-5] if len(st.session_state.observed_risks) >= 5 else 0
       dpt_rate_change = st.session_state.departed_risks[-1] - st.session_state.departed_risks[-5] if len(st.session_state.departed_risks) >= 5 else 0

       # Calculate average departed/observed ratio and trend
       avg_ratio_current = np.mean(np.array(st.session_state.departed_risks) / np.array(st.session_state.observed_risks)) if st.session_state.observed_risks and all(x != 0 for x in st.session_state.observed_risks) else 0
       avg_ratio_previous = np.mean(np.array(st.session_state.departed_risks[:-4]) / np.array(st.session_state.observed_risks[:-4])) if len(st.session_state.observed_risks) >= 5 and all(x != 0 for x in st.session_state.observed_risks[:-4]) else 0
       ratio_trend = "Increased" if avg_ratio_current > avg_ratio_previous else "Decreased" if avg_ratio_current < avg_ratio_previous else "Unchanged"

       # Display results
       st.write(f"Observed Risks Rate Change: {obs_rate_change}")
       st.write(f"Departed Risks Rate Change: {dpt_rate_change}")
       st.write(f"Average Departed/Observed Ratio Trend: {ratio_trend}")

       burn_ratio_trend_graph(st.session_state.observed_risks, st.session_state.departed_risks, "burn_trend", "Cummulative Risk Burndown Trend With Uncertainty and SLA", st.session_state['sla'])

       burn_trend_graph(risk_list=st.session_state.observed_risks, id_val="test_arrive",
                        title_val="Risk Arrivals Over Time with Average Trend and Credible Interval")

       burn_trend_graph(risk_list=st.session_state.departed_risks, id_val="test_depart",
                        title_val="Risk Departures Over Time with Average Trend and Credible Interval")



    #if st.button("Append Risks and Calculate Trends", on_click=append_risks_and_calculate_rates):
    #    pass  # No need for other code here

    st.divider()

    # Reset button
    if st.button("Reset Burndown"):
        # Clear session variables
        for key in st.session_state.keys():
            if key.startswith('m') or key in ['total_open', 'total_fixed', 'observed_risks', 'departed_risks', 'sla', 'sla_val', 'show_graph']:
                del st.session_state[key]

        # Rerun the script to reflect the changes
        st.rerun()
            
    st.divider()
        
    if st.session_state['show_graph']:

        if st.session_state['total_fixed'] > st.session_state['total_open']:
            st.warning("You cannot have more fixed issues than open issues.  Please fix.")

        #prior_open = st.session_state['prior_open']
        #prior_fixed = st.session_state['prior_fixed']

        # Get Total Values
        #total_open = prior_open + mone_open + mtwo_open + mthree_open + mfour_open
        #total_fixed = prior_fixed + mone_fixed + mtwo_fixed + mthree_fixed + mfour_fixed

        # Calculate Beta Distribution Parameters
        alpha = st.session_state['total_fixed'] + 1
        beta_val = st.session_state['total_open'] - st.session_state['total_fixed'] + 1

        # Generate x values for the beta distribution
        x = np.linspace(0, 1, 100)

        # Calculate the probability density function (PDF)
        dist = beta(alpha, beta_val)
        pdf = dist.pdf(x)


        # Calculate HDI
        hdi = calculate_hdi(dist)

        plt.style.use('dark_background')

        # Plot the beta distribution
        fig, ax = plt.subplots()
        #ax.plot(x, pdf, 'r-', lw=2, label=f'Beta({alpha:.2f}, {beta_val:.2f})')
        ax.plot(x, pdf, 'r-', lw=2)

        # Plot the HDI
        ax.axvspan(hdi[0], hdi[1], color='gray', alpha=0.3, label='Confidence')

        # Plot the vertical line
        ax.axvline(st.session_state['sla'], color='red', linestyle='--', label=f'SLA = {round(st.session_state["sla"],3)}')

        ax.set_xlabel('RATE (Probability)')
        ax.set_ylabel('Strength (Density)')
        ax.set_title('Burndown Rate For Last 4 Time Periods – With Uncertainty')
        ax.legend()
        ax.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)

# Display all user data for admin users
def view_probs_data():
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM probs")
    user_data = cursor.fetchall()
    conn.close()


# Display all user data for admin users
def view_user_data():
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM users")
    user_data = cursor.fetchall()
    conn.close()

def random_number_generator(initial_val = 0):
  """Generates a random number between 1 and 40 (non-inclusive)."""
  if initial_val == 0:
    return random.randint(1,43)
  else:
    return random.randint(10, 33)

def track_random_numbers(num_requests=1, initial_value=1):
  """Generates an initial random number and compares subsequent numbers.

  Args:
    num_requests: The number of additional random numbers to generate.

  Returns:
    A list containing the initial value and "right" or "left" for each subsequent number.
  """
  #initial_value = random_number_generator()
  results = [initial_value]

  for _ in range(num_requests):
    new_value = random_number_generator()
    if new_value > initial_value:
      results.append("right")
    elif new_value < initial_value:
      results.append("left")
    else:
      results.append("equal")  # Handle the case where new_value equals initial_value

  return results

def range_scoring_function(lower_bound, upper_bound, forecast_value):
    """Calculates a Brier-like score for a range and forecast value,
       acting as a proper score. Rewards tighter ranges and
       penalizes values outside the range based on distance.
       Penalizes wide ranges even if midpoint is close to forecast value.
       Penalizes ranges based on width when the forecast is the midpoint
       Rewards tighter ranges that contain forecast value better than small distance from midpoint

    Args:
        lower_bound: The lower bound of the range.
        upper_bound: The upper bound of the range.
        forecast_value: The value to be scored within the range.

    Returns:
        The Brier-like score, a value between 0 and infinity.
        Lower scores indicate a better prediction.
    """
    # Check if the forecast value falls within the bounds
    if lower_bound <= forecast_value <= upper_bound:
        # Calculate the normalized distance from the midpoint
        range_width = upper_bound - lower_bound
        midpoint = (lower_bound + upper_bound) / 2
        distance_from_midpoint = abs(forecast_value - midpoint)

        # Prevent zero score for midpoint
        if distance_from_midpoint == 0 and range_width > 0:
            score = range_width / 100  # Penalize based on range width
        else:
            score = (distance_from_midpoint / range_width) ** 2

        # Reward tighter ranges (add range width as a penalty)
        score += range_width / 100  # Adjust the divisor to control the impact
        #score = score * range_width /10 # reward tighter ranges, scale by range width ->REMOVED

        # Penalize wide ranges even if midpoint is close
        if range_width > 20:  # Adjust threshold (20 here) as needed
            score += (range_width - 20) / 50  # Adjust divisor to control penalty scaling

    else:
        # If the forecast value is outside the bounds, apply a penalty based on distance
        if forecast_value < lower_bound:
            distance_outside = lower_bound - forecast_value
        else:  # forecast_value > upper_bound
            distance_outside = forecast_value - upper_bound

        score = 1 + (distance_outside / 10)  # Adjust the divisor to control the penalty scaling

    return round(score,3) * 10

def random_number_game_with_brier_score():
    """Streamlit interface for the random number game with Brier score."""

    st.title("Gambeling With The Rev Bayes")

    st.markdown(
      """
      <style>
      img {
          border: 3px solid darkred; /* Adjust border width and color as needed */
          border-radius: 10px; /* Optional: Add rounded corners */
      }
      </style>
      """,
      unsafe_allow_html=True,
    )
    st.image("data/rev_bayes_pool.png")
    #st.image("./data/rev_bayes_pool.png")

    a = "This is a calibration game. It's goal is to help you understand accuracy, precision and its costs – while confronting some amount of irriducible uncertainty. "
    b = "First, you are learning to forecast using ranges. The closer your range is to the 'true position' of the white ball...the bigger your reward. (Your range based forecasts will be scored.) "
    c = "Conversely, the farther away your range is from the ball, or the more spread out your ranges is (beyond reason)...the bigger your penalties. "
    d = "Also, the more information you seek, the more it costs you. And that cost gets larger the more information you ask for. "
    e = "At the end, you will make a precise guess about the balls location. That guess will also be scored – and you will be penalized if you are far off. "
    f = "NOTE: you will win awards for being close – with a whoppig $100 for being both precise and accurate...meaning by being spot on!"
    bayes_msg = a + b + c + d + e + f
    markdown_box("HOW THE GAME WORKS!", bayes_msg)

    st.divider()

    # Initialize session state variables
    if "initial_value" not in st.session_state:
        st.session_state.initial_value = None
    if "results" not in st.session_state:
        st.session_state.results = []
    if "scores" not in st.session_state:
        st.session_state.scores = []
    if "lower_bounds" not in st.session_state:  # Initialize lower bounds list
        st.session_state.lower_bounds = []
    if "upper_bounds" not in st.session_state:  # Initialize upper bounds list
        st.session_state.upper_bounds = []
    if "counter" not in st.session_state:
        st.session_state.counter = 100
    if "ball_count" not in st.session_state:
        st.session_state.ball_count = 0
    if "forecast_count" not in st.session_state:
        st.session_state.forecast_count = 0

    counter_placeholder = st.empty()

    counter_placeholder.write(f"Money: ${round(st.session_state.counter)}")

    if st.button("Play A New Game"):
        st.session_state.counter = 100
        st.session_state.ball_count = 0
        #st.write(f"Money: ${st.session_state.counter}")
        counter_placeholder.write(f"Money: ${round(st.session_state.counter)}")

        st.session_state.initial_value = random_number_generator(1)
        st.session_state.results = [st.session_state.initial_value]
        st.session_state.scores = []  # Clear scores when starting a new game
        st.session_state.lower_bounds = []  # Clear lower bounds when starting a new game
        st.session_state.upper_bounds = []  # Clear upper bounds when starting a new game
        st.session_state.forecast_count = 0 # set to zero
    else:
        counter_placeholder.write(f"Money: ${round(st.session_state.counter)}")

    if st.session_state.initial_value is not None:

        st.divider()

        if st.button("Roll Ball"):

          # Increment ball count
          st.session_state.ball_count += 1

          # Subtract from counter based on ball count
          if st.session_state.ball_count > 10:
            st.session_state.counter -= 10
          elif st.session_state.ball_count > 7:
            st.session_state.counter -= 7
          else:
            st.session_state.counter -= 5

          counter_placeholder.write(f"Money: ${round(st.session_state.counter)}")
          while True:  # Loop until a valid result is generated
              result = track_random_numbers(1, st.session_state.initial_value)[1]  # Get "right" or "left"
              if result != "equal":  # Check if result is not equal to initial value
                  break  # Exit the loop if result is valid
          st.session_state.results.append(result)
          #st.write(f"Result: {result}")

        # Display counter as dollar
        st.write("Ball Locations:", ", ".join(st.session_state.results[1:]))

        st.divider()
        forecast_lower = st.number_input("Forecast Lower", value=0)
        forecast_higher = st.number_input("Forecast Higher", value=0)

        if st.button("Forecast Range"):
            if st.session_state.initial_value is not None and st.session_state.forecast_count <= 2:

                # Increment forecast counter
                st.session_state.forecast_count += 1

                st.session_state.lower_bounds.append(forecast_lower)  # Store lower bound
                st.session_state.upper_bounds.append(forecast_higher)  # Store upper bound
                score = range_scoring_function(forecast_lower, forecast_higher, st.session_state.initial_value)
                score = round(score,3)

                if forecast_lower <= st.session_state.initial_value <= forecast_higher:
                  if score <= .5:
                    st.session_state.counter += (round((100 - score)) * 2)/st.session_state.forecast_count
                  elif score <= 1:
                    st.session_state.counter += (round((100 - score)) * 1.5)/st.session_state.forecast_count
                  elif score <= 10:
                    st.session_state.counter += (round((100 - score)) * 1.2)/st.session_state.forecast_count
                  else:
                    st.session_state.counter += round(100 - score)/st.session_state.forecast_count
                else:
                    st.session_state.counter += -round(score * 1.5)

                st.session_state.scores.append(score)  # Append score to the list

                st.write(f"Modified Brier Score: {score}")
                counter_placeholder.write(f"Money: ${round(st.session_state.counter)}")



            else:
                st.write(f"You must Play A New Game before you can Forecast A Range – or place your PRECISE bet.")


        st.divider()
        # Guess Ball Location input field
        guess_location = st.number_input("Guess Ball Location", value=0)

        if st.button("Guess Location"):
            if st.session_state.initial_value is not None:
                # Calculate absolute difference
                difference = abs(guess_location - st.session_state.initial_value)

                # Subtract difference * 5 from counter
                if difference == 0:
                  st.session_state.counter += 100
                elif difference == 1:
                  st.session_state.counter += 50
                elif difference == 2:
                  st.session_state.counter += 25
                elif difference == 3:
                  st.session_state.counter += 10
                else:
                  st.session_state.counter -= difference * 5

                # Display updated counter
                #st.write(f"Money: ${st.session_state.counter}")
                counter_placeholder.write(f"Money: ${round(st.session_state.counter)}")
                st.write(f"Original Ball Location: {st.session_state.initial_value}") # Added to reveal after Guess Location

            else:
                st.write("You must start a new game before guessing.")

        # Display all previous scores
        #st.write("Previous Scores:")
        #for i, score in enumerate(st.session_state.scores):
        #   st.write(f"Bet {i + 1}: {score}")

        # Display all previous forecasts
        #st.write("Previous Forecasts:")
        #for i in range(len(st.session_state.lower_bounds)):
        #    st.write(f"Bet {i + 1}: Lower Bound - {st.session_state.lower_bounds[i]}, Upper Bound Ball  - {st.session_state.upper_bounds[i]}")


def create_influence_diagram_from_text(text):
    G = nx.DiGraph()
    lines = text.strip().split('\n')
    node_positions = {}
    shapes = []
    annotations = []
    edges_x = []
    edges_y = []

    # 1. Add nodes to the dictionary first
    for line in lines:
        match = re.match(r'(Decision|Uncertainty|Outcome)\s+"([^"]+)"\s*(?:connects to:\s*([^,]+))?\s*(?:connects from:\s*([^,]+))?', line)
        if match:
            groups = match.groups()
            object_type = groups[0]
            title = groups[1]

            # Use Matplotlib to find the font and calculate text width
            font_prop = fm.FontProperties(family=['DejaVu Sans'], size=12)

            # Calculate the actual width and height of the text (tighter calculation)
            fig, ax = plt.subplots(figsize=(1, 1))
            renderer = fig.canvas.get_renderer()
            text_width, text_height = ax.text(0, 0, title, fontproperties=font_prop, bbox=dict(pad=0)).get_window_extent(renderer).size  # Reduced padding
            plt.close(fig)

            # Adjust shape dimensions, ensuring width is no more than 1.5 times the text width
            shape_x = min(text_width / 72 / 2 * 1.5, 0.2)  # Limit width to 1.5 times text width or 0.2
            shape_y = text_height / 72 / 2 * 1.2  # Keep height scaling as before

            # --- Node Placement Logic with Decision Node on the Right ---
            if object_type == 'Decision':
                # Place decision node on the right
                x = 0.9  # Adjust this value to control the horizontal position

                # Ensure no overlap with existing nodes
                while True:
                    y = random.uniform(0, 1)  # Generate random y-coordinate
                    overlap = False
                    for existing_node, (existing_x, existing_y) in node_positions.items():
                        distance = ((x - existing_x)**2 + (y - existing_y)**2)**0.5
                        if distance < (shape_x + shape_y + 0.1):
                            overlap = True
                            break
                    if not overlap:
                        node_positions[title] = (x, y)
                        break
            else:
                # Place other nodes randomly with overlap check
                while True:
                    x, y = random.uniform(0, 1), random.uniform(0, 1)
                    overlap = False
                    for existing_node, (existing_x, existing_y) in node_positions.items():
                        distance = ((x - existing_x)**2 + (y - existing_y)**2)**0.5
                        if distance < (shape_x + shape_y + 0.1):
                            overlap = True
                            break
                    if not overlap:
                        node_positions[title] = (x, y)
                        break

            # Shape for node
            shape = {
                'Decision': 'rect',
                'Uncertainty': 'ellipse',
                'Outcome': 'diamond'
            }[object_type]

            # Define shape path based on shape type
            if shape == 'rect':
                path = f"M {x - shape_x}, {y - shape_y} L {x + shape_x}, {y - shape_y} L {x + shape_x}, {y + shape_y} L {x - shape_x}, {y + shape_y} Z"
                shapes.append({
                    'type': 'path',
                    'path': path,
                    'fillcolor': '#FFA07A',  # Light red fill color
                    'line': {'color': 'darkblue'},
                    'xref': 'x',
                    'yref': 'y'
                })
            elif shape == 'ellipse':
                shapes.append({
                    'type': 'circle',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': x - shape_x,
                    'y0': y - shape_y,
                    'x1': x + shape_x,
                    'y1': y + shape_y,
                    'fillcolor': '#FFA07A',  # Light red fill color
                    'line': {'color': 'darkblue'}
                })
            else:  # diamond
                path = f"M {x}, {y - shape_y} L {x + shape_x}, {y} L {x}, {y + shape_y} L {x - shape_x}, {y} Z"
                shapes.append({
                    'type': 'path',
                    'path': path,
                    'fillcolor': '#FFA07A',  # Light red fill color
                    'line': {'color': 'darkblue'},
                    'xref': 'x',
                    'yref': 'y'
                })

            # Annotations for text (with bold font)
            annotations.append({
                'x': x,
                'y': y,
                'xref': 'x',
                'yref': 'y',
                'text': title,
                'showarrow': False,
                'xanchor': 'center',
                'yanchor': 'middle',
                'font': dict(family='DejaVu Sans', size=12, color='black', weight='bold')  # Bold font
            })

    # 2. Now process edges
    for line in lines:
        match = re.match(r'(Decision|Uncertainty|Outcome)\s+"([^"]+)"\s*(?:connects to:\s*([^,]+))?\s*(?:connects from:\s*([^,]+))?', line)
        if match:
            groups = match.groups()
            object_type = groups[0]  # Get the object type (Decision, Uncertainty, Outcome)
            title = groups[1]
            connects_to = groups[2] if groups[2] else None
            connects_from = groups[3] if groups[3] else None

            if connects_to:
                for to_node in connects_to.strip().split(','):
                    to_node = to_node.strip()
                    x, y = node_positions[title]
                    to_x, to_y = node_positions[to_node]

                    # Adjust edge endpoint if it connects to a decision node
                    if to_node in node_positions and node_positions[to_node][0] == 0.9:  # Check if target node is a Decision node (x-coordinate 0.9)
                         to_x = node_positions[to_node][0] - shape_x # Adjust to_x to the left edge of the decision node

                    edges_x.extend([x, to_x, None])
                    edges_y.extend([y, to_y, None])

            if connects_from:
                for from_node in connects_from.strip().split(','):
                    from_node = from_node.strip()
                    from_x, from_y = node_positions[from_node]
                    x, y = node_positions[title]
                    edges_x.extend([from_x, x, None])
                    edges_y.extend([from_y, y, None])

    # 3. Plotly Trace for Edges
    edge_trace = go.Scatter(
        x=edges_x,
        y=edges_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # 4. Plotly layout
    layout = go.Layout(
        title="Influence Diagram",
        titlefont_size=16,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        shapes=shapes,
        annotations=annotations,
    )

    fig = go.Figure(data=[edge_trace], layout=layout)
    return fig

def play_crq():
    st.title("AI Assisted Rapid Risk Assessment")

    a = "AI Will attempt to answer every question on this page. "
    b = "The results will require your quick validation. "
    c = "You can also upload your most recent Business Impact Assessment (BIA) "
    d = "As well as a recent cyber insurance questionnaire provided by your broker."

    crq_msg = a + b + c + d
    markdown_box("Rapid Risk Assessment Usage", crq_msg)

    st.write("")
    st.write("")
        
    # Row 1
    col1, col2 = st.columns([4, 1])  # Adjust column ratios as needed
    with col1:
      st.markdown("######")
      input_1 = st.text_input("Input 1", key="input_1")
    with col2:
      st.markdown("###")
      if st.button("Process 1", key="button_1"):
        handle_input("input_1", input_1)

    # Row 2
    col1, col2 = st.columns([4, 1])  # Adjust column ratios as needed
    with col1:
      input_2 = st.text_input("Input 2", key="input_2")
    with col2:
      if st.button("Process 2", key="button_2"):
        handle_input("input_2", input_2)

    # Row 3
    col1, col2 = st.columns([4, 1])  # Adjust column ratios as needed
    with col1:
      input_3 = st.text_input("Input 3", key="input_3")
    with col2:
      if st.button("Process 3", key="button_3"):
        handle_input("input_3", input_3)
        
    st.write("")

    company_revenue = st.number_input("Enter company revenue:", value=0.0)
    market_capitalization = st.number_input("Enter market capitalization:", value=0.0)
    st.write("Please enter values if not auto populated by AI")

    st.divider()
        

# Main function to handle different states
def main():
    st.sidebar.title("Navigation")

    choice = st.sidebar.radio("Go to", ["Sign Up", "Login", "Probability Words", "Forecasting", "Burndown", "Play Pool", "CRQ"])

    if choice == "Sign Up":
        signup()
    elif choice == "Login":
        login()
    elif choice == "Probability Words" and 'role' in st.session_state:
        probability_words()
    elif choice == "Forecasting" and 'role' in st.session_state:
        forecast_elephant()
    elif choice == "Burndown" and 'role' in st.session_state:
        play_burndown()
    elif choice == "Play Pool" and 'role' in st.session_state:
        random_number_game_with_brier_score()
    elif choice == "CRQ":
        play_crq()
    elif choice == "Influence" and 'role' in st.session_state:
        st.title("Simple Influence Diagram")

        a = "Decision \"Choose Strategy\" connects to: Market Share, Profit\n"
        b = "Uncertainty \"Market Conditions\" connects to: Market Share\n"
        c = "Outcome \"Market Share\" connects from: Choose Strategy, Market Conditions connects to: Profit\n"
        d = "Outcome \"Profit\" connects from: Market Share, Choose Strategy"

        inf_msg = a + b + c + d
        inf_msg = inf_msg.replace("\n", "<br>")

        # Render the text using markdown with unsafe_allow_html=True
        st.markdown(inf_msg, unsafe_allow_html=True)
        #markdown_box("EXAMPLE INFLUENCE DIAGRAM", inf_msg)

        text_input = st.text_area("Enter your text description here:")

        if st.button("Generate Diagram"):
            if text_input:
                fig = create_influence_diagram_from_text(text_input)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter a text description.")
    else:
        st.warning("Please log in to access the dashboard.")

# Initialize database on first run
if __name__ == '__main__':
    init_db()
    main()
