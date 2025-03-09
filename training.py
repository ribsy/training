
import sqlite3
import hashlib
import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta

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
            score REAL NOT NULL
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

def write_elephant_values(username, low_cow, high_cow, low_elph, high_elph, score):
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO forecasts (username, low_cow, high_cow, low_elph, high_elph, score) VALUES (?, ?, ?, ?, ?, ?)",
                   (username, low_cow, high_cow, low_elph, high_elph, score))
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
    st.title("FORECASTING ELEPHANTS")

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
            This is your 80% prediction range.
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
            <B>USING COWS TO PREDICT THE WEIGHT OF AN ELEPHANTS</B><br>
            How many cows fit into an elephant? Use your imagination again. You will use the average cow weight you predictive above to do this.
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
            st.write(f"Lower Bound Elephant Weight: {st.session_state['low_elph_lbs']} lbs")
            st.write(f"Upper Bound Elephant Weight: {st.session_state['high_elph_lbs']} lbs")
            score = modified_brier_score(st.session_state['low_elph_lbs'], st.session_state['high_elph_lbs'], 4000, 14000)
            score = round(score,3)
            st.session_state['score'] = score
            #st.write(f"Modified Brier-like Score: {score}")

    if st.button("Submit Elephant Forecast"):
         st.write(f"Forecasting Score: {st.session_state['score']}")
         write_elephant_values(st.session_state['username'], st.session_state['low_cow_lbs'], st.session_state['high_cow_lbs'],
                               st.session_state['low_elph_lbs'], st.session_state['high_elph_lbs'],
                               st.session_state['score'])
         if st.success("Elephant Forecast Saved!"):
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
                The lower your score is the better. A score of 0 means you are exactly right in term of the elephants weight range.
                The futher off you are from that range, the higher the score is.  A score below 100 is incredibly good.
                And anything below 1,000 is not bad.
            </div>
          """,
            unsafe_allow_html=True,
        )

        st.write("\n")
        if st.button("View All Elephant Forecasts and Scores"):
            view_forecasts()


#Modified Brier Score For Elephant Forecasts
def modified_brier_score(forecast_lower, forecast_upper, actual_lower, actual_upper):
    # Calculate the squared differences between the forecast and actual range boundaries
    # Normalize like Brier
    lower_diff = (forecast_lower - actual_lower) ** 2/(forecast_upper - forecast_lower)
    upper_diff = (forecast_upper - actual_upper) ** 2/(forecast_upper - forecast_lower)

    # Calculate the average of the squared differences (Brier-like score)
    score = (lower_diff + upper_diff) / 2
    return score

# Display forecast data
def view_forecasts():
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, low_cow, high_cow, low_elph, high_elph, score FROM forecasts")
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

def play_burndown():
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
        mone_open = st.number_input("", value=0, key = "mone_open")
    with col4:
        mtwo_open = st.number_input("", value=0, step=1, key = "mtwo_open")
    with col5:
        mthree_open = st.number_input("", value=0, step=1, key = "mthree_open")
    with col6:
        mfour_open = st.number_input("", value=0, step=1, key = "mfour_open")
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
        mone_fixed = st.number_input("", value=0, step=1, key = "mone_fixed")
    with col11:
        mtwo_fixed = st.number_input("", value=0, step=1, key = "mtwo_fixed")
    with col12:
        mthree_fixed = st.number_input("", value=0, step=1, key = "mthree_fixed")
    with col13:
        mfour_fixed = st.number_input("", value=0, step=1, key = "mfour_fixed")
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

    if st.button("Calculate Burndown Rate"):
         st.session_state['show_graph'] = True
    else:
        if st.session_state['total_open'] == 0:
            st.session_state['show_graph'] = True

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

        # Plot the beta distribution
        fig, ax = plt.subplots()
        #ax.plot(x, pdf, 'r-', lw=2, label=f'Beta({alpha:.2f}, {beta_val:.2f})')
        ax.plot(x, pdf, 'r-', lw=2)

        # Plot the HDI
        ax.axvspan(hdi[0], hdi[1], color='gray', alpha=0.3, label='Confidence')

        # Plot the vertical line
        ax.axvline(st.session_state['sla'], color='black', linestyle='--', label=f'SLA = {round(st.session_state["sla"],3)}')

        ax.set_xlabel('RATE (Probability)')
        ax.set_ylabel('Strength (Density)')
        ax.set_title('Burndown Rate With Uncertainty')
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

def random_number_generator():
  """Generates a random number between 1 and 40 (non-inclusive)."""
  return random.randint(1, 43)

def track_random_numbers(num_requests=1):
  """Generates an initial random number and compares subsequent numbers.

  Args:
    num_requests: The number of additional random numbers to generate.

  Returns:
    A list containing the initial value and "right" or "left" for each subsequent number.
  """
  initial_value = random_number_generator()
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

  st.title("Playing Pool With The Rev Bayes")
  st.image("./data/rev_bayes_pool.png")

  if "initial_value" not in st.session_state:
    st.session_state.initial_value = None
  if "results" not in st.session_state:
    st.session_state.results = []
  if "scores" not in st.session_state:
    st.session_state.scores = []
    
  # Put buttons on one line using st.columns
  col1, col2, col3 = st.columns(3)

  with col1:
      if st.button("Play A New Game"):
          st.session_state.initial_value = random_number_generator()
          st.session_state.results = [st.session_state.initial_value]
          st.session_state.scores = []  # Clear scores when starting a new game
          st.write(f"Initial value: {st.session_state.initial_value}")

  with col2:
      if st.session_state.initial_value is not None:
          if st.button("Roll Another Ball"):
              result = track_random_numbers(1)[1]  # Get "right" or "left" from the function
              st.session_state.results.append(result)
              st.write(f"Result: {result}")

  with col3:
      if st.session_state.initial_value is not None:
          if st.button("Make A Bet"):
              # ... (your existing logic for "Make A Bet" button) ...
              forecast_lower = st.number_input("Forecast Lower", value=0)
              forecast_higher = st.number_input("Forecast Higher", value=0)
              if st.session_state.initial_value is not None:
                  score = range_scoring_function(forecast_lower, forecast_higher, st.session_state.initial_value)
                  st.session_state.scores.append(score)  # Append score to the list
                  st.write(f"Modified Brier Score: {score}")
              else:
                  st.write(f"You must Play A New Game before you can Make A Bet.")


  st.write("Previous Results:", st.session_state.results)

  # Display all previous scores
  st.write("Previous Scores:")
  for i, score in enumerate(st.session_state.scores):
      st.write(f"Bet {i + 1}: {score}")

# Main function to handle different states
def main():
    st.sidebar.title("Navigation")

    choice = st.sidebar.radio("Go to", ["Sign Up", "Login", "Probability Words", "Forecasting", "Burndown","Play Pool"])

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
    else:
        st.warning("Please log in to access the dashboard.")

# Initialize database on first run
if __name__ == '__main__':
    init_db()
    main()
