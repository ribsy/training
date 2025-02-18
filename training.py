import sqlite3
import hashlib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    #CREATE PROBS TABLE
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

def prob_chart(prob_val, name_val):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.hist(prob_val, bins=100, edgecolor='black')  # Adjust bins as needed
    ax.set_xlim(0, 100)
    ax.set_xlabel(name_val)
    ax.set_ylabel(name_val)
    ax.set_title('Histogram of Probability Words')
    st.pyplot(fig)


# Member dashboard for regular users
def probability_words():

    if 'username' in st.session_state:
        st.title("Probability Words")
        st.write(f"What do these words mean as probabilities, {st.session_state['username']}?")
    else:
        st.warning("Please log in to access the dashboard.")

    # Create four sliders
    slider1 = st.slider("Unlikely", 0, 100, 0)  # Default value is 0
    slider2 = st.slider("Probable", 0, 100, 0)  # Default value is 0
    slider3 = st.slider("Likely", 0, 100, 0)  # Default value is 0
    slider4 = st.slider("Highly Likely", 0, 100, 0)  # Default value is 0


    if 'show_group_results' not in st.session_state:
        st.session_state['show_group_results'] = False
        st.session_state['rerun_group_results'] = False

    # Write slider values to the database when a button is clicked
    if st.button("Save Slider Values"):
        write_slider_values(slider1, slider2, slider3, slider4)
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
            How many 200 lbs males fit into a cow? Use you imagination. Can you see how many fits? What is your upper bound? What is your lower bound?
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
  

# Display all user data for admin users
def view_user_data():
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM users")
    user_data = cursor.fetchall()
    conn.close()
    

# Main function to handle different states
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Sign Up", "Login", "Probability Words", "Forecasting"])

    if choice == "Sign Up":
        signup()
    elif choice == "Login":
        login()
    elif choice == "Probability Words":
        probability_words()
    elif choice == "Forecasting":
        forecast_elephant()
    else:
        st.warning("Please log in to access the dashboard.")

# Initialize database on first run
if __name__ == '__main__':
    init_db()
    main()
