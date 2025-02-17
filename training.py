import sqlite3
import hashlib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
    ax.hist(prob_val, bins=10, edgecolor='black')  # Adjust bins as needed
    #ax.set_xlabel(name_val)
    ax.set_ylabel(name_val)
    #ax.set_title('Histogram of Probability Words')
    st.pyplot(fig)

# Member dashboard for regular users
def member_dashboard():
    st.title("Probability Words")
    st.write(f"What do these words mean as probabilities, {st.session_state['username']}?")

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
        if st.button("Show Group Results"):
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


# Admin dashboard with additional features
def admin_dashboard():
    if st.session_state.get('role') == 'admin':
        st.title("Admin Dashboard")
        view_user_data()
    else:
        st.error("Access Denied: Admins only.")

# Display all user data for admin users
def view_user_data():
    conn = sqlite3.connect('train.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM users")
    user_data = cursor.fetchall()
    conn.close()
    st.dataframe(user_data)

# Main function to handle different states
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Sign Up", "Login", "Probability Words"])

    if choice == "Sign Up":
        signup()
    elif choice == "Login":
        login()
    elif choice == "Probability Words":
        if 'username' in st.session_state:
            if st.session_state['role'] == 'admin':
                admin_dashboard()
            else:
                member_dashboard()
        else:
            st.warning("Please log in to access the dashboard.")

# Initialize database on first run
if __name__ == '__main__':
    init_db()
    main()
