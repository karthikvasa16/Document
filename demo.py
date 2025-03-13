import streamlit as st
import pandas as pd
import numpy as np
import nltk
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import seaborn as sns
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup

# --- Configuration ---
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon=":briefcase:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown(
    """
    <style>
    /* Overall Theme */
    body {
        background-color: #f0f8ff; /* Light blue background color for the entire website */
        color: #333333;
        font-family: sans-serif;
    }
    /* Title */
    .title {
        color: #007bff;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 1em;
    }
    /* Header */
    .header {
        color: #28a745;
        font-size: 2em;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    /* Subheader */
    .subheader {
        color: #17a2b8;
        font-size: 1.5em;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    /* Text */
    .text {
        font-size: 1.1em;
        line-height: 1.6;
    }
    /* Sidebar */
    [data-testid="stSidebarNav"] {
        background-color: #343a40;
        color: #ffffff;
    }
    [data-testid="stSidebarNav"]::before {
        content: "Job Finder Pro";
        margin-left: 20px;
        margin-top: 20px;
        font-size: 30px;
        position: relative;
        font-weight: bold;
        color: #ffc107;
    }
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarForm"] {
        border-top: 1px solid #ffffff;
    }
    /* Dataframe */
    .dataframe {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 1em;
    }
    /* Warning */
    .warning {
        color: #dc3545;
        font-size: 1.2em;
    }
    /* Job Details */
    .detail-container {
        border-radius: 7px;
        padding: 0.75em;
        margin-bottom: 0.5em;
    }

    .detail-label {
        font-weight: bold;
        color: #333;
    }

    .detail-value {
        color: #555;
    }

    /* Search Input Label */
    .stTextInput > label {
        color: black !important;
        padding: 4px;
        border-radius: 4px;
        background-color: transparent !important; /* Make background transparent */
    }
    /* Search Input Box */
    .stTextInput > div > div > input {
        border: 2px solid black; /* Changed border color to black */
        border-radius: 5px;
        padding: 0.5em;
    }

        /* Expander Styling */
    details {
        margin-bottom: 0.5em;
        border: 1px solid #ADD8E6; /* Light blue border */
        border-radius: 8px; /* More rounded corners */
        padding: 0.6em;
        background-color: #e0f2f7; /* Softer background */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08); /* Subtler shadow */
        transition: all 0.3s ease; /* Smooth transition for hover effects */
    }

    details:hover {
        border-color: #77b5fe; /* Slightly darker blue border on hover */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12); /* Increased shadow on hover */
    }

    details > summary {
        list-style: none;
        cursor: pointer;
        padding: 0.5em;
        font-weight: 500; /* Semi-bold */
        color: #444; /* Slightly darker text color */
    }

    details > summary::marker {
      display: none; /* remove the triangle */
    }

    details[open] {
        background-color: #fff;
        border-color: #77b5fe; /* keep the slightly darker blue border when open */
        box-shadow: 0 3px 7px rgba(0, 0, 0, 0.15); /* Slight shadow when expanded */
    }

    details[open] > summary {
        border-bottom: 1px solid #bbb;
    }

    details .detail-value {
        padding: 0.5em;
        color: #555; /* Adjusted value color */
    }

    /* Ensure the expander arrow is visible */
    details > summary::before {
        content: '▶';
        font-size: 0.8em;
        color: #77b5fe; /* use the highlight color for the arrow */
        margin-right: 0.3em;
        transition: transform 0.3s ease-in-out;
        display: inline-block; /* this is important for vertical alignment */
    }
    details[open] > summary::before {
        transform: rotate(90deg);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Suppress warnings
warnings.simplefilter('ignore')

# Download necessary NLTK resources
try:
    nltk.data.find('wordnet')
except LookupError:
    try:
        nltk.download('wordnet')
    except Exception as e:
        st.error(f"Failed to download 'wordnet': {e}")

try:
    nltk.data.find('omw-1.4')
except LookupError:
    try:
        nltk.download('omw-1.4')
    except Exception as e:
        st.error(f"Failed to download 'omw-1.4': {e}")

# File paths
apps_file_path = "Dataset/apps.tsv"
user_history_file_path = "Dataset/user_history.tsv"
jobs_file_path = "Dataset/jobs.csv"
users_file_path = "Dataset/users.tsv"
test_users_file_path = "Dataset/test_users.tsv"


def remove_html_tags(text):
    """Removes HTML tags from a string."""
    if not isinstance(text, str):
        return text
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def clean_text(text):
    """Removes HTML tags and special characters like \r and \n from a string."""
    if not isinstance(text, str):
        return text
    text = remove_html_tags(text)  # Remove HTML tags first
    text = text.replace('\r', '').replace('\n', ' ')  # Remove \r and \n, replace with space
    return text

@st.cache_data
def load_data():
    """Loads the datasets."""
    try:
        apps = pd.read_csv(apps_file_path, delimiter='\t', encoding='utf-8')
        user_history = pd.read_csv(user_history_file_path, delimiter='\t', encoding='utf-8')
        jobs = pd.read_csv(jobs_file_path, delimiter='\t', encoding='utf-8', on_bad_lines='skip')
        users = pd.read_csv(users_file_path, delimiter='\t', encoding='utf-8')
        test_users = pd.read_csv(test_users_file_path, delimiter='\t', encoding='utf-8')

        # Apply cleaning to 'Description' and 'Requirements' columns in 'jobs'
        if jobs is not None and 'Description' in jobs.columns and 'Requirements' in jobs.columns:
            jobs['Description'] = jobs['Description'].astype(str).apply(clean_text)
            jobs['Requirements'] = jobs['Requirements'].astype(str).apply(clean_text)

        print("All datasets loaded successfully.")
        return apps, user_history, jobs, users, test_users
    except FileNotFoundError as e:
        st.error(f"Error loading dataset: {e}.  Please make sure the data files are in the correct location.")
        return None, None, None, None, None

@st.cache_data
def load_jobs_data(file_path):
    """Loads the jobs dataset."""
    try:
        jobs = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', on_bad_lines='skip')
        # Basic data cleaning (handle missing values by filling with empty string)
        jobs = jobs.fillna('')
        # Apply cleaning to 'Description' and 'Requirements' columns
        jobs['Description'] = jobs['Description'].astype(str).apply(clean_text)
        jobs['Requirements'] = jobs['Requirements'].astype(str).apply(clean_text)

        # Combine title, description, and requirements for content-based filtering
        jobs['combined_text'] = jobs['Title'] + ' ' + jobs['Description'] + ' ' + jobs['Requirements']
        return jobs
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None

def create_tfidf_matrix(jobs):
    """Creates a TF-IDF matrix from the combined text data."""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(jobs['combined_text'])
    return tfidf_matrix, tfidf_vectorizer

def get_job_recommendations(job_title, jobs, tfidf_matrix, tfidf_vectorizer, top_n=5):
    """
    Recommends similar jobs based on a given job title using TF-IDF.
    """
    try:
        # Find the index of the job
        idx = jobs[jobs['Title'].str.lower() == job_title.lower()].index[0]
    except IndexError:
        st.warning(f"Job '{job_title}' not found. Please check the title.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Calculate cosine similarities
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get the indices of the top N most similar jobs
    related_job_indices = cosine_similarities.argsort()[:-top_n-2:-1] #Avoid recommending the same job

    # Return the recommended jobs as a DataFrame
    recommended_jobs = jobs.iloc[related_job_indices]
    return recommended_jobs

# Load the data
apps, user_history, jobs, users, test_users = load_data()

# Load data for the recommendation system
jobs_recommendation = load_jobs_data(jobs_file_path)

if jobs_recommendation is not None:
    # Create TF-IDF matrix
    tfidf_matrix, tfidf_vectorizer = create_tfidf_matrix(jobs_recommendation)

# --- Sidebar Styling and Dataset Display ---
with st.sidebar:
    st.header("Datasets")

    # Dataset Selection with Icons
    selected_dataset = st.radio(
        "Choose a dataset:",
        options=[
            "Apps 🗂",
            "User History 📜",
            "Jobs 💼",
            "Users 👤",
            "Test Users 🧪"
        ],
        index=0,
    )

    # Display selected dataset
    if selected_dataset == "Apps 🗂":
        if apps is not None:
            st.subheader("Apps Dataset")
            st.dataframe(apps)
        else:
            st.warning("Apps dataset not loaded.")
    elif selected_dataset == "User History 📜":
        if user_history is not None:
            st.subheader("User History Dataset")
            st.dataframe(user_history)
        else:
            st.warning("User History dataset not loaded.")
    elif selected_dataset == "Jobs 💼":
        if jobs is not None:
            st.subheader("Jobs Dataset")
            st.dataframe(jobs)
        else:
            st.warning("Jobs dataset not loaded.")
    elif selected_dataset == "Users 👤":
        if users is not None:
            st.subheader("Users Dataset")
            st.dataframe(users)
        else:
            st.warning("Users dataset not loaded.")
    elif selected_dataset == "Test Users 🧪":
        if test_users is not None:
            st.subheader("Test Users Dataset")
            st.dataframe(test_users)
        else:
            st.warning("Test Users dataset not loaded.")

# --- Main Section ---
st.markdown("<h1 class='title'>Explore Career Opportunities 💼</h1>", unsafe_allow_html=True)

# Introduction Section
st.markdown("<h2 class='header'>Welcome to the Job Recommendation System</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='text'>
    Unleash your potential and discover exciting career paths with our intelligent job recommendation platform.
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3 class='subheader'>Key Features</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <ul class='text'>
            <li><b>Personalized Job Recommendations:</b> Get job suggestions tailored to your skills and experience.</li>
            <li><b>Data-Driven Insights:</b> Explore comprehensive job market analysis and trends.</li>
            <li><b>Easy Job Search:</b> Find detailed information about various job openings.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    # Job Search Functionality
    st.markdown("<h4 style='color: #e74c3c;'>Enter the job title to search for:</h4>", unsafe_allow_html=True)
    search_title = st.text_input(
        label="",
        value="",
        key="search_title",
        help="Enter a job title to find matching job details.",
        placeholder="e.g., Software Engineer",
        label_visibility="visible",
    )

    def find_job_details_by_title(title_to_search):
        """Finds and displays job details for jobs with the given title, and recommends similar jobs."""
        global jobs, jobs_recommendation, tfidf_matrix, tfidf_vectorizer  # Access the global DataFrames

        if jobs is None or jobs_recommendation is None or tfidf_matrix is None or tfidf_vectorizer is None:
            st.error("Error: 'jobs' DataFrame or recommendation engine components are not loaded.")
            return

        try:
            job = jobs[jobs['Title'].str.lower() == title_to_search.lower()].iloc[0]  # Selects the first matching row
        except IndexError:
            st.write(f"No jobs found with the title '{title_to_search}'.")
            return

        if not job.empty:
            # --- Job Details Display ---
            st.markdown("<h4 style='color: #e74c3c;'>Job Details</h4>", unsafe_allow_html=True)
            # --- Clean and format the text ---
            description = job['Description']
            requirements = job['Requirements']

            # -- Create Expander --
            st.markdown("""
            <style>
            details summary::before {
                content: '▶';
                font-size: 0.8em;
                color: #77b5fe; /* Highlight color for the arrow */
                margin-right: 0.3em;
                transition: transform 0.3s ease-in-out;
                display: inline-block;
                vertical-align: middle;
            }
            details[open] summary::before {
                transform: rotate(90deg);
            }
            </style>
            """, unsafe_allow_html=True)

            with st.expander(label="Description"):
                st.markdown(f"<div class='detail-container'><span class='detail-value'>{description}</span></div>",
                            unsafe_allow_html=True)

            with st.expander(label="Requirements"):
                st.markdown(f"<div class='detail-container'><span class='detail-value'>{requirements}</span></div>",
                            unsafe_allow_html=True)

            # City, State, Country, Zip5
            st.markdown(f"<div class='detail-container'><span class='detail-label'>City:</span>  <span class='detail-value'>{job['City']}</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='detail-container'><span class='detail-label'>State:</span>  <span class='detail-value'>{job['State']}</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='detail-container'><span class='detail-label'>Country:</span>  <span class='detail-value'>{job['Country']}</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='detail-container'><span class='detail-label'>Zip5:</span>  <span class='detail-value'>{job['Zip5']}</span></div>",
                        unsafe_allow_html=True)

            # StartDate and EndDate
            st.markdown(f"<div class='detail-container'><span class='detail-label'>Start Date:</span>  <span class='detail-value'>{job['StartDate']}</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='detail-container'><span class='detail-label'>End Date:</span>  <span class='detail-value'>{job['EndDate']}</span></div>",
                        unsafe_allow_html=True)

            # --- Job Recommendations ---
            st.markdown("<h4 style='color: #e74c3c;'>Recommended Jobs</h4>", unsafe_allow_html=True)
            recommendations = get_job_recommendations(title_to_search, jobs_recommendation, tfidf_matrix, tfidf_vectorizer)

            if not recommendations.empty:
                st.dataframe(recommendations[['Title', 'Description','City', 'State', 'Country']]) # Display relevant columns
            else:
                st.write("No recommendations found.")

        else:
            st.write(f"No jobs found with the title '{title_to_search}'.")

    if search_title:
        find_job_details_by_title(search_title)

with col2:
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(_file_))

    # Construct the absolute path to the image
    image_path = os.path.join(script_dir, "Images/job.png")

    try:
        st.image(image_path, width=500)
    except FileNotFoundError:
        st.error(f"Image not found at: {image_path}")
    except Exception as e:
        st.error(f"Error displaying image: {e}")

# Data Exploration and Visualization
if jobs is not None and users is not None:
    st.markdown("<h2 class='header'>Data Exploration and Visualization</h2>", unsafe_allow_html=True)

    st.markdown("<h3 class='subheader'>Job Data Analysis :bar_chart:</h3>", unsafe_allow_html=True)

    st.markdown("<h4 style='color: #e74c3c;'>Job Openings by Country</h4>", unsafe_allow_html=True)
    Country_wise_job = jobs.groupby(['Country']).size().reset_index(name='Locationwise').sort_values(
        'Locationwise', ascending=False)
    fig_country, ax_country = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Country", y="Locationwise", data=Country_wise_job, ax=ax_country, palette="viridis")
    ax_country.set_xticklabels(ax_country.get_xticklabels(), rotation=45, ha="right", color="#3498db")
    ax_country.set_title('Job Openings by Country', color="#3498db")
    plt.tight_layout()
    st.pyplot(fig_country)

    st.markdown("<h4 style='color: #e74c3c;'>Job Openings by State in the US</h4>", unsafe_allow_html=True)
    jobs_US = jobs.loc[jobs['Country'] == 'US']
    State_wise_job_US = jobs_US.groupby(['State']).size().reset_index(name='Locationwise').sort_values(
        'Locationwise', ascending=False)
    fig_state, ax_state = plt.subplots(figsize=(10, 5))
    sns.barplot(x="State", y="Locationwise", data=State_wise_job_US, ax=ax_state, palette="magma")
    ax_state.set_xticklabels(ax_state.get_xticklabels(), rotation=45, ha="right", color="#3498db")
    ax_state.set_title('Job Openings by State in the US', color="#3498db")
    plt.tight_layout()
    st.pyplot(fig_state)

    st.markdown("<h4 style='color: #e74c3c;'>Job Openings by City in the US</h4>", unsafe_allow_html=True)
    City_wise_location = jobs_US.groupby(['City']).size().reset_index(name='Locationwise').sort_values(
        'Locationwise', ascending=False)
    City_wise_location_th = City_wise_location.loc[City_wise_location['Locationwise'] >= 12]
    fig_city, ax_city = plt.subplots(figsize=(10, 5))
    sns.barplot(x="City", y="Locationwise", data=City_wise_location_th.head(50), ax=ax_city,
                palette="plasma")
    ax_city.set_xticklabels(ax_city.get_xticklabels(), rotation=45, ha="right", color="#3498db")
    ax_city.set_title('Job Openings by City in the US', color="#3498db")
    plt.tight_layout()
    st.pyplot(fig_city)

    st.markdown("<h2 class='header'>User Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Job Seekers by Country :earth_africa:</h3>", unsafe_allow_html=True)
    users_training = users.loc[users['Split'] == 'Train'] if users is not None else None
    Country_wise_user = users_training.groupby(['Country']).size().reset_index(name='Locationwise').sort_values(
        'Locationwise', ascending=False)

    st.dataframe(Country_wise_user.head(10).style.highlight_max(axis=0), height=300)

    st.markdown("<h3 class='subheader'>Job Seekers by State in the US :</h3>", unsafe_allow_html=True)
    user_training_US = users_training.loc[users_training['Country'] == 'US']
    user_training_US_state_wise = user_training_US.groupby(['State']).size().reset_index(
        name='Locationwise_state').sort_values('Locationwise_state', ascending=False)
    user_training_US_th = user_training_US_state_wise.loc[
        user_training_US_state_wise['Locationwise_state'] >= 12]

    fig_user_state, ax_user_state = plt.subplots(figsize=(10, 5))
    sns.barplot(x="State", y="Locationwise_state", data=user_training_US_th.head(50), ax=ax_user_state,
                palette="cividis")
    ax_user_state.set_xticklabels(ax_user_state.get_xticklabels(), rotation=45, ha="right", color="#3498db")
    ax_user_state.set_title('Job Seekers by State in the US', color="#3498db")
    plt.tight_layout()
    st.pyplot(fig_user_state)

    st.markdown("<h3 class='subheader'>Job Seekers by City in the US :</h3>", unsafe_allow_html=True)
    user_training_US_city_wise = user_training_US.groupby(['City']).size().reset_index(
        name='Locationwise_city').sort_values('Locationwise_city', ascending=False)
    user_training_US_City_th = user_training_US_city_wise.loc[
        user_training_US_city_wise['Locationwise_city'] >= 12]
    fig_user_city, ax_user_city = plt.subplots(figsize=(10, 5))
    sns.barplot(x="City", y="Locationwise_city", data=user_training_US_City_th.head(50), ax=ax_user_city,
                palette="Spectral")
    ax_user_city.set_xticklabels(ax_user_city.get_xticklabels(), rotation=45, ha="right", color="#3498db")
    ax_user_city.set_title('Job Seekers by City in the US', color="#3498db")
    plt.tight_layout()
    st.pyplot(fig_user_city)

else:
    st.markdown("<p class='warning'>Data loading failed. Please check the file paths.</p>", unsafe_allow_html=True)