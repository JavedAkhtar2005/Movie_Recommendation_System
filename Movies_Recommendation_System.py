import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Page Configuration

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon=" ",
    layout="centered"
)



# Title

st.title("Movie Recommendation System")

st.write(
    "Find movies and shows similar to your favorite Netflix title."
)

# Load Dataset

@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "netflix_data.csv"

    data = pd.read_csv(file_path)

    return data


data = load_data()

# Prepare Data

required_columns = [
    "title",
    "director",
    "cast",
    "country",
    "listed_in",
    "description"
]

for column in required_columns:
    data[column] = data[column].fillna("")


data["combined_features"] = (
    data["director"] + " " +
    data["cast"] + " " +
    data["country"] + " " +
    data["listed_in"] + " " +
    data["description"]
)

# Create Recommendation Model

@st.cache_resource
def create_model(features):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    matrix = vectorizer.fit_transform(features)

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=6
    )

    model.fit(matrix)

    return vectorizer, matrix, model


vectorizer, matrix, model = create_model(
    data["combined_features"]
)

# Movie Selection

movie_list = sorted(
    data["title"].dropna().unique()
)


selected_movie = st.selectbox(
    "Select a Movie or Show",
    movie_list
)
# Recommendation

if st.button("Recommend Movies"):

    movie_index = data[
        data["title"] == selected_movie
    ].index[0]

    movie_vector = matrix[movie_index]

    distances, indices = model.kneighbors(
        movie_vector,
        n_neighbors=6
    )

    st.subheader(
        f"Movies similar to {selected_movie}"
    )

    count = 0

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        if index == movie_index:
            continue

        movie_title = data.iloc[index]["title"]

        similarity = 1 - distance

        st.success(
            f" {movie_title}"
        )

        st.write(
            f"Similarity Score: {similarity:.2f}"
        )

        if "type" in data.columns:

            st.write(
                f"Type: {data.iloc[index]['type']}"
            )

        count += 1

        if count == 5:
            break

# Sidebar

st.sidebar.title("About")

st.sidebar.write(
    "This Movie Recommendation System "
    "uses TF-IDF and Cosine Similarity "
    "to recommend similar Netflix movies "
    "and shows."
)

st.sidebar.write(
    f"Total Titles: {len(data)}"
)