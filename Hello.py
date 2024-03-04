import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import base64

# Function to perform matching and generate similarity scores
def perform_matching(origin_df, destination_df, selected_columns):
    # Combine the selected columns into a single text column for vectorization
    origin_df['combined_text'] = origin_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
    destination_df['combined_text'] = destination_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Use a pre-trained model for embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Vectorize the combined text
    origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
    destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

    # Create a FAISS index
    dimension = origin_embeddings.shape[1]  # The dimension of vectors
    faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    faiss_index.add(destination_embeddings.astype('float32'))  # Add destination vectors to the index

    # Perform the search for the nearest neighbors
    D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)  # k=1 finds the closest match

    # Calculate similarity score (1 - normalized distance)
    similarity_scores = 1 - (D / np.max(D))

    # Create the output DataFrame with similarity score instead of distance
    matches_df = pd.DataFrame({
        'origin_url': origin_df['Address'],
        'matched_url': destination_df['Address'].iloc[I.flatten()].values,
        'similarity_score': np.round(similarity_scores.flatten(), 4)  # Rounded for better readability
    })

    return matches_df

def main():
    st.title('URL Redirect Similarity Matching App')

    # Sidebar with step-by-step instructions
    st.sidebar.title("URL Redirect Mapping Instructions")
    st.sidebar.markdown("""
    **Table of Contents:**
    - [Step 1: Crawl your live website with Screaming Frog](#step-1-crawl-your-live-website-with-screaming-frog)
    - [Step 2: Export HTML pages with 200 Status Code](#step-2-export-html-pages-with-200-status-code)
    - [Step 3: Repeat steps 1 and 2 for your staging website](#step-3-repeat-steps-1-and-2-for-your-staging-website)
    - [Optional: Find and replace your staging site domain](#optional-find-and-replace-your-staging-site-domain)
    """)

    # Detailed directions below the tool
    st.header("Step-by-Step Instructions")
    
    # Step 1: Crawl your live website with Screaming Frog
    st.subheader("Step 1: Crawl your live website with Screaming Frog")
    st.markdown("""
    You’ll need to perform a standard crawl on your website. Depending on how your website is built, this may or may not require a JavaScript crawl. The goal is to produce a list of as many accessible pages on your site as possible.
    """)
    # (Add more detailed instructions as needed)

    # Step 2: Export HTML pages with 200 Status Code
    st.subheader("Step 2: Export HTML pages with 200 Status Code")
    st.markdown("""
    Once the crawl has been completed, we want to export all of the found HTML URLs with a 200 Status Code. This will provide you with a list of our current live URLs and all of the default metadata Screaming Frog collects about them, such as Titles and Header Tags. Save this file as origin.csv.
    """)
    # (Add more detailed instructions as needed)

    # Step 3: Repeat steps 1 and 2 for your staging website
    st.subheader("Step 3: Repeat steps 1 and 2 for your staging website")
    st.markdown("""
    We now need to gather the same data from our staging website, so we have something to compare to. Depending on how your staging site is secured, you may need to use features such as Screaming Frog’s forms authentication if password protected. Once the crawl has completed, you should export the data and save this file as destination.csv.
    """)
    # (Add more detailed instructions as needed)

    # Optional: Find and replace your staging site domain
    st.subheader("Optional: Find and replace your staging site domain")
    st.markdown("""
    It’s likely your staging website is either on a different subdomain, TLD or even domain that won’t match our actual destination URL. For this reason, I will use a Find and Replace function on my destination.csv to change the path to match the final live site subdomain, domain or TLD.
    """)
    # (Add more detailed instructions as needed)

    st.header("URL Redirect Similarity Matching Tool")
    st.markdown("""
    This app performs similarity matching between two sets of URLs for the purpose of URL redirection mapping. 
    Please follow the instructions provided on the sidebar to prepare your data for matching.
    """)

    origin_file = st.file_uploader("Upload origin.csv", type=['csv'], help="Please upload the CSV file containing the origin URLs")
    destination_file = st.file_uploader("Upload destination.csv", type=['csv'], help="Please upload the CSV file containing the destination URLs")

    if origin_file is not None and destination_file is not None:
        origin_df = pd.read_csv(origin_file)
        destination_df = pd.read_csv(destination_file)

        common_columns = list(set(origin_df.columns) & set(destination_df.columns))

        selected_columns = st.multiselect('Select the columns you want to include for similarity matching:', common_columns)

        if not selected_columns:
            st.warning("Please select at least one column to continue.")
        else:
            if st.button("Let's Go!"):
                matches_df = perform_matching(origin_df, destination_df, selected_columns)
                
                # Display matches in the app
                st.write(matches_df)

                # Save matches to a CSV file and provide download link
                csv = matches_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}">'
