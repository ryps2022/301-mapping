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

    st.markdown("""
    This app performs similarity matching between two sets of URLs for the purpose of URL redirection mapping. 
    Please follow these instructions:
    - **Upload Instructions**:
        - The first CSV upload must be named **"origin.csv"**. This file should contain the URLs you want to redirect.
        - The second CSV upload must be named **"destination.csv"**. This file contains the destination URLs where the origin URLs will be redirected to.
    - **Matching Results**:
        - The vector score represents the similarity between the origin and matched URLs. A higher score indicates a closer match.
        - It is recommended to perform quality assurance (QA) on the matching results before implementing the redirection.
    - **Output**:
        - After matching is complete, a downloadable CSV file with the matching results will be provided.
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
                href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()