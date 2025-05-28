import streamlit as st
import pinecone
import openai
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "sai"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Helper: List all unique PDF names in index
def list_stored_pdfs():
    stats = index.describe_index_stats()
    pdf_names = set()
    for ns in stats.get("namespaces", {}):
        # Can't get individual metadata without fetching matches
        # Assume a consistent list or preload known names
        pass
    # Alternative: hardcode or preload names
    return ["1740030030_67b6c04e6fb4f_agenda.pdf"]

# Helper: Query Pinecone for relevant chunks and generate an answer
def query_vectors(query, selected_pdf):
    vector = openai.Embedding.create(
        input=[query],
        model="text-embedding-3-small"
    ).data[0].embedding

    results = index.query(
        vector=vector,
        top_k=5,
        include_metadata=True,
        filter={"pdf_name": {"$eq": selected_pdf}}
    )

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)

        prompt = (
            f"Refer to the document '{selected_pdf}' and answer the question based on these excerpts:\n\n"
            f"{combined_text}\n\n"
            f"Question: {query}"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides helpful and accurate answers using official documents from the Sports Authority of India."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    else:
        return "Sorry, no relevant information was found in the selected document."

# ---------------- UI ----------------
st.set_page_config(page_title="SAI ChatBot", page_icon="🤖")
st.title("🏛️ Sports Authority of India - Q&A ChatBot")

pdf_list = list_stored_pdfs()
selected_pdf = st.selectbox("Choose a PDF document:", pdf_list)

query = st.text_input("Ask a question about the document:")

if st.button("Get Answer"):
    if selected_pdf and query:
        answer = query_vectors(query, selected_pdf)
        st.markdown("### 📋 Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question and select a document.")
