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

# Helper: Query Pinecone for relevant chunks and generate an answer
def query_vectors(query):
    vector = openai.Embedding.create(
        input=[query],
        model="text-embedding-3-small"
    ).data[0].embedding

    results = index.query(
        vector=vector,
        top_k=5,
        include_metadata=True
        # Removed filter to search all PDFs
    )

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)

        prompt = (
            f"Refer to the following excerpts from the Sports Authority of India documents and answer the question:\n\n"
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
        return "Sorry, no relevant information was found in the documents."

# ---------------- UI ----------------
st.set_page_config(page_title="SAI ChatBot", page_icon="🤖")
st.title("🏛️ Sports Authority of India - Q&A ChatBot")

query = st.text_input("Ask a question about the Sports Authority of India documents:")

if st.button("Get Answer"):
    if query:
        answer = query_vectors(query)
        st.markdown("### 📋 Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
