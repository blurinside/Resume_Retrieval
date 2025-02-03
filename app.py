import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os

# Streamlit App Interface
st.title("Resume Query Retrieval System")
st.write("A query-based resume information retrieval application using Meta Llama 3.1 8B.")

# Define the path to your model and weights
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"  # Adjust based on Hugging Face model name
MODEL_PATH = r"C:\Users\aryan\Downloads\model.pth"  # Adjust this path if necessary

# Load Meta Llama model and tokenizer
try:
    # Initialize the tokenizer and model architecture
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Load the pre-trained weights from your local model.pth
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    st.success("Meta Llama 3.1 8B model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the model: {e}")

# Upload resume file
uploaded_resume = st.file_uploader("Upload Resume Document", type=["txt", "pdf", "docx"])

if uploaded_resume is not None:
    # Read and process resume content
    if uploaded_resume.type == "application/pdf":
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_resume)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
    elif uploaded_resume.type == "text/plain":
        resume_text = uploaded_resume.read().decode("utf-8")
    elif uploaded_resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        doc = Document(uploaded_resume)
        resume_text = "\n".join([para.text for para in doc.paragraphs])
    else:
        st.error("Unsupported file type.")

    st.write("Resume Uploaded Successfully!")

    # Initialize embeddings and text splitter
    embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"  # Adjust as needed
    embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model)
    text_splitter = CharacterTextSplitter(chunk_size=512)

    # Split resume into chunks for processing
    resume_chunks = text_splitter.split_text(resume_text)
    faiss_index = FAISS.from_texts(resume_chunks, embeddings)

    # User query input
    query = st.text_input("Enter a query to search within the resume:")

    if query:
        # Perform similarity search on the resume
        responses = faiss_index.similarity_search(query, k=3)  # Adjust 'k' as needed
        st.write("Results:")
        for idx, result in enumerate(responses):
            st.write(f"{idx + 1}. {result.page_content}")

        # Generate insights using the loaded model
        if st.button("Generate Insights"):
            inputs = tokenizer(query, return_tensors='pt')
            with torch.no_grad():
                output = model.generate(**inputs, max_length=100)
            generated_insights = tokenizer.decode(output[0], skip_special_tokens=True)
            st.write("Generated Insights:")
            st.write(generated_insights)