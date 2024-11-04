import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Initialize the language model with your API key
llm = ChatGroq(
    model_name="mixtral-8x7b-32768", 
    api_key=os.environ.get("GROQ_API_KEY")
)

# Custom CSS loading function
def add_custom_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Footer function to display "Developed By Sehan D Amendra"
def add_footer():
    footer = """
    <div class="footer">
        Developed By Sehan D Amendra
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

# Function to save a single Q&A to the PDF
def save_qa_to_pdf(question, answer, pdf_file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Q: {question}", ln=True)
    pdf.multi_cell(200, 10, txt=f"Answer: {answer}")
    pdf.ln(10)
    pdf.output(pdf_file_path, 'F')

# Function to save all Q&A to the PDF at once
def save_all_qa_to_pdf(qa_pairs, pdf_file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for i, (question, answer) in enumerate(qa_pairs, 1):
        pdf.cell(200, 10, txt=f"Q{i}: {question}", ln=True)
        pdf.multi_cell(200, 10, txt=f"Answer: {answer}")
        pdf.ln(10)
    pdf.output(pdf_file_path)

def main():
    # Load the custom CSS
    add_custom_css()

    st.title("Data Analysis AI Agent")

    if "questions_and_answers" not in st.session_state:
        st.session_state.questions_and_answers = []

    if "pdf_file_path" not in st.session_state:
        st.session_state.pdf_file_path = "questions_and_answers.pdf"

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)

            st.write("Here is a preview of your dataset:")
            st.write(data.head())

            df = SmartDataframe(data, config={"llm": llm})

            query = st.text_input("Ask a question about the dataset:")

            if query:
                answer = df.chat(query)
                st.write("Answer:")
                st.write(answer)

                st.session_state.questions_and_answers.append((query, answer))

               

            if st.session_state.questions_and_answers:
                if st.button("      Save      "):
                    save_all_qa_to_pdf(st.session_state.questions_and_answers, st.session_state.pdf_file_path)
                    with open(st.session_state.pdf_file_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_file,
                            file_name="questions_and_answers.pdf",
                            mime="application/pdf"
                        )

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Display the footer at the bottom of the app
    add_footer()

if __name__ == "__main__":
    main()
