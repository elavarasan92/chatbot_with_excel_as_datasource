import os
import faiss
import numpy as np
import pandas as pd
import psycopg2
import time
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from tabulate import tabulate
import re


# Load environment variables from .env (for local development)
load_dotenv()

# Fetch OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the key is loaded
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found! Make sure it's set in GitHub Secrets or .env")

# OpenAI Client setup
client = OpenAI(api_key=OPENAI_API_KEY)

# PostgreSQL Connection Setup
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

cursor = conn.cursor()

# Create Table for Drawing Data and Embeddings
cursor.execute("""
CREATE TABLE IF NOT EXISTS drawing_embeddings (
    id SERIAL PRIMARY KEY,
    drawing_number TEXT,
    observation TEXT,
    checked_date TEXT,
    owner TEXT,
    project_name TEXT,
    report_number TEXT,
    error TEXT,
    dwg_unique TEXT,
    embedding BYTEA
);
""")
conn.commit()

# Load Drawing Dataset
df = pd.read_csv("data/full_drawing_data.csv")  # Ensure this file is formatted correctly


# Function to Generate OpenAI Embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    start_time = time.time()
    response = client.embeddings.create(input=text, model=model)
    end_time = time.time()
    print(f"get_embedding() took {end_time - start_time:.2f} seconds")
    return np.array(response.data[0].embedding)


# Store Embeddings in PostgreSQL
def store_embeddings():
    start_time = time.time()
    for _, row in df.iterrows():
        combined_text = f"Drawing Number: {row['Drawing Number']}, Observation: {row['Observation']}, " \
                        f"Checked Date: {row['Checked Date']}, Owner: {row['Owner']}, " \
                        f"Project Name: {row['Project Name']}, Report Number: {row['Report Number']}, " \
                        f"Error: {row['Error']}, DWG Unique: {row['Dwg unique']}"

        embedding = get_embedding(combined_text)

        binary_embedding = BytesIO()
        np.save(binary_embedding, embedding)
        binary_embedding.seek(0)

        cursor.execute("""
            INSERT INTO drawing_embeddings (drawing_number, observation, checked_date, owner, project_name, 
            report_number, error, dwg_unique, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (row['Drawing Number'], row['Observation'], row['Checked Date'], row['Owner'],
              row['Project Name'], row['Report Number'], row['Error'], row['Dwg unique'],
              binary_embedding.read()))

    conn.commit()
    end_time = time.time()
    print(f"store_embeddings() took {end_time - start_time:.2f} seconds")
    print("Embeddings stored in PostgreSQL.")


# Load FAISS Index from PostgreSQL
def load_faiss_index():
    start_time = time.time()
    cursor.execute("SELECT id, embedding FROM drawing_embeddings;")
    rows = cursor.fetchall()

    if not rows:
        print("No embeddings found in the database.")
        return None, None

    embeddings, ids = [], []

    for row in rows:
        drawing_id, binary_embedding = row
        embedding = np.load(BytesIO(binary_embedding))
        embeddings.append(embedding)
        ids.append(drawing_id)

    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    end_time = time.time()
    print(f"load_faiss_index() took {end_time - start_time:.2f} seconds")
    return index, ids


# Retrieve Top N Drawings Based on Query
def retrieve_relevant_data(query):
    start_time = time.time()
    query_embedding = get_embedding(query)
    _, faiss_indices = index.search(np.array([query_embedding]), len(df))

    relevant_drawings = []
    for i in faiss_indices[0]:
        cursor.execute("SELECT * FROM drawing_embeddings WHERE id = %s;", (ids[i],))
        relevant_drawings.append(cursor.fetchone())

    end_time = time.time()
    print(f"retrieve_relevant_data() took {end_time - start_time:.2f} seconds")
    return relevant_drawings


# Generate Response Using OpenAI
def generate_response(question):
    start_time = time.time()
    relevant_data = retrieve_relevant_data(question)

    unique_projects = list(set(row[5] for row in relevant_data))

    if "unique" in question.lower() and "projects" in question.lower():
        return unique_project_table(unique_projects)

    '''if contains_keywords(question.lower()):
        print("in defects")
        return defect_percentage(relevant_data)'''


    context = "\n".join([
        f"Drawing Number: {row[1]}, Observation: {row[2]}, Checked Date: {row[3]}, "
        f"Owner: {row[4]}, Project Name: {row[5]}, Report Number: {row[6]}, Error: {row[7]}, DWG Unique: {row[8]}"
        for row in relevant_data
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a drawing inspection assistant. Use the provided data to answer questions accurately."},
            {"role": "user", "content": f"Here is the relevant data:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.7
    )

    end_time = time.time()
    print(f"generate_response() took {end_time - start_time:.2f} seconds")
    return response.choices[0].message.content


def unique_project_table(unique_projects):
    table_data = [[project] for project in unique_projects]
    headers = ["Project Name"]
    unique_projects_table = tabulate(table_data, headers, tablefmt="grid")
    print(f"unique_projects count : {len(unique_projects)}")
    return unique_projects_table

def contains_keywords(text):
    words = ["defect", "defects", "def", "percentage", "%"]
    return any(re.search(rf'\b{re.escape(word)}\b', text) for word in words)

def defect_percentage(relevant_data):
    data = {
        'Dwg unique': [row[8] for row in relevant_data],
        'Error': [row[7] for row in relevant_data]
    }
    df = pd.DataFrame(data)
    # Filter the DataFrame for "Defect" errors
    defect_df = df[df['Error'] == 'Defect']
    # Count distinct non-null 'Dwg unique' values
    distinct_defect_count = defect_df['Dwg unique'].nunique()
    # Apply the IF condition
    m_defect = 0 if distinct_defect_count == 0 else distinct_defect_count
    defect_percentage = (m_defect / len(df)) * 100
    return defect_percentage

# Chatbot Function
def chatbot():
    print("Chatbot: Hello! Ask me anything about drawing records. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = generate_response(user_input)
        print(f"Chatbot: {response}")


# Store embeddings if running for the first time
#store_embeddings()

# Load FAISS index
index, ids = load_faiss_index()

# Run the chatbot
chatbot()
