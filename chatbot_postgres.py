import os

import faiss
import numpy as np
import pandas as pd
import psycopg2
import time
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

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
    dbname="chatbot",
    user="postgres",
    password="Elavarasan92@",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Create Table for Pokémon Embeddings
cursor.execute("""
CREATE TABLE IF NOT EXISTS pokemon_embeddings (
    id SERIAL PRIMARY KEY,
    name TEXT,
    type1 TEXT,
    type2 TEXT,
    total INT,
    hp INT,
    attack INT,
    defense INT,
    sp_atk INT,
    sp_def INT,
    speed INT,
    generation INT,
    legendary BOOLEAN,
    embedding BYTEA
);
""")
conn.commit()

# Load Pokémon Dataset
df = pd.read_csv("pokemon.csv")


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
        combined_text = f"Name: {row['Name']}, Type 1: {row['Type 1']}, Type 2: {row['Type 2']}, " \
                        f"Total: {row['Total']}, HP: {row['HP']}, Attack: {row['Attack']}, " \
                        f"Defense: {row['Defense']}, Sp. Atk: {row['Sp. Atk']}, Sp. Def: {row['Sp. Def']}, " \
                        f"Speed: {row['Speed']}, Generation: {row['Generation']}, Legendary: {row['Legendary']}"

        embedding = get_embedding(combined_text)

        binary_embedding = BytesIO()
        np.save(binary_embedding, embedding)
        binary_embedding.seek(0)

        cursor.execute("""
            INSERT INTO pokemon_embeddings (name, type1, type2, total, hp, attack, defense, sp_atk, sp_def, speed, generation, legendary, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (row['Name'], row['Type 1'], row['Type 2'], row['Total'], row['HP'], row['Attack'], row['Defense'],
              row['Sp. Atk'], row['Sp. Def'], row['Speed'], row['Generation'], row['Legendary'],
              binary_embedding.read()))

    conn.commit()
    end_time = time.time()
    print(f"store_embeddings() took {end_time - start_time:.2f} seconds")
    print("Embeddings stored in PostgreSQL.")


# Load FAISS Index from PostgreSQL
def load_faiss_index():
    start_time = time.time()
    cursor.execute("SELECT id, embedding FROM pokemon_embeddings;")
    rows = cursor.fetchall()

    if not rows:
        print("No embeddings found in the database.")
        return None, None

    embeddings, ids = [], []

    for row in rows:
        pokemon_id, binary_embedding = row
        embedding = np.load(BytesIO(binary_embedding))
        embeddings.append(embedding)
        ids.append(pokemon_id)

    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    end_time = time.time()
    print(f"load_faiss_index() took {end_time - start_time:.2f} seconds")
    return index, ids


# Retrieve Top N Pokémon Based on Query
def retrieve_relevant_data(query):
    start_time = time.time()
    query_embedding = get_embedding(query)
    _, faiss_indices = index.search(np.array([query_embedding]), len(df))

    relevant_pokemon = []
    for i in faiss_indices[0]:
        cursor.execute("SELECT * FROM pokemon_embeddings WHERE id = %s;", (ids[i],))
        relevant_pokemon.append(cursor.fetchone())

    end_time = time.time()
    print(f"retrieve_relevant_data() took {end_time - start_time:.2f} seconds")
    return relevant_pokemon


# Generate Response Using OpenAI
def generate_response(question):
    start_time = time.time()
    relevant_data = retrieve_relevant_data(question)

    context = "\n".join([
        f"Name: {row[1]},  "
        f"Total: {row[4]}, HP: {row[5]}, Attack: {row[6]}, Defense: {row[7]}, "
        f"Sp. Atk: {row[8]}, Sp. Def: {row[9]}, Speed: {row[10]}, Generation: {row[11]}, Legendary: {row[12]}"
        for row in relevant_data
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are a Pokémon expert. Use the provided data to answer questions accurately."},
            {"role": "user", "content": f"Here is the relevant data:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.7
    )

    end_time = time.time()
    print(f"generate_response() took {end_time - start_time:.2f} seconds")
    return response.choices[0].message.content


# Chatbot Function
def chatbot():
    print("Chatbot: Hello! Ask me anything about Pokémon. Type 'exit' to quit.")

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
