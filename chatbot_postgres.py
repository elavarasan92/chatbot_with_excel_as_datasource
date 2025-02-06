import numpy as np
import pandas as pd
import psycopg2
from openai import OpenAI

# Database connection details
DB_NAME = "chatbot"
DB_USER = "elavarasan"
DB_PASSWORD = "Elavarasan92@"
DB_HOST = "localhost"  # Change if using a remote server
DB_PORT = "5432"

# OpenAI API Key
client = OpenAI(
    api_key="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"
    )
model = "gpt-3.5-turbo"
embedding_model = "text-embedding-ada-002"


# Connect to PostgreSQL
def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )


# Create embeddings table if not exists
def create_table():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pokemon_embeddings (
            id SERIAL PRIMARY KEY,
            name TEXT,
            embedding vector(1536) -- Ensure vector size matches the OpenAI model
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


# Get embedding from OpenAI
def get_embedding(text):
    response = client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding


# Store embeddings in PostgreSQL
def store_embeddings():
    df = pd.read_csv("pokemon.csv")
    conn = get_db_connection()
    cur = conn.cursor()

    for _, row in df.iterrows():
        combined_text = (
            f"Name: {row['Name']}, Type 1: {row['Type 1']}, Type 2: {row['Type 2']}, "
            f"Total: {row['Total']}, HP: {row['HP']}, Attack: {row['Attack']}, "
            f"Defense: {row['Defense']}, Sp. Atk: {row['Sp. Atk']}, Sp. Def: {row['Sp. Def']}, "
            f"Speed: {row['Speed']}, Generation: {row['Generation']}, Legendary: {row['Legendary']}"
        )

        embedding = get_embedding(combined_text)
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"  # Convert list to string for PGVector

        cur.execute("INSERT INTO pokemon_embeddings (name, embedding) VALUES (%s, %s)", (row["Name"], embedding_str))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Pokémon embeddings stored in PostgreSQL!")


# Retrieve similar Pokémon using cosine similarity
def retrieve_relevant_data(query):
    conn = get_db_connection()
    cur = conn.cursor()

    query_embedding = get_embedding(query)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    cur.execute("SELECT name FROM pokemon_embeddings ORDER BY embedding <=> %s LIMIT 5;", (query_embedding_str,))
    results = cur.fetchall()

    cur.close()
    conn.close()
    return [row[0] for row in results]


# Generate a response using OpenAI GPT
def generate_response(question):
    relevant_pokemon = retrieve_relevant_data(question)
    context = "\n".join(relevant_pokemon)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a Pokémon expert. Use the provided data to answer questions accurately."},
            {"role": "user", "content": f"Here are some relevant Pokémon:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content


# Chatbot function
def chatbot():
    print("Chatbot: Hello! Ask me anything about Pokémon. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = generate_response(user_input)
        print(f"Chatbot: {response}")


# Main execution
if __name__ == "__main__":
    create_table()  # Ensure table exists
    store_embeddings()  # Store Pokémon embeddings
    chatbot()  # Run chatbot
