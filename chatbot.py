import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(
    api_key="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"
)

model = "gpt-3.5-turbo"

# Load the Pokémon dataset
df = pd.read_csv("pokemon.csv")

# Step 1: Create embeddings for each row in the dataset
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)

# Step 2: Combine all relevant columns into a single text for each row
df["combined"] = df.apply(
    lambda row: f"Name: {row['Name']}, Type 1: {row['Type 1']}, Type 2: {row['Type 2']}, "
                f"Total: {row['Total']}, HP: {row['HP']}, Attack: {row['Attack']}, "
                f"Defense: {row['Defense']}, Sp. Atk: {row['Sp. Atk']}, Sp. Def: {row['Sp. Def']}, "
                f"Speed: {row['Speed']}, Generation: {row['Generation']}, Legendary: {row['Legendary']}",
    axis=1
)

# Step 3: Generate embeddings for all rows and store them in a FAISS index
embeddings = np.vstack(df["combined"].apply(get_embedding).values)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Function to retrieve the top N most relevant rows based on the query
def retrieve_relevant_data(query):
    query_embedding = get_embedding(query)
    _, indices = index.search(np.array([query_embedding]), len(df))
    return df.iloc[indices[0]]

# Step 4: Generate response using OpenAI GPT with retrieved data as context
def generate_response(question):
    relevant_data = retrieve_relevant_data(question)
    context = "\n".join(relevant_data["combined"].values)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Pokémon expert. Use the provided data to answer questions accurately."},
            {"role": "user", "content": f"Here is the relevant data:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# Main chatbot function
def chatbot():
    print("Chatbot: Hello! Ask me anything about Pokémon. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = generate_response(user_input)
        print(f"Chatbot: {response}")

# Run the chatbot
chatbot()
