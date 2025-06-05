"""
Virtual environment- 
python -m venv folder_name
folder_name\Scripts\activate

to check python version: python --version
if it is 3.8
then to create and activate virtual environment:
py -3.12 -m venv folder_name
folder_name\Scripts\activate
"""
#111111111111111
# Program 1
import gensim.downloader as api

print("Loading model...")
wv = api.load("word2vec-google-news-300")
print("Model loaded!")

def show(word1=None, word2=None, word3=None, check_sim=False):
    try:
        if word1 and word2 and word3:
            print(f"\n{word1} - {word2} + {word3}:")
            for w, s in wv.most_similar(positive=[word1, word3], negative=[word2], topn=5):
                print(f"{w}: {s:.4f}")
        elif check_sim:
            print(f"\nSimilarity ({word1}, {word2}): {wv.similarity(word1, word2):.4f}")
        else:
            print(f"\nTop 5 similar to '{word1}':")
            for w, s in wv.most_similar(word1, topn=5):
                print(f"{w}: {s:.4f}")
    except KeyError as e:
        print(f"Error: {e}")

# Examples
show("king", "man", "woman")               # Vector arithmetic
show("cat", "dog", check_sim=True)         # Similarity
show("happy")                              # Similar words


#22222222222222222
# Program 2
import gensim.downloader as api
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

wv = api.load("word2vec-google-news-300")

def analogy(w1, w2, w3):  # word1 - word2 + word3
    try:
        res = wv.most_similar(positive=[w1, w3], negative=[w2], topn=5)
        print(f"\n{w1} - {w2} + {w3} ≈"); [print(f"{w}: {s:.4f}") for w, s in res]
        return [w for w, _ in res]
    except: return []

def plot(words, method='pca', perplexity=5):
    vecs = np.array([wv[w] for w in words])
    red = PCA(2) if method == 'pca' else TSNE(2, perplexity=perplexity)
    coords = red.fit_transform(vecs)
    plt.figure(figsize=(10, 7))
    [plt.scatter(*coords[i]) and plt.text(*coords[i], w) for i, w in enumerate(words)]
    plt.title(f"{method.upper()} Embeddings"); plt.grid(); plt.show()

def similar(w):
    try:
        print(f"\nSimilar to '{w}':"); [print(f"{x}: {s:.4f}") for x, s in wv.most_similar(w, topn=5)]
    except: print("Word not found.")

# --- Run ---
extra = analogy("king", "man", "woman")
royalty = ["king", "man", "woman", "queen", "prince", "princess", "royal", "throne"] + extra
plot(royalty, 'pca'); plot(royalty, 'tsne', 3)

tech = ["computer", "software", "hardware", "algorithm", "data", "network",
        "programming", "machine", "learning", "artificial"]
plot(tech, 'pca'); plot(tech, 'tsne', 3)
similar("computer"); similar("learning")






#33333333333333
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

corpus = [
    "The patient was diagnosed with diabetes and hypertension.",
    "MRI scans reveal brain tissue abnormalities.",
    "Treatment involves antibiotics and monitoring.",
    "Symptoms: fever, fatigue, muscle pain.",
    "Vaccine is effective against viral infections.",
    "Doctors recommend physical therapy.",
    "Trial results published in the journal.",
    "Surgeon performed invasive procedure.",
    "Prescription includes pain relievers.",
    "Diagnosis confirmed a genetic disorder."
]

tokens = [s.lower().split() for s in corpus]
model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, epochs=50)


words = model.wv.index_to_key
vecs = np.array([model.wv[w] for w in words])
tsne = TSNE(n_components=2, perplexity=5, random_state=0)
reduced = tsne.fit_transform(vecs)

plt.figure(figsize=(10, 7))
for i, w in enumerate(words):
    plt.scatter(*reduced[i]); plt.text(*reduced[i], w)
plt.title("Medical Word Embeddings"); plt.grid(); plt.show()

for word in ["treatment", "vaccine"]:
    try:
        print(f"\nSimilar to '{word}':")
        for w, s in model.wv.most_similar(word, topn=5):
            print(f"{w}: {s:.2f}")
    except KeyError:
        print(f"{word} not found.")



#444444444444444444444
# Program 4
# Install: pip install gensim transformers nltk torch
import gensim.downloader as api

print("Loading word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100")
print("Model loaded!")

prompt = input("Enter your GenAI prompt (with a key word): ")
words = prompt.lower().split()

seed = next((w for w in words if w in word_vectors), None)

if seed:
    similar = [w for w, _ in word_vectors.most_similar(seed, topn=3)]
    enriched_prompt = prompt + " (also consider: " + ", ".join(similar) + ")"

    print("\nOriginal Prompt:")
    print(prompt)

    print("\nEnriched Prompt:")
    print(enriched_prompt)

    print("\n--- Simulated AI Responses ---")
    print("\nOriginal Response:")
    print(f"The response focused on '{seed}' and gave a basic explanation.")

    print("\nEnriched Response:")
    print(f"The response included '{seed}' along with '{similar[0]}', '{similar[1]}', and '{similar[2]}', making it more detailed  .")

else:
    print("No valid word in prompt found invocabulary.")


#555555555555555555555555
# Program 5
import gensim.downloader as api
import random

model = api.load("glove-wiki-gigaword-100")

def generate_paragraph(seed):
    try:
        words = [w for w, _ in model.most_similar(seed, topn=5)]
    except KeyError:
        return "Seed word not found."

    templates = [
        f"The {seed} was surrounded by {words[0]} and {words[1]}.",
        f"People associate {seed} with {words[2]} and {words[3]}.",
        f"In the land of {seed}, {words[4]} was common.",
        f"A story about {seed} needs {words[1]} and {words[3]}."
    ]
    return " ".join(random.choices(templates, k=3))

# Run
seed = input("Enter a seed word: ")
print("\nGenerated Paragraph:\n" + generate_paragraph(seed))





#6666666666666666666666
# Program 6
from transformers import pipeline

print("🔍 Loading Sentiment Analysis Model...")
analyze = pipeline("sentiment-analysis")

reviews = [
    "The product is amazing! I love it so much.",
    "I'm very disappointed. The service was terrible.",
    "It was an average experience, nothing special.",
    "Absolutely fantastic quality! Highly recommended.",
    "Not great, but not the worst either."
]

print("\n📢 Customer Sentiment Analysis Results:")
for text in reviews:
    result = analyze(text)[0]
    print(f"\n📝 Input Text: {text}")
    print(f"📊 Sentiment: {result['label']} (Confidence: {result['score']:.4f})")

    
    
    
    
    
    
 #77777777777777777777
 # Program 7
from transformers import pipeline

print("🔍 Loading Summarization Model (BART)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    text = " ".join(text.split())
    settings = {
        "Default": {},
        "High randomness": {"do_sample": True, "temperature": 0.9},
        "Conservative": {"do_sample": False, "num_beams": 5},
        "Diverse sampling": {"do_sample": True, "top_k": 50, "top_p": 0.95},
    }

    print("\n📝 Original Text:\n", text)
    print("\n📌 Summarized Text:")
    for label, params in settings.items():
        summary = summarizer(text, max_length=150, min_length=30, **params)[0]['summary_text']
        print(f"{label}: {summary}")

# Example input
long_text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on creating intelligent machines
capable of mimicking human cognitive functions such as learning, problem-solving, and decision-making.
In recent years, AI has significantly impacted various industries, including healthcare, finance, education,
and entertainment. AI-powered applications, such as chatbots, self-driving cars, and recommendation systems,
have transformed the way we interact with technology. Machine learning and deep learning, subsets of AI,
enable systems to learn from data and improve over time without explicit programming.
However, AI also poses ethical challenges, such as bias in decision-making and concerns over job displacement.
As AI technology continues to advance, it is crucial to balance innovation with ethical considerations to ensure
its responsible development and deployment.
"""

# Run summarization
summarize_text(long_text)







#888888888888888888888
# Program 8
#!pip install lanchain_community langchain cohere
from langchain.llms import Cohere
from langchain import PromptTemplate

# Use regular input instead of getpass
api_key = input("🔑 Enter Cohere API Key: ")

# Load text from file
filepath = "/teach.txt"
with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

# Initialize Cohere LLM
llm = Cohere(model="command", cohere_api_key=api_key)

# Create prompt template
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an AI assistant helping to summarize and analyze a text document.
Here is the document content:

{text}

🔹 Summary: Provide a concise summary.
🔹 Key Takeaways: List 3 key points.
🔹 Sentiment Analysis: Positive, Negative, or Neutral.
"""
)

# Generate and print output
print("\n📌 Output:\n", llm.predict(prompt.format(text=text)))





#99999999999999999999999
#pip install wikipedia-api

import wikipediaapi

wiki = wikipediaapi.Wikipedia(user_agent="Chatbot")

institution_name = input("Enter the institution name: ").strip()
page = wiki.page(institution_name)

if not page.exists():
    print(f"The page for '{institution_name}' does not exist on Wikipedia.")
else:

    founder = "N/A"
    founded = "N/A"
    branches = "N/A"
    number_of_employees = "N/A"

    summary = page.summary

    for line in page.text.split('\n'):
        line_lower = line.lower()
        if 'founder' in line_lower and founder == "N/A":
            founder = line.split(':')[-1].strip()
        elif 'founded' in line_lower and founded == "N/A":
            founded = line.split(':')[-1].strip()
        elif 'branch' in line_lower and branches == "N/A":
            branches = line.split(':')[-1].strip()
        elif 'employee' in line_lower and number_of_employees == "N/A":
            try:
                number_of_employees = int(line.split(':')[-1].strip().replace(',', ''))
            except:
                number_of_employees = "N/A"

    print("\nInstitution Details:")
    print(f"Institution: {institution_name}")
    print(f"Founder: {founder}")
    print(f"Founded: {founded}")
    print(f"Branches: {branches}")
    print(f"Number of Employees: {number_of_employees}")
    print(f"\nSummary:\n{summary}")







#10101010110101010
#program 10
#pip install langchain cohere wikipedia-api pydantic 
#pip install langchain_community 
from IPython import get_ipython
from IPython.display import display

import PyPDF2
from cohere import Client

api_key = "9lvgpUeDFXZpM6hs4sU7EgcYqLwpQpyxUMwHC37v"
co = Client(api_key)

pdf_path = "ipc.pdf"

pdf_file = open("ipc.pdf", "rb")
pdf_reader = PyPDF2.PdfReader(pdf_file)
ipc_text = "".join(page.extract_text() for page in pdf_reader.pages)

print("Indian Penal Code document loaded.\n")

while True:
    question = input("Enter your question about IPC (or type 'exit' to quit): ").strip()
    if question.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    relevant_text = ipc_text[:2000]

    prompt = f"""
    You are an expert on the Indian Penal Code (IPC). Use the following IPC content to answer the question clearly and concisely:

    {relevant_text}

    Question: {question}

    Answer in 3 clear points:
    """

    response = co.generate(prompt=prompt, max_tokens=150).generations[0].text.strip()
    print("\nAnswer:\n", response, "\n")
