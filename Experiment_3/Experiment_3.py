import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Download required resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----------------------------
# Input Headlines
# -----------------------------
headlines = []

# Validate integer input
while True:
    try:
        n = int(input("Enter number of headlines: "))
        if n < 2:
            print("Please enter at least 2 headlines.")
            continue
        break
    except ValueError:
        print("Please enter a valid integer.")

# Read headlines
for i in range(n):
    headline = input(f"Enter headline {i+1}: ")
    headlines.append(headline)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(headlines)

# -----------------------------
# Cosine Similarity
# -----------------------------
print("\nCosine Similarity Matrix:")
print(cosine_similarity(X))

# -----------------------------
# K-Means Clustering
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X)

print("\nHeadline Clusters:")
for i in range(len(headlines)):
    print(f"{headlines[i]} -> Cluster {kmeans.labels_[i]}")

# -----------------------------
# WordNet Similarity
# -----------------------------
w1 = input("\nEnter first word: ")
w2 = input("Enter second word: ")

s1 = wn.synsets(w1)
s2 = wn.synsets(w2)

if s1 and s2:
    similarity = s1[0].path_similarity(s2[0])

    if similarity is not None:
        print("\nWordNet Similarity:", similarity)
    else:
        print("\nNo path similarity found.")
else:
    print("\nWord not found in WordNet.")