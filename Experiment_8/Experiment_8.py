import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# Input reviews
reviews = []
n = int(input("Enter number of reviews: "))

for i in range(n):
    reviews.append(input("Enter review: "))

# Convert text to document-term matrix
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# LDA Model
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Display Topics
words = vectorizer.get_feature_names_out()

print("\nTopics:")
for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i+1}")
    top_words = topic.argsort()[-5:]
    for j in top_words:
        print(words[j])

# t-SNE Visualization
X_dense = X.toarray()

# Perplexity should be less than number of reviews
perplexity = min(2, len(reviews)-1)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=perplexity
)

X_tsne = tsne.fit_transform(X_dense)

print("\nt-SNE Coordinates:")
for i, point in enumerate(X_tsne):
    print(f"Review {i+1}: {point}")

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], s=80)

# Review labels
for i in range(len(reviews)):
    plt.text(X_tsne[i,0], X_tsne[i,1], f"R{i+1}", fontsize=12)

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(False)
plt.show()