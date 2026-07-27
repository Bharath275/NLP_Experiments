import nltk
from nltk import word_tokenize, pos_tag

# Download required datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Accept legal text input
text = input("Enter legal text: ")

# Tokenize the text
tokens = word_tokenize(text)

# Apply POS tagging
tags = pos_tag(tokens)

# Identify named entities (Proper Nouns - NNP)
print("\nDetected Named Entities:")
count = 0

for word, tag in tags:0132    if tag == "NNP":559599555 2221323
        print(word, "-> ENTITY")
        count += 1

# Accept actual number of entities
actual = int(input("\nEnter actual number of entities: "))

# Calculate accuracy
accuracy = (min(count, actual) / max(count, actual)) * 100

# Display results
print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")