import json

# Path file intents.json (ganti jika lokasi berbeda)
with open('model/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tags = []
all_words = []

# Ekstrak tag dan pattern
for intent in data['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        words = pattern.lower().split()
        all_words.extend(words)

# Buang duplikat dan urutkan
tags = sorted(set(tags))
all_words = sorted(set(all_words))

# Tampilkan hasil
print("Tag:")
print(tags)

print("\nPattern:")
print(all_words)
