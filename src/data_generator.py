import json
import random
import string
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------
# Base STT-like dictionaries
# ---------------------------------------------------------
DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three",
    "4": "four", "5": "five", "6": "six", "7": "seven",
    "8": "eight", "9": "nine"
}

FILLERS = [
    "uh", "umm", "basically", "like", "you know", "i mean",
    "sorta", "kinda", "honestly", "actually"
]

CONV_PREFIXES = [
    "so i was saying",
    "and then he told me",
    "yesterday we were talking",
    "listen i wanted to say",
    "so basically what happened was",
    "then later in the afternoon",
    "i remember telling him",
    "when we were discussing"
]

CONV_SUFFIXES = [
    "and then we moved on",
    "after that i mentioned",
    "then she said",
    "you know what i mean",
    "so anyway",
    "and i guess that is it",
    "but that was earlier"
]

HOMOPHONES = {"for": "four", "four": "for", "to": "two", "two": "to"}

NAMES = ["rohan kumar", "alice reddy", "john doe", "vikram singh", "maria fernandez"]
CITIES = ["chennai", "mumbai", "delhi", "austin", "seattle"]
LOCATIONS = ["main street", "sector twelve", "building five", "bay area", "local market"]
EMAIL_DOMAINS = ["gmail dot com", "yahoo dot com", "outlook dot com"]


# ---------------------------------------------------------
# Light STT noise functions
# ---------------------------------------------------------
def maybe_noise(word):
    """Light STT noise."""
    if word in HOMOPHONES and random.random() < 0.05:
        return HOMOPHONES[word]

    # occasional slur
    if random.random() < 0.03:
        return word.replace(" ", "")

    return word


def add_sentence_noise(sentence):
    """Applies light noise only."""
    tokens = sentence.split()

    # deletion
    if len(tokens) > 6 and random.random() < 0.1:
        del tokens[random.randrange(len(tokens))]

    # repetition
    if len(tokens) > 6 and random.random() < 0.1:
        i = random.randrange(len(tokens))
        tokens.insert(i, tokens[i])

    # token noise
    tokens = [maybe_noise(t) for t in tokens]

    # merge two words
    if len(tokens) > 6 and random.random() < 0.07:
        i = random.randint(1, len(tokens)-2)
        tokens[i] = tokens[i] + tokens[i+1]
        del tokens[i+1]

    return " ".join(tokens)


# ---------------------------------------------------------
# Entity generators
# ---------------------------------------------------------
def gen_phone():
    return " ".join(DIGITS[random.choice(string.digits)] for _ in range(10))

def gen_credit_card():
    return " ".join(DIGITS[random.choice(string.digits)] for _ in range(16))

def gen_email():
    name = random.choice(["alice", "rohan", "john", "maria"])
    domain = random.choice(EMAIL_DOMAINS)
    return f"{name} at {domain}"

def gen_person_name():
    return random.choice(NAMES)

def gen_date():
    day = random.randint(1, 28)
    month = random.choice(["january", "february", "march", "april", "may", "june"])
    year = random.choice(["twenty twenty four", "twenty twenty three"])
    return f"{day} {month} {year}"

def gen_city():
    return random.choice(CITIES)

def gen_location():
    return random.choice(LOCATIONS)


ENTITY_GENERATORS = {
    "PHONE": gen_phone,
    "CREDIT_CARD": gen_credit_card,
    "EMAIL": gen_email,
    "PERSON_NAME": gen_person_name,
    "DATE": gen_date,
    "CITY": gen_city,
    "LOCATION": gen_location,
}

ALL_ENTITY_TYPES = list(ENTITY_GENERATORS.keys())


# ---------------------------------------------------------
# Conversational template builder
# ---------------------------------------------------------
def build_conversational_sentence(entities):
    parts = []

    if random.random() < 0.7:
        parts.append(random.choice(CONV_PREFIXES))

    if random.random() < 0.5:
        parts.append(random.choice(FILLERS))

    for etype, val in entities:
        clause = random.choice([
            f"i mentioned my {etype.lower().replace('_',' ')} which is {val}",
            f"i also said the {etype.lower().replace('_',' ')} was {val}",
            f"the {etype.lower().replace('_',' ')} is actually {val}",
            f"i told her that my {etype.lower().replace('_',' ')} is {val}",
            f"then later i shared the {etype.lower().replace('_',' ')} {val}",
        ])
        parts.append(clause)

    if random.random() < 0.6:
        parts.append(random.choice(CONV_SUFFIXES))

    text = " ".join(" ".join(parts).split())
    text = add_sentence_noise(text)

    return text


# ---------------------------------------------------------
# Create a labeled example (1–3 entities)
# ---------------------------------------------------------
def create_labeled_example(types):
    values = [ENTITY_GENERATORS[t]() for t in types]
    txt = build_conversational_sentence(list(zip(types, values)))

    entities = []
    for etype, val in zip(types, values):
        try:
            s = txt.index(val)
            e = s + len(val)
            entities.append({"start": s, "end": e, "label": etype})
        except ValueError:
            continue

    return {"id": f"utt_{random.randint(10000,99999)}", "text": txt, "entities": entities}


# ---------------------------------------------------------
# Create negative (no entity) examples
# ---------------------------------------------------------
def create_negative_example():
    parts = []

    # conversational fluff
    parts.append(random.choice(CONV_PREFIXES))
    parts.append(random.choice(FILLERS))

    # unrelated content
    random_words = [
        "we talked about the movie scene yesterday",
        "i was walking down the road thinking about food",
        "then later we discussed the weather and traffic",
        "he told me about his trip to the market",
        "i mentioned nothing important basically",
    ]
    parts.append(random.choice(random_words))
    parts.append(random.choice(CONV_SUFFIXES))

    text = " ".join(" ".join(parts).split())
    text = add_sentence_noise(text)

    return {
        "id": f"utt_{random.randint(10000,99999)}",
        "text": text,
        "entities": []
    }


# ---------------------------------------------------------
# Write a balanced dataset
# ---------------------------------------------------------
def write_balanced_dataset(path, total_samples):
    NEG_RATIO = 0.15  # 15% negative samples
    num_neg = int(total_samples * NEG_RATIO)
    num_pos = total_samples - num_neg

    per_entity = num_pos // len(ALL_ENTITY_TYPES)

    with open(path, "w", encoding="utf-8") as f:

        # --- Balanced labeled examples ---
        for etype in ALL_ENTITY_TYPES:
            for _ in range(per_entity):
                # 1–3 entities: always include the main entity type
                num_entities = random.randint(1, 3)
                types = [etype] + random.sample(ALL_ENTITY_TYPES, num_entities - 1)
                ex = create_labeled_example(types)
                f.write(json.dumps(ex) + "\n")

        # --- Negative examples ---
        for _ in range(num_neg):
            ex = create_negative_example()
            f.write(json.dumps(ex) + "\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    write_balanced_dataset("data/train.jsonl", 1000)
    write_balanced_dataset("data/dev.jsonl", 200)
    write_balanced_dataset("data/test.jsonl", 200)
    print("✓ dataset generated!")
