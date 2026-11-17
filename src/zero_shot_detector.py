
from transformers import pipeline

print("Loading zero-shot classification model...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

print("Model loaded successfully!")

CANDIDATE_LABELS = [
    "a person asking an ordinary question about a health symptom",
    "a person expressing excessive worry about their health, "
    "such as repeatedly checking symptoms or fearing serious illness"
]

LABEL_MAPPING = {
    CANDIDATE_LABELS[0]: "normal_health_concern",
    CANDIDATE_LABELS[1]: "health_anxiety"
}


def detect_health_anxiety(text):

    result = classifier(
        text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This message is {}."
    )

    prediction = LABEL_MAPPING[result["labels"][0]]

    scores = {
        LABEL_MAPPING[label]: score
        for label, score in zip(
            result["labels"],
            result["scores"]
        )
    }

    return {
        "prediction": prediction,
        "scores": scores
    }
