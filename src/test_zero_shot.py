
from zero_shot_detector import detect_health_anxiety
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 28 manual test examples
test_examples = [
    {"text": "I've had a headache since this morning. What could be causing it?", "expected": "normal_health_concern"},
    {"text": "I'm terrified that my headache means I have a brain tumour.", "expected": "health_anxiety"},
    {"text": "My throat has been sore for two days. Should I be concerned?", "expected": "normal_health_concern"},
    {"text": "I keep checking my throat because I'm scared something serious is wrong.", "expected": "health_anxiety"},
    {"text": "What are common causes of an elevated heart rate?", "expected": "normal_health_concern"},
    {"text": "I constantly check my heart rate because I'm afraid my heart is going to stop.", "expected": "health_anxiety"},
    {"text": "My doctor said my test results are normal. Can symptoms still persist?", "expected": "normal_health_concern"},
    {"text": "My doctor said everything is fine, but I'm still convinced they missed something.", "expected": "health_anxiety"},
    {"text": "Can stress cause stomach discomfort?", "expected": "normal_health_concern"},
    {"text": "Every time my stomach hurts, I panic and think I have a serious illness.", "expected": "health_anxiety"},
    {"text": "I've noticed a new skin mark. What characteristics should I ask my doctor about?", "expected": "normal_health_concern"},
    {"text": "I keep examining this skin mark because I'm convinced it could be cancer.", "expected": "health_anxiety"},
    {"text": "I'm scared about what might be causing these symptoms.", "expected": "health_anxiety"},
    {"text": "I'm frightened that this headache could be something serious.", "expected": "health_anxiety"},
    {"text": "I know it is probably nothing, but I cannot stop worrying about my symptoms.", "expected": "health_anxiety"},
    {"text": "I keep thinking about my health even when I'm trying to focus on other things.", "expected": "health_anxiety"},
    {"text": "My doctor reassured me, but I still need to know for certain that nothing is wrong.", "expected": "health_anxiety"},
    {"text": "I've asked several people whether this symptom is dangerous, but I still feel worried.", "expected": "health_anxiety"},
    {"text": "Every small change in my body makes me think I have a serious disease.", "expected": "health_anxiety"},
    {"text": "I immediately assume the worst whenever I notice something unusual about my health.", "expected": "health_anxiety"},
    {"text": "I'm worried about my fever because it has lasted several days. Should I contact my doctor?", "expected": "normal_health_concern"},
    {"text": "I'm nervous about my upcoming surgery. What should I expect?", "expected": "normal_health_concern"},
    {"text": "I'm scared because my doctor found something unusual and wants to investigate it further.", "expected": "normal_health_concern"},
    {"text": "I feel anxious about my blood test tomorrow.", "expected": "normal_health_concern"},
    {"text": "What are the common symptoms of iron deficiency?", "expected": "normal_health_concern"},
    {"text": "Can dehydration cause headaches?", "expected": "normal_health_concern"},
    {"text": "How long does a typical cold last?", "expected": "normal_health_concern"},
    {"text": "What should I expect during a routine blood test?", "expected": "normal_health_concern"}
]


y_true = []
y_pred = []

print("\n--- ZERO-SHOT CLASSIFICATION RESULTS ---\n")

for i, example in enumerate(test_examples, start=1):

    result = detect_health_anxiety(example["text"])

    prediction = result["prediction"]
    expected = example["expected"]

    y_true.append(expected)
    y_pred.append(prediction)

    print(f"Example {i}")
    print(f"Text: {example['text']}")
    print(f"Expected: {expected}")
    print(f"Predicted: {prediction}")

    for label, score in result["scores"].items():
        print(f"{label}: {score:.4f}")

    print(f"Correct: {prediction == expected}")
    print("-" * 60)


# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred,
    pos_label="health_anxiety",
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    pos_label="health_anxiety",
    zero_division=0
)

f1 = f1_score(
    y_true, y_pred,
    pos_label="health_anxiety",
    zero_division=0
)


print("\n--- EVALUATION METRICS ---")
print(f"Total examples: {len(test_examples)}")
print(f"Correct predictions: {sum(a == b for a, b in zip(y_true, y_pred))}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_true,
    y_pred,
    labels=["normal_health_concern", "health_anxiety"],
    zero_division=0
))


print("--- CONFUSION MATRIX ---")
print("Rows = Actual labels; Columns = Predicted labels")
print("Label order: normal_health_concern, health_anxiety")

print(confusion_matrix(
    y_true,
    y_pred,
    labels=["normal_health_concern", "health_anxiety"]
))
