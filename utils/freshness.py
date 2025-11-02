def classify_vegetable(class_name: str, confidence: float) -> tuple[str, str, float, str]:
    """
    Classify vegetable into Healthy/Damaged/Rotten with proper recommendation.

    Args:
        class_name: YOLO class name (e.g., "carrot", "tomato", "damaged_carrot", "rotten_tomato")
        confidence: Detection confidence (0-1 scale)

    Returns:
        Tuple of (classified_name, status, freshness_level, recommendation)
        - classified_name: e.g., "healthy_carrot", "damaged_carrot", "rotten_carrot"
        - status: "good", "caution", or "bad"
        - freshness_level: 0-100 percentage
        - recommendation: User-friendly text
    """
    class_lower = class_name.lower()

    # Extract base vegetable name
    base_veggie = class_lower
    for prefix in ['healthy_', 'fresh_', 'damaged_', 'rotten_', 'bad_', 'old_', 'aging_']:
        if class_lower.startswith(prefix):
            base_veggie = class_lower.replace(prefix, '')
            break

    # Determine freshness category
    if 'rotten' in class_lower or 'bad' in class_lower:
        # Rotten vegetables
        classified_name = f"rotten_{base_veggie}"
        status = "bad"
        freshness_level = max(0, min(40, int(confidence * 40)))  # 0-40%
        recommendation = f"Not recommended for consumption. This {base_veggie} shows signs of significant spoilage and rot. Please dispose of it."

    elif 'damaged' in class_lower or 'old' in class_lower or 'aging' in class_lower:
        # Damaged/aging vegetables
        classified_name = f"damaged_{base_veggie}"
        status = "caution"
        freshness_level = 40 + int(confidence * 35)  # 40-75%
        recommendation = f"Caution advised. This {base_veggie} shows some deterioration. Cook thoroughly or use in prepared dishes soon."

    else:
        # Healthy/fresh vegetables
        classified_name = f"healthy_{base_veggie}"
        status = "good"
        freshness_level = 70 + int(confidence * 30)  # 70-100%
        recommendation = f"Recommended for consumption. This {base_veggie} appears fresh and healthy. Safe to eat raw or cooked."

    return classified_name, status, freshness_level, recommendation


def compute_freshness(confidence: float, labels: list[str]) -> tuple[float, str, str]:
    """
    Legacy function for backward compatibility.
    Compute freshness score based on confidence and detected labels.

    Args:
        confidence: YOLOv8 detection confidence (0-1 scale)
        labels: List of detected labels

    Returns:
        Tuple of (freshness_score, status, recommendation)
    """
    # Convert confidence to percentage
    freshness = confidence * 100

    # Check for damage/rot indicators in labels
    has_rotten = any('rotten' in str(label).lower() or 'bad' in str(label).lower() for label in labels)
    has_damage = any('damaged' in str(label).lower() or 'old' in str(label).lower() or 'aging' in str(label).lower() for label in labels)

    if has_rotten:
        freshness = min(40, freshness * 0.4)  # 0-40%
        status = "Bad"
        recommendation = "Not recommended for consumption. Significant spoilage detected."
    elif has_damage:
        freshness = 40 + (freshness * 0.35)  # 40-75%
        status = "Caution"
        recommendation = "Consume soon. Some deterioration detected. Best for cooking."
    else:
        freshness = 70 + (freshness * 0.3)  # 70-100%
        status = "Good"
        recommendation = "Recommended for consumption. Appears fresh and healthy."

    return freshness, status, recommendation

