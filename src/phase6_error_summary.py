
import csv
from pathlib import Path

INPUT_FILE = Path("outputs/phase6_error_analysis/per_image_metrics.csv")
OUTPUT_FILE = Path("outputs/phase6_error_analysis/error_summary.csv")


def main():
    rows = []

    with open(INPUT_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            tp = int(row["TP"])
            fp = int(row["FP"])
            fn = int(row["FN"])
            tn = int(row["TN"])

            actual_pixels = tp + fn
            predicted_pixels = tp + fp
            total_errors = fp + fn

            if actual_pixels == 0:
                error_type = "False detections - no actual change"

            elif predicted_pixels > actual_pixels * 1.5 and fp > fn:
                error_type = "Likely over-segmentation"

            elif fn > fp:
                error_type = "Mostly missed changes"

            else:
                error_type = "Mixed segmentation errors"

            row.update({
                "actual_change_pixels": actual_pixels,
                "predicted_change_pixels": predicted_pixels,
                "total_error_pixels": total_errors,
                "error_type": error_type
            })

            rows.append(row)

    # Rank by total incorrect pixels, highest first
    rows.sort(
        key=lambda row: int(row["total_error_pixels"]),
        reverse=True
    )

    fieldnames = list(rows[0].keys())

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nPHASE 6 - ERROR SUMMARY")
    print("-" * 75)

    categories = {}

    for row in rows:
        category = row["error_type"]
        categories[category] = categories.get(category, 0) + 1

    print("\nError category counts:")

    for category, count in categories.items():
        print(f"{category}: {count} images")

    print("\nTOP 10 IMAGES BY TOTAL ERROR PIXELS")
    print("-" * 75)

    for row in rows[:10]:
        print(
            f"Image {int(row['validation_index']):03d} | "
            f"Errors: {row['total_error_pixels']} | "
            f"Actual: {row['actual_change_pixels']} | "
            f"Predicted: {row['predicted_change_pixels']} | "
            f"{row['error_type']}"
        )

    print(f"\nSaved summary to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
