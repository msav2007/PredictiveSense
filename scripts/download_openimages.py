import fiftyone.zoo as foz

classes = [
    "Watch",
    "Glasses",
    "Headphones",
    "Mobile phone",
    "Bottle",
    "Bowl",
    "Mug",
    "Coffee cup",
    "Computer keyboard",
    "Computer mouse",
    "Pen",
    "Pencil",
    "Tin can",
    "Cocktail shaker",
]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=classes,
    max_samples=5000,
)

print(dataset)
print("Downloaded Open Images subset successfully.")