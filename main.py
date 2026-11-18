import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (100, 100)
CAPTURE_COUNT = 15
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001
CONFIDENCE_THRESHOLD = 0.65


class SimpleFaceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def preprocess_face(gray_face):
    resized = cv2.resize(gray_face, FACE_SIZE)
    equalized = cv2.equalizeHist(resized)
    return equalized


def extract_face_features(face):
    normalized = face.astype(np.float32) / 255.0
    edges = cv2.Canny(face, 60, 120).astype(np.float32) / 255.0
    return np.stack([normalized, edges], axis=0)


def augment_face_samples(face):
    return [
        face,
        cv2.flip(face, 1),
        cv2.GaussianBlur(face, (3, 3), 0),
    ]


def detect_faces(gray_image, cascade):
    return cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )


def detect_largest_face(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, cascade)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return preprocess_face(gray[y:y + h, x:x + w])


def iter_training_images():
    KNOWN_FACES_DIR.mkdir(exist_ok=True)

    for person_dir in KNOWN_FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        for image_path in person_dir.iterdir():
            if image_path.suffix.lower() in VALID_EXTENSIONS:
                yield person_dir.name, image_path

    for image_path in KNOWN_FACES_DIR.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            yield image_path.stem, image_path


def load_training_data(cascade):
    feature_maps = []
    labels = []
    label_names = {}
    name_to_label = {}
    person_counts = defaultdict(int)

    for person_name, image_path in iter_training_images():
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        face = detect_largest_face(image, cascade)
        if face is None:
            print(f"Skipping {image_path.name}: no face found")
            continue

        if person_name not in name_to_label:
            label = len(name_to_label)
            name_to_label[person_name] = label
            label_names[label] = person_name

        label = name_to_label[person_name]
        for sample in augment_face_samples(face):
            feature_maps.append(extract_face_features(sample))
            labels.append(label)
        person_counts[person_name] += 1

    if not feature_maps:
        return None, None, {}, {}

    x = torch.tensor(np.array(feature_maps), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y, label_names, dict(person_counts)


def train_model(features, labels, num_classes):
    model = SimpleFaceCNN(num_classes)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.2%}")

    return model


def detect_webcam_faces(gray, cascade):
    height, width = gray.shape
    scale = min(1.0, 480 / max(height, width))
    if scale == 1.0:
        return detect_faces(gray, cascade)

    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_height, small_width = small.shape
    faces = cascade.detectMultiScale(
        small,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(24, round(60 * scale)),) * 2,
    )

    return [
        (
            round(x * width / small_width),
            round(y * height / small_height),
            round(w * width / small_width),
            round(h * height / small_height),
        )
        for x, y, w, h in faces
    ]


@torch.inference_mode()
def predict_faces(model, gray, boxes, label_names):
    if len(boxes) == 0:
        return []

    features = np.stack([
        extract_face_features(preprocess_face(gray[y:y + h, x:x + w]))
        for x, y, w, h in boxes
    ])
    probabilities = model(torch.from_numpy(features)).softmax(dim=1)
    confidences, labels = probabilities.max(dim=1)
    return [
        (label_names[label] if confidence >= CONFIDENCE_THRESHOLD else "Unknown", confidence)
        for confidence, label in zip(confidences.tolist(), labels.tolist())
    ]


def open_camera():
    for index in (0, 1, 2):
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            return camera
        camera.release()
    return None


def capture_faces(person_name, cascade):
    person_dir = KNOWN_FACES_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    camera = open_camera()
    if camera is None:
        print("Could not open webcam.")
        return

    saved = 0
    print(f"Capturing {CAPTURE_COUNT} samples for {person_name}.")
    print("Look at the camera and press Q to stop.")

    while saved < CAPTURE_COUNT:
        ok, frame = camera.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = detect_faces(gray, cascade)

        for x, y, w, h in detected_faces:
            face = preprocess_face(gray[y:y + h, x:x + w])
            output_path = person_dir / f"{saved + 1:02d}.png"
            cv2.imwrite(str(output_path), face)
            saved += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Saved {saved}/{CAPTURE_COUNT}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            break

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(150) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images in {person_dir}")


def recognize_faces(cascade):
    features, labels, label_names, person_counts = load_training_data(cascade)
    if features is None:
        print("No training faces found.")
        print(f"Checked folder: {KNOWN_FACES_DIR}")
        print("Use this layout:")
        print("known_faces/alice/1.png")
        print("known_faces/bob/1.png")
        print("known_faces/bob/2.png")
        print("or run: python test.py")
        return

    if len(label_names) < 2:
        print("Add at least 2 different people for the PyTorch model.")
        print("Right now I found:")
        for person_name, count in sorted(person_counts.items()):
            print(f"- {person_name}: {count} image(s)")
        return

    print("Loaded training data:")
    for person_name, count in sorted(person_counts.items()):
        print(f"- {person_name}: {count} image(s)")

    print("Training PyTorch model...")
    model = train_model(features, labels, num_classes=len(label_names))
    model.eval()

    camera = open_camera()
    if camera is None:
        print("Could not open webcam.")
        return

    print("Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to read from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = detect_webcam_faces(gray, cascade)
            predictions = predict_faces(model, gray, detected_faces, label_names)
            status_text = "No face detected"
            status_color = (0, 180, 255)

            for (x, y, w, h), (name, confidence) in zip(detected_faces, predictions):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                text = f"{name} ({confidence:.2f})"

                status_text = text
                status_color = color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.imshow("PyTorch Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face recognition with OpenCV + PyTorch")
    parser.add_argument("--capture", metavar="NAME", help="capture webcam face samples for one person")
    args = parser.parse_args()

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print("Could not load Haar cascade.")
        return

    if args.capture:
        capture_faces(args.capture, cascade)
    else:
        recognize_faces(cascade)


if __name__ == "__main__":
    main()
