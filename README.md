# Webcam Face Recognition

A Python project that detects faces in a webcam feed with OpenCV and classifies them using a small PyTorch CNN. The model trains locally from your reference photos each time you launch recognition.

## Setup on Windows

You need Python 3.12 and a webcam. Run these commands in PowerShell from the project folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Add reference photos

Add clear, front-facing photos for at least two people. Use a few photos per person with different lighting or angles. The script accepts JPEG and PNG files and uses the largest face in each image.

```text
known_faces/
  person_one/
    01.jpg
    02.jpg
  person_two/
    01.jpg
    02.jpg
```

You can also capture up to 15 webcam samples for a new person. Press **Q** in the capture window to stop early. Use a new folder name for each capture session to avoid overwriting previously numbered captures.

```powershell
.\.venv\Scripts\python.exe test.py --capture person_one
```

## Run recognition

```powershell
.\.venv\Scripts\python.exe test.py
```

The terminal reports the images used for training and the accuracy on those training samples. After training, the webcam window shows each predicted name and its model score. Scores below 0.65 display **Unknown**. Press **Q** in the webcam window to quit.

Live detection runs on a reduced-size frame. The classifier uses crops from the original frame and processes all detected faces together. Smaller or distant faces can be harder to detect. An unfamiliar face can still receive a high score for a known person, so keep this project for experimentation.

## Push to Git

The included `.gitignore` excludes `known_faces/`, virtual environments, and Python caches. Keep reference photos local and use images from people who agree to participate.

```powershell
git init
git add README.md requirements.txt .gitignore test.py
git commit -m "Add webcam face recognition project"
git branch -M main
git remote add origin "<repository-url>"
git push -u origin main
```

Replace `<repository-url>` with the URL of an empty remote repository.

