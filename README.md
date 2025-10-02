

# Face Recognition & Distance Estimation

This project is a **real-time face recognition system** with **distance estimation** using OpenCV, DeepFace, and a K-Nearest Neighbors (KNN) classifier.
It was built during a **robotics camp** with the help of coaches as a collaborative learning project.

---

## 🚀 Features

* Real-time **face detection** using Haar Cascades (`OpenCV`).
* **Face recognition** with DeepFace embeddings.
* **KNN classifier** for identifying known faces.
* **Distance estimation** using focal length calibration.
* Smooth distance updates for more stable results.
* Simple live **UI overlay** showing name, confidence, and distance.

---

## 📦 Requirements

Install the following Python libraries before running:

```bash
pip install opencv-python deepface scikit-learn numpy
```

---

Got it 👍 I’ll adjust the README so it doesn’t sound rushed, and I’ll add the points about:

* Errors showing up during testing (not because it was rushed).
* Possible mistaken recognition if multiple faces are in the dataset.
* Making it clear that the **`Images/` folder must exist on your laptop**.

Here’s the updated version of those sections:

---

## 📂 Project Structure

```
.
├── Images/             # Folder containing reference face images (must exist on your laptop)
├── main.py             # Main program
└── README.md           # Documentation
```

* The `Images/` folder must be created in the **same directory as the script**.
* Place one or more **face images** in the folder.

  * The **filename (without extension)** is used as the person’s name.
  * Example: `Images/Alice.jpg` → recognized as "Alice".
* ⚠️ If you have **more than one image**, the program might occasionally mistake one person for another.

---


## ▶️ Usage

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/face-recognition-distance.git
   cd face-recognition-distance
   ```

2. Run the script:

   ```bash
   python main.py
   ```

3. Press **`q`** to quit the webcam feed.

---

## 📏 Distance Estimation

* The system estimates distance based on a **known face width** and **calibrated focal length**.
* Default values:

  * `KNOWN_DISTANCE = 20 cm`
  * `KNOWN_WIDTH = 15 cm`

---

## ⚠️ Notes & Troubleshooting

* Keep your **face within close distance of the laptop camera** for best recognition.
* Use **good lighting** so the system can properly detect your face.
* Make sure the **distance values in code** (`KNOWN_DISTANCE`, `KNOWN_WIDTH`) are realistic and close to your actual setup, otherwise distance calibration will be off.
* As we tested the project more, **different errors appeared** depending on the images, lighting, and camera conditions — this is normal for early testing.
* If the system fails to detect you, try:

  * Moving closer to the camera
  * Using a clear, frontal face image in the `Images/` folder
  * Reducing the number of reference images to avoid confusion

* Reminder: This was a **learning project** made with guidance during a robotics camp, not a finished production tool.

---

## 🏆 Credits

* Built at **Robotics Camp** with guidance from coaches.
* Libraries used: [OpenCV](https://opencv.org/), [DeepFace](https://github.com/serengil/deepface), [scikit-learn](https://scikit-learn.org/).

---

✨ Future improvements could include:

* Multi-face tracking
* Better distance smoothing
* GUI instead of command-line



---

