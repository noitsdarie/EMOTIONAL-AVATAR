# Creating an Emotional Avatar with Dynamic Facial Expressions During Human-Computer Interaction

## Project Description

This software application is designed to analyze the user's facial movements using a webcam and reproduce them on an interactive three-dimensional avatar in real time.

The system uses computer vision techniques to detect facial landmarks, analyze facial expression parameters, and track the user's head position. The collected data is transmitted to the rendering module, which updates the 3D model according to changes in the user's facial expressions.

### Main System Features

- Real-time face tracking
- Facial landmark detection
- Facial movement analysis
- Basic emotional state recognition
- Interactive 3D avatar rendering
- Facial mesh visualization mode
- Facial landmark visualization over the video stream
- Compatibility with standard webcams without additional hardware

---

## Technologies Used

- Python 3.11+
- OpenCV
- MediaPipe
- NumPy
- Pygame

---

## System Requirements

### Minimum Requirements

| Component | Requirements |
|-----------|---------|
| Processor | Intel Core i3 or equivalent |
| RAM | 4 GB |
| Camera | 720p webcam |
| Python | Version 3.11 or later |

### Recommended Requirements

| Component | Requirements |
|-----------|---------|
| Processor | Intel Core i5 / Ryzen 5 or newer |
| RAM | 8 GB or more |
| Camera | Full HD webcam |
| Lighting | Stable and evenly distributed |

---

## Installation and Launch Recommendations

> **Important!**
>
> To avoid conflicts with libraries and software already installed on the user's computer, it is recommended to run the project in a separate Python virtual environment.

Using a virtual environment allows you to:

- Isolate project dependencies
- Avoid library version conflicts
- Simplify component installation and updates
- Prevent interference with other Python projects installed on the computer

---

## Creating a Virtual Environment

### Windows

Navigate to the project directory:

```bash
cd project_directory
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Once successfully activated, the following prefix will appear in the command prompt:

```text
(venv)
```

### Linux

Create the environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### macOS

Create the environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## Installing Dependencies

After activating the virtual environment, run:

```bash
pip install -r requirements.txt
```

All required libraries will be installed automatically.

---

## Running the Application

After installing the dependencies, run:

```bash
python main.py
```

or

```bash
python3 main.py
```

depending on your operating system configuration.

---

## Operating Modes

### 3D Avatar Mode

The primary operating mode of the system.

Features:

- User facial expression analysis
- Head position tracking
- 3D avatar geometry updates
- Real-time animation rendering

---

### Mesh Mode

Designed to demonstrate the polygonal structure of the face.

The following elements are displayed:

- Model vertices
- Edges connecting the vertices
- 3D mesh structure

---

### Facial Landmark Mode

Used to analyze the performance of computer vision algorithms.

The following elements are displayed:

- Live camera feed
- All detected facial landmarks
- Face Landmarker processing results

---

## Shutting Down the Application

To exit the application, simply close the main program window. After shutting down the application, it is recommended to deactivate the virtual environment:

```bash
deactivate
```

---

## Troubleshooting

### Camera Not Detected

Check the following:

- Whether the webcam is properly connected
- Whether it is being used by another application
- Whether permission to access the camera has been granted

### Library Import Errors

Make sure the following command:

```bash
pip install -r requirements.txt
```

has completed successfully.

### Poor Performance

The following actions are recommended:

- Close unnecessary background applications
- Ensure adequate lighting conditions
- Reduce CPU usage by closing resource-intensive applications

---

## Author

Bachelor's thesis by Dariia Oleksandrivna Dudiak, titled:

**"Creating an Emotional Avatar with Dynamic Facial Expressions During Human-Computer Interaction"**
