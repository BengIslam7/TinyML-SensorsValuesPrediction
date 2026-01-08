# TinyML-IoT

A collection of Tiny Machine Learning (TinyML) experiments and IoT projects built with Python and Jupyter Notebooks.
This repository focuses on **sensor data prediction and anomaly detection** workflows using machine learning models suited for deployment on edge devices (such as ESP32 boards) and for experimentation on desktop environments.

---

## 📌 Contents

### 📁 Notebooks

* **Anomaly_Detection_Model.ipynb**
  A Jupyter Notebook implementing one or more TinyML models for detecting anomalies in time-series sensor data. This notebook typically covers:

  * Data loading and preprocessing
  * Feature engineering for anomaly detection
  * Training machine learning or deep learning models
  * Evaluating model performance
  * Visualization of anomalous vs. normal behavior

* **Temperature_Humidity_Prediction.ipynb**
  A Jupyter Notebook to build a predictive model for temperature and humidity sensor data. Typical tasks include:

  * Data exploration and analysis
  * Regression modeling for continuous sensor values
  * Performance evaluation and visual plots
  * Preparing models for TinyML export (e.g., TensorFlow Lite)

---

## 🧠 What Is TinyML?

TinyML refers to the deployment of machine learning models on **resource-constrained embedded systems** such as microcontrollers and small IoT devices. It enables on-device intelligence with:

* Low latency
* Offline capability
* Reduced power and network dependency
  Even with limited CPU/RAM resources, TinyML can perform predictive tasks or detect anomalous patterns in sensor streams. ([GitHub][2])

---

## 🚀 How to Use This Repository

### 🛠 Requirements

Install Python and the following key libraries:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib jupyter
```

### 📓 Run the Notebooks

1. Open the repository in your local environment or JupyterLab/VS Code.
2. Launch:

   ```bash
   jupyter notebook
   ```
3. Navigate to and run either notebook:

   * `Anomaly_Detection_Model.ipynb`
   * `Temperature_Humidity_Prediction.ipynb`

### 📦 Exporting Models for Embedded Deployment

After training your model in Python:

1. Convert the trained model to **TensorFlow Lite** format (e.g., `.tflite`).
2. (Optional) Quantize the model for reduced size and faster inference on devices like ESP32.
3. Deploy to the target device using a framework such as TensorFlow Lite Micro.

---

## 📌 Examples of Possible Extensions

These notebooks can be extended to:

* Export models to microcontroller firmware (e.g., ESP32 via Arduino or PlatformIO)
* Perform real-time prediction from connected sensors
* Integrate anomaly outputs with IoT dashboards (MQTT, WebSockets)
* Quantize models to 8-bit fixed point (TinyML optimization)

---

## 📄 License

This repository is open source — feel free to use and modify the notebooks for personal or educational projects.
