Delirium Detector
=================
This project implements a machine learning pipeline to detect and predict delirium risk in hospitalized elderly patients using sensor data. The core model is an LSTM (Long Short-Term Memory) neural network designed to analyze time-series physiological data collected from patients, identifying early signs of hospital-induced delirium to enable timely interventions.

The solution integrates with a MongoDB database storing patient sensor data and current assignments + patient statuses, running continuous inference and updating patient status in near real-time.

Made by Harry Lu :3