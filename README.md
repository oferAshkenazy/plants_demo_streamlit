plants_demo_streamlit

A Streamlit application for plant disease classification, leveraging a fine-tuned TensorFlow model and interactive Colab notebooks for experimentation.

Table of Contents

Overview

Features

Installation

Usage

Directory Structure

File Descriptions

Overview

This project provides:

A Streamlit web app (appv4.py) for uploading leaf images and predicting plant diseases.

Utility code (labels.py) to map model outputs to human-readable disease names.

Interactive Jupyter notebooks in the Colab/ directory demonstrating model conversion, training, and inference workflows.

Saved model artifacts and training histories in the Models/ directory for reproducibility.

Features

Web interface for image upload and real-time inference.

Colab demos for step-by-step guidance on model conversion to .h5 and running inference.

Deployment-ready via Poetry (pyproject.toml) or pip (requirements.txt).

Consistent runtime specification with runtime.txt for Streamlit Community Cloud or Heroku.

Installation

Clone the repository

git clone https://github.com/oferAshkenazy/plants_demo_streamlit.git
cd plants_demo_streamlit

Using Poetry (recommended)

poetry install
poetry run streamlit run appv4.py

Using pip

pip install -r requirements.txt
streamlit run appv4.py

Usage

Navigate to http://localhost:8501 in your browser.

Use the file uploader to select a leaf image (.jpg, .png, etc.).

View the predicted disease label and recommended treatment.

Directory Structure

├── .idea/               # JetBrains IDE settings (local development only)
├── Colab/               # Jupyter notebooks for demos
├── Models/              # Saved model weights and training histories
├── README.md            # Project overview and instructions (this file)
├── appv4.py             # Main Streamlit application script
├── labels.py            # Label-loading utilities for mapping model outputs
├── pyproject.toml       # Poetry configuration file
├── requirements.txt     # pip dependencies for non-Poetry environments
└── runtime.txt          # Python runtime specification for deployment

File Descriptions

.idea/

Contains IDE-specific configuration (workspace settings, run configurations) for JetBrains IDEs. Exclude from production deployments.

Colab/

Interactive Jupyter notebooks illustrating:

kerasTOh5.ipynb: Converting TensorFlow checkpoints to standalone .h5 model files.

plants_demo_streamlit_demo.ipynb: Running the Streamlit app in Google Colab and visualizing results.

Other notebooks: Model training, evaluation, and visualization workflows.

Models/

rMobileNetV2_finetuned_24.h5: Fine-tuned Keras model weights for plant disease classification.

history_rMobileNetV2_finetuned_24.pkl: Pickled training history (accuracy/loss per epoch) for plotting metrics.

README.md

The main documentation (this file) explaining setup, usage, and project structure.

appv4.py

Streamlit application:

Loads a model from Models/.

Presents a file uploader widget for leaf images.

Preprocesses images, runs inference, and displays predictions with recommended treatments.

labels.py

Provides functions to load and parse class labels, ensuring alignment between model output indices and human-readable disease names.

pyproject.toml

Poetry project file containing:

Project metadata (name, version, authors).

Python version constraints.

Dependency specifications for reproducible environments.

requirements.txt

List of packages for pip install workflows, including:

streamlit

tensorflow

opencv-python

Other dependencies needed by the app.

runtime.txt

Specifies the Python runtime (e.g., python-3.10.8) for platforms like Streamlit Community Cloud or Heroku to ensure consistent deployment environments.


