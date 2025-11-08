# RecycleVision - Recyclable Object Classifier

A web application that uses deep learning to identify and classify recyclable objects from images.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Make sure you have your data organized:
- Run `dataSplitting.py` to split your data into train/val/test sets
- Or ensure `splitData/` directory exists with `train/`, `val/`, and `test/` subdirectories
- Each subdirectory should contain folders for each recyclable category (cardboard, glass, metal, etc.)

### 3. Train the Model

Train the model using:
```bash
python3 main.py
```

This will create a trained model in the `model/` directory.

### 4. Run the Streamlit App

```bash
streamlit run website.py
```

The app will open in your default web browser, typically at `http://localhost:8501`

## Usage

1. **Capture or Upload an Image**: Use the webcam or upload an image file
2. **Click Analyze**: The model will predict the recyclable category
3. **View Results**: See the predicted material type, confidence score, and recycling tips

## Features

- **Local Model Inference**: Uses your trained model for predictions
- **Multiple Categories**: Classifies 9 recyclable categories
- **Recycling Tips**: Provides material-specific recycling guidance
- **Confidence Scores**: Shows prediction confidence and alternative possibilities

## Troubleshooting

- **Model not found**: Make sure you've trained the model using `main.py` first
- **Import errors**: Install all dependencies with `pip install -r requirements.txt`
- **Port already in use**: Streamlit will automatically try the next available port

