# Spam-Email-Detection
    # Spam Email Detection System

This is a machine learning project that detects whether an email is **Spam** or **Not Spam** using Natural Language Processing (NLP).

## About the Project

The goal of this project is to classify emails based on their content. The text is cleaned and converted into numerical features using TF-IDF, and multiple machine learning models are trained and compared to find the best-performing one.

## Features

- Spam and Ham email classification
- Text preprocessing
- TF-IDF vectorization
- Multiple ML models for comparison
- Streamlit web application for predictions

## Technologies Used

- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- NLTK
- XGBoost

## How to Run

1. Clone this repository.
2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Train the model.

```bash
python train_model.py
```

4. Run the application.

```bash
streamlit run app.py
```

## Future Improvements

- Improve prediction accuracy with larger datasets
- Add phishing URL detection
- Integrate with Gmail API

## Author

**Avani Krishna**
