# Kestrel AI Service Request Routing

## 1. Project Overview

Kestrel AI Service Request Routing is a machine-learning-based application that classifies home appliance service requests into the appropriate service teams.

The system uses historical service request data to predict the initial routing team. It provides a REST API and a Streamlit dashboard for submitting requests and viewing predictions.

## 2. Project Objectives

- Automate service request classification.
- Reduce manual routing effort.
- Route customer requests into seven service categories.
- Provide a simple interface for employees.
- Evaluate model performance using historical data.

## 3. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Text Processing:** TF-IDF Vectorization
- **Classification Model:** Linear Support Vector Machine (LinearSVC)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Visualisation:** Matplotlib, Seaborn

## 4. Machine Learning Approach

The system uses a TF-IDF vectorizer with unigram and bigram features to convert customer request text into numerical features.

A Linear Support Vector Machine classifier predicts the appropriate service team.

The model uses the following input fields:

- Customer request text
- Product family
- Warranty status
- Request channel
- Request source

Historical labels were standardized to the current team names.

## 5. Model Evaluation

The model was evaluated using an 80:20 chronological train-validation split.

| Metric | Result |
|---|---:|
| Training records | 8,657 |
| Validation records | 2,165 |
| Accuracy | 95.20% |
| Macro F1 Score | 95.00% |

The final model was retrained using all 10,822 historical training records.

## 6. Service Teams

The model predicts one of seven teams:

1. Billing
2. Filters & Consumables
3. Installs & Demo
4. Product Advice
5. Repairs
6. Returns & Replacement
7. Warranty Claims

## 7. Project Structure

```text
kestrel-service-routing/
│
├── api/
│   └── main.py
│
├── app/
│   └── app.py
│
├── data/
│   ├── train.csv
│   ├── test_unlabelled.csv
│   └── sample_submission.csv
│
├── evidence/
│   ├── evaluation_report.txt
│   └── confusion_matrix.png
│
├── models/
│   └── routing_model.pkl
│
├── src/
│   └── train.py
│
├── predictions.csv
├── requirements.txt
└── README.md
```

## 8. Installation and Setup

### Prerequisites

- Python installed
- pip package manager

### Step 1: Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd kestrel-service-routing
```

### Step 2: Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Add the required datasets

Place the following files inside the `data/` directory:

- train.csv
- test_unlabelled.csv
- sample_submission.csv

These files are required for training and prediction generation. They should be obtained through the authorized assignment data pack.

## 9. Train the Model and Generate Predictions

Run:

```bash
python src/train.py
```

This script:

- Loads and preprocesses the datasets.
- Creates TF-IDF text features.
- Trains a Linear SVM classifier.
- Evaluates the model.
- Saves the confusion matrix and evaluation report.
- Retrains the final model on all historical training records.
- Saves `models/routing_model.pkl`.
- Generates `predictions.csv`.

The prediction file contains 2,178 test records.

## 10. Run the Backend API

Open a terminal and execute:

```bash
python -m uvicorn api.main:app --reload
```

API address:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Prediction endpoint:

```text
POST /predict
```

The endpoint accepts customer request details in JSON format and returns the predicted service team along with request information.

## 11. Run the Frontend Dashboard

Open a second terminal and execute:

```bash
python -m streamlit run app/app.py
```

Dashboard address:

```text
http://localhost:8501
```

The dashboard includes:

- New service request submission
- Predicted team display
- Predicted team distribution
- Model performance metrics
- Classification report
- Confusion matrix
- API connection status

## 12. Limitations

- The model predicts historical initial bot-assigned teams, not necessarily the final team that resolved the request.
- Prediction quality depends on the quality and coverage of historical data.
- Ambiguous requests may require employee review.
- The displayed request details are not independent explanations of the model's decisions.
- The validation score does not guarantee the same performance on future requests.
- The application has not yet been demonstrated in a live production environment.

## 13. Cost and Deployment

The application uses locally installed open-source Python libraries and a locally trained model.

No paid AI API key is required for inference.

The prototype runs locally using FastAPI and Streamlit. Production deployment would require additional hosting, monitoring, access control, and operational testing.

## 14. Conclusion

The project demonstrates an end-to-end machine learning workflow for home appliance service request classification, including data preprocessing, model training, evaluation, API development, and an employee-facing dashboard.