🧾 1. Project Overview
This project builds a complete end‑to‑end Natural Language Processing (NLP) and Machine Learning pipeline to classify hotel guest reviews into specific service areas. The goal is to help hotel management automatically understand what guests are talking about—whether it’s Room Cleanliness, Check‑In, Breakfast, Staff Service, Location, or Booking Issues.

The workflow follows the same structure as a real‑world machine learning project:

Understanding the business problem

Loading and inspecting the dataset

Text preprocessing

Exploratory text analysis

POS tagging and Named Entity Recognition

Feature extraction using TF‑IDF

Building a classification model

Evaluating the model

Business interpretation and insights

This README explains each step in detail so anyone can understand the full pipeline.

🏨 2. Business Problem
Lakeside Hotel Group collects guest reviews from multiple platforms such as Google Reviews, TripAdvisor, Booking.com, and email surveys. These reviews contain valuable information about guest experiences, but manually reading and categorizing them is time‑consuming.

The hotel wants to automatically classify each review into one of six service areas:

Room_Cleanliness

Check_In

Breakfast

Staff_Service

Location

Booking_Issue

This classification helps hotel managers:

Identify which departments receive the most complaints

Detect recurring operational issues

Improve service quality

Prioritize staff training

Enhance overall guest satisfaction

This is a multi‑class text classification problem, where the goal is to predict the service area based on the review text.

📂 3. Dataset Description
The dataset contains 120 hotel guest reviews with the following columns:

GuestReview → The raw text written by the guest

ServiceArea → The target label (6 categories)

ReviewChannel → Source of the review

HotelCity → City of the hotel

RoomType, GuestType, Rating → Additional metadata

For this project:

GuestReview is used as the main text feature

ServiceArea is the target variable

🧹 4. Text Preprocessing
To prepare the text for machine learning, several NLP preprocessing steps were applied:

✔️ Lowercasing
All text was converted to lowercase to ensure consistency.

✔️ Removing punctuation
Symbols like “!”, “?”, “#”, and commas were removed.

✔️ Tokenization
Text was split into individual words using NLTK.

✔️ Stopword removal
Common words like the, is, in, at were removed because they do not add meaning.

✔️ Lemmatization
Words were reduced to their base form (e.g., cleaning → clean).

✔️ CleanedReview column
A new column was created containing the cleaned text.

These steps reduce noise and help the model focus on meaningful words.

🔍 5. Exploratory Text Analysis
Before modeling, the text was analyzed to understand common patterns.

✔️ Most frequent words
Words like room, breakfast, staff, location, reservation, dirty, wait, key appeared frequently.

✔️ Interpretation
Frequent mentions of dirty, towel, floor → cleanliness concerns

Words like waited, key, check → check‑in issues

Words like coffee, buffet, eggs → breakfast service

Words like location, walking, near → location satisfaction

Words like reservation, confirmation → booking problems

This confirms that the dataset naturally aligns with the six service areas.

🧠 6. POS Tagging & Named Entity Recognition (NLTK)
Using NLTK:

✔️ POS Tagging
Identified nouns, verbs, and adjectives:

Nouns: room, breakfast, staff, reservation, location

Verbs: waited, ignored, charged, handled

Adjectives: dirty, cold, noisy, helpful

These reveal what guests talk about and how they describe their experience.

✔️ Named Entity Recognition
Detected:

People: Michael, Anna

Locations: Toronto, Niagara Falls, Halifax

Dates: July 2, June 10

Organizations: TripAdvisor, Booking.com

These entities help identify patterns across cities, staff members, and platforms.

🔢 7. Feature Extraction (TF‑IDF)
The cleaned text was converted into numerical features using TF‑IDF (Term Frequency–Inverse Document Frequency).

TF‑IDF highlights important words in each review

A vocabulary of 2000 features was created

The output is a sparse matrix used for model training

TF‑IDF is ideal for text classification because it emphasizes meaningful words like dirty, reservation, breakfast, location.

🤖 8. Model Building (Logistic Regression)
A Logistic Regression model was trained using:

X_train → TF‑IDF features

y_train → ServiceArea labels

Logistic Regression was chosen because:

It works well with high‑dimensional TF‑IDF data

It supports multi‑class classification

It is fast and interpretable

The model learned patterns in the text to predict the correct service area.

📊 9. Model Evaluation
The model was evaluated using:

✔️ Accuracy
Shows how many reviews were correctly classified.

✔️ Classification Report
Provides precision, recall, and F1‑score for each service area.

✔️ Confusion Matrix
Shows which categories were confused with each other.

Common correct predictions:
Location (clear keywords like “walking”, “near”)

Breakfast (words like “coffee”, “buffet”, “eggs”)

Common misclassifications:
Staff_Service ↔ Check_In

Booking_Issue ↔ Check_In

Room_Cleanliness ↔ Breakfast

These errors occur when reviews mention multiple topics or ambiguous wording.

💼 10. Business Interpretation
✔️ What the model predicts
The model predicts which hotel department a guest is referring to in their review.

✔️ Why this is useful
Helps managers identify problem areas

Reduces manual review time

Supports data‑driven decision making

Improves guest satisfaction by addressing recurring issues

✔️ Key insight from the data
Guests frequently mention:

Staff behavior

Check‑in delays

Breakfast quality

Room cleanliness

These areas strongly influence guest satisfaction.

⚠️ 11. Limitations
Small dataset (120 reviews)

Some reviews are short and lack detail

Overlapping vocabulary between categories

TF‑IDF does not capture deep semantic meaning

A larger dataset or transformer models (e.g., BERT) would improve accuracy

🏁 12. Conclusion
This project demonstrates a complete NLP and machine learning pipeline for classifying hotel guest reviews. By combining text preprocessing, TF‑IDF feature extraction, and Logistic Regression, the model can automatically categorize reviews into meaningful service areas. This helps hotel management quickly identify operational issues and improve service quality.
