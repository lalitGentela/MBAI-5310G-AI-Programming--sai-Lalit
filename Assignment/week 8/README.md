1. Business Overview

Lakeside Hotel Group is a multi‑city hospitality brand offering family rooms, suites, conference accommodations, and business travel services. The company receives hundreds of guest reviews across platforms such as Google Reviews, TripAdvisor, Booking.com, and email surveys. These reviews contain valuable feedback about guest experiences, but manually reading and categorizing them is time‑consuming and inconsistent.

To improve service quality and operational efficiency, the hotel wants to automatically classify each guest review into one of its core service departments:

Room Cleanliness

Check‑In Experience

Breakfast Service

Staff Service

Location

Booking Issues

| Business Element | Description |
| --- | --- |
| Company Name | Lakeside Hotel Analytics |
| Business Area | Hospitality, Guest Experience, Service Quality |
| Business Decision | Identify which department each guest review refers to |
| ML Task | Multi‑class text classification using NLP |
| Target Variable | ServiceArea |
| Goal | Automatically classify reviews to support faster issue resolution and service improvement |

2. Business Problem

Guest reviews contain critical information about operational strengths and weaknesses. However, the hotel faces several challenges:

| Business Challenge | Why It Matters | Possible Business Action |
| --- | --- | --- |
| Manual review of feedback is slow | Important complaints may be missed | Automate classification to route issues instantly |
| Mixed topics in reviews | Hard to identify which department is responsible | Assign each review to the correct service area |
| Repeated complaints in certain areas | Indicates systemic service failures | Prioritize staff training or operational fixes |
| High volume of reviews | Staff cannot read everything | Use NLP to summarize and categorize feedback |
| Multi‑platform reviews | Data is scattered | Centralize insights for management |


3. Why Text Classification?

Unlike numeric datasets, guest reviews are unstructured text. NLP (Natural Language Processing) allows the hotel to convert this text into meaningful insights.

| Concept | Numeric ML | NLP Text Classification (This Assignment) |
| --- | --- | --- |
| Input | Numbers | Raw text reviews |
| Goal | Predict numeric/label outcome | Predict service area from text |
| Algorithm | Logistic Regression, Decision Tree | Logistic Regression with TF‑IDF |
| Output | Numeric prediction | Service area classification |

4. Dataset & Columns Used

The dataset contains 120 guest reviews. Each row represents one review submitted by a hotel guest.

| Column | Meaning |
| --- | --- |
| GuestReview | Raw text written by the guest |
| ServiceArea | Target label (6 categories) |
| ReviewChannel | Source of review (Google, TripAdvisor, etc.) |
| HotelCity | City of the hotel |
| RoomType | Type of room booked |
| GuestType | Family, Business, Solo, etc. |
| Rating | Guest rating from 1–5 |

5. Notebook Workflow

5.1 Data Loading & Inspection

The dataset is loaded using pandas. Initial inspection includes:

df.shape

df.columns

df.head()

df.isnull().sum()

df['ServiceArea'].value_counts()

This helps understand the structure and distribution of service categories.

5.2 Text Preprocessing

A cleaned version of the review text is created using:

Lowercasing

Removing punctuation

Removing stopwords

Tokenization

Lemmatization

Creating a CleanedReview column

This step ensures the text is standardized and ready for feature extraction.

5.3 Exploratory Text Analysis

Word frequency analysis is performed using:

Counter()

Bar charts of most common words

Insights include:

Frequent mentions of room, breakfast, staff, location, reservation

Negative words like dirty, cold, waited

Positive words like helpful, fresh, professional

This confirms that reviews naturally align with the six service areas.

5.4 POS Tagging & Named Entity Recognition

Using NLTK:

POS Tagging identifies nouns, verbs, adjectives

NER identifies names, locations, dates, and organizations

Examples:

PERSON: Michael, Anna

GPE: Toronto, Niagara Falls

DATE: July 2

ORG: TripAdvisor, Booking.com

These insights help understand guest sentiment and operational context.

5.5 Feature Extraction (TF‑IDF)

TF‑IDF Vectorizer converts cleaned text into numerical features.

Vocabulary size: 2000 words

Output: Sparse matrix representing word importance

TF‑IDF highlights meaningful words like dirty, reservation, breakfast, location.

5.6 Train-Test Split

The dataset is split into:

80% training

20% testing

Stratified by service area to maintain class balance

5.7 Model Training — Logistic Regression

Logistic Regression is used because:

It handles high‑dimensional TF‑IDF data well

It supports multi‑class classification

It is fast and interpretable

6. Model Evaluation

6.1 Accuracy

Accuracy measures how many reviews were correctly classified.

6.2 Classification Report

Shows precision, recall, and F1‑score for each service area.

Typical findings:

Location and Breakfast perform best

Booking_Issue and Staff_Service have lower scores due to overlapping vocabulary

6.3 Confusion Matrix

| Actual | Predicted | Reason |
| --- | --- | --- |
| Staff_Service | Check_In | Both involve staff interactions |
| Booking_Issue | Check_In | Booking problems occur at arrival |
| Room_Cleanliness | Breakfast | Words like “dirty” appear in both |

7. Business Interpretation of Model Results

| Insight | Business Meaning |
| --- | --- |
| Frequent cleanliness complaints | Housekeeping needs improvement |
| Check‑in delays mentioned often | Front desk staffing or process issues |
| Breakfast complaints about empty items | Food service needs better replenishment |
| Staff names appear in reviews | Identify high-performing or low-performing employees |
| Booking issues repeated | Reservation system or third‑party platforms need review |

The model helps the hotel:

Route complaints to the correct department

Identify recurring operational issues

Improve service quality

Enhance guest satisfaction

Reduce negative reviews online

8. Limitations

Dataset is small (120 reviews)

Some reviews are short and lack detail

Overlapping vocabulary reduces classification accuracy

TF‑IDF does not capture deep semantic meaning

More advanced models (BERT, LSTM) could improve performance

Reviews may contain bias or emotional exaggeration

Human oversight is required for sensitive feedback

# Repository Structure

     Week8/
     ├── AI_Assignment_8.pdf
     ├── NLP_Dataset_5_Hotel_Guest_Service_Area.xlsx
     └── README.md






