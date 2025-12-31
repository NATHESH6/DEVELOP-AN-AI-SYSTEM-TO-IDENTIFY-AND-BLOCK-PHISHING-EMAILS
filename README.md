# DEVELOP-AN-AI-SYSTEM-TO-IDENTIFY-AND-BLOCK-PHISHING-EMAILS
Here's a comprehensive `README.md` file for your phishing email detection project, based on the project report:

```markdown
 Phishing Email Detection System

📌 Project Overview
This project implements an intelligent AI-powered system to detect and block phishing emails using machine learning algorithms. The system analyzes email content, URLs, and metadata to classify emails as either phishing or legitimate (ham) with high accuracy.

🎯 Key Features
- Machine Learning Models: Utilizes multiple ML algorithms including XGBoost, Random Forest, SVM, Naive Bayes, and Logistic Regression
- NLP Processing         : Employs Natural Language Processing techniques for text analysis
- Web Interface          : User-friendly Flask-based web application for real-time email analysis
- High Accuracy          : Achieves up to 96.1% accuracy with XGBoost model
- Feature Extraction     : Comprehensive feature engineering including lexical, URL-based, header-based, and content-based features

📊 Performance Metrics

|----------------------------------------------------------|
| Algorithm     | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| XGBoost       | 96.1%    | 95.3%     | 96.9%  | 96.1%    |
| Random Forest | 95.2%    | 94.6%     | 95.8%  | 95.2%    |
| SVM           | 93.0%    | 91.5%     | 93.8%  | 92.6%    |
| Logistic      |          |           |        |          |
| Regression    | 91.1%    | 89.7%     | 92.0%  | 90.8%    |
| Naive Bayes   | 89.5%    | 87.2%     | 91.4%  | 89.2%    |
|----------------------------------------------------------|

🏗️ System Architecture

Modules:
1. Data Input Module        : Receives email content via web interface
2. Preprocessing Module     : Cleans and standardizes email text
3. Feature Extraction Module: Converts text to numerical features using TF-IDF
4. Classification Module    : ML model predicts phishing/legitimate status
5. Evaluation Module        : Monitors performance and logs results
6. User Interface Module    : Displays results to users
7. Deployment Module        : Integrates with email systems

🛠️ Technology Stack
- Programming Language: Python 3.x
- Machine Learning: Scikit-learn, XGBoost, Pandas, NLTK
- Web Framework: Flask
- Frontend: HTML, CSS, JavaScript
- Data Processing: TF-IDF Vectorization, NLP techniques

```
📁 Project Structure

phishing-detection-system/
├── app.py              # Flask application
├── train_model.py      # Model training script
├── model.pkl           # Trained model and vectorizer
├── templates/          # HTML templates
│   ├── home.html       # Home page
│   └── index.html      # Email scanner page
├── static/             # CSS and static files
├── datasets/           # Training datasets
└── requirements.txt    # Python dependencies
```

🚀 Installation & Setup

Prerequisites
- Python 3.7+
- pip package manager

Installation Steps
1. Clone the repository
   ```bash
   git clone <repository-url>
   cd phishing-detection-system
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional - pre-trained model included)
   ```bash
   python train_model.py
   ```

4. Run the application
   ```bash
   python app.py
   ```

5. Access the web interface
   - Open browser and navigate to: `http://127.0.0.1:5000`

📋 Dataset Information
The system is trained on a balanced dataset containing:
- 10,000 Phishing emails** from sources like Phish-Tank
- 10,000 Legitimate emails** from Spam-Assassin and Enron datasets

Features Extracted:
- Lexical Features: Special characters, word length, capitalization
- URL Features    : URL length, IP usage, shortening services
- Content Features: Suspicious keywords, tone analysis
- Header Features : SPF/DKIM status, sender-reply mismatch
- NLP Features    : TF-IDF vectors, BERT embeddings

💻 Usage Instructions

Web Interface:
1. Navigate to the home page
2. Click "Go to Email Scanner"
3. Paste email content in the text area
4. Click "Scan Email"
5. View results: "Phishing Email Detected!" or "Safe Email."

API Usage:
The system can be integrated via REST API endpoints for automated email scanning.

📈 Model Training Process
1. Data Collection    : Gather phishing and legitimate emails
2. Preprocessing      : Clean HTML tags, remove special characters
3. Feature Engineering: Extract relevant features
4. Model Training     : Train multiple ML algorithms
5. Evaluation         : Compare performance using accuracy, precision, recall, F1-score
6. Deployment         : Integrate best-performing model into web application

🔍 Key Algorithms Implemented

1. XGBoost (Primary Model)
- Gradient boosting algorithm with tree ensemble
- Handles imbalanced data effectively
- Provides feature importance scores

2. Random Forest
- Ensemble of decision trees
- Reduces overfitting
- Good for high-dimensional data

3. Support Vector Machine (SVM)
- Effective for text classification
- Works well with TF-IDF features

4. Naive Bayes
- Fast and efficient for text data
- Good baseline model

5. Logistic Regression
- Simple and interpretable
- Good for binary classification

🎨 User Interface Screenshots
1. Home Page    : Welcome screen with navigation to email scanner
2. Email Scanner: Text area for email input and scan button
3. Results Page : Clear indication of phishing detection status

📚 References
1. Sebastian, F. (2002). Machine learning in automated text categorization.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
3. Zhang, Y., & Luo, X. (2021). Phishing email detection using NLP techniques.
4. Scikit-learn Documentation (2023).
5. Flask Documentation (2023).

👥 Team Members
- Angeshkumar P (71052202301)
- Anisuthan S (71052202007)
- Nathesh P V (71052202303)
- Sanjayram S (71052202208)

🏫 Institution
Coimbatore Institute of Engineering and Technology  
Computer Science and Engineering Department  
Coimbatore – 641 109  
May 2025

📄 License
This project is developed as part of academic requirements for Bachelor of Engineering in Computer Science and Engineering.

🤝 Acknowledgments
We express our gratitude to:
- Dr. K. Pushpalatha (HOD, CSE Department)
- Ms. V. Saranya (Project Supervisor)
- All faculty members and staff of CIET
- Open-source community for tools and datasets

---

For more details, refer to the complete project report in `main_project_report.pdf`
```

This README file provides:
1. Comprehensive project overview
2. Clear installation and setup instructions
3. Technical specifications
4. Usage guidelines
5. Performance metrics
6. Team and institutional information
7. References and acknowledgments

The file is structured to be informative for both technical users (developers, researchers) and non-technical stakeholders (instructors, project evaluators).





