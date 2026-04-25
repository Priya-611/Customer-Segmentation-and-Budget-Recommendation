# Customer-Segmentation-and-Budget-Recommendation

A Machine Learning-based web application that helps users analyze their monthly expenses and provides intelligent budget recommendations.

This project uses **K-Means Clustering** to group spending patterns and suggest improvements to achieve savings goals.

---

## 🚀 Features

* 📊 Analyze monthly expenses
* 🤖 ML-based budget recommendations
* 💡 Smart suggestions to reduce overspending
* 🎯 Check if your savings goal is achievable
* 🌐 Interactive frontend (HTML UI)
* ⚡ Fast backend using Flask API

---

## 🛠️ Tech Stack

* **Frontend**: HTML, CSS, JavaScript *(AI-assisted UI design)* 
* **Backend**: Python, Flask *(self-developed)* 
* **Machine Learning**:

  * K-Means Clustering
  * StandardScaler
  * Pandas, NumPy 

---

## 📂 Project Structure

```
├── data.csv               # datset
├── app.py                 # Flask backend API
├── analysis.py            # Model training script
├── expenses_model.pkl     # Trained ML model
├── index.html             # Frontend UI
└── README.md
```

---

## ⚙️ How It Works

1. User enters:

   * Income
   * Desired savings
   * Monthly expenses

2. Backend:

   * Scales input data
   * Predicts cluster using K-Means
   * Compares user spending with cluster averages

3. Output:

   * Savings feasibility
   * Personalized suggestions to reduce expenses

---

## 🧠 Machine Learning Logic

* Data is standardized using **StandardScaler**
* **K-Means clustering** is applied to group similar spending behaviors
* Optimal clusters selected using **Elbow Method**
* Recommendations are generated based on **cluster centroids**

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <project-folder>
```

### 2. Install dependencies

```bash
pip install flask pandas numpy scikit-learn
```

### 3. Run the Flask app

```bash
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:8000
```

---

## 📸 UI Preview

* Clean modern glassmorphism UI
* User-friendly input fields
* Real-time results with suggestions

---

## 📌 Example Output

* ✅ “Your saving goal is achievable”
* ❌ “Reduce Entertainment from ₹3000 → ₹1800”

---

## ⚠️ Notes

* The ML model is pre-trained and saved as `expenses_model.pkl`
* Ensure correct file paths when running locally
* Input values must be non-negative numbers

---

## 🙌 Credits

* Frontend UI: Generated with AI assistance
* Backend + ML Model: Developed manually

