# 💰 Financial AI Agent - Your Personal Stock Market Analyst  

## 📌 Project Overview  
The **Financial AI Agent** is an **intelligent stock analysis tool** that provides real-time financial insights using **machine learning, web search, and stock market APIs**. It combines **Google Gemini AI, YFinance API, and DuckDuckGo Search** to deliver **stock price analysis, analyst recommendations, and latest financial news**.

✅ **Enter a stock ticker (e.g., AAPL, NVDA), and the AI agent provides:**  
- 🔹 **Current stock price & key financial metrics.**  
- 🔹 **Analyst recommendations (Buy/Hold/Sell).**  
- 🔹 **Latest news & trends from the internet.**  
- 🔹 **Structured insights with tables & key takeaways.**  

This AI-powered tool **empowers investors** with **quick, data-driven financial decisions.**  

---

## 📊 Technologies Used  
- **Python** (Backend)  
- **Streamlit** (Web App Interface)  
- **Google Gemini Pro API** (LLM-powered financial insights)  
- **YFinance API** (Real-time stock data & fundamentals)  
- **DuckDuckGo API** (Fetching latest financial news)  
- **Phidata AI Framework** (Agent-based multi-AI system)  
- **Hugging Face Cloud** (Deployment platform)  
- **Dotenv** (Environment variable management)  

---

## ⚙️ Installation & Setup  

### **1️⃣ Clone the repository**  
```bash
git clone https://github.com/yourusername/Financial-AI-Agent.git
cd Financial-AI-Agent
```
### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set up API Keys**
* Obtain a Google Gemini API Key from Google AI Studio
* Add the API key to a .env file in the project directory:
```bash
GOOGLE_API_KEY=your_google_api_key
```

### **4️⃣ Run the Streamlit app**
```bash
streamlit run app.py
```

## **🏗️ Project Workflow**:
1️⃣ User enters a stock ticker (e.g., AAPL, TSLA, NVDA).
2️⃣ The Financial AI Agent performs two key tasks:

* 🔍 Searches the internet for the latest stock news using DuckDuckGo.
* 📊 Fetches stock price, fundamentals, and analyst recommendations using YFinance API.
  
3️⃣ The multi-agent AI system processes & summarizes insights.
4️⃣ Gemini AI generates a structured, easy-to-read financial report.
5️⃣ The report is displayed in the Streamlit web app.

### **🖥️ Usage**:
1. Run the app:
```bash
streamlit run app.py
```

2. Enter a stock ticker (e.g., AAPL, MSFT, GOOG).
3. Click "Analyze Stock" to get insights.
4. View AI-generated stock recommendations & news updates.

## **📊 Sample Output**:

![fin_1](https://github.com/user-attachments/assets/1e9fdddd-cb86-47c2-920d-6cf89ddacfa3)
![fin_2](https://github.com/user-attachments/assets/580ff436-6f85-496c-a41b-055429b89ffc)

## 📩 Contact & Support
📧 Email: vishal1594@outlook.com
🔗 LinkedIn: https://www.linkedin.com/in/vishal-shivnani-87487110a/
