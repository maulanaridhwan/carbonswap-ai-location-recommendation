# 🌱 CarbonSwap: Plantation Location Recommendation API
https://carbonswap-ai-location-recommendation.onrender.com/recommendations/

**CarbonSwap** is a digital marketplace that connects companies with credible NGOs to simplify the carbon offset process in a transparent and measurable way.

This repository contains one of CarbonSwap's AI-powered features — the **Plantation Location Recommendation API**.  
It provides intelligent insights for users to identify and rank plantation sites based on both **project performance** and **socio-ecological indicators**.  
This API will be deployed on **Render** and integrated into the **CarbonSwap Web Dashboard**.

---

## 🏢 CarbonSwap Overview

### 🌍 Problem Statement

Between **2019 and 2023**, greenhouse gas emissions from Indonesia's industry sector increased by **16.25%**, while household emissions rose by only **1.29%**.  
Globally, CO₂ concentration growth rates have tripled since the 1960s — from **0.8 ppm/year** to **2.4 ppm/year (2011–2020)**, with **a record 3.5 ppm increase in 2023–2024**.

Indonesia holds vast potential to offset emissions through its forests and conservation areas, capable of absorbing **hundreds of billions of tons of CO₂**.  
However, companies still face major challenges in fulfilling ESG (Environmental, Social, and Governance) responsibilities:

1. Difficulty identifying credible NGOs.  
2. Lack of transparency and real-time monitoring.  
3. Complex ESG compliance and documentation.

---

## 💡 Solution: The CarbonSwap Platform

**CarbonSwap** solves these challenges by connecting companies and NGOs in a transparent, measurable, and secure marketplace.

### 🔑 Key Features
- **Verified NGO Partnerships** — connect companies with trusted carbon offset projects.  
- **AI-Powered Insights (Swappy AI)** — suggest impactful projects and predict carbon outcomes.  
- **Proof of Ownership** — every transaction generates a verifiable ownership certificate.  
- **Bank Collaboration (Custodian Role)** — ensures secure and transparent fund flow.  
- **Real-Time Monitoring & Certification** — automatic ESG reporting with project progress tracking.

### 👥 Target Users
- **Buyers (Companies):** seeking reliable carbon offset projects.  
- **Sellers (NGOs):** offering verified emission reduction projects.

---

## 🌿 Plantation Location Recommendation Feature

This API is designed as an **insight feature** within CarbonSwap to recommend the best plantation locations based on multiple performance and ecological factors.  
It analyzes project data from `location.csv` and ranks each location using a weighted scoring system.

### 🧩 Indicator Categories

#### 1. 🌳 Project Performance Indicators
These measure the operational success of plantation projects.
| Feature                | Description                      | Weight |
| ---------------------- | -------------------------------- | ------ |
| `trees_planted`        | Total number of trees planted    | 0.10   |
| `area_ha`              | Total plantation area (hectares) | 0.15   |
| `annual_survival_rate` | Tree survival rate (%)           | 0.20   |
| `review_count`         | Number of user reviews           | 0.05   |

#### 2. 🌾 Socio-Ecological Indicators
These reflect sustainability, social engagement, and environmental impact.
| Feature           | Description                             | Weight |
| ----------------- | --------------------------------------- | ------ |
| `carbon_absorbed` | CO₂ absorbed annually (tons)            | 0.25   |
| `people_involved` | Number of people engaged in the project | 0.08   |
| `review_rating`   | Average user rating                     | 0.07   |
| `nlp_sentiment`   | Sentiment score from project feedback   | 0.10   |


All indicators are normalized and combined to generate an **overall environmental performance score**.

---

## ⚙️ API Overview

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/recommendations` | GET | Returns top recommended plantation sites by score |
| `/ranked_full` | GET | Returns full ranked dataset |

### Example Response
```json
[
  {
        "locName": "Bedono Village",
        "locDesc": "A coastal area highly vulnerable to rising sea levels and land subsidence, making mangrove restoration vital to protect the shoreline and local communities.",
        "locImage": "https://storage.googleapis.com/restor2-prod-1-sites/fa58735c-b79b-4e1c-9695-420e6236f53f/fe0d5f7e-eb3f-4145-b72e-8ff3fe55cb13/600",
        "province": "JawaTengah",
        "treeType": "Mangrove Rhizopora",
        "score": 0.7255
    },
    {
        "locName": "Trimulyo Coast",
        "locDesc": "A coastal green area that serves as a natural barrier protecting the northern coast and the city from flooding. However, severe abrasion, tidal flooding, and waste pollution have increasingly threatened the area.",
        "locImage": "https://storage.googleapis.com/restor2-prod-1-sites/84a10553-69b7-4189-b424-cde450ac7ef7/b06cac74-eccf-435b-8ae8-55f99dfc4596/600",
        "province": "JawaTengah",
        "treeType": "Mangrove Rhizopora",
        "score": 0.5417
    },
]
```

---

## 🧠 Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,fastapi,pytorch,sklearn,react,git,github" />
</p>

| Category | Tools |
|-----------|--------|
| **Backend** | FastAPI, Uvicorn |
| **AI/NLP** | Hugging Face Transformers, PyTorch, SentencePiece |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Deployment** | Render, Koyeb, Railway |
| **Frontend Integration** | React (CarbonSwap Dashboard) |
| **Versioning** | Git, GitHub |

---

## 🧰 Installation & Local Development

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/carbonswap-plantation-recommendation.git
cd carbonswap-plantation-recommendation
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run locally
```bash
uvicorn app:app --reload
```
Visit: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** → Swagger UI

---

## 📊 Dataset

`location.csv` — contains plantation project data with these columns:
| Column                 | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `id`                   | Unique identifier                                  |
| `location_name`        | Name of plantation site                            |
| `province`             | Province where the project is located              |
| `description`          | Project and ecosystem summary                      |
| `seed_type`            | Tree species used in the project                   |
| `area_ha`              | Total plantation area (hectares)                   |
| `total_campaigns`      | Total number of campaigns run                      |
| `people_involved`      | Number of participants or local community members  |
| `trees_planted`        | Number of trees planted                            |
| `carbon_absorbed`      | Annual CO₂ absorption (tons)                       |
| `annual_survival_rate` | Tree survival rate (%)                             |
| `mean_annual_temp`     | Average annual temperature (°C)                    |
| `mean_annual_precip`   | Average annual precipitation (mm)                  |
| `soil_ph`              | Average soil pH value                              |
| `review_rating`        | Average review score                               |
| `review_count`         | Total number of reviews                            |
| `image`                | Project image URL                                  |
| `review_1`–`review_4`  | Project feedback (used for NLP sentiment analysis) |

---

## 🔍 References
1. Badan Pusat Statistik Indonesia (BPS)  
2. World Meteorological Organization (WMO)  
3. Indonesia Carbon Market Whitepaper  
4. Restor.eco

---

## 🎯 Vision
> “To make carbon offset accessible, transparent, and measurable — empowering Indonesian companies to meet their ESG goals while restoring nature.”

---

**Developed with 💚 by the CarbonSwap Team**
