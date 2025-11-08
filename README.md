# 🌱 CarbonSwap: Plantation Location Recommendation API

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
Reflect how effectively a project operates and grows.
| Feature | Description | Weight |
|----------|--------------|--------|
| `pohon_tertanam` | Number of trees planted | 0.10 |
| `luas_ha` | Total plantation area (ha) | 0.15 |
| `annual_survival_rate` | Tree survival rate (%) | 0.20 |
| `jumlah_ulasan` | Number of reviews | 0.05 |

#### 2. 🌾 Socio-Ecological Indicators
Represent both ecological and social sustainability impacts.
| Feature | Description | Weight |
|----------|--------------|--------|
| `karbon_terserap` | CO₂ absorbed (tons/year) | 0.25 |
| `orang_terlibat` | Number of people involved | 0.08 |
| `rating_ulasan` | Average project rating | 0.07 |
| `nlp_sentiment` | Sentiment score from feedback (via NLP) | 0.10 |

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
    "nama_lokasi": "Mangrove Bali",
    "jenis_bibit": "Rhizophora sp.",
    "skor_total": 0.8743
  },
  {
    "nama_lokasi": "Hutan Kalimantan",
    "jenis_bibit": "Meranti",
    "skor_total": 0.8421
  }
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
```
nama_lokasi,jenis_bibit,luas_ha,total_kampanye,orang_terlibat,pohon_tertanam,karbon_terserap,annual_survival_rate,mean_annual_temp,mean_annual_precip,soil_ph,rating_ulasan,jumlah_ulasan,ulasan_1,ulasan_2,ulasan_3,ulasan_4
```

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
