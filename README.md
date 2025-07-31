# 🎬 Movie Recommendation Backend

Simple FastAPI backend for movie recommendations. Models are loaded from Hugging Face.

## 🚀 Quick Deploy to Render

1. **Push to GitHub:**
```bash
git add backend/
git commit -m "Add movie recommendation backend"
git push origin main
```

2. **Deploy on Render:**
- Create new **Web Service**
- Connect your GitHub repo
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python -m uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Environment:** Python 3.11

3. **Get your API URL:**
- After deployment: `https://your-app-name.onrender.com`
- Test: `https://your-app-name.onrender.com/health`

## 📝 API Endpoints

- `GET /health` - Check if API is working
- `POST /recommendations` - Get movie recommendations
- `POST /search` - Search movies
- `GET /random` - Get random movies

## 🤗 Hugging Face Models

Models auto-download from: `Abuzaid01/movi_recommender_system`
- movie_data.pkl
- movie_info.pkl  
- similarity_matrix.pkl
- count_vectorizer.pkl

## 🧪 Test Locally

```bash
pip install -r requirements.txt
python app.py
```

API will run at: http://localhost:8000
