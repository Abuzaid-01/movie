"""
Movie Recommendation Backend API
FastAPI backend that serves movie recommendations using Hugging Face models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import logging

from models import (
    MovieRecommendationRequest, 
    MovieRecommendationResponse,
    SearchMoviesRequest,
    SearchMoviesResponse,
    MovieDetailsResponse,
    RandomMoviesResponse
)
from recommender import get_recommender, HuggingFaceMovieRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🎬 Movie Recommendation API",
    description="AI-powered movie recommendation system backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender (will download models from Hugging Face)
recommender = None

@app.on_event("startup")
async def startup_event():
    """Initialize the recommender when the app starts"""
    global recommender
    logger.info("🚀 Starting Movie Recommendation API...")
    logger.info("📥 Loading models from Hugging Face...")
    recommender = get_recommender()
    recommender.load_models()
    logger.info("✅ API ready!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎬 Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global recommender
    if recommender is None or not recommender.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "total_movies": recommender.get_total_movies()
    }

@app.post("/recommendations", response_model=MovieRecommendationResponse)
async def get_recommendations(request: MovieRecommendationRequest):
    """Get movie recommendations based on a selected movie"""
    global recommender
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        logger.info(f"🎯 Getting recommendations for: {request.movie_title}")
        
        recommendations = recommender.get_movie_recommendations(
            request.movie_title, 
            request.num_recommendations
        )
        
        if recommendations is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Movie '{request.movie_title}' not found"
            )
        
        return MovieRecommendationResponse(
            selected_movie=request.movie_title,
            recommendations=recommendations,
            total_found=len(recommendations)
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchMoviesResponse)
async def search_movies(request: SearchMoviesRequest):
    """Search for movies by title or keywords"""
    global recommender
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        logger.info(f"🔍 Searching for: {request.query}")
        
        results = recommender.search_movies(request.query, request.limit)
        
        # Apply minimum rating filter if specified
        if request.min_rating is not None:
            results = [
                movie for movie in results 
                if movie['rating'] >= request.min_rating
            ]
        
        return SearchMoviesResponse(
            query=request.query,
            results=results,
            total_found=len(results)
        )
        
    except Exception as e:
        logger.error(f"❌ Error searching movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movie/{movie_title}", response_model=MovieDetailsResponse)
async def get_movie_details(movie_title: str):
    """Get detailed information about a specific movie"""
    global recommender
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        logger.info(f"📽️ Getting details for: {movie_title}")
        
        movie_details = recommender.get_movie_details(movie_title)
        
        if movie_details is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Movie '{movie_title}' not found"
            )
        
        return MovieDetailsResponse(**movie_details)
        
    except Exception as e:
        logger.error(f"❌ Error getting movie details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/random", response_model=RandomMoviesResponse)
async def get_random_movies(count: int = 6):
    """Get random movies for discovery"""
    global recommender
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        logger.info(f"🎲 Getting {count} random movies")
        
        random_movies = recommender.get_random_movies(count)
        
        return RandomMoviesResponse(
            movies=random_movies,
            count=len(random_movies)
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting random movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies/titles")
async def get_all_movie_titles():
    """Get all movie titles for autocomplete"""
    global recommender
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        titles = recommender.get_all_movie_titles()
        return {
            "titles": titles,
            "total_count": len(titles)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting movie titles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
