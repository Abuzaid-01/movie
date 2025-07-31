"""
Pydantic models for the Movie Recommendation API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MovieRecommendationRequest(BaseModel):
    """Request model for getting movie recommendations"""
    movie_title: str = Field(..., description="Title of the movie to base recommendations on")
    num_recommendations: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")

class MovieInfo(BaseModel):
    """Model for movie information"""
    title: str = Field(..., description="Movie title")
    overview: str = Field(..., description="Movie plot summary")
    rating: float = Field(..., description="Movie rating (0-10)")
    poster_path: Optional[str] = Field(None, description="Path to movie poster")
    similarity_score: Optional[float] = Field(None, description="Similarity score (0-1)")

class MovieRecommendationResponse(BaseModel):
    """Response model for movie recommendations"""
    selected_movie: str = Field(..., description="The movie used for recommendations")
    recommendations: List[MovieInfo] = Field(..., description="List of recommended movies")
    total_found: int = Field(..., description="Total number of recommendations found")

class SearchMoviesRequest(BaseModel):
    """Request model for searching movies"""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    min_rating: Optional[float] = Field(None, ge=0, le=10, description="Minimum rating filter")

class SearchMoviesResponse(BaseModel):
    """Response model for movie search"""
    query: str = Field(..., description="Original search query")
    results: List[MovieInfo] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")

class MovieDetailsResponse(BaseModel):
    """Response model for movie details"""
    title: str = Field(..., description="Movie title")
    overview: str = Field(..., description="Movie plot summary")
    rating: float = Field(..., description="Movie rating (0-10)")
    poster_path: Optional[str] = Field(None, description="Path to movie poster")

class RandomMoviesResponse(BaseModel):
    """Response model for random movies"""
    movies: List[MovieInfo] = Field(..., description="List of random movies")
    count: int = Field(..., description="Number of movies returned")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    total_movies: Optional[int] = Field(None, description="Total number of movies in database")

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
