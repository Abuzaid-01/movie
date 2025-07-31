"""
Movie Recommender Backend with Hugging Face Integration
"""

import os
import pickle
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceMovieRecommender:
    """
    Movie Recommendation System with Hugging Face model loading
    """
    
    def __init__(self, repo_id: str = "Abuzaid01/movi_recommender_system"):
        """
        Initialize the recommender with Hugging Face repo
        
        Args:
            repo_id: Hugging Face repository ID containing the models
        """
        self.repo_id = repo_id
        self.movie_data = None
        self.movie_info = None
        self.similarity_matrix = None
        self.count_vectorizer = None
        self.models_loaded = False
        self.data_dir = "models"
        
        # Create models directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_models_from_hf(self) -> bool:
        """
        Download models from Hugging Face repository
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading models from Hugging Face repo: {self.repo_id}")
            
            # List of model files to download
            model_files = [
                "movie_data.pkl",
                "movie_info.pkl", 
                "similarity_matrix.pkl",
                "count_vectorizer.pkl"
            ]
            
            for file_name in model_files:
                logger.info(f"Downloading {file_name}...")
                
                # Download file from Hugging Face
                downloaded_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file_name,
                    cache_dir=self.data_dir,
                    local_dir=self.data_dir,
                    local_dir_use_symlinks=False
                )
                
                logger.info(f"Downloaded {file_name} to {downloaded_path}")
            
            logger.info("All models downloaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading models from Hugging Face: {str(e)}")
            return False
    
    def load_models(self) -> bool:
        """
        Load all models and data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Download models if not already present
            model_files = [
                os.path.join(self.data_dir, "movie_data.pkl"),
                os.path.join(self.data_dir, "movie_info.pkl"),
                os.path.join(self.data_dir, "similarity_matrix.pkl"),
                os.path.join(self.data_dir, "count_vectorizer.pkl")
            ]
            
            # Check if all files exist
            missing_files = [f for f in model_files if not os.path.exists(f)]
            
            if missing_files:
                logger.info("Some model files are missing. Downloading from Hugging Face...")
                if not self.download_models_from_hf():
                    return False
            
            logger.info("Loading movie data...")
            with open(os.path.join(self.data_dir, "movie_data.pkl"), 'rb') as f:
                self.movie_data = pickle.load(f)
            
            logger.info("Loading movie info...")
            with open(os.path.join(self.data_dir, "movie_info.pkl"), 'rb') as f:
                self.movie_info = pickle.load(f)
            
            logger.info("Loading similarity matrix...")
            with open(os.path.join(self.data_dir, "similarity_matrix.pkl"), 'rb') as f:
                self.similarity_matrix = pickle.load(f)
            
            logger.info("Loading count vectorizer...")
            with open(os.path.join(self.data_dir, "count_vectorizer.pkl"), 'rb') as f:
                self.count_vectorizer = pickle.load(f)
            
            self.models_loaded = True
            logger.info(f"All models loaded successfully! Total movies: {len(self.movie_data)}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
            return False
    
    def get_movie_recommendations(self, movie_title: str, num_recommendations: int = 5) -> List[Dict]:
        """
        Get movie recommendations based on a given movie title
        
        Args:
            movie_title: Title of the movie to base recommendations on
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with details
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            # Find the movie in the dataset
            movie_matches = self.movie_data[
                self.movie_data['title'].str.contains(movie_title, case=False, na=False)
            ]
            
            if movie_matches.empty:
                # Try partial matching
                movie_matches = self.movie_data[
                    self.movie_data['title'].str.lower().str.contains(
                        movie_title.lower(), na=False
                    )
                ]
            
            if movie_matches.empty:
                return []
            
            # Get the first match
            selected_movie = movie_matches.iloc[0]
            movie_index = selected_movie.name
            
            # Get similarity scores
            similarity_scores = list(enumerate(self.similarity_matrix[movie_index]))
            
            # Sort by similarity (excluding the movie itself)
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]
            
            # Get top recommendations
            recommendations = []
            for i, (idx, score) in enumerate(similarity_scores[:num_recommendations]):
                movie = self.movie_data.iloc[idx]
                
                recommendation = {
                    'title': movie['title'],
                    'overview': movie.get('overview', 'No overview available'),
                    'rating': float(movie.get('vote_average', 0.0)),
                    'poster_path': self._get_poster_url(movie.get('poster_path')),
                    'similarity_score': float(score)
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations for '{movie_title}': {str(e)}")
            return []
    
    def search_movies(self, query: str, limit: int = 10, min_rating: Optional[float] = None) -> List[Dict]:
        """
        Search movies by title or content
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_rating: Minimum rating filter
            
        Returns:
            List of matching movies
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            # Search in titles and overviews
            title_matches = self.movie_data[
                self.movie_data['title'].str.contains(query, case=False, na=False)
            ]
            
            overview_matches = self.movie_data[
                self.movie_data['overview'].str.contains(query, case=False, na=False)
            ]
            
            # Combine and remove duplicates
            all_matches = pd.concat([title_matches, overview_matches]).drop_duplicates()
            
            # Apply rating filter
            if min_rating is not None:
                all_matches = all_matches[all_matches['vote_average'] >= min_rating]
            
            # Sort by rating
            all_matches = all_matches.sort_values('vote_average', ascending=False)
            
            # Limit results
            all_matches = all_matches.head(limit)
            
            results = []
            for _, movie in all_matches.iterrows():
                result = {
                    'title': movie['title'],
                    'overview': movie.get('overview', 'No overview available'),
                    'rating': float(movie.get('vote_average', 0.0)),
                    'poster_path': self._get_poster_url(movie.get('poster_path'))
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching movies with query '{query}': {str(e)}")
            return []
    
    def get_random_movies(self, count: int = 10) -> List[Dict]:
        """
        Get random movies
        
        Args:
            count: Number of random movies to return
            
        Returns:
            List of random movies
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            # Get random sample
            random_movies = self.movie_data.sample(n=min(count, len(self.movie_data)))
            
            results = []
            for _, movie in random_movies.iterrows():
                result = {
                    'title': movie['title'],
                    'overview': movie.get('overview', 'No overview available'),
                    'rating': float(movie.get('vote_average', 0.0)),
                    'poster_path': self._get_poster_url(movie.get('poster_path'))
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting random movies: {str(e)}")
            return []
    
    def get_movie_details(self, movie_title: str) -> Optional[Dict]:
        """
        Get details for a specific movie
        
        Args:
            movie_title: Title of the movie
            
        Returns:
            Movie details or None if not found
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            movie_matches = self.movie_data[
                self.movie_data['title'].str.contains(movie_title, case=False, na=False)
            ]
            
            if movie_matches.empty:
                return None
            
            movie = movie_matches.iloc[0]
            
            return {
                'title': movie['title'],
                'overview': movie.get('overview', 'No overview available'),
                'rating': float(movie.get('vote_average', 0.0)),
                'poster_path': self._get_poster_url(movie.get('poster_path'))
            }
            
        except Exception as e:
            logger.error(f"Error getting movie details for '{movie_title}': {str(e)}")
            return None
    
    def get_all_movie_titles(self) -> List[str]:
        """
        Get all available movie titles
        
        Returns:
            List of all movie titles
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        return self.movie_data['title'].tolist()
    
    def get_total_movies(self) -> int:
        """
        Get total number of movies in the database
        
        Returns:
            Total number of movies
        """
        if not self.models_loaded:
            return 0
        
        return len(self.movie_data)
    
    def _get_poster_url(self, poster_path: Optional[str]) -> Optional[str]:
        """
        Generate full poster URL
        
        Args:
            poster_path: Relative poster path
            
        Returns:
            Full poster URL or None
        """
        if poster_path and pd.notna(poster_path):
            base_url = "https://image.tmdb.org/t/p/w500"
            return f"{base_url}{poster_path}"
        return None

# Global recommender instance
recommender = HuggingFaceMovieRecommender()

def get_recommender() -> HuggingFaceMovieRecommender:
    """
    Get the global recommender instance
    
    Returns:
        HuggingFaceMovieRecommender instance
    """
    return recommender
