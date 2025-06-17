"""
Hệ thống phát hiện câu hỏi trùng lặp sử dụng Sentence Transformers
Tối ưu hiệu suất và bộ nhớ cho xử lý large datasets
"""
from time import time
from datetime import datetime, timedelta
import logging
import numpy as np
import torch
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Set, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
import hashlib
from tqdm import tqdm
import traceback

from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F


# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DuplicateDetectorConfig:
    """Cấu hình cho Duplicate Detector"""
    model_name: str = "AITeamVN/Vietnamese_Embedding_v2"
    max_seq_length: int = 512
    similarity_threshold: float = 0.85
    batch_size: int = 32
    device: str = "auto"
    cache_embeddings: bool = True
    cache_dir: str = "./cache"


@dataclass
class DuplicateAnalysisResult:
    """Kết quả phân tích duplicate"""
    total_questions: int
    duplicate_pairs: List[Tuple[int, int]]
    duplicate_indices: Set[int]
    similarity_matrix: Optional[torch.Tensor] = None
    statistics: Optional[Dict] = None


class DuplicateDetectorError(Exception):
    """Custom exception cho Duplicate Detector"""
    pass


logger = logging.getLogger(__name__)

class QuestionDuplicateDetector:
    """
    Lớp phát hiện câu hỏi trùng lặp sử dụng semantic similarity với caching tối ưu
    """
    
    def __init__(self, config: Optional[DuplicateDetectorConfig] = None):
        """
        Khởi tạo Duplicate Detector
        
        Args:
            config: Cấu hình cho detector
        """
        self.config = config or DuplicateDetectorConfig()
        self.device = self._setup_device()
        self.model = None  # Lazy loading
        self.cache_dir = Path(self.config.cache_dir)
        
        # Enhanced caching features
        self._memory_cache = {}  # In-memory cache for recent embeddings
        self._cache_metadata = {}  # Track cache usage and timestamps
        self._max_memory_cache_size = 100  # Max items in memory cache
        
        if self.config.cache_embeddings:
            self.cache_dir.mkdir(exist_ok=True)
            self._load_cache_metadata()

    def _setup_device(self) -> torch.device:
        """Thiết lập device với kiểm tra bộ nhớ"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_gb = gpu_memory / (1024**3)
                    
                    device = torch.device("cuda")
                    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name()}")
                    logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
                    
                    if gpu_memory_gb < 2.0:
                        logger.warning("GPU memory thấp, có thể gặp vấn đề với model lớn")
                        
                except Exception as e:
                    logger.warning(f"GPU check failed: {e}, falling back to CPU")
                    device = torch.device("cpu")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        else:
            device = torch.device(self.config.device)
            logger.info(f"Sử dụng device: {device}")
        
        return device

    def _load_model(self) -> SentenceTransformer:
        """Load và cấu hình model với error handling cải tiến"""
        if self.model is not None:
            return self.model

        try:
            logger.info(f"Loading model: {self.config.model_name} onto device: {self.device}")
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Try multiple loading strategies
            loading_strategies = [
                lambda: SentenceTransformer(
                    self.config.model_name,
                    device=str(self.device),
                    trust_remote_code=True
                ),
                lambda: self._load_cpu_then_transfer(),
                lambda: self._load_with_fallback()
            ]
            
            model = None
            last_error = None
            
            for i, strategy in enumerate(loading_strategies, 1):
                try:
                    logger.info(f"Trying loading strategy {i}")
                    model = strategy()
                    logger.info(f"Model loaded successfully with strategy {i}")
                    break
                except Exception as e:
                    logger.warning(f"Strategy {i} failed: {e}")
                    last_error = e
                    continue
            
            if model is None:
                raise DuplicateDetectorError(f"All loading strategies failed. Last error: {last_error}")

            # Configure model
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = self.config.max_seq_length
                
            # Ensure model is on correct device
            if hasattr(model, 'device') and str(model.device) != str(self.device):
                logger.info(f"Moving model from {model.device} to {self.device}")
                model = model.to(self.device)

            logger.info(f"Model loaded successfully. Max sequence length: {self.config.max_seq_length}")
            
            self.model = model
            return model

        except Exception as e:
            self._log_detailed_error(e)
            raise DuplicateDetectorError(f"Không thể load model: {str(e)}")

    def _load_cpu_then_transfer(self) -> SentenceTransformer:
        """Load model on CPU first, then transfer to target device"""
        model = SentenceTransformer(self.config.model_name, device='cpu')
        if str(self.device) != 'cpu':
            model = model.to(self.device)
        return model

    def _load_with_fallback(self) -> SentenceTransformer:
        """Load with fallback options"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = SentenceTransformer(self.config.model_name)
        
        try:
            model = model.to(self.device)
        except RuntimeError as e:
            if "meta tensor" in str(e):
                logger.info("Using to_empty to fix meta tensor issue")
                model = model.to_empty(device=self.device)
            else:
                raise e
        
        return model

    def _log_detailed_error(self, e: Exception):
        """Log detailed error information"""
        logger.error("=== DETAILED ERROR INFORMATION ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Device: {self.device}")
        logger.error(f"Model name: {self.config.model_name}")
        
        if torch.cuda.is_available():
            logger.error(f"CUDA device count: {torch.cuda.device_count()}")
            try:
                logger.error(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                logger.error(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            except:
                pass

    def _load_cache_metadata(self):
        """Load cache metadata from disk"""
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self._cache_metadata = json.load(f)
                logger.debug("Cache metadata loaded")
            except Exception as e:
                logger.warning(f"Could not load cache metadata: {e}")
                self._cache_metadata = {}

    def _save_cache_metadata(self):
        """Save cache metadata to disk"""
        metadata_file = self.cache_dir / "cache_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self._cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache metadata: {e}")

    def _generate_cache_key(self, questions: List[str]) -> str:
        """Tạo cache key từ danh sách câu hỏi với thông tin model"""
        # Include model info and config in cache key
        content = "|".join(sorted(questions))
        model_info = f"{self.config.model_name}_{self.config.max_seq_length}_{self.config.batch_size}"
        full_content = f"{model_info}|{content}"
        return hashlib.md5(full_content.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid (not expired)"""
        if not self.config.cache_embeddings:
            return False
            
        metadata = self._cache_metadata.get(cache_key)
        if not metadata:
            return False
            
        # Check if cache has expired (if cache_ttl is set)
        if hasattr(self.config, 'cache_ttl_hours') and self.config.cache_ttl_hours:
            created_time = datetime.fromisoformat(metadata['created_at'])
            expiry_time = created_time + timedelta(hours=self.config.cache_ttl_hours)
            if datetime.now() > expiry_time:
                logger.debug(f"Cache {cache_key} has expired")
                return False
                
        return True

    def _get_from_memory_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """Get embeddings from in-memory cache"""
        if cache_key in self._memory_cache:
            logger.debug("Using embeddings from memory cache")
            return self._memory_cache[cache_key]['embeddings']
        return None

    def _save_to_memory_cache(self, cache_key: str, embeddings: torch.Tensor):
        """Save embeddings to in-memory cache with LRU eviction"""
        # Remove oldest items if cache is full
        while len(self._memory_cache) >= self._max_memory_cache_size:
            oldest_key = min(self._memory_cache.keys(), 
                           key=lambda k: self._memory_cache[k]['access_time'])
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = {
            'embeddings': embeddings.clone().detach(),
            'access_time': time()
        }

    def _save_embeddings(self, embeddings: torch.Tensor, cache_key: str) -> None:
        """Lưu embeddings vào cache với metadata"""
        if not self.config.cache_embeddings:
            return
        
        try:
            cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"
            
            # Save embeddings
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings.cpu().numpy(), f)
            
            # Update metadata
            self._cache_metadata[cache_key] = {
                'created_at': datetime.now().isoformat(),
                'file_path': str(cache_file),
                'shape': embeddings.shape,
                'access_count': 1
            }
            self._save_cache_metadata()
            
            # Also save to memory cache
            self._save_to_memory_cache(cache_key, embeddings)
            
            logger.debug(f"Embeddings saved to cache: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Không thể lưu cache: {str(e)}")

    def _load_embeddings(self, cache_key: str) -> Optional[torch.Tensor]:
        """Load embeddings từ cache với fallback chain"""
        if not self.config.cache_embeddings:
            return None
        
        # First try memory cache
        embeddings = self._get_from_memory_cache(cache_key)
        if embeddings is not None:
            return embeddings
        
        # Then try disk cache
        if not self._is_cache_valid(cache_key):
            return None
        
        try:
            cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    embeddings_np = pickle.load(f)
                embeddings = torch.tensor(embeddings_np, dtype=torch.float32).to(self.device)
                
                # Update access count and save to memory cache
                if cache_key in self._cache_metadata:
                    self._cache_metadata[cache_key]['access_count'] += 1
                    self._save_cache_metadata()
                
                self._save_to_memory_cache(cache_key, embeddings)
                
                logger.debug(f"Embeddings loaded from disk cache: {cache_file}")
                return embeddings
                
        except Exception as e:
            logger.warning(f"Không thể load cache: {str(e)}")
        
        return None

    def _validate_questions(self, questions: List[str]) -> List[str]:
        """Validate và clean danh sách câu hỏi"""
        if not questions:
            raise DuplicateDetectorError("Danh sách câu hỏi không được để trống")
        
        if not isinstance(questions, list):
            raise DuplicateDetectorError("Questions phải là list")
        
        # Filter và clean questions
        cleaned_questions = []
        for i, q in enumerate(questions):
            if q is None:
                cleaned_questions.append("")
            elif isinstance(q, str):
                cleaned_questions.append(q.strip())
            else:
                cleaned_questions.append(str(q).strip())
        
        # Check có câu hỏi hợp lệ không
        valid_count = sum(1 for q in cleaned_questions if q)
        if valid_count == 0:
            raise DuplicateDetectorError("Không có câu hỏi hợp lệ nào")
        
        if valid_count != len(questions):
            logger.warning(f"Có {len(questions) - valid_count} câu hỏi trống/không hợp lệ")
        
        return cleaned_questions

    def encode_questions(self, questions: List[str], use_cache: bool = True, 
                        force_refresh: bool = False) -> torch.Tensor:
        """
        Encode danh sách câu hỏi thành embeddings với enhanced caching
        
        Args:
            questions: Danh sách câu hỏi
            use_cache: Có sử dụng cache không
            force_refresh: Force tạo embeddings mới (bỏ qua cache)
            
        Returns:
            torch.Tensor: Embeddings tensor
        """
        cleaned_questions = self._validate_questions(questions)
        
        # Load model nếu chưa load
        model = self._load_model()
        
        # Generate cache key
        cache_key = self._generate_cache_key(cleaned_questions) if use_cache else None
        
        # Try load from cache (unless force refresh)
        if use_cache and not force_refresh and cache_key:
            cached_embeddings = self._load_embeddings(cache_key)
            if cached_embeddings is not None:
                logger.info(f"✓ Sử dụng embeddings từ cache (shape: {cached_embeddings.shape})")
                return cached_embeddings
        
        # Generate new embeddings
        logger.info(f"🔄 Encoding {len(cleaned_questions)} câu hỏi...")
        start_time = time()
        
        try:
            all_embeddings = []
            
            with torch.no_grad():
                progress_bar = tqdm(range(0, len(cleaned_questions), self.config.batch_size), 
                                  desc="Encoding questions", 
                                  unit="batch")
                
                for i in progress_bar:
                    batch = cleaned_questions[i:i + self.config.batch_size]
                    
                    # Handle empty questions
                    batch_processed = [q if q else "[EMPTY]" for q in batch]
                    
                    batch_embeddings = model.encode(
                        batch_processed,
                        convert_to_tensor=True,
                        device=self.device,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    all_embeddings.append(batch_embeddings)
                    
                    # Update progress bar with current batch info
                    progress_bar.set_postfix({
                        'batch_size': len(batch),
                        'total_processed': min(i + self.config.batch_size, len(cleaned_questions))
                    })
            
            # Concatenate all embeddings
            embeddings = torch.cat(all_embeddings, dim=0)
            
            encoding_time = time() - start_time
            logger.info(f"✓ Encoding hoàn thành trong {encoding_time:.2f}s. Shape: {embeddings.shape}")
            
            # Save to cache
            if use_cache and cache_key:
                self._save_embeddings(embeddings, cache_key)
                logger.info("💾 Embeddings đã được lưu vào cache")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi encode câu hỏi: {str(e)}")
            raise DuplicateDetectorError(f"Lỗi khi encode câu hỏi: {str(e)}")

    def get_cache_info(self) -> Dict:
        """Lấy thông tin về cache hiện tại"""
        cache_info = {
            'memory_cache_size': len(self._memory_cache),
            'max_memory_cache_size': self._max_memory_cache_size,
            'disk_cache_entries': len(self._cache_metadata),
            'cache_dir': str(self.cache_dir),
            'total_disk_size_mb': 0
        }
        
        # Calculate total disk cache size
        try:
            total_size = 0
            for file_path in self.cache_dir.glob("embeddings_*.pkl"):
                total_size += file_path.stat().st_size
            cache_info['total_disk_size_mb'] = total_size / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not calculate cache size: {e}")
        
        return cache_info

    def clear_cache(self, clear_memory: bool = True, clear_disk: bool = False):
        """Clear cache"""
        if clear_memory:
            self._memory_cache.clear()
            logger.info("Memory cache cleared")
        
        if clear_disk:
            try:
                for file_path in self.cache_dir.glob("embeddings_*.pkl"):
                    file_path.unlink()
                self._cache_metadata.clear()
                self._save_cache_metadata()
                logger.info("Disk cache cleared")
            except Exception as e:
                logger.error(f"Error clearing disk cache: {e}")

    def encode_single_question(self, question: str, use_cache: bool = True) -> torch.Tensor:
        """
        Encode một câu hỏi đơn lẻ (optimized cho single question)
        
        Args:
            question: Câu hỏi cần encode
            use_cache: Có sử dụng cache không
            
        Returns:
            torch.Tensor: Embedding vector của câu hỏi
        """
        return self.encode_questions([question], use_cache=use_cache)[0:1]
    
    def compute_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Tính similarity matrix hiệu quả cho large datasets
        
        Args:
            embeddings: Tensor embeddings (đã normalized)
            
        Returns:
            torch.Tensor: Similarity matrix
        """
        logger.info(f"Tính similarity matrix cho {embeddings.shape[0]} embeddings...")
        
        try:
            with torch.no_grad():
                # Embeddings đã được normalize trong encode_questions
                # Compute similarity matrix using matrix multiplication
                similarity_matrix = torch.mm(embeddings, embeddings.t())
                
                # Clamp values to handle numerical precision
                similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
                
                logger.info(f"Similarity matrix computed. Shape: {similarity_matrix.shape}")
                return similarity_matrix
                
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA out of memory, switching to CPU...")
            embeddings_cpu = embeddings.cpu()
            similarity_matrix = torch.mm(embeddings_cpu, embeddings_cpu.t())
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            return similarity_matrix.to(self.device)
        
        except Exception as e:
            raise DuplicateDetectorError(f"Lỗi khi tính similarity matrix: {str(e)}")
    
    def find_duplicate_pairs(self, 
                           similarity_matrix: torch.Tensor, 
                           threshold: float = None) -> List[Tuple[int, int]]:
        """
        Tìm các cặp câu hỏi trùng lặp
        
        Args:
            similarity_matrix: Ma trận similarity
            threshold: Ngưỡng similarity (mặc định từ config)
            
        Returns:
            List[Tuple[int, int]]: Danh sách các cặp index trùng lặp
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        logger.info(f"Tìm duplicate pairs với threshold = {threshold}")
        
        duplicate_pairs = []
        n = similarity_matrix.shape[0]
        
        # Convert to numpy for faster iteration if on CPU
        if similarity_matrix.device.type == 'cpu':
            sim_matrix_np = similarity_matrix.numpy()
            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix_np[i, j] > threshold:
                        duplicate_pairs.append((i, j))
        else:
            # Use tensor operations for GPU
            # Get upper triangle indices
            triu_indices = torch.triu_indices(n, n, offset=1, device=similarity_matrix.device)
            triu_values = similarity_matrix[triu_indices[0], triu_indices[1]]
            
            # Find indices where similarity > threshold
            duplicate_mask = triu_values > threshold
            duplicate_i = triu_indices[0][duplicate_mask].cpu().numpy()
            duplicate_j = triu_indices[1][duplicate_mask].cpu().numpy()
            
            duplicate_pairs = list(zip(duplicate_i, duplicate_j))
        
        logger.info(f"Tìm thấy {len(duplicate_pairs)} cặp duplicate")
        return duplicate_pairs
    
    def analyze_duplicates(self, 
                         questions: List[str], 
                         threshold: float = None,
                         return_similarity_matrix: bool = False) -> DuplicateAnalysisResult:
        """
        Phân tích comprehensive các câu hỏi trùng lặp
        
        Args:
            questions: Danh sách câu hỏi
            threshold: Ngưỡng similarity
            return_similarity_matrix: Có trả về similarity matrix không
            
        Returns:
            DuplicateAnalysisResult: Kết quả phân tích
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        logger.info(f"Bắt đầu phân tích duplicate cho {len(questions)} câu hỏi")
        
        # Encode questions
        embeddings = self.encode_questions(questions)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Find duplicate pairs
        duplicate_pairs = self.find_duplicate_pairs(similarity_matrix, threshold)
        
        # Compute duplicate indices
        duplicate_indices = set()
        for i, j in duplicate_pairs:
            duplicate_indices.add(i)
            duplicate_indices.add(j)
        
        # Generate statistics
        statistics = self._generate_statistics(
            len(questions), duplicate_pairs, duplicate_indices, similarity_matrix
        )
        
        result = DuplicateAnalysisResult(
            total_questions=len(questions),
            duplicate_pairs=duplicate_pairs,
            duplicate_indices=duplicate_indices,
            similarity_matrix=similarity_matrix if return_similarity_matrix else None,
            statistics=statistics
        )
        
        logger.info(f"Phân tích hoàn thành: {len(duplicate_pairs)} cặp, "
                   f"{len(duplicate_indices)} câu hỏi bị trùng")
        
        return result
    
    def _generate_statistics(self, 
                           total_questions: int,
                           duplicate_pairs: List[Tuple[int, int]], 
                           duplicate_indices: Set[int],
                           similarity_matrix: torch.Tensor) -> Dict:
        """Tạo thống kê chi tiết"""
        
        # Tính phân bố similarity scores
        n = similarity_matrix.shape[0]
        upper_triangle_indices = torch.triu_indices(n, n, offset=1, device=similarity_matrix.device)
        similarity_scores = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
        
        # Count by similarity ranges
        similarity_ranges = {
            "0.95-1.0": ((similarity_scores >= 0.95) & (similarity_scores <= 1.0)).sum().item(),
            "0.9-0.95": ((similarity_scores >= 0.9) & (similarity_scores < 0.95)).sum().item(),
            "0.8-0.9": ((similarity_scores >= 0.8) & (similarity_scores < 0.9)).sum().item(),
            "0.7-0.8": ((similarity_scores >= 0.7) & (similarity_scores < 0.8)).sum().item(),
            "0.6-0.7": ((similarity_scores >= 0.6) & (similarity_scores < 0.7)).sum().item(),
            "< 0.6": (similarity_scores < 0.6).sum().item()
        }
        
        # Duplicate frequency per question
        duplicate_frequency = Counter()
        for i, j in duplicate_pairs:
            duplicate_frequency[i] += 1
            duplicate_frequency[j] += 1
        
        statistics = {
            "total_questions": total_questions,
            "unique_questions": total_questions - len(duplicate_indices),
            "duplicate_questions": len(duplicate_indices),
            "duplicate_pairs": len(duplicate_pairs),
            "duplicate_rate": len(duplicate_indices) / total_questions * 100 if total_questions > 0 else 0,
            "similarity_distribution": similarity_ranges,
            "max_similarity": similarity_scores.max().item() if len(similarity_scores) > 0 else 0,
            "min_similarity": similarity_scores.min().item() if len(similarity_scores) > 0 else 0,
            "avg_similarity": similarity_scores.mean().item() if len(similarity_scores) > 0 else 0,
            "most_duplicated_questions": duplicate_frequency.most_common(100) if duplicate_frequency else []
        }
        
        return statistics
    
    def get_duplicate_groups(self, duplicate_pairs: List[Tuple[int, int]]) -> List[Set[int]]:
        """
        Nhóm các câu hỏi duplicate thành các groups
        
        Args:
            duplicate_pairs: Danh sách các cặp duplicate
            
        Returns:
            List[Set[int]]: Danh sách các groups duplicate
        """
        if not duplicate_pairs:
            return []
        
        # Build graph and find connected components
        graph = defaultdict(set)
        for i, j in duplicate_pairs:
            graph[i].add(j)
            graph[j].add(i)
        
        visited = set()
        groups = []
        
        def dfs(node, current_group):
            visited.add(node)
            current_group.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_group)
        
        for node in graph:
            if node not in visited:
                group = set()
                dfs(node, group)
                groups.append(group)
        
        return groups
    
    def remove_duplicates(self, 
                         questions: List[str], 
                         threshold: float = None,
                         strategy: str = "keep_first") -> Tuple[List[str], List[int], List[Set[int]]]:
        """
        Loại bỏ câu hỏi trùng lặp
        
        Args:
            questions: Danh sách câu hỏi
            threshold: Ngưỡng similarity
            strategy: Chiến lược loại bỏ ("keep_first", "keep_longest", "keep_shortest")
            
        Returns:
            Tuple[List[str], List[int], List[Set[int]]]: (câu hỏi unique, indices được giữ lại, duplicate groups)
        """
        result = self.analyze_duplicates(questions, threshold)
        duplicate_groups = self.get_duplicate_groups(result.duplicate_pairs)
        
        to_remove = set()
        
        for group in duplicate_groups:
            group_list = list(group)
            
            if strategy == "keep_first":
                # Keep the smallest index (first occurrence)
                keep_idx = min(group_list)
                to_remove.update(idx for idx in group_list if idx != keep_idx)
                
            elif strategy == "keep_longest":
                # Keep the question with maximum length
                keep_idx = max(group_list, key=lambda x: len(questions[x]))
                to_remove.update(idx for idx in group_list if idx != keep_idx)
                
            elif strategy == "keep_shortest":
                # Keep the question with minimum length
                keep_idx = min(group_list, key=lambda x: len(questions[x]))
                to_remove.update(idx for idx in group_list if idx != keep_idx)
                
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
        
        # Create filtered list
        unique_questions = []
        kept_indices = []
        
        for i, question in enumerate(questions):
            if i not in to_remove:
                unique_questions.append(question)
                kept_indices.append(i)
        
        logger.info(f"Removed {len(to_remove)} duplicate questions using strategy '{strategy}'. "
                   f"Kept {len(unique_questions)} unique questions.")
        
        return unique_questions, kept_indices, duplicate_groups
    
    def print_analysis_report(self, result: DuplicateAnalysisResult, questions: List[str] = None) -> None:
        """In báo cáo phân tích chi tiết"""
        print("\n" + "="*70)
        print("📊 BÁO CÁO PHÂN TÍCH DUPLICATE QUESTIONS")
        print("="*70)
        
        stats = result.statistics
        
        print(f"📈 TỔNG QUAN:")
        print(f"   • Tổng số câu hỏi: {stats['total_questions']:,}")
        print(f"   • Câu hỏi unique: {stats['unique_questions']:,}")
        print(f"   • Câu hỏi duplicate: {stats['duplicate_questions']:,}")
        print(f"   • Tỷ lệ duplicate: {stats['duplicate_rate']:.2f}%")
        print(f"   • Số cặp duplicate: {stats['duplicate_pairs']:,}")
        
        print(f"\n📊 PHÂN BỐ SIMILARITY:")
        for range_name, count in stats['similarity_distribution'].items():
            percentage = (count / (stats['total_questions'] * (stats['total_questions'] - 1) // 2)) * 100 if stats['total_questions'] > 1 else 0
            print(f"   • {range_name}: {count:,} cặp ({percentage:.2f}%)")
        
        print(f"\n📈 THỐNG KÊ SIMILARITY:")
        print(f"   • Max similarity: {stats['max_similarity']:.4f}")
        print(f"   • Min similarity: {stats['min_similarity']:.4f}")
        print(f"   • Avg similarity: {stats['avg_similarity']:.4f}")
        
        if stats['most_duplicated_questions']:
            print(f"\n🔄 CÂU HỎI BỊ DUPLICATE NHIỀU NHẤT:")
            for idx, count in stats['most_duplicated_questions']:
                question_preview = questions[idx] if questions and len(questions[idx]) > 50 else (questions[idx] if questions else "")
                print(f"   • Question {idx}: {count} duplicates")
                if question_preview:
                    print(f"     \"{question_preview}\"")
        
        print("="*70)
    
    def export_results(self, result: DuplicateAnalysisResult, 
                      questions: List[str], 
                      output_file: str = "duplicate_analysis.json") -> None:
        """Export kết quả phân tích ra file JSON"""
        
        # Prepare export data
        export_data = {
            "config": {
                "model_name": self.config.model_name,
                "similarity_threshold": self.config.similarity_threshold,
                "max_seq_length": self.config.max_seq_length
            },
            "statistics": result.statistics,
            "duplicate_pairs": [
                {
                    "indices": [i, j],
                    "questions": [questions[i], questions[j]],
                    "similarity": result.similarity_matrix[i, j].item() if result.similarity_matrix is not None else None
                }
                for i, j in result.duplicate_pairs
            ],
            "duplicate_groups": [
                {
                    "group_id": idx,
                    "indices": list(group),
                    "questions": [questions[i] for i in group]
                }
                for idx, group in enumerate(self.get_duplicate_groups(result.duplicate_pairs))
            ]
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Kết quả đã được export ra file: {output_file}")


def main():
    """Hàm main để test"""
    
    # Tạo sample questions cho demo
    sample_questions = all_questions
    try:
        # Khởi tạo detector
        config = DuplicateDetectorConfig(
            similarity_threshold=0.75,  # Lower threshold for demo
            batch_size=8,
            cache_embeddings=True
        )
        detector = QuestionDuplicateDetector(config)
        
        print("🚀 Bắt đầu phân tích duplicate questions...")
        
        # Phân tích duplicates
        result = detector.analyze_duplicates(sample_questions, return_similarity_matrix=True)
        
        # In báo cáo
        detector.print_analysis_report(result, sample_questions)
        
        # Hiện duplicate pairs chi tiết
        if result.duplicate_pairs:
            print(f"\n🔍 CHI TIẾT CÁC CẶP DUPLICATE (Top 10):")
            for i, (idx1, idx2) in enumerate(result.duplicate_pairs[:10], 1):
                sim_score = result.similarity_matrix[idx1, idx2].item()
                print(f"\n{i}. Similarity: {sim_score:.4f}")
                print(f"   [{idx1}] \"{sample_questions[idx1]}\"")
                print(f"   [{idx2}] \"{sample_questions[idx2]}\"")
        
        # Export results (optional)
        print(f"\n💾 Exporting results...")
        detector.export_results(result, sample_questions, "duplicate_analysis_demo.json")
        print("✅ Export completed!")
        
    except DuplicateDetectorError as e:
        print(f"❌ Lỗi Duplicate Detector: {e}")
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()


# if __name__ == "__main__":
#     main()