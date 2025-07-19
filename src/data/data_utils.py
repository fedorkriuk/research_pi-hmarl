"""Data utilities for PI-HMARL

This module provides utility functions for data loading, preprocessing,
and management for the PI-HMARL framework.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator
import json
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from dataclasses import dataclass
from multiprocessing import Pool
import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    batch_size: int = 32
    sequence_length: int = 100
    stride: int = 10
    num_workers: int = 4
    shuffle: bool = True
    normalize: bool = True
    augment: bool = False
    cache_size: int = 1000  # Number of sequences to cache
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Preprocessing
    position_scale: float = 100.0  # Scale positions to reasonable range
    velocity_scale: float = 10.0
    acceleration_scale: float = 5.0
    
    # Augmentation parameters
    position_noise_std: float = 0.1
    velocity_noise_std: float = 0.05
    rotation_augment: bool = True


class PIHMARLDataset(Dataset):
    """PyTorch dataset for PI-HMARL training data"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: DataConfig,
        split: str = "train",
        transform: Optional[Any] = None
    ):
        """Initialize the dataset
        
        Args:
            data_dir: Directory containing .h5 data files
            config: Data configuration
            split: Data split (train, val, test)
            transform: Optional data transformation
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Find all data files
        self.data_files = sorted(self.data_dir.glob("*.h5"))
        
        # Split files
        self._split_files()
        
        # Build sequence index
        self.sequence_index = []
        self._build_sequence_index()
        
        # Cache for loaded sequences
        self.cache = {}
        
        logger.info(
            f"Initialized {split} dataset with {len(self.sequence_index)} sequences "
            f"from {len(self.split_files)} files"
        )
    
    def _split_files(self):
        """Split files into train/val/test sets"""
        n_files = len(self.data_files)
        
        train_end = int(n_files * self.config.train_split)
        val_end = train_end + int(n_files * self.config.val_split)
        
        if self.split == "train":
            self.split_files = self.data_files[:train_end]
        elif self.split == "val":
            self.split_files = self.data_files[train_end:val_end]
        elif self.split == "test":
            self.split_files = self.data_files[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _build_sequence_index(self):
        """Build index of all sequences in the dataset"""
        for file_idx, file_path in enumerate(self.split_files):
            with h5py.File(file_path, 'r') as f:
                # Get trajectory length
                positions = f['trajectories/positions']
                n_timesteps = positions.shape[0]
                
                # Create sequences with stride
                for start_idx in range(0, n_timesteps - self.config.sequence_length + 1,
                                     self.config.stride):
                    self.sequence_index.append({
                        'file_idx': file_idx,
                        'file_path': file_path,
                        'start_idx': start_idx,
                        'end_idx': start_idx + self.config.sequence_length
                    })
    
    def __len__(self) -> int:
        return len(self.sequence_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of data
        
        Args:
            idx: Sequence index
            
        Returns:
            Dictionary of tensors
        """
        # Check cache
        if idx in self.cache and len(self.cache) < self.config.cache_size:
            return self.cache[idx]
        
        # Get sequence info
        seq_info = self.sequence_index[idx]
        
        # Load data
        with h5py.File(seq_info['file_path'], 'r') as f:
            start = seq_info['start_idx']
            end = seq_info['end_idx']
            
            # Load trajectories
            data = {
                'positions': f['trajectories/positions'][start:end],
                'velocities': f['trajectories/velocities'][start:end],
                'accelerations': f['trajectories/accelerations'][start:end],
                'orientations': f['trajectories/orientations'][start:end],
            }
            
            # Load energy data
            if 'energy' in f:
                data['battery_soc'] = f['energy/battery_soc'][start:end]
                data['power_consumption'] = f['energy/power_consumption'][start:end]
            
            # Load constraint labels
            if 'constraints' in f:
                data['energy_satisfied'] = f['constraints/energy_satisfied'][start:end]
                data['velocity_satisfied'] = f['constraints/velocity_satisfied'][start:end]
                data['collision_distances'] = f['constraints/collision_distances'][start:end]
        
        # Preprocessing
        data = self._preprocess(data)
        
        # Apply augmentation if training
        if self.split == "train" and self.config.augment:
            data = self._augment(data)
        
        # Convert to tensors
        for key, value in data.items():
            data[key] = torch.from_numpy(value).float()
        
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
        
        # Cache if space available
        if len(self.cache) < self.config.cache_size:
            self.cache[idx] = data
        
        return data
    
    def _preprocess(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess data
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Preprocessed data
        """
        if self.config.normalize:
            # Normalize positions
            if 'positions' in data:
                data['positions'] = data['positions'] / self.config.position_scale
            
            # Normalize velocities
            if 'velocities' in data:
                data['velocities'] = data['velocities'] / self.config.velocity_scale
            
            # Normalize accelerations
            if 'accelerations' in data:
                data['accelerations'] = data['accelerations'] / self.config.acceleration_scale
        
        return data
    
    def _augment(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply data augmentation
        
        Args:
            data: Preprocessed data
            
        Returns:
            Augmented data
        """
        # Add noise to positions
        if 'positions' in data and self.config.position_noise_std > 0:
            noise = np.random.normal(0, self.config.position_noise_std, 
                                   data['positions'].shape)
            data['positions'] = data['positions'] + noise
        
        # Add noise to velocities
        if 'velocities' in data and self.config.velocity_noise_std > 0:
            noise = np.random.normal(0, self.config.velocity_noise_std,
                                   data['velocities'].shape)
            data['velocities'] = data['velocities'] + noise
        
        # Random rotation augmentation
        if self.config.rotation_augment and 'positions' in data:
            # Random yaw rotation
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Apply rotation to positions and velocities
            for key in ['positions', 'velocities', 'accelerations']:
                if key in data:
                    x, y = data[key][..., 0], data[key][..., 1]
                    data[key][..., 0] = cos_a * x - sin_a * y
                    data[key][..., 1] = sin_a * x + cos_a * y
        
        return data
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute dataset statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'positions': {'mean': [], 'std': []},
            'velocities': {'mean': [], 'std': []},
            'accelerations': {'mean': [], 'std': []},
            'power_consumption': {'mean': [], 'std': []}
        }
        
        # Sample subset of data
        sample_size = min(100, len(self))
        indices = np.random.choice(len(self), sample_size, replace=False)
        
        for idx in indices:
            data = self[idx]
            for key in stats:
                if key in data:
                    values = data[key].numpy()
                    stats[key]['mean'].append(np.mean(values))
                    stats[key]['std'].append(np.std(values))
        
        # Aggregate statistics
        for key in stats:
            if stats[key]['mean']:
                stats[key]['mean'] = np.mean(stats[key]['mean'])
                stats[key]['std'] = np.mean(stats[key]['std'])
            else:
                stats[key]['mean'] = 0.0
                stats[key]['std'] = 1.0
        
        return stats


def create_dataloaders(
    data_dir: Union[str, Path],
    config: DataConfig,
    batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing data files
        config: Data configuration
        batch_size: Override batch size from config
        
    Returns:
        Train, validation, and test dataloaders
    """
    if batch_size is not None:
        config.batch_size = batch_size
    
    # Create datasets
    train_dataset = PIHMARLDataset(data_dir, config, split="train")
    val_dataset = PIHMARLDataset(data_dir, config, split="val")
    test_dataset = PIHMARLDataset(data_dir, config, split="test")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def merge_h5_files(
    input_files: List[Path],
    output_file: Path,
    compression: str = "gzip"
):
    """Merge multiple HDF5 files into one
    
    Args:
        input_files: List of input file paths
        output_file: Output file path
        compression: HDF5 compression type
    """
    with h5py.File(output_file, 'w') as out_f:
        # Track total samples
        total_timesteps = 0
        
        # First pass: determine total size
        for file_path in input_files:
            with h5py.File(file_path, 'r') as in_f:
                if 'trajectories/positions' in in_f:
                    total_timesteps += in_f['trajectories/positions'].shape[0]
        
        # Create datasets
        datasets_created = False
        current_idx = 0
        
        # Second pass: copy data
        for file_path in tqdm.tqdm(input_files, desc="Merging files"):
            with h5py.File(file_path, 'r') as in_f:
                # Copy metadata from first file
                if not datasets_created and 'metadata' in in_f:
                    in_f.copy('metadata', out_f)
                
                # Copy trajectory data
                for group_name in ['trajectories', 'energy', 'constraints']:
                    if group_name in in_f:
                        in_group = in_f[group_name]
                        
                        if not datasets_created:
                            # Create output group
                            out_group = out_f.create_group(group_name)
                            
                            # Create datasets
                            for dataset_name in in_group:
                                shape = list(in_group[dataset_name].shape)
                                shape[0] = total_timesteps
                                
                                out_group.create_dataset(
                                    dataset_name,
                                    shape=shape,
                                    dtype=in_group[dataset_name].dtype,
                                    compression=compression
                                )
                        
                        # Copy data
                        for dataset_name in in_group:
                            data = in_group[dataset_name][:]
                            n_samples = data.shape[0]
                            
                            out_f[f'{group_name}/{dataset_name}'][
                                current_idx:current_idx + n_samples
                            ] = data
                
                if 'trajectories/positions' in in_f:
                    current_idx += in_f['trajectories/positions'].shape[0]
                
                datasets_created = True
        
        # Add merge metadata
        out_f.attrs['merged_from'] = [str(f) for f in input_files]
        out_f.attrs['total_timesteps'] = total_timesteps
    
    logger.info(f"Merged {len(input_files)} files into {output_file}")


def compute_trajectory_metrics(
    trajectory_file: Path
) -> Dict[str, Any]:
    """Compute metrics for a trajectory file
    
    Args:
        trajectory_file: Path to HDF5 trajectory file
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    with h5py.File(trajectory_file, 'r') as f:
        # Basic info
        if 'trajectories/positions' in f:
            positions = f['trajectories/positions'][:]
            metrics['duration'] = positions.shape[0] * 0.01  # Assuming 100Hz
            metrics['num_agents'] = positions.shape[1]
            metrics['num_timesteps'] = positions.shape[0]
            
            # Trajectory length
            trajectory_lengths = []
            for agent_idx in range(positions.shape[1]):
                agent_pos = positions[:, agent_idx]
                distances = np.linalg.norm(np.diff(agent_pos, axis=0), axis=1)
                trajectory_lengths.append(np.sum(distances))
            
            metrics['avg_trajectory_length'] = np.mean(trajectory_lengths)
            metrics['total_distance'] = np.sum(trajectory_lengths)
        
        # Energy metrics
        if 'energy/battery_soc' in f:
            battery_soc = f['energy/battery_soc'][:]
            metrics['min_battery_soc'] = np.min(battery_soc)
            metrics['avg_battery_depletion'] = 1.0 - np.mean(battery_soc[-1])
        
        # Constraint violations
        if 'constraints/energy_satisfied' in f:
            energy_satisfied = f['constraints/energy_satisfied'][:]
            metrics['energy_violation_rate'] = 1.0 - np.mean(energy_satisfied)
        
        if 'constraints/collision_distances' in f:
            collision_distances = f['constraints/collision_distances'][:]
            min_distances = np.min(collision_distances + np.eye(
                collision_distances.shape[1]
            )[None] * 1000, axis=(1, 2))
            metrics['min_separation'] = np.min(min_distances)
            metrics['avg_min_separation'] = np.mean(min_distances)
    
    return metrics


def parallel_data_generation(
    generator_func: Callable,
    num_scenarios: int,
    output_dir: Path,
    num_workers: int = 4
):
    """Generate data in parallel using multiple processes
    
    Args:
        generator_func: Function to generate single scenario
        num_scenarios: Total number of scenarios
        output_dir: Output directory
        num_workers: Number of parallel workers
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tasks
    tasks = []
    for i in range(num_scenarios):
        output_path = output_dir / f"scenario_{i:04d}.h5"
        tasks.append((i, output_path))
    
    # Worker function
    def worker(args):
        idx, output_path = args
        try:
            generator_func(idx, output_path)
            return f"Generated {output_path}"
        except Exception as e:
            return f"Error generating {output_path}: {e}"
    
    # Execute in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm.tqdm(
            pool.imap(worker, tasks),
            total=len(tasks),
            desc="Generating scenarios"
        ))
    
    # Log results
    for result in results:
        logger.info(result)


# Convenience functions
def load_trajectory(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load trajectory from HDF5 file
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        Dictionary of trajectory data
    """
    data = {}
    
    with h5py.File(file_path, 'r') as f:
        # Load all datasets
        for group_name in f:
            if isinstance(f[group_name], h5py.Group):
                for dataset_name in f[group_name]:
                    key = f"{group_name}/{dataset_name}"
                    data[key] = f[key][:]
    
    return data


def save_trajectory(
    data: Dict[str, np.ndarray],
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
):
    """Save trajectory to HDF5 file
    
    Args:
        data: Dictionary of trajectory data
        file_path: Output file path
        metadata: Optional metadata
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        # Save data
        for key, value in data.items():
            if '/' in key:
                group_name, dataset_name = key.rsplit('/', 1)
                if group_name not in f:
                    f.create_group(group_name)
                f[key] = value
            else:
                f[key] = value
        
        # Save metadata
        if metadata:
            meta_group = f.create_group('metadata') if 'metadata' not in f else f['metadata']
            for key, value in metadata.items():
                meta_group.attrs[key] = value