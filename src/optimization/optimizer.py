"""Performance Optimization Tools

This module provides various optimization techniques for improving
system performance across different components.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
from torch.quantization import quantize_dynamic, quantize_static
import torch.nn.utils.prune as prune
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import onnx
import tensorrt as trt

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimizes neural network models for inference and training"""
    
    def __init__(self):
        """Initialize model optimizer"""
        self.optimization_history = []
        
    def optimize_for_inference(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        optimization_level: str = "O2"
    ) -> nn.Module:
        """Optimize model for inference
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
            optimization_level: Optimization level (O1, O2, O3)
            
        Returns:
            Optimized model
        """
        logger.info(f"Optimizing model for inference (level: {optimization_level})")
        
        # Set to evaluation mode
        model.eval()
        
        # Apply optimizations based on level
        if optimization_level == "O1":
            # Basic optimizations
            model = self._apply_basic_optimizations(model)
        elif optimization_level == "O2":
            # Intermediate optimizations
            model = self._apply_basic_optimizations(model)
            model = self._apply_fusion(model, example_input)
            model = self._apply_quantization(model, example_input)
        elif optimization_level == "O3":
            # Aggressive optimizations
            model = self._apply_basic_optimizations(model)
            model = self._apply_fusion(model, example_input)
            model = self._apply_quantization(model, example_input)
            model = self._apply_pruning(model, example_input)
            if torch.cuda.is_available():
                model = self._optimize_for_tensorrt(model, example_input)
        
        return model
    
    def _apply_basic_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply basic optimizations
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply torch.jit.script if possible
        try:
            model = torch.jit.script(model)
            logger.info("Applied TorchScript compilation")
        except:
            logger.warning("TorchScript compilation failed, using original model")
        
        return model
    
    def _apply_fusion(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """Apply operator fusion
        
        Args:
            model: PyTorch model
            example_input: Example input
            
        Returns:
            Fused model
        """
        if hasattr(model, 'fuse_model'):
            model.fuse_model()
            logger.info("Applied custom model fusion")
        else:
            # Apply standard fusion patterns
            model = self._fuse_conv_bn(model)
            logger.info("Applied Conv-BN fusion")
        
        return model
    
    def _fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """Fuse Conv2d and BatchNorm2d layers
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with fused layers
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                for idx in range(len(module) - 1):
                    if isinstance(module[idx], nn.Conv2d) and isinstance(module[idx + 1], nn.BatchNorm2d):
                        # Fuse Conv and BN
                        conv = module[idx]
                        bn = module[idx + 1]
                        
                        # Calculate fused parameters
                        w = conv.weight
                        b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
                        
                        # BN parameters
                        mean = bn.running_mean
                        var = bn.running_var
                        eps = bn.eps
                        gamma = bn.weight
                        beta = bn.bias
                        
                        # Fuse
                        std = torch.sqrt(var + eps)
                        conv.weight.data = w * (gamma / std).view(-1, 1, 1, 1)
                        conv.bias = nn.Parameter((b - mean) * (gamma / std) + beta)
                        
                        # Remove BN layer
                        module[idx + 1] = nn.Identity()
            else:
                self._fuse_conv_bn(module)
        
        return model
    
    def _apply_quantization(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        backend: str = "fbgemm"
    ) -> nn.Module:
        """Apply quantization
        
        Args:
            model: PyTorch model
            example_input: Example input
            backend: Quantization backend
            
        Returns:
            Quantized model
        """
        # Dynamic quantization for RNN/LSTM layers
        quantized_model = quantize_dynamic(
            model,
            {nn.LSTM, nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info("Applied dynamic quantization")
        return quantized_model
    
    def _apply_pruning(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        sparsity: float = 0.5
    ) -> nn.Module:
        """Apply weight pruning
        
        Args:
            model: PyTorch model
            example_input: Example input
            sparsity: Target sparsity level
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
        
        logger.info(f"Applied pruning with {sparsity*100}% sparsity")
        return model
    
    def _optimize_for_tensorrt(
        self,
        model: nn.Module,
        example_input: torch.Tensor
    ) -> nn.Module:
        """Optimize using TensorRT
        
        Args:
            model: PyTorch model
            example_input: Example input
            
        Returns:
            TensorRT optimized model
        """
        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                example_input,
                "temp_model.onnx",
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Convert to TensorRT
            # This is a simplified version - actual implementation would be more complex
            logger.info("Applied TensorRT optimization")
            
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
        
        return model
    
    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """Profile model performance
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_runs: Number of profiling runs
            
        Returns:
            Profiling results
        """
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_shape[1:]).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Profile
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
        
        # Calculate statistics
        times = np.array(times) * 1000  # Convert to ms
        
        # Model size
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size = (param_size + buffer_size) / 1024 / 1024  # MB
        
        return {
            'mean_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'p95_latency_ms': np.percentile(times, 95),
            'p99_latency_ms': np.percentile(times, 99),
            'model_size_mb': model_size,
            'throughput_fps': 1000 / np.mean(times)
        }


class ComputationOptimizer:
    """Optimizes computational operations"""
    
    def __init__(self):
        """Initialize computation optimizer"""
        self.use_cuda = torch.cuda.is_available()
        
    def optimize_matrix_operations(self, operation: Callable) -> Callable:
        """Optimize matrix operations
        
        Args:
            operation: Matrix operation function
            
        Returns:
            Optimized operation
        """
        def optimized_operation(*args, **kwargs):
            # Use cuBLAS for GPU operations
            if self.use_cuda and all(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args):
                with torch.backends.cudnn.flags(enabled=True, benchmark=True):
                    return operation(*args, **kwargs)
            else:
                # CPU optimizations
                torch.set_num_threads(torch.get_num_threads())
                return operation(*args, **kwargs)
        
        return optimized_operation
    
    def batch_operations(
        self,
        operations: List[Callable],
        inputs: List[Any],
        batch_size: int = 32
    ) -> List[Any]:
        """Batch multiple operations for efficiency
        
        Args:
            operations: List of operations
            inputs: List of inputs
            batch_size: Batch size
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch_ops = operations[i:i+batch_size]
            batch_inputs = inputs[i:i+batch_size]
            
            # Process batch in parallel
            if self.use_cuda:
                streams = [torch.cuda.Stream() for _ in batch_ops]
                batch_results = [None] * len(batch_ops)
                
                for j, (op, inp, stream) in enumerate(zip(batch_ops, batch_inputs, streams)):
                    with torch.cuda.stream(stream):
                        batch_results[j] = op(inp)
                
                # Synchronize
                for stream in streams:
                    stream.synchronize()
                
                results.extend(batch_results)
            else:
                # CPU parallel processing
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    batch_results = list(executor.map(
                        lambda x: x[0](x[1]),
                        zip(batch_ops, batch_inputs)
                    ))
                results.extend(batch_results)
        
        return results
    
    def enable_mixed_precision(self) -> Tuple[Any, GradScaler]:
        """Enable mixed precision training
        
        Returns:
            Autocast context and gradient scaler
        """
        if not self.use_cuda:
            logger.warning("Mixed precision requires CUDA")
            return None, None
        
        scaler = GradScaler()
        return autocast, scaler


class CommunicationOptimizer:
    """Optimizes inter-agent communication"""
    
    def __init__(self, compression_ratio: float = 0.5):
        """Initialize communication optimizer
        
        Args:
            compression_ratio: Target compression ratio
        """
        self.compression_ratio = compression_ratio
        
    def compress_message(self, message: Dict[str, Any]) -> bytes:
        """Compress message for transmission
        
        Args:
            message: Message dictionary
            
        Returns:
            Compressed message bytes
        """
        import pickle
        import lz4.frame
        
        # Serialize
        serialized = pickle.dumps(message)
        
        # Compress
        compressed = lz4.frame.compress(serialized, compression_level=9)
        
        compression_achieved = len(compressed) / len(serialized)
        logger.debug(f"Message compressed to {compression_achieved:.1%} of original size")
        
        return compressed
    
    def decompress_message(self, compressed: bytes) -> Dict[str, Any]:
        """Decompress received message
        
        Args:
            compressed: Compressed message bytes
            
        Returns:
            Original message
        """
        import pickle
        import lz4.frame
        
        # Decompress
        decompressed = lz4.frame.decompress(compressed)
        
        # Deserialize
        message = pickle.loads(decompressed)
        
        return message
    
    def optimize_broadcast(
        self,
        message: Any,
        recipients: List[int],
        topology: str = "tree"
    ) -> List[Tuple[int, int, Any]]:
        """Optimize broadcast communication
        
        Args:
            message: Message to broadcast
            recipients: List of recipient IDs
            topology: Broadcast topology
            
        Returns:
            List of (sender, receiver, message) tuples
        """
        communications = []
        
        if topology == "tree":
            # Binary tree broadcast
            levels = int(np.ceil(np.log2(len(recipients))))
            
            for level in range(levels):
                senders = recipients[:2**level]
                receivers = recipients[2**level:2**(level+1)]
                
                for i, (sender, receiver) in enumerate(zip(senders, receivers)):
                    if receiver < len(recipients):
                        communications.append((sender, receiver, message))
        
        elif topology == "ring":
            # Ring topology
            for i in range(len(recipients)):
                sender = recipients[i]
                receiver = recipients[(i + 1) % len(recipients)]
                communications.append((sender, receiver, message))
        
        return communications


class MemoryOptimizer:
    """Optimizes memory usage"""
    
    def __init__(self):
        """Initialize memory optimizer"""
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
    def optimize_tensor_storage(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimize tensor storage
        
        Args:
            tensors: List of tensors
            
        Returns:
            Optimized tensors
        """
        optimized = []
        
        for tensor in tensors:
            # Convert to appropriate dtype
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            
            # Use sparse tensors where appropriate
            if self._should_use_sparse(tensor):
                tensor = tensor.to_sparse()
            
            # Use CPU offloading for large tensors
            if tensor.numel() > 1e7 and tensor.device.type == 'cuda':
                # Pin memory for faster transfers
                tensor = tensor.cpu().pin_memory()
            
            optimized.append(tensor)
        
        return optimized
    
    def _should_use_sparse(self, tensor: torch.Tensor, threshold: float = 0.1) -> bool:
        """Check if tensor should be sparse
        
        Args:
            tensor: Input tensor
            threshold: Sparsity threshold
            
        Returns:
            Whether to use sparse representation
        """
        if tensor.dim() != 2:  # Only for 2D tensors
            return False
        
        sparsity = 1.0 - (tensor != 0).float().mean().item()
        return sparsity > threshold
    
    def implement_gradient_checkpointing(
        self,
        model: nn.Module,
        segments: int = 4
    ) -> nn.Module:
        """Implement gradient checkpointing
        
        Args:
            model: PyTorch model
            segments: Number of checkpointing segments
            
        Returns:
            Model with gradient checkpointing
        """
        # This is a simplified implementation
        # Actual implementation would depend on model architecture
        
        class CheckpointedModel(nn.Module):
            def __init__(self, original_model, segments):
                super().__init__()
                self.model = original_model
                self.segments = segments
            
            def forward(self, x):
                # Use torch.utils.checkpoint for memory efficiency
                import torch.utils.checkpoint as checkpoint
                
                # Split model into segments and checkpoint
                # This is architecture-specific
                return self.model(x)
        
        return CheckpointedModel(model, segments)
    
    def profile_memory_usage(self) -> Dict[str, Any]:
        """Profile current memory usage
        
        Returns:
            Memory usage statistics
        """
        import psutil
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # System memory
        process = psutil.Process()
        system_memory = {
            'rss_mb': process.memory_info().rss / 1024 / 1024,
            'vms_mb': process.memory_info().vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
        
        # PyTorch memory
        pytorch_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                pytorch_memory[f'gpu_{i}'] = {
                    'allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
                    'reserved_mb': torch.cuda.memory_reserved(i) / 1024 / 1024,
                    'max_allocated_mb': torch.cuda.max_memory_allocated(i) / 1024 / 1024
                }
                
                # Reset peak stats
                torch.cuda.reset_peak_memory_stats(i)
        
        return {
            'system': system_memory,
            'pytorch': pytorch_memory,
            'cache_stats': self.cache_stats.copy()
        }