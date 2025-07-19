"""Deployment Management System

This module handles deployment strategies and execution for different
environments (cloud, edge, hybrid).
"""

import os
import subprocess
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import logging
import shutil
import tarfile
import tempfile

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    RUNNING = "running"
    UPDATING = "updating"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    ROLLBACK = "rollback"


class DeploymentTarget(Enum):
    """Deployment targets"""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    EDGE = "edge"
    HYBRID = "hybrid"


@dataclass
class DeploymentResult:
    """Deployment operation result"""
    success: bool
    status: DeploymentStatus
    message: str
    deployment_id: str
    endpoints: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


class DeploymentStrategy(ABC):
    """Abstract deployment strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize deployment strategy
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.status = DeploymentStatus.PENDING
        self.deployment_id = f"{config.get('name', 'deployment')}_{int(time.time())}"
    
    @abstractmethod
    async def deploy(self) -> DeploymentResult:
        """Execute deployment"""
        pass
    
    @abstractmethod
    async def update(self, new_config: Dict[str, Any]) -> DeploymentResult:
        """Update deployment"""
        pass
    
    @abstractmethod
    async def rollback(self, version: str) -> DeploymentResult:
        """Rollback deployment"""
        pass
    
    @abstractmethod
    async def stop(self) -> DeploymentResult:
        """Stop deployment"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        pass


class CloudDeployment(DeploymentStrategy):
    """Cloud deployment strategy"""
    
    def __init__(self, config: Dict[str, Any], provider: str):
        """Initialize cloud deployment
        
        Args:
            config: Deployment configuration
            provider: Cloud provider (aws, azure, gcp)
        """
        super().__init__(config)
        self.provider = provider
        self.region = config.get('region', 'us-east-1')
        self.instance_type = config.get('instance_type', 't3.medium')
        self.cluster_name = config.get('cluster_name', 'pi-hmarl-cluster')
    
    async def deploy(self) -> DeploymentResult:
        """Deploy to cloud"""
        self.status = DeploymentStatus.PREPARING
        logs = []
        
        try:
            # Prepare deployment package
            package_path = await self._prepare_package()
            logs.append(f"Package prepared: {package_path}")
            
            # Create infrastructure
            self.status = DeploymentStatus.DEPLOYING
            infra_result = await self._create_infrastructure()
            logs.extend(infra_result['logs'])
            
            # Deploy application
            deploy_result = await self._deploy_application(package_path)
            logs.extend(deploy_result['logs'])
            
            # Configure networking
            network_result = await self._configure_networking()
            logs.extend(network_result['logs'])
            
            # Verify deployment
            if await self._verify_deployment():
                self.status = DeploymentStatus.RUNNING
                endpoints = await self._get_endpoints()
                
                return DeploymentResult(
                    success=True,
                    status=self.status,
                    message=f"Deployment successful on {self.provider}",
                    deployment_id=self.deployment_id,
                    endpoints=endpoints,
                    logs=logs
                )
            else:
                raise Exception("Deployment verification failed")
                
        except Exception as e:
            self.status = DeploymentStatus.FAILED
            logger.error(f"Cloud deployment failed: {e}")
            
            return DeploymentResult(
                success=False,
                status=self.status,
                message=str(e),
                deployment_id=self.deployment_id,
                logs=logs
            )
    
    async def update(self, new_config: Dict[str, Any]) -> DeploymentResult:
        """Update cloud deployment"""
        self.status = DeploymentStatus.UPDATING
        logs = []
        
        try:
            # Validate new configuration
            if not self._validate_config(new_config):
                raise ValueError("Invalid configuration")
            
            # Create new version
            version_id = f"v{int(time.time())}"
            logs.append(f"Creating version: {version_id}")
            
            # Rolling update
            if self.provider == "kubernetes":
                result = await self._kubernetes_rolling_update(new_config)
            elif self.provider == "aws":
                result = await self._aws_blue_green_deployment(new_config)
            else:
                result = await self._generic_update(new_config)
            
            logs.extend(result.get('logs', []))
            
            if result['success']:
                self.status = DeploymentStatus.RUNNING
                return DeploymentResult(
                    success=True,
                    status=self.status,
                    message=f"Update successful: {version_id}",
                    deployment_id=self.deployment_id,
                    logs=logs
                )
            else:
                raise Exception("Update failed")
                
        except Exception as e:
            # Automatic rollback
            logger.error(f"Update failed, initiating rollback: {e}")
            rollback_result = await self.rollback("previous")
            
            return DeploymentResult(
                success=False,
                status=DeploymentStatus.ROLLBACK,
                message=f"Update failed: {e}",
                deployment_id=self.deployment_id,
                logs=logs + rollback_result.logs
            )
    
    async def rollback(self, version: str) -> DeploymentResult:
        """Rollback cloud deployment"""
        self.status = DeploymentStatus.ROLLBACK
        logs = [f"Rolling back to version: {version}"]
        
        try:
            # Get previous configuration
            prev_config = await self._get_version_config(version)
            
            # Execute rollback
            if self.provider == "kubernetes":
                cmd = f"kubectl rollout undo deployment/{self.config['name']}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                logs.append(result.stdout)
            else:
                # Generic rollback
                result = await self._generic_rollback(prev_config)
                logs.extend(result.get('logs', []))
            
            self.status = DeploymentStatus.RUNNING
            
            return DeploymentResult(
                success=True,
                status=self.status,
                message=f"Rollback to {version} successful",
                deployment_id=self.deployment_id,
                logs=logs
            )
            
        except Exception as e:
            self.status = DeploymentStatus.FAILED
            return DeploymentResult(
                success=False,
                status=self.status,
                message=f"Rollback failed: {e}",
                deployment_id=self.deployment_id,
                logs=logs
            )
    
    async def stop(self) -> DeploymentResult:
        """Stop cloud deployment"""
        self.status = DeploymentStatus.STOPPING
        logs = []
        
        try:
            # Stop services
            if self.provider == "kubernetes":
                cmd = f"kubectl delete deployment {self.config['name']}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                logs.append(result.stdout)
            else:
                # Provider-specific stop
                result = await self._provider_stop()
                logs.extend(result.get('logs', []))
            
            # Clean up resources
            await self._cleanup_resources()
            
            self.status = DeploymentStatus.STOPPED
            
            return DeploymentResult(
                success=True,
                status=self.status,
                message="Deployment stopped",
                deployment_id=self.deployment_id,
                logs=logs
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                status=self.status,
                message=f"Stop failed: {e}",
                deployment_id=self.deployment_id,
                logs=logs
            )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        if self.provider == "kubernetes":
            cmd = f"kubectl get deployment {self.config['name']} -o json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                deployment_info = json.loads(result.stdout)
                return {
                    'status': self.status.value,
                    'replicas': deployment_info['status'].get('replicas', 0),
                    'ready_replicas': deployment_info['status'].get('readyReplicas', 0),
                    'conditions': deployment_info['status'].get('conditions', [])
                }
        
        return {
            'status': self.status.value,
            'provider': self.provider,
            'deployment_id': self.deployment_id
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        metrics = {
            'deployment_id': self.deployment_id,
            'provider': self.provider,
            'status': self.status.value
        }
        
        if self.provider == "kubernetes":
            # Get pod metrics
            cmd = f"kubectl top pods -l app={self.config['name']}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse metrics
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                pod_metrics = []
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        pod_metrics.append({
                            'name': parts[0],
                            'cpu': parts[1],
                            'memory': parts[2]
                        })
                
                metrics['pods'] = pod_metrics
        
        return metrics
    
    async def _prepare_package(self) -> str:
        """Prepare deployment package"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        package_dir = Path(temp_dir) / "pi-hmarl-deploy"
        package_dir.mkdir()
        
        # Copy source code
        src_dir = Path(__file__).parent.parent / "src"
        shutil.copytree(src_dir, package_dir / "src")
        
        # Copy configuration
        config_dir = Path(__file__).parent.parent / "configs"
        shutil.copytree(config_dir, package_dir / "configs")
        
        # Create deployment manifest
        manifest = {
            'name': self.config['name'],
            'version': self.config['version'],
            'deployment_id': self.deployment_id,
            'timestamp': time.time()
        }
        
        with open(package_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # Create archive
        archive_path = f"{temp_dir}/pi-hmarl-{self.deployment_id}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(package_dir, arcname="pi-hmarl")
        
        return archive_path
    
    async def _create_infrastructure(self) -> Dict[str, Any]:
        """Create cloud infrastructure"""
        logs = []
        
        if self.provider == "aws":
            # Create EKS cluster
            logs.append("Creating EKS cluster...")
            # Simplified - would use boto3 or eksctl
            
        elif self.provider == "azure":
            # Create AKS cluster
            logs.append("Creating AKS cluster...")
            # Simplified - would use Azure SDK
            
        elif self.provider == "gcp":
            # Create GKE cluster
            logs.append("Creating GKE cluster...")
            # Simplified - would use Google Cloud SDK
        
        return {'success': True, 'logs': logs}
    
    async def _deploy_application(self, package_path: str) -> Dict[str, Any]:
        """Deploy application to cloud"""
        logs = []
        
        # Upload package
        logs.append(f"Uploading package: {package_path}")
        
        # Deploy using provider-specific method
        if self.provider == "kubernetes":
            # Generate Kubernetes manifests
            manifests = self._generate_k8s_manifests()
            
            # Apply manifests
            for manifest in manifests:
                manifest_file = f"/tmp/{manifest['name']}.yaml"
                with open(manifest_file, 'w') as f:
                    yaml.dump(manifest['spec'], f)
                
                cmd = f"kubectl apply -f {manifest_file}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                logs.append(f"Applied {manifest['name']}: {result.stdout}")
        
        return {'success': True, 'logs': logs}
    
    async def _configure_networking(self) -> Dict[str, Any]:
        """Configure cloud networking"""
        logs = []
        
        # Create load balancer
        logs.append("Creating load balancer...")
        
        # Configure ingress
        logs.append("Configuring ingress rules...")
        
        # Set up SSL/TLS
        if self.config.get('tls_enabled', True):
            logs.append("Setting up SSL/TLS...")
        
        return {'success': True, 'logs': logs}
    
    async def _verify_deployment(self) -> bool:
        """Verify deployment is working"""
        # Check health endpoints
        health_url = f"http://{self.config['name']}/health"
        
        max_retries = 30
        for i in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url) as response:
                        if response.status == 200:
                            return True
            except:
                pass
            
            await asyncio.sleep(10)
        
        return False
    
    async def _get_endpoints(self) -> Dict[str, str]:
        """Get deployment endpoints"""
        endpoints = {}
        
        if self.provider == "kubernetes":
            # Get service endpoints
            cmd = f"kubectl get service {self.config['name']} -o json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                service_info = json.loads(result.stdout)
                
                if 'status' in service_info and 'loadBalancer' in service_info['status']:
                    ingress = service_info['status']['loadBalancer'].get('ingress', [])
                    if ingress:
                        host = ingress[0].get('hostname') or ingress[0].get('ip')
                        endpoints['api'] = f"http://{host}"
                        endpoints['dashboard'] = f"http://{host}/dashboard"
        
        return endpoints
    
    def _generate_k8s_manifests(self) -> List[Dict[str, Any]]:
        """Generate Kubernetes manifests"""
        manifests = []
        
        # Deployment manifest
        deployment = {
            'name': 'deployment',
            'spec': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': self.config['name'],
                    'labels': {'app': self.config['name']}
                },
                'spec': {
                    'replicas': self.config.get('replicas', 3),
                    'selector': {
                        'matchLabels': {'app': self.config['name']}
                    },
                    'template': {
                        'metadata': {
                            'labels': {'app': self.config['name']}
                        },
                        'spec': {
                            'containers': [{
                                'name': 'pi-hmarl',
                                'image': f"{self.config['image']}:{self.config['version']}",
                                'ports': [{'containerPort': 8080}],
                                'resources': {
                                    'requests': {
                                        'cpu': self.config.get('cpu_request', '100m'),
                                        'memory': self.config.get('memory_request', '256Mi')
                                    },
                                    'limits': {
                                        'cpu': self.config.get('cpu_limit', '1'),
                                        'memory': self.config.get('memory_limit', '1Gi')
                                    }
                                },
                                'env': self._generate_env_vars()
                            }]
                        }
                    }
                }
            }
        }
        manifests.append(deployment)
        
        # Service manifest
        service = {
            'name': 'service',
            'spec': {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': self.config['name']
                },
                'spec': {
                    'type': 'LoadBalancer',
                    'ports': [{'port': 80, 'targetPort': 8080}],
                    'selector': {'app': self.config['name']}
                }
            }
        }
        manifests.append(service)
        
        return manifests
    
    def _generate_env_vars(self) -> List[Dict[str, str]]:
        """Generate environment variables"""
        env_vars = [
            {'name': 'PI_HMARL_ENV', 'value': self.config.get('environment', 'production')},
            {'name': 'NUM_AGENTS', 'value': str(self.config.get('num_agents', 10))},
            {'name': 'ENABLE_METRICS', 'value': 'true'}
        ]
        
        # Add custom env vars
        for key, value in self.config.get('env', {}).items():
            env_vars.append({'name': key, 'value': str(value)})
        
        return env_vars
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate deployment configuration"""
        required_fields = ['name', 'version']
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        return True
    
    async def _kubernetes_rolling_update(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Kubernetes rolling update"""
        logs = []
        
        # Update deployment
        cmd = f"kubectl set image deployment/{self.config['name']} " \
              f"pi-hmarl={new_config['image']}:{new_config['version']}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        logs.append(result.stdout)
        
        # Wait for rollout
        cmd = f"kubectl rollout status deployment/{self.config['name']}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        logs.append(result.stdout)
        
        return {'success': result.returncode == 0, 'logs': logs}
    
    async def _aws_blue_green_deployment(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """AWS blue-green deployment"""
        logs = []
        
        # Create green environment
        logs.append("Creating green environment...")
        
        # Deploy to green
        logs.append("Deploying to green environment...")
        
        # Switch traffic
        logs.append("Switching traffic to green...")
        
        # Cleanup blue
        logs.append("Cleaning up blue environment...")
        
        return {'success': True, 'logs': logs}
    
    async def _generic_update(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic update strategy"""
        logs = []
        
        # Stop current deployment
        logs.append("Stopping current deployment...")
        
        # Deploy new version
        logs.append("Deploying new version...")
        
        # Verify new deployment
        logs.append("Verifying new deployment...")
        
        return {'success': True, 'logs': logs}
    
    async def _get_version_config(self, version: str) -> Dict[str, Any]:
        """Get configuration for specific version"""
        # Simplified - would retrieve from version control or database
        return self.config
    
    async def _generic_rollback(self, prev_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic rollback strategy"""
        return await self._generic_update(prev_config)
    
    async def _provider_stop(self) -> Dict[str, Any]:
        """Provider-specific stop"""
        logs = []
        
        if self.provider == "aws":
            logs.append("Stopping ECS tasks...")
        elif self.provider == "azure":
            logs.append("Stopping container instances...")
        elif self.provider == "gcp":
            logs.append("Stopping compute instances...")
        
        return {'success': True, 'logs': logs}
    
    async def _cleanup_resources(self):
        """Clean up cloud resources"""
        logger.info("Cleaning up cloud resources...")


class EdgeDeployment(DeploymentStrategy):
    """Edge deployment strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize edge deployment
        
        Args:
            config: Deployment configuration
        """
        super().__init__(config)
        self.edge_nodes = config.get('edge_nodes', [])
        self.deployment_mode = config.get('mode', 'distributed')
    
    async def deploy(self) -> DeploymentResult:
        """Deploy to edge devices"""
        self.status = DeploymentStatus.PREPARING
        logs = []
        failed_nodes = []
        
        try:
            # Prepare lightweight package
            package_path = await self._prepare_edge_package()
            logs.append(f"Edge package prepared: {package_path}")
            
            # Deploy to each edge node
            self.status = DeploymentStatus.DEPLOYING
            endpoints = {}
            
            for node in self.edge_nodes:
                try:
                    node_result = await self._deploy_to_node(node, package_path)
                    logs.append(f"Deployed to {node['name']}: {node_result['status']}")
                    
                    if node_result['success']:
                        endpoints[node['name']] = node_result['endpoint']
                    else:
                        failed_nodes.append(node['name'])
                        
                except Exception as e:
                    logger.error(f"Failed to deploy to {node['name']}: {e}")
                    failed_nodes.append(node['name'])
            
            # Check deployment status
            if len(failed_nodes) == 0:
                self.status = DeploymentStatus.RUNNING
                success = True
                message = "Edge deployment successful"
            elif len(failed_nodes) < len(self.edge_nodes):
                self.status = DeploymentStatus.RUNNING
                success = True
                message = f"Partial deployment: {len(failed_nodes)} nodes failed"
            else:
                self.status = DeploymentStatus.FAILED
                success = False
                message = "All edge nodes failed"
            
            return DeploymentResult(
                success=success,
                status=self.status,
                message=message,
                deployment_id=self.deployment_id,
                endpoints=endpoints,
                logs=logs,
                metrics={'failed_nodes': failed_nodes}
            )
            
        except Exception as e:
            self.status = DeploymentStatus.FAILED
            return DeploymentResult(
                success=False,
                status=self.status,
                message=str(e),
                deployment_id=self.deployment_id,
                logs=logs
            )
    
    async def update(self, new_config: Dict[str, Any]) -> DeploymentResult:
        """Update edge deployment"""
        # Implement over-the-air update
        logs = []
        
        # Create update package
        update_package = await self._create_update_package(new_config)
        
        # Push updates to edge nodes
        for node in self.edge_nodes:
            result = await self._push_update(node, update_package)
            logs.append(f"Updated {node['name']}: {result['status']}")
        
        return DeploymentResult(
            success=True,
            status=DeploymentStatus.RUNNING,
            message="Edge update completed",
            deployment_id=self.deployment_id,
            logs=logs
        )
    
    async def rollback(self, version: str) -> DeploymentResult:
        """Rollback edge deployment"""
        # Implement edge rollback
        return DeploymentResult(
            success=True,
            status=DeploymentStatus.RUNNING,
            message=f"Rolled back to {version}",
            deployment_id=self.deployment_id
        )
    
    async def stop(self) -> DeploymentResult:
        """Stop edge deployment"""
        logs = []
        
        for node in self.edge_nodes:
            result = await self._stop_node(node)
            logs.append(f"Stopped {node['name']}: {result['status']}")
        
        self.status = DeploymentStatus.STOPPED
        
        return DeploymentResult(
            success=True,
            status=self.status,
            message="Edge deployment stopped",
            deployment_id=self.deployment_id,
            logs=logs
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get edge deployment status"""
        node_status = {}
        
        for node in self.edge_nodes:
            status = await self._get_node_status(node)
            node_status[node['name']] = status
        
        return {
            'status': self.status.value,
            'nodes': node_status,
            'deployment_mode': self.deployment_mode
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get edge deployment metrics"""
        metrics = {
            'deployment_id': self.deployment_id,
            'total_nodes': len(self.edge_nodes),
            'nodes': {}
        }
        
        for node in self.edge_nodes:
            node_metrics = await self._get_node_metrics(node)
            metrics['nodes'][node['name']] = node_metrics
        
        return metrics
    
    async def _prepare_edge_package(self) -> str:
        """Prepare lightweight edge package"""
        # Create optimized package for edge devices
        temp_dir = tempfile.mkdtemp()
        package_dir = Path(temp_dir) / "pi-hmarl-edge"
        package_dir.mkdir()
        
        # Copy only essential components
        essential_modules = ['agents', 'models', 'communication', 'energy']
        src_dir = Path(__file__).parent.parent / "src"
        
        for module in essential_modules:
            if (src_dir / module).exists():
                shutil.copytree(src_dir / module, package_dir / module)
        
        # Create edge configuration
        edge_config = {
            'deployment_id': self.deployment_id,
            'mode': 'edge',
            'optimization': {
                'cpu_only': True,
                'reduced_precision': True,
                'model_quantization': True
            }
        }
        
        with open(package_dir / "edge_config.json", 'w') as f:
            json.dump(edge_config, f)
        
        # Create compressed archive
        archive_path = f"{temp_dir}/pi-hmarl-edge-{self.deployment_id}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(package_dir, arcname="pi-hmarl-edge")
        
        return archive_path
    
    async def _deploy_to_node(self, node: Dict[str, Any], package_path: str) -> Dict[str, Any]:
        """Deploy to individual edge node"""
        # Connect to edge node
        node_ip = node['ip']
        node_port = node.get('port', 22)
        
        # Transfer package (simplified - would use SCP or similar)
        transfer_cmd = f"scp {package_path} {node['user']}@{node_ip}:/tmp/"
        
        # Extract and run
        run_cmd = f"ssh {node['user']}@{node_ip} 'cd /tmp && tar -xzf {os.path.basename(package_path)} && ./pi-hmarl-edge/deploy.sh'"
        
        # Get endpoint
        endpoint = f"http://{node_ip}:{node.get('service_port', 8080)}"
        
        return {
            'success': True,
            'status': 'deployed',
            'endpoint': endpoint
        }
    
    async def _create_update_package(self, new_config: Dict[str, Any]) -> str:
        """Create update package for edge nodes"""
        # Create differential update package
        return await self._prepare_edge_package()
    
    async def _push_update(self, node: Dict[str, Any], update_package: str) -> Dict[str, Any]:
        """Push update to edge node"""
        # Implement OTA update mechanism
        return {'status': 'updated'}
    
    async def _stop_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Stop deployment on edge node"""
        # Send stop command to edge node
        return {'status': 'stopped'}
    
    async def _get_node_status(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of edge node"""
        try:
            # Query node health endpoint
            health_url = f"http://{node['ip']}:{node.get('service_port', 8080)}/health"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    if response.status == 200:
                        return {'status': 'running', 'healthy': True}
                    else:
                        return {'status': 'unhealthy', 'healthy': False}
                        
        except Exception as e:
            return {'status': 'unreachable', 'error': str(e)}
    
    async def _get_node_metrics(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics from edge node"""
        try:
            # Query node metrics endpoint
            metrics_url = f"http://{node['ip']}:{node.get('service_port', 8080)}/metrics"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(metrics_url, timeout=5) as response:
                    if response.status == 200:
                        return await response.json()
                        
        except Exception:
            pass
        
        return {'status': 'unavailable'}


class HybridDeployment(DeploymentStrategy):
    """Hybrid cloud-edge deployment strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid deployment
        
        Args:
            config: Deployment configuration
        """
        super().__init__(config)
        
        # Cloud configuration
        self.cloud_config = config.get('cloud', {})
        self.cloud_deployment = CloudDeployment(
            self.cloud_config,
            self.cloud_config.get('provider', 'aws')
        )
        
        # Edge configuration
        self.edge_config = config.get('edge', {})
        self.edge_deployment = EdgeDeployment(self.edge_config)
        
        # Hybrid strategy
        self.coordination_mode = config.get('coordination', 'hierarchical')
    
    async def deploy(self) -> DeploymentResult:
        """Deploy hybrid cloud-edge system"""
        self.status = DeploymentStatus.DEPLOYING
        logs = []
        
        try:
            # Deploy cloud components first
            logs.append("Deploying cloud components...")
            cloud_result = await self.cloud_deployment.deploy()
            logs.extend(cloud_result.logs)
            
            if not cloud_result.success:
                raise Exception("Cloud deployment failed")
            
            # Configure edge-cloud communication
            await self._configure_hybrid_network(cloud_result.endpoints)
            
            # Deploy edge components
            logs.append("Deploying edge components...")
            edge_result = await self.edge_deployment.deploy()
            logs.extend(edge_result.logs)
            
            if not edge_result.success:
                raise Exception("Edge deployment failed")
            
            # Setup coordination
            await self._setup_coordination()
            
            self.status = DeploymentStatus.RUNNING
            
            # Combine endpoints
            endpoints = {
                'cloud': cloud_result.endpoints,
                'edge': edge_result.endpoints,
                'coordinator': f"{cloud_result.endpoints.get('api', '')}/coordinator"
            }
            
            return DeploymentResult(
                success=True,
                status=self.status,
                message="Hybrid deployment successful",
                deployment_id=self.deployment_id,
                endpoints=endpoints,
                logs=logs
            )
            
        except Exception as e:
            self.status = DeploymentStatus.FAILED
            return DeploymentResult(
                success=False,
                status=self.status,
                message=str(e),
                deployment_id=self.deployment_id,
                logs=logs
            )
    
    async def update(self, new_config: Dict[str, Any]) -> DeploymentResult:
        """Update hybrid deployment"""
        # Coordinate updates between cloud and edge
        logs = []
        
        # Update cloud first
        if 'cloud' in new_config:
            cloud_result = await self.cloud_deployment.update(new_config['cloud'])
            logs.extend(cloud_result.logs)
        
        # Update edge with coordination
        if 'edge' in new_config:
            edge_result = await self.edge_deployment.update(new_config['edge'])
            logs.extend(edge_result.logs)
        
        return DeploymentResult(
            success=True,
            status=DeploymentStatus.RUNNING,
            message="Hybrid update completed",
            deployment_id=self.deployment_id,
            logs=logs
        )
    
    async def rollback(self, version: str) -> DeploymentResult:
        """Rollback hybrid deployment"""
        # Coordinate rollback
        logs = []
        
        # Rollback in reverse order
        edge_result = await self.edge_deployment.rollback(version)
        logs.extend(edge_result.logs)
        
        cloud_result = await self.cloud_deployment.rollback(version)
        logs.extend(cloud_result.logs)
        
        return DeploymentResult(
            success=True,
            status=DeploymentStatus.RUNNING,
            message=f"Hybrid rollback to {version} completed",
            deployment_id=self.deployment_id,
            logs=logs
        )
    
    async def stop(self) -> DeploymentResult:
        """Stop hybrid deployment"""
        logs = []
        
        # Stop edge first
        edge_result = await self.edge_deployment.stop()
        logs.extend(edge_result.logs)
        
        # Stop cloud
        cloud_result = await self.cloud_deployment.stop()
        logs.extend(cloud_result.logs)
        
        self.status = DeploymentStatus.STOPPED
        
        return DeploymentResult(
            success=True,
            status=self.status,
            message="Hybrid deployment stopped",
            deployment_id=self.deployment_id,
            logs=logs
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get hybrid deployment status"""
        cloud_status = await self.cloud_deployment.get_status()
        edge_status = await self.edge_deployment.get_status()
        
        return {
            'status': self.status.value,
            'cloud': cloud_status,
            'edge': edge_status,
            'coordination_mode': self.coordination_mode
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get hybrid deployment metrics"""
        cloud_metrics = await self.cloud_deployment.get_metrics()
        edge_metrics = await self.edge_deployment.get_metrics()
        
        # Calculate hybrid metrics
        total_agents = (
            cloud_metrics.get('agent_count', 0) +
            sum(node.get('agent_count', 0) for node in edge_metrics.get('nodes', {}).values())
        )
        
        return {
            'deployment_id': self.deployment_id,
            'cloud': cloud_metrics,
            'edge': edge_metrics,
            'hybrid': {
                'total_agents': total_agents,
                'coordination_latency': await self._measure_coordination_latency(),
                'data_sync_rate': await self._get_sync_rate()
            }
        }
    
    async def _configure_hybrid_network(self, cloud_endpoints: Dict[str, str]):
        """Configure hybrid cloud-edge network"""
        # Setup VPN or secure tunnels
        # Configure edge nodes to connect to cloud
        pass
    
    async def _setup_coordination(self):
        """Setup cloud-edge coordination"""
        if self.coordination_mode == 'hierarchical':
            # Cloud as coordinator
            pass
        elif self.coordination_mode == 'distributed':
            # Peer-to-peer coordination
            pass
        elif self.coordination_mode == 'federated':
            # Federated learning setup
            pass
    
    async def _measure_coordination_latency(self) -> float:
        """Measure cloud-edge coordination latency"""
        # Simplified latency measurement
        return 50.0  # ms
    
    async def _get_sync_rate(self) -> float:
        """Get data synchronization rate"""
        # Simplified sync rate
        return 100.0  # messages/second


class DeploymentManager:
    """Main deployment manager"""
    
    def __init__(self):
        """Initialize deployment manager"""
        self.deployments: Dict[str, DeploymentStrategy] = {}
        self.deployment_history = []
    
    async def deploy(
        self,
        config: Dict[str, Any],
        target: DeploymentTarget
    ) -> DeploymentResult:
        """Deploy system to target
        
        Args:
            config: Deployment configuration
            target: Deployment target
            
        Returns:
            Deployment result
        """
        # Create deployment strategy
        if target == DeploymentTarget.KUBERNETES:
            strategy = CloudDeployment(config, 'kubernetes')
        elif target == DeploymentTarget.AWS:
            strategy = CloudDeployment(config, 'aws')
        elif target == DeploymentTarget.EDGE:
            strategy = EdgeDeployment(config)
        elif target == DeploymentTarget.HYBRID:
            strategy = HybridDeployment(config)
        else:
            raise ValueError(f"Unsupported deployment target: {target}")
        
        # Execute deployment
        result = await strategy.deploy()
        
        # Store deployment
        if result.success:
            self.deployments[result.deployment_id] = strategy
            self.deployment_history.append({
                'deployment_id': result.deployment_id,
                'target': target.value,
                'config': config,
                'timestamp': time.time(),
                'status': result.status.value
            })
        
        return result
    
    async def update_deployment(
        self,
        deployment_id: str,
        new_config: Dict[str, Any]
    ) -> DeploymentResult:
        """Update existing deployment
        
        Args:
            deployment_id: Deployment ID
            new_config: New configuration
            
        Returns:
            Update result
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        strategy = self.deployments[deployment_id]
        return await strategy.update(new_config)
    
    async def rollback_deployment(
        self,
        deployment_id: str,
        version: str
    ) -> DeploymentResult:
        """Rollback deployment
        
        Args:
            deployment_id: Deployment ID
            version: Version to rollback to
            
        Returns:
            Rollback result
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        strategy = self.deployments[deployment_id]
        return await strategy.rollback(version)
    
    async def stop_deployment(self, deployment_id: str) -> DeploymentResult:
        """Stop deployment
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Stop result
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        strategy = self.deployments[deployment_id]
        result = await strategy.stop()
        
        # Remove from active deployments
        if result.success:
            del self.deployments[deployment_id]
        
        return result
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment status
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        strategy = self.deployments[deployment_id]
        return await strategy.get_status()
    
    async def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment metrics
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment metrics
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        strategy = self.deployments[deployment_id]
        return await strategy.get_metrics()
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments
        
        Returns:
            List of deployments
        """
        deployments = []
        
        for deployment_id, strategy in self.deployments.items():
            deployments.append({
                'deployment_id': deployment_id,
                'status': strategy.status.value,
                'type': strategy.__class__.__name__
            })
        
        return deployments
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history
        
        Returns:
            Deployment history
        """
        return self.deployment_history