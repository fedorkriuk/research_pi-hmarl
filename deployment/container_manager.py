"""Container Management Module

This module provides container-based deployment support for Docker,
Kubernetes, and Docker Swarm.
"""

import os
import json
import yaml
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import docker
from kubernetes import client, config
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


class ContainerStatus(Enum):
    """Container status"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    EXITED = "exited"
    DEAD = "dead"
    REMOVING = "removing"


@dataclass
class ContainerConfig:
    """Container configuration"""
    name: str
    image: str
    tag: str = "latest"
    command: Optional[List[str]] = None
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, int] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=lambda: {
        'cpu_limit': '1',
        'memory_limit': '1G',
        'gpu_enabled': False
    })
    replicas: int = 1
    restart_policy: str = "on-failure"
    labels: Dict[str, str] = field(default_factory=dict)


class ContainerManager:
    """Base container manager"""
    
    def __init__(self):
        """Initialize container manager"""
        self.containers: Dict[str, Any] = {}
        
    async def create_container(self, config: ContainerConfig) -> str:
        """Create container
        
        Args:
            config: Container configuration
            
        Returns:
            Container ID
        """
        raise NotImplementedError
    
    async def start_container(self, container_id: str) -> bool:
        """Start container
        
        Args:
            container_id: Container ID
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    async def stop_container(self, container_id: str, timeout: int = 10) -> bool:
        """Stop container
        
        Args:
            container_id: Container ID
            timeout: Stop timeout
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    async def remove_container(self, container_id: str, force: bool = False) -> bool:
        """Remove container
        
        Args:
            container_id: Container ID
            force: Force removal
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    async def get_container_status(self, container_id: str) -> ContainerStatus:
        """Get container status
        
        Args:
            container_id: Container ID
            
        Returns:
            Container status
        """
        raise NotImplementedError
    
    async def get_container_logs(
        self,
        container_id: str,
        tail: int = 100
    ) -> List[str]:
        """Get container logs
        
        Args:
            container_id: Container ID
            tail: Number of lines
            
        Returns:
            Log lines
        """
        raise NotImplementedError


class DockerDeployment(ContainerManager):
    """Docker-based deployment"""
    
    def __init__(self):
        """Initialize Docker deployment"""
        super().__init__()
        
        try:
            self.client = docker.from_env()
            logger.info("Connected to Docker daemon")
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise
    
    async def create_container(self, config: ContainerConfig) -> str:
        """Create Docker container"""
        try:
            # Build container configuration
            container_config = {
                'image': f"{config.image}:{config.tag}",
                'name': config.name,
                'environment': config.environment,
                'labels': config.labels,
                'restart_policy': {
                    'Name': config.restart_policy,
                    'MaximumRetryCount': 3
                }
            }
            
            # Add command if specified
            if config.command:
                container_config['command'] = config.command
            
            # Add volumes
            if config.volumes:
                container_config['volumes'] = {
                    host: {'bind': container, 'mode': 'rw'}
                    for host, container in config.volumes.items()
                }
            
            # Add ports
            if config.ports:
                container_config['ports'] = {
                    f"{container}/tcp": host
                    for container, host in config.ports.items()
                }
            
            # Add resource limits
            if config.resources:
                container_config['cpu_quota'] = int(
                    float(config.resources.get('cpu_limit', '1')) * 100000
                )
                container_config['mem_limit'] = config.resources.get(
                    'memory_limit', '1G'
                )
                
                # GPU support
                if config.resources.get('gpu_enabled'):
                    container_config['runtime'] = 'nvidia'
                    container_config['environment']['NVIDIA_VISIBLE_DEVICES'] = 'all'
            
            # Create container
            container = self.client.containers.create(**container_config)
            self.containers[container.id] = container
            
            logger.info(f"Created container {config.name} ({container.id})")
            
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            raise
    
    async def start_container(self, container_id: str) -> bool:
        """Start Docker container"""
        try:
            if container_id in self.containers:
                container = self.containers[container_id]
            else:
                container = self.client.containers.get(container_id)
                self.containers[container_id] = container
            
            container.start()
            
            logger.info(f"Started container {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return False
    
    async def stop_container(self, container_id: str, timeout: int = 10) -> bool:
        """Stop Docker container"""
        try:
            if container_id in self.containers:
                container = self.containers[container_id]
            else:
                container = self.client.containers.get(container_id)
            
            container.stop(timeout=timeout)
            
            logger.info(f"Stopped container {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False
    
    async def remove_container(self, container_id: str, force: bool = False) -> bool:
        """Remove Docker container"""
        try:
            if container_id in self.containers:
                container = self.containers[container_id]
            else:
                container = self.client.containers.get(container_id)
            
            container.remove(force=force)
            
            if container_id in self.containers:
                del self.containers[container_id]
            
            logger.info(f"Removed container {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove container: {e}")
            return False
    
    async def get_container_status(self, container_id: str) -> ContainerStatus:
        """Get Docker container status"""
        try:
            if container_id in self.containers:
                container = self.containers[container_id]
            else:
                container = self.client.containers.get(container_id)
            
            container.reload()
            status = container.status
            
            return ContainerStatus(status)
            
        except Exception as e:
            logger.error(f"Failed to get container status: {e}")
            return ContainerStatus.DEAD
    
    async def get_container_logs(
        self,
        container_id: str,
        tail: int = 100
    ) -> List[str]:
        """Get Docker container logs"""
        try:
            if container_id in self.containers:
                container = self.containers[container_id]
            else:
                container = self.client.containers.get(container_id)
            
            logs = container.logs(tail=tail, stream=False)
            
            if isinstance(logs, bytes):
                logs = logs.decode('utf-8')
            
            return logs.split('\n')
            
        except Exception as e:
            logger.error(f"Failed to get container logs: {e}")
            return []
    
    async def build_image(
        self,
        dockerfile_path: Path,
        image_name: str,
        tag: str = "latest"
    ) -> bool:
        """Build Docker image
        
        Args:
            dockerfile_path: Dockerfile path
            image_name: Image name
            tag: Image tag
            
        Returns:
            Success status
        """
        try:
            # Build image
            image, logs = self.client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=dockerfile_path.name,
                tag=f"{image_name}:{tag}",
                rm=True,
                forcerm=True
            )
            
            # Log build output
            for log in logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
            
            logger.info(f"Built image {image_name}:{tag}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build image: {e}")
            return False
    
    async def push_image(
        self,
        image_name: str,
        tag: str = "latest",
        registry: Optional[str] = None
    ) -> bool:
        """Push image to registry
        
        Args:
            image_name: Image name
            tag: Image tag
            registry: Registry URL
            
        Returns:
            Success status
        """
        try:
            full_name = f"{image_name}:{tag}"
            
            if registry:
                # Tag for registry
                registry_name = f"{registry}/{full_name}"
                image = self.client.images.get(full_name)
                image.tag(registry_name)
                full_name = registry_name
            
            # Push image
            self.client.images.push(full_name)
            
            logger.info(f"Pushed image {full_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push image: {e}")
            return False


class KubernetesDeployment(ContainerManager):
    """Kubernetes-based deployment"""
    
    def __init__(self, kubeconfig: Optional[str] = None):
        """Initialize Kubernetes deployment
        
        Args:
            kubeconfig: Kubernetes config file path
        """
        super().__init__()
        
        try:
            if kubeconfig:
                config.load_kube_config(config_file=kubeconfig)
            else:
                # Try in-cluster config first, then default config
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()
            
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            
            logger.info("Connected to Kubernetes cluster")
            
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
            raise
    
    async def create_deployment(
        self,
        config: ContainerConfig,
        namespace: str = "default"
    ) -> str:
        """Create Kubernetes deployment
        
        Args:
            config: Container configuration
            namespace: Kubernetes namespace
            
        Returns:
            Deployment name
        """
        try:
            # Create deployment spec
            container = client.V1Container(
                name=config.name,
                image=f"{config.image}:{config.tag}",
                env=[
                    client.V1EnvVar(name=k, value=v)
                    for k, v in config.environment.items()
                ]
            )
            
            # Add command if specified
            if config.command:
                container.command = config.command
            
            # Add resource limits
            if config.resources:
                resources = client.V1ResourceRequirements(
                    limits={
                        'cpu': config.resources.get('cpu_limit', '1'),
                        'memory': config.resources.get('memory_limit', '1Gi')
                    },
                    requests={
                        'cpu': str(float(config.resources.get('cpu_limit', '1')) * 0.5),
                        'memory': config.resources.get('memory_limit', '512Mi')
                    }
                )
                container.resources = resources
            
            # Add volumes
            volumes = []
            volume_mounts = []
            
            for i, (host_path, container_path) in enumerate(config.volumes.items()):
                vol_name = f"{config.name}-vol-{i}"
                
                volumes.append(
                    client.V1Volume(
                        name=vol_name,
                        host_path=client.V1HostPathVolumeSource(path=host_path)
                    )
                )
                
                volume_mounts.append(
                    client.V1VolumeMount(
                        name=vol_name,
                        mount_path=container_path
                    )
                )
            
            if volume_mounts:
                container.volume_mounts = volume_mounts
            
            # Create pod spec
            pod_spec = client.V1PodSpec(
                containers=[container],
                volumes=volumes if volumes else None,
                restart_policy="Always"
            )
            
            # Create deployment
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(
                    name=config.name,
                    labels=config.labels
                ),
                spec=client.V1DeploymentSpec(
                    replicas=config.replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": config.name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": config.name}
                        ),
                        spec=pod_spec
                    )
                )
            )
            
            # Create deployment
            self.apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Created Kubernetes deployment {config.name}")
            
            return config.name
            
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            raise
    
    async def create_service(
        self,
        config: ContainerConfig,
        namespace: str = "default",
        service_type: str = "ClusterIP"
    ) -> str:
        """Create Kubernetes service
        
        Args:
            config: Container configuration
            namespace: Kubernetes namespace
            service_type: Service type
            
        Returns:
            Service name
        """
        try:
            # Create service spec
            ports = []
            for container_port, host_port in config.ports.items():
                ports.append(
                    client.V1ServicePort(
                        port=host_port,
                        target_port=int(container_port),
                        protocol="TCP"
                    )
                )
            
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(
                    name=f"{config.name}-svc",
                    labels=config.labels
                ),
                spec=client.V1ServiceSpec(
                    selector={"app": config.name},
                    ports=ports,
                    type=service_type
                )
            )
            
            # Create service
            self.v1.create_namespaced_service(
                namespace=namespace,
                body=service
            )
            
            logger.info(f"Created Kubernetes service {config.name}-svc")
            
            return f"{config.name}-svc"
            
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            raise
    
    async def scale_deployment(
        self,
        name: str,
        replicas: int,
        namespace: str = "default"
    ) -> bool:
        """Scale deployment
        
        Args:
            name: Deployment name
            replicas: Number of replicas
            namespace: Namespace
            
        Returns:
            Success status
        """
        try:
            # Get deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=namespace
            )
            
            # Update replicas
            deployment.spec.replicas = replicas
            
            # Patch deployment
            self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment {name} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False
    
    async def delete_deployment(
        self,
        name: str,
        namespace: str = "default"
    ) -> bool:
        """Delete deployment
        
        Args:
            name: Deployment name
            namespace: Namespace
            
        Returns:
            Success status
        """
        try:
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=namespace,
                propagation_policy='Foreground'
            )
            
            logger.info(f"Deleted deployment {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            return False
    
    async def get_pod_logs(
        self,
        pod_name: str,
        namespace: str = "default",
        tail: int = 100
    ) -> List[str]:
        """Get pod logs
        
        Args:
            pod_name: Pod name
            namespace: Namespace
            tail: Number of lines
            
        Returns:
            Log lines
        """
        try:
            logs = self.v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail
            )
            
            return logs.split('\n')
            
        except Exception as e:
            logger.error(f"Failed to get pod logs: {e}")
            return []


class DockerSwarmDeployment(ContainerManager):
    """Docker Swarm deployment"""
    
    def __init__(self):
        """Initialize Docker Swarm deployment"""
        super().__init__()
        
        try:
            self.client = docker.from_env()
            
            # Verify swarm mode
            if not self.client.swarm.attrs:
                logger.warning("Docker not in swarm mode, initializing...")
                self.client.swarm.init()
            
            logger.info("Connected to Docker Swarm")
            
        except Exception as e:
            logger.error(f"Failed to connect to Docker Swarm: {e}")
            raise
    
    async def create_service(self, config: ContainerConfig) -> str:
        """Create swarm service
        
        Args:
            config: Container configuration
            
        Returns:
            Service ID
        """
        try:
            # Build service spec
            service_spec = {
                'name': config.name,
                'image': f"{config.image}:{config.tag}",
                'env': [f"{k}={v}" for k, v in config.environment.items()],
                'labels': config.labels,
                'mode': {
                    'Replicated': {
                        'Replicas': config.replicas
                    }
                },
                'restart_policy': {
                    'Condition': config.restart_policy,
                    'MaxAttempts': 3
                }
            }
            
            # Add command
            if config.command:
                service_spec['command'] = config.command
            
            # Add mounts
            if config.volumes:
                service_spec['mounts'] = [
                    {
                        'Source': host,
                        'Target': container,
                        'Type': 'bind'
                    }
                    for host, container in config.volumes.items()
                ]
            
            # Add ports
            if config.ports:
                service_spec['endpoint_spec'] = {
                    'Ports': [
                        {
                            'Protocol': 'tcp',
                            'PublishedPort': host,
                            'TargetPort': int(container)
                        }
                        for container, host in config.ports.items()
                    ]
                }
            
            # Add resource limits
            if config.resources:
                service_spec['resources'] = {
                    'Limits': {
                        'NanoCPUs': int(float(config.resources.get('cpu_limit', '1')) * 1e9),
                        'MemoryBytes': self._parse_memory(
                            config.resources.get('memory_limit', '1G')
                        )
                    }
                }
            
            # Create service
            service = self.client.services.create(**service_spec)
            
            logger.info(f"Created swarm service {config.name} ({service.id})")
            
            return service.id
            
        except Exception as e:
            logger.error(f"Failed to create swarm service: {e}")
            raise
    
    async def update_service(
        self,
        service_id: str,
        **kwargs
    ) -> bool:
        """Update swarm service
        
        Args:
            service_id: Service ID
            **kwargs: Update parameters
            
        Returns:
            Success status
        """
        try:
            service = self.client.services.get(service_id)
            service.update(**kwargs)
            
            logger.info(f"Updated swarm service {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update service: {e}")
            return False
    
    async def scale_service(
        self,
        service_id: str,
        replicas: int
    ) -> bool:
        """Scale swarm service
        
        Args:
            service_id: Service ID
            replicas: Number of replicas
            
        Returns:
            Success status
        """
        try:
            service = self.client.services.get(service_id)
            service.scale(replicas)
            
            logger.info(f"Scaled service {service_id} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale service: {e}")
            return False
    
    async def remove_service(self, service_id: str) -> bool:
        """Remove swarm service
        
        Args:
            service_id: Service ID
            
        Returns:
            Success status
        """
        try:
            service = self.client.services.get(service_id)
            service.remove()
            
            logger.info(f"Removed swarm service {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove service: {e}")
            return False
    
    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to bytes
        
        Args:
            memory_str: Memory string (e.g., "1G", "512M")
            
        Returns:
            Memory in bytes
        """
        units = {
            'K': 1024,
            'M': 1024 * 1024,
            'G': 1024 * 1024 * 1024
        }
        
        for unit, multiplier in units.items():
            if memory_str.upper().endswith(unit):
                return int(float(memory_str[:-1]) * multiplier)
        
        # Default to bytes
        return int(memory_str)


# Example Dockerfile generation
def generate_dockerfile(
    base_image: str = "python:3.9-slim",
    workdir: str = "/app",
    requirements_file: str = "requirements.txt",
    entrypoint: List[str] = ["python", "-m", "src.main"]
) -> str:
    """Generate Dockerfile content
    
    Args:
        base_image: Base Docker image
        workdir: Working directory
        requirements_file: Requirements file
        entrypoint: Container entrypoint
        
    Returns:
        Dockerfile content
    """
    dockerfile = f"""# PI-HMARL Dockerfile
FROM {base_image}

# Set working directory
WORKDIR {workdir}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY {requirements_file} .

# Install Python dependencies
RUN pip install --no-cache-dir -r {requirements_file}

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 5000 8080 9090

# Set entrypoint
ENTRYPOINT {json.dumps(entrypoint)}
"""
    
    return dockerfile


# Example Kubernetes manifest generation
def generate_k8s_manifest(
    app_name: str,
    image: str,
    replicas: int = 3,
    cpu_limit: str = "1",
    memory_limit: str = "1Gi"
) -> Dict[str, Any]:
    """Generate Kubernetes manifest
    
    Args:
        app_name: Application name
        image: Docker image
        replicas: Number of replicas
        cpu_limit: CPU limit
        memory_limit: Memory limit
        
    Returns:
        Kubernetes manifest
    """
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": app_name,
            "labels": {
                "app": app_name,
                "component": "pi-hmarl"
            }
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app": app_name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": app_name
                    }
                },
                "spec": {
                    "containers": [{
                        "name": app_name,
                        "image": image,
                        "resources": {
                            "limits": {
                                "cpu": cpu_limit,
                                "memory": memory_limit
                            },
                            "requests": {
                                "cpu": str(float(cpu_limit) * 0.5),
                                "memory": "512Mi"
                            }
                        },
                        "env": [
                            {
                                "name": "POD_NAME",
                                "valueFrom": {
                                    "fieldRef": {
                                        "fieldPath": "metadata.name"
                                    }
                                }
                            },
                            {
                                "name": "POD_NAMESPACE",
                                "valueFrom": {
                                    "fieldRef": {
                                        "fieldPath": "metadata.namespace"
                                    }
                                }
                            }
                        ],
                        "livenessProbe": {
                            "httpGet": {
                                "path": "/health",
                                "port": 8080
                            },
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10
                        },
                        "readinessProbe": {
                            "httpGet": {
                                "path": "/ready",
                                "port": 8080
                            },
                            "initialDelaySeconds": 10,
                            "periodSeconds": 5
                        }
                    }]
                }
            }
        }
    }
    
    return manifest