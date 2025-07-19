"""Collision Detection System

This module implements efficient collision detection for multi-agent systems
with support for various shapes and spatial optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollisionShape:
    """Collision shape definition"""
    shape_type: str  # sphere, box, cylinder, mesh
    size: np.ndarray  # Shape-specific size parameters
    offset: np.ndarray = None  # Offset from object center
    
    def __post_init__(self):
        if self.offset is None:
            self.offset = np.zeros(3)


@dataclass
class CollisionInfo:
    """Collision information"""
    object1_id: int
    object2_id: int
    contact_point: np.ndarray
    contact_normal: np.ndarray
    penetration_depth: float
    relative_velocity: np.ndarray
    impulse: float = 0.0


class SpatialHash:
    """Spatial hash for broad-phase collision detection"""
    
    def __init__(self, cell_size: float = 5.0):
        """Initialize spatial hash
        
        Args:
            cell_size: Size of hash grid cells
        """
        self.cell_size = cell_size
        self.hash_table: Dict[Tuple[int, int, int], List[int]] = {}
    
    def clear(self):
        """Clear the spatial hash"""
        self.hash_table.clear()
    
    def insert(self, object_id: int, position: np.ndarray, radius: float):
        """Insert object into spatial hash
        
        Args:
            object_id: Object identifier
            position: Object position
            radius: Bounding radius
        """
        # Get all cells the object overlaps
        min_cell = self._position_to_cell(position - radius)
        max_cell = self._position_to_cell(position + radius)
        
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell_key = (x, y, z)
                    if cell_key not in self.hash_table:
                        self.hash_table[cell_key] = []
                    self.hash_table[cell_key].append(object_id)
    
    def query(self, position: np.ndarray, radius: float) -> List[int]:
        """Query objects near a position
        
        Args:
            position: Query position
            radius: Query radius
            
        Returns:
            List of potentially colliding object IDs
        """
        potential_collisions = set()
        
        min_cell = self._position_to_cell(position - radius)
        max_cell = self._position_to_cell(position + radius)
        
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell_key = (x, y, z)
                    if cell_key in self.hash_table:
                        potential_collisions.update(self.hash_table[cell_key])
        
        return list(potential_collisions)
    
    def _position_to_cell(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert position to cell coordinates"""
        return (
            int(np.floor(position[0] / self.cell_size)),
            int(np.floor(position[1] / self.cell_size)),
            int(np.floor(position[2] / self.cell_size))
        )


class CollisionDetector:
    """Efficient collision detection system"""
    
    def __init__(
        self,
        enable_spatial_hash: bool = True,
        collision_margin: float = 0.01,
        max_contacts_per_pair: int = 4
    ):
        """Initialize collision detector
        
        Args:
            enable_spatial_hash: Use spatial hashing for broad phase
            collision_margin: Collision detection margin
            max_contacts_per_pair: Maximum contacts per collision pair
        """
        self.enable_spatial_hash = enable_spatial_hash
        self.collision_margin = collision_margin
        self.max_contacts_per_pair = max_contacts_per_pair
        
        # Object registry
        self.objects: Dict[int, Dict[str, Any]] = {}
        
        # Spatial hash for broad phase
        self.spatial_hash = SpatialHash() if enable_spatial_hash else None
        
        # Collision groups and masks
        self.collision_groups: Dict[int, int] = {}  # object_id -> group
        self.collision_masks: Dict[int, int] = {}   # object_id -> mask
        
        logger.info("Initialized CollisionDetector")
    
    def add_object(
        self,
        object_id: int,
        shape: CollisionShape,
        position: np.ndarray,
        orientation: np.ndarray,
        collision_group: int = 1,
        collision_mask: int = -1
    ):
        """Add object to collision detection
        
        Args:
            object_id: Object identifier
            shape: Collision shape
            position: Object position
            orientation: Object orientation (quaternion)
            collision_group: Collision group (bit flag)
            collision_mask: Collision mask (bit flag)
        """
        self.objects[object_id] = {
            "shape": shape,
            "position": position.copy(),
            "orientation": orientation.copy(),
            "bounding_radius": self._calculate_bounding_radius(shape)
        }
        
        self.collision_groups[object_id] = collision_group
        self.collision_masks[object_id] = collision_mask
        
        logger.debug(f"Added object {object_id} to collision detection")
    
    def remove_object(self, object_id: int):
        """Remove object from collision detection"""
        if object_id in self.objects:
            del self.objects[object_id]
            del self.collision_groups[object_id]
            del self.collision_masks[object_id]
    
    def update_object(
        self,
        object_id: int,
        position: np.ndarray,
        orientation: np.ndarray
    ):
        """Update object transform"""
        if object_id in self.objects:
            self.objects[object_id]["position"] = position.copy()
            self.objects[object_id]["orientation"] = orientation.copy()
    
    def detect_collisions(self) -> List[CollisionInfo]:
        """Detect all collisions
        
        Returns:
            List of collision information
        """
        collisions = []
        
        # Update spatial hash if enabled
        if self.spatial_hash:
            self._update_spatial_hash()
        
        # Check all pairs
        object_ids = list(self.objects.keys())
        
        for i, id1 in enumerate(object_ids):
            # Get potential collisions
            if self.spatial_hash:
                obj1 = self.objects[id1]
                potential_ids = self.spatial_hash.query(
                    obj1["position"], obj1["bounding_radius"]
                )
                # Filter to avoid duplicate checks
                potential_ids = [id2 for id2 in potential_ids 
                               if id2 in object_ids[i+1:]]
            else:
                potential_ids = object_ids[i+1:]
            
            for id2 in potential_ids:
                # Check collision groups/masks
                if not self._should_collide(id1, id2):
                    continue
                
                # Narrow phase collision detection
                collision_info = self._check_collision_pair(id1, id2)
                if collision_info:
                    collisions.append(collision_info)
        
        return collisions
    
    def _update_spatial_hash(self):
        """Update spatial hash with current object positions"""
        self.spatial_hash.clear()
        
        for object_id, obj_data in self.objects.items():
            self.spatial_hash.insert(
                object_id,
                obj_data["position"],
                obj_data["bounding_radius"]
            )
    
    def _should_collide(self, id1: int, id2: int) -> bool:
        """Check if two objects should collide based on groups/masks"""
        group1 = self.collision_groups.get(id1, 1)
        group2 = self.collision_groups.get(id2, 1)
        mask1 = self.collision_masks.get(id1, -1)
        mask2 = self.collision_masks.get(id2, -1)
        
        # Objects collide if each object's group matches the other's mask
        return (group1 & mask2) != 0 and (group2 & mask1) != 0
    
    def _check_collision_pair(self, id1: int, id2: int) -> Optional[CollisionInfo]:
        """Check collision between two objects"""
        obj1 = self.objects[id1]
        obj2 = self.objects[id2]
        
        shape1 = obj1["shape"]
        shape2 = obj2["shape"]
        
        # Dispatch to appropriate collision test
        if shape1.shape_type == "sphere" and shape2.shape_type == "sphere":
            return self._sphere_sphere_collision(id1, obj1, id2, obj2)
        elif shape1.shape_type == "box" and shape2.shape_type == "box":
            return self._box_box_collision(id1, obj1, id2, obj2)
        elif (shape1.shape_type == "sphere" and shape2.shape_type == "box") or \
             (shape1.shape_type == "box" and shape2.shape_type == "sphere"):
            return self._sphere_box_collision(id1, obj1, id2, obj2)
        else:
            # Fallback to bounding sphere test
            return self._bounding_sphere_collision(id1, obj1, id2, obj2)
    
    def _sphere_sphere_collision(
        self, id1: int, obj1: Dict, id2: int, obj2: Dict
    ) -> Optional[CollisionInfo]:
        """Check collision between two spheres"""
        pos1 = obj1["position"] + obj1["shape"].offset
        pos2 = obj2["position"] + obj2["shape"].offset
        
        radius1 = obj1["shape"].size[0]
        radius2 = obj2["shape"].size[0]
        
        # Calculate distance
        delta = pos2 - pos1
        distance = np.linalg.norm(delta)
        
        # Check collision
        if distance < radius1 + radius2 + self.collision_margin:
            # Calculate collision info
            if distance > 0:
                normal = delta / distance
            else:
                normal = np.array([0, 0, 1])  # Arbitrary normal for exact overlap
            
            penetration = radius1 + radius2 - distance
            contact_point = pos1 + normal * (radius1 - penetration / 2)
            
            return CollisionInfo(
                object1_id=id1,
                object2_id=id2,
                contact_point=contact_point,
                contact_normal=normal,
                penetration_depth=penetration,
                relative_velocity=np.zeros(3)  # Would need velocity info
            )
        
        return None
    
    def _box_box_collision(
        self, id1: int, obj1: Dict, id2: int, obj2: Dict
    ) -> Optional[CollisionInfo]:
        """Check collision between two boxes (simplified AABB test)"""
        # This is a simplified axis-aligned bounding box test
        # For proper oriented box collision, use SAT algorithm
        
        pos1 = obj1["position"] + obj1["shape"].offset
        pos2 = obj2["position"] + obj2["shape"].offset
        
        half_size1 = obj1["shape"].size / 2
        half_size2 = obj2["shape"].size / 2
        
        # Check overlap on each axis
        overlap = np.zeros(3)
        for i in range(3):
            min1 = pos1[i] - half_size1[i]
            max1 = pos1[i] + half_size1[i]
            min2 = pos2[i] - half_size2[i]
            max2 = pos2[i] + half_size2[i]
            
            if max1 < min2 or max2 < min1:
                return None  # No collision
            
            # Calculate overlap
            overlap[i] = min(max1, max2) - max(min1, min2)
        
        # Find axis of minimum penetration
        min_axis = np.argmin(overlap)
        penetration = overlap[min_axis]
        
        # Calculate normal
        normal = np.zeros(3)
        normal[min_axis] = 1.0 if pos1[min_axis] < pos2[min_axis] else -1.0
        
        # Calculate contact point (center of overlap region)
        contact_point = (pos1 + pos2) / 2
        
        return CollisionInfo(
            object1_id=id1,
            object2_id=id2,
            contact_point=contact_point,
            contact_normal=normal,
            penetration_depth=penetration,
            relative_velocity=np.zeros(3)
        )
    
    def _sphere_box_collision(
        self, id1: int, obj1: Dict, id2: int, obj2: Dict
    ) -> Optional[CollisionInfo]:
        """Check collision between sphere and box"""
        # Ensure sphere is first
        if obj1["shape"].shape_type != "sphere":
            id1, id2 = id2, id1
            obj1, obj2 = obj2, obj1
        
        sphere_pos = obj1["position"] + obj1["shape"].offset
        sphere_radius = obj1["shape"].size[0]
        
        box_pos = obj2["position"] + obj2["shape"].offset
        box_half_size = obj2["shape"].size / 2
        
        # Find closest point on box to sphere center
        closest_point = np.clip(
            sphere_pos,
            box_pos - box_half_size,
            box_pos + box_half_size
        )
        
        # Check distance
        delta = sphere_pos - closest_point
        distance = np.linalg.norm(delta)
        
        if distance < sphere_radius + self.collision_margin:
            # Calculate collision info
            if distance > 0:
                normal = delta / distance
            else:
                # Sphere center inside box
                # Find closest face
                face_distances = np.concatenate([
                    box_half_size - (sphere_pos - box_pos),
                    box_half_size + (sphere_pos - box_pos)
                ])
                min_face = np.argmin(face_distances)
                normal = np.zeros(3)
                normal[min_face % 3] = -1 if min_face < 3 else 1
            
            penetration = sphere_radius - distance
            
            return CollisionInfo(
                object1_id=id1,
                object2_id=id2,
                contact_point=closest_point,
                contact_normal=normal,
                penetration_depth=penetration,
                relative_velocity=np.zeros(3)
            )
        
        return None
    
    def _bounding_sphere_collision(
        self, id1: int, obj1: Dict, id2: int, obj2: Dict
    ) -> Optional[CollisionInfo]:
        """Fallback bounding sphere collision test"""
        pos1 = obj1["position"]
        pos2 = obj2["position"]
        
        radius1 = obj1["bounding_radius"]
        radius2 = obj2["bounding_radius"]
        
        distance = np.linalg.norm(pos2 - pos1)
        
        if distance < radius1 + radius2:
            # Simple collision info
            normal = (pos2 - pos1) / distance if distance > 0 else np.array([0, 0, 1])
            
            return CollisionInfo(
                object1_id=id1,
                object2_id=id2,
                contact_point=(pos1 + pos2) / 2,
                contact_normal=normal,
                penetration_depth=radius1 + radius2 - distance,
                relative_velocity=np.zeros(3)
            )
        
        return None
    
    def _calculate_bounding_radius(self, shape: CollisionShape) -> float:
        """Calculate bounding sphere radius for shape"""
        if shape.shape_type == "sphere":
            return shape.size[0]
        elif shape.shape_type == "box":
            return np.linalg.norm(shape.size) / 2
        elif shape.shape_type == "cylinder":
            radius = shape.size[0]
            height = shape.size[1]
            return np.sqrt(radius**2 + (height/2)**2)
        else:
            # Default conservative estimate
            return np.max(shape.size)