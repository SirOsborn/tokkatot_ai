"""
Centroid-Based Object Tracker
Tracks detected objects (feces) across frames to avoid double-counting.
Uses centroid distance matching with scipy spatial distance.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist


class CentroidTracker:
    """
    Track detected objects (feces) across frames to avoid double-counting.
    Uses centroid distance matching.
    """
    
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.maxDisappeared = maxDisappeared
        self.object_classes = {}
        self.object_confidence = {}
        
    def register(self, centroid, class_name, confidence):
        """Register a new object."""
        self.objects[self.nextObjectID] = centroid
        self.object_classes[self.nextObjectID] = class_name
        self.object_confidence[self.nextObjectID] = confidence
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        """Deregister an object."""
        del self.objects[objectID]
        del self.object_classes[objectID]
        del self.object_confidence[objectID]
        del self.disappeared[objectID]
        
    def update(self, rects: List[Tuple], classes: List[str], confidences: List[float]):
        """Update tracker with new detections."""
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        
        inputCentroids = np.zeros((len(rects), 2))
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = (startX + endX) // 2
            cY = (startY + endY) // 2
            inputCentroids[i] = [cX, cY]
        
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], classes[i], confidences[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = np.array([self.objects[objID] for objID in objectIDs])
            
            D = cdist(objectCentroids, inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D[rows, :].argmin(axis=1)
            
            usedRows = set()
            usedCols = set()
            
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > 50:  # Distance threshold
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.object_classes[objectID] = classes[col]
                self.object_confidence[objectID] = confidences[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], classes[col], confidences[col])
        
        return self.objects
