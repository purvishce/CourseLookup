from dataclasses import dataclass
from datetime import date
from typing import Dict

@dataclass
class CourseRecord:
    '''id: int
    title: str
    summary: str
    full_text: str
    assigned_to: str
    category: str
    date: date
    status: str'''
    title: str
    description: str
    instructor: str
    rating: float
    reviewcount: int
    duration: str
    lectures: str
    level: str

@dataclass
class Chunk:
    text: str
    metadata: Dict
