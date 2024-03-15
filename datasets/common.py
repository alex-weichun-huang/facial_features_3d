"""
This is the emotion to label mapping shared across all the dataset.
This design choice is made to make merging of different datasets easier.
"""

registered_emotions = {
    # common emotions
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6,
    
    # dataset specific emotions
    "Ambiguous": 7,
    "Contempt": 8,
}