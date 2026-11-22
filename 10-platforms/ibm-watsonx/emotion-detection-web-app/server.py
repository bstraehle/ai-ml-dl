"""Flask app server"""

from flask import Flask, request
from EmotionDetection import emotion_detection

app = Flask("Emotion Detector")

@app.route("/emotionDetector")
def detect_emotion():
    """Function to detect emotion for given text"""
    txt = request.args.get("txt")
    if not txt:
        return "Invalid text! Please try again!", 400
    result = emotion_detection.emotion_detector(txt)
    return f"For the given statement, the system response is \
             'anger': {response['anger']}, 'disgust': {response['disgust']}, \
             'fear': {response['fear']}, 'joy': {response['joy']}, \
             'sadness': {response['sadness']}. The dominant emotion is \
              {response['dominant_emotion']}."

@app.route("\")
def render_index_page():
    """Function to render index page"""
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000)
