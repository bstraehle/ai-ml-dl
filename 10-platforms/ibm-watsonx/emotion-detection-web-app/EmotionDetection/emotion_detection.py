import json, requests, operator

def emotion_detector(text_to_analyse):
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    body = { "raw_document": { "text": text_to_analyse } }
    result = requests.post(url, json = body, headers = headers)
    result_json = json.loads(result.text)
    emotions = result_json["emotionPredictions"][0]["emotion"]
    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": sorted(emotions.items(), key = operator.itemgetter(1), reverse = True)[0][0]
    }