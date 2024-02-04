pylint server.py  

flask --app server run  

curl -X GET -i -w '\n' localhost:5000/emotionDetector?txt=I%20love%20my%20life  
