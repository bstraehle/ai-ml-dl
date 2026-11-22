Run the Flask backend, then send it a message:

```bash
python3.11 -m pip install flask
python3.11 gitconnect.py

curl -X POST -H "Content-Type: application/json" -d '{"user_message": "Python"}' http://127.0.0.1:5000/chat
```
