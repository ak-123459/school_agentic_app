import requests

url = "http://localhost:8000/chat"

payload = {
    "query": "what is the timetable of class 6",
    "last_3_turn": [

    ]
}







response = requests.post(url, json=payload)
print(response.json())  # {'response': '...'}