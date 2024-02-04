from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# TODO: Replace
github_url = "<github json url>"

# TODO: Replace
github_token = "<github personal access token>"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the user's message from the request
        user_message = request.json.get("user_message")

        if not user_message:
            return jsonify({"response_message": "Please provide a user message."})

        # Retrieve JSON data from GitHub with authentication
        headers = {"Authorization": f"token {github_token}"}
        response = requests.get(github_url, headers=headers)

        if response.status_code == 200:
            json_data = response.json()
            
            # Split the user message into keywords
            keywords = user_message.lower().split()
            
            # Filter the JSON data based on both keywords in the "name" field
            filtered_data = [item for item in json_data if all(keyword in item.get("name", "").lower() for keyword in keywords)]
            
            if not filtered_data:
                # If no matches are found by name, then search in the "description" field with both keywords
                filtered_data = [item for item in json_data if all(keyword in item.get("description", "").lower() for keyword in keywords)]
            
            if filtered_data:
                # Extract "description" and "name" fields from the filtered data
                filtered_result = [{"description": item["description"], "name": item["name"]} for item in filtered_data]
                return jsonify({"response_message": filtered_result})
            else:
                return jsonify({"response_message": "No matching results found."})

        else:
            return jsonify({"response_message": "Failed to retrieve JSON data from GitHub."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
