def validate_predict_request(data):
    
    #Validate incoming JSON for prediction.
    if not data or "text" not in data:
        return False, "Missing 'text' field."

    text = data["text"]

    # check text datatype and length
    if not (isinstance(text, str) and len(text.strip()) > 3):
        return False, "Text is too short or invalid."

    return True, text
