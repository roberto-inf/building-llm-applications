import json 

def to_obj(s):
    try:
        return json.loads(s)
    except Exception:
        return {}