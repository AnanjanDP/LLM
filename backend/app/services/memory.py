memory_store = {}

def get_history(session_id):
    return memory_store.get(session_id, [])

def update_history(session_id, user, bot):
    if session_id not in memory_store:
        memory_store[session_id] = []
    memory_store[session_id].append({"user": user, "bot": bot})