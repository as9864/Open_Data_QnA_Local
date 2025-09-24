from flask import Flask, request, jsonify
app = Flask(__name__)
store = {}
@app.post("/api/chat/callback/<chatId>")
def cb(chatId):
    store[chatId] = request.get_json(silent=True)
    print("[CALLBACK]", chatId, store[chatId])
    return jsonify({"ok": True})
@app.get("/__callbacks__")
def list_(): return jsonify(store)
app.run(host="0.0.0.0", port=3010)