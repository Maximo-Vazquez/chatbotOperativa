import hmac, hashlib, json

secret_key = '3mM44WkB26Fhsc_BwDrs4HFByH96x8BjN4KYW'

payload = {
    "evento": "suscripcion.creada",
    "datos": {
        "suscripcion_id_externa": "12345",
        "tipo": "PREMIUM",
        "fecha_fin": "2025-12-31"
    }
}

body = json.dumps(payload).encode("utf-8")

signature = hmac.new(
    secret_key.encode("utf-8"),
    body,
    hashlib.sha256
).hexdigest()

print(signature)
