from flask import Flask, request, jsonify
import subprocess
import os
import uuid
import json

app = Flask(__name__)

@app.route("/", methods=["POST"])
def recibir_datos():
    try:
        datos = request.get_json()

        # Crear archivo temporal CSV
        nombre = f"entrada_{uuid.uuid4().hex[:8]}.csv"
        with open(nombre, "w", encoding="utf-8") as f:
            headers = datos[0].keys()
            f.write(",".join(headers) + "\n")
            for fila in datos:
                f.write(",".join(map(str, [fila[col] for col in headers])) + "\n")

        # Ejecutar el script con ese CSV
        comando = ["python", "nova1b_rev04.py", nombre]
        resultado = subprocess.run(comando, capture_output=True, text=True)

        # Limpiar archivo temporal
        os.remove(nombre)

        if resultado.returncode == 0:
            return jsonify({"status": "ok", "mensaje": "Script ejecutado con éxito."})
        else:
            return jsonify({"status": "error", "mensaje": "Falló la ejecución del script.", "detalles": resultado.stderr}), 500

    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
