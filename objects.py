import requests

class ThreadClient:
    """
    Cliente para gestionar hilos a través de la API de OpenAI.
    Contiene métodos para crear, recuperar, modificar y eliminar un hilo.
    """
    def __init__(self, api_key, base_url="https://api.openai.com/v1/threads"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "assistants=v2"
        }

    def crear_hilo(self, mensajes=None, recursos_de_herramientas=None, metadatos=None):
        """
        Crea un hilo.

        Parámetros opcionales:
          - mensajes: lista de mensajes para iniciar el hilo.
          - recursos_de_herramientas: conjunto de recursos disponibles para las herramientas (por ejemplo, file_ids para code_interpreter).
          - metadatos: diccionario con hasta 16 pares clave-valor para almacenar información adicional.

        Devuelve el objeto JSON con la información del hilo creado.
        """
        payload = {}
        if mensajes is not None:
            payload["messages"] = mensajes
        if recursos_de_herramientas is not None:
            payload["tool_resources"] = recursos_de_herramientas
        if metadatos is not None:
            payload["metadata"] = metadatos

        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def recuperar_hilo(self, id_hilo):
        """
        Recupera un hilo dado su ID.

        Parámetro:
          - id_hilo: cadena que identifica el hilo a recuperar.

        Devuelve el objeto JSON correspondiente.
        """
        url = f"{self.base_url}/{id_hilo}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def modificar_hilo(self, id_hilo, recursos_de_herramientas=None, metadatos=None):
        """
        Modifica un hilo determinado por su ID.
        Solo es posible modificar 'tool_resources' y 'metadata'.

        Parámetro:
          - id_hilo: cadena que identifica el hilo a modificar.
          - recursos_de_herramientas: (opcional) nuevo conjunto de recursos para las herramientas.
          - metadatos: (opcional) nuevo diccionario de metadatos.

        Devuelve el objeto JSON del hilo modificado.
        """
        url = f"{self.base_url}/{id_hilo}"
        payload = {}
        if recursos_de_herramientas is not None:
            payload["tool_resources"] = recursos_de_herramientas
        if metadatos is not None:
            payload["metadata"] = metadatos

        # Se utiliza PATCH para actualizar parcialmente el objeto.
        response = requests.patch(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def eliminar_hilo(self, id_hilo):
        """
        Elimina un hilo dado su ID.

        Parámetro:
          - id_hilo: cadena que identifica el hilo a eliminar.

        Devuelve el objeto JSON con el estado de eliminación.
        """
        url = f"{self.base_url}/{id_hilo}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class Thread:
    """
    Representa un hilo según lo retornado por la API.
    """
    def __init__(self, data):
        self.id = data.get("id")
        self.objeto = data.get("object")
        self.creado_en = data.get("created_at")
        self.metadatos = data.get("metadata")
        self.recursos_de_herramientas = data.get("tool_resources")

    def __repr__(self):
        return f"<Thread id={self.id} creado_en={self.creado_en}>"



class AssistantClient:
    """
    Cliente para gestionar asistentes a través de la API de OpenAI.
    Contiene métodos para crear, listar, recuperar, modificar y eliminar un asistente.
    """
    def __init__(self, api_key, base_url="https://api.openai.com/v1/assistants"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "assistants=v2"
        }

    def crear_asistente(self, model, name=None, description=None, instructions=None,
                         tools=None, tool_resources=None, metadata=None,
                         temperature=1, top_p=1, response_format="auto"):
        """
        Crea un asistente.

        Parámetros obligatorios y opcionales:
          - model: Identificador del modelo a utilizar (obligatorio).
          - name: Nombre del asistente (máx. 256 caracteres).
          - description: Descripción del asistente (máx. 512 caracteres).
          - instructions: Instrucciones del sistema (máx. 256,000 caracteres).
          - tools: Lista de herramientas habilitadas para el asistente (máx. 128 herramientas). 
                   Por ejemplo: [{"type": "code_interpreter"}].
          - tool_resources: Recursos específicos para las herramientas.
          - metadata: Diccionario con hasta 16 pares clave-valor (claves máx. 64 caracteres y valores máx. 512 caracteres).
          - temperature: Valor entre 0 y 2 para la temperatura (default=1).
          - top_p: Valor para nucleus sampling (default=1).
          - response_format: Formato de respuesta (default="auto").
        
        Devuelve el objeto JSON con la información del asistente creado.
        """
        payload = {
            "model": model,
            "name": name,
            "description": description,
            "instructions": instructions,
            "tools": tools if tools is not None else [],
            "tool_resources": tool_resources,
            "metadata": metadata,
            "temperature": temperature,
            "top_p": top_p,
            "response_format": response_format
        }
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def listar_asistentes(self, limit=20, order="desc", after=None, before=None):
        """
        Devuelve una lista de asistentes.

        Parámetros opcionales:
          - limit: Límite de asistentes a retornar (entre 1 y 100, default=20).
          - order: Orden de clasificación por created_at ("asc" o "desc", default="desc").
          - after: Cursor para paginación (recupera la página siguiente).
          - before: Cursor para paginación (recupera la página anterior).

        Devuelve el objeto JSON con la lista de asistentes.
        """
        params = {
            "limit": limit,
            "order": order
        }
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before

        response = requests.get(
            self.base_url,
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    def recuperar_asistente(self, assistant_id):
        """
        Recupera un asistente dado su ID.

        Parámetro:
          - assistant_id: Identificador del asistente a recuperar.

        Devuelve el objeto JSON con la información del asistente.
        """
        url = f"{self.base_url}/{assistant_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def modificar_asistente(self, assistant_id, model=None, name=None, description=None,
                            instructions=None, tools=None, tool_resources=None,
                            metadata=None, temperature=None, top_p=None, response_format=None):
        """
        Modifica un asistente.

        Parámetro obligatorio:
          - assistant_id: Identificador del asistente a modificar.

        Parámetros opcionales (los que se incluyan en el payload serán modificados):
          - model, name, description, instructions, tools,
            tool_resources, metadata, temperature, top_p, response_format.

        Devuelve el objeto JSON del asistente modificado.
        """
        payload = {}
        if model is not None:
            payload["model"] = model
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if instructions is not None:
            payload["instructions"] = instructions
        if tools is not None:
            payload["tools"] = tools
        if tool_resources is not None:
            payload["tool_resources"] = tool_resources
        if metadata is not None:
            payload["metadata"] = metadata
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if response_format is not None:
            payload["response_format"] = response_format

        url = f"{self.base_url}/{assistant_id}"
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def eliminar_asistente(self, assistant_id):
        """
        Elimina un asistente dado su ID.

        Parámetro:
          - assistant_id: Identificador del asistente a eliminar.

        Devuelve el objeto JSON con el estado de la eliminación.
        """
        url = f"{self.base_url}/{assistant_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class Assistant:
    """
    Representa un asistente tal como lo retorna la API.
    """
    def __init__(self, data):
        self.id = data.get("id")
        self.objeto = data.get("object")
        self.created_at = data.get("created_at")
        self.name = data.get("name")
        self.description = data.get("description")
        self.model = data.get("model")
        self.instructions = data.get("instructions")
        self.tools = data.get("tools")
        self.tool_resources = data.get("tool_resources")
        self.metadata = data.get("metadata")
        self.temperature = data.get("temperature")
        self.top_p = data.get("top_p")
        self.response_format = data.get("response_format")

    def __repr__(self):
        return f"<Assistant id={self.id} name={self.name} model={self.model}>"

class ClienteMensajes:
    """
    Cliente para gestionar mensajes dentro de un hilo a través de la API de OpenAI.
    Permite crear, listar, recuperar, modificar y eliminar mensajes.
    """
    def __init__(self, api_key, base_url="https://api.openai.com/v1/threads"):
        """
        Inicializa el cliente configurando el API key y la URL base.
        """
        self.api_key = api_key
        # La URL base para los hilos; el endpoint de mensajes se agrega según el id del hilo.
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "assistants=v2"
        }

    def crear_mensaje(self, id_hilo, rol, contenido, adjuntos=None, metadatos=None):
        """
        Crea un mensaje dentro de un hilo.

        Parámetros:
          - id_hilo: Identificador del hilo (string).
          - rol: Rol de la entidad que envía el mensaje ("user" o "assistant").
          - contenido: Contenido del mensaje (string o array). Se puede enviar
                       directamente un string o una lista de estructuras de contenido.
          - adjuntos: (Opcional) Lista de archivos o información de los adjuntos.
          - metadatos: (Opcional) Diccionario de metadatos (hasta 16 pares clave-valor).

        Devuelve el objeto JSON con la información del mensaje creado.
        """
        url = f"{self.base_url}/{id_hilo}/messages"
        payload = {
            "role": rol,
            "content": contenido,
            "attachments": adjuntos,
            "metadata": metadatos
        }
        # Eliminamos los campos con valor None para no enviarlos
        payload = {k: v for k, v in payload.items() if v is not None}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def listar_mensajes(self, id_hilo, limite=20, orden="desc", despues=None, antes=None, run_id=None):
        """
        Lista los mensajes de un hilo.

        Parámetros:
          - id_hilo: Identificador del hilo.
          - limite: Número máximo de mensajes a retornar (entre 1 y 100, default=20).
          - orden: Orden de los mensajes por la marca de tiempo "created_at" ("asc" o "desc", default="desc").
          - despues: (Opcional) Cursor para paginación, para obtener la página siguiente.
          - antes: (Opcional) Cursor para paginación, para obtener la página anterior.
          - run_id: (Opcional) Filtrar mensajes por el run_id que los generó.

        Devuelve el objeto JSON con la lista de mensajes.
        """
        url = f"{self.base_url}/{id_hilo}/messages"
        params = {
            "limit": limite,
            "order": orden
        }
        if despues is not None:
            params["after"] = despues
        if antes is not None:
            params["before"] = antes
        if run_id is not None:
            params["run_id"] = run_id

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def recuperar_mensaje(self, id_hilo, id_mensaje):
        """
        Recupera un mensaje específico de un hilo.

        Parámetros:
          - id_hilo: Identificador del hilo.
          - id_mensaje: Identificador del mensaje.

        Devuelve el objeto JSON con la información del mensaje.
        """
        url = f"{self.base_url}/{id_hilo}/messages/{id_mensaje}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def modificar_mensaje(self, id_hilo, id_mensaje, metadatos):
        """
        Modifica un mensaje dentro de un hilo.

        Parámetros:
          - id_hilo: Identificador del hilo.
          - id_mensaje: Identificador del mensaje.
          - metadatos: Diccionario con los nuevos metadatos (hasta 16 pares clave-valor).

        Devuelve el objeto JSON con el mensaje modificado.
        """
        url = f"{self.base_url}/{id_hilo}/messages/{id_mensaje}"
        payload = {
            "metadata": metadatos
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def eliminar_mensaje(self, id_hilo, id_mensaje):
        """
        Elimina un mensaje dentro de un hilo.

        Parámetros:
          - id_hilo: Identificador del hilo.
          - id_mensaje: Identificador del mensaje a eliminar.

        Devuelve el objeto JSON con el estado de la eliminación.
        """
        url = f"{self.base_url}/{id_hilo}/messages/{id_mensaje}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class Mensaje:
    """
    Representa un mensaje dentro de un hilo.
    """
    def __init__(self, data):
        self.id = data.get("id")
        self.objeto = data.get("object")  # Siempre "thread.message"
        self.creado_en = data.get("created_at")
        self.id_asistente = data.get("assistant_id")
        self.id_hilo = data.get("thread_id")
        self.run_id = data.get("run_id")
        self.rol = data.get("role")
        self.contenido = data.get("content")
        self.adjuntos = data.get("attachments")
        self.metadatos = data.get("metadata")
        # Si hubiera otros campos (status, incomplete_details, etc.), se pueden agregar

    def __repr__(self):
        return f"<Mensaje id={self.id} rol={self.rol}>"

class RunClient:
    """
    Cliente para gestionar ejecuciones (runs) en un thread.
    Permite crear una ejecución en un hilo.
    """
    def __init__(self, api_key, base_url="https://api.openai.com/v1/threads"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "assistants=v2"
        }

    def crear_run(self, thread_id, assistant_id, model=None, instructions=None, additional_instructions=None,
                  additional_messages=None, tools=None, metadata=None, temperature=None, top_p=None,
                  stream=None, max_prompt_tokens=None, max_completion_tokens=None, truncation_strategy=None,
                  tool_choice=None, parallel_tool_calls=True, response_format="auto", include=None):
        """
        Crea una ejecución (run) para un thread.

        Parámetros:
          - thread_id (str): ID del hilo en el que se ejecutará el run.
          - assistant_id (str): ID del asistente a usar en la ejecución.
          - model (str, opcional): ID del modelo a utilizar; si se provee, sobreescribe el asociado al asistente.
          - instructions (str o None, opcional): Instrucciones para el run.
          - additional_instructions (str o None, opcional): Instrucciones adicionales que se anexan a las instrucciones.
          - additional_messages (list o None, opcional): Mensajes adicionales que se agregarán al thread antes de ejecutar el run.
          - tools (list o None, opcional): Lista de herramientas a utilizar en este run.
          - metadata (dict, opcional): Diccionario con información adicional (hasta 16 pares clave-valor).
          - temperature (number o None, opcional): Temperatura de muestreo (entre 0 y 2; default=1 si se omite).
          - top_p (number o None, opcional): Valor para nucleus sampling (default=1 si se omite).
          - stream (bool o None, opcional): Si es True, la respuesta se devuelve como stream.
          - max_prompt_tokens (int o None, opcional): Número máximo de tokens del prompt.
          - max_completion_tokens (int o None, opcional): Número máximo de tokens para la respuesta.
          - truncation_strategy (object o None, opcional): Estrategia de truncamiento para el thread antes del run.
          - tool_choice (str o dict, opcional): Controla qué herramienta se llama durante el run.
          - parallel_tool_calls (bool, opcional): Habilita llamadas paralelas a herramientas (default True).
          - response_format ("auto" o dict, opcional): Especifica el formato de respuesta (default "auto").
          - include (list, opcional): Lista de campos adicionales a incluir en la respuesta. 
            Ejemplo: ["step_details.tool_calls[*].file_search.results[*].content"]

        Devuelve:
          El objeto JSON con la información del run creado.
        """
        url = f"{self.base_url}/{thread_id}/runs"
        params = {}
        if include is not None:
            # Usamos include[] para pasar el query parameter según la documentación.
            params["include[]"] = include

        payload = {
            "assistant_id": assistant_id,
            "model": model,
            "instructions": instructions,
            "additional_instructions": additional_instructions,
            "additional_messages": additional_messages,
            "tools": tools,
            "metadata": metadata,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_prompt_tokens": max_prompt_tokens,
            "max_completion_tokens": max_completion_tokens,
            "truncation_strategy": truncation_strategy,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "response_format": response_format
        }
        # Eliminamos los campos que tengan valor None para no enviarlos innecesariamente
        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(url, headers=self.headers, json=payload, params=params)
        response.raise_for_status()
        return response.json()
    
    
    def recuperar_run(self, thread_id, run_id):
        """
        Recupera la información de un run dado el id del hilo y el id del run.
        """
        url = f"{self.base_url}/{thread_id}/runs/{run_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()