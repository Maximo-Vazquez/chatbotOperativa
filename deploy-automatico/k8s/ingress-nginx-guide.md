# Guía de migración a Nginx Ingress (obsoleta)

> **Importante:** Desde la adopción del nuevo modelo multi-tenant ya no se utiliza Kubernetes ni la aplicación de orquestación incluida originalmente en este repositorio. La información que sigue se conserva únicamente como referencia histórica para despliegues antiguos.

## 1. Instalar el controlador

Ejecutar una sola vez en el clúster:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

Esto despliega el controlador en el *namespace* `ingress-nginx` y expone el servicio `ingress-nginx-controller`.

## 2. Servicios de las aplicaciones

Cada aplicación Django debe tener su `Service` de tipo **ClusterIP**. El orquestador ya crea estos servicios, pero sin exponer puertos externos. Ejemplo:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: django-service-miapp
spec:
  selector:
    app: django-miapp
  ports:
    - protocol: TCP
      port: 8000   # puerto interno que usará el Ingress
      targetPort: 8000
  type: ClusterIP
```

## 3. Definir los Ingress

Por cada aplicación se crea un recurso `Ingress` que enruta el dominio correspondiente al `Service` anterior. El orquestador puede generar YAML similar al siguiente:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress-miapp
spec:
  rules:
    - host: ejemplo.mi-dominio.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: django-service-miapp
                port:
                  number: 8000
```

## 4. Manejo de archivos estáticos

Se mantiene el despliegue existente de Nginx dedicado a servir archivos estáticos. Puede seguir expuesto como `NodePort` o a través del propio Ingress, por ejemplo:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress-estaticos
spec:
  rules:
    - host: static.mi-dominio.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: nginx-service   # servicio del Nginx de estáticos
                port:
                  number: 80
```

## 5. Adaptaciones en el orquestador

1. Al desplegar una aplicación nueva, generar además del `Deployment` y el `Service` un `Ingress` con el dominio indicado.
2. Eliminar la lógica que modificaba el ConfigMap de Nginx para cada alta/baja de aplicación.
3. Conservar el despliegue del Nginx de estáticos (Deployment + Service) tal como se utiliza hoy.

Con estos cambios las aplicaciones quedarán accesibles a través del Nginx Ingress Controller sin necesidad de editar configuraciones manualmente.
