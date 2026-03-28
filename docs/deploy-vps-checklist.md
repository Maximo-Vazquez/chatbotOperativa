# Deploy VPS - Checklist de Referencia

Esta guia resume lo minimo necesario para desplegar `chatbot-operativa` en VPS usando GitHub Actions.

## Workflows involucrados

- Build imagen: `.github/workflows/1-construir-imagen.yaml`
- Deploy VPS: `.github/workflows/3-deploy-manual-vps.yaml`

## Nombres de recursos en Kubernetes

- Deployment: `chatbot-operativa-deployment`
- Service: `chatbot-operativa-service`
- Secret DB: `db-credentials-chatbot`
- Secret app (Google OAuth): `chatbot-operativa-app-secrets` (se crea/actualiza desde GitHub Secrets en cada deploy)
- Dominio esperado: `ia.indutienda.com`
- Imagen: `ghcr.io/maximo-vazquez/chatbotoperativa`

## GitHub Secrets requeridos (repositorio)

### Acceso SSH al VPS

- `VPS_HOST`: IP o dominio del servidor
- `VPS_PORT`: puerto SSH (ejemplo: `22`)
- `VPS_USER`: usuario SSH que ejecuta deploy
- `VPS_SSH_KEY`: clave privada SSH (contenido completo, no `.pub`)

### Variables de Google OAuth (aplicacion)

- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`

Nota: estos secretos se usan para crear/actualizar en el cluster el secret `chatbot-operativa-app-secrets`.
`GOOGLE_CLIENT_KEY` es opcional y puede omitirse.

## Requisitos en el VPS/cluster

- `kubectl` o `k3s kubectl` disponible para el usuario de deploy
- kubeconfig legible en `~/.kube/config`
- acceso al cluster con permisos para:
  - `apply` de manifests
  - `create/apply` de secrets
  - `set image`, `set env`, `rollout`
- Secret de DB creado en cluster: `db-credentials-chatbot` con keys:
  - `host`
  - `port`
  - `user`
  - `password`

## Flujo recomendado

1. Ejecutar build de imagen (`1-construir-imagen.yaml`) en `main` o tag `v*`.
2. Ejecutar deploy (`3-deploy-manual-vps.yaml`):
   - si usas `image_tag=auto`, busca tags `v*` en GHCR
   - si no hay tags `v*`, usar `image_tag=latest` manualmente
3. Verificar rollout:
   - `kubectl rollout status deployment/chatbot-operativa-deployment`

## Fallas comunes y causa

- `can't connect without a private SSH key or password`
  - falta `VPS_SSH_KEY` o esta vacio
- `ssh: unable to authenticate, attempted methods [none publickey]`
  - la clave privada no corresponde con la publica en `~/.ssh/authorized_keys` del `VPS_USER`
- `No se encontro ningun tag versionado v* en GHCR`
  - se ejecuto con `image_tag=auto` y no existen tags `v*` publicados
