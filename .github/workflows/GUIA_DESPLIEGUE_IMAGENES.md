# 🚀 Guía Rápida: Generar y Subir Imágenes Docker

Este documento explica cómo actualizar las imágenes de las aplicaciones
(Ventas o Indumentaria) en el registro privado de GitHub.

**Pre-requisito:** Ya configuraste el archivo
`.github/workflows/build-push.yml` en el repositorio.

------------------------------------------------------------------------

## 1. Flujo de Trabajo Diario

Trabajá en tu código normalmente. Hacé todos los `commits` y `push` que
quieras.\
**Nada se subirá al registro Docker todavía.** Esto es solo para guardar
tu código.

``` bash
git add .
git commit -m "Trabajo en progreso..."
git push
```

------------------------------------------------------------------------

## 2. Cómo Lanzar una Nueva Versión (Disparador)

Cuando tu código esté listo y quieras que se construya la imagen Docker
nueva para producción:

### **Paso A: Definir la versión**

Usá el comando `tag`. Siempre debe empezar con `v`.\
**Ejemplos:** `v1.0`, `v1.0.1`, `v2.0`

``` bash
git tag v1.0.5
```

### **Paso B: Enviar la orden a GitHub**

Al subir el tag, el robot de GitHub se despierta, construye la imagen y
la guarda.

``` bash
git push origin v1.0.5
```

------------------------------------------------------------------------

## 3. ¿Dónde están mis imágenes?

Una vez que el proceso termina (tarda unos 2--3 minutos), tus imágenes
estarán disponibles en:

-   **URL del Registro:** `ghcr.io`
-   **Tu Imagen:**\
    `ghcr.io/TU_USUARIO_GITHUB/NOMBRE_DEL_REPO:v1.0.5`

**Etiqueta Automática:** También se actualiza automáticamente la
etiqueta `:latest`.

Podés verlas visualmente en tu perfil de GitHub → pestaña **Packages**.
