# Manual de uso: de la guía web de CUDA a Markdown en español

Este proyecto descarga la [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) de NVIDIA, convierte cada página a Markdown y la traduce al español latinoamericano con un modelo local de Ollama.

El proceso tiene tres pasos, y cada uno deja sus archivos junto a los del paso anterior:

```text
programming-model.html  ──(1. wget)
programming-model.md    ──(2. html_to_md.py)
programming-model[es].md ─(3. translate.py + Ollama en la Jetson)
```

## Requisitos

| Qué | Dónde | Para qué |
| --- | --- | --- |
| MSYS2 con zsh, `wget` y `curl` | `C:\S\Msys2\msys64` | Terminal y descarga del sitio |
| Python 3 con el entorno virtual `.venv` | raíz del proyecto | Conversión y traducción |
| Ollama con `translategemma:latest` | Jetson en `192.168.0.100:11434` | Modelo de traducción |

Todos los comandos de este manual se ejecutan desde la raíz del proyecto, en zsh.

### Preparar el entorno de Python (una sola vez)

```zsh
# Crear el entorno virtual si no existe (con uv o con el Python de Windows)
uv venv .venv            # o bien: python -m venv .venv

# Instalar las dependencias dentro del venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Las dependencias son `beautifulsoup4` y `markdownify`, y solo las usa el conversor. El traductor usa nada más que la biblioteca estándar.

> [!TIP]
> Si activás el entorno (`source .venv/Scripts/activate`), en los comandos siguientes podés escribir `python` en lugar de `.venv/Scripts/python.exe`.

## Paso 1: copiar el sitio con wget

```zsh
wget --recursive --page-requisites --adjust-extension --convert-links --no-parent \
     https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
```

| Opción | Efecto |
| --- | --- |
| `--recursive` | Sigue los enlaces y conserva la estructura de carpetas del servidor (hasta 5 niveles, alcanza para la guía) |
| `--page-requisites` | Incluye imágenes (`_images/`), estilos y scripts (`_static/`) |
| `--adjust-extension` | Asegura que las páginas terminen en `.html` |
| `--convert-links` | Reescribe los enlaces para navegar la copia local |
| `--no-parent` | No sube por encima de `/cuda/cuda-programming-guide/` (no baja todo docs.nvidia.com) |

Opcionalmente, `--wait=1` deja un segundo entre pedidos para no sobrecargar el servidor.

En Windows, el `wget` de MSYS2 cambia los caracteres que Windows no admite en los nombres de archivo. Por eso aparecen archivos como `copybutton.js@v=65e89d2a`: el `?` de la URL pasó a `@`. No afecta la conversión.

El resultado queda en `docs.nvidia.com/cuda/cuda-programming-guide/`:

```text
docs.nvidia.com/cuda/cuda-programming-guide/
├── index.html, part1.html … part5.html
├── 01-introduction/   introduction.html, programming-model.html, cuda-platform.html
├── 02-basics/
├── 03-advanced/
├── 04-special-topics/
├── 05-appendices/
├── _images/           figuras referenciadas desde las páginas
└── _static/           CSS y JavaScript del sitio (no se usan después)
```

### Ver la copia local (opcional)

Cada página carga un script de cookies de OneTrust (`cdn.cookielaw.org/.../otSDKStub.js`). Si abrís la copia local, por ejemplo con Live Preview de VS Code, aparece un cartel de cookies que no se puede cerrar. Para desactivarlo en todas las páginas:

```zsh
.venv/Scripts/python.exe disable_cookie_banner.py            # por defecto: docs.nvidia.com
```

El script deja comentada la línea `<script src="https://cdn.cookielaw.org/...">` con `<!-- ... -->`. Se puede correr las veces que haga falta: si ya está comentada, no la toca. Conserva los saltos de línea originales y no afecta la conversión del paso 2.

## Paso 2: convertir HTML a Markdown

```zsh
# Una página
.venv/Scripts/python.exe html_to_md.py docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

# Todo el sitio (recorre las subcarpetas)
.venv/Scripts/python.exe html_to_md.py docs.nvidia.com
```

Cada `nombre.html` genera `nombre.md` en la misma carpeta.

| Opción | Efecto |
| --- | --- |
| `-o salida.md` | Elige otro archivo de salida (solo con una página) |
| `-f`, `--force` | Sobrescribe los `.md` que ya existen |

> [!IMPORTANT]
> Sin `--force`, los `.md` que ya tienen contenido se omiten y aparece `omitido (ya existe, usá --force)`. Así no se pierden los archivos que hayas retocado a mano.

### Qué hace el conversor

Toma solo el contenido de la página (`<article class="bd-article">`) y descarta los menús, el índice lateral, el bloque "On this page" y los botones. Después adapta cada elemento de Sphinx:

| En el HTML | En el Markdown |
| --- | --- |
| Títulos con su numeración | `#`, `##`, `###`… con la numeración (`## 1.2.1. Heterogeneous Systems`) |
| Código con resaltado (`highlight-cuda`, `c++`, `python`, `bash`…) | Bloques ```` ```cuda ```` y similares, con el texto limpio |
| Note, Hint, Warning… | Avisos de GitHub: `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]` |
| Figuras | `![texto alternativo](../_images/x.png)` y debajo **Figure N.** *pie de figura* |
| Fórmulas | `$...$` en línea y `$$...$$` en bloque |
| Notas al pie | `[^1]` en el texto y `[^1]: ...` al final |
| Pestañas (p. ej. "Unified Memory" / "Explicit Memory") | El nombre de la pestaña en **negrita** y su contenido debajo |
| Tablas simples | Tablas Markdown |
| Tablas con celdas combinadas o con código adentro | HTML limpio (GitHub lo muestra bien) |
| Tablas de una sola celda (contenedor de código) | Se desarman y queda el bloque de código |
| Enlaces a otras páginas (`foo.html#x`) | `foo.md#x` |
| Subíndices y superíndices | `<sub>` y `<sup>` |

Las imágenes quedan con la misma ruta relativa (`../_images/...`), así que se ven desde los `.md` sin copiar nada.

## Paso 3: traducir al español

### 3.1 Crear el modelo en la Jetson (una vez, o cada vez que cambie el Modelfile)

Copiá `Modelfile-translate` a la Jetson y ejecutá ahí:

```bash
ollama pull translategemma:latest          # solo la primera vez
ollama create cuda-translator -f Modelfile-translate
```

El Modelfile configura lo siguiente:

- `num_ctx 16384`: entran la entrada y la salida de la sección más larga de la guía.
- `temperature 0.1`: traducción fiel y estable.
- Una plantilla (`TEMPLATE`) que une la instrucción y el texto en un solo mensaje, que es el formato con el que se entrenó TranslateGemma.
- El prompt oficial de TranslateGemma, en inglés y corto. Es un traductor, no un modelo que siga instrucciones: con reglas largas pierde los marcadores de código. Por eso las protecciones están en `translate.py`.

Para comprobar que el modelo existe:

```zsh
curl -s http://192.168.0.100:11434/api/tags | jq -r '.models[].name'
```

### 3.2 Traducir

```zsh
# Una página
.venv/Scripts/python.exe translate.py docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.md

# Todo el sitio (todos los .md que no terminan en [es])
.venv/Scripts/python.exe translate.py docs.nvidia.com
```

Cada `nombre.md` genera `nombre[es].md` en la misma carpeta.

| Opción | Efecto |
| --- | --- |
| `-o salida.md` | Elige otro archivo de salida (solo con una página) |
| `-f`, `--force` | Sobrescribe las traducciones que ya existen |
| `--no-cache` | Vuelve a traducir todas las secciones, sin usar la caché |
| `--host URL` | Otro servidor Ollama (por defecto `http://192.168.0.100:11434`) |
| `--model NOMBRE` | Otro modelo (por defecto `cuda-translator`) |
| `--timeout SEG` | Tiempo máximo por llamada (por defecto 900) |

> [!WARNING]
> En zsh, los corchetes son comodines. Para nombrar un archivo traducido, ponelo entre comillas: `less "programming-model[es].md"`. Sin comillas, zsh responde `no matches found`.

Mientras traduce, el script muestra el avance:

```text
programming-model.md -> programming-model[es].md (17 secciones)
  [1/17] # 1.2. Programming Model
    9.8s
  [2/17] ## 1.2.1. Heterogeneous Systems
    21.4s
  [3/17] (caché) ## 1.2.2. GPU Hardware Model
```

### Cómo traduce

1. **Una sección por llamada.** Cada encabezado, del nivel que sea (`#` a `######`), inicia una sección nueva. Los `#` dentro de bloques de código, como `# comentario` en bash o python, no cuentan.
2. **El código no pasa por el modelo.** Los bloques de código, las tablas HTML y las fórmulas `$$` se reemplazan por marcadores `<!-- keep:N -->` y después de traducir se restauran tal cual. Los bloques que están al principio o al final de una sección ni siquiera se envían.
3. **Enlaces e imágenes protegidos.** El modelo traduce rutas y nombres de archivo: `../03-advanced/driver-api.md` salía como `../03-avanzado/...` y `active-warp-lanes.png` como `hilos-activos.png`. Por eso los destinos de `[texto](destino)` y `![alt](ruta)` se reemplazan por fichas (`](L0)`, `](L1)`…) y se restauran después. El texto del enlace sí se traduce.
4. **Validación y plan B.** Después de cada traducción, el script verifica que no falten marcadores de código, fichas de enlaces ni notas al pie (`[^1]`). Si falta algo, avisa `traduciendo por partes` y traduce cada bloque de texto por separado. Si un bloque sigue fallando, prueba párrafo por párrafo. Si un párrafo no se puede traducir sin romper nada, lo deja en inglés y avisa `!! párrafo dejado en inglés`. Así se ve qué hay que traducir a mano, y no queda ningún enlace roto escondido.
5. **Avisos en inglés.** El modelo a veces traduce las etiquetas (`[!NOTA]`, `[¡NOTA!]`, `[Advertencia]`). El script las vuelve a poner como `[!NOTE]`, `[!WARNING]`, etc., para que GitHub las muestre como avisos.
6. **Caché.** Cada sección traducida se guarda en `.translate_cache/` (está en `.gitignore`). Si la traducción se corta, o si una página cambia, al volver a correrlo solo se traducen las secciones nuevas o modificadas.

Cada sección tarda entre 10 y 60 segundos en la Jetson, así que una página grande puede llevar media hora o más. La guía completa tiene unas 950 secciones: conviene traducirla por carpetas, en varias sesiones.

## Flujo completo

```zsh
# 0. Entorno (una vez)
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 1. Descargar
wget --recursive --page-requisites --adjust-extension --convert-links --no-parent \
     https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

# 1b. (Opcional) Quitar el cartel de cookies para ver la copia local
.venv/Scripts/python.exe disable_cookie_banner.py

# 2. Convertir
.venv/Scripts/python.exe html_to_md.py docs.nvidia.com

# 3. Traducir (con el modelo cuda-translator ya creado en la Jetson)
.venv/Scripts/python.exe translate.py docs.nvidia.com/cuda/cuda-programming-guide/01-introduction
```

Para actualizar la guía cuando NVIDIA publique cambios, repetí los tres pasos con `--force` en el paso 2. El `wget` sobrescribe los HTML y el script de cookies vuelve a quedar activo: si querés ver la copia local, corré de nuevo `disable_cookie_banner.py`. En el paso 3, **no uses `--force` sobre una carpeta** si corregiste traducciones a mano. Sobrescribiría todos los `[es].md` de esa carpeta, y la caché guarda la traducción original del modelo, no tus correcciones. Para una página que quieras regenerar, pasá el archivo solo, con `--force`. Las secciones que no cambiaron salen de la caché.

> [!CAUTION]
> Sin `--force`, los scripts nunca sobrescriben un archivo que ya tiene contenido. Ese es el comportamiento que protege tus correcciones manuales.

## Problemas comunes

| Síntoma | Causa y solución |
| --- | --- |
| `Error al conectar con Ollama en http://192.168.0.100:11434` | La Jetson está apagada o Ollama no está corriendo. Probá con `curl http://192.168.0.100:11434/api/tags`. |
| `aviso: la respuesta se cortó por límite de contexto (num_ctx)` | La sección no entra en el contexto. Subí `num_ctx` en `Modelfile-translate` y volvé a crear el modelo. |
| La Jetson se queda sin memoria o Ollama se reinicia | Liberá la caché del sistema con `liberar_memoria_jetson.sh` (en la Jetson). Si sigue pasando, bajá `num_ctx` a 12288. |
| `omitido (ya existe, usá --force)` | El archivo de salida ya existe. Agregá `--force` para regenerarlo. |
| `zsh: no matches found: ...[es].md` | Faltan comillas alrededor del nombre (ver el aviso del paso 3.2). |
| Términos traducidos que deberían quedar en inglés ("hilo" en lugar de "thread", "núcleo" en lugar de "kernel") | TranslateGemma respeta el glosario del Modelfile solo en parte. Se corrige a mano, o se usa otro modelo con `--model`. |

## Archivos del proyecto

| Archivo | Función |
| --- | --- |
| `disable_cookie_banner.py` | Comenta el script de cookies en los HTML descargados (para verlos localmente) |
| `html_to_md.py` | Conversor de HTML (Sphinx) a Markdown |
| `translate.py` | Traductor sección por sección con Ollama |
| `Modelfile-translate` | Definición del modelo `cuda-translator` para Ollama |
| `requirements.txt` | Dependencias de Python |
| `liberar_memoria_jetson.sh` | Libera la caché de memoria de la Jetson (`drop_caches`) |
| `.translate_cache/` | Caché de secciones traducidas (se puede borrar sin problema) |
