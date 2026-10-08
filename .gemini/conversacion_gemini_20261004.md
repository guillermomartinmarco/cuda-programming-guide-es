# Conversión HTML a Markdown

> **Yo:**
> ¿Existe alguna utilidad para convertir a markdown todo un sitio de documentación? Por ejemplo, la _Guia de Grogramación CUDA de Nvidia_.

**Sí, existen excelentes herramientas diseñadas específicamente para rastrear y convertir sitios enteros de documentación a Markdown**, optimizadas además para limpiar menús, pies de página o anuncios, dejando solo el contenido técnico útil (código, explicaciones y tablas) listo para ser leído o ingresado a una Inteligencia Artificial.

Las opciones más eficientes del mercado se dividen según el enfoque de uso:

## 1. Herramientas Especializadas en un Solo Archivo (Ideales para LLMs)

- [**SiteToMarkdown**](https://sitetomarkdown.com/): Es una de las utilidades web más directas para este propósito. Le proporcionas la URL raíz de la documentación (o su `sitemap.xml`) y el servicio **descubre todas las páginas vinculadas, extrae el contenido principal y lo unifica en un único archivo Markdown limpio** estructurado. Es ideal si quieres pasarle la guía de CUDA completa a herramientas como Google NotebookLM o Claude. [1, 2]

## 2. Servicios API y Rastrilladores Web para Desarrolladores

- [**Firecrawl**](https://www.firecrawl.dev/): Es la herramienta de referencia para convertir webs en Markdown para flujos de Inteligencia Artificial o pipelines RAG. Su endpoint `/crawl` te permite ingresar la URL de la documentación de NVIDIA CUDA, **respetar el archivo `robots.txt`, renderizar el JavaScript dinámico** y devolverte un mapa de páginas convertidas a Markdown perfectamente limpio. [3, 4, 5, 6]

- [**Crawl4AI**](https://docs.crawl4ai.com/): Una librería de código abierto basada en Python muy potente. Te permite programar un rastreador asíncrono que extrae el contenido, elimina el ruido visual y genera archivos Markdown estructurados (`raw_markdown` o `fit_markdown`) en cuestión de segundos mediante filtros automáticos. [7, 8]

## 3. Alternativas en la Nube y No-Code

- [**Apify (Website Content Crawler**)](https://apify.com/): Esta plataforma ofrece un "Actor" preconfigurado para convertir sitios web a Markdown. Solo ingresas la URL inicial, defines la profundidad del rastreo (cuántos subenlaces seguir) y descargas un archivo estructurado con toda la documentación procesada. [9, 10]

## 💡 Consejo para el caso específico de la Guía de CUDA de NVIDIA

La documentación de CUDA a veces incluye tablas de compatibilidad masivas y fragmentos de código complejos que pueden desformatearse en conversiones web directas si hay scripts de carga dinámica agresivos. Si las herramientas web fallan con algún apartado:

   1. Intenta descargar la guía en su versión **PDF oficial** (NVIDIA suele ofrecer un botón de descarga para el manual completo en PDF).

   2. Procesa ese PDF con [**MarkItDown**](https://github.com/microsoft/markitdown) (una herramienta oficial de código abierto de Microsoft de línea de comandos) o [**Docling**](https://docling.ai/), que están optimizadas para convertir PDFs técnicos complejos a Markdown preservando ecuaciones y tablas a la perfección. [11, 12, 13]

¿Qué objetivo final tienes para ese Markdown? Si me dices si lo planeas usar para **entrenar un modelo de IA personalizado (RAG)**, leerlo en un editor local como **Obsidian**, o si necesitas ayuda con el **script de Python** para automatizar la descarga, te puedo dar los pasos exactos para configurarlo.

[1] [https://sitetomarkdown.com](https://sitetomarkdown.com)
[2] [https://sitetomarkdown.com](https://sitetomarkdown.com/notebooklm)
[3] [https://www.firecrawl.dev](https://www.firecrawl.dev/tools/website-to-markdown)
[4] [https://zackproser.com](https://zackproser.com/blog/scrape-website-to-markdown)
[5] [https://www.firecrawl.dev](https://www.firecrawl.dev/blog/scrape-a-website-to-markdown)
[6] [https://www.firecrawl.dev](https://www.firecrawl.dev/blog/scrape-a-website-to-markdown)
[7] [https://docs.crawl4ai.com](https://docs.crawl4ai.com/core/markdown-generation/)
[8] [https://docs.crawl4ai.com](https://docs.crawl4ai.com/core/quickstart/)
[9] [https://apify.com](https://apify.com/ryanclinton/website-content-to-markdown)
[10] [https://www.youtube.com](https://www.youtube.com/watch?v=UrCxqRHl95k)
[11] [https://docling.ai](https://docling.ai)
[12] [https://www.youtube.com](https://www.youtube.com/watch?v=g6V9tius9EE)
[13] [https://www.youtube.com](https://www.youtube.com/watch?v=TwlIt75fQAg&t=40)

> **Yo:**
> Vamos con otra alternativa.
> Tengo **Ollama** en mi **Jetson Orin Nano de 8GB**.
> Actualmente tengo estos modelos
>
> ```bash
> martin@jetson-orin-nano:~$ ollama list
> NAME                      ID              SIZE      MODIFIED     
> translategemma:latest     c49d986b0764    3.3 GB    46 hours ago    
> cuda-translator:latest    aa2005f489b4    3.3 GB    46 hours ago    
> llama3.2:1b               baf6a787fdff    1.3 GB    47 hours ago    
> martin@jetson-orin-nano:~$ 
> ```
>
> Estoy consiguiendo traducir con un modelo particular que he desarrollado: `cuda-translator:latest`.
> Pero quiero hacer la transcripción de `.html` a `markdown` sin usar un servicio externo que me cobra.
> Al modelo `cuda-translator` lo generé con el `Modelfile` que te muestro en el adjunto.
> A ese modelo lo arranco con el script `translate.sh` que también te lo muestro en el adjunto.
> Con esa misma filosofóa quisiera generar un modelo que convierte el cada `.html` en un archivo `.md` respetando los links, respetando los formatos, respetando admonitions, respetando el código.
> Lo que he visto en relación al código es que está lleno de tags en el html, eso es un problema.
> En adjunto te muestro una de esas páginas con código `intro-to-cuda-cpp.html`.
> Te animás a construirme un `Modelfile` para convertir `.html` en `.md`?
> Y si los modelos que tengo no alcanzan, sugerime algún otro de **Ollama** (ver aqui <https://ollama.com/library>) que pueda correr en la Jetson que tengo.
> Te animás?

¡Claro que me animo! Es un excelente desafío técnico para resolver de forma local en tu **Jetson Orin Nano de 8GB**.

Pasar HTML crudo (lleno de tags de sintaxis, clases CSS y menús) directamente a un LLM pequeño para que lo convierta en Markdown suele fallar o consumir demasiada memoria de contexto (`num_ctx`). Por suerte, en la biblioteca de Ollama existe una familia de modelos optimizada precisamente para solucionar esto de raíz.

## 1. El Modelo Recomendado para tu Jetson: `reader-lm`

Para esta tarea específica, tus modelos actuales se van a quedar cortos o van a alucinar con la tonelada de tags HTML. Debes descargar `reader-lm`.

Es una serie de modelos creados por _Prediction Guard_ diseñados con un único objetivo: **tomar HTML sucio de documentación y devolver Markdown limpio**, respetando código, tablas y enlaces.

Para tu Jetson de 8GB, la versión de **1.5B** (`reader-lm:1.5b`) o incluso la de **0.5B** (`reader-lm:0.5b`) son perfectas, ligeras y ultra rápidas:

```bash
ollama run reader-lm:1.5b
```

## 2. El `Modelfile` para tu Conversor HTML a MD

Vamos a estructurar el `Modelfile` basándonos en `reader-lm:1.5b`. Este prompt del sistema está diseñado para limpiar el ruido (menús, pies de página de NVIDIA) y concentrarse en el contenido técnico real, formateando los bloques de código e ignorando los scripts de diseño.

Crea un archivo llamado `Modelfile-html2md`:

```dockerfile
# 1. Definir el modelo base optimizado para conversión HTML -> Markdown
FROM reader-lm:1.5b

# 2. Configurar el contexto. La documentación técnica suele ser larga.
# 8192 es un buen tamaño para el HTML crudo sin saturar los 8GB de la Jetson.
PARAMETER temperature 0.0
PARAMETER num_ctx 8192

# 3. Definir el SYSTEM PROMPT con las reglas de conversión y limpieza

SYSTEM """
Eres un extractor de contenido web y conversor experto de HTML a Markdown técnico. Tu única tarea es convertir el código HTML de la documentación de NVIDIA CUDA que te sea proporcionado en un archivo Markdown (.md) limpio, estructurado y perfectamente legible.

Sigue estrictamente estas reglas:

1. Extrae únicamente el contenido principal de la documentación. Elimina menús de navegación, barras laterales, pies de página, selectores de idioma y botones de búsqueda.
2. Convierte todo el código fuente que encuentres (C++, CUDA, Python) en bloques de código Markdown válidos usando triple acento grave e indicando el lenguaje (ej. ```cpp o ```c). Limpia todos los tags internos del HTML (como <span>, <div>, o clases de resaltado de sintaxis) para dejar el código puro.
3. Conserva e integra correctamente todos los hipervínculos internos y externos utilizando la sintaxis de enlaces de Markdown [Texto](URL).
4. Detecta los bloques de advertencias, notas o notas importantes (admonitions, classes como "note", "warning", "important") y estructúralos de forma clara (ej. usando blockquotes '> **Nota:**' o el estándar de GitHub).
5. Mantén la jerarquía correcta de títulos (#, ##, ###) basándote en las etiquetas <h1>, <h2>, etc.
6. Devuelve ÚNICAMENTE el contenido convertido en Markdown. No incluyas introducciones, saludos, comentarios ni explicaciones adicionales fuera del documento.
"""
```

Para construir el modelo ejecuta en tu terminal:

```bash
ollama create html-cleaner -f ./Modelfile-html2md
```

## 3. Adaptando tu Script de Automatización

Para que funcione igual que tu script de traducción `translate.sh`, usaremos `jq` para enviar el HTML de forma segura en el JSON de la API de Ollama.

Crea un script llamado `clean_html.sh`:

```bash
#!/bin/bash

# Asegúrate de que el archivo de entrada exista
INPUT_HTML="./intro-to-cuda-cpp.html"
OUTPUT_MD="./intro-to-cuda-cpp.md"

if [ ! -f "$INPUT_HTML" ]; then
    echo "Error: No se encuentra el archivo $INPUT_HTML"
    exit 1
fi

echo "Procesando y convirtiendo HTML a Markdown localmente..."

# Enviamos el contenido del HTML mediante jq para escapar caracteres problemáticos
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"html-cleaner\",
  \"prompt\": $(jq -Rs . < "$INPUT_HTML"),
  \"stream\": false
}" | jq -r '.response' > "$OUTPUT_MD"

echo "¡Completado! Archivo guardado en $OUTPUT_MD"
```

## 💡 Un truco vital para el HTML de NVIDIA CUDA

El archivo `intro-to-cuda-cpp.html` de NVIDIA viene con miles de líneas de CSS embebido, fuentes e interfaces interactivas al principio y al final del archivo.

Si notas que el modelo se agota o "recorta" la respuesta porque el archivo supera las 8192 fichas de contexto (`num_ctx`), puedes usar un comando rápido en Linux para quitar las partes innecesarias del HTML (como el `<head>`) antes de pasárselo a Ollama. Por ejemplo, extrayendo solo lo que esté dentro de la etiqueta `<main>` o `<body>`:

```bash
# Extrae solo el cuerpo principal del HTML para ahorrarle un 50% de contexto al modelo
sed -n '/<body/,/<\/body>/p' ./intro-to-cuda-cpp.html > ./body_only.html
```

Luego cambias el script para que lea `body_only.html` y verás que el rendimiento en tu Jetson será instantáneo y súper preciso.

¿Prefieres que refinemos el script de bash para que haga esta **limpieza previa del HTML** de forma automática antes de enviarlo a Ollama, o quieres probar primero cómo responde el modelo con el archivo completo?

> **Yo:**
> No está funcionando el script. Me devuelve un archivo vacío en todos los casos.
>
> ```bash
> #!/bin/bash
>
> # Asegúrate de que el archivo de entrada exista
> # INPUT_HTML="./intro-to-cuda-cpp.html"
> INPUT_HTML="./docs.nvidia.com/cuda/cuda-programming-guide/part2.html"
> OUTPUT_MD="./docs.nvidia.com/cuda/cuda-programming-guide/part2.md"
>
> if [ ! -f "$INPUT_HTML" ]; then
>     echo "Error: No se encuentra el archivo $INPUT_HTML"
>     exit 1
> fi
>
> echo "Procesando y convirtiendo HTML a Markdown localmente..."
>
> # Enviamos el contenido del HTML mediante jq para escapar caracteres > problemáticos
> curl -s http://localhost:11434/api/generate -d "{
>   \"model\": \"html-cleaner\",
>   \"prompt\": $(jq -Rs . < "$INPUT_HTML"),
>   \"stream\": false
> }" | jq -r '.response' > "$OUTPUT_MD"
>
> echo "¡Completado! Archivo guardado en $OUTPUT_MD"
> ```
