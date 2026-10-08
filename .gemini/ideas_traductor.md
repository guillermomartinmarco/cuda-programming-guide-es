# 📖 Flujo de Trabajo Optimizado en VS Code para Libros Técnicos y GPU

## 1. Adaptación al Enfoque: Pure Markdown (GFM) + Libros de GPUs

En el estudio de arquitectura de GPUs (CUDA, Vulkan, DirectX, C++, shaders, modelo de memoria SIMD/SIMT), el formato no debe distraer.

### Principios Fundamentales

* **Zero HTML Tags:** Nada de `<details>`, `<summary>` o estilos incrustados. Usaremos **GitHub Flavored Markdown (GFM)** estándar (`#`, `##`, listas, blockquotes `>` y bloques de código explícitos).

* **Tratamiento del Dominio GPU / Hardware:** El léxico de GPUs (*warp, wavefront, streaming multiprocessor, register pressure, barrier, occupancy, memory coalescing, thread block, SPIR-V, HLSL/GLSL*) debe mantenerse en inglés cuando no exista un equivalente técnico inequívoco en castellano.

* **Limpieza de Artefactos de PDF:** La extracción desde PDF suele introducir saltos de línea a mitad de oración, guiones de partición de palabras (*hyphenation*), cabeceras de página y números de página flotantes. El script debe limpiar estos artefactos en la misma pasada.

## 2. Automatización Directa en VS Code

Para evitar salir de VS Code, el script `traducir_gpu_md.py` procesa archivos `.md` o textos pegados en borrador, eliminando la basura del PDF y traduciendo el capítulo respetando estrictamente los bloques de código y la terminología de arquitectura de computadores.

### Script de Automatización: `traducir_gpu_md.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traducir_gpu_md.py - Automatización de traducción técnica de libros de GPUs/Sistemas.
Diseñado para VS Code, compatibilidad GFM pura y limpieza de texto extraído de PDF.
"""

import os
import sys
import argparse
from google import genai
from google.genai import types

PROMPT_SISTEMA_GPU = """
Eres un especialista en traducción e ingeniería de software de bajo nivel, gráficos por computadora y arquitectura de GPUs (CUDA, Vulkan, C++, HLSL, GLSL, ensamblador, paralelo).

Tu tarea es tomar un texto o capítulo en Markdown procedente de la extracción de un libro en PDF y devolver un documento en castellano técnico neutro que cumpla estrictamente lo siguiente:

REGLAS DE FORMATO Y ESTILO (MARKDOWN PURO):
1. Devuelve ÚNICAMENTE Markdown compatible con GitHub (GFM). PROHIBIDO usar etiquetas HTML (<details>, <div>, <br>, etc.).
2. Elimina artefactos típicos de PDF: une oraciones cortadas por saltos de línea erróneos, elimina encabezados/pies de página flotantes y elimina números de página aislados.
3. Conserva intactos los bloques de código (```cpp, ```cuda, ```hlsl, ```glsl, ```python, ```asm). NO traduzcas las instrucciones, palabras clave ni nombres de variables/funciones. Traduce únicamente los comentarios dentro del código.
4. Mantén intactas las fórmulas matemáticas en LaTeX ($...$ o $$...$$).

GLOSARIO Y TERMINOLOGÍA GPU / PARALELO (REGLA DE CONSERVACIÓN):
- Conserva en inglés términos clave de la industria donde la traducción resulta confusa o entorpece la lectura:
  'Warp', 'Wavefront', 'Streaming Multiprocessor (SM)', 'Thread Block', 'Grid',
  'Register Pressure', 'Occupancy', 'Memory Coalescing', 'Shared Memory', 'Global Memory',
  'Barrier', 'Fence', 'Bank Conflict', 'Tuning', 'Pipeline', 'Dispatch', 'Draw Call',
  'Descriptor Set', 'Command Buffer', 'Ray Tracing', 'Tensor Core', 'Shader'.
- Traduce con precisión términos de teoría general de computación:
  'Latency' -> Latencia, 'Throughput' -> Rendimiento/Ancho de banda, 'Thread' -> Hilo,
  'Host' -> Host/Anfitrión, 'Device' -> Dispositivo/Device.
"""

def procesar_capitulo(ruta_entrada: str, ruta_salida: str = None):
    if not os.path.exists(ruta_entrada):
        print(f"❌ Error: El archivo '{ruta_entrada}' no existe.")
        sys.exit(1)

    with open(ruta_entrada, "r", encoding="utf-8") as f:
        contenido_bruto = f.read()

    print(f"📖 Procesando '{ruta_entrada}' ({len(contenido_bruto)} caracteres)...")
    print("⚡ Limpiando PDF + Traduciendo dominio GPU con Gemini 2.5 Flash...")

    client = genai.Client()

    prompt_usuario = (
        f"Procesa, limpia de artefactos PDF y traduce el siguiente texto técnico:\n\n{contenido_bruto}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[PROMPT_SISTEMA_GPU, prompt_usuario],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Temperatura muy baja para precisión matemática y de código
            ),
        )
    except Exception as e:
        print(f"❌ Error en API Gemini: {e}")
        sys.exit(1)

    if not ruta_salida:
        base, ext = os.path.splitext(ruta_entrada)
        ruta_salida = f"{base}_ES{ext}"

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(response.text)

    print("✅ Proceso completado exitosamente.")
    print(f"💾 Resultado en GFM escrito en: '{ruta_salida}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traducción y limpieza de libros de GPUs/Bajo nivel en Markdown"
    )
    parser.add_argument("archivo", help="Ruta del archivo Markdown (.md) a procesar")
    parser.add_argument("-o", "--output", help="Ruta de salida (opcional)")

    args = parser.parse_args()
    procesar_capitulo(args.archivo, args.output)
```

## 3. Integración Directa como VS Code Task

Puedes ejecutar este script con una sola combinación de teclas dentro de VS Code sin abrir una terminal independiente.

### Paso 1: Configurar la Tarea (`.vscode/tasks.json`)

En la carpeta del proyecto donde guardas tus archivos `.md`, crea o edita el archivo `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Traducir Capítulo GPU (Gemini)",
      "type": "shell",
      "command": "python",
      "args": [
        "${workspaceFolder}/traducir_gpu_md.py",
        "${file}"
      ],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "shared"
      },
      "problemMatcher": []
    }
  ]
}


```

### Paso 2: Flujo de Uso en VS Code

1. Pegas el texto extraído del PDF en tu archivo `capitulo_03.md`.

2. Presionas `Ctrl + Shift + B` (o `Cmd + Shift + B` en macOS).

3. Gemini limpia los saltos de línea feos del PDF, respeta tu código CUDA/C++ y genera en segundos el archivo `capitulo_03_ES.md`.

4. Abrís `capitulo_03_ES.md` al lado en VS Code con `Ctrl + K, V` (Markdown Preview) y leés fluidamente.

## 4. Comparativa de Métodos para Lectura de Estudio

| Método | Esfuerzo de Formato | Manejo de Código / Math | Preservación de Jerga GPU | Velocidad |
 | ----- | ----- | ----- | ----- | ----- |
| **PDF** $\rightarrow$ **DeepL Web (Copia/Pega <1500 chars)** | Alto (rompe párrafos y listas) | Malo (rompe sangrías y palabras clave) | Variable (traduce términos que deberían ir en inglés) | 🐢 Muy lento |
| **PDF** $\rightarrow$ **Clean GFM con `traducir_gpu_md.py`** | Bajo (el LLM limpia el PDF) | Excelente (C++, CUDA, HLSL e $i/o$ sin alteración) | Nativo (mantiene warps, occupancy, etc.) | ⚡ Rápido (1 atajo) |
