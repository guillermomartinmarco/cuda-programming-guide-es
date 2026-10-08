# Guía de Programación de CUDA

## CUDA y la Guía de Programación de CUDA

CUDA es una plataforma y modelo de programación para computación paralela desarrollada por NVIDIA que permite aumentar drásticamente el rendimiento de la computación aprovechando el poder de la GPU. Permite a los desarrolladores acelerar aplicaciones intensivas en cómputo y se utiliza ampliamente en campos como el aprendizaje profundo, la computación científica y la computación de alto rendimiento (HPC).

Esta Guía de Programación de CUDA es el recurso oficial y completo sobre el modelo de programación de CUDA y cómo escribir código que se ejecuta en la GPU utilizando la plataforma CUDA. Esta guía cubre todo, desde el modelo de programación de CUDA y la plataforma CUDA hasta los detalles de las extensiones del lenguaje, y cubre cómo aprovechar las características específicas del hardware y el software. Esta guía proporciona un camino para que los desarrolladores aprendan CUDA si son nuevos, y también proporciona un recurso esencial para los desarrolladores a medida que construyen aplicaciones con CUDA.

## Organización de esta Guía

Incluso para los desarrolladores que utilizan principalmente bibliotecas, marcos o DSL, comprender el modelo de programación de CUDA y cómo las GPUs ejecutan el código es valioso para saber qué está sucediendo detrás de las capas de abstracción.

Esta guía comienza con un capítulo sobre el [modelo de programación de CUDA](./01-introduction/programming-model[es].md) fuera de cualquier lenguaje de programación específico, que es aplicable a cualquier persona interesada en comprender cómo funciona CUDA, incluso a no desarrolladores.

La guía se divide en cinco partes principales:

- Parte 1: Introducción y Modelo de Programación Abstracto

  - Una descripción general sin lenguaje del modelo de programación de CUDA, así como un breve recorrido por la plataforma CUDA.

  - Esta sección está destinada a ser leída por cualquier persona que desee comprender las GPUs y los conceptos de ejecución de código en las GPUs, incluso si no son desarrolladores.
- Parte 2: Programación de GPUs en CUDA

  - Los conceptos básicos de la programación de GPUs en C++ y Python.

  - Esta sección está destinada a ser leída por cualquier persona que desee comenzar a programar en GPUs.

  - Esta sección está destinada a ser instructiva, no completa, y enseña las partes más importantes y comunes de la programación de CUDA, incluyendo algunas consideraciones de rendimiento comunes.
- Parte 3: CUDA Avanzada

  - Introduce algunas características más avanzadas de CUDA que permiten un control más fino y más oportunidades para maximizar el rendimiento, incluyendo el uso de múltiples GPUs en una sola aplicación.

  - Esta sección concluye con un [recorrido de las características cubiertas en la Parte 4](./03-advanced/feature-survey[es].md) con una breve introducción al propósito y la función de cada una, ordenadas por cuándo y por qué un desarrollador puede encontrar útil cada característica.
- Parte 4: Características de CUDA

  - Esta sección contiene una cobertura completa de características específicas de CUDA, como los gráficos CUDA, la programación dinámica, la interoperabilidad con las APIs gráficas, y la memoria unificada.

  - Esta sección debe consultarse cuando se necesita una comprensión completa de una característica específica de CUDA. Siempre que sea posible, se ha hecho un esfuerzo para introducir y motivar las características cubiertas en esta sección en secciones anteriores.
- Parte 5: Apéndices Técnicos

  - Los apéndices técnicos proporcionan documentación de referencia sobre el soporte de lenguaje de alto nivel de C++ de CUDA, especificaciones de hardware, y otras especificaciones técnicas.

  - Esta sección está destinada como referencia técnica para la descripción de la sintaxis, la semántica y el comportamiento técnico de los elementos de CUDA.

Las partes 1-3 proporcionan una experiencia de aprendizaje guiada para los desarrolladores nuevos en CUDA, aunque también proporcionan información y actualizaciones útiles para los desarrolladores de CUDA de cualquier nivel de experiencia.

Las partes 4 y 5 proporcionan una gran cantidad de información sobre características específicas y temas detallados, y están destinadas a proporcionar una referencia bien organizada y curada para los desarrolladores que necesitan saber más detalles a medida que escriben aplicaciones de CUDA.
