# Uso de `cuda::ceil_div`

> Capítulo **2.1.2.3.1. Verificación de límites**

Hemos visto cómo el rendimiento de la GPU se ve alterado si no realizamos un buen cálculo de la cantidad mínima de bloques de hilos que necesitamos para calcular, por ejemplo, la suma vectorial de dos vectores, atribuyendo a cada hilo, el cálculo de un elemento particular.

Recordemos, la llamada a la función cuda se realiza así:

```cpp
funcion_cuda<<<bloques, hilosPorBloque>>>(argumentos...);
```

Si queremos realizar la suma de dos vectores de dimensión 1000000 y definimos (arbitrariamente) que cada bloque de hilos tiene que ser de 256 hilos, entonces necesitamos, siendo finos

$$\cfrac{1000000}{256}=3906,25 \ \text{bloques}$$

Pero no podemos tener una cantidad de bloques de hilos con decimales. Tenemos varias alternativas:

1. Podríamos tomar la parte entera $3906$ y sumarle $1$. Entonces $3907$ seria la cantidad efectiva de bloques de 256 hilos que necesitamos pasarle al primer argumento de la llamada.

2. Podríamos restar $1$ de la cantidad de `hilosPorBloque` que hemos seleccionado arbitrariamente, realizar la división, y tomar la parte entera. Una división entera automáticamente devuelve este número entero

   $$\cfrac{1000000}{256-1}=\cfrac{1000000}{255}=3921,56...\ \text{bloques}$$

   ```cpp
   int dimVector = 1000000;
   int hilosPorBloque = 256;
   int bloques = dimVector / (hilosPorBloque - 1); // Esto resulta en 3921 bloques
   ```

   Siendo muy observadores, tenemos un sobrante de $3921-3907=14$ hilos de más que no harían absolutamente nada dentro del kernel, a pesar de ser lanzados.

3. Otra técnica que podemos usar:

   $$
   \begin{aligned}
   \mathtt{bloques} &= \cfrac{\mathtt{dimVector} + \mathtt{hilosPorBloque}-1}{\mathtt{hilosPorBloque}} \\
   \mathtt{bloques} &= \cfrac{1000000 + 256-1}{256} = 3907,24... \ \text{bloques}
   \end{aligned}
   $$

   La parte entera es $3907$, es decir, la misma mínima cantidad de hilos que calculamos en la primera alternativa. La ventaja de esta alternativa es que, si hacemos la división entera, nos devuelve automáticamente $3907$.

   ```cpp
   int dimVector = 1000000;
   int hilosPorBloque = 256;
   int bloques = (dimVector + hilosPorBloque - 1) / hilosPorBloque;
   // Esto resulta en 3907 bloques
   ```

4. Usar la función `cuda::ceil_div` de la biblioteca **CCCL**, que hace esta división con redondeo y calcula el número de bloques de hilos que necesitamos para lanzar el kernel de manera estandarizada:

   ```cpp
   int dimVector = 1000000;
   int hilosPorBloque = 256;
   int bloques = cuda::ceil_div(dimVector, hilosPorBloque); // 3907
   funcion_cuda<<<bloques, hilosPorBloque>>>(argumentos...);
   ```

Modificaremos un poco el ejercicio de la suma de vectores para hacer uso de esta función.
