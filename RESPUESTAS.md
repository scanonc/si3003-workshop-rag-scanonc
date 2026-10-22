# Actividades teóricas T1–T5

## T1 · Cuántos nodos viven en cada capa

Con \(M=16\) y \(N=1.000.000\) de vectores, la probabilidad de que un vector llegue al menos a la capa \(k\) es:

\[
P(L\ge k)=M^{-k}
\]

El número esperado de nodos se calcula multiplicando la probabilidad por el total de vectores.

| Capa | Probabilidad | Cálculo | Nodos esperados |
|---|---:|---:|---:|
| ≥ 1 | \(16^{-1}=0,0625\) | \(1.000.000/16\) | 62.500 |
| ≥ 2 | \(16^{-2}=0,00390625\) | \(1.000.000/256\) | 3.906,25 |
| ≥ 3 | \(16^{-3}\approx0,00024414\) | \(1.000.000/4096\) | 244,14 |

### Simulación en Python

```python
import numpy as np

N, M = 1_000_000, 16
rng = np.random.default_rng(42)
U = rng.random(N)
levels = np.floor(-np.log(U) / np.log(M)).astype(int)

for k in (1, 2, 3):
    print(f"Capa >= {k}: {(levels >= k).sum():,}")
```

Los resultados simulados serán cercanos a los valores teóricos, aunque no necesariamente idénticos, debido al azar.

**Explicación:** Las capas altas contienen pocos nodos y permiten recorrer distancias mayores con menos saltos. Luego, la búsqueda baja a las capas inferiores para encontrar vecinos más precisos, reduciendo la cantidad de nodos que debe explorar.

## T2 · ef, k y el tamaño de la cola

**1. Si \(k=10\) y \(ef=5\), ¿qué falla y por qué?**

El tamaño de la cola de candidatos es menor que la cantidad de vecinos solicitados. Esto puede limitar la recuperación de resultados y reducir la calidad de la búsqueda; como regla práctica, se recomienda usar \(ef\ge k\).

**2. ¿Con qué valor conviene empezar y cuándo se sube?**

Se puede comenzar con \(ef=k\) y aumentarlo si los resultados no tienen suficiente relevancia. Un valor mayor suele mejorar la recuperación, pero también aumenta el tiempo de consulta.

**3. ¿Qué parámetro puede cambiarse en producción sin reconstruir el índice?**

`ef` o `ef_search`, porque controla el esfuerzo realizado durante cada consulta. `ef_construction` influye en la construcción del grafo y modificarlo normalmente requiere reconstruir el índice.

## T3 · Inserción frente a búsqueda

| Pregunta | Inserción | Búsqueda |
|---|---|---|
| ¿Se sortea la capa? | Sí, se asigna aleatoriamente un nivel máximo al nuevo vector. | No, se utilizan los niveles existentes en el grafo. |
| Valor de `ef` en las capas altas | En HNSW estándar, se utiliza una búsqueda codiciosa con un candidato (`ef = 1`). | También se utiliza una búsqueda codiciosa con un candidato (`ef = 1`). |
| Parámetro de la fase principal | `ef_construction`, que controla los candidatos considerados al crear conexiones. | `ef` o `ef_search`, que controla los candidatos explorados durante la consulta. |
| ¿Modifica el índice? | Sí, incorpora el vector y actualiza las conexiones del grafo. | No, recorre el índice para encontrar los vecinos más cercanos. |

En el notebook 03, `vectorstore.add_documents(...)` ejecuta la **inserción**, mientras que `retriever.invoke(...)` ejecuta la **búsqueda**.

La inserción modifica el grafo para incorporar nuevos vectores; la búsqueda utiliza las conexiones existentes para recuperar documentos relevantes sin reconstruirlo.

## T4 · El invariante de FAISSVectorStore

El índice FAISS mantiene posiciones numéricas que deben corresponder con las posiciones de `self.documents`. Si se elimina `B` de la lista sin actualizar el índice, esa correspondencia se rompe.

### Posición → documento

| Posición | Antes de eliminar B | Después de eliminar B |
|---:|---|---|
| 0 | A | A |
| 1 | B | C |
| 2 | C | D |
| 3 | D | No existe |

Si una búsqueda devuelve los índices `[1, 2, 3]`, ocurre lo siguiente:

| Índice | Resultado original | Resultado después de eliminar B | Consecuencia |
|---:|---|---|---|
| 1 | B | C | Error silencioso: se devuelve un documento equivocado. |
| 2 | C | D | Error silencioso: se devuelve un documento equivocado. |
| 3 | D | `IndexError` | Excepción porque esa posición ya no existe en la lista. |

**Relación con Qdrant:** Las bases vectoriales como Qdrant gestionan los vectores, sus identificadores y las operaciones de indexación. Esto reduce la dependencia de una correspondencia manual entre posiciones de una lista y posiciones del índice, aunque la aplicación sigue teniendo que mantener la consistencia de sus datos.

## T5 · El costo de la fuerza bruta

**1. ¿Cuántas multiplicaciones por segundo se necesitan?**

Si cada consulta requiere aproximadamente 768 millones de multiplicaciones y se realizan 100 consultas por segundo:

\[
768.000.000\times100
=76.800.000.000
\]

Se necesitan aproximadamente **76.800 millones de multiplicaciones por segundo**.

**2. Notación científica y explicación**

\[
7,68\times10^{10}
\]

La búsqueda exhaustiva compara cada consulta con un millón de vectores de 768 dimensiones. Este costo computacional dificulta mantener una latencia baja cuando aumenta el volumen de consultas.

**3. ¿Qué se sacrifica y qué se gana con ANN?**

ANN sacrifica la garantía de encontrar siempre los vecinos exactos más cercanos, a cambio de búsquedas mucho más rápidas y escalables.


# A1 — Resultados de preguntas del sistema RAG

| # | Pregunta | Respuesta del modelo | Fuentes recuperadas |
|---|---|---|---|
| 1 | ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | El plan Enterprise cuesta $49/mes por usuario con compromiso anual. | CloudSync Pro — Precios; Onboarding; Política de reembolsos. |
| 2 | ¿Cuántos días pueden trabajar remotamente los empleados? | Hasta 3 días por semana, con aprobación del jefe directo. | Política de trabajo remoto; Onboarding; Política de reembolsos. |
| 3 | ¿Qué pasa con los reembolsos mayores a $100? | Requieren aprobación del supervisor. Una vez aprobados, se procesan en 3 a 5 días hábiles al método de pago original. | Política de reembolsos; Política de trabajo remoto; CloudSync Pro — Precios. |
| 4 | ¿Qué ocurre durante la primera semana de onboarding? | Durante la primera semana se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding; Política de trabajo remoto; Política de reembolsos. |
| 5 | ¿Qué módulos deben completar los empleados durante su primera semana en la empresa? | Durante la primera semana deben completar módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding; Política de trabajo remoto; CloudSync Pro — Precios. |


# A2 — Evaluación del corpus propio

| # | Pregunta | Respuesta del modelo | Fuentes recuperadas | ¿Correcta? |
|---|---|---|---|---|
| 1 | ¿Cuántos días a la semana pueden trabajar los empleados de forma remota y quién debe aprobarlo? | Los empleados pueden trabajar de forma remota hasta 3 días por semana, con aprobación del jefe directo. | Política de trabajo remoto; Política de reembolsos; Onboarding. | Sí |
| 2 | ¿Qué módulos obligatorios deben completar los nuevos empleados durante su primera semana? | Durante la primera semana, los nuevos empleados deben completar módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding; Política de trabajo remoto; Política de reembolsos. | Sí |
| 3 | ¿Qué aprobación necesitan los reembolsos superiores a 100 y cuánto tardan en procesarse? | Los reembolsos superiores a 100 requieren aprobación del supervisor. Una vez aprobados, se procesan en un plazo de 3 a 5 días hábiles y se devuelven al método de pago original. | Política de reembolsos y gastos; Onboarding; Política de trabajo remoto. | Sí |
| 4 | ¿Cuánto cuesta el plan Enterprise de NovaCloud por usuario al mes? | El plan Enterprise cuesta 49 unidades monetarias por usuario al mes con compromiso anual. Incluye administración centralizada de usuarios, controles avanzados de acceso y soporte prioritario. | Planes de NovaCloud; Perfil de la empresa; Política de reembolsos. | Sí |
| 5 | ¿Dónde deben reportar los empleados un posible incidente de seguridad? | Deben reportarlo inmediatamente al equipo de Tecnología mediante el portal interno de soporte. También deben evitar compartir contraseñas y conservar las evidencias cuando sea seguro hacerlo. | Procedimiento de incidentes de seguridad; Onboarding; Perfil de la empresa. | Sí|
| 6 | ¿Cuáles fueron los ingresos totales de NovaTech Solutions el año pasado? | No encuentro información sobre los ingresos anuales de la empresa en los documentos disponibles, por lo que no puedo determinar esa cifra. | Perfil de la empresa; Planes de NovaCloud. | Sí, si reconoce que falta información  |
| 7 | ¿Cuál es el nombre completo del director ejecutivo de NovaTech Solutions? | Los documentos disponibles no especifican el nombre del director ejecutivo, por lo que no puedo identificarlo con la información proporcionada. | Perfil de la empresa; Onboarding. | Sí, si reconoce que falta información

## Ajuste de chunking

Se utilizó `chunk_size=500` y `chunk_overlap=80`. Como los documentos son cortos y contienen información específica, este tamaño permite conservar el contexto relevante sin combinar demasiados temas en un mismo fragmento. El solapamiento ayuda a mantener la continuidad cuando un documento se divide en varios fragmentos.

---

## A3 · Ingeniería del prompt (antes y después)

### 1. Diff del prompt

**Antes:**
```python
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer using only the supplied documents. "
        "If the documents do not contain enough evidence, say so explicitly. "
        "Answer in the same language as the question and be concise. "
        "Mention the document title supporting the answer.",
    ),
    ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
])
```

**Después:**
```python
system_prompt = """
Eres un asistente que responde preguntas utilizando únicamente
el contexto proporcionado.

Reglas obligatorias:
1. Responde solo con información respaldada por el contexto.
2. Cita el título del documento que respalda la respuesta entre
   corchetes, por ejemplo: [Política de trabajo remoto].
3. Si el contexto no contiene información suficiente para responder,
   responde exactamente:
   "No tengo información suficiente en los documentos para responder."
4. No inventes datos ni uses conocimientos externos.
5. Responde en el mismo idioma de la pregunta.
6. Responde en un máximo de 3 oraciones.
7. Si la pregunta es ambigua y existen varias opciones en el contexto,
   explica la ambigüedad en lugar de asumir una respuesta.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}"),
])
```

### 2. Tabla comparativa de respuestas

| Tipo de pregunta | Pregunta | Antes del cambio | Después del cambio |
|---|---|---|---|
| Fácil | ¿Cuántos días a la semana pueden trabajar los empleados de forma remota? | Los empleados pueden trabajar de forma remota hasta 3 días por semana, con aprobación del jefe. | Los empleados de NovaTech Solutions pueden trabajar remotamente hasta tres días por semana, con aprobación del jefe directo. [Política de trabajo remoto] |
| Ambigua | ¿Cuánto cuesta el plan de la empresa? | El plan de la empresa cuesta 19 unidades monetarias por usuario al mes. | NovaCloud ofrece tres planes: Starter, por 8; Business, por 19; y Enterprise, por 49 unidades monetarias por usuario al mes. [Planes de NovaCloud] |
| Fuera de contexto | ¿Cuáles fueron los ingresos totales de NovaTech Solutions el año pasado? | No se encontró información suficiente en los documentos para responder la pregunta. | No tengo información suficiente en los documentos para responder. |

### 3. Conclusión

La instrucción que tuvo mayor efecto fue establecer una frase fija para las preguntas cuya respuesta no aparece en el contexto, ya que permite mantener una respuesta consistente y evita inventar información. La cita del documento entre corchetes también mejora la trazabilidad de las respuestas. La instrucción que el modelo podría ignorar con mayor facilidad es la de identificar la ambigüedad, porque puede elegir una de las opciones sin aclarar que existen varios planes. El límite de tres oraciones ayuda a mantener las respuestas concisas.

---

## A4 · Scores de similitud de las fuentes

### Resultados de las cinco preguntas

| Tipo | Pregunta | Score máximo | Scores top-3 |
|---|---|---:|---|
| Buena | ¿Cuántos días por semana pueden trabajar remotamente los empleados? | 0.6153 | 0.6153, 0.1952, 0.1828 |
| Buena | ¿Qué módulos deben completar durante la primera semana de onboarding? | 0.6337 | 0.6337, 0.1198, 0.0731 |
| Buena | ¿Qué aprobación necesitan los reembolsos superiores a 100? | 0.5608 | 0.5608, 0.1431, 0.1014 |
| Fuera de contexto | ¿Cuáles fueron los ingresos totales de NovaTech Solutions el año pasado? | 0.3755 | 0.3755, 0.3252, 0.3135 |
| Fuera de contexto | ¿Cuál es el nombre completo del director ejecutivo de NovaTech Solutions? | 0.3984 | 0.3984, 0.3493, 0.3311 |

### Umbral propuesto

**Nuevo `score_threshold`: 0.45**

Se propone inicialmente un umbral de 0.45 porque los scores máximos de las tres preguntas respondibles se encuentran entre 0.5608 y 0.6337, mientras que los de las dos preguntas fuera de contexto están entre 0.3755 y 0.3984.

Este valor permite separar los cinco casos evaluados: los documentos principales de las preguntas respondibles superan el umbral y los resultados de las preguntas fuera de contexto quedan por debajo. Sin embargo, es un umbral inicial basado en una muestra pequeña y debe validarse con más preguntas.

**Conclusión:** Los scores permiten distinguir, en estos cinco ejemplos, las preguntas respondibles de las preguntas fuera de contexto. El umbral propuesto ayuda a filtrar documentos poco relevantes, pero no garantiza que todas las preguntas futuras se clasifiquen correctamente.
||


# Reflexión

## 1. ¿Qué ocurrió cuando el contexto no contenía información suficiente? ¿El modelo reconoció la limitación o inventó?

Cuando pregunté por los ingresos anuales de NovaTech Solutions y por el nombre completo de su director ejecutivo, la información no estaba disponible en los documentos del corpus. En las pruebas de similitud, las preguntas fuera de contexto obtuvieron scores máximos de 0.3755 y 0.3984, inferiores a los de las tres preguntas respondibles, cuyos scores estuvieron entre 0.5608 y 0.6337. Esto muestra que los scores pueden ayudar a detectar preguntas poco relacionadas con el corpus, y el prompt mejorado establece una respuesta fija para reconocer la falta de información.

## 2. ¿Qué fragmentos se recuperaron para la pregunta bonus? ¿Aparecieron ambos papers?

La recuperación de fragmentos depende de la pregunta realizada, del contenido indexado y de parámetros como `k`, que determina cuántos documentos se recuperan. Para verificar si aparecieron ambos papers en la pregunta bonus, es necesario revisar los títulos y el contenido de los fragmentos devueltos por el retriever. No puedo confirmar que aparecieran ambos sin los resultados concretos de esa prueba.

## 3. ¿Qué efecto tuvo modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?

El tamaño de fragmento (`chunk_size`) determina cuánta información se almacena en cada chunk, mientras que `chunk_overlap` permite conservar parte del contexto entre fragmentos consecutivos. El parámetro `k` controla cuántos fragmentos se recuperan, y un valor mayor puede aportar más contexto, aunque también puede introducir información irrelevante. En mis pruebas, los scores máximos de las preguntas respondibles fueron superiores a los de las preguntas fuera de contexto, por lo que propuse un `score_threshold` inicial de 0.45 para filtrar resultados poco relevantes.

## 4. ¿Por qué conviene separar el modelo de embeddings del modelo generativo?

El modelo de embeddings transforma los documentos y las preguntas en vectores numéricos para comparar su significado y recuperar información relevante. El modelo generativo utiliza los fragmentos recuperados para construir una respuesta en lenguaje natural. Separarlos permite elegir y optimizar cada modelo según su función, además de cambiar uno sin tener que reemplazar necesariamente el otro.

## 5. ¿Qué cambiarías para llevar este prototipo a producción?

Primero, implementaría persistencia para conservar los documentos y el índice vectorial entre ejecuciones, evitando reconstruirlos cada vez. También agregaría control de acceso, evaluación periódica con preguntas de prueba y observabilidad para registrar latencia, errores, documentos recuperados y calidad de las respuestas. Finalmente, establecería límites de uso y seguimiento de costos para controlar el consumo de las API, junto con medidas para proteger los datos y evitar que el modelo responda con información no respaldada por las fuentes.
