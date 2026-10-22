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
