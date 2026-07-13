"""Generador de dataset sintetico de noticias en espanol (v2 enriquecida).

Mejoras vs v1:
  - Vocabulario mucho mas diverso (sujetos, acciones, contextos por categoria).
  - Textos mas largos y realistas (80-150 palabras vs 30 antes).
  - Estructuras de oracion variadas (no solo plantilla sujeto+accion).
  - Adjetivos, conectores y vocabulario periodistico real.
  - Ruido: variaciones de redaccion, citas, datos numericos.

Objetivo: que los modelos generalicen a noticias reales, no solo a plantillas.
"""
import random
import json
import os
from pathlib import Path

CATEGORIAS = [
    "politica",
    "economia",
    "deportes",
    "tecnologia",
    "salud",
    "internacional",
    "cultura",
]

# ---------------------------------------------------------------------------
# Sujetos diversos por categoria (entidades que aparecen en noticias reales).
# ---------------------------------------------------------------------------
SUJETOS = {
    "politica": [
        "el gobierno", "el presidente", "el ministro de estado", "la oposicion",
        "el congreso", "el senado", "la camara de diputados", "el partido gobernante",
        "el alcalde", "la asamblea legislativa", "el gabinete ministerial",
        "el tribunal electoral", "la autoridad electoral", "el secretario general",
        "el vicepresidente", "la ministra de gobierno", "el partido politico",
        "la coalicion opositora", "el comite de partido", "la comision de gobierno",
        "el lider opositor", "la bancada oficialista", "el jefe de estado",
        "la comision de etica", "el gobernador", "el ministerio publico",
        "la defensoria del pueblo", "el contralor general",
    ],
    "economia": [
        "el banco central", "el ministerio de economia", "la bolsa de valores",
        "el fondo monetario internacional", "el banco mundial", "las empresas privadas",
        "el sector exportador", "la camara de comercio", "la asociacion de banqueros",
        "el instituto nacional de estadistica", "la superintendencia de bancos",
        "el servicio de impuestos", "los gremios empresariales", "la federacion de comerciantes",
        "los analistas financieros", "el mercado bursatil", "la corporacion andina de fomento",
        "el banco de desarrollo", "los inversionistas extranjeros", "la bolsa comercial",
        "la camara nacional de industria", "el servicio aduanero", "la direccion de tributacion",
        "los empresarios del rubro", "el gremio de exportadores", "la junta monetaria",
    ],
    "deportes": [
        "el seleccionado nacional", "el club deportivo", "el entrenador del equipo",
        "el jugador estrella", "la federacion deportiva", "el comite olimpico",
        "la liga profesional", "el cuerpo tecnico", "los atletas olmpicos",
        "la asociacion de futbol", "el arbitro central", "el capitan del equipo",
        "los deportistas locales", "el director tecnico", "la seleccion sub-23",
        "el club campeón", "los hinchas del equipo", "la liga departamental",
        "el medallista olmpico", "la maraton internacional", "el ciclista profesional",
        "el tenista nacional", "la seleccion femenina", "el equipo visitante",
        "el club local", "la confederacion sudamericana", "el boxeador categoria peso",
    ],
    "tecnologia": [
        "la empresa tecnologica", "los desarrolladores de software", "la startup emergente",
        "el gigante informatico", "los investigadores en inteligencia artificial",
        "la compania de telecomunicaciones", "el fabricante de dispositivos",
        "el laboratorio de innovacion", "la plataforma digital", "los ingenieros de sistemas",
        "la comunidad open source", "el centro de computacion", "la firma ciberseguridad",
        "los creadores de contenido", "la agencia espacial", "el consorcio tecnologico",
        "la division de datos", "los especialistas en robotics", "el equipo de desarrollo",
        "la empresa de software", "los hackers eticos", "la compania de internet",
        "el proveedor de servicios cloud", "los pioneros en blockchain",
        "la corporacion digital", "el instituto de tecnologia",
    ],
    "salud": [
        "el ministerio de salud", "los medicos especialistas", "el hospital general",
        "la organizacion mundial de la salud", "los centros de salud publica",
        "el instituto nacional de laboratorios", "los profesionales de la salud",
        "la sociedad medica", "el colegio medico", "la direccion de epidemiologia",
        "los servicios de emergencia", "el personal sanitario", "la facultad de medicina",
        "los investigadores clinicos", "el comite de bioetica", "la asociacion de enfermeras",
        "los voluntarios de salud", "el cuerpo medico", "la cruz roja",
        "los tecnicos de laboratorio", "el servicio nacional de salud",
        "la fundacion contra el cancer", "los nutricionistas", "el centro de rehabilitacion",
        "la brigada medica", "los residentes de medicina",
    ],
    "internacional": [
        "el gobierno extranjero", "la organizacion de naciones unidas", "el lider mundial",
        "la cancilleria", "las potencias occidentales", "el secretario general de la onu",
        "la union europea", "el consejo de seguridad", "los paises miembros",
        "la comunidad internacional", "el cuerpo diplomatico", "la liga de naciones arabes",
        "los ministros de exteriores", "la organizacion de estados americanos",
        "el embajador acreditado", "las fuerzas de paz", "la coalicion internacional",
        "los paises vecinos", "el bloque regional", "la cumbre iberoamericana",
        "los representantes diplomaticos", "el enviado especial", "la agencia de cooperacion",
        "los observadores internacionales", "el alto comisionado", "la asamblea general",
        "el tratado de libre comercio", "la organizacion mundial del comercio",
    ],
    "cultura": [
        "los artistas locales", "el museo nacional", "la compania teatral",
        "el escritor reconocido", "el festival internacional", "los cineastas nacionales",
        "la direccion de cultura", "la orquesta sinfonica", "el ballet clasico",
        "los musicos populares", "la fundacion cultural", "el centro cultural",
        "la asociacion de artistas", "los pintores contemporaneos", "el conservatorio de musica",
        "la academia de bellas artes", "los dramaturgos", "la compania de danza",
        "el instituto de cultura", "la biblioteca nacional", "los arquitectos locales",
        "la casa de la cultura", "el colectivo artistico", "los artesanos de la region",
        "la revista literaria", "el circuló de escritores", "la filarmonica nacional",
    ],
}

# ---------------------------------------------------------------------------
# Acciones (verbos) por categoria. Mucho mas variadas que v1.
# ---------------------------------------------------------------------------
ACCIONES = {
    "politica": [
        "anuncio un proyecto de ley que busca reformar el sistema electoral",
        "rechazo la propuesta legislativa presentada por la bancada opositora",
        "convoco a elecciones generales para el proximo periodo gubernamental",
        "firmo un decreto supremo que establece nuevas medidas administrativas",
        "presento su renuncia irrevocable al cargo tras las recientes controversias",
        "asumio el nuevo cargo en una ceremonia realizada en el palacio de gobierno",
        "debatio la reforma constitucional en una sesion que duro varias horas",
        "aprobo el presupuesto nacional para el proximo ano fiscal con mayoria simple",
        "critico la gestion anterior señalando deficiencias en la administracion publica",
        "busco alianzas con la oposicion para lograr consenso en el congreso",
        "sanciono la nueva ley de transparencia y acceso a la informacion publica",
        "promulgo el codigo penal tras un largo proceso de discusion parlamentaria",
        "ratifico el gabinete ministerial con cambios en tres ministerios clave",
        "insto a la unidad nacional en medio de la crisis politica que atraviesa el pais",
        "convoco a un referendo constitucional para definir el modelo economico",
        "prometio invertir en obra publica como parte de su programa de gobierno",
        "advirtio que no tolerara actos de corrupcion dentro de la administracion",
        "negocio con los sindicatos para evitar un paro nacional indefinido",
        "retiro el proyecto de ley tras las fuertes criticas de la sociedad civil",
        "implemento un plan de digitalizacion de los servicios del estado",
        "presento un paquete de medidas para combatir la inseguridad ciudadana",
        "fijo la fecha para las elecciones departamentales y municipales",
        "sometio a consulta popular la modificacion de la constitucion politica",
        "ratifico el compromiso con la democracia y el estado de derecho",
    ],
    "economia": [
        "anuncio un plan de estímulo economico para reactivar los sectores productivos",
        "reviso las tasas de interes en respuesta a la situacion del mercado financiero",
        "reporto el crecimiento del producto interno bruto en el ultimo trimestre",
        "advirtio sobre la inflacion creciente y sus efectos en el poder adquisitivo",
        "firmo un acuerdo comercial bilateral que abrira nuevos mercados de exportacion",
        "proyecto una recuperacion gradual de la economia nacional para el proximo ano",
        "elevo la meta de inflacion anual para adaptarse al contexto internacional",
        "confirmo la devaluacion de la moneda nacional frente al dolar estadounidense",
        "anuncio incentivos fiscales para la inversion extranjera directa en el pais",
        "revelo datos del desempleo trimestral que muestran una tendencia a la baja",
        "redujo las tasas de interes para estimular el credito al sector privado",
        "informo sobre el comportamiento de la balanza comercial en el ultimo semestre",
        "destino fondos del tesoro publico para apoyar a las pequenas y medianas empresas",
        "negocio con el fondo monetario internacional una nueva linea de credito",
        "establecio controles de cambio para frenar la fuga de divisas",
        "publico el informe mensual sobre la evolucion de los precios al consumidor",
        "anuncio la emision de bonos soberanos para financiar el presupuesto nacional",
        "reviso el tipo de cambio oficial tras las fluctuaciones del mercado",
        "pronostico un crecimiento economico moderado para el cierre del ano fiscal",
        "implemento medidas arancelarias para proteger la produccion nacional",
        "reporto un superavit en la balanza comercial gracias al aumento de las exportaciones",
        "advirtio sobre los riesgos de recesion en la economia global",
        "promovio la formalizacion del comercio mediante incentivos tributarios",
        "anuncio un programa de financiamiento para el sector agricola y agroindustrial",
    ],
    "deportes": [
        "gano el clasico nacional con un marcador amplio ante su eterno rival",
        "convoco a los jugadores para la proxima fecha de las eliminatorias mundialistas",
        "renovo el contrato del tecnico por dos temporadas mas tras los buenos resultados",
        "sufrio una lesion grave que lo marginara del resto del torneo",
        "clasifico a la siguiente ronda del torneo continental tras una victoria ajustada",
        "anuncio el retiro definitivo del futbol profesional despues de una destacada carrera",
        "celebro el titulo obtenido en la final del campeonato nacional",
        "cayo derrotado en la final del torneo internacional ante un rival superior",
        "anoto dos goles en el partido que definio al campeon de la liga",
        "logro el record mundial en la disciplina de atletismo durante la competencia",
        "obtuvo la medalla de oro en la competencia internacional de natacion",
        "aseguro su clasificacion al mundial tras vencer al seleccionado visitante",
        "firmo un contrato millonario con el club europeo por cinco temporadas",
        "dirigio su ultimo partido como entrenador tras una decada en el banquillo",
        "rompio el record de goles en una sola temporada de la liga profesional",
        "anuncio su incorporacion al cuerpo tecnico de la seleccion nacional",
        "sufrio una derrota ante el equipo visitante en el estadio principal",
        "conquisto el bicampeonato nacional tras una temporada invicta",
        "fue expulsado en el partido tras una fuerte falta sobre el rival",
        "anuncio la contratacion de un delantero extranjero para la proxima temporada",
        "obtuvo la clasificacion a los juegos olimpicos en la prueba de fondo",
        "celebro el ascenso a la categoria de honor de la liga departamental",
        "revelo la lista de convocados para el proximo compromiso internacional",
        "cerro su participacion en el torneo con una actuacion destacada",
    ],
    "tecnologia": [
        "lanzo un nuevo dispositivo movil con funciones de inteligencia artificial integradas",
        "presento una plataforma en la nube orientada a empresas y desarrolladores",
        "adquirio una startup emergente especializada en procesamiento de datos",
        "reporto una falla de seguridad que afecto a millones de usuarios globalmente",
        "anuncio una alianza estrategica con otra compania del sector tecnologico",
        "mostro avances significativos en modelos de lenguaje de gran escala",
        "libero una version beta de su aplicacion para pruebas con usuarios seleccionados",
        "confirmo la filtracion de datos personales de sus clientes en un ciberataque",
        "desarrollo un nuevo algoritmo de aprendizaje automatico para deteccion de fraudes",
        "integro herramientas de automatizacion en su suite de productos empresariales",
        "anuncio el lanzamiento de un procesador de nueva generacion con mayor eficiencia",
        "presento un sistema de reconocimiento facial con aplicaciones en seguridad",
        "implemeto medidas de cifrado de extremo a extremo para proteger las comunicaciones",
        "revelo una vulnerabilidad en sus servidores que fue corregida de inmediato",
        "anuncio la expansion de su red de cobertura a nuevas regiones del pais",
        "firmo un convenio con el sector educativo para donar equipos tecnologicos",
        "desarrollo una aplicacion movil para facilitar tramites administrativos",
        "presento un prototipo de vehiculo autonomo con tecnologia de conduccion asistida",
        "anuncio la apertura de un centro de innovacion y desarrollo de software",
        "lanzo una plataforma de streaming de contenido digital para america latina",
        "integro tecnologia blockchain para garantizar la trazabilidad de transacciones",
        "reporto un crecimiento sostenido en suscripciones a sus servicios digitales",
        "anuncio el desarrollo de un asistente virtual con capacidad de dialogo natural",
        "presento avances en robotics aplicada a la industria manufacturera",
    ],
    "salud": [
        "confirmo un brote epidemiologico en la region norte del pais",
        "recomendo medidas preventivas a la poblacion ante el aumento de casos",
        "reporto una baja significativa en los casos de la enfermedad en el ultimo mes",
        "anuncio una campana de vacunacion masiva para toda la poblacion infantil",
        "advirtio sobre el consumo excesivo de azucar y sus efectos en la salud publica",
        "inauguro un nuevo centro asistencial equipado con tecnologia de punta",
        "publico un estudio clinico sobre los efectos del nuevo tratamiento medico",
        "detecto una nueva variante del virus que esta circulando en la region",
        "autorizo el uso de emergencia del medicamento tras ensayos clinicos exitosos",
        "reporto avances significativos en el tratamiento de enfermedades cronicas",
        "declaro alerta sanitaria en varios municipios por contaminacion del agua",
        "anuncio un plan de mejora en la infraestructura hospitalaria del sistema publico",
        "recomendo la vacunacion anual como medida de prevencion para grupos de riesgo",
        "informo sobre el aumento de casos de dengue en la temporada de lluvias",
        "coordino con los gobiernos locales acciones de prevencion y control sanitario",
        "reporto un descenso en la tasa de mortalidad materna e infantil",
        "anuncio la distribucion gratuita de medicamentos esenciales en centros de salud",
        "advirtio sobre los riesgos del consumo de tabaco y alcohol en jovenes",
        "implanto un sistema de vigilancia epidemiologica a nivel nacional",
        "revelo datos alarmantes sobre el aumento de obesidad en la poblacion adulta",
        "anuncio la capacitacion de personal medico en tecnicas de atencion especializada",
        "confirmo la erradicacion de una enfermedad transmisible en el territorio nacional",
        "reporto el cumplimiento de las metas del programa de salud publica",
        "llamo a la poblacion a mantener las medidas de higiene personal y colectiva",
    ],
    "internacional": [
        "firmo un tratado bilateral de cooperacion en materia de seguridad y defensa",
        "convoco a una cumbre diplomatica para abordar la crisis regional",
        "condeno los actos de violencia y llamo al dialogo entre las partes involucradas",
        "reunio a los ministros de exteriores para discutir la agenda regional",
        "anuncio sanciones economicas contra el regimen que vulnera los derechos humanos",
        "acordo un cese al fuego temporal para permitir la entrega de ayuda humanitaria",
        "reclamo medidas contra el cambio climatico en la cumbre ambiental mundial",
        "negocio un acuerdo migratorio para regular el flujo de personas entre los paises",
        "romoio relaciones diplomaticas tras el incidente que afecto a los ciudadanos",
        "recibio al enviado especial que trae un mensaje del gobierno extranjero",
        "expreso su preocupacion por la situacion politica y social del pais vecino",
        "anuncio un paquete de ayuda internacional para los paises afectados por la crisis",
        "logro un consenso en la resolucion sobre comercio y desarrollo sostenible",
        "rechazo la intervencion extranjera en los asuntos internos de la region",
        "reafirmo el compromiso con la paz y la seguridad internacional",
        "promovio una iniciativa para el desarme y la no proliferacion de armamentos",
        "coordino acciones de cooperacion para combatir el narcotrafico y el crimen organizado",
        "anuncio la apertura de una nueva embajada en el pais despues de varios anos",
        "llamo a la comunidad internacional a responder ante la crisis humanitaria",
        "sometio a votacion una resolucion sobre derechos humanos en el consejo",
        "firmo un acuerdo de cooperacion tecnica y cientifica con paises aliados",
        "advirtio sobre los riesgos de un conflicto armado si no se retoma el dialogo",
        "recibio el respaldo de la comunidad internacional para su gestion diplomatica",
        "celebro el acuerdo alcanzado tras intensas jornadas de negociacion",
    ],
    "cultura": [
        "inauguro una exposicion de arte contemporaneo que reune obras de artistas nacionales",
        "gano un premio internacional en reconocimiento a su trayectoria artistica",
        "presento una obra teatral basada en hechos historicos del pais",
        "anuncio el programa oficial del festival que se realizara el proximo mes",
        "rindo homenaje al artista local considerado patrimonio cultural de la nacion",
        "estreno una pelicula aclamada por la critica en el festival de cine internacional",
        "lanzo un libro de gran aceptacion entre el publico y la critica literaria",
        "recibio el galardon nacional en la categoria de artes visuales y plasticas",
        "abrio las puertas del museo remodelado tras varios meses de trabajo",
        "celebro el dia del patrimonio con actividades culturales gratuitas para todo publico",
        "anuncio la restauracion de monumentos historicos en el centro de la ciudad",
        "presento un concierto sinfonico con obras de compositores clasicos y contemporaneos",
        "organizo un ciclo de cine que recupera las producciones nacionales mas destacadas",
        "inauguro la feria internacional del libro con la participacion de autores extranjeros",
        "anuncio un programa de fomento a la lectura en escuelas y bibliotecas publicas",
        "presento una muestra de artesania tradicional de las regiones del pais",
        "rindio tributo al musico nacional en un concierto conmemorativo",
        "anuncio la creacion de una beca para jovenes talentos en las artes escénicas",
        "organizo un taller de pintura y escultura abierto a la comunidad",
        "recupero un manuscrito historico que se creia perdido para la cultura nacional",
        "presento un documental sobre las tradiciones y costumbres de los pueblos originarios",
        "inauguro una sala de conciertos con acustica de nivel internacional",
        "celebro los cien anos del teatro municipal con una gala especial",
        "anuncio la declaracion de patrimonio cultural inmaterial para una tradicion local",
    ],
}

# ---------------------------------------------------------------------------
# Vocabulario de contexto por categoria (refuerza la señal tematica).
# ---------------------------------------------------------------------------
CONTEXTO_CAT = {
    "politica": [
        "congreso nacional", "parlamento", "elecciones generales", "partido politico",
        "votacion", "campana electoral", "senado", "legislatura", "ley electoral",
        "reforma constitucional", "decreto supremo", "gabinete ministerial", "presupuesto nacional",
        "crisis politica", "gestion gubernamental", "administracion publica", "palacio de gobierno",
        "asamblea legislativa", "comision de gobierno", "tribunal electoral",
        "ley de transparencia", "codigo penal", "obra publica", "estado de derecho",
        "corrupcion administrativa", "unidad nacional", "referendo constitucional",
        "sindicatos", "paro nacional", "consulta popular", "seguridad ciudadana",
    ],
    "economia": [
        "bolsa de valores", "mercado financiero", "inflacion", "exportaciones",
        "producto interno bruto", "finanzas publicas", "banca comercial", "comercio exterior",
        "tasas de interes", "politica monetaria", "devaluacion", "moneda nacional",
        "dolar estadounidense", "inversion extranjera", "desempleo", "credito",
        "balanza comercial", "pequenas y medianas empresas", "fondo monetario internacional",
        "controles de cambio", "precios al consumidor", "bonos soberanos", "presupuesto fiscal",
        "tipo de cambio", "recesion economica", "formalizacion comercial", "sector agricola",
        "superavit comercial", "medidas arancelarias", "linea de credito", "tesoro publico",
    ],
    "deportes": [
        "torneo", "liga profesional", "seleccion nacional", "estadio",
        "campeonato", "futbol", "entrenamiento", "temporada", "eliminatorias mundialistas",
        "lesion", "marcador", "rival", "banquillo", "goles", "final",
        "medalla de oro", "atletismo", "natacion", "record mundial", "competencia internacional",
        "club deportivo", "contrato millonario", "cuerpo tecnico", "ascenso",
        "expulsion", "delantero extranjero", "juegos olimpicos", "categoria de honor",
        "bicampeonato", "falta", "arbitro", "capitan del equipo",
    ],
    "tecnologia": [
        "software", "internet", "datos", "plataforma digital", "servidores",
        "innovacion", "algoritmo", "digital", "inteligencia artificial", "nube",
        "dispositivo movil", "startup", "ciberataque", "ciberseguridad", "filtreacion de datos",
        "automatizacion", "procesador", "reconocimiento facial", "cifrado", "vulnerabilidad",
        "cobertura de red", "vehiculo autonomo", "blockchain", "streaming",
        "robotics", "aprendizaje automatico", "modelos de lenguaje", "asistente virtual",
        "aplicacion movil", "centro de innovacion", "desarrollo de software",
    ],
    "salud": [
        "hospital", "pacientes", "virus", "vacuna", "epidemia", "clinica",
        "medicos", "sintomas", "brote epidemiologico", "medidas preventivas",
        "campana de vacunacion", "salud publica", "centro asistencial", "estudio clinico",
        "variante del virus", "medicamento", "tratamiento medico", "enfermedades cronicas",
        "alerta sanitaria", "infraestructura hospitalaria", "grupos de riesgo",
        "tasa de mortalidad", "dengue", "vigilancia epidemiologica", "obesidad",
        "medicamentos esenciales", "tabaco", "alcohol", "higiene personal",
        "erradicacion", "programa de salud", "capacitacion medica",
    ],
    "internacional": [
        "cumbre diplomatica", "tratado bilateral", "frontera", "relaciones exteriores",
        "embajador", "diplomacia", "region", "organizacion de naciones unidas",
        "consejo de seguridad", "paises miembros", "comunidad internacional", "cuerpo diplomatico",
        "derechos humanos", "sanciones economicas", "cese al fuego", "ayuda humanitaria",
        "cambio climatico", "acuerdo migratorio", "relaciones diplomaticas", "enviado especial",
        "crisis humanitaria", "paz internacional", "desarme", "narcotrafico",
        "crimen organizado", "embajada", "resolucion", "cooperacion tecnica",
        "conflicto armado", "negociacion", "apoyo internacional",
    ],
    "cultura": [
        "museo", "arte", "teatro", "cine", "literatura", "festival",
        "patrimonio cultural", "exposicion", "obra teatral", "premio internacional",
        "trayectoria artistica", "pelicula", "critica literaria", "concierto sinfonico",
        "compositores", "artesania tradicional", "monumentos historicos", "artes visuales",
        "feria internacional del libro", "fomento a la lectura", "bibliotecas publicas",
        "pintura y escultura", "musica nacional", "artes escénicas", "documental",
        "tradiciones", "pueblos originarios", "sala de conciertos", "teatro municipal",
        "patrimonio cultural inmaterial", "gala especial",
    ],
}

# ---------------------------------------------------------------------------
# Frases neutras de relleno periodistico (aparecen en cualquier categoria).
# ---------------------------------------------------------------------------
RELLENO = [
    "La conferencia de prensa se realizo en la sede central durante la manana de ayer.",
    "Los detalles de la medida se daran a conocer en los proximos dias mediante un comunicado oficial.",
    "La informacion fue confirmada por fuentes oficiales que prefirieron mantener el anonimato.",
    "Se espera una pronta declaracion al respecto por parte de las autoridades competentes.",
    "La medida entrara en vigencia a partir del proximo mes segun lo establecido en la normativa.",
    "Hasta el momento no ha habido comentarios adicionales por parte de los involucrados.",
    "El anuncio se produjo tras una reunion prolongada que se extendio por varias horas.",
    "Las partes involucradas aun no han emitido comunicados oficiales sobre el particular.",
    "El evento conto con la presencia de autoridades y representantes de diversos sectores.",
    "La situacion sigue su curso habitual mientras se aguardan definiciones mas claras.",
    "Pronto se conoceran mas especificaciones tecnicas del proyecto anunciado.",
    "La decision fue tomada por mayoria tras un amplio debate entre los participantes.",
    "Segun informaron fuentes cercanas al proceso, la implementacion sera gradual.",
    "El organismo difundio un comunicado en el que explica los alcances de la medida.",
    "Se aguarda la reaccion de los distintos sectores que podrian verse afectados.",
    "Los expertos consultados señalaron que es necesario analizar el impacto a largo plazo.",
    "La proxima semana se llevara a cabo una nueva reunion para evaluar los avances.",
    "Segun el cronograma establecido, las acciones comenzaran a ejecutarse de inmediato.",
    "Representantes de distintos sectores participaron activamente en la jornada de trabajo.",
    "El encuentro tuvo lugar en las instalaciones principales con cobertura de prensa acreditada.",
]

# ---------------------------------------------------------------------------
# Acciones AMBIGUAS: verbos genericos que aparecen en varias categorias.
# Estas obligan al modelo a mirar el contexto, no solo el verbo.
# Se mezclan con el contexto de la categoria para desambiguar.
# ---------------------------------------------------------------------------
ACCIONES_AMBIGUAS = [
    "anuncio una serie de medidas que seran implementadas progresivamente",
    "presento un nuevo plan destinado a mejorar la situacion actual",
    "confirmo que se llevaran a cabo acciones concretas en los proximos dias",
    "revelo informacion importante sobre el proceso en curso",
    "informo sobre los avances logrados en las ultimas semanas",
    "destaco la importancia de las acciones emprendidas recientemente",
    "advirtio que la situacion requiere atencion inmediata de las autoridades",
    "rechazo las criticas y defenso la gestion realizada hasta el momento",
    "llamo a la unidad y al trabajo conjunto de todos los sectores involucrados",
    "expreso su compromiso con las medidas adoptadas y sus resultados",
]

# ---------------------------------------------------------------------------
# Hard negatives: contextos que parecen de otra categoria pero la etiqueta
# correcta es distinta. El modelo debe aprender que el CONTEXTO dominante
# (no una palabra aislada) define la categoria.
# Formato: (categoria_real, frase_contextual) donde la frase menciona temas
# de otra categoria pero la noticia es de la categoria_real.
# ---------------------------------------------------------------------------
HARD_NEGATIVES = [
    # Politica que habla de economia pero es decision politica
    ("politica", "La medida, que afecta el presupuesto nacional, fue dictada mediante decreto supremo por el presidente en el palacio de gobierno."),
    ("politica", "Tras un intenso debate en el congreso, la ley fue aprobada con votos del partido gobernante y la oposicion."),
    # Economia que menciona salud/tecnologia pero es tema economico
    ("economia", "La inversion en el sector salud genero un crecimiento del producto interno bruto en el ultimo trimestre."),
    ("economia", "Las empresas tecnologicas lideraron las exportaciones y mejoraron la balanza comercial del pais."),
    # Deportes que menciona politica/economia
    ("deportes", "El club deportivo firmo un contrato millonario que supera las cifras de cualquier transferencia previa en la liga."),
    ("deportes", "El seleccionado nacional disputara el clasico regional tras vencer a su rival en el estadio."),
    # Tecnologia que menciona salud/cultura
    ("tecnologia", "La plataforma digital desarrollo un sistema de inteligencia artificial para procesar grandes volumenes de datos."),
    ("tecnologia", "La startup lanzo una aplicacion movil con funciones de automatizacion y procesamiento en la nube."),
    # Salud que menciona politica/economia
    ("salud", "El hospital general implemento un protocolo de prevencion tras confirmar un brote epidemiologico en la region."),
    ("salud", "El ministerio de salud reporto una baja en los casos tras la campana de vacunacion masiva."),
    # Internacional que menciona economia/politica
    ("internacional", "La organizacion de naciones unidas convoco a una cumbre diplomatica para abordar la crisis regional."),
    ("internacional", "El consejo de seguridad debatio las sanciones economicas contra el regimen que vulnera los derechos humanos."),
    # Cultura que menciona economia/politica
    ("cultura", "El museo nacional inauguro una exposicion de arte contemporaneo con obras de artistas locales reconocidos."),
    ("cultura", "El festival internacional de cine presento una pelicula aclamada por la critica especializada."),
]
COLETILLAS_POS = [
    "Los sectores involucrados celebraron la decision y la calificaron de acertada.",
    "La noticia fue recibida con optimismo por los analistas consultados.",
    "Se trata de un avance significativo para el pais segun la mayoria de los observadores.",
    "Expertos calificaron el hecho de historicamente positivo para la region.",
    "La poblacion manifesto su apoyo generalizado a la medida anunciada.",
    "Los beneficiarios directos expresaron su satisfaccion por la gestion realizada.",
]
COLETILLAS_NEG = [
    "Criticos cuestionaron la falta de planificacion en la implementacion de la medida.",
    "La oposicion rechazo de plano la decision y anuncio que pedira su reconsideracion.",
    "Se temen efectos adversos en los proximos meses segun advirtieron los especialistas.",
    "Voces autorizadas advirtieron graves consecuencias si no se corrige el rumbo.",
    "El sector afectado manifesto su profunda preocupacion por los resultados obtenidos.",
    "Organizaciones civiles exigieron mayor claridad sobre el alcance de la medida.",
]
COLETILLAS_NEU = [
    "El organismo difundio un comunidado sin emitir opiniones al respecto.",
    "Se aguardan mas detalles en breve sobre los proximos pasos a seguir.",
    "No hubo declaraciones oficiales adicionales tras el encuentro.",
    "La situacion permanece sin cambios sustanciales segun los observadores.",
    "El tema sigue en evaluacion por parte de los tecnicos responsables.",
]

# ---------------------------------------------------------------------------
# Marcadores de tono sensacionalista.
# ---------------------------------------------------------------------------
CLICKBAIT_INICIOS = [
    "NO CREERAS lo que paso",
    "ESCANDALO total al descubrirse",
    "Te sorprendera saber que",
    "EXCLUSIVO: filtran que",
    "IMPACTANTE: confirman que",
    "Nadie esperaba esto:",
    "El secreto mejor guardado:",
    "URGENTE: alertan que",
]

PALABRAS_EMOCIONALES = [
    "conmocionado", "escandalizado", "estremecedor", "impactante",
    "increible", "bomba", "escandalo", "catastrofico", "historico",
    "sin precedentes", "explosivo", "alarmante", "devastador",
]

# Datos numericos realistas para enriquecer el texto.
NUMEROS = [
    "320 millones", "1.5 por ciento", "15 mil millones", "48 horas",
    "el ultimo trimestre", "la presente gestion", "los ultimos seis meses",
    "el 30 por ciento", "mas de dos mil personas", "cinco departamentos",
    "12 millones de habitantes", "el periodo 2024-2025", "tres fases",
    "un plazo de 90 dias", "una inversion de 50 millones",
]


def generar_noticia(categoria, tono, sentimiento, semilla=None):
    """Genera una noticia sintetica rica en vocabulario.

    Con probabilidad ~20% usa una accion AMBIGUA (verbo generico que aparece
    en varias categorias), obligando al modelo a usar el contexto para
    desambiguar. Con probabilidad ~10% usa un HARD NEGATIVE (frase que
    menciona temas de otra categoria pero la etiqueta correcta es 'categoria').
    """
    rng = random.Random(semilla)
    sujeto = rng.choice(SUJETOS[categoria])

    # 20% de las veces: accion ambigua (verbo generico).
    if rng.random() < 0.20:
        accion = rng.choice(ACCIONES_AMBIGUAS)
    else:
        accion = rng.choice(ACCIONES[categoria])

    # Coletilla de sentimiento.
    if sentimiento == "positivo":
        coletilla = rng.choice(COLETILLAS_POS)
    elif sentimiento == "negativo":
        coletilla = rng.choice(COLETILLAS_NEG)
    else:
        coletilla = rng.choice(COLETILLAS_NEU)

    # Contexto de categoria (1-2 frases con vocabulario tematico).
    # Cuando la accion es ambigua, el contexto es lo que desambigua.
    contexto1 = rng.choice(CONTEXTO_CAT[categoria])
    contexto2 = rng.choice(CONTEXTO_CAT[categoria]) if rng.random() < 0.5 else None

    # Relleno periodistico (1-2 frases neutras).
    relleno1 = rng.choice(RELLENO)
    relleno2 = rng.choice(RELLENO) if rng.random() < 0.5 else None

    # Dato numerico ocasional.
    numero = rng.choice(NUMEROS) if rng.random() < 0.4 else None

    # Hard negative: 10% de las veces, anade una frase que menciona otra
    # categoria pero la etiqueta sigue siendo 'categoria' (definida por
    # el sujeto + contexto dominante).
    hard_neg = None
    if rng.random() < 0.10:
        hards_cat = [h for c, h in HARD_NEGATIVES if c == categoria]
        if hards_cat:
            hard_neg = rng.choice(hards_cat)

    # Construir oraciones segun tono.
    if tono == "sensacionalista":
        inicio = rng.choice(CLICKBAIT_INICIOS)
        emocional = rng.choice(PALABRAS_EMOCIONALES)
        oraciones = [
            f"{inicio}, {sujeto} {accion}",
            f"En un hecho {emocional}, la medida guarda relacion con {contexto1}.",
        ]
        if contexto2:
            oraciones.append(f"El tema de {contexto2} tambien esta en el centro del debate.")
        if hard_neg:
            oraciones.append(hard_neg)
        oraciones.append(relleno1)
        oraciones.append(coletilla)
        if relleno2:
            oraciones.append(relleno2)
        if numero:
            oraciones.append(f"Segun datos oficiales, la cifra asciende a {numero}.")
        if rng.random() < 0.7:
            oraciones[-1] += "!"
        texto = " ".join(oraciones)
    else:
        oraciones = [
            f"{sujeto.capitalize()} {accion}.",
        ]
        if contexto1:
            oraciones.append(f"La decision esta vinculada al ambito de {contexto1}.")
        if contexto2:
            oraciones.append(f"Tambien se considera el impacto sobre {contexto2}.")
        if hard_neg:
            oraciones.append(hard_neg)
        oraciones.append(relleno1)
        oraciones.append(coletilla)
        if relleno2:
            oraciones.append(relleno2)
        if numero:
            oraciones.append(f"Segun el reporte oficial, los datos alcanzan {numero}.")
        # Limpiar: cada oracion termina con punto, sin dobles.
        limpias = []
        for f in oraciones:
            f = f.strip()
            if not f.endswith((".", "!", "?")):
                f += "."
            limpias.append(f)
        texto = " ".join(limpias)

    return {
        "texto": texto,
        "categoria": categoria,
        "tono": tono,
        "sentimiento": sentimiento,
    }


def generar_dataset(n_total=1500, semilla=42):
    """Genera el dataset balanceado por categoria."""
    rng = random.Random(semilla)
    noticias = []
    por_cat = n_total // len(CATEGORIAS)
    resto = n_total - por_cat * len(CATEGORIAS)
    for i, cat in enumerate(CATEGORIAS):
        n_cat = por_cat + (1 if i < resto else 0)
        for j in range(n_cat):
            tono = rng.choices(["informativo", "sensacionalista"], weights=[0.6, 0.4])[0]
            sent = rng.choices(["positivo", "negativo", "neutro"], weights=[0.4, 0.4, 0.2])[0]
            noticia = generar_noticia(cat, tono, sent, semilla=semilla + i * 10000 + j)
            noticias.append(noticia)
    rng.shuffle(noticias)
    for i, n in enumerate(noticias):
        n["id"] = i
    return noticias


def dividir_dataset(noticias, test_size=0.15, val_size=0.15, semilla=42):
    rng = random.Random(semilla)
    por_cat = {c: [n for n in noticias if n["categoria"] == c] for c in CATEGORIAS}
    train, val, test = [], [], []
    for c in CATEGORIAS:
        items = por_cat[c][:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def guardar_dataset(noticias, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(noticias, f, ensure_ascii=False, indent=2)


def cargar_dataset(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base = Path(__file__).parent
    noticias = generar_dataset(n_total=1500, semilla=42)
    train, val, test = dividir_dataset(noticias, semilla=42)
    guardar_dataset(train, base / "train.json")
    guardar_dataset(val, base / "val.json")
    guardar_dataset(test, base / "test.json")
    guardar_dataset(noticias, base / "full.json")
    print(f"Dataset generado: {len(noticias)} noticias totales")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    from collections import Counter
    # Vocabulario unico.
    vocab = set()
    for n in noticias:
        vocab.update(n["texto"].lower().split())
    print(f"  Vocabulario unico: {len(vocab)} tokens")
    lens = [len(n["texto"].split()) for n in noticias]
    print(f"  Palabras/noticia: min={min(lens)} max={max(lens)} media={sum(lens)/len(lens):.0f}")
    print("\nDistribucion por categoria:")
    for cat, n in sorted(Counter(n["categoria"] for n in noticias).items()):
        print(f"  {cat}: {n}")
    print("\nEjemplos:")
    for n in noticias[:3]:
        print(f"  [{n['categoria']}|{n['tono']}|{n['sentimiento']}] {n['texto'][:110]}...")


if __name__ == "__main__":
    main()
