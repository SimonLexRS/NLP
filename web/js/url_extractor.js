// url_extractor.js — Extrae el texto de una noticia desde su URL.
//
// Como el sitio es 100% estático (GitHub Pages, sin backend), usamos
// servicios de proxy/reader para evitar el bloqueo CORS del navegador.
//
// Estrategia (en orden de fallback):
//   1. https://r.jina.ai/            (Jina Reader: devuelve Markdown limpio, CORS-friendly)
//   2. https://api.allorigins.win/    (proxy CORS crudo, para sitios con SSR)
//   3. https://corsproxy.io/          (ultimo recurso)
//
// Ademas incluye validacion: detecta URLs que no son noticias (landing pages
// de marketing, encuestas, spam) y devuelve una advertencia.

const READERS = [
  {
    nombre: "Jina Reader",
    build: (url) => `https://r.jina.ai/${url}`,
    esMarkdown: true,
  },
  {
    nombre: "AllOrigins",
    build: (url) => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
    esMarkdown: false,
  },
  {
    nombre: "CorsProxy",
    build: (url) => `https://corsproxy.io/?url=${encodeURIComponent(url)}`,
    esMarkdown: false,
  },
];

/**
 * Descarga el contenido de una URL vía los servicios de reader/proxy.
 * Prueba en orden hasta que uno funcione.
 * Devuelve: { contenido, esMarkdown, readerUsado }
 */
async function fetchContenido(url) {
  let ultimoError = null;
  for (const reader of READERS) {
    try {
      const readerUrl = reader.build(url);
      const resp = await fetch(readerUrl, {
        method: "GET",
        signal: AbortSignal.timeout(20000),
        headers: { Accept: reader.esMarkdown ? "text/plain" : "text/html" },
      });
      if (resp.ok) {
        const contenido = await resp.text();
        if (contenido && contenido.length > 200) {
          return { contenido, esMarkdown: reader.esMarkdown, readerUsado: reader.nombre };
        }
      }
      ultimoError = new Error(`${reader.nombre}: HTTP ${resp.status}`);
    } catch (e) {
      ultimoError = new Error(`${reader.nombre}: ${e.message}`);
    }
  }
  throw new Error(
    `No se pudo descargar la URL desde ningún servicio. ${ultimoError ? ultimoError.message : ""}. ` +
      `El sitio puede bloquear la extracción automática o requerir JavaScript.`
  );
}

// Frases tipicas de UI/navegacion que Jina Reader extrae como ruido.
// Se eliminan del texto final para dejar solo contenido periodistico.
const FRASES_BASURA = [
  "continuar leyendo", "ver más", "ver mas", "leer más", "leer mas",
  "más para ti", "mas para ti", "más información", "mas informacion",
  "suscríbete", "suscribete", "suscríbete ahora", "regístrate", "registrate",
  "iniciar sesión", "iniciar sesion", "inicia sesión", "inicia sesion",
  "comentarios", "comentar", "deja tu comentario",
  "compartir", "comparte", "twittear", "enviar por email",
  "reproducir video", "ver video", "escucha el audio",
  "recomendado", "recomendados", "también te puede interesar", "tambien te puede interesar",
  "publicidad", "anuncio", "patrocinado", "contenido patrocinado",
  "temas relacionados", "relacionados", "enlaces relacionados",
  "volver arriba", "ir al inicio", "página principal", "pagina principal",
  "seguir leyendo", "siguiente", "anterior",
  "© todos los derechos reservados", "política de privacidad", "politica de privacidad",
  "términos de uso", "terminos de uso", "cookies",
  "actualizado", "publicado",
  "fuentes", "fuente",
];

/**
 * Extrae texto util de una respuesta Markdown (Jina Reader).
 * El Markdown de Jina suele tener: "Title: ...\n\nURL Source: ...\n\nMarkdown Content:\n\n..."
 * Filtra frases de UI/navegacion y parrafos muy cortos.
 */
function extraerDeMarkdown(md) {
  // Jina Reader anade cabeceras; buscar "Markdown Content:" si existe.
  let contenido = md;
  const marker = "Markdown Content:";
  const idx = md.indexOf(marker);
  if (idx >= 0) {
    contenido = md.slice(idx + marker.length);
  }

  // Extraer titulo (Title: ...).
  let titulo = "";
  const titleMatch = md.match(/^Title:\s*(.+)$/m);
  if (titleMatch) titulo = titleMatch[1].trim();

  // Limpiar Markdown: quitar enlaces [texto](url) -> texto, imagenes, etc.
  let texto = contenido
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "") // imagenes
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // enlaces -> texto
    .replace(/^#{1,6}\s+/gm, "") // titulos markdown
    .replace(/\*\*(.+?)\*\*/g, "$1") // negritas
    .replace(/\*(.+?)\*/g, "$1") // cursivas
    .replace(/`{1,3}[^`]*`{1,3}/g, "") // codigo
    .replace(/^\s*[-*+]\s+/gm, "") // listas
    .replace(/^\s*\d+\.\s+/gm, "") // listas numeradas
    .replace(/^\s*>\s+/gm, "") // citas
    .replace(/---+/g, "") // separadores
    .trim();

  // Filtrar parrafos: eliminar frases basura (UI/navegacion) y parrafos cortos.
  const parrafos = texto.split(/\n+/).map((p) => p.trim()).filter(Boolean);
  const parrafosLimpios = parrafos.filter((p, idx) => {
    const pLower = p.toLowerCase();
    // Eliminar si es una frase basura exacta.
    const esBasura = FRASES_BASURA.some((fb) => pLower === fb || pLower === fb + ".");
    if (esBasura) return false;
    // Mantener siempre el primer parrafo (suele ser el titular, aunque sea corto).
    if (idx === 0) return true;
    // Eliminar parrafos muy cortos (probable UI: botones, labels) a partir del 2do.
    if (p.split(/\s+/).length < 3) return false;
    return true;
  });

  // Unir preservando estructura de parrafos (sin colapsar TODO a una linea).
  texto = parrafosLimpios.join("\n\n").trim();

  return { titulo, texto };
}

/**
 * Parsea HTML y extrae el texto de la noticia (heuristica).
 */
function extraerTextoDeHTML(html, url) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, "text/html");

  const getMeta = (prop) => {
    const el =
      doc.querySelector(`meta[property="${prop}"]`) ||
      doc.querySelector(`meta[name="${prop}"]`);
    return el?.getAttribute("content")?.trim() || "";
  };

  let titulo = getMeta("og:title") || doc.querySelector("h1")?.textContent?.trim() || "";
  let descripcion =
    getMeta("og:description") ||
    getMeta("description") ||
    getMeta("twitter:description") ||
    "";

  let contenedor =
    doc.querySelector("article") ||
    doc.querySelector("main") ||
    doc.querySelector('[role="main"]') ||
    doc.querySelector(".article-body") ||
    doc.querySelector(".post-content") ||
    doc.querySelector(".entry-content") ||
    doc.querySelector(".news-text") ||
    doc.querySelector("#cuerpo-noticia");

  let parrafos = [];
  if (contenedor) {
    parrafos = Array.from(contenedor.querySelectorAll("p"))
      .map((p) => p.textContent.trim())
      .filter((t) => t.length > 20);
  }
  if (parrafos.length === 0) {
    parrafos = Array.from(doc.querySelectorAll("p"))
      .map((p) => p.textContent.trim())
      .filter((t) => t.length > 30);
  }

  const partes = [];
  if (titulo) partes.push(titulo);
  if (descripcion) partes.push(descripcion);
  if (parrafos.length > 0) partes.push(parrafos.join(" "));
  const texto = partes.join("\n\n").replace(/\s+/g, " ").trim();

  return { titulo, descripcion, texto, nParrafos: parrafos.length };
}

// ---------------------------------------------------------------------------
// Deteccion de URLs que NO son noticias (marketing, spam, landing pages).
// ---------------------------------------------------------------------------

// Palabras/frases tipicas de marketing, encuestas, spam (no periodismo).
const PALABRAS_MARKETING = [
  "gana dinero", "ganar dinero", "encuestas pagadas", "encuesta remunerada",
  "trabaja desde casa", "trabajar desde casa", "ingresos extra", "ingresos pasivos",
  "oportunidad de negocio", "sé tu propio jefe", "ser tu propio jefe",
  "registrarse gratis", "regístrate gratis", "registrate ahora", "registrarte",
  "oferta limitada", "por tiempo limitado", "solo hoy", "ultima oportunidad",
  "descuento exclusivo", "promocion exclusiva", "oferta exclusiva",
  "dinero facil", "dinero rapido", "hacerte rico", "volverte rico",
  "ingresos garantizados", "ganancias garantizadas",
  "metodo probado", "secreto revelado", "truco infalible",
  "gana premios", "participa y gana", "premios en efectivo",
  "paypal", "transferencia bancaria", "retirar tus ganancias",
  "click aqui", "haz click", "descarga gratis", "descarga ahora",
  "bono de bienvenida", "bono gratis", "bonus de registro",
  "sin inversion", "sin experiencia", "no se requiere experiencia",
  "gana criptomonedas", "minado de criptomonedas", "airdrop",
  "apuestas deportivas", "casino online", "tiradas gratis",
];

// Vocabulario periodistico (si el texto tiene alguno, probablemente es noticia).
const VOCABLO_PERIODISTICO = [
  "gobierno", "presidente", "congreso", "ministerio", "senado", "asamblea",
  "parlamento", "elecciones", "partido", "diputado", "senador", "alcalde",
  "ley", "decreto", "reforma", "constitucion", "legislatura",
  "banco central", "economia", "inflacion", "producto interno bruto",
  "bolsa", "mercado", "exportaciones", "importaciones", "inversion",
  "seleccion", "equipo", "torneo", "liga", "campeonato", "partido",
  "jugador", "entrenador", "gol", "estadio", "futbol",
  "tecnologia", "software", "aplicacion", "plataforma", "internet",
  "inteligencia artificial", "algoritmo", "datos", "innovacion",
  "hospital", "pacientes", "medicos", "vacuna", "virus", "epidemia",
  "salud", "tratamiento", "enfermedad", "brote",
  "organizacion de naciones unidas", "onu", "cumbre", "tratado",
  "embajador", "diplomatico", "relaciones exteriores", "frontera",
  "museo", "exposicion", "festival", "artista", "concierto",
  "cine", "pelicula", "obra", "literatura", "patrimonio",
  "policial", "robado", "detenido", "fiscalia", "policia",
  "manifestantes", "protesta", "paro", "huelga",
  "informe", "reportaje", "corresponsal", "redaccion", "según",
];

/**
 * Valida si el texto extraido parece una noticia o es marketing/landing.
 *
 * Devuelve: { esNoticia: bool, razon: string, confianza: float, senales: [] }
 */
export function validarEsNoticia(texto) {
  if (!texto || typeof texto !== "string") {
    return { esNoticia: false, razon: "No se extrajo texto de la página.", confianza: 1.0, senales: ["texto vacío"] };
  }

  const textoLower = texto.toLowerCase();
  const senales = [];

  // 1. Palabras de marketing detectadas.
  const marketingHits = PALABRAS_MARKETING.filter((p) => textoLower.includes(p));
  if (marketingHits.length > 0) {
    senales.push(`Vocabulario de marketing/spam: "${marketingHits.slice(0, 3).join('", "')}"`);
  }

  // 2. Texto muy corto (probablemente landing page).
  const palabras = textoLower.split(/\s+/).filter(Boolean);
  if (palabras.length < 40) {
    senales.push(`Texto muy corto (${palabras.length} palabras) — parece una landing page, no un artículo`);
  }

  // 3. Densidad de vocabulario periodistico.
  const periodismoHits = VOCABLO_PERIODISTICO.filter((p) => textoLower.includes(p));

  // Decision.
  let esNoticia = true;
  let razon = "El texto parece una noticia.";

  if (marketingHits.length >= 2) {
    esNoticia = false;
    razon = `Se detectaron ${marketingHits.length} señales de marketing/spam. La URL probablemente no es una noticia.`;
  } else if (marketingHits.length >= 1 && periodismoHits.length === 0) {
    esNoticia = false;
    razon = "Se detectó vocabulario de marketing y no se encontró vocabulario periodístico. La URL no parece ser una noticia.";
  } else if (periodismoHits.length === 0 && palabras.length < 100) {
    esNoticia = false;
    razon = "El texto es corto y no contiene vocabulario periodístico. Probablemente es una landing page u otro tipo de contenido.";
  } else if (periodismoHits.length >= 1) {
    razon = `Contenido periodístico detectado (${periodismoHits.length} términos del dominio).`;
  }

  const confianza = Math.min(
    (periodismoHits.length * 0.15 + (marketingHits.length === 0 ? 0.3 : 0) + (palabras.length > 100 ? 0.2 : 0)),
    1.0
  );

  return { esNoticia, razon, confianza, senales, nPalabras: palabras.length, nPeriodismo: periodismoHits.length, nMarketing: marketingHits.length };
}

/**
 * Funcion principal: dada una URL, descarga y extrae el texto de la noticia.
 *
 * onEstado: callback(mensaje, tipo) para informar progreso al usuario.
 * Devuelve: { titulo, descripcion, texto, url, nParrafos, readerUsado, esMarkdown }
 */
export async function extraerNoticiaDeURL(url, onEstado = null) {
  if (!url) throw new Error("URL vacía");

  // Validar/normalizar URL.
  let urlNormalizada = url.trim();
  if (!/^https?:\/\//i.test(urlNormalizada)) {
    urlNormalizada = "https://" + urlNormalizada;
  }
  try {
    new URL(urlNormalizada);
  } catch {
    throw new Error("URL inválida");
  }

  if (onEstado) onEstado("Descargando página (probando servicios de lectura)...", "info");

  const { contenido, esMarkdown, readerUsado } = await fetchContenido(urlNormalizada);

  if (onEstado) onEstado(`Extraído vía ${readerUsado}. Procesando texto...`, "info");

  let resultado;
  if (esMarkdown) {
    const { titulo, texto } = extraerDeMarkdown(contenido);
    resultado = {
      titulo: titulo || "(sin título)",
      descripcion: "",
      texto: texto || "(no se pudo extraer texto)",
      url: urlNormalizada,
      nParrafos: texto ? texto.split(/\n+/).filter((p) => p.length > 20).length : 0,
      readerUsado,
      esMarkdown: true,
    };
  } else {
    const r = extraerTextoDeHTML(contenido, urlNormalizada);
    resultado = {
      titulo: r.titulo || "(sin título)",
      descripcion: r.descripcion || "",
      texto: r.texto || "(no se pudo extraer texto)",
      url: urlNormalizada,
      nParrafos: r.nParrafos,
      readerUsado,
      esMarkdown: false,
    };
  }

  if (resultado.texto.length < 50) {
    throw new Error(
      "Se descargó la página pero no se pudo extraer texto suficiente. " +
        "El sitio puede usar JavaScript para renderizar contenido o bloquear la extracción."
    );
  }

  return resultado;
}
