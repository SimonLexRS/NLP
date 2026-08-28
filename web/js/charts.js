// charts.js — Gráficos simples con Canvas vanilla (sin librerías externas).
// Dibuja barras horizontales y donuts para visualizar resultados.

// Paleta de colores por categoría.
const COLORES_CATEGORIA = {
  politica: "#dc2626",       // rojo
  economia: "#2563eb",       // azul
  deportes: "#16a34a",       // verde
  tecnologia: "#7c3aed",     // violeta
  salud: "#0891b2",          // cian
  internacional: "#ea580c",  // naranja
  cultura: "#db2777",        // rosa
};

const COLORES_SENTIMIENTO = {
  positivo: "#16a34a",
  negativo: "#dc2626",
  neutro: "#6b7280",
};

/**
 * Lee una custom property de :root. Los gráficos son SVG inline con colores
 * hardcodeados, así que toman el fondo/etiquetas del tema activo para seguir
 * siendo legibles en modo oscuro.
 */
function colorTema(nombre, alternativa) {
  if (typeof window === "undefined" || !document.documentElement) return alternativa;
  const valor = getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
  return valor || alternativa;
}

// Colores del tema, resueltos al dibujar (no al cargar el módulo) por si el
// usuario cambia el tema con la página abierta.
const pista = () => colorTema("--chart-track", "#e5e7eb");
const etiqueta = () => colorTema("--chart-label", "#374151");
const valorTxt = () => colorTema("--chart-value", "#6b7280");
const centro = () => colorTema("--chart-label", "#1f2937");

/**
 * Dibuja barras horizontales para una distribución de probabilidades.
 * container: elemento DOM donde se inserta el SVG.
 * datos: { etiqueta: valor } (valores 0-1).
 * colorMap: { etiqueta: color } opcional.
 */
export function dibujarBarras(container, datos, colorMap = {}) {
  const entradas = Object.entries(datos).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entradas.map(([, v]) => v), 0.001);

  const ancho = 320;
  const altoBarra = 26;
  const gap = 6;
  const alto = entradas.length * (altoBarra + gap) + 10;

  let svg = `<svg width="100%" height="${alto}" viewBox="0 0 ${ancho} ${alto}" preserveAspectRatio="xMinYMid meet" xmlns="http://www.w3.org/2000/svg">`;
  entradas.forEach(([label, valor], i) => {
    const y = i * (altoBarra + gap) + 5;
    const w = (valor / max) * (ancho - 120);
    const color = colorMap[label] || "#3b82f6";
    // Barra de fondo.
    svg += `<rect x="100" y="${y}" width="${ancho - 120}" height="${altoBarra}" fill="${pista()}" rx="3"/>`;
    // Barra de valor.
    svg += `<rect x="100" y="${y}" width="${Math.max(w, 1)}" height="${altoBarra}" fill="${color}" rx="3"/>`;
    // Etiqueta.
    svg += `<text x="95" y="${y + altoBarra / 2 + 4}" text-anchor="end" font-size="12" fill="${etiqueta()}" font-family="sans-serif">${label}</text>`;
    // Valor.
    const pct = (valor * 100).toFixed(1);
    svg += `<text x="${ancho - 8}" y="${y + altoBarra / 2 + 4}" text-anchor="end" font-size="11" fill="${valorTxt()}" font-family="sans-serif">${pct}%</text>`;
  });
  svg += "</svg>";

  container.innerHTML = svg;
}

/**
 * Dibuja un donut (gráfico circular con agujero) para una distribución.
 * container: elemento DOM.
 * datos: { etiqueta: valor }.
 * colorMap: { etiqueta: color }.
 * centroLabel: texto del centro.
 */
export function dibujarDonut(container, datos, colorMap = {}, centroLabel = "") {
  const entradas = Object.entries(datos);
  const total = entradas.reduce((s, [, v]) => s + v, 0) || 1;
  const size = 160;
  const r = 60;
  const rInterno = 38;
  const cx = size / 2;
  const cy = size / 2;

  let svg = `<svg width="100%" height="auto" viewBox="0 0 ${size} ${size}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" style="max-width:${size}px">`;
  let anguloInicio = -Math.PI / 2; // empezar arriba

  entradas.forEach(([label, valor]) => {
    const fraccion = valor / total;
    const anguloFin = anguloInicio + fraccion * 2 * Math.PI;
    const color = colorMap[label] || "#3b82f6";

    // Coordenadas del arco.
    const x1 = cx + r * Math.cos(anguloInicio);
    const y1 = cy + r * Math.sin(anguloInicio);
    const x2 = cx + r * Math.cos(anguloFin);
    const y2 = cy + r * Math.sin(anguloFin);
    const x1i = cx + rInterno * Math.cos(anguloInicio);
    const y1i = cy + rInterno * Math.sin(anguloInicio);
    const x2i = cx + rInterno * Math.cos(anguloFin);
    const y2i = cy + rInterno * Math.sin(anguloFin);
    const largeArc = fraccion > 0.5 ? 1 : 0;

    const path = [
      `M ${x1} ${y1}`,
      `A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`,
      `L ${x2i} ${y2i}`,
      `A ${rInterno} ${rInterno} 0 ${largeArc} 0 ${x1i} ${y1i}`,
      "Z",
    ].join(" ");

    svg += `<path d="${path}" fill="${color}" stroke="white" stroke-width="1.5"/>`;
    anguloInicio = anguloFin;
  });

  // Texto central.
  if (centroLabel) {
    svg += `<text x="${cx}" y="${cy + 5}" text-anchor="middle" font-size="14" font-weight="bold" fill="${centro()}" font-family="sans-serif">${centroLabel}</text>`;
  }

  svg += "</svg>";
  container.innerHTML = svg;
}

/**
 * Dibuja una barra de progreso simple (para confianza de una predicción).
 */
export function dibujarBarraConfianza(container, valor, color = "#3b82f6") {
  const pct = Math.round(valor * 100);
  const ancho = 200;
  const alto = 12;
  const w = valor * (ancho - 2);
  const svg = `<svg width="100%" height="${alto}" viewBox="0 0 ${ancho} ${alto}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="1" y="1" width="${ancho - 2}" height="${alto - 2}" fill="${pista()}" rx="5"/>
    <rect x="1" y="1" width="${Math.max(w, 1)}" height="${alto - 2}" fill="${color}" rx="5"/>
    <text x="${ancho / 2}" y="${alto - 2}" text-anchor="middle" font-size="9" fill="white" font-family="sans-serif" font-weight="bold">${pct}%</text>
  </svg>`;
  container.innerHTML = svg;
}

export { COLORES_CATEGORIA, COLORES_SENTIMIENTO };
