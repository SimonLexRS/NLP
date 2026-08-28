#!/usr/bin/env node
/**
 * Renderiza web/assets/og-image.svg → web/assets/og-image.png (1200x630).
 *
 * Uso local (desde la raíz del repo):
 *   npm install --no-save --no-package-lock @resvg/resvg-js
 *   node scripts/render-og-image.mjs
 *
 * El deploy (.github/workflows/deploy-pages.yml) ejecuta esto antes de subir el
 * artefacto de GitHub Pages, de modo que el PNG exista en producción sin
 * versionar binarios que nadie puede regenerar a mano.
 *
 * Tipografías: si están instalados los paquetes @expo-google-fonts/* (TTF) se
 * usan IBM Plex Sans/Mono y Source Serif 4; si no, resvg cae a las fuentes del
 * sistema. El diseño del SVG está pensado para verse bien con ambos métricas.
 */
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const svgPath = path.join(repoRoot, 'web', 'assets', 'og-image.svg');
const pngPath = path.join(repoRoot, 'web', 'assets', 'og-image.png');

let Resvg;
try {
  ({ Resvg } = await import('@resvg/resvg-js'));
} catch {
  console.error('Falta @resvg/resvg-js: npm install --no-save --no-package-lock @resvg/resvg-js');
  process.exit(1);
}

// Recolecta los .ttf de las fuentes de marca si el paquete está instalado.
async function collectFonts(dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await collectFonts(full, out);
    else if (entry.name.toLowerCase().endsWith('.ttf')) out.push(full);
  }
  return out;
}

const fontFiles = await collectFonts(path.join(repoRoot, 'node_modules', '@expo-google-fonts'));
if (fontFiles.length) {
  console.log(`Tipografías de marca: ${fontFiles.length} archivos TTF`);
} else {
  console.log('Sin TTFs de marca: se usan las fuentes del sistema (fallback del SVG).');
}

const svg = await readFile(svgPath, 'utf8');
const resvg = new Resvg(svg, {
  fitTo: { mode: 'original' },
  font: {
    loadSystemFonts: true,
    fontFiles,
    defaultFontFamily: 'DejaVu Sans',
  },
  background: '#0b1220',
});

const rendered = resvg.render();
const png = rendered.asPng();
await writeFile(pngPath, png);

const kb = (png.length / 1024).toFixed(0);
console.log(`OK ${path.relative(repoRoot, pngPath)} → ${rendered.width}x${rendered.height}, ${kb} KB`);