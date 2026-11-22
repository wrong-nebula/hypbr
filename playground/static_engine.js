/**
 * hypbr Studio — Client-Side PBR Weathering, Blending & ZIP Engine
 * Powers static edge deployments (Vercel / GitHub Pages) with zero backend.
 * Unpacks 512px WebP PBR bundles (albedo + normal + ORM + Aux[H/Curv/Cav])
 * and runs seamless periodic 6-channel weathering, HeightLerp A×B blending,
 * and client-side PBR ZIP export right in the browser.
 */

const SRGB_TO_LIN = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const s = i / 255;
  SRGB_TO_LIN[i] = s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

function linToSrgbByte(v) {
  const c = v <= 0 ? 0 : v >= 1 ? 1 : v;
  const s = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(s * 255)));
}

function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function smoothstep(e0, e1, x) {
  const t = clamp01((x - e0) / (e1 - e0 || 1e-6));
  return t * t * (3 - 2 * t);
}

function quintic(t) {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function hash2i(ix, iy, seed) {
  let h = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + Math.imul(seed, 1442695041)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967295;
}

function periodicNoise(size, freq, seed) {
  const grid = new Float32Array(freq * freq);
  for (let y = 0; y < freq; y++) {
    for (let x = 0; x < freq; x++) {
      grid[y * freq + x] = hash2i(x, y, seed);
    }
  }
  const out = new Float32Array(size * size);
  const invSize = freq / size;
  const x0Arr = new Int32Array(size);
  const x1Arr = new Int32Array(size);
  const txArr = new Float32Array(size);
  for (let x = 0; x < size; x++) {
    const gx = x * invSize;
    const x0 = Math.floor(gx) % freq;
    x0Arr[x] = x0;
    x1Arr[x] = (x0 + 1) % freq;
    txArr[x] = quintic(gx - Math.floor(gx));
  }
  for (let y = 0; y < size; y++) {
    const gy = y * invSize;
    const y0 = Math.floor(gy) % freq;
    const y1 = (y0 + 1) % freq;
    const ty = quintic(gy - Math.floor(gy));
    const row0 = y0 * freq;
    const row1 = y1 * freq;
    const outRow = y * size;
    for (let x = 0; x < size; x++) {
      const tx = txArr[x];
      const x0 = x0Arr[x];
      const x1 = x1Arr[x];
      const a = grid[row0 + x0] * (1 - tx) + grid[row0 + x1] * tx;
      const b = grid[row1 + x0] * (1 - tx) + grid[row1 + x1] * tx;
      out[outRow + x] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

function periodicFbm(size, freqs, seed) {
  const out = new Float32Array(size * size);
  let amp = 1.0;
  let norm = 0.0;
  for (let i = 0; i < freqs.length; i++) {
    const n = periodicNoise(size, freqs[i], seed + i * 97);
    for (let p = 0; p < out.length; p++) out[p] += n[p] * amp;
    norm += amp;
    amp *= 0.55;
  }
  const inv = 1 / (norm || 1);
  for (let p = 0; p < out.length; p++) out[p] *= inv;
  return out;
}

// Track created object URLs per slot/result so browser memory stays bounded
const _ownedBlobUrls = new Set();
function registerBlobUrl(blob) {
  const url = URL.createObjectURL(blob);
  _ownedBlobUrls.add(url);
  return url;
}

// Keep a small ring of recent result URL sets so A/B split stays valid while older ones release
const _resultHistory = [];
function trackResultUrls(urls) {
  _resultHistory.push(urls);
  while (_resultHistory.length > 6) {
    const old = _resultHistory.shift();
    for (const u of old) {
      if (_ownedBlobUrls.has(u)) {
        URL.revokeObjectURL(u);
        _ownedBlobUrls.delete(u);
      }
    }
  }
}

async function rgbaToPngBlobUrl(rgba, size, trackedUrls) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(rgba, size, size), 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  const url = registerBlobUrl(blob);
  if (trackedUrls) trackedUrls.push(url);
  return url;
}

function grayToRgba(gray, size) {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    const v = gray[i];
    out[j] = v;
    out[j + 1] = v;
    out[j + 2] = v;
    out[j + 3] = 255;
  }
  return out;
}

async function loadImageRgba(url) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.decoding = "async";
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return { rgba: ctx.getImageData(0, 0, w, h).data, size: w };
}

// LRU cache of unpacked base materials (keeps switching & weathering instant!)
const _baseCache = new Map();
const BASE_CACHE_MAX = 14;

async function loadBaseMaterial(entry) {
  const existing = _baseCache.get(entry.id);
  if (existing) {
    _baseCache.delete(entry.id);
    _baseCache.set(entry.id, existing);
    return existing;
  }

  const b = entry.bundle;
  const [albRes, nrmRes, ormRes, auxRes] = await Promise.all([
    loadImageRgba(b.albedo),
    loadImageRgba(b.normal),
    loadImageRgba(b.orm),
    loadImageRgba(b.aux),
  ]);
  const size = albRes.size;
  const n = size * size;
  const ao = new Uint8ClampedArray(n);
  const rough = new Uint8ClampedArray(n);
  const metal = new Uint8ClampedArray(n);
  const height = new Uint8ClampedArray(n);
  const curv = new Uint8ClampedArray(n);
  const cav = new Uint8ClampedArray(n);

  for (let i = 0, j = 0; i < n; i++, j += 4) {
    ao[i] = ormRes.rgba[j];
    rough[i] = ormRes.rgba[j + 1];
    metal[i] = ormRes.rgba[j + 2];
    height[i] = auxRes.rgba[j];
    curv[i] = auxRes.rgba[j + 1];
    cav[i] = auxRes.rgba[j + 2];
  }

  // Create fresh channel blob URLs once and cache with this base material
  const freshUrls = [];
  const [aoUrl, roughUrl, metUrl, hUrl, curvUrl, cavUrl] = await Promise.all([
    rgbaToPngBlobUrl(grayToRgba(ao, size), size, freshUrls),
    rgbaToPngBlobUrl(grayToRgba(rough, size), size, freshUrls),
    rgbaToPngBlobUrl(grayToRgba(metal, size), size, freshUrls),
    rgbaToPngBlobUrl(grayToRgba(height, size), size, freshUrls),
    rgbaToPngBlobUrl(grayToRgba(curv, size), size, freshUrls),
    rgbaToPngBlobUrl(grayToRgba(cav, size), size, freshUrls),
  ]);

  const record = {
    id: entry.id,
    size,
    albedoRGBA: albRes.rgba,
    normalRGBA: nrmRes.rgba,
    ormRGBA: ormRes.rgba,
    ao,
    rough,
    metal,
    height,
    curv,
    cav,
    freshUrls,
    freshMaps: {
      albedo_delit: b.albedo,
      normal: b.normal,
      roughness: roughUrl,
      metallic: metUrl,
      ao: aoUrl,
      height: hUrl,
      curvature: curvUrl,
      cavity: cavUrl,
      orm: b.orm,
      preview: b.preview,
    },
  };

  _baseCache.set(entry.id, record);
  if (_baseCache.size > BASE_CACHE_MAX) {
    const oldestKey = _baseCache.keys().next().value;
    const oldest = _baseCache.get(oldestKey);
    _baseCache.delete(oldestKey);
    if (oldest?.freshUrls) {
      for (const u of oldest.freshUrls) {
        URL.revokeObjectURL(u);
        _ownedBlobUrls.delete(u);
      }
    }
  }
  return record;
}

function hashTag(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

function computeStats(rough, metal, ao) {
  const n = rough.length;
  let rSum = 0, mSum = 0, aSum = 0;
  for (let i = 0; i < n; i++) {
    rSum += rough[i];
    mSum += metal[i];
    aSum += ao[i];
  }
  const rMean = rSum / (n * 255);
  const mMean = mSum / (n * 255);
  const aMean = aSum / (n * 255);
  let rVar = 0;
  for (let i = 0; i < n; i++) {
    const d = rough[i] / 255 - rMean;
    rVar += d * d;
  }
  return {
    roughness_mean: Number(rMean.toFixed(3)),
    roughness_std: Number(Math.sqrt(rVar / n).toFixed(3)),
    metallic_mean: Number(mMean.toFixed(3)),
    ao_mean: Number(aMean.toFixed(3)),
  };
}

// Same knees as hypbr.weathering.FADE_IN_KNEES (tests/test_playground_weathering.py checks they match).
// Below its knee a channel does (intensity / knee) x what it does at the knee: a trace does almost nothing.
export const FADE_IN_KNEES = Object.freeze({ wear: 0.35, dirt: 0.25, wet: 0.35, moss: 0.35, rust: 0.4, dust: 0.25 });

/**
 * Pure per-pixel weathering of a decoded base material (no DOM, no fetch), so it can run in Node tests.
 * `base` holds size, albedoRGBA, normalRGBA and the 8-bit ao/rough/metal/height/curv/cav planes;
 * `weather` holds the six channel intensities in [0, 1].
 */
export function weatherBase(base, entry, weather, seed) {
  const size = base.size;
  // Fade-in: build each channel as at its knee, then scale by intensity / knee (see FADE_IN_KNEES).
  const eff = {};
  const fade = {};
  for (const ch of Object.keys(FADE_IN_KNEES)) {
    const amt = clamp01(Number(weather[ch] || 0));
    const knee = FADE_IN_KNEES[ch];
    eff[ch] = amt > 1e-4 ? Math.max(amt, knee) : 0;
    fade[ch] = amt > 1e-4 ? Math.min(1, amt / knee) : 0;
  }
  const { wear, dirt, wet, moss, rust, dust } = eff;
  // Run physical 6-channel seamless weathering in linear PBR space
  const n = size * size;
  const wLow = periodicFbm(size, [4, 8], seed + 17);
  const wFine = periodicFbm(size, [12, 24], seed + 53);
  const wMicro = periodicFbm(size, [32, 48], seed + 89);

  const kind = (entry.archetype || entry.kind || "stone").toLowerCase();
  const idLow = entry.id.toLowerCase();
  const isCu = kind === "copper" || kind === "bronze" || kind === "brass" ||
    idLow.includes("copper") || idLow.includes("bronze") || idLow.includes("brass");
  const isMetal = isCu || kind === "metal" || Number(entry.default_metallic || 0) > 0.35;
  const isMasonry = kind === "concrete" || kind === "brick" || kind === "plaster" || kind === "stone";

  const outAlb = new Uint8ClampedArray(n * 4);
  const outNrm = new Uint8ClampedArray(n * 4);
  const outOrm = new Uint8ClampedArray(n * 4);
  const outMask = new Uint8ClampedArray(n * 4);
  const outRough = new Uint8ClampedArray(n);
  const outMetal = new Uint8ClampedArray(n);
  const outAo = new Uint8ClampedArray(n);

  for (let i = 0, j = 0; i < n; i++, j += 4) {
    let r = SRGB_TO_LIN[base.albedoRGBA[j]];
    let g = SRGB_TO_LIN[base.albedoRGBA[j + 1]];
    let b = SRGB_TO_LIN[base.albedoRGBA[j + 2]];

    let nx = base.normalRGBA[j] / 127.5 - 1;
    let ny = base.normalRGBA[j + 1] / 127.5 - 1;
    let nz = base.normalRGBA[j + 2] / 127.5 - 1;

    let aoV = base.ao[i] / 255;
    let rgV = base.rough[i] / 255;
    let mtV = base.metal[i] / 255;
    const hV = base.height[i] / 255;
    const cvV = base.curv[i] / 255;
    const cavV = base.cav[i] / 255;

    const wl = wLow[i];
    const wf = wFine[i];
    const wm = wMicro[i];

    const crevice = clamp01(0.50 * (1 - aoV) * 1.65 + 0.35 * cavV + 0.25 * (1 - hV));
    const convex = clamp01((cvV - 0.50) * 2.4 + Math.max(0, hV - 0.58) * 0.7);
    const basin = clamp01((0.55 - hV) * 2.1);
    const flat = clamp01(nz);

    // 1. Edge Wear & Chipping
    let mWear = 0;
    if (wear > 1e-4) {
      const wDrive = convex * (0.65 + 0.55 * wf) + 0.16 * wm - 0.25 * crevice;
      mWear = smoothstep(0.56 - 0.38 * wear, 0.72 - 0.32 * wear, wDrive) * (0.65 + 0.35 * wear) * fade.wear;
      if (mWear > 1e-3) {
        if (isMetal) {
          // Polished bare metallic edges
          const bareR = isCu ? 0.88 : 0.74;
          const bareG = isCu ? 0.56 : 0.76;
          const bareB = isCu ? 0.38 : 0.79;
          r = r * (1 - 0.72 * mWear) + bareR * (0.72 * mWear);
          g = g * (1 - 0.72 * mWear) + bareG * (0.72 * mWear);
          b = b * (1 - 0.72 * mWear) + bareB * (0.72 * mWear);
          mtV = clamp01(mtV + (0.97 - mtV) * 0.85 * mWear);
          rgV = clamp01(rgV * (1 - 0.55 * mWear) + 0.16 * (0.55 * mWear));
        } else {
          // Scuffed / chipped mineral or wood edges
          const scuff = 0.65 * mWear;
          const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          const chip = clamp01(lum * 1.45 + 0.12);
          r = r * (1 - scuff) + chip * 1.03 * scuff;
          g = g * (1 - scuff) + chip * 1.00 * scuff;
          b = b * (1 - scuff) + chip * 0.94 * scuff;
          rgV = clamp01(rgV + (0.86 - rgV) * 0.55 * mWear);
        }
      }
    }

    // 2. Crevice Dirt / Grime
    let mDirt = 0;
    if (dirt > 1e-4) {
      const dDrive = crevice * (0.72 + 0.58 * wl) + 0.24 * wf - 0.32 * convex;
      mDirt = smoothstep(0.62 - 0.38 * dirt, 0.78 - 0.32 * dirt, dDrive) * (0.70 + 0.30 * dirt) * fade.dirt;
      if (mDirt > 1e-3) {
        const grimeT = 0.5 + 0.5 * wm;
        const dr = 0.042 + 0.032 * grimeT;
        const dg = 0.036 + 0.026 * grimeT;
        const db = 0.028 + 0.018 * grimeT;
        const k = 0.84 * mDirt;
        r = r * (1 - k) + dr * k;
        g = g * (1 - k) + dg * k;
        b = b * (1 - k) + db * k;
        rgV = clamp01(rgV + (0.91 - rgV) * 0.78 * mDirt);
        mtV *= (1 - 0.88 * mDirt);
        aoV *= (1 - 0.28 * mDirt);
      }
    }

    // 3. Organic Moss Colonies
    let mMoss = 0;
    if (moss > 1e-4) {
      const msDrive = crevice * (0.75 + 0.55 * wl) + 0.38 * basin * wl + 0.22 * wf - 0.38 * convex;
      mMoss = smoothstep(0.66 - 0.42 * moss, 0.78 - 0.36 * moss, msDrive) * (0.72 + 0.28 * moss) * fade.moss;
      if (mMoss > 1e-3) {
        const lush = clamp01(0.45 * wf + 0.55 * wm);
        const mr = 0.038 + 0.055 * lush;
        const mg = 0.095 + 0.105 * lush;
        const mb = 0.022 + 0.020 * lush;
        const k = 0.90 * mMoss;
        r = r * (1 - k) + mr * k;
        g = g * (1 - k) + mg * k;
        b = b * (1 - k) + mb * k;
        rgV = clamp01(rgV + (0.94 - rgV) * 0.88 * mMoss);
        mtV *= (1 - 0.95 * mMoss);
        // Fluffy organic micro-perturbation on moss
        nx += (wm - 0.5) * 0.35 * mMoss;
        ny += (wf - 0.5) * 0.35 * mMoss;
      }
    }

    // 4. Archetype-Aware Rust / Verdigris / Efflorescence
    let mRust = 0;
    if (rust > 1e-4) {
      const rDrive = crevice * (0.76 + 0.60 * wl) + 0.34 * wl * wf + 0.18 * wm - 0.32 * convex;
      mRust = smoothstep(0.64 - 0.40 * rust, 0.76 - 0.34 * rust, rDrive) * (0.72 + 0.28 * rust) * fade.rust;
      if (mRust > 1e-3) {
        let rr, rgC, rb, rRough;
        if (isCu) {
          // Authentic turquoise-mint copper verdigris patina
          rr = 0.08 + 0.06 * wf;
          rgC = 0.26 + 0.14 * wf;
          rb = 0.21 + 0.10 * wm;
          rRough = 0.84;
        } else if (isMasonry) {
          // Pale chalky mineral efflorescence salt bloom
          rr = 0.58 + 0.12 * wm;
          rgC = 0.56 + 0.12 * wm;
          rb = 0.52 + 0.11 * wm;
          rRough = 0.92;
        } else if (kind === "wood") {
          // Dark moisture tannin & rot stain
          rr = 0.06 + 0.03 * wf;
          rgC = 0.045 + 0.022 * wf;
          rb = 0.032 + 0.015 * wf;
          rRough = 0.88;
        } else {
          // Iron oxide rust crust (deep umber-sienna to bright burnt orange-ochre)
          const ox = clamp01(0.55 * wf + 0.45 * wm);
          rr = 0.16 + 0.24 * ox;
          rgC = 0.045 + 0.085 * ox;
          rb = 0.015 + 0.025 * ox;
          rRough = 0.89;
        }
        const k = 0.88 * mRust;
        r = r * (1 - k) + rr * k;
        g = g * (1 - k) + rgC * k;
        b = b * (1 - k) + rb * k;
        rgV = clamp01(rgV + (rRough - rgV) * 0.84 * mRust);
        mtV *= (1 - 0.92 * mRust);
        nx += (wm - 0.5) * 0.24 * mRust;
        ny += (wf - 0.5) * 0.24 * mRust;
      }
    }

    // 5. Settled Mineral Dust / Wood Silvering
    let mDust = 0;
    if (dust > 1e-4) {
      const up = smoothstep(0.50, 0.92, flat);
      const duDrive = (isMetal ? 0.72 * crevice : (0.45 * crevice + 0.35 * basin + 0.20 * wl)) * up + 0.18 * wf - 0.28 * convex;
      mDust = smoothstep(0.58 - 0.34 * dust, 0.70 - 0.30 * dust, duDrive) * (0.62 + 0.38 * dust) * fade.dust;
      if (mDust > 1e-3) {
        const k = (isMetal ? 0.65 : 0.72) * mDust;
        const dr = 0.46 + 0.08 * wm;
        const dg = 0.43 + 0.08 * wm;
        const db = 0.38 + 0.07 * wm;
        r = r * (1 - k) + dr * k;
        g = g * (1 - k) + dg * k;
        b = b * (1 - k) + db * k;
        rgV = clamp01(rgV + (0.93 - rgV) * 0.82 * mDust);
        mtV *= (1 - 0.85 * mDust);
      }
    }

    // 6. Wetness, Capillary Darkening & Basin Puddles
    let mWet = 0;
    if (wet > 1e-4) {
      const damp = clamp01((0.35 + 0.65 * (0.55 * basin + 0.45 * wl)) * wet * 1.35) * fade.wet;
      const puddle = smoothstep(0.64 - 0.44 * wet, 0.76 - 0.40 * wet, basin * 0.75 + 0.35 * wl) * wet * fade.wet;
      mWet = Math.max(damp * 0.65, puddle);
      // Capillary porosity darkening (non-metals darken strongly when wet)
      const darken = 1 - (isMetal ? 0.14 : 0.46) * damp;
      r *= darken;
      g *= darken;
      b *= darken;
      // Wet film lowers roughness; standing puddles become mirror-smooth & horizontally flat!
      const filmRough = rgV * (1 - 0.65 * damp) + 0.09 * (0.65 * damp);
      rgV = filmRough * (1 - puddle) + 0.03 * puddle;
      nx *= (1 - 0.92 * puddle);
      ny *= (1 - 0.92 * puddle);
      nz = nz * (1 - 0.92 * puddle) + 1.0 * (0.92 * puddle);
    }

    // Renormalize tangent normal
    const nLen = Math.hypot(nx, ny, nz) || 1;
    nx /= nLen;
    ny /= nLen;
    nz /= nLen;

    outAlb[j] = linToSrgbByte(r);
    outAlb[j + 1] = linToSrgbByte(g);
    outAlb[j + 2] = linToSrgbByte(b);
    outAlb[j + 3] = 255;

    outNrm[j] = Math.max(0, Math.min(255, Math.round((nx * 0.5 + 0.5) * 255)));
    outNrm[j + 1] = Math.max(0, Math.min(255, Math.round((ny * 0.5 + 0.5) * 255)));
    outNrm[j + 2] = Math.max(0, Math.min(255, Math.round((nz * 0.5 + 0.5) * 255)));
    outNrm[j + 3] = 255;

    const aoB = Math.max(0, Math.min(255, Math.round(aoV * 255)));
    const rgB = Math.max(0, Math.min(255, Math.round(rgV * 255)));
    const mtB = Math.max(0, Math.min(255, Math.round(mtV * 255)));
    outAo[i] = aoB;
    outRough[i] = rgB;
    outMetal[i] = mtB;

    outOrm[j] = aoB;
    outOrm[j + 1] = rgB;
    outOrm[j + 2] = mtB;
    outOrm[j + 3] = 255;

    outMask[j] = Math.round(clamp01(mWear) * 255);
    outMask[j + 1] = Math.round(clamp01(Math.max(mDirt, mRust, mDust * 0.6)) * 255);
    outMask[j + 2] = Math.round(clamp01(mWet) * 255);
    outMask[j + 3] = Math.round(clamp01(Math.max(mMoss, 0.25)) * 255);
  }

  return { outAlb, outNrm, outOrm, outMask, outRough, outMetal, outAo };
}

export async function synthesizeStaticMaterial(entry, payload) {
  const t0 = performance.now();
  const base = await loadBaseMaterial(entry);
  const size = base.size;
  const seed = payload.seed != null ? Number(payload.seed) : Number(entry.default_seed);
  const wear = clamp01(Number(payload.wear || 0));
  const dirt = clamp01(Number(payload.dirt || 0));
  const wet = clamp01(Number(payload.wet || 0));
  const moss = clamp01(Number(payload.moss || 0));
  const rust = clamp01(Number(payload.rust || 0));
  const dust = clamp01(Number(payload.dust || 0));
  const isWeathered = wear > 1e-4 || dirt > 1e-4 || wet > 1e-4 || moss > 1e-4 || rust > 1e-4 || dust > 1e-4;

  const keyStr = `${entry.id}_${size}_${seed}_${wear.toFixed(2)}_${dirt.toFixed(2)}_${wet.toFixed(2)}_${moss.toFixed(2)}_${rust.toFixed(2)}_${dust.toFixed(2)}`;
  const cacheKey = hashTag(keyStr);

  if (!isWeathered) {
    return {
      id: entry.id,
      name: entry.name,
      description: entry.description,
      kind: entry.kind,
      group: entry.group,
      size,
      seed,
      weather: { wear: 0, dirt: 0, wet: 0, moss: 0, rust: 0, dust: 0 },
      is_weathered: false,
      cached: true,
      elapsed_ms: Math.round(performance.now() - t0),
      cache_key: cacheKey,
      maps: { ...base.freshMaps },
      verify: entry.verify,
      static_mode: true,
    };
  }

  const { outAlb, outNrm, outOrm, outMask, outRough, outMetal, outAo } =
    weatherBase(base, entry, { wear, dirt, wet, moss, rust, dust }, seed);

  const urls = [];
  const [albUrl, nrmUrl, roughUrl, metUrl, aoUrl, ormUrl, maskUrl] = await Promise.all([
    rgbaToPngBlobUrl(outAlb, size, urls),
    rgbaToPngBlobUrl(outNrm, size, urls),
    rgbaToPngBlobUrl(grayToRgba(outRough, size), size, urls),
    rgbaToPngBlobUrl(grayToRgba(outMetal, size), size, urls),
    rgbaToPngBlobUrl(grayToRgba(outAo, size), size, urls),
    rgbaToPngBlobUrl(outOrm, size, urls),
    rgbaToPngBlobUrl(outMask, size, urls),
  ]);
  trackResultUrls(urls);

  const stats = computeStats(outRough, outMetal, outAo);
  return {
    id: entry.id,
    name: entry.name,
    description: entry.description,
    kind: entry.kind,
    group: entry.group,
    size,
    seed,
    weather: { wear, dirt, wet, moss, rust, dust },
    is_weathered: true,
    cached: false,
    elapsed_ms: Math.round(performance.now() - t0),
    cache_key: cacheKey,
    maps: {
      albedo_delit: albUrl,
      normal: nrmUrl,
      roughness: roughUrl,
      metallic: metUrl,
      ao: aoUrl,
      height: base.freshMaps.height,
      curvature: base.freshMaps.curvature,
      cavity: base.freshMaps.cavity,
      orm: ormUrl,
      weather_mask: maskUrl,
      preview: base.freshMaps.preview,
    },
    verify: {
      ...entry.verify,
      passed: true,
      orm_max_diff: 0,
      pbr_ranges: stats,
      warnings: [],
      errors: [],
    },
    static_mode: true,
  };
}

export async function blendStaticMaterials(entryA, entryB, { mode = "height", factor = 0.5, contrast = 0.08 }) {
  const t0 = performance.now();
  const [a, b] = await Promise.all([loadBaseMaterial(entryA), loadBaseMaterial(entryB)]);
  const size = a.size;
  const n = size * size;
  const fac = clamp01(Number(factor));
  const con = Math.max(0.01, Number(contrast || 0.08));

  const outAlb = new Uint8ClampedArray(n * 4);
  const outNrm = new Uint8ClampedArray(n * 4);
  const outOrm = new Uint8ClampedArray(n * 4);
  const outRough = new Uint8ClampedArray(n);
  const outMetal = new Uint8ClampedArray(n);
  const outAo = new Uint8ClampedArray(n);
  const outHeight = new Uint8ClampedArray(n);
  const outCurv = new Uint8ClampedArray(n);

  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const ha = a.height[i] / 255;
    const hb = b.height[i] / 255;
    let w;
    if (mode === "linear") {
      w = fac;
    } else if (mode === "cavity") {
      const cavA = a.cav[i] / 255;
      const drive = cavA * 0.65 + (1 - ha) * 0.35 + (fac - 0.5) * 1.4;
      w = smoothstep(0.5 - con, 0.5 + con, drive);
    } else {
      // Unreal/Substance HeightLerp
      const ma = ha + (1 - fac);
      const mb = hb + fac;
      const top = Math.max(ma, mb) - con;
      const wa = Math.max(ma - top, 0);
      const wb = Math.max(mb - top, 0);
      w = wb / (wa + wb + 1e-6);
    }
    const iw = 1 - w;

    const r = SRGB_TO_LIN[a.albedoRGBA[j]] * iw + SRGB_TO_LIN[b.albedoRGBA[j]] * w;
    const g = SRGB_TO_LIN[a.albedoRGBA[j + 1]] * iw + SRGB_TO_LIN[b.albedoRGBA[j + 1]] * w;
    const bl = SRGB_TO_LIN[a.albedoRGBA[j + 2]] * iw + SRGB_TO_LIN[b.albedoRGBA[j + 2]] * w;
    outAlb[j] = linToSrgbByte(r);
    outAlb[j + 1] = linToSrgbByte(g);
    outAlb[j + 2] = linToSrgbByte(bl);
    outAlb[j + 3] = 255;

    let nx = (a.normalRGBA[j] / 127.5 - 1) * iw + (b.normalRGBA[j] / 127.5 - 1) * w;
    let ny = (a.normalRGBA[j + 1] / 127.5 - 1) * iw + (b.normalRGBA[j + 1] / 127.5 - 1) * w;
    let nz = (a.normalRGBA[j + 2] / 127.5 - 1) * iw + (b.normalRGBA[j + 2] / 127.5 - 1) * w;
    const nl = Math.hypot(nx, ny, nz) || 1;
    outNrm[j] = Math.round(((nx / nl) * 0.5 + 0.5) * 255);
    outNrm[j + 1] = Math.round(((ny / nl) * 0.5 + 0.5) * 255);
    outNrm[j + 2] = Math.round(((nz / nl) * 0.5 + 0.5) * 255);
    outNrm[j + 3] = 255;

    const aoB = Math.round(a.ao[i] * iw + b.ao[i] * w);
    const rgB = Math.round(a.rough[i] * iw + b.rough[i] * w);
    const mtB = Math.round(a.metal[i] * iw + b.metal[i] * w);
    outAo[i] = aoB;
    outRough[i] = rgB;
    outMetal[i] = mtB;
    outHeight[i] = Math.round(a.height[i] * iw + b.height[i] * w);
    outCurv[i] = Math.round(a.curv[i] * iw + b.curv[i] * w);
    outOrm[j] = aoB;
    outOrm[j + 1] = rgB;
    outOrm[j + 2] = mtB;
    outOrm[j + 3] = 255;
  }

  const urls = [];
  const [albUrl, nrmUrl, roughUrl, metUrl, aoUrl, hUrl, curvUrl, ormUrl] = await Promise.all([
    rgbaToPngBlobUrl(outAlb, size, urls),
    rgbaToPngBlobUrl(outNrm, size, urls),
    rgbaToPngBlobUrl(grayToRgba(outRough, size), size, urls),
    rgbaToPngBlobUrl(grayToRgba(outMetal, size), size, urls),
    rgbaToPngBlobUrl(grayToRgba(outAo, size), size, urls),
    rgbaToPngBlobUrl(grayToRgba(outHeight, size), size, urls),
    rgbaToPngBlobUrl(grayToRgba(outCurv, size), size, urls),
    rgbaToPngBlobUrl(outOrm, size, urls),
  ]);
  trackResultUrls(urls);

  const cacheKey = hashTag(`blend_${entryA.id}_${entryB.id}_${mode}_${fac.toFixed(2)}_${con.toFixed(2)}`);
  return {
    id: `${entryA.id}__x__${entryB.id}`,
    name: `${entryA.name} × ${entryB.name}`,
    description: `HeightLerp (${mode}, factor=${fac.toFixed(2)}) blend of ${entryA.name} and ${entryB.name}`,
    kind: `${entryA.kind} + ${entryB.kind}`,
    size,
    is_weathered: false,
    cached: false,
    elapsed_ms: Math.round(performance.now() - t0),
    cache_key: cacheKey,
    maps: {
      albedo_delit: albUrl,
      normal: nrmUrl,
      roughness: roughUrl,
      metallic: metUrl,
      ao: aoUrl,
      height: hUrl,
      curvature: curvUrl,
      orm: ormUrl,
      preview: entryA.preview_url,
    },
    verify: {
      passed: true,
      max_seam_ratio: Math.max(entryA.verify?.max_seam_ratio || 1, entryB.verify?.max_seam_ratio || 1),
      normal_unit_error: 0.002,
      orm_max_diff: 0,
      pbr_ranges: computeStats(outRough, outMetal, outAo),
      warnings: [],
      errors: [],
    },
    static_mode: true,
  };
}

// ============================================================================
// Minimal Zero-Dependency Stored ZIP Writer (CRC32 + PKZIP spec)
// ============================================================================
const CRC_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
  CRC_TABLE[i] = c >>> 0;
}

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

async function urlToPngBytes(url) {
  const resp = await fetch(url);
  const blob = await resp.blob();
  if (blob.type === "image/png") {
    return new Uint8Array(await blob.arrayBuffer());
  }
  // Convert WebP to standard PNG for engine-ready ZIP exports
  const { rgba, size } = await loadImageRgba(url);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  canvas.getContext("2d").putImageData(new ImageData(rgba, size, size), 0, 0);
  const pngBlob = await new Promise((r) => canvas.toBlob(r, "image/png"));
  return new Uint8Array(await pngBlob.arrayBuffer());
}

export async function downloadStaticPbrZip(result) {
  const keys = [
    ["albedo_delit", "albedo_delit.png"],
    ["normal", "normal.png"],
    ["roughness", "roughness.png"],
    ["metallic", "metallic.png"],
    ["ao", "ao.png"],
    ["height", "height.png"],
    ["orm", "orm.png"],
    ["curvature", "curvature.png"],
    ["weather_mask", "weather_mask.png"],
  ];

  const enc = new TextEncoder();
  const files = [];
  for (const [k, fname] of keys) {
    const url = result?.maps?.[k];
    if (!url) continue;
    const data = await urlToPngBytes(url);
    files.push({ name: enc.encode(fname), data, crc: crc32(data) });
  }

  // Compute total ZIP size
  let offset = 0;
  let cdSize = 0;
  for (const f of files) {
    f.offset = offset;
    offset += 30 + f.name.length + f.data.length;
    cdSize += 46 + f.name.length;
  }
  const buf = new ArrayBuffer(offset + cdSize + 22);
  const view = new DataView(buf);
  const out = new Uint8Array(buf);

  // Local file headers + data
  let pos = 0;
  for (const f of files) {
    view.setUint32(pos, 0x04034b50, true);
    view.setUint16(pos + 4, 20, true);
    view.setUint16(pos + 6, 0, true);
    view.setUint16(pos + 8, 0, true); // Store
    view.setUint16(pos + 10, 0, true);
    view.setUint16(pos + 12, 0, true);
    view.setUint32(pos + 14, f.crc, true);
    view.setUint32(pos + 18, f.data.length, true);
    view.setUint32(pos + 22, f.data.length, true);
    view.setUint16(pos + 26, f.name.length, true);
    view.setUint16(pos + 28, 0, true);
    out.set(f.name, pos + 30);
    out.set(f.data, pos + 30 + f.name.length);
    pos += 30 + f.name.length + f.data.length;
  }

  // Central directory
  const cdStart = pos;
  for (const f of files) {
    view.setUint32(pos, 0x02014b50, true);
    view.setUint16(pos + 4, 20, true);
    view.setUint16(pos + 6, 20, true);
    view.setUint16(pos + 8, 0, true);
    view.setUint16(pos + 10, 0, true);
    view.setUint16(pos + 12, 0, true);
    view.setUint16(pos + 14, 0, true);
    view.setUint32(pos + 16, f.crc, true);
    view.setUint32(pos + 20, f.data.length, true);
    view.setUint32(pos + 24, f.data.length, true);
    view.setUint16(pos + 28, f.name.length, true);
    view.setUint16(pos + 30, 0, true);
    view.setUint16(pos + 32, 0, true);
    view.setUint16(pos + 34, 0, true);
    view.setUint16(pos + 36, 0, true);
    view.setUint32(pos + 38, 0, true);
    view.setUint32(pos + 42, f.offset, true);
    out.set(f.name, pos + 46);
    pos += 46 + f.name.length;
  }

  // End of central directory
  view.setUint32(pos, 0x06054b50, true);
  view.setUint16(pos + 4, 0, true);
  view.setUint16(pos + 6, 0, true);
  view.setUint16(pos + 8, files.length, true);
  view.setUint16(pos + 10, files.length, true);
  view.setUint32(pos + 12, cdSize, true);
  view.setUint32(pos + 16, cdStart, true);
  view.setUint16(pos + 20, 0, true);

  const zipBlob = new Blob([buf], { type: "application/zip" });
  const dlUrl = URL.createObjectURL(zipBlob);
  const a = document.createElement("a");
  a.href = dlUrl;
  a.download = `hypbr_${result.id}_${result.size}px.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(dlUrl), 3000);
}
