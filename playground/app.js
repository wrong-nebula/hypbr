import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { Pass } from "three/addons/postprocessing/Pass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { SMAAPass } from "three/addons/postprocessing/SMAAPass.js";
import {
  synthesizeStaticMaterial,
  blendStaticMaterials,
  downloadStaticPbrZip,
} from "./static_engine.js";

// ============================================================================
// Constants & Channel Metadata
// ============================================================================
const WEATHER_KEYS = ["wear", "dirt", "wet", "moss", "rust", "dust"];

const DIAGNOSTIC_MAPS = [
  { key: "preview", label: "Studio Preview", space: "sRGB" },
  { key: "albedo_delit", label: "Albedo (De-lit)", space: "sRGB" },
  { key: "normal", label: "Normal (OpenGL +Y)", space: "Linear" },
  { key: "roughness", label: "Roughness", space: "Linear" },
  { key: "metallic", label: "Metallic", space: "Linear" },
  { key: "ao", label: "8-Azimuth HBAO", space: "Linear" },
  { key: "height", label: "Height (8-bit)", space: "Linear" },
  { key: "curvature", label: "Curvature", space: "Linear" },
  { key: "orm", label: "glTF ORM", space: "R:AO G:R B:M" },
  { key: "bent_normal", label: "Bent Normal", space: "Linear" },
  { key: "cavity", label: "Cavity", space: "Linear" },
  { key: "weather_mask", label: "Weather Mask", space: "RGBA8" },
];

const LIGHT_RIGS = {
  studio: {
    envIntensity: 1.05,
    keyColor: 0xfff6eb,
    keyIntensity: 2.4,
    keyElevDeg: 36,
    keyRadius: 4.8,
    fillColor: 0x9ec5ff,
    fillIntensity: 0.78,
    fillPos: [-3.8, 1.5, 2.2],
    rimColor: 0xdbeafe,
    rimIntensity: 1.1,
    rimPos: [-1.8, 2.8, -3.8],
    hemiSky: 0xd8e2f0,
    hemiGround: 0x161a22,
    hemiIntensity: 0.26,
    bgTop: 0x11151d,
    bgHorizon: 0x1b212d,
    bgBottom: 0x0a0c11,
    shadowOpacity: 0.34,
  },
  raking: {
    envIntensity: 0.48,
    keyColor: 0xfff0d8,
    keyIntensity: 3.55,
    keyElevDeg: 14,
    keyRadius: 5.2,
    fillColor: 0x6f93c9,
    fillIntensity: 0.26,
    fillPos: [-3.5, 1.2, 2.5],
    rimColor: 0xfde68a,
    rimIntensity: 0.95,
    rimPos: [-2.2, 1.5, -4.0],
    hemiSky: 0x7c8ea8,
    hemiGround: 0x0b0d12,
    hemiIntensity: 0.14,
    bgTop: 0x0a0c11,
    bgHorizon: 0x141822,
    bgBottom: 0x06070a,
    shadowOpacity: 0.46,
  },
  golden: {
    envIntensity: 0.82,
    keyColor: 0xffb76b,
    keyIntensity: 3.2,
    keyElevDeg: 18,
    keyRadius: 5.0,
    fillColor: 0x5f88d6,
    fillIntensity: 0.92,
    fillPos: [-4.0, 2.6, 1.6],
    rimColor: 0xffcf99,
    rimIntensity: 1.25,
    rimPos: [-2.0, 1.8, -4.0],
    hemiSky: 0x7da2db,
    hemiGround: 0x2a1d14,
    hemiIntensity: 0.34,
    bgTop: 0x110e0d,
    bgHorizon: 0x211914,
    bgBottom: 0x090707,
    shadowOpacity: 0.42,
  },
  overcast: {
    envIntensity: 1.48,
    keyColor: 0xf4f7fb,
    keyIntensity: 1.15,
    keyElevDeg: 58,
    keyRadius: 4.5,
    fillColor: 0xe2e8f0,
    fillIntensity: 0.9,
    fillPos: [-3.0, 3.2, 2.5],
    rimColor: 0xdbe4f0,
    rimIntensity: 0.52,
    rimPos: [0.0, 3.5, -3.5],
    hemiSky: 0xe6edf6,
    hemiGround: 0x2f3540,
    hemiIntensity: 0.55,
    bgTop: 0x171c26,
    bgHorizon: 0x232936,
    bgBottom: 0x0f1218,
    shadowOpacity: 0.24,
  },
  darkrim: {
    envIntensity: 0.34,
    keyColor: 0xfff8ee,
    keyIntensity: 2.35,
    keyElevDeg: 22,
    keyRadius: 4.9,
    fillColor: 0x3b4f70,
    fillIntensity: 0.24,
    fillPos: [-3.5, -0.5, 2.2],
    rimColor: 0xf5d08a,
    rimIntensity: 2.75,
    rimPos: [-1.6, 2.4, -3.6],
    hemiSky: 0x3d485c,
    hemiGround: 0x06070a,
    hemiIntensity: 0.09,
    bgTop: 0x06070a,
    bgHorizon: 0x0f1219,
    bgBottom: 0x040507,
    shadowOpacity: 0.50,
  },
};

const GEOM_DEFAULT_REPEAT = {
  sphere: 4.0,
  knot: 4.0,
  cylinder: 3.5,
  box: 2.0,
  plane: 2.0,
  wall2x2: 1.0,
};

// Best automatic preview scene (geometry, UV repeat, lighting rig) per catalog material.
// Defaults to Raking Grazing (best micro-relief readability) and 4.0x UV repeat on Shaderball.
const MATERIAL_STAGE_MAP = {
  // Shaderball (sphere) — organic, woven, sculpted, hammered, pebbled & specular curved surfaces
  hammered_copper_dimple:     { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  carbon_fiber_weave:         { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  cast_iron_skillet_pebble:   { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  dimpled_golf_grip:          { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  patinated_verdigris_bronze: { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  damascus_folded_steel:      { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  woven_denim_twill:          { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  ribbed_corduroy_velvet:     { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  woven_nylon_strap:          { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  perforated_leather_auto:    { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  terry_velcro_loops:         { geom: "sphere",   repeat: 4.5, lightRig: "raking" },
  cable_knit_wool:            { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  herringbone_tweed_wool:     { geom: "sphere",   repeat: 4.5, lightRig: "raking" },
  waffle_cotton_linen:        { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  burlap_jute_sackcloth:      { geom: "sphere",   repeat: 4.0, lightRig: "raking" },
  quilted_leather_stitch:     { geom: "sphere",   repeat: 3.0, lightRig: "raking" },
  tufted_emerald_velvet:      { geom: "sphere",   repeat: 3.0, lightRig: "raking" },

  // Capped Cylinder — cylindrical barrels, knurled grips, bezels & vertical fluted glass
  knurled_metal_grip:         { geom: "cylinder", repeat: 3.0, lightRig: "raking" },
  knurled_straight_flutes:    { geom: "cylinder", repeat: 3.5, lightRig: "raking" },
  reeded_coin_milling:        { geom: "cylinder", repeat: 4.0, lightRig: "raking" },
  micro_hex_grip:             { geom: "cylinder", repeat: 3.5, lightRig: "raking" },
  fluted_reeded_glass:        { geom: "cylinder", repeat: 3.0, lightRig: "studio" },

  // Rounded Studio Cube — structural metal housings, meshes, fins & architectural stone blocks
  brushed_anodized_plate:     { geom: "box",      repeat: 2.0, lightRig: "raking" },
  heatsink_extrusion_fins:    { geom: "box",      repeat: 2.0, lightRig: "raking" },
  waffle_radiator_fins:       { geom: "box",      repeat: 2.0, lightRig: "raking" },
  speaker_mesh_hex:           { geom: "box",      repeat: 2.5, lightRig: "raking" },
  expanded_metal_mesh:        { geom: "box",      repeat: 2.0, lightRig: "raking" },
  chainlink_wire_mesh:        { geom: "box",      repeat: 2.5, lightRig: "raking" },
  circular_perf_sheet:        { geom: "box",      repeat: 2.5, lightRig: "raking" },
  terrazzo_speckled_stone:    { geom: "box",      repeat: 2.0, lightRig: "studio" },
  carrara_veined_marble:      { geom: "box",      repeat: 1.5, lightRig: "studio" },
  acoustic_kerf_felt:         { geom: "box",      repeat: 2.0, lightRig: "raking" },

  // 2x2 Seamless Wall — repeating architectural cladding, mosaics, plaster & porous stone
  penny_round_mosaic:         { geom: "wall2x2",  repeat: 1.0, lightRig: "raking" },
  hex_mosaic_tile:            { geom: "wall2x2",  repeat: 1.0, lightRig: "raking" },
  mermaid_fishscale_tile:     { geom: "wall2x2",  repeat: 1.0, lightRig: "raking" },
  corrugated_sheet_metal:     { geom: "wall2x2",  repeat: 1.0, lightRig: "raking" },
  riveted_sheet_seam:         { geom: "wall2x2",  repeat: 1.0, lightRig: "raking" },
  corrugated_cardboard:       { geom: "wall2x2",  repeat: 1.0, lightRig: "raking" },
  woven_rattan_cane:          { geom: "wall2x2",  repeat: 1.5, lightRig: "raking" },
  venetian_plaster_trowel:    { geom: "wall2x2",  repeat: 1.0, lightRig: "raking" },
  travertine_porous_stone:    { geom: "wall2x2",  repeat: 1.0, lightRig: "raking" },

  // Flat Displaced Tile — architectural floors, panels, grates, vents, treads & electronics
  hardops_plate:              { geom: "plane",    repeat: 1.0, lightRig: "raking" },
  wood_parquet_herringbone:   { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  subway_metro_tile:          { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  sidewalk_expansion_joint:   { geom: "plane",    repeat: 1.5, lightRig: "raking" },
  street_storm_grate:         { geom: "plane",    repeat: 1.5, lightRig: "raking" },
  diamond_tread_plate:        { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  hvac_slotted_vent:          { geom: "plane",    repeat: 1.5, lightRig: "raking" },
  stamped_louver_micro:       { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  stamped_vent_slots_stagger: { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  drip_tray_sieve_grate:      { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  solar_cell_grid:            { geom: "plane",    repeat: 1.5, lightRig: "raking" },
  pcb_solder_mask_traces:     { geom: "plane",    repeat: 1.5, lightRig: "raking" },
  keyboard_keycap_grid:       { geom: "plane",    repeat: 1.5, lightRig: "raking" },
  modular_stud_grid:          { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  waffle_silicone_pad:        { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  acoustic_pegboard:          { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  acoustic_slat_wood:         { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  acoustic_pyramid_foam:      { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  window_blinds_slats:        { geom: "plane",    repeat: 1.5, lightRig: "raking" },
  gilded_tin_ceiling_tile:    { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  embossed_braille_strip:     { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  stair_nosing_grooves:       { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  ribbed_rubber_mat:          { geom: "plane",    repeat: 2.5, lightRig: "raking" },
  rubber_tread_stair:         { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  frosted_prism_diffuser:     { geom: "plane",    repeat: 2.0, lightRig: "studio" },
  corkboard_pin_surface:      { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  woven_bamboo_tatami:        { geom: "plane",    repeat: 2.0, lightRig: "raking" },
  fine_asphalt_tarmac:        { geom: "plane",    repeat: 2.0, lightRig: "raking" },
};

function getRecommendedStage(entry) {
  if (!entry) return { geom: "sphere", repeat: 4.0, lightRig: "raking" };
  if (MATERIAL_STAGE_MAP[entry.id]) return MATERIAL_STAGE_MAP[entry.id];
  if (entry.stage && entry.stage.geom) return entry.stage;

  const kind = entry.kind || "";
  if (["fabric", "leather", "copper", "bronze", "carbon"].includes(kind)) {
    return { geom: "sphere", repeat: 4.0, lightRig: "raking" };
  }
  if (["glass"].includes(kind)) {
    return { geom: "plane", repeat: 2.0, lightRig: "studio" };
  }
  if (["concrete", "asphalt", "ceramic", "wood", "brass", "rubber"].includes(kind)) {
    return { geom: "plane", repeat: 2.0, lightRig: "raking" };
  }
  if (["stone", "plaster", "paper", "natural"].includes(kind)) {
    return { geom: "wall2x2", repeat: 1.0, lightRig: "raking" };
  }
  return { geom: "sphere", repeat: 4.0, lightRig: "raking" };
}

// ============================================================================
// Application State
// ============================================================================
const state = {
  catalog: [],
  catalogById: new Map(),
  filtered: [],
  groups: ["All"],
  groupCounts: {},
  weatherPresets: {},
  specCache: new Map(),
  atlasSpec: null,
  activeGroup: "All",
  searchQuery: "",

  selectedId: null,
  staticMode: false,
  size: 512,
  seed: null,
  activePreset: "fresh",
  weather: { wear: 0, dirt: 0, wet: 0, moss: 0, rust: 0, dust: 0 },
  autoSynth: true,
  autoStage: true,
  userTouchedRepeat: false,

  // Active synthesized payload
  currentResult: null,
  freshResult: null,
  blendModeActive: false,
  blendConfig: { id_b: null, mode: "height", factor: 0.5, contrast: 0.08 },

  // Viewport controls
  geom: "sphere",
  channel: "pbr",
  lightRig: "raking",
  azimuthDeg: 38,
  elevationDeg: 14,
  iblMult: 1.0,
  specAA: 1.0,
  exposure: 1.08,
  repeat: 4.0,
  normalScale: 1.0,
  displacement: 0.012,
  roughMult: 1.0,
  metalScale: 1.0,
  autoRotate: false,
  showSeamGrid: false,

  // A/B comparison
  splitActive: false,
  splitFraction: 0.5,
  holdingFresh: false,

  // Async sequencing
  synthSeq: 0,
  synthAbort: null,
  freshAbort: null,
  debounceTimer: null,
  lightboxKey: null,
};

// ============================================================================
// DOM References
// ============================================================================
const $ = (id) => document.getElementById(id);
const dom = {
  headerMatName: $("headerMatName"),
  headerMatKind: $("headerMatKind"),
  headerDeviceBadge: $("headerDeviceBadge"),
  btnSplitAB: $("btnSplitAB"),
  btnHoldFresh: $("btnHoldFresh"),
  btnDownloadZip: $("btnDownloadZip"),
  catalogCount: $("catalogCount"),
  catalogSearch: $("catalogSearch"),
  btnClearSearch: $("btnClearSearch"),
  categoryPills: $("categoryPills"),
  materialList: $("materialList"),
  viewportWrap: $("viewportWrap"),
  glCanvas: $("glCanvas"),
  geomSelector: $("geomSelector"),
  lightRigSelect: $("lightRigSelect"),
  btnAutoStage: $("btnAutoStage"),
  btnAutoRotate: $("btnAutoRotate"),
  btnSeamGrid: $("btnSeamGrid"),
  btnSnapshot: $("btnSnapshot"),
  btnResetCam: $("btnResetCam"),
  lightHudPill: $("lightHudPill"),
  splitOverlay: $("splitOverlay"),
  splitHandle: $("splitHandle"),
  freshHoldBanner: $("freshHoldBanner"),
  vpStatusPill: $("vpStatusPill"),
  vpStatusText: $("vpStatusText"),
  vpCaption: $("vpCaption"),
  mapGridOverlay: $("mapGridOverlay"),
  mapGridSubtitle: $("mapGridSubtitle"),
  mapGridContent: $("mapGridContent"),
  chkGrid2x2: $("chkGrid2x2"),
  btnCloseGrid: $("btnCloseGrid"),
  mapLightbox: $("mapLightbox"),
  lightboxTitle: $("lightboxTitle"),
  lightboxStage: $("lightboxStage"),
  btnLightbox3D: $("btnLightbox3D"),
  btnLightboxClose: $("btnLightboxClose"),
  channelBar: $("channelBar"),
  weatherPresets: $("weatherPresets"),
  btnResetWeather: $("btnResetWeather"),
  chkAutoSynth: $("chkAutoSynth"),
  inputSeed: $("inputSeed"),
  btnRandSeed: $("btnRandSeed"),
  btnApplySynth: $("btnApplySynth"),
  btnResetSurface: $("btnResetSurface"),
  blendMatALabel: $("blendMatALabel"),
  blendMatBSelect: $("blendMatBSelect"),
  blendModeSeg: $("blendModeSeg"),
  slBlendFactor: $("slBlendFactor"),
  valBlendFactor: $("valBlendFactor"),
  slBlendContrast: $("slBlendContrast"),
  valBlendContrast: $("valBlendContrast"),
  btnRunBlend: $("btnRunBlend"),
  btnExitBlend: $("btnExitBlend"),
  qaBadge: $("qaBadge"),
  qaLatency: $("qaLatency"),
  qaSeam: $("qaSeam"),
  qaNormErr: $("qaNormErr"),
  qaRough: $("qaRough"),
  qaMetal: $("qaMetal"),
  qaAO: $("qaAO"),
  qaOrm: $("qaOrm"),
  qaMessages: $("qaMessages"),
  qaCacheHash: $("qaCacheHash"),
  btnCopyPath: $("btnCopyPath"),
  atlasBody: $("atlasBody"),
  atlasEmpty: $("atlasEmpty"),
  btnAgentAtlas: $("btnAgentAtlas"),
  agentModal: $("agentModal"),
  agentModalBody: $("agentModalBody"),
  btnAgentClose: $("btnAgentClose"),
  toast: $("toast"),
};

const ATLAS_PUBLIC_URL = "https://hypbr.vercel.app";

// ============================================================================
// Three.js Studio Stage (HDR 8x MSAA + SMAA + Specular AA + Softbox IBL)
// ============================================================================
function getEffectivePixelRatio() {
  const dpr = window.devicePixelRatio || 1;
  // Supersample 1x displays to 1.5x; run native up to 2.0x on Retina displays
  return Math.min(Math.max(dpr, 1.5), 2.0);
}

const renderer = new THREE.WebGLRenderer({
  canvas: dom.glCanvas,
  antialias: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(getEffectivePixelRatio());
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.AgXToneMapping;
renderer.toneMappingExposure = state.exposure;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(LIGHT_RIGS.raking.bgBottom);

// Custom High-Contrast Procedural Studio Softbox HDRI Environment (PMREM)
// Provides crisp vertical strip reflectors + warm overhead softbox + cool bounce
// so metallic, brushed, glazed, and wet surfaces show true photographic speculars.
function buildStudioEnvironment(pmrem) {
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x0b0e14);

  const disposableGeoms = [];
  const disposableMats = [];
  function addPanel(w, h, color, intensity, pos, lookTarget = [0, 0, 0]) {
    const g = new THREE.PlaneGeometry(w, h);
    const c = new THREE.Color(color).multiplyScalar(intensity);
    const m = new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide });
    const p = new THREE.Mesh(g, m);
    p.position.set(...pos);
    p.lookAt(...lookTarget);
    envScene.add(p);
    disposableGeoms.push(g);
    disposableMats.push(m);
  }

  // Warm-white overhead key softbox
  addPanel(7.5, 5.5, 0xfff4e4, 2.8, [1.4, 6.2, 1.8]);
  // Crisp vertical right specular strip (studio look-dev highlight)
  addPanel(1.3, 6.5, 0xfff8ee, 5.2, [5.6, 1.8, 3.4]);
  // Cool left fill softbox
  addPanel(4.2, 4.8, 0x8fb8ff, 1.85, [-5.8, 2.2, 2.4]);
  // Warm-gold rear rim strip
  addPanel(2.0, 5.2, 0xffd9a0, 3.4, [-2.4, 2.8, -5.8]);
  // Subtle horizon bounce band & dark studio floor
  addPanel(12.0, 2.2, 0x4b5c78, 0.55, [0.0, 0.3, -6.5]);
  addPanel(16.0, 16.0, 0x10141c, 0.35, [0.0, -5.5, 0.0]);

  const rt = pmrem.fromScene(envScene, 0.025);
  for (const g of disposableGeoms) g.dispose();
  for (const m of disposableMats) m.dispose();
  return rt;
}

const pmremGenerator = new THREE.PMREMGenerator(renderer);
const envRT = buildStudioEnvironment(pmremGenerator);
scene.environment = envRT.texture;
pmremGenerator.dispose();

// Studio Photographic Radial Gradient Cyclorama Dome (included in PNG snapshots!)
const backdropUniforms = {
  uTop: { value: new THREE.Color(LIGHT_RIGS.raking.bgTop) },
  uHorizon: { value: new THREE.Color(LIGHT_RIGS.raking.bgHorizon) },
  uBottom: { value: new THREE.Color(LIGHT_RIGS.raking.bgBottom) },
};
const backdropDome = new THREE.Mesh(
  new THREE.SphereGeometry(26, 32, 20),
  new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: backdropUniforms,
    vertexShader: `
      varying vec3 vWorldDir;
      void main() {
        vWorldDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 uTop;
      uniform vec3 uHorizon;
      uniform vec3 uBottom;
      varying vec3 vWorldDir;
      void main() {
        float y = vWorldDir.y;
        float hGlow = exp(-pow(y * 2.6, 2.0));
        vec3 base = y >= 0.0 ? mix(uHorizon, uTop, smoothstep(0.0, 0.72, y))
                             : mix(uHorizon, uBottom, smoothstep(0.0, 0.65, -y));
        vec3 col = mix(base, uHorizon * 1.18, hGlow * 0.45);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  })
);
backdropDome.renderOrder = -10;
scene.add(backdropDome);

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 90);
camera.position.set(0, 0.32, 3.35);

const controls = new OrbitControls(camera, dom.glCanvas);
controls.enableDamping = true;
controls.dampingFactor = 0.065;
controls.minDistance = 1.15;
controls.maxDistance = 8.5;
controls.autoRotateSpeed = 1.4;

// Reusable lighting rig (key casts soft 2K PCF shadows)
const keyLight = new THREE.DirectionalLight(0xfff0d8, 3.55);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 16;
keyLight.shadow.camera.left = -2.4;
keyLight.shadow.camera.right = 2.4;
keyLight.shadow.camera.top = 2.4;
keyLight.shadow.camera.bottom = -2.4;
keyLight.shadow.bias = -0.0005;
keyLight.shadow.normalBias = 0.022;
keyLight.shadow.radius = 3.5;

const fillLight = new THREE.DirectionalLight(0x6f93c9, 0.26);
const rimLight = new THREE.DirectionalLight(0xfde68a, 0.95);
const hemiLight = new THREE.HemisphereLight(0x7c8ea8, 0x0b0d12, 0.14);
scene.add(keyLight, fillLight, rimLight, hemiLight);

// Grounded Studio Shadow Catcher + Subtle Contact Halo Disc (for 3D meshes)
const GEOM_FLOOR_Y = {
  sphere: -1.03,
  box: -0.72,
  knot: -0.96,
  cylinder: -0.79,
  plane: -1.6,
  wall2x2: -1.6,
};
const shadowMat = new THREE.ShadowMaterial({ opacity: 0.44 });
const shadowFloor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), shadowMat);
shadowFloor.rotation.x = -Math.PI / 2;
shadowFloor.position.y = GEOM_FLOOR_Y.sphere;
shadowFloor.receiveShadow = true;
scene.add(shadowFloor);

const studioHaloDisc = new THREE.Mesh(
  new THREE.CircleGeometry(2.35, 64),
  new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: { uColor: { value: new THREE.Color(0x000000) } },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      varying vec2 vUv;
      void main() {
        float r = length(vUv - 0.5) * 2.0;
        float ao = exp(-r * r * 5.2) * 0.34;
        float ring = smoothstep(0.84, 0.87, r) * (1.0 - smoothstep(0.87, 0.92, r)) * 0.08;
        gl_FragColor = vec4(mix(uColor, vec3(0.45, 0.52, 0.65), ring * 5.0), ao + ring);
      }
    `,
  })
);
studioHaloDisc.rotation.x = -Math.PI / 2;
studioHaloDisc.position.y = GEOM_FLOOR_Y.sphere - 0.002;
scene.add(studioHaloDisc);

// Reusable high-subdivision geometries (with uv2 copied from uv for aoMap)
function prepGeom(g) {
  if (g.attributes.uv && !g.attributes.uv2) {
    g.setAttribute("uv2", g.attributes.uv);
  }
  return g;
}

const GEOMETRIES = {
  sphere: prepGeom(new THREE.SphereGeometry(1.02, 288, 224)),
  box: prepGeom(new RoundedBoxGeometry(1.42, 1.42, 1.42, 20, 0.14)),
  knot: prepGeom(new THREE.TorusKnotGeometry(0.68, 0.235, 300, 72)),
  cylinder: prepGeom(new THREE.CylinderGeometry(0.76, 0.76, 1.56, 200, 112, false)),
  plane: prepGeom(new THREE.PlaneGeometry(2.35, 2.35, 256, 256)),
  wall2x2: prepGeom(new THREE.PlaneGeometry(2.72, 2.72, 256, 256)),
};

// Visual 2x2 seam crosshair lines (mounted in front of wall2x2 when toggled)
function buildSeamGuide() {
  const half = 1.36;
  const z = 0.095;
  const pts = [
    new THREE.Vector3(0, -half, z), new THREE.Vector3(0, half, z),
    new THREE.Vector3(-half, 0, z), new THREE.Vector3(half, 0, z),
  ];
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const m = new THREE.LineBasicMaterial({
    color: 0xe5a93b,
    transparent: true,
    opacity: 0.78,
    depthTest: false,
  });
  const lines = new THREE.LineSegments(g, m);
  lines.visible = false;
  lines.renderOrder = 5;
  return lines;
}
const seamGuide = buildSeamGuide();
scene.add(seamGuide);

// GPU Studio Anti-Aliasing Suite:
// 1. 5-tap texel-filtered vertex displacement (eliminates vertex-grid sawtooth faceting!)
// 2. 4-tap GPU Cubic B-Spline magnification (eliminates close-up bilinear normal diamond artifacts!)
// 3. 4-tap Rotated Grid Super-Sampling (RGSS) minification (eliminates grazing/distance moire!)
// 4. Dual Specular AA: Kaplanyan/Filament screen-space normal variance + Toksvig mipmap variance
const specAAUniform = { value: state.specAA };

function applySpecularAA(mat) {
  mat.customProgramCacheKey = () => "hypbr-pbr-studio-aa-v3";
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uSpecAA = specAAUniform;

    // 5-tap smooth texel-filtered vertex displacement eliminates coarse triangle spikes
    shader.vertexShader = shader.vertexShader.replace(
      "#include <displacementmap_vertex>",
      `#ifdef USE_DISPLACEMENTMAP
        vec2 dispTexel = 1.0 / vec2( textureSize( displacementMap, 0 ) );
        float dC = texture2D( displacementMap, vDisplacementMapUv ).x;
        float dL = texture2D( displacementMap, vDisplacementMapUv - vec2( dispTexel.x, 0.0 ) ).x;
        float dR = texture2D( displacementMap, vDisplacementMapUv + vec2( dispTexel.x, 0.0 ) ).x;
        float dD = texture2D( displacementMap, vDisplacementMapUv - vec2( 0.0, dispTexel.y ) ).x;
        float dU = texture2D( displacementMap, vDisplacementMapUv + vec2( 0.0, dispTexel.y ) ).x;
        float dispSmooth = 0.40 * dC + 0.15 * ( dL + dR + dD + dU );
        transformed += normalize( objectNormal ) * ( dispSmooth * displacementScale + displacementBias );
      #endif`
    );

    const aaHelpers = `
      uniform float uSpecAA;
      // Exact 4-tap GPU Cubic B-Spline lookup (Ruijters / Sigg-Hadwiger): strictly non-negative weights, C2-smooth, zero overshoot!
      vec3 sampleCubicBSpline3( sampler2D tex, vec2 uv, vec2 texSize ) {
        vec2 invSize = 1.0 / texSize;
        vec2 tc = uv * texSize - 0.5;
        vec2 i = floor( tc );
        vec2 f = tc - i;
        vec2 f2 = f * f;
        vec2 f3 = f2 * f;
        vec2 omf = 1.0 - f;
        vec2 w0 = ( omf * omf * omf ) * ( 1.0 / 6.0 );
        vec2 w1 = ( 3.0 * f3 - 6.0 * f2 + 4.0 ) * ( 1.0 / 6.0 );
        vec2 w2 = ( -3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0 ) * ( 1.0 / 6.0 );
        vec2 w3 = f3 * ( 1.0 / 6.0 );
        vec2 s0 = w0 + w1;
        vec2 s1 = w2 + w3;
        vec2 t0 = ( i - 0.5 + w1 / s0 ) * invSize;
        vec2 t1 = ( i + 1.5 + w3 / s1 ) * invSize;
        vec3 c00 = texture2D( tex, vec2( t0.x, t0.y ) ).xyz;
        vec3 c10 = texture2D( tex, vec2( t1.x, t0.y ) ).xyz;
        vec3 c01 = texture2D( tex, vec2( t0.x, t1.y ) ).xyz;
        vec3 c11 = texture2D( tex, vec2( t1.x, t1.y ) ).xyz;
        return ( c00 * s0.x + c10 * s1.x ) * s0.y + ( c01 * s0.x + c11 * s1.x ) * s1.y;
      }
      // 4-tap Rotated Grid Super-Sampling (RGSS) for crisp oblique/distance minification AA
      vec3 sampleRGSS3( sampler2D tex, vec2 uv, vec2 dx, vec2 dy ) {
        vec3 s = vec3( 0.0 );
        s += texture2D( tex, uv + dx *  0.375 + dy *  0.125 ).xyz;
        s += texture2D( tex, uv + dx * -0.375 + dy * -0.125 ).xyz;
        s += texture2D( tex, uv + dx * -0.125 + dy *  0.375 ).xyz;
        s += texture2D( tex, uv + dx *  0.125 + dy * -0.375 ).xyz;
        return 0.25 * s;
      }
    `;

    shader.fragmentShader =
      aaHelpers +
      shader.fragmentShader.replace(
        "#include <normal_fragment_maps>",
        `#ifdef USE_NORMALMAP_OBJECTSPACE
          normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
          #ifdef FLIP_SIDED
            normal = - normal;
          #endif
          #ifdef DOUBLE_SIDED
            normal = normal * faceDirection;
          #endif
          normal = normalize( normalMatrix * normal );
        #elif defined( USE_NORMALMAP_TANGENTSPACE )
          vec2 nTexSize = vec2( textureSize( normalMap, 0 ) );
          vec2 nDx = dFdx( vNormalMapUv );
          vec2 nDy = dFdy( vNormalMapUv );
          float nFootprint = max( length( nDx * nTexSize ), length( nDy * nTexSize ) );
          vec3 nBaseSample = texture2D( normalMap, vNormalMapUv ).xyz;
          vec3 nFiltered = nBaseSample;
          if ( uSpecAA > 0.001 ) {
            if ( nFootprint < 0.95 ) {
              float wCubic = smoothstep( 0.95, 0.45, nFootprint ) * clamp( uSpecAA, 0.0, 1.0 );
              nFiltered = mix( nBaseSample, sampleCubicBSpline3( normalMap, vNormalMapUv, nTexSize ), wCubic );
            } else if ( nFootprint > 1.05 ) {
              float wRgss = smoothstep( 1.05, 1.85, nFootprint ) * clamp( uSpecAA, 0.0, 1.0 );
              nFiltered = mix( nBaseSample, sampleRGSS3( normalMap, vNormalMapUv, nDx, nDy ), wRgss );
            }
          }
          vec3 rawMapN = nFiltered * 2.0 - 1.0;
          vec3 mapN = rawMapN;
          #if defined( USE_PACKED_NORMALMAP )
            mapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );
          #endif
          mapN.xy *= normalScale;
          normal = normalize( tbn * mapN );
        #elif defined( USE_BUMPMAP )
          normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
        #endif
        if ( uSpecAA > 0.001 ) {
          vec3 dndu = dFdx( normal );
          vec3 dndv = dFdy( normal );
          float screenVar = 0.25 * ( dot( dndu, dndu ) + dot( dndv, dndv ) );
          float kRough = min( 2.0 * screenVar * uSpecAA, 0.24 );
          #ifdef USE_NORMALMAP_TANGENTSPACE
            float nLen = clamp( length( rawMapN ), 0.05, 1.0 );
            float toksvigVar = ( 1.0 - nLen ) / nLen;
            float nScaleMag = 0.5 * ( abs( normalScale.x ) + abs( normalScale.y ) );
            kRough += min( toksvigVar * 0.65 * max( nScaleMag, 0.5 ) * uSpecAA, 0.28 );
          #endif
          roughnessFactor = sqrt( clamp( roughnessFactor * roughnessFactor + kRough, 0.0, 1.0 ) );
        }`
      );
  };
}

// Primary & Fresh (A/B) Material Pairs
function makePbrMaterial() {
  const mat = new THREE.MeshPhysicalMaterial({
    roughness: 1.0,
    metalness: 1.0,
    normalScale: new THREE.Vector2(1.0, 1.0),
    displacementScale: state.displacement,
    displacementBias: -0.5 * state.displacement,
    aoMapIntensity: 1.0,
    clearcoat: 0.0,
    clearcoatRoughness: 0.12,
    sheen: 0.0,
    sheenRoughness: 0.68,
    sheenColor: new THREE.Color(0xffffff),
    side: THREE.FrontSide,
  });
  applySpecularAA(mat);
  return mat;
}

function makeFlatMaterial() {
  return new THREE.MeshBasicMaterial({ side: THREE.FrontSide });
}

const matMainPbr = makePbrMaterial();
const matMainFlat = makeFlatMaterial();
const matFreshPbr = makePbrMaterial();
const matFreshFlat = makeFlatMaterial();

const mesh = new THREE.Mesh(GEOMETRIES.sphere, matMainPbr);
mesh.castShadow = true;
mesh.receiveShadow = true;
scene.add(mesh);

// WebGL Context Loss Recovery
dom.glCanvas.addEventListener("webglcontextlost", (e) => {
  e.preventDefault();
  setStatus("WebGL context lost — waiting to restore…", true);
});
dom.glCanvas.addEventListener("webglcontextrestored", () => {
  setStatus("", false);
  if (state.currentResult) {
    applyTexturesToSlot(state.currentResult, "main");
  }
});

// ============================================================================
// Texture Lifecycle Management (Zero Memory Leaks)
// ============================================================================
const texLoader = new THREE.TextureLoader();
const maxAniso = Math.min(renderer.capabilities.getMaxAnisotropy(), 16);

const textureSlots = {
  main: { cacheKey: null, map: new Map() },
  fresh: { cacheKey: null, map: new Map() },
};

function disposeTextureSlot(slotName) {
  const slot = textureSlots[slotName];
  for (const tex of slot.map.values()) {
    tex.dispose();
  }
  slot.map.clear();
  slot.cacheKey = null;
}

function effectiveRepeatXY() {
  const rep = state.repeat;
  if (state.geom === "wall2x2") {
    const r = Math.max(2, rep * 2);
    return [r, r];
  }
  // Compensate curved UV aspect ratios so tiles remain 1:1 square with zero stretch:
  // Sphere equirectangular UV is 2:1 -> (2*rep, rep) displays rep x rep square tiles on the front hemisphere!
  if (state.geom === "sphere") return [rep * 2, rep];
  if (state.geom === "cylinder") return [rep * 2, rep];
  if (state.geom === "knot") return [rep * 4, rep];
  return [rep, rep];
}

function configureTexture(tex, srgb) {
  const [rx, ry] = effectiveRepeatXY();
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(rx, ry);
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = maxAniso;
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function loadTexturePromise(url, srgb) {
  return new Promise((resolve, reject) => {
    texLoader.load(
      url,
      (tex) => resolve(configureTexture(tex, srgb)),
      undefined,
      (err) => reject(err)
    );
  });
}

const CORE_3D_KEYS = [
  { key: "albedo_delit", srgb: true },
  { key: "normal", srgb: false },
  { key: "roughness", srgb: false },
  { key: "metallic", srgb: false },
  { key: "ao", srgb: false },
  { key: "height", srgb: false },
];

async function ensureChannelTexture(slotName, result, key) {
  const slot = textureSlots[slotName];
  if (slot.map.has(key)) return slot.map.get(key);
  const url = result?.maps?.[key];
  if (!url) return null;
  const srgb = key === "albedo_delit" || key === "preview";
  const tex = await loadTexturePromise(url, srgb);
  // Guard if slot switched while loading
  if (slot.cacheKey !== result.cache_key) {
    tex.dispose();
    return null;
  }
  slot.map.set(key, tex);
  return tex;
}

async function applyTexturesToSlot(result, slotName) {
  const slot = textureSlots[slotName];
  if (slot.cacheKey === result.cache_key && slot.map.size >= 6) {
    bindSlotToMaterials(slotName);
    return;
  }

  // Load new core PBR textures before disposing old ones so there is no black flash
  const loaded = new Map();
  const entries = CORE_3D_KEYS.filter(({ key }) => Boolean(result.maps?.[key]));
  // If inspecting a non-core channel (curvature / weather_mask), preload it too
  if (
    state.channel !== "pbr" &&
    state.channel !== "grid2d" &&
    !entries.some((e) => e.key === state.channel) &&
    result.maps?.[state.channel]
  ) {
    entries.push({ key: state.channel, srgb: state.channel === "albedo_delit" });
  }

  const results = await Promise.allSettled(
    entries.map(async ({ key, srgb }) => {
      const tex = await loadTexturePromise(result.maps[key], srgb);
      return [key, tex];
    })
  );

  for (const r of results) {
    if (r.status === "fulfilled") {
      const [k, tex] = r.value;
      loaded.set(k, tex);
    }
  }

  // Dispose old GPU textures cleanly
  disposeTextureSlot(slotName);
  slot.cacheKey = result.cache_key;
  for (const [k, tex] of loaded.entries()) {
    slot.map.set(k, tex);
  }
  bindSlotToMaterials(slotName);
}

const GLAZED_IDS = new Set([
  "subway_metro_tile",
  "penny_round_mosaic",
  "mermaid_fishscale_tile",
  "fluted_reeded_glass",
  "frosted_prism_diffuser",
  "solar_cell_grid",
  "carrara_veined_marble",
]);

function applyPhysicalMaterialProps(pbr, isFreshSlot) {
  const entry = state.selectedId ? state.catalogById.get(state.selectedId) : null;
  const kind = entry?.kind || "";
  const isFabric = kind === "fabric";
  const isGlazed = kind === "glass" || (entry && GLAZED_IDS.has(entry.id));
  const wet = isFreshSlot ? 0 : (state.weather.wet || 0);

  pbr.roughness = state.roughMult;
  pbr.metalness = state.metalScale;
  pbr.normalScale.set(state.normalScale, state.normalScale);
  const useDisp = state.geom !== "box" ? state.displacement : state.displacement * 0.45;
  pbr.displacementScale = useDisp;
  pbr.displacementBias = -0.5 * useDisp;

  const baseCoat = isGlazed ? 0.32 : 0.0;
  pbr.clearcoat = Math.min(0.85, baseCoat + wet * 0.62);
  pbr.clearcoatRoughness = wet > 0.02 ? 0.05 : 0.11;
  pbr.sheen = isFabric ? 0.42 : 0.0;
  pbr.sheenRoughness = 0.72;
}

function bindSlotToMaterials(slotName) {
  const slot = textureSlots[slotName];
  const isFresh = slotName === "fresh";
  const pbr = isFresh ? matFreshPbr : matMainPbr;
  const flat = isFresh ? matFreshFlat : matMainFlat;

  pbr.map = slot.map.get("albedo_delit") || null;
  pbr.normalMap = slot.map.get("normal") || null;
  pbr.roughnessMap = slot.map.get("roughness") || null;
  pbr.metalnessMap = slot.map.get("metallic") || null;
  pbr.aoMap = slot.map.get("ao") || null;
  pbr.displacementMap = slot.map.get("height") || null;

  applyPhysicalMaterialProps(pbr, isFresh);
  const doubleSide = state.geom === "plane" || state.geom === "wall2x2";
  pbr.side = doubleSide ? THREE.DoubleSide : THREE.FrontSide;
  pbr.needsUpdate = true;

  if (state.channel !== "pbr" && state.channel !== "grid2d") {
    const chTex = slot.map.get(state.channel) || null;
    flat.map = chTex;
    flat.color.setHex(chTex ? 0xffffff : 0x1a1e26);
    flat.side = doubleSide ? THREE.DoubleSide : THREE.FrontSide;
    flat.needsUpdate = true;
  }
}

function syncAllTextureRepeats() {
  const [rx, ry] = effectiveRepeatXY();
  for (const slot of Object.values(textureSlots)) {
    for (const tex of slot.map.values()) {
      tex.repeat.set(rx, ry);
    }
  }
}

function updateMaterialScalars() {
  specAAUniform.value = state.specAA;
  applyPhysicalMaterialProps(matMainPbr, false);
  applyPhysicalMaterialProps(matFreshPbr, true);
}

// ============================================================================
// Lighting & Camera Controls
// ============================================================================
function updateStageFloor() {
  const showFloor = state.channel === "pbr" && state.geom !== "plane" && state.geom !== "wall2x2";
  const fy = GEOM_FLOOR_Y[state.geom] ?? -1.03;
  shadowFloor.position.y = fy;
  studioHaloDisc.position.y = fy - 0.002;
  shadowFloor.visible = showFloor;
  studioHaloDisc.visible = showFloor;
}

function applyLightingRig(preserveElevation = false) {
  const rig = LIGHT_RIGS[state.lightRig] || LIGHT_RIGS.raking;
  scene.background.setHex(rig.bgBottom);
  backdropUniforms.uTop.value.setHex(rig.bgTop);
  backdropUniforms.uHorizon.value.setHex(rig.bgHorizon);
  backdropUniforms.uBottom.value.setHex(rig.bgBottom);
  scene.environmentIntensity = rig.envIntensity * state.iblMult;
  shadowMat.opacity = rig.shadowOpacity ?? 0.4;

  keyLight.color.setHex(rig.keyColor);
  keyLight.intensity = rig.keyIntensity;
  fillLight.color.setHex(rig.fillColor);
  fillLight.intensity = rig.fillIntensity;
  fillLight.position.set(...rig.fillPos);
  rimLight.color.setHex(rig.rimColor);
  rimLight.intensity = rig.rimIntensity;
  rimLight.position.set(...rig.rimPos);
  hemiLight.color.setHex(rig.hemiSky);
  hemiLight.groundColor.setHex(rig.hemiGround);
  hemiLight.intensity = rig.hemiIntensity;

  if (!preserveElevation) {
    state.elevationDeg = rig.keyElevDeg ?? 14;
    const slE = $("slElevation");
    const valE = $("valElevation");
    if (slE) slE.value = String(state.elevationDeg);
    if (valE) valE.textContent = `${Math.round(state.elevationDeg)}°`;
  }

  updateLightAzimuth();
}

function updateLightAzimuth() {
  const rig = LIGHT_RIGS[state.lightRig] || LIGHT_RIGS.raking;
  const azRad = (state.azimuthDeg * Math.PI) / 180;
  const elRad = (state.elevationDeg * Math.PI) / 180;
  const r = rig.keyRadius || 5.0;
  const horiz = Math.cos(elRad) * r;
  keyLight.position.set(
    Math.cos(azRad) * horiz,
    Math.sin(elRad) * r,
    Math.sin(azRad) * horiz
  );
  // Rotate PMREM studio softbox HDRI in lockstep with key light azimuth!
  if (scene.environmentRotation) {
    scene.environmentRotation.y = -((state.azimuthDeg - 38) * Math.PI) / 180;
  }
  // Orbit rim opposite key slightly for responsive specular inspection
  const rimRad = azRad + Math.PI * 0.78;
  rimLight.position.set(
    Math.cos(rimRad) * 4.2,
    rig.rimPos[1],
    Math.sin(rimRad) * 4.2
  );
}

function setRepeatValue(val) {
  state.repeat = val;
  $("slRepeat").value = String(val);
  $("valRepeat").textContent = `${val.toFixed(1)}×`;
  syncAllTextureRepeats();
}

function applyRecommendedStage(entry) {
  const stage = getRecommendedStage(entry);
  state.userTouchedRepeat = false;
  state.lightRig = stage.lightRig || "raking";
  dom.lightRigSelect.value = state.lightRig;
  applyLightingRig();
  setRepeatValue(stage.repeat ?? GEOM_DEFAULT_REPEAT[stage.geom] ?? 4.0);
  setGeometry(stage.geom || "sphere", { fromAutoStage: true });
}

function setGeometry(geomKey, opts = {}) {
  if (!GEOMETRIES[geomKey]) return;
  const prevGeom = state.geom;
  state.geom = geomKey;
  mesh.geometry = GEOMETRIES[geomKey];

  if (!opts.fromAutoStage && state.autoStage && !state.userTouchedRepeat && prevGeom !== geomKey) {
    const defRep = GEOM_DEFAULT_REPEAT[geomKey];
    if (defRep != null) {
      state.repeat = defRep;
      $("slRepeat").value = String(defRep);
      $("valRepeat").textContent = `${defRep.toFixed(1)}×`;
    }
  }

  const isWall = geomKey === "wall2x2";
  dom.btnSeamGrid.classList.toggle("hidden", !isWall);
  seamGuide.visible = isWall && state.showSeamGrid;

  // Adjust camera subtly for flat planes vs 3D shaderballs
  if (geomKey === "plane" || geomKey === "wall2x2") {
    camera.position.set(0, 0.05, 3.65);
    controls.target.set(0, 0, 0);
  } else {
    if (Math.abs(camera.position.y) < 0.12 && camera.position.z > 3.5) {
      camera.position.set(0, 0.32, 3.35);
      controls.target.set(0, 0, 0);
    }
  }
  controls.update();

  syncAllTextureRepeats();
  updateStageFloor();
  bindSlotToMaterials("main");
  if (textureSlots.fresh.cacheKey) bindSlotToMaterials("fresh");

  dom.geomSelector.querySelectorAll(".vp-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.geom === geomKey);
  });
}

function resetCamera() {
  if (state.geom === "plane" || state.geom === "wall2x2") {
    camera.position.set(0, 0.05, 3.65);
  } else {
    camera.position.set(0, 0.32, 3.35);
  }
  controls.target.set(0, 0, 0);
  controls.update();
}

// ============================================================================
// Three.js r187 Post-Processing: Linear HDR 8x MSAA -> SMAA -> AgX OutputPass
// ============================================================================
let vpWidth = 1;
let vpHeight = 1;

function activeMaterialFor(slotName) {
  if (state.channel === "pbr") {
    return slotName === "fresh" ? matFreshPbr : matMainPbr;
  }
  return slotName === "fresh" ? matFreshFlat : matMainFlat;
}

class SplitScenePass extends Pass {
  constructor() {
    super();
    this.needsSwap = false;
  }

  render(rend, writeBuffer, readBuffer) {
    const oldAutoClear = rend.autoClear;
    rend.autoClear = false;
    const target = this.renderToScreen ? null : readBuffer;
    rend.setRenderTarget(target);
    rend.clear(true, true, true);

    if (state.splitActive && textureSlots.fresh.cacheKey) {
      if (target) {
        const w = target.width;
        const h = target.height;
        const splitX = Math.max(1, Math.min(w - 1, Math.round(w * state.splitFraction)));
        target.scissorTest = true;

        target.scissor.set(0, 0, splitX, h);
        mesh.material = activeMaterialFor("fresh");
        rend.render(scene, camera);

        target.scissor.set(splitX, 0, w - splitX, h);
        mesh.material = activeMaterialFor("main");
        rend.render(scene, camera);

        target.scissorTest = false;
        target.scissor.set(0, 0, w, h);
      } else {
        const splitX = Math.max(1, Math.min(vpWidth - 1, Math.round(vpWidth * state.splitFraction)));
        rend.setScissorTest(true);
        rend.setScissor(0, 0, splitX, vpHeight);
        mesh.material = activeMaterialFor("fresh");
        rend.render(scene, camera);
        rend.setScissor(splitX, 0, vpWidth - splitX, vpHeight);
        mesh.material = activeMaterialFor("main");
        rend.render(scene, camera);
        rend.setScissorTest(false);
      }
    } else {
      const slotName = state.holdingFresh && textureSlots.fresh.cacheKey ? "fresh" : "main";
      mesh.material = activeMaterialFor(slotName);
      rend.render(scene, camera);
    }

    rend.autoClear = oldAutoClear;
  }
}

let composer = null;
let smaaPass = null;
let useComposer = false;

try {
  const msaaSamples = Math.min(8, renderer.capabilities.maxSamples || 4);
  const hdrTarget = new THREE.WebGLRenderTarget(1, 1, {
    type: THREE.HalfFloatType,
    samples: msaaSamples,
  });
  composer = new EffectComposer(renderer, hdrTarget);
  composer.setPixelRatio(getEffectivePixelRatio());
  composer.addPass(new SplitScenePass());
  // In Three.js r187, SMAAPass runs in linear-srgb right BEFORE OutputPass!
  smaaPass = new SMAAPass();
  composer.addPass(smaaPass);
  composer.addPass(new OutputPass());
  useComposer = true;
} catch (err) {
  console.warn("HDR MSAA+SMAA composer unavailable, falling back to direct MSAA:", err);
  useComposer = false;
}

function resizeViewport() {
  const rect = dom.viewportWrap.getBoundingClientRect();
  vpWidth = Math.max(1, Math.floor(rect.width));
  vpHeight = Math.max(1, Math.floor(rect.height));
  const dpr = getEffectivePixelRatio();
  camera.aspect = vpWidth / vpHeight;
  camera.updateProjectionMatrix();
  renderer.setPixelRatio(dpr);
  renderer.setSize(vpWidth, vpHeight, false);
  if (useComposer && composer) {
    composer.setPixelRatio(dpr);
    composer.setSize(vpWidth, vpHeight);
  }
}

const resizeObserver = new ResizeObserver(resizeViewport);
resizeObserver.observe(dom.viewportWrap);
resizeViewport();

function renderFrame() {
  if (useComposer && composer) {
    try {
      composer.render();
      return;
    } catch (err) {
      console.warn("Composer render failed, falling back to direct MSAA:", err);
      useComposer = false;
    }
  }

  if (state.splitActive && textureSlots.fresh.cacheKey) {
    const splitX = Math.max(1, Math.min(vpWidth - 1, Math.round(vpWidth * state.splitFraction)));
    renderer.setScissorTest(true);

    renderer.setScissor(0, 0, splitX, vpHeight);
    mesh.material = activeMaterialFor("fresh");
    renderer.render(scene, camera);

    renderer.setScissor(splitX, 0, vpWidth - splitX, vpHeight);
    mesh.material = activeMaterialFor("main");
    renderer.render(scene, camera);

    renderer.setScissorTest(false);
  } else {
    const slotName = state.holdingFresh && textureSlots.fresh.cacheKey ? "fresh" : "main";
    mesh.material = activeMaterialFor(slotName);
    renderer.render(scene, camera);
  }
}

function saveViewportSnapshot() {
  if (state.channel === "grid2d") return;
  renderFrame();
  try {
    const dataUrl = dom.glCanvas.toDataURL("image/png");
    const a = document.createElement("a");
    const slug = state.selectedId || "material";
    const preset = state.activePreset || "custom";
    a.href = dataUrl;
    a.download = `hypbr_${slug}_${preset}_${state.size}px.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (err) {
    console.error("Snapshot failed:", err);
  }
}

renderer.setAnimationLoop(() => {
  controls.autoRotate = state.autoRotate && !state.splitActive;
  controls.update();

  if (state.channel === "grid2d") return;
  renderFrame();
});

// ============================================================================
// API Layer & Synthesis Flow
// ============================================================================
function setStatus(text, show) {
  dom.vpStatusText.textContent = text;
  dom.vpStatusPill.classList.toggle("hidden", !show);
}

function isAnyWeatherActive() {
  return WEATHER_KEYS.some((k) => state.weather[k] > 1e-4);
}

async function fetchCatalog() {
  setStatus("Loading catalog…", true);
  let data = null;
  // Auto-detect live local Python server on localhost; load static edge bundle directly on remote hosts (zero 404s!)
  const isLocalHost = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
  if (isLocalHost) {
    try {
      const res = await fetch("./api/catalog", { headers: { Accept: "application/json" } });
      const ct = res.headers.get("content-type") || "";
      if (res.ok && ct.includes("application/json")) {
        data = await res.json();
        state.staticMode = false;
      }
    } catch {
      // Fall through to static bundle
    }
  }

  if (!data) {
    const res = await fetch("./assets/catalog.json");
    if (!res.ok) throw new Error(`Catalog fetch failed (${res.status})`);
    data = await res.json();
    state.staticMode = true;
  }

  state.catalog = data.materials || [];
  state.catalogById.clear();
  for (const m of state.catalog) state.catalogById.set(m.id, m);
  state.groups = data.groups || ["All"];
  state.groupCounts = data.group_counts || {};
  if (!data.group_counts) {
    for (const m of state.catalog) state.groupCounts[m.group] = (state.groupCounts[m.group] || 0) + 1;
  }
  state.weatherPresets = data.weather_presets || {};
  // Default to 512px in UI (even if an older server instance is still running)
  state.size = 512;
  document.querySelectorAll(".seg-btn[data-size]").forEach((b) => {
    const sz = Number(b.dataset.size);
    b.classList.toggle("active", sz === state.size);
    if (state.staticMode && sz !== 512) {
      b.classList.add("hidden");
    }
  });
  if (state.staticMode && dom.btnCopyPath) {
    dom.btnCopyPath.classList.add("hidden");
  }

  const devTag = state.staticMode ? "WEBGL" : String(data.device || "cpu").toUpperCase();
  dom.headerDeviceBadge.textContent = `${devTag} · ${state.catalog.length}`;

  renderCategoryPills();
  renderWeatherPresets();
  populateBlendPicker();
  applyCatalogFilter();

  if (state.catalog.length > 0) {
    // Start on a rich curved Shaderball hero (sphere @ 4.0x repeat + raking grazing + 512px)
    const preferred = state.catalogById.has("hammered_copper_dimple")
      ? "hammered_copper_dimple"
      : (state.catalogById.has("wood_parquet_herringbone")
        ? "wood_parquet_herringbone"
        : state.catalog[0].id);
    selectMaterial(preferred);
  }
}

async function synthesizeCurrent() {
  if (!state.selectedId) return;
  state.blendModeActive = false;
  dom.btnExitBlend.classList.add("hidden");

  if (state.synthAbort) state.synthAbort.abort();
  const controller = new AbortController();
  state.synthAbort = controller;
  const seq = ++state.synthSeq;

  const payload = {
    id: state.selectedId,
    size: state.size,
    seed: state.seed,
    wear: state.weather.wear,
    dirt: state.weather.dirt,
    wet: state.weather.wet,
    moss: state.weather.moss,
    rust: state.weather.rust,
    dust: state.weather.dust,
  };

  setStatus(state.staticMode ? `Loading ${state.size}px…` : `Synthesizing ${state.size}px…`, true);

  try {
    let result;
    if (state.staticMode) {
      const entry = state.catalogById.get(state.selectedId);
      result = await synthesizeStaticMaterial(entry, payload);
    } else {
      const resp = await fetch("./api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }
      result = await resp.json();
    }
    if (seq !== state.synthSeq) return;

    state.currentResult = result;
    if (!result.is_weathered) {
      state.freshResult = result;
      updateCatalogRowCache(result);
    }

    updateHeaderAndQA(result);
    updateWeatherMaskButtonState(result);

    setStatus("Uploading PBR maps…", true);
    await applyTexturesToSlot(result, "main");
    if (seq !== state.synthSeq) return;

    if (!result.is_weathered) {
      // Keep fresh slot synced if split mode is on
      if (state.splitActive) {
        await applyTexturesToSlot(result, "fresh");
      }
    } else if (state.splitActive || state.holdingFresh) {
      await ensureFreshLoaded();
    }

    if (state.channel === "grid2d") {
      renderDiagnosticGrid();
    }

    setStatus("", false);
  } catch (err) {
    if (err.name === "AbortError") return;
    console.error("Synthesis failed:", err);
    setStatus(`Error: ${err.message}`, true);
  }
}

async function ensureFreshLoaded() {
  if (!state.selectedId) return;
  const desiredSeed = state.seed;
  if (
    state.freshResult &&
    state.freshResult.id === state.selectedId &&
    state.freshResult.size === state.size &&
    (desiredSeed === null || state.freshResult.seed === desiredSeed)
  ) {
    await applyTexturesToSlot(state.freshResult, "fresh");
    return;
  }

  if (state.freshAbort) state.freshAbort.abort();
  const ctrl = new AbortController();
  state.freshAbort = ctrl;

  try {
    let fresh;
    if (state.staticMode) {
      const entry = state.catalogById.get(state.selectedId);
      fresh = await synthesizeStaticMaterial(entry, {
        id: state.selectedId,
        size: state.size,
        seed: desiredSeed,
        wear: 0, dirt: 0, wet: 0, moss: 0, rust: 0, dust: 0,
      });
    } else {
      const resp = await fetch("./api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: state.selectedId,
          size: state.size,
          seed: desiredSeed,
          wear: 0,
          dirt: 0,
          wet: 0,
          moss: 0,
          rust: 0,
          dust: 0,
        }),
        signal: ctrl.signal,
      });
      if (!resp.ok) return;
      fresh = await resp.json();
    }
    state.freshResult = fresh;
    updateCatalogRowCache(fresh);
    await applyTexturesToSlot(fresh, "fresh");
  } catch (err) {
    if (err.name !== "AbortError") console.error("Fresh load error:", err);
  }
}

async function runBlendSynthesis() {
  const idA = state.selectedId;
  const idB = dom.blendMatBSelect.value;
  if (!idA || !idB) return;

  if (state.synthAbort) state.synthAbort.abort();
  const controller = new AbortController();
  state.synthAbort = controller;
  const seq = ++state.synthSeq;

  setStatus(`Blending A × B (${state.size}px)…`, true);

  try {
    let result;
    if (state.staticMode) {
      const entryA = state.catalogById.get(idA);
      const entryB = state.catalogById.get(idB);
      result = await blendStaticMaterials(entryA, entryB, {
        mode: state.blendConfig.mode,
        factor: state.blendConfig.factor,
        contrast: state.blendConfig.contrast,
      });
    } else {
      const resp = await fetch("./api/blend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id_a: idA,
          id_b: idB,
          size: state.size,
          mode: state.blendConfig.mode,
          factor: state.blendConfig.factor,
          contrast: state.blendConfig.contrast,
        }),
        signal: controller.signal,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }
      result = await resp.json();
    }
    if (seq !== state.synthSeq) return;

    state.blendModeActive = true;
    state.currentResult = result;
    dom.btnExitBlend.classList.remove("hidden");

    updateHeaderAndQA(result);
    updateWeatherMaskButtonState(result);

    setStatus("Uploading blended maps…", true);
    await applyTexturesToSlot(result, "main");
    if (seq !== state.synthSeq) return;
    if (state.channel === "grid2d") renderDiagnosticGrid();
    setStatus("", false);
  } catch (err) {
    if (err.name === "AbortError") return;
    console.error("Blend failed:", err);
    setStatus(`Blend error: ${err.message}`, true);
  }
}

function scheduleAutoSynth() {
  if (!state.autoSynth) return;
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    synthesizeCurrent();
  }, 160);
}

// ============================================================================
// UI Rendering: Catalog, Presets, QA Telemetry, 2D Grid
// ============================================================================
function renderCategoryPills() {
  dom.categoryPills.innerHTML = "";
  for (const grp of state.groups) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "cat-pill" + (grp === state.activeGroup ? " active" : "");
    btn.dataset.group = grp;
    const count = grp === "All" ? state.catalog.length : (state.groupCounts[grp] || 0);
    btn.innerHTML = "";
    const label = document.createElement("span");
    label.textContent = grp === "All" ? "All" : grp;
    const cnt = document.createElement("span");
    cnt.className = "cat-count";
    cnt.textContent = String(count);
    btn.append(label, cnt);
    btn.addEventListener("click", () => {
      state.activeGroup = grp;
      dom.categoryPills.querySelectorAll(".cat-pill").forEach((el) => {
        el.classList.toggle("active", el.dataset.group === grp);
      });
      applyCatalogFilter();
    });
    dom.categoryPills.appendChild(btn);
  }
}

function applyCatalogFilter() {
  const q = state.searchQuery.trim().toLowerCase().replace(/sci[-\s]fi/g, "scifi");
  const toks = q.split(/[\s,]+/).filter((t) => t.length > 0).map((t) => (t.length > 3 && t.endsWith("s") && !t.endsWith("ss") ? t.slice(0, -1) : t));
  const scored = [];
  for (const m of state.catalog) {
    if (state.activeGroup !== "All" && m.group !== state.activeGroup) continue;
    if (!toks.length) {
      scored.push([0, m]);
      continue;
    }
    if (!m._hay) {
      m._hayName = `${m.name} ${m.id}`.toLowerCase().replace(/_/g, " ");
      m._hayObj = (m.best_objects || []).join(" ").toLowerCase().replace(/_/g, " ");
      m._hay = [m._hayName, m._hayObj, (m.scene_roles || []).join(" "), (m.tags || []).join(" ").replace(/_/g, " "),
        m.subcategory || "", m.kind || "", m.group || "", m.description || ""].join(" ").toLowerCase().replace(/_/g, " ");
    }
    let score = 0;
    let ok = true;
    for (const t of toks) {
      if (!m._hay.includes(t)) { ok = false; break; }
      score += (m._hayName.includes(t) ? 3 : 0) + (m._hayObj.includes(t) ? 2 : 0) +
        ((m.subcategory || "").replace(/_/g, " ").includes(t) ? 3 : 0) + 1;
    }
    if (ok) scored.push([score, m]);
  }
  if (toks.length) scored.sort((a, b) => b[0] - a[0]);
  state.filtered = scored.map((x) => x[1]);

  dom.catalogCount.textContent = `${state.filtered.length} / ${state.catalog.length}`;
  dom.btnClearSearch.classList.toggle("hidden", q.length === 0);
  renderMaterialList();
}

function renderMaterialList() {
  dom.materialList.innerHTML = "";
  const frag = document.createDocumentFragment();

  for (const item of state.filtered) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "mat-row" + (item.id === state.selectedId ? " active" : "");
    row.dataset.id = item.id;
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(item.id === state.selectedId));
    row.title = item.description;

    const thumb = document.createElement("div");
    thumb.className = "mat-thumb";
    if (item.preview_url) {
      const img = document.createElement("img");
      img.src = item.preview_url;
      img.alt = "";
      img.loading = "lazy";
      img.decoding = "async";
      img.width = 40;
      img.height = 40;
      thumb.appendChild(img);
    } else {
      const sw = document.createElement("div");
      sw.className = "mat-swatch";
      sw.style.backgroundColor = item.tint_hex;
      thumb.appendChild(sw);
    }

    const info = document.createElement("div");
    info.className = "mat-info";
    const nameEl = document.createElement("div");
    nameEl.className = "mat-name";
    nameEl.textContent = item.name;
    const subEl = document.createElement("div");
    subEl.className = "mat-sub";
    const sz = item.size_m ? ` · ${item.size_m[0]}×${item.size_m[1]} m` : "";
    subEl.textContent = `${(item.subcategory || item.kind).replace(/_/g, " ")}${sz}`;
    info.append(nameEl, subEl);

    const dot = document.createElement("span");
    dot.className = "mat-cache-dot" + (item.cached ? " cached" : "");
    dot.title = item.cached ? "Cached on disk" : "Not cached yet";

    row.append(thumb, info, dot);
    row.addEventListener("click", () => selectMaterial(item.id));
    frag.appendChild(row);
  }
  dom.materialList.appendChild(frag);
}

function updateCatalogRowCache(result) {
  const item = state.catalogById.get(result.id);
  if (!item) return;
  item.cached = true;
  if (result.maps?.preview) item.preview_url = result.maps.preview;

  const row = dom.materialList.querySelector(`.mat-row[data-id="${CSS.escape(result.id)}"]`);
  if (!row) return;
  const dot = row.querySelector(".mat-cache-dot");
  if (dot) {
    dot.classList.add("cached");
    dot.title = "Cached on disk";
  }
  const thumb = row.querySelector(".mat-thumb");
  if (thumb && item.preview_url && !thumb.querySelector("img")) {
    thumb.innerHTML = "";
    const img = document.createElement("img");
    img.src = item.preview_url;
    img.alt = "";
    thumb.appendChild(img);
  }
}

function selectMaterial(matId, scrollIntoView = false) {
  if (!state.catalogById.has(matId)) return;
  state.selectedId = matId;
  state.freshResult = null;
  disposeTextureSlot("fresh");

  const entry = state.catalogById.get(matId);
  dom.blendMatALabel.textContent = `${entry.name} (${entry.kind})`;

  // Reset seed placeholder to this material's default seed
  dom.inputSeed.placeholder = String(entry.default_seed);

  // Automatically pick best geometry, lighting rig & UV repeat for this material
  if (state.autoStage) {
    applyRecommendedStage(entry);
  }

  // Highlight active row
  dom.materialList.querySelectorAll(".mat-row").forEach((el) => {
    const active = el.dataset.id === matId;
    el.classList.toggle("active", active);
    el.setAttribute("aria-selected", String(active));
    if (active && scrollIntoView) {
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });

  synthesizeCurrent();
  loadAtlasInspector(matId);
}

function renderWeatherPresets() {
  dom.weatherPresets.innerHTML = "";
  const order = ["fresh", "aged", "rain", "mossy", "rusty", "dusty", "coastal", "abandoned"];
  const names = order.filter((n) => n in state.weatherPresets);
  for (const name of names) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset-pill" + (name === state.activePreset ? " active" : "");
    btn.dataset.preset = name;
    btn.textContent = name;
    btn.addEventListener("click", () => applyWeatherPreset(name));
    dom.weatherPresets.appendChild(btn);
  }
}

function applyWeatherPreset(name) {
  const vals = state.weatherPresets[name];
  if (!vals) return;
  state.activePreset = name;
  for (const k of WEATHER_KEYS) {
    state.weather[k] = Number(vals[k] || 0);
  }
  syncWeatherSliderDOM();
  dom.weatherPresets.querySelectorAll(".preset-pill").forEach((el) => {
    el.classList.toggle("active", el.dataset.preset === name);
  });
  synthesizeCurrent();
}

function syncWeatherSliderDOM() {
  const mapIds = {
    wear: ["slWear", "valWear"],
    dirt: ["slDirt", "valDirt"],
    wet: ["slWet", "valWet"],
    moss: ["slMoss", "valMoss"],
    rust: ["slRust", "valRust"],
    dust: ["slDust", "valDust"],
  };
  for (const [k, [slId, valId]] of Object.entries(mapIds)) {
    const v = state.weather[k];
    $(slId).value = String(v);
    $(valId).textContent = v.toFixed(2);
  }
}

function populateBlendPicker() {
  dom.blendMatBSelect.innerHTML = "";
  const byGroup = new Map();
  for (const item of state.catalog) {
    if (!byGroup.has(item.group)) {
      const og = document.createElement("optgroup");
      og.label = item.group;
      byGroup.set(item.group, og);
      dom.blendMatBSelect.appendChild(og);
    }
    const opt = document.createElement("option");
    opt.value = item.id;
    opt.textContent = item.name;
    byGroup.get(item.group).appendChild(opt);
  }
  const defB = state.catalogById.has("sidewalk_expansion_joint")
    ? "sidewalk_expansion_joint"
    : state.catalog[1]?.id || state.catalog[0]?.id;
  if (defB) {
    dom.blendMatBSelect.value = defB;
    state.blendConfig.id_b = defB;
  }
}

function updateHeaderAndQA(result) {
  dom.headerMatName.textContent = result.name;
  dom.headerMatKind.textContent = result.kind;
  dom.vpCaption.textContent = result.description || "";

  if (result.download_url) {
    dom.btnDownloadZip.href = result.download_url;
    dom.btnDownloadZip.classList.remove("disabled");
  } else if (state.staticMode) {
    dom.btnDownloadZip.href = "#download-zip";
    dom.btnDownloadZip.classList.remove("disabled");
  }

  const qa = result.verify || {};
  const passed = Boolean(qa.passed);
  dom.qaBadge.textContent = passed ? "PASS" : "FAIL";
  dom.qaBadge.className = "qa-badge " + (passed ? "qa-pass" : "qa-fail");

  const latTag = result.cached
    ? `CACHE · ${result.size}px`
    : `${result.elapsed_ms} ms · ${result.size}px`;
  dom.qaLatency.textContent = latTag;

  const sr = qa.seam_ratios || {};
  dom.qaSeam.textContent = qa.max_seam_ratio != null ? `${qa.max_seam_ratio.toFixed(2)}×` : "—";
  dom.qaSeam.title = `H:${sr.height ?? "—"} N:${sr.normal ?? "—"} A:${sr.albedo ?? "—"}`;
  dom.qaNormErr.textContent = qa.normal_unit_error != null ? qa.normal_unit_error.toFixed(4) : "—";

  const pbr = qa.pbr_ranges || {};
  dom.qaRough.textContent =
    pbr.roughness_mean != null
      ? `${pbr.roughness_mean.toFixed(2)}±${(pbr.roughness_std ?? 0).toFixed(2)}`
      : "—";
  dom.qaMetal.textContent = pbr.metallic_mean != null ? pbr.metallic_mean.toFixed(2) : "—";
  dom.qaAO.textContent = pbr.ao_mean != null ? pbr.ao_mean.toFixed(2) : "—";
  dom.qaOrm.textContent = qa.orm_max_diff != null ? String(qa.orm_max_diff) : "—";

  const msgs = [...(qa.errors || []), ...(qa.warnings || [])];
  if (msgs.length > 0) {
    dom.qaMessages.textContent = msgs.join(" · ");
    dom.qaMessages.classList.remove("hidden");
    dom.qaMessages.classList.toggle("has-error", (qa.errors || []).length > 0);
  } else {
    dom.qaMessages.classList.add("hidden");
    dom.qaMessages.textContent = "";
  }

  dom.qaCacheHash.textContent = `#${result.cache_key}`;
  dom.qaCacheHash.title = result.cache_dir || result.cache_key;
}

function updateWeatherMaskButtonState(result) {
  const hasMask = Boolean(result?.maps?.weather_mask);
  const btn = dom.channelBar.querySelector('[data-channel="weather_mask"]');
  if (btn) {
    btn.classList.toggle("muted", !hasMask);
    btn.title = hasMask
      ? "RGBA8 Weather Mask (R:Wear G:Dirt/Rust B:Wet A:Moss)"
      : "Apply a weathering layer to generate weather_mask.png";
  }
}

async function setChannel(chanKey) {
  state.channel = chanKey;
  dom.channelBar.querySelectorAll(".chan-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.channel === chanKey);
  });

  if (chanKey === "grid2d") {
    dom.mapGridOverlay.classList.remove("hidden");
    renderDiagnosticGrid();
    return;
  }

  dom.mapGridOverlay.classList.add("hidden");
  dom.mapLightbox.classList.add("hidden");

  if (chanKey !== "pbr" && state.currentResult) {
    if (!state.currentResult.maps?.[chanKey]) {
      setStatus(`No ${chanKey} map for Fresh material — enable weathering first`, true);
      setTimeout(() => setStatus("", false), 2200);
    } else {
      await ensureChannelTexture("main", state.currentResult, chanKey);
      if (state.splitActive && state.freshResult) {
        await ensureChannelTexture("fresh", state.freshResult, chanKey);
      }
    }
  }
  updateStageFloor();
  bindSlotToMaterials("main");
  if (textureSlots.fresh.cacheKey) bindSlotToMaterials("fresh");
}

function renderDiagnosticGrid() {
  const res = state.currentResult;
  if (!res) return;
  dom.mapGridSubtitle.textContent = `${res.name} · ${res.size}×${res.size} · #${res.cache_key}`;
  dom.mapGridContent.innerHTML = "";
  const tiled = dom.chkGrid2x2.checked;

  const frag = document.createDocumentFragment();
  for (const m of DIAGNOSTIC_MAPS) {
    const url = res.maps?.[m.key];
    const tile = document.createElement("div");
    tile.className = "map-tile";

    const imgWrap = document.createElement("div");
    imgWrap.className = "map-tile-img" + (tiled && url ? " tiled-2x2" : "");
    if (url) {
      imgWrap.style.backgroundImage = `url("${url}")`;
    } else {
      const empty = document.createElement("div");
      empty.className = "map-tile-empty";
      empty.textContent = m.key === "weather_mask" ? "Fresh material\n(no weather mask)" : "Unavailable";
      imgWrap.appendChild(empty);
    }

    const foot = document.createElement("div");
    foot.className = "map-tile-foot";
    const lbl = document.createElement("span");
    lbl.className = "map-tile-label";
    lbl.textContent = m.label;
    const sp = document.createElement("span");
    sp.className = "map-tile-space";
    sp.textContent = m.space;
    foot.append(lbl, sp);

    tile.append(imgWrap, foot);
    if (url) {
      tile.addEventListener("click", () => openMapLightbox(m, url));
    }
    frag.appendChild(tile);
  }
  dom.mapGridContent.appendChild(frag);
}

function openMapLightbox(mapMeta, url) {
  state.lightboxKey = mapMeta.key;
  dom.lightboxTitle.textContent = `${mapMeta.label} (${mapMeta.space}) — ${state.currentResult?.size || 256}px`;
  dom.lightboxStage.innerHTML = "";
  const img = document.createElement("img");
  img.src = url;
  img.alt = mapMeta.label;
  dom.lightboxStage.appendChild(img);

  const can3D = Boolean(dom.channelBar.querySelector(`[data-channel="${mapMeta.key}"]`));
  dom.btnLightbox3D.classList.toggle("hidden", !can3D);
  dom.mapLightbox.classList.remove("hidden");
}

// ============================================================================
// Split-Screen & Hold-Fresh A/B Comparison
// ============================================================================
async function toggleSplitAB() {
  state.splitActive = !state.splitActive;
  dom.btnSplitAB.classList.toggle("active", state.splitActive);
  dom.splitOverlay.classList.toggle("hidden", !state.splitActive);

  if (state.splitActive) {
    updateSplitHandleUI();
    setStatus("Loading Fresh (A) for split compare…", true);
    await ensureFreshLoaded();
    if (state.channel !== "pbr" && state.channel !== "grid2d" && state.freshResult) {
      await ensureChannelTexture("fresh", state.freshResult, state.channel);
      bindSlotToMaterials("fresh");
    }
    setStatus("", false);
  }
}

function updateSplitHandleUI() {
  const pct = (state.splitFraction * 100).toFixed(2);
  dom.splitHandle.style.left = `${pct}%`;
  dom.splitHandle.setAttribute("aria-valuenow", String(Math.round(state.splitFraction * 100)));
}

async function beginHoldFresh() {
  if (state.holdingFresh) return;
  state.holdingFresh = true;
  dom.btnHoldFresh.classList.add("active");
  dom.freshHoldBanner.classList.remove("hidden");
  await ensureFreshLoaded();
  if (state.channel !== "pbr" && state.channel !== "grid2d" && state.freshResult) {
    await ensureChannelTexture("fresh", state.freshResult, state.channel);
    bindSlotToMaterials("fresh");
  }
}

function endHoldFresh() {
  if (!state.holdingFresh) return;
  state.holdingFresh = false;
  dom.btnHoldFresh.classList.remove("active");
  dom.freshHoldBanner.classList.add("hidden");
}

// ============================================================================
// Event Wiring
// ============================================================================
function wireEvents() {
  // Resolution toggle
  document.querySelectorAll(".topbar .seg-btn[data-size]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const sz = Number(btn.dataset.size);
      if (sz === state.size) return;
      state.size = sz;
      state.freshResult = null;
      disposeTextureSlot("fresh");
      document.querySelectorAll(".topbar .seg-btn[data-size]").forEach((b) => {
        b.classList.toggle("active", Number(b.dataset.size) === sz);
      });
      if (state.blendModeActive) runBlendSynthesis();
      else synthesizeCurrent();
    });
  });

  // Client-side ZIP download in static edge mode (Vercel / GitHub Pages)
  dom.btnDownloadZip.addEventListener("click", async (e) => {
    if (!state.staticMode) return;
    e.preventDefault();
    if (!state.currentResult) return;
    setStatus("Packaging PNG ZIP…", true);
    try {
      await downloadStaticPbrZip(state.currentResult);
    } catch (err) {
      console.error("ZIP export failed:", err);
      setStatus(`ZIP error: ${err.message}`, true);
      return;
    }
    setStatus("", false);
  });

  // Search
  dom.catalogSearch.addEventListener("input", () => {
    state.searchQuery = dom.catalogSearch.value;
    applyCatalogFilter();
  });
  dom.btnClearSearch.addEventListener("click", () => {
    dom.catalogSearch.value = "";
    state.searchQuery = "";
    applyCatalogFilter();
    dom.catalogSearch.focus();
  });

  // Geometry switcher
  dom.geomSelector.addEventListener("click", (e) => {
    const btn = e.target.closest(".vp-btn[data-geom]");
    if (btn) setGeometry(btn.dataset.geom);
  });

  // Lighting rig & viewport quick buttons
  dom.lightRigSelect.addEventListener("change", () => {
    state.lightRig = dom.lightRigSelect.value;
    applyLightingRig();
  });
  dom.btnAutoStage.addEventListener("click", () => {
    state.autoStage = !state.autoStage;
    dom.btnAutoStage.classList.toggle("active", state.autoStage);
    if (state.autoStage && state.selectedId) {
      applyRecommendedStage(state.catalogById.get(state.selectedId));
    }
  });
  dom.btnAutoRotate.addEventListener("click", () => {
    state.autoRotate = !state.autoRotate;
    dom.btnAutoRotate.classList.toggle("active", state.autoRotate);
  });
  dom.btnSeamGrid.addEventListener("click", () => {
    state.showSeamGrid = !state.showSeamGrid;
    dom.btnSeamGrid.classList.toggle("active", state.showSeamGrid);
    seamGuide.visible = state.geom === "wall2x2" && state.showSeamGrid;
  });
  dom.btnSnapshot.addEventListener("click", saveViewportSnapshot);
  dom.btnResetCam.addEventListener("click", resetCamera);

  // Double-click canvas to reset camera framing
  dom.glCanvas.addEventListener("dblclick", resetCamera);

  // Interactive Shift + Drag studio light & HDRI orbit (Marmoset/Substance-style)
  let lightDragging = false;
  let lightDragStartX = 0;
  let lightDragStartY = 0;
  let lightStartAz = 38;
  let lightStartEl = 14;
  let lightHudTimer = null;

  function showLightHud() {
    if (!dom.lightHudPill) return;
    dom.lightHudPill.textContent = `KEY ${Math.round(state.azimuthDeg)}° · ELEV ${Math.round(state.elevationDeg)}°`;
    dom.lightHudPill.classList.remove("hidden");
    clearTimeout(lightHudTimer);
    if (!lightDragging) {
      lightHudTimer = setTimeout(() => dom.lightHudPill.classList.add("hidden"), 900);
    }
  }

  dom.glCanvas.addEventListener("pointerdown", (e) => {
    if (e.shiftKey && e.button === 0) {
      lightDragging = true;
      controls.enabled = false;
      lightDragStartX = e.clientX;
      lightDragStartY = e.clientY;
      lightStartAz = state.azimuthDeg;
      lightStartEl = state.elevationDeg;
      dom.glCanvas.classList.add("light-orbiting");
      dom.glCanvas.setPointerCapture(e.pointerId);
      showLightHud();
      e.preventDefault();
      e.stopPropagation();
    }
  }, true);

  dom.glCanvas.addEventListener("pointermove", (e) => {
    if (!lightDragging) return;
    const dx = e.clientX - lightDragStartX;
    const dy = e.clientY - lightDragStartY;
    state.azimuthDeg = ((lightStartAz + dx * 0.55) % 360 + 360) % 360;
    state.elevationDeg = Math.max(6, Math.min(78, lightStartEl - dy * 0.35));
    $("slAzimuth").value = String(Math.round(state.azimuthDeg));
    $("valAzimuth").textContent = `${Math.round(state.azimuthDeg)}°`;
    $("slElevation").value = String(Math.round(state.elevationDeg));
    $("valElevation").textContent = `${Math.round(state.elevationDeg)}°`;
    updateLightAzimuth();
    showLightHud();
  });

  function stopLightDrag(e) {
    if (!lightDragging) return;
    lightDragging = false;
    controls.enabled = true;
    dom.glCanvas.classList.remove("light-orbiting");
    try { dom.glCanvas.releasePointerCapture(e.pointerId); } catch {}
    showLightHud();
  }
  dom.glCanvas.addEventListener("pointerup", stopLightDrag);
  dom.glCanvas.addEventListener("pointercancel", stopLightDrag);

  // Channel Inspector Bar
  dom.channelBar.addEventListener("click", (e) => {
    const btn = e.target.closest(".chan-btn[data-channel]");
    if (btn) setChannel(btn.dataset.channel);
  });
  dom.chkGrid2x2.addEventListener("change", renderDiagnosticGrid);
  dom.btnCloseGrid.addEventListener("click", () => setChannel("pbr"));
  dom.btnLightboxClose.addEventListener("click", () => dom.mapLightbox.classList.add("hidden"));
  dom.btnLightbox3D.addEventListener("click", () => {
    if (state.lightboxKey) setChannel(state.lightboxKey);
  });

  // Split A/B + Hold Fresh
  dom.btnSplitAB.addEventListener("click", toggleSplitAB);
  dom.btnHoldFresh.addEventListener("pointerdown", () => beginHoldFresh());
  window.addEventListener("pointerup", () => endHoldFresh());
  dom.btnHoldFresh.addEventListener("pointerleave", () => endHoldFresh());

  let draggingSplit = false;
  dom.splitHandle.addEventListener("pointerdown", (e) => {
    draggingSplit = true;
    dom.splitHandle.setPointerCapture(e.pointerId);
    e.stopPropagation();
  });
  dom.splitHandle.addEventListener("pointermove", (e) => {
    if (!draggingSplit) return;
    const rect = dom.viewportWrap.getBoundingClientRect();
    state.splitFraction = Math.max(0.05, Math.min(0.95, (e.clientX - rect.left) / rect.width));
    updateSplitHandleUI();
  });
  dom.splitHandle.addEventListener("pointerup", () => {
    draggingSplit = false;
  });

  // Right sidebar tabs
  document.querySelectorAll(".rtab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.rtab;
      document.querySelectorAll(".rtab-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.rtab === target);
      });
      $("rtabWeather").classList.toggle("active", target === "weather");
      $("rtabBlend").classList.toggle("active", target === "blend");
      $("rtabAtlas").classList.toggle("active", target === "atlas");
    });
  });

  // Weathering sliders
  document.querySelectorAll("#weatherSliders input[type=range]").forEach((sl) => {
    sl.addEventListener("input", () => {
      const key = sl.dataset.wkey;
      const val = parseFloat(sl.value);
      state.weather[key] = val;
      syncWeatherSliderDOM();
      // Mark preset as custom if it no longer matches
      state.activePreset = isAnyWeatherActive() ? null : "fresh";
      dom.weatherPresets.querySelectorAll(".preset-pill").forEach((el) => {
        el.classList.toggle("active", el.dataset.preset === state.activePreset);
      });
      scheduleAutoSynth();
    });
  });

  dom.btnResetWeather.addEventListener("click", () => applyWeatherPreset("fresh"));
  dom.chkAutoSynth.addEventListener("change", () => {
    state.autoSynth = dom.chkAutoSynth.checked;
  });

  dom.inputSeed.addEventListener("change", () => {
    const raw = dom.inputSeed.value.trim();
    state.seed = raw === "" ? null : Math.max(0, Math.min(999999, parseInt(raw, 10) || 0));
    if (state.autoSynth) synthesizeCurrent();
  });

  dom.btnRandSeed.addEventListener("click", () => {
    const s = Math.floor(Math.random() * 99999) + 1;
    state.seed = s;
    dom.inputSeed.value = String(s);
    synthesizeCurrent();
  });

  dom.btnApplySynth.addEventListener("click", () => synthesizeCurrent());

  // Surface & Lighting real-time sliders
  $("slRepeat").addEventListener("input", (e) => {
    state.userTouchedRepeat = true;
    state.repeat = parseFloat(e.target.value);
    $("valRepeat").textContent = `${state.repeat.toFixed(1)}×`;
    syncAllTextureRepeats();
  });
  $("slNormal").addEventListener("input", (e) => {
    state.normalScale = parseFloat(e.target.value);
    $("valNormal").textContent = state.normalScale.toFixed(2);
    updateMaterialScalars();
  });
  $("slDisp").addEventListener("input", (e) => {
    state.displacement = parseFloat(e.target.value);
    $("valDisp").textContent = state.displacement.toFixed(3);
    updateMaterialScalars();
  });
  $("slRough").addEventListener("input", (e) => {
    state.roughMult = parseFloat(e.target.value);
    $("valRough").textContent = `${state.roughMult.toFixed(2)}×`;
    updateMaterialScalars();
  });
  $("slMetal").addEventListener("input", (e) => {
    state.metalScale = parseFloat(e.target.value);
    $("valMetal").textContent = `${state.metalScale.toFixed(2)}×`;
    updateMaterialScalars();
  });
  $("slAzimuth").addEventListener("input", (e) => {
    state.azimuthDeg = parseFloat(e.target.value);
    $("valAzimuth").textContent = `${Math.round(state.azimuthDeg)}°`;
    updateLightAzimuth();
  });
  $("slElevation").addEventListener("input", (e) => {
    state.elevationDeg = parseFloat(e.target.value);
    $("valElevation").textContent = `${Math.round(state.elevationDeg)}°`;
    updateLightAzimuth();
  });
  $("slIbl").addEventListener("input", (e) => {
    state.iblMult = parseFloat(e.target.value);
    $("valIbl").textContent = `${state.iblMult.toFixed(2)}×`;
    const rig = LIGHT_RIGS[state.lightRig] || LIGHT_RIGS.raking;
    scene.environmentIntensity = rig.envIntensity * state.iblMult;
  });
  $("slSpecAA").addEventListener("input", (e) => {
    state.specAA = parseFloat(e.target.value);
    $("valSpecAA").textContent = `${state.specAA.toFixed(2)}×`;
    specAAUniform.value = state.specAA;
  });
  $("slExposure").addEventListener("input", (e) => {
    state.exposure = parseFloat(e.target.value);
    $("valExposure").textContent = state.exposure.toFixed(2);
    renderer.toneMappingExposure = state.exposure;
  });

  dom.btnResetSurface.addEventListener("click", () => {
    const rec = state.selectedId
      ? getRecommendedStage(state.catalogById.get(state.selectedId))
      : { geom: "sphere", repeat: 4.0, lightRig: "raking" };
    const rig = LIGHT_RIGS[rec.lightRig || state.lightRig] || LIGHT_RIGS.raking;
    state.userTouchedRepeat = false;
    state.repeat = state.autoStage ? rec.repeat : (GEOM_DEFAULT_REPEAT[state.geom] ?? 4.0);
    state.normalScale = 1.0;
    state.displacement = 0.012;
    state.roughMult = 1.0;
    state.metalScale = 1.0;
    state.azimuthDeg = 38;
    state.elevationDeg = rig.keyElevDeg ?? 14;
    state.iblMult = 1.0;
    state.specAA = 1.0;
    state.exposure = 1.08;
    $("slRepeat").value = String(state.repeat); $("valRepeat").textContent = `${state.repeat.toFixed(1)}×`;
    $("slNormal").value = "1"; $("valNormal").textContent = "1.00";
    $("slDisp").value = "0.012"; $("valDisp").textContent = "0.012";
    $("slRough").value = "1"; $("valRough").textContent = "1.00×";
    $("slMetal").value = "1"; $("valMetal").textContent = "1.00×";
    $("slAzimuth").value = "38"; $("valAzimuth").textContent = "38°";
    $("slElevation").value = String(state.elevationDeg); $("valElevation").textContent = `${Math.round(state.elevationDeg)}°`;
    $("slIbl").value = "1"; $("valIbl").textContent = "1.00×";
    $("slSpecAA").value = "1"; $("valSpecAA").textContent = "1.00×";
    $("slExposure").value = "1.08"; $("valExposure").textContent = "1.08";
    renderer.toneMappingExposure = state.exposure;
    if (state.autoStage) {
      applyRecommendedStage(state.catalogById.get(state.selectedId));
    } else {
      applyLightingRig();
      syncAllTextureRepeats();
    }
    updateMaterialScalars();
  });

  // Blend Controls
  dom.blendModeSeg.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn[data-bmode]");
    if (!btn) return;
    state.blendConfig.mode = btn.dataset.bmode;
    dom.blendModeSeg.querySelectorAll(".seg-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.bmode === state.blendConfig.mode);
    });
  });
  dom.slBlendFactor.addEventListener("input", (e) => {
    state.blendConfig.factor = parseFloat(e.target.value);
    dom.valBlendFactor.textContent = state.blendConfig.factor.toFixed(2);
  });
  dom.slBlendContrast.addEventListener("input", (e) => {
    state.blendConfig.contrast = parseFloat(e.target.value);
    dom.valBlendContrast.textContent = state.blendConfig.contrast.toFixed(2);
  });
  dom.btnRunBlend.addEventListener("click", runBlendSynthesis);
  dom.btnExitBlend.addEventListener("click", () => {
    state.blendModeActive = false;
    dom.btnExitBlend.classList.add("hidden");
    synthesizeCurrent();
  });

  // Copy Cache Path
  dom.btnCopyPath.addEventListener("click", async () => {
    const p = state.currentResult?.cache_dir;
    if (!p) return;
    try {
      await navigator.clipboard.writeText(p);
      const orig = dom.btnCopyPath.textContent;
      dom.btnCopyPath.textContent = "Copied!";
      setTimeout(() => { dom.btnCopyPath.textContent = orig; }, 1200);
    } catch {
      window.prompt("Cache folder path:", p);
    }
  });

  // Global Keyboard Navigation (Up/Down catalog flip, Space hold fresh, / search, Esc close)
  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName;
    const isTyping = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";

    if (e.key === "/" && !isTyping) {
      e.preventDefault();
      dom.catalogSearch.focus();
      dom.catalogSearch.select();
      return;
    }

    if (e.key === "Escape") {
      if (!dom.mapLightbox.classList.contains("hidden")) {
        dom.mapLightbox.classList.add("hidden");
      } else if (state.channel === "grid2d") {
        setChannel("pbr");
      } else if (document.activeElement === dom.catalogSearch) {
        dom.catalogSearch.blur();
      }
      return;
    }

    if (isTyping) return;

    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (state.filtered.length === 0) return;
      e.preventDefault();
      const curIdx = state.filtered.findIndex((m) => m.id === state.selectedId);
      const delta = e.key === "ArrowDown" ? 1 : -1;
      const nextIdx = curIdx < 0
        ? 0
        : (curIdx + delta + state.filtered.length) % state.filtered.length;
      selectMaterial(state.filtered[nextIdx].id, true);
      return;
    }

    if (e.code === "Space" && !e.repeat) {
      e.preventDefault();
      beginHoldFresh();
      return;
    }

    const geomKeys = ["sphere", "box", "knot", "cylinder", "plane", "wall2x2"];
    if (e.key >= "1" && e.key <= "6" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      setGeometry(geomKeys[Number(e.key) - 1]);
      return;
    }

    if ((e.key === "r" || e.key === "R") && !e.metaKey && !e.ctrlKey) {
      state.autoRotate = !state.autoRotate;
      dom.btnAutoRotate.classList.toggle("active", state.autoRotate);
      return;
    }

    if ((e.key === "l" || e.key === "L") && !e.metaKey && !e.ctrlKey) {
      const rigOrder = ["raking", "studio", "golden", "overcast", "darkrim"];
      const next = rigOrder[(rigOrder.indexOf(state.lightRig) + 1) % rigOrder.length];
      state.lightRig = next;
      dom.lightRigSelect.value = next;
      applyLightingRig();
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
      endHoldFresh();
    }
  });
}

// ============================================================================
// Atlas Inspector (agent metadata for the selected material)
// ============================================================================
function showToast(msg) {
  if (!dom.toast) return;
  dom.toast.textContent = msg;
  dom.toast.classList.add("show");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => dom.toast.classList.remove("show"), 1600);
}

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    showToast(`Copied ${label}`);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); showToast(`Copied ${label}`); } catch { showToast("Copy failed"); }
    ta.remove();
  }
}

async function fetchSpec(matId) {
  if (state.specCache.has(matId)) return state.specCache.get(matId);
  const res = await fetch(`./atlas/mats/${encodeURIComponent(matId)}.json`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`spec ${res.status}`);
  const spec = await res.json();
  state.specCache.set(matId, spec);
  return spec;
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function chipRow(items, cls) {
  const row = el("div", "chip-row");
  for (const it of items) row.appendChild(el("span", `chip ${cls || ""}`, String(it).replace(/_/g, " ")));
  return row;
}

function atlasSection(title, child) {
  const sec = el("section", "ctrl-section atlas-sec");
  const head = el("div", "section-head");
  head.appendChild(el("span", "panel-title", title));
  sec.appendChild(head);
  if (child) sec.appendChild(child);
  return sec;
}

function applyRecommendedWeathering(spec) {
  const hp = spec?.recommended_weathering?.hypbr_params;
  if (!hp) return;
  for (const k of WEATHER_KEYS) state.weather[k] = Number(hp[k] || 0);
  state.activePreset = null;
  syncWeatherSliderDOM();
  dom.weatherPresets.querySelectorAll(".preset-pill").forEach((p) => p.classList.remove("active"));
  synthesizeCurrent();
  showToast(`Applied recommended weathering (${spec.recommended_weathering.preset})`);
}

function openBlendWithPartner(partnerId) {
  if (!state.catalogById.has(partnerId)) return;
  dom.blendMatBSelect.value = partnerId;
  dom.blendMatBSelect.dispatchEvent(new Event("change"));
  state.blendConfig.id_b = partnerId;
  document.querySelector('.rtab-btn[data-rtab="blend"]')?.click();
  showToast(`Blend slot B: ${state.catalogById.get(partnerId).name}`);
}

function renderAtlasInspector(spec) {
  const body = dom.atlasBody;
  if (!body) return;
  body.innerHTML = "";
  if (!spec) {
    dom.atlasEmpty?.classList.remove("hidden");
    return;
  }
  dom.atlasEmpty?.classList.add("hidden");
  const [w, h] = spec.real_world_size_m;
  const pj = spec.projection;
  const ss = spec.shader_setup;

  // Header summary
  const head = el("section", "ctrl-section atlas-head");
  const title = el("div", "atlas-title", spec.name);
  const sub = el("div", "mono-dim", `${spec.category.replace(/_/g, " ")} · ${spec.subcategory.replace(/_/g, " ")} · ${spec.generator.kind}`);
  const desc = el("p", "helper-copy", spec.description);
  const stats = el("div", "atlas-stats");
  const stat = (k, v) => {
    const d = el("div", "atlas-stat");
    d.append(el("span", "qa-k", k), el("span", "qa-v", v));
    return d;
  };
  stats.append(
    stat("Tile size", `${w} × ${h} m`),
    stat("Projection", pj.recommended + (pj.triplanar_fallback ? " · tri ok" : "")),
    stat("Grain axis", pj.anisotropic_axis),
    stat("Repeat on 4×2.7 m", `${(4 / w).toFixed(2)} × ${(2.7 / h).toFixed(2)}`),
  );
  head.append(title, sub, desc, stats);
  body.appendChild(head);

  // Objects
  const objWrap = el("div");
  objWrap.append(el("div", "field-lbl", "Best on"), chipRow(spec.best_objects, "chip-good"),
    el("div", "field-lbl", "Avoid on"), chipRow(spec.avoid_objects, "chip-bad"),
    el("div", "field-lbl", "Scene roles"), chipRow(spec.scene_roles, ""));
  body.appendChild(atlasSection("OBJECTS & ROLES", objWrap));

  // UV / projection
  const uv = el("div");
  uv.append(el("code", "atlas-code", pj.uv_repeat_formula), el("p", "helper-copy", pj.tiling_note));
  if (pj.requires_tangents) uv.appendChild(el("p", "helper-copy mono-dim", "Directional: generate MikkTSpace tangents / set anisotropy rotation."));
  body.appendChild(atlasSection("UV & PROJECTION", uv));

  // Shader table
  const tbl = el("table", "atlas-table");
  const rows = [
    ["albedo_delit", `${ss.albedo_colorspace}`],
    ["normal", `${ss.normal_space} · ${ss.data_colorspace}`],
    ["orm", `${ss.orm_packing}`],
    ["aux", `${ss.aux_packing}`],
    ["displacement", `${ss.displacement_scale_m} m`],
    ["ior", `${ss.ior}${ss.transmission ? ` · transmission ${ss.transmission}` : ""}`],
    ["roughness", `${ss.roughness_range[0]} – ${ss.roughness_range[1]}${ss.ranges_measured ? " (measured)" : ""}`],
    ["metallic", `${(ss.metallic_range || [0, 0]).join(" – ")}${ss.is_metallic ? " · metal" : " · dielectric"}`],
  ];
  for (const [k, v] of rows) {
    const tr = el("tr");
    tr.append(el("td", "atlas-k", k), el("td", "atlas-v", v));
    tbl.appendChild(tr);
  }
  body.appendChild(atlasSection("SHADER SETUP", tbl));

  // Weathering
  const wx = spec.recommended_weathering;
  const ww = el("div");
  const bars = el("div", "atlas-weather");
  for (const k of ["wear", "dirt", "moisture", "moss", "rust", "frost"]) {
    const r = el("div", "aw-row");
    const bar = el("div", "aw-bar");
    const fill = el("div", "aw-fill");
    fill.style.width = `${Math.round((wx[k] || 0) * 100)}%`;
    bar.appendChild(fill);
    r.append(el("span", "aw-k", k), bar, el("span", "mono-val", Number(wx[k] || 0).toFixed(2)));
    bars.appendChild(r);
  }
  const applyBtn = el("button", "primary-btn atlas-apply", `Apply Recommended Weathering · ${wx.preset}`);
  applyBtn.type = "button";
  applyBtn.addEventListener("click", () => applyRecommendedWeathering(spec));
  ww.append(bars, el("p", "helper-copy", wx.note), applyBtn);
  body.appendChild(atlasSection("RECOMMENDED WEATHERING", ww));

  // Pairs
  const pw = el("div", "pair-list");
  for (const pid of spec.pairs_well_with || []) {
    const item = state.catalogById.get(pid);
    if (!item) continue;
    const row = el("div", "pair-row");
    const main = el("button", "pair-main");
    main.type = "button";
    main.title = `Load ${item.name}`;
    if (item.preview_url) {
      const img = document.createElement("img");
      img.src = item.preview_url;
      img.alt = "";
      img.loading = "lazy";
      main.appendChild(img);
    }
    main.appendChild(el("span", "pair-name", item.name));
    main.addEventListener("click", () => selectMaterial(pid, true));
    const blendBtn = el("button", "pair-blend", "Blend B");
    blendBtn.type = "button";
    blendBtn.title = "Open in HeightLerp Blend slot B";
    blendBtn.addEventListener("click", () => openBlendWithPartner(pid));
    row.append(main, blendBtn);
    pw.appendChild(row);
  }
  body.appendChild(atlasSection("PAIRS WELL WITH", pw));

  // Copy actions
  const acts = el("div", "atlas-actions");
  const mk = (label, fn) => {
    const b = el("button", "secondary-btn", label);
    b.type = "button";
    b.addEventListener("click", fn);
    return b;
  };
  acts.append(
    mk("Copy Agent Spec JSON", () => {
      const { snippets, ...lean } = spec;
      copyText(JSON.stringify(lean, null, 2), "agent spec JSON");
    }),
    mk("Copy Three.js Snippet", () => copyText(spec.snippets?.threejs || "", "Three.js snippet")),
    mk("Copy Blender bpy Snippet", () => copyText(spec.snippets?.blender_bpy || "", "Blender bpy snippet")),
    mk("Open Spec URL ↗", () => window.open(spec.spec_url || `./atlas/mats/${spec.id}.json`, "_blank", "noopener")),
  );
  body.appendChild(atlasSection("AGENT EXPORT", acts));
}

async function loadAtlasInspector(matId) {
  try {
    const spec = await fetchSpec(matId);
    if (state.selectedId !== matId) return;
    state.atlasSpec = spec;
    renderAtlasInspector(spec);
  } catch (err) {
    console.warn("Atlas spec unavailable", err);
    state.atlasSpec = null;
    renderAtlasInspector(null);
  }
}

function atlasBaseUrl() {
  const h = window.location.hostname;
  return ["localhost", "127.0.0.1", "[::1]"].includes(h) ? ATLAS_PUBLIC_URL : window.location.origin;
}

function renderAgentModal() {
  const base = atlasBaseUrl();
  const body = dom.agentModalBody;
  body.innerHTML = "";
  const links = [
    ["llms.txt", `${base}/llms.txt`, "LLM entrypoint (start here)"],
    ["agent.md", `${base}/agent.md`, "Full integration guide"],
    ["atlas.json", `${base}/atlas.json`, "Manifest, schema & URL templates"],
    ["atlas/index.json", `${base}/atlas/index.json`, `Compact index of all ${state.catalog.length} materials`],
    ["spec card", `${base}/atlas/mats/${state.selectedId || "classic_red_brick"}.json`, "Current material spec card"],
  ];
  for (const [label, url, hint] of links) {
    const row = el("div", "agent-link");
    const info = el("div", "agent-link-info");
    const a = el("a", "agent-url", url);
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener";
    info.append(el("span", "agent-label", label), a, el("span", "mono-dim", hint));
    const b = el("button", "secondary-btn", "Copy");
    b.type = "button";
    b.addEventListener("click", () => copyText(url, label));
    row.append(info, b);
    body.appendChild(row);
  }
  const prompt = `You have access to the hypbr PBR Material Atlas: ${state.catalog.length || 500} seamless, physically calibrated PBR materials.\n` +
    `1. Read ${base}/llms.txt (then ${base}/agent.md for details).\n` +
    `2. Find materials in ${base}/atlas/index.json by best_objects, scene_roles and tags; avoid any whose avoid_objects match.\n` +
    `3. Fetch ${base}/atlas/mats/<id>.json and obey real_world_size_m (repeat = surface_m / size_m), projection, shader_setup (sRGB albedo, OpenGL +Y normal, ORM = R AO / G roughness / B metallic) and recommended_weathering.\n` +
    `4. Load textures from ${base}/assets/mats/<id>/{albedo_delit,normal,orm,aux}.webp (CORS enabled).`;
  const pre = el("pre", "agent-prompt", prompt);
  const copyPrompt = el("button", "primary-btn", "Copy System Prompt");
  copyPrompt.type = "button";
  copyPrompt.addEventListener("click", () => copyText(prompt, "system prompt"));
  const sec = el("div", "agent-prompt-wrap");
  sec.append(el("div", "panel-title", "SYSTEM PROMPT SNIPPET"), pre, copyPrompt);
  body.appendChild(sec);
}

function openAgentModal() {
  renderAgentModal();
  dom.agentModal.classList.remove("hidden");
  dom.agentModal.setAttribute("aria-hidden", "false");
  dom.btnAgentClose?.focus();
}

function closeAgentModal() {
  dom.agentModal.classList.add("hidden");
  dom.agentModal.setAttribute("aria-hidden", "true");
}

dom.btnAgentAtlas?.addEventListener("click", openAgentModal);
dom.btnAgentClose?.addEventListener("click", closeAgentModal);
dom.agentModal?.addEventListener("click", (e) => {
  if (e.target === dom.agentModal) closeAgentModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && dom.agentModal && !dom.agentModal.classList.contains("hidden")) closeAgentModal();
});

// ============================================================================
// Bootstrap
// ============================================================================
applyLightingRig();
wireEvents();
fetchCatalog().catch((err) => {
  console.error(err);
  setStatus(`Failed to load catalog: ${err.message}`, true);
});
