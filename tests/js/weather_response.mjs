// Response curves of the playground's client-side weathering (playground/static_engine.js).
// usage: node weather_response.mjs <case.json>   (planes written by tests/test_playground_weathering.py)
// Prints {channel: [effect at each level]} where effect = mean |albedo| + |roughness| change vs. an
// unweathered pass, in 0..1 units.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const { weatherBase, FADE_IN_KNEES } = await import(pathToFileURL(join(here, "..", "..", "playground", "static_engine.js")).href);

const spec = JSON.parse(readFileSync(process.argv[2], "utf8"));
const plane = (name) => new Uint8ClampedArray(readFileSync(join(dirname(process.argv[2]), `${name}.bin`)));
const base = {
  size: spec.size, albedoRGBA: plane("albedo"), normalRGBA: plane("normal"),
  ao: plane("ao"), rough: plane("rough"), metal: plane("metal"), height: plane("height"), curv: plane("curv"), cav: plane("cav"),
};
const zero = { wear: 0, dirt: 0, wet: 0, moss: 0, rust: 0, dust: 0 };
const ref = weatherBase(base, spec.entry, zero, spec.entry.default_seed);
const effect = (out) => {
  let a = 0, r = 0;
  for (let i = 0; i < out.outAlb.length; i++) if ((i & 3) !== 3) a += Math.abs(out.outAlb[i] - ref.outAlb[i]);
  for (let i = 0; i < out.outRough.length; i++) r += Math.abs(out.outRough[i] - ref.outRough[i]);
  return a / (255 * 3 * out.outRough.length) + r / (255 * out.outRough.length);
};
const result = {};
for (const ch of spec.channels) {
  result[ch] = spec.levels.map((lv) => effect(weatherBase(base, spec.entry, { ...zero, [ch]: lv }, spec.entry.default_seed)));
}
process.stdout.write(JSON.stringify({ curves: result, knees: FADE_IN_KNEES }));
