# Cubica

A lightweight, zero-dependency **WebGPU voxel path tracer** for the browser. Write a voxel scene in a few lines of JavaScript and get physically-based lighting, glass, emission, soft shadows, and progressive accumulation — all running on the GPU, with an optional orbit camera controller.

Designed for generative art and interactive 3D experiments. Interesting options supported are the ability to create gaps between voxels as well as supporting curved voxel edges, giving a more organic look compared to typical voxel apps.

Inspiration comes from [vixel](https://github.com/wwwtyro/vixel), a WebGL voxel path tracer.

Vibe-coded using Google Antigravity (Gemini 3.1 Pro and Claude Sonnet 4.6).

---

## What it does

Cubica renders a 3-D grid of coloured voxels using a **Monte Carlo path tracer** written in WGSL (WebGPU Shading Language). Each frame, the GPU traces rays from your camera into the scene, bouncing them around to gather light physically. The result accumulates over many frames into a noise-free image.

Because every pixel is computed in parallel on the GPU and the algorithm is physically based, you get:

- Soft shadows and area lighting for free
- Global illumination (indirect bounce light)
- Correct glass refraction and metallic reflection
- Emissive voxels that act as light sources
- Progressive refinement — it keeps getting better the longer you let it run

---

## Technical stack

| Layer | Technology |
|-------|-----------|
| Renderer | WebGPU compute shader (WGSL) |
| Algorithm | Monte Carlo path tracing with Next Event Estimation |
| Traversal | DDA (Amanatides & Woo 1987) |
| Sampling | PCG hash PRNG seeded with Interleaved Gradient Noise (IGN) |
| Hemisphere sampling | Cosine-weighted (importance sampling) |
| Tone mapping | ACES filmic + gamma correction (2.2) |
| Accumulation | Ping-pong `rgba16float` textures |
| Host language | Vanilla ES module JavaScript |
| Dependencies | **None** |

---

## Browser requirements

WebGPU is required. As of 2026 this means:

- **Chrome 113+** (desktop, enabled by default)
- **Edge 113+**
- **Safari 18+** (macOS/iOS)
- Firefox: behind a flag (`dom.webgpu.enabled`)

The library throws a descriptive `Error` if WebGPU is unavailable.

---

## Quick start

```html
<canvas id="canvas"></canvas>
<script type="module" src="demo.js"></script>
```

```js
// demo.js
import { Cubica, CubicaOrbitControls } from './cubica.js';

const canvas = document.getElementsByTagName('canvas')[0];
canvas.width  = window.innerWidth;
canvas.height = window.innerHeight;

const cubica = new Cubica();

await cubica.init({
    canvas,
    dimensions: [64, 64, 64],
    env: {
        sunDirection:  [0.6, 1.0, 0.4],
        sunColor:      [255, 210, 180],
        sunIntensity:  2.0,
        ambientColor:  [30, 40, 60],
        ambientIntensity: 0.4,
        maxBounces:    4
    }
});

// --- Define materials ---
const stone = cubica.setMaterial(160, 155, 140, 0.9, 1.0, 0.0);
const concrete = cubica.setMaterial(160, 165, 175, 0.8, 1.0, 0.0);

// --- Build a scene ---
cubica.currentMaterial = stone;
for (let x = 0; x < 64; x++)
    for (let z = 0; z < 64; z++)
        cubica.set(x, 0, z);

cubica.currentMaterial = concrete;
for (let i = 0; i < 1000; i++) {
    const x = Math.floor(Math.random() * 64);
    const y = Math.floor(Math.random() * 20);
    const z = Math.floor(Math.random() * 64);
    cubica.set(x, y, z);
}

cubica.updateBuffers();

// --- Camera & controls ---
const controls = new CubicaOrbitControls(cubica, {
    radius: 80,
    target: [32, 0, 32],
});

// --- Render loop ---
function frame() {
    controls.update();
    cubica.trace(controls.isDragging ? 1 : 4);
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

---

## API Reference

### `Cubica`

The main class. Import it as an ES module.

```js
import { Cubica } from './cubica.js';
```

---

#### `new Cubica()`

Creates a new cubica instance. All configuration is passed via `init({ env })`.

---

#### `cubica.init(options)` → `Promise<void>`

Initialises WebGPU, compiles the WGSL shader, and allocates all GPU buffers. Must be `await`-ed before calling any other method.

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `canvas` | `HTMLCanvasElement` | ✅ | The canvas to render into |
| `dimensions` | `[x, y, z]` | ✅ | Voxel grid size. Larger grids cost more memory and DDA traversal time. 128³ is a practical upper bound for interactive framerates |
| `env` | `object` | — | Initial environment settings (see [Environment](#environment-options)) |

> **Note:** `maxBounces` and `cornerRadius` are baked into the shader at compile time. They cannot be changed after `init()` without re-initialising.

---

#### `cubica.setMaterial(r, g, b, roughness, ior, emission)` → `number`

Registers a material and returns its **material ID**. Up to 255 materials (+ material 0 = air).

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `r, g, b` | `number` | `0–255` | Base colour |
| `roughness` | `number` | `0.0–1.0` | `0` = perfect mirror, `1` = fully diffuse (Lambertian) |
| `ior` | `number` | `1.0–∞` | Index of refraction. `1.0` = opaque, `1.33` = water, `1.5` = glass. Values above `1.05` trigger the refractive code path |
| `emission` | `number` | `0.0–∞` | Emission multiplier. `0` = no emission. `4–8` gives a strong neon glow |

Returns the integer material ID. Store this to use with `cubica.currentMaterial`.

```js
const matGlass  = cubica.setMaterial(220, 240, 255,  0.0, 1.5, 0.0);
const matMirror = cubica.setMaterial(220, 220, 230,  0.0, 1.0, 0.0);
const matGlow   = cubica.setMaterial(255,  80, 120,  0.5, 1.0, 8.0);
```

---

#### `cubica.currentMaterial`

`number` — The active material ID. Set this before calling `set()`.

```js
cubica.currentMaterial = matGlass;
cubica.set(10, 5, 10);
```

---

#### `cubica.set(x, y, z, gapSize?)` 

Places a voxel at grid position `(x, y, z)` using `currentMaterial`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x, y, z` | `number` | — | Integer grid coordinates |
| `gapSize` | `number` | `0.0` | `0.0` = flush/solid fill. `0.05–0.3` shrinks the voxel geometry, creating a gap between adjacent voxels (useful for floating cubes, windows, decorative detail). Value is clamped to `[0, 1]` |

Out-of-bounds coordinates are silently ignored.

---

#### `cubica.remove(x, y, z)`

Sets a voxel back to air (material 0).

---

#### `cubica.updateBuffers()`

Uploads the current voxel grid and material table to the GPU. **Must be called** after building or modifying the scene, before `trace()`.

Also resets the accumulation buffer (starts re-rendering from scratch).

---

#### `cubica.camera(options)`

Positions the camera. Called automatically by `CubicaOrbitControls.update()`.

| Option | Type | Description |
|--------|------|-------------|
| `position` | `[x, y, z]` | World-space camera position |
| `target` | `[x, y, z]` | World-space look-at point |
| `fov` | `number` (optional) | Override `env.fov` for this call only |

Returns `true` if the camera moved (accumulation will be reset automatically).

```js
cubica.camera({
    position: [100, 60, 100],
    target:   [64,  10, 64],
});
```

---

#### `cubica.setEnvironment(opts)`

Updates environment lighting at runtime. Merges with existing settings. Resets accumulation.

```js
cubica.setEnvironment({
    sunIntensity: 5.0,
    sunDirection: [1.0, 0.5, 0.0],
});
```

See [Environment options](#environment-options) for all fields.

---

#### `cubica.trace(batchSize?)`

Runs one render step: dispatches the compute shader for `batchSize` samples per pixel, then blits the accumulated result to the canvas.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batchSize` | `number` | `env.batchSize` | Samples per pixel per call. Higher = faster convergence but longer GPU frame time. `1` during interaction, `4–8` at rest is a common pattern |

Call this every animation frame.

```js
function frame() {
    cubica.trace(controls.isDragging ? 1 : 4);
    requestAnimationFrame(frame);
}
```

---

#### `cubica.clear()`

Resets the accumulated sample count and flips the ping-pong index. The next `trace()` will start accumulation fresh. Called automatically when the camera moves or the environment changes.

---

#### `cubica.sampleCount`

`number` — Total samples accumulated so far. Useful for UI display.

```js
document.getElementById('samples').textContent = `Samples: ${cubica.sampleCount}`;
```

---

#### `cubica.dispose()`

Destroys all GPU resources (buffers, textures, device). Call when the tracer is no longer needed, e.g. when navigating away in a SPA.

---

### Environment options

Passed to `init({ env })` or `setEnvironment()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fov` | `number` | `50` | Vertical field of view in degrees |
| `ambientColor` | `[r, g, b]` | `[0, 0, 0]` | Sky / ambient light colour (0–255 per channel) |
| `ambientIntensity` | `number` | `0.0` | Brightness of the ambient sky gradient |
| `sunDirection` | `[x, y, z]` | `[0.5, 1.0, 0.2]` | World-space sun direction vector (does not need to be normalised) |
| `sunColor` | `[r, g, b]` | `[255, 255, 230]` | Sun disc colour |
| `sunIntensity` | `number` | `0.0` | Sun brightness. `0` disables the sun and all shadow rays entirely. `2–5` gives natural outdoor lighting |
| `sunSize` | `number` | `0.99` | Angular size of the sun disc. `0.999` = small crisp disc with sharp soft-shadows. `0.9` = very large area light with very soft shadows |
| `maxBounces` | `number` | `4` | Maximum ray bounces per path. **Baked into the shader at init** — cannot be changed at runtime |
| `cornerRadius` | `number` | `0.0` | Fake surface curvature at voxel edges via normal bending. `0` = hard cube edges, `0.5` = very rounded. **Baked into the shader at init** |
| `batchSize` | `number` | `4` | Default samples per pixel per `trace()` call |
| `focusDistance` | `number` | `100` | World-space distance from the camera to the perfectly sharp focal plane. Objects closer or further will be blurred. Has no effect when `aperture` is `0` |
| `aperture` | `number` | `0.0` | Radius of the virtual lens disc. `0` = pinhole camera (no blur, default). `0.5–3.0` gives a shallow depth-of-field bokeh effect. Larger values blur more aggressively |

---

### `CubicaOrbitControls`

An optional but ready-to-use orbit camera controller. Handles mouse and touch input.

```js
import { CubicaOrbitControls } from './cubica.js';
```

#### `new CubicaOrbitControls(cubica, options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `radius` | `number` | `150` | Initial distance from the target |
| `target` | `[x, y, z]` | `[0, 0, 0]` | Point the camera orbits around |
| `theta` | `number` (radians) | `π/4` | Initial horizontal angle |
| `phi` | `number` (radians) | `π/3` | Initial vertical angle (`0` = top-down, `π/2` = equator) |
| `rotateSpeed` | `number` | `0.005` | Mouse/touch rotation sensitivity |
| `panSpeed` | `number` | `0.002` | Right-click/two-finger pan speed (scales with `radius`) |
| `zoomSpeed` | `number` | `0.1` | Scroll wheel / pinch zoom speed |

**Controls:**

| Input | Action |
|-------|--------|
| Left-drag | Orbit |
| Right-drag | Pan |
| Scroll wheel | Zoom |
| One-finger drag | Orbit (touch) |
| Two-finger drag | Pan (touch) |
| Pinch | Zoom (touch) |

---

#### `controls.update()`

Recomputes the camera position from `theta`/`phi`/`radius`/`target` and calls `cubica.camera()`. Call once per animation frame before `cubica.trace()`.

---

#### `controls.isDragging`

`boolean` — `true` while a mouse button or touch is held. Useful for reducing `batchSize` during interaction to maintain responsiveness.

---

#### `controls.dispose()`

Removes all event listeners (uses `AbortController` internally). Call when tearing down the renderer.

---

## The material model

Every material has three behaviours, selected by its parameters:

| Behaviour | How to set it |
|-----------|---------------|
| **Diffuse / matte** | `roughness = 1.0`, `ior = 1.0`, `emission = 0` |
| **Metallic / glossy** | `roughness = 0.0–0.3`, `ior = 1.0`, `emission = 0`. Lower roughness = sharper reflection |
| **Glass / refractive** | `ior > 1.05` (e.g. `1.5`). `roughness` blends the specular lobe |
| **Emissive / light** | `emission > 0`. Acts as a light source. Colour × emission = emitted radiance |

Materials are fixed once `updateBuffers()` is called and persist until the next call.

---

## The voxel format

Internally each grid cell is a 32-bit unsigned integer:

```
[ 8 bits: gap size (0–255) | 8 bits: material ID (0–255) | 16 bits: reserved ]
```

- **Material 0** is always air (empty). You never define it.
- **Gap size** is the amount the voxel geometry shrinks inside its cell. `0` = flush/solid. Encoded as `gapSize * 255` and decoded in the shader.
- Access the raw grid via `cubica.gridData` (a `Uint32Array`) if you need to do bulk procedural generation before uploading.

---

## Performance tips

- **Keep `dimensions` as small as your art allows.** DDA traversal cost scales with the number of voxels traversed per ray, not the total count — but a larger grid means longer diagonal rays.
- **Use `batchSize = 1` while dragging.** The demo does this: `cubica.trace(controls.isDragging ? 1 : 4)`.
- **`sunIntensity = 0` disables shadow rays entirely.** If you only want emissive or ambient lighting, turning the sun off saves one full DDA traversal per diffuse bounce.
- **Emission is cheap.** Emissive voxels are hit-and-done; the path terminates immediately. Use them freely as neon lights and indicators.
- **`gapSize` triggers sub-voxel AABB intersection.** Flush voxels (`gapSize = 0`) use a faster code path. Use gap sizes only where visual needed.

---

## Renderer internals (for the curious)

The path tracer runs as a WebGPU **compute shader** dispatched at 8×8 workgroup size. Each invocation handles one pixel.

**Per frame:**
1. `trace()` writes updated camera/environment uniforms to a 144-byte uniform buffer.
2. The compute pass runs `batchSize` full paths per pixel. Each path bounces up to `maxBounces` times.
3. The resulting colour is blended into the previous accumulation texture (ping-pong between two `rgba16float` textures).
4. A fullscreen render pass reads the latest accumulation texture, applies ACES filmic tone mapping and γ 2.2 correction, and writes to the canvas swap chain.

**Notable details:**
- **PRNG:** PCG hash seeded per-pixel with Interleaved Gradient Noise (IGN, Jimenez 2016) for a blue-noise-like spatial distribution that reduces visible clumping at low sample counts.
- **Diffuse sampling:** Cosine-weighted hemisphere sampling (importance sampled toward the normal, so throughput correctly simplifies to `mat.color`).
- **Next Event Estimation:** Shadow rays are traced toward the sun on every diffuse bounce for efficient direct lighting.
- **Russian Roulette:** Paths with low throughput are terminated stochastically after bounce 3, saving computation on dark paths.
- **Normal bending:** The `cornerRadius` parameter fakes rounded cube edges by perturbing the geometric normal toward a spherical approximation on the face edges — purely optical, no geometry change.

---

## File structure

```
cubica.js      — Cubica and CubicaOrbitControls (ES module, no dependencies)
cubica.wgsl    — WGSL compute + vertex + fragment shaders
demo.html      — Minimal HTML entry point
demo.js        — Generative art demo (cyberpunk cityscape)
demo.jpg       — Screenshot of the demo
LICENSE.txt    — MIT License
README.md      — This file
```

## License

MIT license, refer license.txt