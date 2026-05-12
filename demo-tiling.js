import { Cubica, CubicaOrbitControls } from './cubica.js';

async function init() {
    const canvas = document.getElementsByTagName('canvas')[0];
    const statusEl = document.getElementById('status');
    const fpsEl = document.getElementById('fps');
    const samplesEl = document.getElementById('samples');

    if (!navigator.gpu) {
        statusEl.innerText = "WebGPU not supported on this browser.";
        return;
    }

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const cubica = new Cubica();
    await cubica.init({
        canvas: canvas,
        dimensions: [256, 128, 256],
        env: {
            fov: 60,
            ambientColor: [120, 160, 220],
            ambientIntensity: 0.4,
            sunDirection: [0.4, 1.0, -0.3],
            sunColor: [255, 248, 230],
            sunIntensity: 2.0,
            sunSize: 0.995,
            maxBounces: 4,
            cornerRadius: 0.0,
            batchSize: 2,
            focusDistance: 100,
            aperture: 0.4,
            volumetricDensity: 0.01
        }
    });

    // --- Materials — bright Minecraft-style palette ---
    const matWater      = cubica.setMaterial(50,  120, 240, 0.1, 1.33, 0.0);
    const matDeepWater  = cubica.setMaterial(15,  45,  140, 0.1, 1.33, 0.0);
    const matGrass      = cubica.setMaterial(75,  175, 55,  0.9, 1.0, 0.0);
    const matDirt       = cubica.setMaterial(150, 95,  45,  0.9, 1.0, 0.0);
    const matStone      = cubica.setMaterial(130, 130, 135, 0.8, 1.0, 0.0);
    const matSnow       = cubica.setMaterial(248, 250, 255, 0.9, 1.0, 0.0);
    const matSand       = cubica.setMaterial(220, 200, 140, 0.8, 1.0, 0.0);
    const matWood       = cubica.setMaterial(110, 65,  30,  0.8, 1.0, 0.0);
    const matLeaves     = cubica.setMaterial(50,  130, 45,  0.9, 1.0, 0.0);
    const matMossy      = cubica.setMaterial(80,  110, 72,  0.8, 1.0, 0.0);
    const matCloud      = cubica.setMaterial(252, 254, 255, 0.7, 1.0, 0.0);

    const SIZE    = 256;
    const HMAX    = 128;
    const WL      = 22; // water level
    const SEED    = Math.floor(Math.random() * 10000);

    // --- Simple value noise with smoothstep interpolation ---
    function hash(x, z) {
        let h = (x * 127.100 + z * 311.700 + 17.0) * 43758.5453;
        return h - Math.floor(h);
    }

    function snoise(x, z, seed) {
        let ix = Math.floor(x), iz = Math.floor(z);
        let fx = x - ix, fz = z - iz;
        fx = fx * fx * (3 - 2 * fx);
        fz = fz * fz * (3 - 2 * fz);

        let t0 = hash(ix, iz + seed) * (1 - fx) + hash(ix + 1, iz + seed) * fx;
        let t1 = hash(ix, iz + 1 + seed) * (1 - fx) + hash(ix + 1, iz + 1 + seed) * fx;
        return t0 * (1 - fz) + t1 * fz;
    }

    function fbm(x, z, oct, seed) {
        let v = 0, a = 0.5, f = 1;
        for (let i = 0; i < oct; i++) {
            v += a * snoise(x * f, z * f, seed + i * 77);
            f *= 2;
            a *= 0.5;
        }
        return v;
    }

    function ridged(x, z, oct, seed) {
        let v = 0, a = 0.5, f = 1;
        for (let i = 0; i < oct; i++) {
            v += a * (1 - Math.abs(2 * snoise(x * f, z * f, seed + i * 77 + 50) - 1));
            f *= 2;
            a *= 0.5;
        }
        return v;
    }

    // --- Noise-based terrain ---
    function getHeight(x, z) {
        let h = 0;
        h += fbm(x * 0.007, z * 0.007, 6, SEED) * 55;        // Large mountains
        h += ridged(x * 0.015, z * 0.015, 4, SEED + 200) * 28;     // Sharp ridges
        h += fbm(x * 0.04, z * 0.04, 4, SEED + 400) * 10;          // Mid bumps
        h += fbm(x * 0.09, z * 0.09, 3, SEED + 600) * 4;           // Fine detail
        return Math.floor(h);
    }

    // --- Canyon (running roughly diagonally) ---
    function canyonDepth(x, z) {
        let angle = SEED * 0.001;
        let ca = Math.cos(angle), sa = Math.sin(angle);
        let rx = x * ca - z * sa;
        let rz = x * sa + z * ca;
        let d = Math.abs(rx * 0.65 + Math.sin(x * 0.012 + SEED) * 25 + Math.sin(z * 0.018 + SEED * 1.3) * 15);
        if (d < 10) return 50 + (1 - d / 10) * 22;
        if (d < 26) return (1 - (d - 10) / 16) * 22;
        return 0;
    }

    // --- Lake basin detection ---
    function isLakeBasin(x, z) {
        return fbm(x * 0.014, z * 0.014, 3, SEED + 800) > 0.40
            && getHeight(x, z) < 30;
    }

    // --- Scan entire heightmap ---
    const heights = new Int32Array(SIZE * SIZE);
    for (let x = 0; x < SIZE; x++) {
        for (let z = 0; z < SIZE; z++) {
            let h = getHeight(x, z);
            let canyon = canyonDepth(x, z);
            let lakeDep = isLakeBasin(x, z) ? 3 : 0;
            h = Math.max(2, Math.min(h - Math.floor(canyon) - lakeDep, HMAX - 2));
            heights[x * SIZE + z] = h;
        }
    }

    // --- Fill voxel terrain ---
    const H = heights;
    for (let x = 0; x < SIZE; x++) {
        for (let z = 0; z < SIZE; z++) {
            let h = H[x * SIZE + z];
            for (let y = 0; y < SIZE; y++) {
                if (y >= h) break;

                let mat;
                if (y < WL) {
                    mat = y < WL - 5 ? matDeepWater : matWater;
                } else if (y === h - 1) {
                    // Surface block
                    if (h <= WL + 2) {
                        mat = Math.random() > 0.3 ? matSand : matMossy;
                    } else if (h > 70) {
                        mat = matSnow;
                    } else if (h > 58) {
                        mat = Math.random() > 0.45 ? matSnow : matStone;
                    } else if (Math.random() > 0.12) {
                        mat = matGrass;
                    } else {
                        mat = matMossy;
                    }
                } else if (y >= h - 5) {
                    mat = matDirt;
                } else {
                    mat = matStone;
                }

                cubica.currentMaterial = mat;
                cubica.set(x, y, z);
            }
        }
    }

    // --- Trees (voxel: trunk + leaf dome) ---
    function placeTree(tx, ty, tz) {
        let th = Math.floor(Math.random() * 3) + 4;
        cubica.currentMaterial = matWood;
        for (let i = 0; i < th; i++) cubica.set(tx, ty + i, tz);

        let top = ty + th;
        let r = Math.floor(Math.random() * 2) + 3;
        cubica.currentMaterial = matLeaves;
        for (let dx = -r; dx <= r; dx++) {
            for (let dy = -1; dy <= r; dy++) {
                for (let dz = -r; dz <= r; dz++) {
                    let d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                    if (d <= r + (dy < 0 ? -0.5 : 0)) {
                        cubica.set(tx + dx, top + dy, tz + dz, 0.05);
                    }
                }
            }
        }
    }

    for (let x = 3; x < SIZE - 3; x += 3) {
        for (let z = 3; z < SIZE - 3; z += 3) {
            let h = H[x * SIZE + z];
            if (h > WL + 6 && h < 62 && Math.random() < 0.14) {
                placeTree(x, h, z);
            }
        }
    }

    // --- Floating voxel clouds ---
    function placeCloud(clx, cly, clz) {
        cubica.currentMaterial = matCloud;
        let rx = Math.floor(Math.random() * 3) + 5;
        let ry = Math.floor(Math.random() * 2) + 2;
        let rz = Math.floor(Math.random() * 3) + 5;
        for (let x = -rx; x <= rx; x++) {
            for (let y = -ry; y <= ry; y++) {
                for (let z = -rz; z <= rz; z++) {
                    let d = Math.sqrt(
                        (x * x) / (rx * rx) +
                        (y * y) / (ry * ry) +
                        (z * z) / (rz * rz));
                    if (d < 0.75 + Math.random() * 0.35) {
                        cubica.set(clx + x, cly + y, clz + z);
                    }
                }
            }
        }
    }

    for (let i = 0; i < 35; i++) {
        let cx = Math.floor(Math.random() * SIZE);
        let cz = Math.floor(Math.random() * SIZE);
        let cy = Math.floor(Math.random() * 18) + 78;
        placeCloud(cx, cy, cz);
    }

    cubica.updateBuffers();
    statusEl.innerText = "Hit 's' to save";

    // --- Controls & render loop ---
    const controls = new CubicaOrbitControls(cubica, {
        radius: 130,
        target: [128, 25, 128],
        phi: 1.35,
        theta: Math.PI / 3
     });

    window.addEventListener('resize', () => {
        if (!isSaving) {
            cubica.resize(window.innerWidth, window.innerHeight);
        }
    });

    let isSaving = false;
    let saveSamples = 0;
    const targetSamples = 200;
    let tiles = [];
    let currentTileIndex = 0;
    const TILE_SIZE = 512;
    let outputWidth = window.innerWidth * 2;
    let outputHeight= window.innerHeight * 2;

    window.addEventListener('keydown', (e) => {
        if (e.key === 's' && !isSaving) {
            isSaving = true;
            cubica.resize(outputWidth, outputHeight);
            
            tiles = [];
            for (let y = 0; y < outputHeight; y += TILE_SIZE) {
                for (let x = 0; x < outputWidth; x += TILE_SIZE) {
                    tiles.push({x, y, w: TILE_SIZE, h: TILE_SIZE});
                }
            }
            currentTileIndex = 0;
            saveSamples = 0;
            cubica.clear();
            statusEl.innerText = "Rendering high-res (Tiled)...";
        }
    });

    let lastTime = performance.now();
    let frames = 0;

    function frame() {
        const now = performance.now();
        frames++;
        if (now - lastTime >= 1000) {
            fpsEl.innerText = `FPS: ${frames}`;
            frames = 0;
            lastTime = now;
        }
        controls.update();
        
        if (isSaving) {
            const tile = tiles[currentTileIndex];
            cubica.setRenderScale(1.0);
            cubica.trace(cubica.env.batchSize, tile);
            saveSamples += cubica.env.batchSize;
            
            statusEl.innerText = `Rendering Tile ${currentTileIndex + 1}/${tiles.length} ... ${saveSamples}/${targetSamples} samples`;
            
            if (saveSamples >= targetSamples) {
                currentTileIndex++;
                if (currentTileIndex < tiles.length) {
                    saveSamples = 0;
                    cubica.clear();
                } else {
                    isSaving = false;
                    cubica.device.queue.onSubmittedWorkDone().then(() => {
                        const dataURL = canvas.toDataURL('image/png');
                        const a = document.createElement('a');
                        a.href = dataURL;
                        a.download = 'cubica_render_hires.png';
                        a.click();
                        
                        cubica.resize(window.innerWidth, window.innerHeight);
                        statusEl.innerText = 'Ready';
                    });
                }
            }
        } else {
            if (controls.isDragging) {
                cubica.setRenderScale(0.3);
            } else {
                cubica.setRenderScale(1.0);
            }
            cubica.trace(controls.isDragging ? 1 : 4);
        }

        samplesEl.innerText = `Samples: ${cubica.sampleCount}`;
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

init();
