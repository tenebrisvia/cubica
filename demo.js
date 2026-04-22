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
        dimensions: [128, 64, 128],
        env: {
            fov: 50,
            ambientColor: [75, 75, 100],
            ambientIntensity: 0.4,
            sunDirection: [0.6, 1.0, 0.4],
            sunColor: [255, 210, 200],
            sunIntensity: 1.5, 
            sunSize: 0.999,
            maxBounces: 4,
            cornerRadius: 0.2,
            batchSize: 4,
            focusDistance: 40,
            aperture: 0.05
        }
    });

    // Materials
    const matWater = cubica.setMaterial(90, 120, 230, 0.0, 1.33, 0.0);
    const matFloor = cubica.setMaterial(20, 25, 30, 0.9, 1.0, 0.0);
    const matConcrete = cubica.setMaterial(160, 165, 175, 0.8, 1.0, 0.0);
    const matDarkFacade = cubica.setMaterial(40, 45, 55, 0.5, 1.0, 0.0);
    const matNeonCyan = cubica.setMaterial(0, 255, 255, 0.6, 1.0, 8.0);
    const matNeonPink = cubica.setMaterial(255, 50, 150, 0.6, 1.0, 8.0);
    const matWindowGold = cubica.setMaterial(255, 200, 100, 0.2, 1.0, 4.0);

    // Build Ground
    cubica.currentMaterial = matFloor;
    for (let x = 1; x < 127; x++) {
        for (let z = 1; z < 127; z++) {
            cubica.set(x, 0, z);
        }
    }

    const waterLevel = 5;

    // City Blocks
    for (let x = 12; x < 116; x += Math.floor(Math.random() * 8 + 6)) {
        for (let z = 12; z < 116; z += Math.floor(Math.random() * 8 + 6)) {
            let dx = x - 64;
            let dz = z - 64;
            let dist = Math.sqrt(dx * dx + dz * dz);
            let maxH = 80 * Math.exp(-(dist * dist) / 1200);
            let h = Math.floor(Math.random() * maxH + 4);
            let w = Math.floor(Math.random() * 8 + 3);
            let d = Math.floor(Math.random() * 8 + 3);
            let isDark = Math.random() > 0.3;
            let baseMat = isDark ? matDarkFacade : matConcrete;

            for (let bx = x; bx < x + w && bx < 128; bx++) {
                for (let bz = z; bz < z + d && bz < 128; bz++) {
                    for (let by = 1; by < h; by++) {
                        cubica.currentMaterial = baseMat;
                        let currentGap = 0.04;

                        if (by > waterLevel && Math.random() > 0.95) {
                            let accent = Math.random();
                            if (accent > 0.6) { cubica.currentMaterial = matWindowGold; currentGap = 0.08; }
                            else if (accent > 0.3) { cubica.currentMaterial = matNeonCyan; currentGap = 0.08; }
                            else { cubica.currentMaterial = matNeonPink; currentGap = 0.08; }
                        }
                        
                        cubica.set(bx, by, bz, currentGap);
                    }
                }
            }

            // Neon tops
            if (h > waterLevel + 10 && Math.random() > 0.6) {
                cubica.currentMaterial = Math.random() > 0.5 ? matNeonCyan : matNeonPink;
                let topW = w - 2;
                let topD = d - 2;
                if (topW > 0 && topD > 0) {
                    for (let bx = x + 1; bx < x + 1 + topW; bx++) {
                        for (let bz = z + 1; bz < z + 1 + topD; bz++) {
                            cubica.set(bx, h, bz);
                        }
                    }
                }
            }
        }
    }

    // Water Layer
    cubica.currentMaterial = matWater;
    for (let x = 0; x < 128; x++) {
        for (let z = 0; z < 128; z++) {
            for (let y = 0; y <= waterLevel; y++) {
                if (cubica.gridData[cubica.getIndex(x, y, z)] === 0) {
                    cubica.set(x, y, z);
                }
            }
        }
    }

    // Ensure CPU buffers are sent to GPU
    cubica.updateBuffers();

    statusEl.innerText = "Tracing...";

    // Setup UI Controls
    const controls = new CubicaOrbitControls(cubica, {
        radius: 120,
        target: [64, 0, 64],
        phi: Math.PI / 3,
        theta: Math.PI / 4
    });

    // Rebuild textures whenever the window resizes so the render
    // is always full-res rather than stretching the texture
    window.addEventListener('resize', () => {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
        cubica.createAccumulationTexture();
        cubica.updateBindGroups();
        cubica.clear();
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
        cubica.trace(controls.isDragging ? 1 : 4); //when dragging drop to 1 sample per frame
        samplesEl.innerText = `Samples: ${cubica.sampleCount}`;

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

init();