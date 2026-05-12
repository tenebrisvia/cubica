// Uniform buffer slot offsets (f32 index = byte offset / 4).
// These MUST mirror the WGSL `Uniforms` struct field order exactly.
// struct Uniforms {
//   camPos:vec3f, fov:f32,                    // slots  0-3
//   camDir:vec3f, sampleCount:u32,            // slots  4-7
//   camRight:vec3f, rngSeed:u32,              // slots  8-11
//   camUp:vec3f, gridDimX:u32,               // slots 12-15
//   ambientColor:vec3f, gridDimY:u32,         // slots 16-19
//   sunDirection:vec3f, gridDimZ:u32,         // slots 20-23
//   sunColor:vec3f, sunSize:f32,              // slots 24-27
//   ambientIntensity:f32, sunIntensity:f32, maxHops:u32, batchSize:u32 // slots 28-31
//   focusDistance:f32, aperture:f32, _pad2:f32, _pad3:f32  // slots 32-35
// }
const UBO = Object.freeze({
    camPosX: 0,  camPosY: 1,  camPosZ: 2,  fov: 3,
    camDirX: 4,  camDirY: 5,  camDirZ: 6,  sampleCount: 7,
    camRightX: 8, camRightY: 9, camRightZ: 10, rngSeed: 11,
    camUpX: 12,  camUpY: 13,  camUpZ: 14,  gridDimX: 15,
    ambR: 16,    ambG: 17,    ambB: 18,    gridDimY: 19,
    sunDirX: 20, sunDirY: 21, sunDirZ: 22, gridDimZ: 23,
    sunR: 24,    sunG: 25,    sunB: 26,    sunSize: 27,
    ambientIntensity: 28, sunIntensity: 29, maxHops: 30, batchSize: 31,
    focusDistance: 32,    aperture: 33,   volumetricDensity: 34,
});

export class Cubica {
    constructor() {
        this.materials = [];
        this.currentMaterial = 0; // 0 will be empty space / air
        this.sampleCount = 0;
        
        // Default camera
        this.camPos = [0, 0, 0];
        this.camTarget = [0, 0, -1];
        this._sunDirNorm = [0, 1, 0];
        
        // Default environment
        this.env = {
            fov: 50,
            ambientColor: [0, 0, 0],
            ambientIntensity: 0.0,
            sunDirection: [0.5, 1.0, 0.2],
            sunColor: [255, 255, 230],
            sunIntensity: 0.0,
            sunSize: 0.99,
            maxBounces: 4,
            cornerRadius: 0.0,
            batchSize: 4,
            focusDistance: 100,
            aperture: 0.0,
            volumetricDensity: 0.0
        };
        
        // Add a default "air" material at index 0
        this.materials.push({
            r: 0, g: 0, b: 0,
            roughness: 0, ior: 1.0, emission: 0
        });
    }

    setEnvironment(opts) {
        this.env = { ...this.env, ...opts };
        this._normaliseSun();
        this.clear();
    }

    async init({ canvas, dimensions, env = {} }) {
        this.canvas = canvas;
        this.dimensions = dimensions; 
        this.gridSize = dimensions[0] * dimensions[1] * dimensions[2];
        this.gridData = new Uint32Array(this.gridSize);
        this.env = { ...this.env, ...env };
        this._normaliseSun();

        if (!navigator.gpu) {
            throw new Error("WebGPU not supported");
        }
        
        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
            throw new Error("No WebGPU adapter found");
        }
        
        this.device = await this.adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        
        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'opaque'
        });

        // Buffers
        this.voxelBuffer = this.device.createBuffer({
            size: this.gridSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Max 256 materials, 8 floats per material (32 bytes)
        // struct Material { color: vec3f, roughness: f32, emission: f32, ior: f32, pad1, pad2 }
        this.materialBuffer = this.device.createBuffer({
            size: 256 * 8 * 4, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Uniforms — 36 floats = 144 bytes (32 base + 4 for DOF)
        this.uniformBufferArray = new ArrayBuffer(144);
        this.uniformDataFloat = new Float32Array(this.uniformBufferArray);
        this.uniformDataUint = new Uint32Array(this.uniformBufferArray);
        this.uniformBuffer = this.device.createBuffer({
            size: 144,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Accumulation Texture
        this.frameIndex = 0;
        this.createAccumulationTexture();

        // Load Shaders
        const shaderUrl = new URL('cubica.wgsl', import.meta.url);
        const req = await fetch(shaderUrl);
        let wgslCode = await req.text();
        
        // Dynamic Compile-Time Shader Replacements
        wgslCode = wgslCode.replace(/const MAX_BOUNCES:\s*u32\s*=\s*\d+u;/, `const MAX_BOUNCES: u32 = ${this.env.maxBounces}u;`);
        wgslCode = wgslCode.replace(/const CORNER_RADIUS:\s*f32\s*=\s*[\d.]+;/, `const CORNER_RADIUS: f32 = ${this.env.cornerRadius.toFixed(4)};`);
        
        this.shaderModule = this.device.createShaderModule({ code: wgslCode });

        // Compute Pipeline
        this.computePipeline = await this.device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: this.shaderModule,
                entryPoint: 'main_pt'
            }
        });

        // Fullscreen Quad Render Pipeline
        this.renderPipeline = await this.device.createRenderPipelineAsync({
            layout: 'auto',
            vertex: {
                module: this.shaderModule,
                entryPoint: 'fs_vertex'
            },
            fragment: {
                module: this.shaderModule,
                entryPoint: 'fs_fragment',
                targets: [{ format: this.presentationFormat }]
            },
            primitive: { topology: 'triangle-list' }
        });

        this.updateBindGroups();

        // Write uniform slots that are fixed for the lifetime of this grid
        this.uniformDataUint[UBO.gridDimX] = this.dimensions[0];
        this.uniformDataUint[UBO.gridDimY] = this.dimensions[1];
        this.uniformDataUint[UBO.gridDimZ] = this.dimensions[2];
        this.uniformDataUint[UBO.maxHops]  = this.dimensions[0] + this.dimensions[1] + this.dimensions[2] + 16;
    }

    createAccumulationTexture() {
        if (this.accumTexture0) this.accumTexture0.destroy();
        if (this.accumTexture1) this.accumTexture1.destroy();
        
        const desc = {
            size: [this.canvas.width, this.canvas.height, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        };
        this.accumTexture0 = this.device.createTexture(desc);
        this.accumTexture1 = this.device.createTexture(desc);
        this._wgX = Math.ceil(this.canvas.width / 8);
        this._wgY = Math.ceil(this.canvas.height / 8);
    }

    updateBindGroups() {
        this.computeGroupLayout = this.computePipeline.getBindGroupLayout(0);
        
        this.computeBindGroup0 = this.device.createBindGroup({
            layout: this.computeGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.voxelBuffer } },
                { binding: 1, resource: { buffer: this.materialBuffer } },
                { binding: 2, resource: { buffer: this.uniformBuffer } },
                { binding: 3, resource: this.accumTexture0.createView() },
                { binding: 4, resource: this.accumTexture1.createView() }
            ]
        });

        this.computeBindGroup1 = this.device.createBindGroup({
            layout: this.computeGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.voxelBuffer } },
                { binding: 1, resource: { buffer: this.materialBuffer } },
                { binding: 2, resource: { buffer: this.uniformBuffer } },
                { binding: 3, resource: this.accumTexture1.createView() },
                { binding: 4, resource: this.accumTexture0.createView() }
            ]
        });

        const renderGroupLayout = this.renderPipeline.getBindGroupLayout(0);
        
        this.renderBindGroup0 = this.device.createBindGroup({
            layout: renderGroupLayout,
            entries: [
                { binding: 1, resource: this.accumTexture1.createView() }
            ]
        });

        this.renderBindGroup1 = this.device.createBindGroup({
            layout: renderGroupLayout,
            entries: [
                { binding: 1, resource: this.accumTexture0.createView() }
            ]
        });
    }

    setMaterial(r, g, b, roughness, ior, emission) {
        const id = this.materials.length;
        if (id >= 256) {
            console.warn("Max 256 materials supported");
            return 255;
        }
        this.materials.push({ r, g, b, roughness, ior, emission });
        return id;
    }

    getIndex(x, y, z) {
        if (x < 0 || y < 0 || z < 0 || x >= this.dimensions[0] || y >= this.dimensions[1] || z >= this.dimensions[2]) return -1;
        return x + y * this.dimensions[0] + z * this.dimensions[0] * this.dimensions[1];
    }

    set(x, y, z, gapSize = 0.0) {
        const idx = this.getIndex(x, y, z);
        if (idx !== -1) {
            const gapByte = Math.min(255, Math.max(0, Math.floor(gapSize * 255.0)));
            this.gridData[idx] = (gapByte << 8) | this.currentMaterial;
        }
    }

    remove(x, y, z) {
        const idx = this.getIndex(x, y, z);
        if (idx !== -1) {
            this.gridData[idx] = 0;
        }
    }

    updateBuffers() {
        // Upload voxels
        this.device.queue.writeBuffer(this.voxelBuffer, 0, this.gridData);
        
        // Upload materials
        const matData = new Float32Array(this.materials.length * 8);
        for(let i=0; i<this.materials.length; i++) {
            const m = this.materials[i];
            matData[i*8 + 0] = m.r / 255.0;
            matData[i*8 + 1] = m.g / 255.0;
            matData[i*8 + 2] = m.b / 255.0;
            matData[i*8 + 3] = m.roughness;
            matData[i*8 + 4] = m.emission;
            matData[i*8 + 5] = m.ior;
            matData[i*8 + 6] = 0;
            matData[i*8 + 7] = 0;
        }
        this.device.queue.writeBuffer(this.materialBuffer, 0, matData);

        this.clear();
    }

    normalize(v) {
        let len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if(len===0) return [0,0,0];
        return [v[0]/len, v[1]/len, v[2]/len];
    }

    // Caches a normalised copy of env.sunDirection into _sunDirNorm.
    // Called once at init and once per setEnvironment() — not per frame.
    _normaliseSun() {
        const sd = this.env.sunDirection;
        const len = Math.sqrt(sd[0]*sd[0] + sd[1]*sd[1] + sd[2]*sd[2]) || 1;
        this._sunDirNorm[0] = sd[0] / len;
        this._sunDirNorm[1] = sd[1] / len;
        this._sunDirNorm[2] = sd[2] / len;
    }
    
    cross(a, b) {
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ];
    }

    camera({ position, target, fov }) {
        let moved = false;
        if(this.camPos[0]!==position[0] || this.camPos[1]!==position[1] || this.camPos[2]!==position[2] ||
           this.camTarget[0]!==target[0] || this.camTarget[1]!==target[1] || this.camTarget[2]!==target[2]) {
            moved = true;
        }

        this.camPos    = [position[0], position[1], position[2]];
        this.camTarget = [target[0],   target[1],   target[2]];
        if (fov != null) {
            this.env.fov = fov;
        }

        let w = this.normalize([target[0] - position[0], target[1] - position[1], target[2] - position[2]]);
        // Approximate Up
        let up = [0, 1, 0];
        if (Math.abs(w[1]) > 0.99) up = [0, 0, -Math.sign(w[1])];
        
        let u = this.normalize(this.cross(w, up));
        let v = this.cross(u, w);

        // Update uniforms
        this.uniformDataFloat[UBO.camPosX] = this.camPos[0];
        this.uniformDataFloat[UBO.camPosY] = this.camPos[1];
        this.uniformDataFloat[UBO.camPosZ] = this.camPos[2];
        this.uniformDataFloat[UBO.fov]     = this.env.fov * Math.PI / 180.0;
        
        this.uniformDataFloat[UBO.camDirX] = w[0];
        this.uniformDataFloat[UBO.camDirY] = w[1];
        this.uniformDataFloat[UBO.camDirZ] = w[2];
        // UBO.sampleCount filled in trace() as u32
        
        this.uniformDataFloat[UBO.camRightX] = u[0];
        this.uniformDataFloat[UBO.camRightY] = u[1];
        this.uniformDataFloat[UBO.camRightZ] = u[2];
        // UBO.rngSeed filled in trace() as u32

        this.uniformDataFloat[UBO.camUpX] = v[0];
        this.uniformDataFloat[UBO.camUpY] = v[1];
        this.uniformDataFloat[UBO.camUpZ] = v[2];

        if (moved) this.clear();
        return moved;
    }

    clear() {
        this.sampleCount = 0;
        this.frameIndex = 0; // Reset ping-pong index
    }

    trace(batchSizeOverride) {
        const currentBatch = batchSizeOverride !== undefined ? batchSizeOverride : this.env.batchSize;
        
        // Write env uniforms once per trace() call
        // Sun direction is pre-normalised in _normaliseSun() and only recomputed on env change
        const sd = this._sunDirNorm;
        
        this.uniformDataFloat[UBO.ambR] = this.env.ambientColor[0] / 255.0;
        this.uniformDataFloat[UBO.ambG] = this.env.ambientColor[1] / 255.0;
        this.uniformDataFloat[UBO.ambB] = this.env.ambientColor[2] / 255.0;
        this.uniformDataFloat[UBO.sunDirX] = sd[0];
        this.uniformDataFloat[UBO.sunDirY] = sd[1];
        this.uniformDataFloat[UBO.sunDirZ] = sd[2];
        this.uniformDataFloat[UBO.sunR]    = this.env.sunColor[0] / 255.0;
        this.uniformDataFloat[UBO.sunG]    = this.env.sunColor[1] / 255.0;
        this.uniformDataFloat[UBO.sunB]    = this.env.sunColor[2] / 255.0;
        this.uniformDataFloat[UBO.sunSize] = this.env.sunSize;
        
        this.uniformDataFloat[UBO.ambientIntensity] = this.env.ambientIntensity;
        this.uniformDataFloat[UBO.sunIntensity]     = this.env.sunIntensity;
        
        this.uniformDataFloat[UBO.batchSize] = 0;
        this.uniformDataUint[UBO.batchSize]  = currentBatch;

        this.uniformDataFloat[UBO.focusDistance] = this.env.focusDistance;
        this.uniformDataFloat[UBO.aperture]      = this.env.aperture;
        this.uniformDataFloat[UBO.volumetricDensity] = this.env.volumetricDensity;

        // Iterate Sample Counters locally
        this.sampleCount += currentBatch;
        this.uniformDataUint[UBO.sampleCount] = this.sampleCount;
        this.uniformDataUint[UBO.rngSeed]     = Math.random() * 0xFFFFFFFF >>> 0;

        // Execute unified GPU dispatch limits
        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformBufferArray);
        const commandEncoder = this.device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        
        const cbg = (this.frameIndex % 2 === 0) ? this.computeBindGroup0 : this.computeBindGroup1;
        computePass.setBindGroup(0, cbg);
        computePass.dispatchWorkgroups(this._wgX, this._wgY);
        computePass.end();
        this.frameIndex++;

        // Single Render Pass at the end — blit accumulation texture to canvas
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        // frameIndex has been incremented, so the last written texture is (frameIndex-1) % 2
        const rbg = ((this.frameIndex - 1) % 2 === 0) ? this.renderBindGroup0 : this.renderBindGroup1;
        renderPass.setBindGroup(0, rbg);
        renderPass.draw(6, 1, 0, 0);
        renderPass.end();

        // Single submit for everything
        this.device.queue.submit([commandEncoder.finish()]);
    }

    // Release all GPU resources. Call when the tracer is no longer needed.
    dispose() {
        this.voxelBuffer.destroy();
        this.materialBuffer.destroy();
        this.uniformBuffer.destroy();
        this.accumTexture0.destroy();
        this.accumTexture1.destroy();
        this.device.destroy();
    }
}

export class CubicaOrbitControls {
    constructor(tracer, options = {}) {
        this.tracer = tracer;
        this.canvas = tracer.canvas;
        
        this.theta = options.theta !== undefined ? options.theta : Math.PI / 4;
        this.phi = options.phi !== undefined ? options.phi : Math.PI / 3;
        this.radius = options.radius !== undefined ? options.radius : 150;
        this.target = options.target || [0, 0, 0];
        
        this.panSpeed = options.panSpeed !== undefined ? options.panSpeed : 0.002;
        this.zoomSpeed = options.zoomSpeed !== undefined ? options.zoomSpeed : 0.1;
        this.rotateSpeed = options.rotateSpeed !== undefined ? options.rotateSpeed : 0.005;

        this.isDragging = false;
        this.dragMode = 0; // 1 = rotate, 2 = pan
        this.lastX = 0;
        this.lastY = 0;
        this.lastPinchDist = 0;
        this._posArr    = [0, 0, 0];
        this._targetArr = [0, 0, 0];

        this._attach();
    }

    _attach() {
        // All listeners share a single AbortController so dispose() removes them atomically.
        this._ac = new AbortController();
        const sig = this._ac.signal;

        this.canvas.addEventListener('contextmenu', e => e.preventDefault(), { signal: sig });

        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            if (e.button === 0) this.dragMode = 1;
            else if (e.button === 2) this.dragMode = 2;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
        }, { signal: sig });

        window.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.dragMode = 0;
        }, { signal: sig });

        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;

            let dx = e.clientX - this.lastX;
            let dy = e.clientY - this.lastY;
            this.lastX = e.clientX;
            this.lastY = e.clientY;

            if (this.dragMode === 1) {
                this.theta += dx * this.rotateSpeed;
                this.phi -= dy * this.rotateSpeed;
                this.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.phi));
            } else if (this.dragMode === 2) {
                this._pan(dx, dy);
            }
        }, { signal: sig });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.radius += e.deltaY * this.zoomSpeed;
            this.radius = Math.max(5.0, Math.min(2000.0, this.radius));
        }, { passive: false, signal: sig });

        // --- Touch Support ---

        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.isDragging = true;
            if (e.touches.length === 1) {
                this.dragMode = 1;
                this.lastX = e.touches[0].clientX;
                this.lastY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                this.dragMode = 2;
                let dx = e.touches[0].clientX - e.touches[1].clientX;
                let dy = e.touches[0].clientY - e.touches[1].clientY;
                this.lastPinchDist = Math.sqrt(dx * dx + dy * dy);
                this.lastX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                this.lastY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
            }
        }, { passive: false, signal: sig });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (!this.isDragging) return;

            if (e.touches.length === 1 && this.dragMode === 1) {
                let dx = e.touches[0].clientX - this.lastX;
                let dy = e.touches[0].clientY - this.lastY;
                this.lastX = e.touches[0].clientX;
                this.lastY = e.touches[0].clientY;

                this.theta += dx * this.rotateSpeed;
                this.phi -= dy * this.rotateSpeed;
                this.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.phi));
                
            } else if (e.touches.length === 2 && this.dragMode === 2) {
                let cx2 = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                let cy2 = (e.touches[0].clientY + e.touches[1].clientY) / 2;
                let dx = cx2 - this.lastX;
                let dy = cy2 - this.lastY;
                this.lastX = cx2;
                this.lastY = cy2;

                this._pan(dx, dy);

                let pinchDx = e.touches[0].clientX - e.touches[1].clientX;
                let pinchDy = e.touches[0].clientY - e.touches[1].clientY;
                let currentPinchDist = Math.sqrt(pinchDx * pinchDx + pinchDy * pinchDy);
                let pinchDelta = this.lastPinchDist - currentPinchDist;
                this.lastPinchDist = currentPinchDist;

                this.radius += pinchDelta * this.zoomSpeed * 0.5;
                this.radius = Math.max(5.0, Math.min(2000.0, this.radius));
            }
        }, { passive: false, signal: sig });

        this.canvas.addEventListener('touchend', (e) => {
            if (e.touches.length === 0) {
                this.isDragging = false;
                this.dragMode = 0;
            } else if (e.touches.length === 1) {
                this.dragMode = 1;
                this.lastX = e.touches[0].clientX;
                this.lastY = e.touches[0].clientY;
            }
        }, { signal: sig });
    }

    // Uses the camera basis cached by update()
    _pan(dx, dy) {
        const u = this._camRight || [1, 0, 0];
        const v = this._camUp    || [0, 1, 0];
        const speed = this.radius * this.panSpeed;
        this.target[0] += (-dx * u[0] + dy * v[0]) * speed;
        this.target[1] += (-dx * u[1] + dy * v[1]) * speed;
        this.target[2] += (-dx * u[2] + dy * v[2]) * speed;
    }

    // Computes the camera position, caches the right/up basis vectors for _pan(),
    // then submits the camera update to the tracer
    update() {
        const sinPhi = Math.sin(this.phi), cosPhi = Math.cos(this.phi);
        const cosTheta = Math.cos(this.theta), sinTheta = Math.sin(this.theta);

        const cx = this.target[0] + this.radius * sinPhi * cosTheta;
        const cy = this.target[1] + this.radius * cosPhi;
        const cz = this.target[2] + this.radius * sinPhi * sinTheta;

        // Cache right/up basis so _pan() doesn't repeat this work
        const t = this.tracer;
        const w = t.normalize([this.target[0]-cx, this.target[1]-cy, this.target[2]-cz]);
        let up = [0, 1, 0];
        if (Math.abs(w[1]) > 0.99) up = [0, 0, -Math.sign(w[1])];
        this._camRight = t.normalize(t.cross(w, up)); // camera right (u)
        this._camUp    = t.cross(this._camRight, w);   // camera up    (v)

        this._posArr[0] = cx; this._posArr[1] = cy; this._posArr[2] = cz;
        this._targetArr[0] = this.target[0]; this._targetArr[1] = this.target[1]; this._targetArr[2] = this.target[2];
        t.camera({ position: this._posArr, target: this._targetArr });
    }

    dispose() {
        this._ac.abort();
    }
}