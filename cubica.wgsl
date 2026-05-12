struct Material {
    color: vec3f,
    roughness: f32,
    emission: f32,
    ior: f32,
    pad1: f32,
    pad2: f32,
}

struct Uniforms {
    camPos: vec3f,
    fov: f32,
    camDir: vec3f,
    sampleCount: u32,
    camRight: vec3f,
    rngSeed: u32,
    camUp: vec3f,
    gridDimX: u32,       // u32 avoids repeated casts in DDA hot path
    ambientColor: vec3f,
    gridDimY: u32,
    sunDirection: vec3f, // pre-normalised on CPU
    gridDimZ: u32,
    sunColor: vec3f,
    sunSize: f32,
    ambientIntensity: f32,
    sunIntensity: f32,
    maxHops: u32,
    batchSize: u32,
    // --- Depth of Field (thin-lens model) ---
    // aperture == 0 reduces to a standard pinhole camera with no extra cost.
    focusDistance: f32,  // world-space distance to the in-focus plane
    aperture: f32,       // radius of the lens disc; 0 = pinhole (no blur)
    volumetricDensity: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> grid: array<u32>;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var accumMapOld: texture_2d<f32>;
@group(0) @binding(4) var accumMapNew: texture_storage_2d<rgba16float, write>;

const MAX_BOUNCES: u32 = 4u;
const CORNER_RADIUS: f32 = 0.16;
const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618; // 1/PI — Lambertian BRDF normalisation factor

// --- PRNG ---
// PCG (Permuted Congruential Generator) Hash
// For pseudo-random number generation on GPUs.
var<private> rng_state: u32;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand() -> f32 {
    rng_state = pcg_hash(rng_state);
    return f32(rng_state) / 4294967296.0;
}

// Generate a random 3D direction vector (uniform sphere)
fn rand_dir() -> vec3f {
    let z = rand() * 2.0 - 1.0;
    let a = rand() * 2.0 * PI;
    let r = sqrt(1.0 - z * z);
    let x = r * cos(a);
    let y = r * sin(a);
    return vec3f(x, y, z);
}

// Cosine-weighted hemisphere sample (importance-sampled toward the surface normal).
// PDF = cos(theta) / PI — the cos and PI factors cancel exactly in the Lambertian
// throughput estimator, so throughput simply multiplies by mat.color with no extra term.
fn cosine_hemisphere(n: vec3f) -> vec3f {
    let u1  = rand();
    let u2  = rand();
    let r   = sqrt(u1);
    let phi = 2.0 * PI * u2;
    // Build an orthonormal tangent frame around n
    var up  = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let t   = normalize(cross(up, n));
    let b   = cross(n, t);
    return t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1));
}

// --- Interleaved Gradient Noise (IGN) ---
// Source: Jorge Jimenez, "Temporal Anti-Aliasing in Uncharted 4" (2016).
//
// WHY IGN INSTEAD OF PURE WHITE NOISE?
// The PCG hash above is white noise — every pixel's random stream
// is statistically independent. But independent means *uncorrelated*, which
// means noise errors form visible blotchy clumps (low spatial frequency
// content). The human visual system is far more sensitive to low-frequency
// clumps than to fine-grained, evenly-spread error.
//
// IGN has a high-pass ("blue") power spectrum: adjacent pixels tend to produce
// complementary values, so at 1–8 samples per pixel the noise is spread
// smoothly across the image rather than clustering. This looks significantly
// cleaner during camera movement (1 spp) or in the first frames after a scene
// change, without changing variance or long-run convergence at all.
//
// HOW IT WORKS:
// The formula is two nested fractional-multiply hashes tuned to produce
// a 3×3 interleaved pattern across pixels. Animating by adding a
// frame-scaled offset (golden-ratio irrational 5.588…) to the pixel
// coordinates shifts the pattern each frame without repeating for thousands
// of frames, decorrelating the noise temporally as well.
//
// HOW IT IS USED HERE:
// IGN produces exactly ONE value in [0, 1) per pixel per frame. We use it
// only to *seed* the PCG state, not as a direct sample. All actual random
// numbers consumed during ray bounces come from the PCG stream that follows.
// This way we get blue-noise spatial placement of the initial seed, while
// the PCG still provides the independent, high-quality stream needed for
// multi-bounce path tracing.
fn ign(px: u32, py: u32, frame: u32) -> f32 {
    // Irrational multiplier ~= 5.588... chosen so that adding it per frame
    // steps through the noise pattern without low-period repetition.
    let x = f32(px) + 5.588238 * f32(frame);
    let y = f32(py) + 5.588238 * f32(frame);
    return fract(52.9829189 * fract(0.06711056 * x + 0.00583715 * y));
}

// --- Utils ---
struct Ray {
    origin: vec3f,
    dir: vec3f,
    invDir: vec3f,
}

struct HitRecord {
    hit: bool,
    dist: f32,
    normal: vec3f,
    mat_id: u32,
    pos: vec3f,
}

// Check intersection with the grid's bounding box to start DDA
fn intersectAABB(ray: Ray, boxMin: vec3f, boxMax: vec3f, tmin: ptr<function, f32>, tmax: ptr<function, f32>) -> bool {
    var t1 = (boxMin - ray.origin) * ray.invDir;
    var t2 = (boxMax - ray.origin) * ray.invDir;
    
    var tmin_v = min(t1, t2);
    var tmax_v = max(t1, t2);
    
    *tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    *tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
    
    return *tmax >= *tmin && *tmax >= 0.0;
}

fn getVoxel(x: i32, y: i32, z: i32) -> u32 {
    let dimX = i32(uniforms.gridDimX);
    let dimY = i32(uniforms.gridDimY);
    let dimZ = i32(uniforms.gridDimZ);
    if (x < 0 || y < 0 || z < 0 || x >= dimX || y >= dimY || z >= dimZ) {
        return 0u; 
    }
    let idx = u32(x) + u32(y) * uniforms.gridDimX + u32(z) * uniforms.gridDimX * uniforms.gridDimY;
    return grid[idx] & 0xFFu;
}

fn unsafeGetVoxel(x: i32, y: i32, z: i32) -> u32 {
    let idx = u32(x) + u32(y) * uniforms.gridDimX + u32(z) * uniforms.gridDimX * uniforms.gridDimY;
    return grid[idx] & 0xFFu;
}

// Precondition: (x, y, z) must be within grid bounds — only call after confirming
// mat_id != 0 inside the DDA or entry-voxel paths (same contract as unsafeGetVoxel).
fn getVoxelGap(x: i32, y: i32, z: i32) -> f32 {
    let idx = u32(x) + u32(y) * uniforms.gridDimX + u32(z) * uniforms.gridDimX * uniforms.gridDimY;
    return f32((grid[idx] >> 8u) & 0xFFu) / 255.0;
}

// DDA traversal
// Implementation of "A Fast Voxel Traversal Algorithm for Ray Tracing" 
// by John Amanatides and Andrew Woo (1987).
fn intersect_sub_voxel(ray: Ray, mapX: i32, mapY: i32, mapZ: i32, hit_rec: ptr<function, HitRecord>, mat_id: u32, shrink: f32) -> bool {
    // Exact dynamic sub-voxel shrinkage boundary bounds
    let vMin = vec3f(f32(mapX), f32(mapY), f32(mapZ)) + shrink;
    let vMax = vec3f(f32(mapX), f32(mapY), f32(mapZ)) + 1.0 - shrink;
    
    var v_tNear: f32;
    var v_tFar: f32;
    if (intersectAABB(ray, vMin, vMax, &v_tNear, &v_tFar)) {
        if (v_tNear < 0.0 && v_tFar > 0.0) { v_tNear = 0.0; } // Extreme internal spawn failsafe
        
        (*hit_rec).hit = true;
        (*hit_rec).dist = v_tNear;
        (*hit_rec).mat_id = mat_id;
        
        // Find exact mathematical normal of the sub-face physically struck
        let hit_pos = ray.origin + ray.dir * v_tNear;
        let c = (vMin + vMax) * 0.5;
        let p = hit_pos - c;
        let d = abs(p) / ((vMax - vMin) * 0.5);
        
        var n = vec3f(0.0);
        if (d.x > d.y && d.x > d.z) { n = vec3f(sign(p.x), 0.0, 0.0); }
        else if (d.y > d.z) { n = vec3f(0.0, sign(p.y), 0.0); }
        else { n = vec3f(0.0, 0.0, sign(p.z)); }
        
        // --- Optical Normal Bending (Fakes rounded geometric curvature) ---
        let flat_thresh = 1.0 - CORNER_RADIUS;
        let bend = max(vec3f(0.0), d - flat_thresh) / CORNER_RADIUS;
        
        let primary_axis = abs(n);
        let edge_axes = 1.0 - primary_axis;
        let u_len = length(bend * edge_axes);
        
        if (u_len > 0.0) {
            let u_clamped = min(1.0, u_len);
            let h = sqrt(1.0 - u_clamped * u_clamped);
            // Reconstruct the normal using the mathematical curve formula for cylinders/spheres
            let nx = select(bend.x * sign(p.x), h * sign(p.x), primary_axis.x > 0.5);
            let ny = select(bend.y * sign(p.y), h * sign(p.y), primary_axis.y > 0.5);
            let nz = select(bend.z * sign(p.z), h * sign(p.z), primary_axis.z > 0.5);
            n = normalize(vec3f(nx, ny, nz));
        }
        
        (*hit_rec).normal = n;
        (*hit_rec).pos = hit_pos;
        return true;
    }
    return false;
}

fn traverse(ray: Ray, hit_rec: ptr<function, HitRecord>) {
    let boxMin = vec3f(0.0);
    let boxMax = vec3f(f32(uniforms.gridDimX), f32(uniforms.gridDimY), f32(uniforms.gridDimZ));
    
    var tmin: f32;
    var tmax: f32;
    
    if (!intersectAABB(ray, boxMin, boxMax, &tmin, &tmax)) {
        (*hit_rec).hit = false;
        return;
    }
    
    var t = max(0.0, tmin + 0.0001);
    var pos = ray.origin + ray.dir * t;
    
    // Limits
    let dimX = i32(uniforms.gridDimX);
    let dimY = i32(uniforms.gridDimY);
    let dimZ = i32(uniforms.gridDimZ);
    var mapX = clamp(i32(floor(pos.x)), 0, dimX - 1);
    var mapY = clamp(i32(floor(pos.y)), 0, dimY - 1);
    var mapZ = clamp(i32(floor(pos.z)), 0, dimZ - 1);
    
    let stepX = select(-1, 1, ray.dir.x > 0.0);
    let stepY = select(-1, 1, ray.dir.y > 0.0);
    let stepZ = select(-1, 1, ray.dir.z > 0.0);
    
    let tDeltaX = select(1e30, abs(ray.invDir.x), ray.dir.x != 0.0);
    let tDeltaY = select(1e30, abs(ray.invDir.y), ray.dir.y != 0.0);
    let tDeltaZ = select(1e30, abs(ray.invDir.z), ray.dir.z != 0.0);
    
    var tMaxX = select((f32(mapX) - pos.x) * ray.invDir.x, (f32(mapX) + 1.0 - pos.x) * ray.invDir.x, ray.dir.x > 0.0);
    var tMaxY = select((f32(mapY) - pos.y) * ray.invDir.y, (f32(mapY) + 1.0 - pos.y) * ray.invDir.y, ray.dir.y > 0.0);
    var tMaxZ = select((f32(mapZ) - pos.z) * ray.invDir.z, (f32(mapZ) + 1.0 - pos.z) * ray.invDir.z, ray.dir.z > 0.0);
    
    var hitNormal = vec3f(0.0);
    
    var last_mat_id = 0u; // Air
    if (tmin < 0.0) {
        // Camera spawned physically inside the grid volume
        last_mat_id = getVoxel(mapX, mapY, mapZ);
    }
    
    let entry_mat_id = getVoxel(mapX, mapY, mapZ);
    if (entry_mat_id != last_mat_id) {
        let shrink = getVoxelGap(mapX, mapY, mapZ);
        if (entry_mat_id == 0u || shrink == 0.0) {
            // Ray struck the exterior flush boundary of a homogenous volume
            (*hit_rec).hit = true;
            (*hit_rec).dist = t;
            (*hit_rec).mat_id = entry_mat_id;
            
            // Calculate which geometric plane of the AABB we just hit purely via Distance
            let tmin_vec = min((vec3f(0.0) - ray.origin) * ray.invDir, (boxMax - ray.origin) * ray.invDir);
            if (tmin == tmin_vec.x) { (*hit_rec).normal = vec3f(-sign(ray.dir.x), 0.0, 0.0); }
            else if (tmin == tmin_vec.y) { (*hit_rec).normal = vec3f(0.0, -sign(ray.dir.y), 0.0); }
            else { (*hit_rec).normal = vec3f(0.0, 0.0, -sign(ray.dir.z)); }
            
            (*hit_rec).pos = pos;
            return;
        } else {
            if (intersect_sub_voxel(ray, mapX, mapY, mapZ, hit_rec, entry_mat_id, shrink)) {
                return;
            }
            // If missed, do NOT synchronize last_mat_id, ensuring the ray fully perceives it slipped through the physical perimeter
        }
    } else {
        last_mat_id = entry_mat_id; // Synchronize for internal DDA stepping
    }

    // Traverse (Max structural bounds driven physically by dynamic parameters)
    for (var i = 0u; i < uniforms.maxHops; i++) {
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                mapX += stepX;
                if (mapX < 0 || mapX >= dimX) { break; }
                t = tMaxX;
                tMaxX += tDeltaX;
                hitNormal = vec3f(f32(-stepX), 0.0, 0.0);
            } else {
                mapZ += stepZ;
                if (mapZ < 0 || mapZ >= dimZ) { break; }
                t = tMaxZ;
                tMaxZ += tDeltaZ;
                hitNormal = vec3f(0.0, 0.0, f32(-stepZ));
            }
        } else {
            if (tMaxY < tMaxZ) {
                mapY += stepY;
                if (mapY < 0 || mapY >= dimY) { break; }
                t = tMaxY;
                tMaxY += tDeltaY;
                hitNormal = vec3f(0.0, f32(-stepY), 0.0);
            } else {
                mapZ += stepZ;
                if (mapZ < 0 || mapZ >= dimZ) { break; }
                t = tMaxZ;
                tMaxZ += tDeltaZ;
                hitNormal = vec3f(0.0, 0.0, f32(-stepZ));
            }
        }
        
        let mat_id = unsafeGetVoxel(mapX, mapY, mapZ);
        if (mat_id != last_mat_id) {
            
            let shrink = getVoxelGap(mapX, mapY, mapZ);
            if (mat_id == 0u || shrink == 0.0) {
                // Exact flush homogeneous meta-volume physics transition
                (*hit_rec).hit = true;
                (*hit_rec).dist = t;
                
                if (mat_id != 0u) {
                    (*hit_rec).normal = hitNormal;
                    (*hit_rec).mat_id = mat_id;
                } else {
                    (*hit_rec).normal = -hitNormal;
                    (*hit_rec).mat_id = last_mat_id;
                }
                
                (*hit_rec).pos = ray.origin + ray.dir * (*hit_rec).dist;
                return;
            } else {
                // Complex separated sub-voxel geometric evaluation
                if (intersect_sub_voxel(ray, mapX, mapY, mapZ, hit_rec, mat_id, shrink)) {
                    return; // Successfully crashed into the shrunken floating brick
                }
                // Missed... Ray slipped effortlessly right through the cracks in the architecture.
                // We physically avoid synchronizing last_mat_id so the ray never realizes it was supposed to stop
            }
        }
    }
    
    (*hit_rec).hit = false;
}

// Schlick's approximation for reflectance (Fresnel equations)
// Approximates how much light reflects vs refracts based on viewing angle.
// Formulated by Christophe Schlick (1994).
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// Compute Pass -> Path Tracing
@compute @workgroup_size(8, 8, 1)
fn main_pt(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(accumMapNew);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    
    let pixelPos = vec2f(f32(id.x), f32(id.y));
    let resolution = vec2f(f32(dims.x), f32(dims.y));
    
    // Seed PRNG with Interleaved Gradient Noise (IGN).
    // IGN gives each pixel a starting seed drawn from a blue-noise-like
    // distribution, so noise errors spread evenly across the image rather
    // than forming visible clumps. The frame index animates the pattern
    // so it decorrelates across frames too. PCG then advances from this
    // seed for all per-bounce random decisions.
    let px      = id.x + id.y * dims.x; // linear pixel index
    let frame   = uniforms.sampleCount / uniforms.batchSize;
    let bn_seed = u32(ign(id.x, id.y, frame) * 4294967296.0);
    rng_state   = pcg_hash(bn_seed ^ (px * 2891336453u) ^ uniforms.rngSeed);
    
    var batch_color = vec3f(0.0);
    
    // Hoist per-frame constants out of sub_sample loop - these never change between samples
    let aspect = resolution.x / resolution.y;
    let vh = 2.0 * tan(uniforms.fov / 2.0);
    let halfVw = vh * aspect * 0.5;
    let halfVh = vh * 0.5;

    for (var sub_sample = 0u; sub_sample < uniforms.batchSize; sub_sample++) {
        // Anti-aliasing jitter - unique per sample
        let jitter = vec2f(rand() - 0.5, rand() - 0.5);
        let uv = ((pixelPos + jitter) / resolution - 0.5) * 2.0;

        // Ray setup using pre-computed half-extents
        var ray_dir = normalize(uniforms.camDir +
                                uniforms.camRight * uv.x * halfVw -
                                uniforms.camUp    * uv.y * halfVh);

        // --- Thin-Lens Depth of Field ---
        // A pinhole camera focuses everything equally because all rays originate
        // from a single point. A real lens has a finite aperture: rays from
        // different points on the lens converge only at the focal plane, blurring
        // everything else. We simulate this cheaply because path tracing already
        // generates multiple samples per pixel.
        //
        // 1. The pinhole ray determines where we're *looking* — it hits the focal
        //    plane at `focal_point`, which will be perfectly sharp.
        // 2. We then pick a random point on an aperture disc and re-aim the ray
        //    from there toward `focal_point`. Different samples scatter across the
        //    disc; accumulation averages them into smooth defocus blur (bokeh).
        // 3. When aperture == 0, lens_offset == vec3(0) and the ray origin and
        //    direction are exactly the pinhole result — zero overhead.
        let focal_point  = uniforms.camPos + ray_dir * uniforms.focusDistance;
        let lens_r       = sqrt(rand()) * uniforms.aperture; // sqrt for uniform disc distribution
        let lens_phi     = rand() * 2.0 * PI;
        let lens_offset  = uniforms.camRight * (lens_r * cos(lens_phi))
                         + uniforms.camUp    * (lens_r * sin(lens_phi));
        let dof_origin   = uniforms.camPos + lens_offset;

        var ray = Ray(dof_origin, normalize(focal_point - dof_origin), vec3f(0.0));
        ray.invDir = 1.0 / ray.dir;

        var color = vec3f(0.0);
        var throughput = vec3f(1.0);
        var last_bounce_was_specular = true;
        
        for (var b = 0u; b < MAX_BOUNCES; b++) {
            var hit_rec: HitRecord;
            traverse(ray, &hit_rec);

            // --- Monte Carlo Single Scattering (God Rays) ---
            if (b == 0u && uniforms.volumetricDensity > 0.0 && uniforms.sunIntensity > 0.0) {
                let boxMin = vec3f(0.0);
                let boxMax = vec3f(f32(uniforms.gridDimX), f32(uniforms.gridDimY), f32(uniforms.gridDimZ));
                var tmin_b: f32;
                var tmax_b: f32;
                if (intersectAABB(ray, boxMin, boxMax, &tmin_b, &tmax_b)) {
                    let ray_entry = max(0.0, tmin_b);
                    let ray_exit = select(tmax_b, hit_rec.dist, hit_rec.hit);
                    let travel_dist = ray_exit - ray_entry;
                    
                    if (travel_dist > 0.0) {
                        let t_scatter = ray_entry + rand() * travel_dist;
                        let scatter_pos = ray.origin + ray.dir * t_scatter;
                        
                        // Shadow ray
                        let sun_jitter = rand_dir() * (1.0 - uniforms.sunSize) * 2.0;
                        let shadow_dir = normalize(uniforms.sunDirection + sun_jitter);
                        var shadow_ray = Ray(scatter_pos, shadow_dir, 1.0 / shadow_dir);
                        var shadow_hit: HitRecord;
                        traverse(shadow_ray, &shadow_hit);
                        
                        if (!shadow_hit.hit) {
                            let cos_theta = dot(ray.dir, shadow_dir);
                            let g = 0.6; // Forward scattering
                            let g2 = 0.36;
                            let phase = (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
                            
                            let scatter_color = uniforms.sunColor * uniforms.sunIntensity * uniforms.volumetricDensity * phase * travel_dist;
                            color += throughput * scatter_color;
                        }
                    }
                }
            }
            
            if (!hit_rec.hit) {
                // Miss - hit sky
                let t = 0.5 * (ray.dir.y + 1.0);
                
                // Smooth gradient for ambient
                let skyBase = uniforms.ambientColor * uniforms.ambientIntensity;
                var envLight = mix(skyBase * 0.5, skyBase, t);
                
                if (uniforms.sunIntensity > 0.0 && last_bounce_was_specular) {
                    // sunDirection is pre-normalised on CPU
                    let sunCos = dot(ray.dir, uniforms.sunDirection);
                    if (sunCos > uniforms.sunSize) {
                        let sunGlow = smoothstep(uniforms.sunSize, 1.0, sunCos);
                        envLight += uniforms.sunColor * uniforms.sunIntensity * sunGlow;
                    }
                }
                
                color += throughput * envLight;
                break;
            }

            // Material properties
            let mat = materials[hit_rec.mat_id];
            
            // Emission
            if (mat.emission > 0.0) {
                // Because optical curvature only alters the normal vector, pure emission mathematically ignores it and looks perfectly flat.
                // We fake a neon glass-tube rim-darkening effect here by evaluating the curved normal against the viewing angle
                let view_angle = max(0.0, dot(-ray.dir, hit_rec.normal));
                let rim_glow = mix(0.4, 1.0, pow(view_angle, 0.5)); // Drops to 40% brightness at absolute perpendicular grazing edges
                
                color += throughput * mat.color * mat.emission * rim_glow;
                break; // Stop after hitting light
            }
            
            let isRefractive = mat.ior > 1.05;
            
            if (isRefractive) {
                last_bounce_was_specular = true;
                // Glass material (Snell's Law: 1.0 -> IOR when entering, IOR -> 1.0 when exiting)
                let refraction_ratio = select(1.0 / mat.ior, mat.ior, dot(ray.dir, hit_rec.normal) > 0.0);
                var normal = hit_rec.normal;
                if (dot(ray.dir, hit_rec.normal) > 0.0) { normal = -normal; }

                let cos_theta = min(dot(-ray.dir, normal), 1.0);
                let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                
                var out_dir: vec3f;
                if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rand()) {
                    // Reflect
                    out_dir = reflect(ray.dir, normal);
                } else {
                    // Refract
                    // Snell's law implementation
                    let r_out_perp = refraction_ratio * (ray.dir + cos_theta * normal);
                    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * normal;
                    out_dir = r_out_perp + r_out_parallel;
                    
                    // Physically dye the ray with the transmission color of the volume
                    throughput *= mat.color;
                }
                
                ray.origin = hit_rec.pos + out_dir * 0.001;
                ray.dir = out_dir;
                ray.invDir = 1.0 / ray.dir;
                throughput *= mat.color; //note: remove if too dark / saturated

            } else {
                // Matte/Metallic/Glossy combination
                
                // Combine based on some simplified metallic/specular logic
                let isSpecular = rand() > mat.roughness;
                last_bounce_was_specular = isSpecular;

                // Next Event Estimation (Shadow Ray) — skip entirely when sun is off
                if (uniforms.sunIntensity > 0.0) {
                    let sun_jitter = rand_dir() * (1.0 - uniforms.sunSize) * 2.0;
                    let shadow_dir = normalize(uniforms.sunDirection + sun_jitter);
                    let NdotL = max(dot(hit_rec.normal, shadow_dir), 0.0);
                    
                    if (NdotL > 0.0) {
                        var shadow_ray = Ray(hit_rec.pos + hit_rec.normal * 0.001, shadow_dir, 1.0 / shadow_dir);
                        var shadow_hit: HitRecord;
                        traverse(shadow_ray, &shadow_hit);
                        if (!shadow_hit.hit) {
                            // Direct lighting (diffuse or specular)
                            color += throughput * mat.color * uniforms.sunColor * uniforms.sunIntensity * NdotL * INV_PI;
                        }
                    }
                }

                // Diffuse bounce — cosine-weighted hemisphere (importance-sampled toward normal,
                // PDF = cos/PI cancels the Lambertian BRDF so throughput just multiplies by mat.color)
                var diffuse_dir = cosine_hemisphere(hit_rec.normal);
                
                // Specular/Metallic bounce
                let reflect_dir = reflect(ray.dir, hit_rec.normal);
                let spec_dir = normalize(mix(reflect_dir, diffuse_dir, mat.roughness));

                var out_dir = select(diffuse_dir, spec_dir, isSpecular);

                ray.origin = hit_rec.pos + hit_rec.normal * 0.001;
                ray.dir = out_dir;
                ray.invDir = 1.0 / ray.dir;
                
                // Attenuate
                throughput *= mat.color;
            }

            // Russian Roulette
            if (b > 2u) {
                let p = min(max(throughput.x, max(throughput.y, throughput.z)), 1.0);
                if (rand() > p) {
                    break;
                }
                throughput /= p;
            }
        }
        batch_color += color;
    }
    
    let avg_batch_color = batch_color / f32(uniforms.batchSize);
    
    // Accumulate
    let pixelCoord = vec2i(id.xy);
    var old_color = textureLoad(accumMapOld, pixelCoord, 0); // Need mip level 0 for non-storage texture reading
    
    if (uniforms.sampleCount <= uniforms.batchSize) {
        old_color = vec4f(0.0);
    }
    
    let mix_factor = f32(uniforms.batchSize) / f32(uniforms.sampleCount);
    let final_color = mix(old_color.rgb, avg_batch_color, mix_factor);
    
    textureStore(accumMapNew, pixelCoord, vec4f(final_color, 1.0));
}

// --- Render Pass ---

// Simple fullscreen quad
@vertex
fn fs_vertex(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4f {
    var pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0)
    );
    return vec4f(pos[VertexIndex], 0.0, 1.0);
}

@group(0) @binding(1) var myTexture: texture_2d<f32>;

// ACES Tone mapping
// "Academy Color Encoding System" standard curve.
// Specifically Krzysztof Narkowicz's polynomial approximation (2015) 
// widely used in game engines like Unreal Engine to perfectly handle HDR blow-outs.
fn acesFilm(x: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs_fragment(@builtin(position) coord : vec4f) -> @location(0) vec4f {
    // WebGPU fragment coord Y is down, texture Y is down.
    var color = textureLoad(myTexture, vec2i(coord.xy), 0).rgb;
    
    // Tone mapping
    color = acesFilm(color); // Simple exposure multiplier

    // Gamma correction
    color = pow(color, vec3f(1.0 / 2.2));

    return vec4f(color, 1.0);
}