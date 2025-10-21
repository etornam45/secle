export const ShaderCode = {
  // Matrix multiplication: result[a_rows x b_cols] = a[a_rows x a_cols] * b[a_cols x b_cols]
  matmul: /* wgsl */ `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
  @group(0) @binding(3) var<uniform> params: vec3<u32>; // a_rows, a_cols, b_cols

  @compute @workgroup_size(16, 16, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row: u32 = gid.x;
    let col: u32 = gid.y;
    let a_rows: u32 = params.x;
    let a_cols: u32 = params.y;
    let b_cols: u32 = params.z;
    if (row >= a_rows || col >= b_cols) { return; }

    var sum: f32 = 0.0;
    var k: u32 = 0u;
    loop {
      if (k >= a_cols) { break; }
      let a_index: u32 = row * a_cols + k;
      let b_index: u32 = k * b_cols + col;
      sum = sum + a[a_index] * b[b_index];
      k = k + 1u;
    }
    let out_index: u32 = row * b_cols + col;
    outBuf[out_index] = sum;
  }
`,

  // Elementwise add: out[i] = a[i] + b[i]; n = number of elements
  add: /* wgsl */ `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
  @group(0) @binding(3) var<uniform> n: u32;

  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= n) { return; }
    outBuf[i] = a[i] + b[i];
  }
`,

  // Elementwise relu: out[i] = max(0, a[i])
  relu: /* wgsl */ `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
  @group(0) @binding(3) var<uniform> n: u32;

  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= n) { return; }
    let x: f32 = a[i];
    outBuf[i] = select(0.0, x, x > 0.0);
  }
`,

  // Elementwise sigmoid: out[i] = 1 / (1 + exp(-a[i]))
  sigmoid: /* wgsl */ `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
  @group(0) @binding(3) var<uniform> n: u32;

  fn exp_approx(x: f32) -> f32 {
    // WGSL has exp() intrinsic, use it directly
    return exp(x);
  }

  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= n) { return; }
    let x: f32 = a[i];
    outBuf[i] = 1.0 / (1.0 + exp_approx(-x));
  } `
}