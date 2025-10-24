export type Shape = number[];
export type Device = "webgpu" | "cpu"

export async function initDevice(device: Device) {
  if (device === "webgpu") {
    await WebGPU.init();
  }
}

export class Tensor {
  data: Float32Array;
  shape: Shape;
  requires_grad: boolean;
  _grad?: Tensor;
  _prev: Tensor[] = [];
  _name?: string;
  _backward?: () => void;

  device: Device = "cpu"

  _data_buffer?: GPUBuffer
  _grad_buffer?: GPUBuffer

  constructor(
    data: Float32Array | number[],
    shape: Shape,
    requires_grad: boolean = false,
    device: Device = "cpu"
  ) {
    if (data.length !== shape.reduce((a, b) => a * b, 1)) {
      throw new Error(`Data length ${data.length} does not match shape ${shape.join(", ")}.`);
    }
    if (data instanceof Float32Array) {
      this.data = data;
    } else {
      this.data = new Float32Array(data);
    }
    this.shape = shape;
    this.requires_grad = requires_grad;
    this.device = device
  }

  static fromArray(arr: number[], shape: Shape, requires_grad: boolean = false, device: Device = "cpu"): Tensor {
    return new Tensor(arr, shape, requires_grad, device);
  }

  static zeros(shape: Shape, requires_grad: boolean = false, device: Device = "cpu"): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape, requires_grad, device);
  }

  static ones(shape: Shape, requires_grad: boolean = false, device: Device = "cpu"): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(1);
    return new Tensor(data, shape, requires_grad, device);
  }

  static randn(shape: Shape, requires_grad: boolean = false, device: Device = "cpu"): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    function randomNormal(size: number): Float32Array {
      const data = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        data[i] = z0;
      }
      return data;
    }
    const data = randomNormal(size);
    return new Tensor(data, shape, requires_grad, device);
  }

  toString(): string {
    
    return `Tensor(shape=[${this.shape}])`;
  }


  backward(grad?: Tensor): void {
    if (!this.requires_grad) {
      throw new Error("Cannot call backward on a tensor that does not require gradients.");
    }
    if (!grad) {
      if (this.size() !== 1) throw new Error('grad must be provided for non-scalar tensors');
      grad = Tensor.ones(this.shape, this.requires_grad);
    }
    this._grad = grad;

    
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const buildTopo = (tensor: Tensor) => {
      if (visited.has(tensor)) return;
      visited.add(tensor);
      for (const child of tensor._prev) {
        buildTopo(child);
      }
      topo.push(tensor);
    }

    buildTopo(this);
    topo.reverse();

    for (let i = 0; i < topo.length; i++) {
      const tensor = topo[i];
      if (tensor._backward) {
        tensor._backward();
      }
    }
  }

  size(): number {
    return this.data.length;
  }

  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("Transpose is only implemented for 2D tensors.");
    }
    if (this.device === "webgpu") {
      return WebGPU.transpose(this);
    }
    const [rows, cols] = this.shape;
    const result = Tensor.zeros([cols, rows], this.requires_grad, this.device);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.data[j * rows + i] = this.data[i * cols + j];
      }
    }
    return result;
  }

  zero_grad() {
    if (this._grad) {
      if (this.device === "webgpu") {
        WebGPU.fill(this._grad, 0)
      } else {
        this._grad.data.fill(0);
        this._grad = undefined;
      }
    }
  }

  item(): number {
    if (this.data.length !== 1) throw new Error(`Tensor is not scalar. Got Shape(${this.shape}) and Lenght(${this.data.length})`);
    return this.data[0];
  }

  
  async to(device: Device): Promise<Tensor> {
    if (this.device === device) {
      return this;
    }

    if (device === "cpu") {
      const d = await WebGPU.readTensorData(this)
      this.data = new Float32Array(d);
      this._data_buffer = undefined;
      this._grad_buffer = undefined;
      this.device = "cpu";
      return this;
    }

    if (device === "webgpu") {
      this._data_buffer = WebGPU.getDataBuffer(this);
      this._grad_buffer = WebGPU.getGradBuffer(this);
      this.device = "webgpu";
      return this;
    }

    throw new Error(`Device ${device} not supported.`);
  }
}
export const ShaderCode = {
  
  matmul:  `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
  @group(0) @binding(3) var<uniform> params: vec3<u32>; 

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

  
  add:  `
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

  add_a_number:  `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<uniform> b: f32;
  @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
  @group(0) @binding(3) var<uniform> n: u32;

  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= n) { return; }
    outBuf[i] = a[i] + b;
  }
`,

  
  relu:  `
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

  
  sigmoid:  `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
  @group(0) @binding(3) var<uniform> n: u32;

  fn exp_approx(x: f32) -> f32 {
    
    return exp(x);
  }

  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= n) { return; }
    let x: f32 = a[i];
    outBuf[i] = 1.0 / (1.0 + exp_approx(-x));
  } `,


  sub: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
    @group(0) @binding(3) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      outBuf[i] = a[i] - b[i];
    }
  `,

  mul: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;
    @group(0) @binding(3) var<uniform> n: u32;
    
    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      out[i] = a[i] * b[i];
    }
  `,

  mul_scalar: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<uniform> b: f32;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;
    @group(0) @binding(3) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      out[i] = a[i] * b;
    }
  `,

  neg: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<f32>;
    @group(0) @binding(2) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      out[i] = -a[i];
    }
  `,

  
  sum: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<f32>;
    @group(0) @binding(2) var<uniform> n: u32;

    var<workgroup> shared_data: array<f32, 256>;
    const WG_SIZE: u32 = 256u;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
        let i: u32 = gid.x;
        
        if (i < n) {
            shared_data[lid.x] = a[i];
        } else {
            shared_data[lid.x] = 0.0;
        }

        
        workgroupBarrier();

        
        var stride: u32 = WG_SIZE / 2u;
        while (stride > 0u) {
            if (lid.x < stride) {
                shared_data[lid.x] = shared_data[lid.x] + shared_data[lid.x + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        
        if (lid.x == 0u) {
            
            out[0] = shared_data[0];
        }
    }
  `,



  pow: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<uniform> b: f32;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;
    @group(0) @binding(3) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      out[i] = pow(a[i], b);
    }
  `,

  div: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;
    @group(0) @binding(3) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      out[i] = a[i] / b[i];
    }
  `,

  
  tanh: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<f32>;
    @group(0) @binding(2) var<uniform> n: u32;
    
    fn exp_approx(x: f32) -> f32 {
      
      return exp(x);
    }

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      let e_pos: f32 = exp_approx(a[i]);
      let e_neg: f32 = exp_approx(-a[i]);
      out[i] = (e_pos - e_neg) / (e_pos + e_neg);
    }
  `,

  
  sgd_step:  `
    @group(0) @binding(0) var<storage, read_write> param: array<f32>;
    @group(0) @binding(1) var<storage, read> grad: array<f32>;
    @group(0) @binding(2) var<uniform> lr: f32;
    @group(0) @binding(3) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      param[i] = param[i] - lr * grad[i];
    }
  `,

  sgd_momentum:  `
    @group(0) @binding(0) var<storage, read_write> param: array<f32>;
    @group(0) @binding(1) var<storage, read> grad: array<f32>;
    @group(0) @binding(2) var<storage, read_write> velocity: array<f32>;
    @group(0) @binding(3) var<uniform> lr: f32;
    @group(0) @binding(4) var<uniform> momentum: f32;
    @group(0) @binding(5) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i: u32 = gid.x;
      if (i >= n) { return; }
      velocity[i] = momentum * velocity[i] + grad[i];
      param[i] = param[i] - lr * velocity[i];
    }
  `,

  
  transpose:  `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<f32>;
    @group(0) @binding(2) var<uniform> params: vec2<u32>; 
  
    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let row = gid.x;
        let col = gid.y;
        let rows = params.x;
        let cols = params.y;
    
        if (row >= rows || col >= cols) {
            return;
        }
    
        let a_index = row * cols + col;
        let out_index = col * rows + row;
        out[out_index] = a[a_index];
    }
  `,

  relu_backward:  `
  @group(0) @binding(0) var<storage, read> out_grad: array<f32>;
  @group(0) @binding(1) var<storage, read> a: array<f32>;
  @group(0) @binding(2) var<storage, read_write> a_grad: array<f32>;
  @group(0) @binding(3) var<uniform> n: u32;

  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= n) { return; }

    if (a[i] > 0.0) {
      a_grad[i] = out_grad[i];
    } else {
      a_grad[i] = 0.0;
    }
  }
  `,

  broadcast:  `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<f32>;
    @group(0) @binding(2) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= n) { return; }
        out[i] = a[0];
    }
  `,

  fill:  `
    @group(0) @binding(0) var<storage, read_write> out: array<f32>;
    @group(0) @binding(1) var<uniform> value: f32;
    @group(0) @binding(2) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= n) { return; }
        out[i] = value;
    }
  `,

  sigmoid_backward:  `
    @group(0) @binding(0) var<storage, read> out_grad: array<f32>;
    @group(0) @binding(1) var<storage, read> out: array<f32>;
    @group(0) @binding(2) var<storage, read_write> a_grad: array<f32>;
    @group(0) @binding(3) var<uniform> n: u32;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= n) { return; }

        let sigmoid_out = out[i];
        let sigmoid_derivative = sigmoid_out * (1.0 - sigmoid_out);
        a_grad[i] = out_grad[i] * sigmoid_derivative;
    }
  `,
}
let device: GPUDevice | undefined
let queue: GPUQueue | undefined

export const WebGPU = {
  name: "webgpu",
  is_available: async (): Promise<boolean> => {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter?.requestDevice();
      return adapter !== null && device !== undefined;
    } catch (_error) {
      console.error(`Error checking WebGPU availability: ${_error}`);
      return false
    }
  },

  async init() {
    if (!(await this.is_available())) {
      throw new Error("WebGPU is not supported in this environment.");
    }
    try {
      const adapter = await navigator.gpu.requestAdapter()

      device = await adapter?.requestDevice()
      queue = device?.queue

    } catch (error) {
      console.log(`Error initializing WebGPU: ${error}`);
      throw error
    }
  },

  getDevice() {
    if (!device) {
      throw new Error(`WebGPU device is not initialized. Call WebGPU.init() first.`);
    }
    return device;
  },

  getQueue() {
    if (!queue) {
      throw new Error(`WebGPU queue is not initialized. Call WebGPU.init() first.`);
    }
    return queue;
  },

  getDataBuffer(tensor: Tensor): GPUBuffer {
    if (!tensor._data_buffer) {
      const device = this.getDevice();
      tensor._data_buffer = device.createBuffer({
        label: `tensor-data-${tensor._name ?? ''}`,
        size: tensor.data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      const arrayBuffer = tensor._data_buffer.getMappedRange();
      new Float32Array(arrayBuffer).set(tensor.data);
      tensor._data_buffer.unmap();
    }
    return tensor._data_buffer;
  },

  getGradBuffer(tensor: Tensor): GPUBuffer {
    if (!tensor._grad_buffer) {
      const device = this.getDevice();
      tensor._grad_buffer = device.createBuffer({
        label: `tensor-grad-${tensor._name ?? ''}`,
        size: tensor.data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        
        mappedAtCreation: false,
      });
    }
    return tensor._grad_buffer;
  },

  createShaderModule(code: string): GPUShaderModule {
    const device = this.getDevice();
    return device.createShaderModule({
      code: code,
    });
  },

  async readTensorData(tensor: Tensor): Promise<Float32Array> {
    const device = this.getDevice();
    const sizeBytes = tensor.data.byteLength;
    const staging = device.createBuffer({
      label: `staging-read-${tensor._name ?? ''}`,
      size: sizeBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.getDataBuffer(tensor), 0, staging, 0, sizeBytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return copy;
  },

  createUniformBufferFromU32(values: Uint32Array | number[]): GPUBuffer {
    const device = this.getDevice();
    const arr = values instanceof Uint32Array ? values : new Uint32Array(values);
    const buffer = device.createBuffer({
      label: `uniform-${Math.random().toString(36).slice(2, 8)}`,
      size: Math.max(4, arr.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    const mapped = buffer.getMappedRange();
    new Uint32Array(mapped).set(arr);
    buffer.unmap();
    return buffer;
  },

  
  createBindGroupLayoutForABOutUniform(): GPUBindGroupLayout {
    const device = this.getDevice();
    return device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
  },

  matmul(a: Tensor, b: Tensor): Tensor {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("Both tensors must be 2-dimensional for matmul.");
    }
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Incompatible tensor shapes for matmul. Got shapes ${a.shape} and ${b.shape}.`);
    }
    const aBuffer = this.getDataBuffer(a);
    const bBuffer = this.getDataBuffer(b);
    const result = Tensor.zeros([a.shape[0], b.shape[1]], a.requires_grad || b.requires_grad, "webgpu");
    const resultBuffer = this.getDataBuffer(result);

    const shaderModule = this.createShaderModule(ShaderCode.matmul);
    const bindGroupLayout = this.createBindGroupLayoutForABOutUniform();
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: "main" },
    });

    const uniform = this.createUniformBufferFromU32([a.shape[0], a.shape[1], b.shape[1]]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: resultBuffer } },
        { binding: 3, resource: { buffer: uniform } },
      ],
    });

    const commandEncoder = this.getDevice().createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupSizeX = 16;
    const workgroupSizeY = 16;
    const dispatchX = Math.ceil(a.shape[0] / workgroupSizeX);
    const dispatchY = Math.ceil(b.shape[1] / workgroupSizeY);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
    passEncoder.end();
    this.getQueue().submit([commandEncoder.finish()]);

    return result;
  },

  add(a: Tensor, b: Tensor): Tensor {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for addition. Got sizes ${a.size()} and ${b.size()}.`);
    }
    const out = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const bBuf = this.getDataBuffer(b);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.add);
    const bindGroupLayout = this.createBindGroupLayoutForABOutUniform();
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  add_a_number(a: Tensor, b: number): Tensor {
    const out = Tensor.zeros(a.shape, a.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.add_a_number);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });

    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });
    const scalarBuf = this.getDevice().createBuffer({
      label: `scalar-uniform-${Math.random().toString(36).slice(2, 8)}`,
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.getDevice().queue.writeBuffer(scalarBuf, 0, new Float32Array([b]));
    const n = this.createUniformBufferFromU32([a.size()]);

    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: scalarBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(a.size() / 256));
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },
  relu(a: Tensor): Tensor {
    const out = Tensor.zeros(a.shape, a.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.relu);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  sigmoid(a: Tensor): Tensor {
    const out = Tensor.zeros(a.shape, a.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.sigmoid);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  sub(a: Tensor, b: Tensor): Tensor {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for addition. Got sizes ${a.size()} and ${b.size()}.`);
    }
    const out = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const bBuf = this.getDataBuffer(b);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.sub);
    const bindGroupLayout = this.createBindGroupLayoutForABOutUniform();
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  mul(a: Tensor, b: Tensor): Tensor {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for multiplication. Got sizes ${a.size()} and ${b.size()}.`);
    }
    const out = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const bBuf = this.getDataBuffer(b);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.mul);
    const bindGroupLayout = this.createBindGroupLayoutForABOutUniform();
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  mul_scalar(a: Tensor, b: number): Tensor {
    const out = Tensor.zeros(a.shape, a.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.mul_scalar);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const scalarBuf = this.getDevice().createBuffer({
      label: `scalar-uniform-${Math.random().toString(36).slice(2, 8)}`,
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.getDevice().queue.writeBuffer(scalarBuf, 0, new Float32Array([b]));

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: scalarBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  neg(a: Tensor): Tensor {
    const out = Tensor.zeros(a.shape, a.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.neg);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  sum(a: Tensor): Tensor {
    const out = Tensor.zeros([1, 1], a.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.sum);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  mean(a: Tensor): Tensor {
    const sumTensor = this.sum(a);
    const meanValue = this.mul_scalar(sumTensor, 1 / a.size());
    return meanValue;
  },

  pow(a: Tensor, b: number): Tensor {
    const out = Tensor.zeros(a.shape, a.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.pow);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const powerBuf = this.getDevice().createBuffer({
      label: `power-uniform-${Math.random().toString(36).slice(2, 8)}`,
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.getDevice().queue.writeBuffer(powerBuf, 0, new Float32Array([b]));

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: powerBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  div(a: Tensor, b: Tensor): Tensor {
    const out = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad, "webgpu")
    const aBuf = this.getDataBuffer(a);
    const bBuf = this.getDataBuffer(b);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.div);
    const bindGroupLayout = this.createBindGroupLayoutForABOutUniform();
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  tanh(a: Tensor): Tensor {
    const out = Tensor.zeros(a.shape, a.requires_grad, "webgpu");
    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.tanh);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 256;
    const groups = Math.ceil(a.size() / wg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  
  sgdStep(parameter: Tensor, gradient: Tensor, learningRate: number): void {
    const paramBuffer = this.getDataBuffer(parameter);
    const gradBuffer = this.getDataBuffer(gradient);

    const shaderModule = this.createShaderModule(ShaderCode.sgd_step);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });

    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: "main" },
    });

    
    const lrBuffer = this.getDevice().createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.getDevice().queue.writeBuffer(lrBuffer, 0, new Float32Array([learningRate]));

    const nBuffer = this.createUniformBufferFromU32([parameter.size()]);

    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramBuffer } },
        { binding: 1, resource: { buffer: gradBuffer } },
        { binding: 2, resource: { buffer: lrBuffer } },
        { binding: 3, resource: { buffer: nBuffer } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(parameter.size() / 256));
    pass.end();
    this.getQueue().submit([encoder.finish()]);
  },

  
  sgdMomentumStep(parameter: Tensor, gradient: Tensor, velocity: GPUBuffer, learningRate: number, momentum: number): void {
    const paramBuffer = this.getDataBuffer(parameter);
    const gradBuffer = this.getDataBuffer(gradient);

    const shaderModule = this.createShaderModule(ShaderCode.sgd_momentum);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });

    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: "main" },
    });

    
    const lrBuffer = this.getDevice().createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.getDevice().queue.writeBuffer(lrBuffer, 0, new Float32Array([learningRate]));

    const momentumBuffer = this.getDevice().createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.getDevice().queue.writeBuffer(momentumBuffer, 0, new Float32Array([momentum]));

    const nBuffer = this.createUniformBufferFromU32([parameter.size()]);

    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramBuffer } },
        { binding: 1, resource: { buffer: gradBuffer } },
        { binding: 2, resource: { buffer: velocity } },
        { binding: 3, resource: { buffer: lrBuffer } },
        { binding: 4, resource: { buffer: momentumBuffer } },
        { binding: 5, resource: { buffer: nBuffer } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(parameter.size() / 256));
    pass.end();
    this.getQueue().submit([encoder.finish()]);
  },

  transpose(a: Tensor): Tensor {
    if (a.shape.length !== 2) {
      throw new Error("Transpose is only implemented for 2D tensors.");
    }
    const [rows, cols] = a.shape;
    const out = Tensor.zeros([cols, rows], a.requires_grad, "webgpu");

    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);
    const module = this.createShaderModule(ShaderCode.transpose);

    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: "main" },
    });

    const uniform = this.createUniformBufferFromU32([rows, cols]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: uniform } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rows / 16), Math.ceil(cols / 16));
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  relu_backward(out_grad: Tensor, a: Tensor): Tensor {
    const a_grad = Tensor.zeros(a.shape, a.requires_grad, "webgpu");

    const outGradBuf = this.getDataBuffer(out_grad);
    const aBuf = this.getDataBuffer(a);
    const aGradBuf = this.getDataBuffer(a_grad);

    const module = this.createShaderModule(ShaderCode.relu_backward);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([a.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: outGradBuf } },
        { binding: 1, resource: { buffer: aBuf } },
        { binding: 2, resource: { buffer: aGradBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(a.size() / 256));
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return a_grad;
  },

  broadcast(a: Tensor, shape: number[]): Tensor {
    const out = Tensor.zeros(shape, a.requires_grad, "webgpu");

    const aBuf = this.getDataBuffer(a);
    const outBuf = this.getDataBuffer(out);

    const module = this.createShaderModule(ShaderCode.broadcast);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([out.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(out.size() / 256));
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return out;
  },

  fill(tensor: Tensor, value: number): void {
    const outBuf = this.getDataBuffer(tensor);

    const module = this.createShaderModule(ShaderCode.fill);
    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: "main" },
    });

    const valueBuffer = this.getDevice().createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.getDevice().queue.writeBuffer(valueBuffer, 0, new Float32Array([value]));

    const n = this.createUniformBufferFromU32([tensor.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: outBuf } },
        { binding: 1, resource: { buffer: valueBuffer } },
        { binding: 2, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(tensor.size() / 256));
    pass.end();
    this.getQueue().submit([encoder.finish()]);
  },

  sigmoid_backward(out_grad: Tensor, out: Tensor): Tensor {
    const a_grad = Tensor.zeros(out.shape, out.requires_grad, "webgpu");

    const outGradBuf = this.getDataBuffer(out_grad);
    const outBuf = this.getDataBuffer(out);
    const aGradBuf = this.getDataBuffer(a_grad);

    const module = this.createShaderModule(ShaderCode.sigmoid_backward);


    const bindGroupLayout = this.getDevice().createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    const pipeline = this.getDevice().createComputePipeline({
      layout: this.getDevice().createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: "main" },
    });

    const n = this.createUniformBufferFromU32([out.size()]);
    const bindGroup = this.getDevice().createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: outGradBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: aGradBuf } },
        { binding: 3, resource: { buffer: n } },
      ],
    });

    const encoder = this.getDevice().createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(out.size() / 256));
    pass.end();
    this.getQueue().submit([encoder.finish()]);

    return a_grad;
  }

}
export const CPU = {
  name: "cpu",
  is_available: (): boolean => true,
  matmul: (a: Tensor, b: Tensor): Tensor => {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("Both tensors must be 2-dimensional for matmul." + " Got shapes " + a.shape + " and " + b.shape);
    }
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Incompatible tensor shapes for matmul. Got shapes ${a.shape} and ${b.shape}.`);
    }

    const result = Tensor.zeros([a.shape[0], b.shape[1]], a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.shape[0]; i++) {
      for (let j = 0; j < b.shape[1]; j++) {
        for (let k = 0; k < a.shape[1]; k++) {
          result.data[i * b.shape[1] + j] += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
        }
      }
    }
    return result;
  },

  add: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for addition. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] + b.data[i];
    }
    return result;
  },

  add_a_number: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] + b;
    }
    return result;
  },

  sub: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for subtraction. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] - b.data[i];
    }
    return result;
  },

  mul: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for multiplication. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] * b.data[i];
    }
    return result;
  },

  
  mul_scalar: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] * b;
    }
    return result;
  },

  div: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for division. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] / b.data[i];
    }
    return result;
  },

  neg: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = -a.data[i];
    }
    return result;
  },



  sum: (a: Tensor): Tensor => {
    let result = 0;
    for (let i = 0; i < a.size(); i++) {
      result += a.data[i];
    }
    return new Tensor([result], [1, 1], a.requires_grad);
  },

  mean: (a: Tensor): Tensor => {
    const result = CPU.sum(a);
    return new Tensor([result.data[0] / a.size()], [1, 1], a.requires_grad);
  },

  
  pow: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = Math.pow(a.data[i], b);
    }
    return result;
  },

  relu: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = Math.max(0, a.data[i]);
    }
    return result;
  },

  sigmoid: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = 1 / (1 + Math.exp(-a.data[i]));
    }
    return result;
  },
}
export class Module {
  _name?: string;
  parameters: Tensor[] = [];

  constructor(_name?: string) {
    this._name = _name;
    
    for (const key in this) {
      if (this[key] instanceof Tensor && this[key].requires_grad) {
        this.parameters.push(this[key]);
      } else if (this[key] instanceof Module) {
        
        this.register_parameters(this[key].parameters);
      }
    }
  }

  register_parameters(_parameters?: Tensor[] | Module[]): void {
    if (!_parameters) {
      for (const key in this) {
        if (this[key] instanceof Tensor && this[key].requires_grad) {
          this.parameters.push(this[key]);
        } else if (this[key] instanceof Module) {
          
          this.register_parameters(this[key].parameters);
        }
      }
      return;
    };
    for (const parameter of _parameters) {
      if (parameter instanceof Tensor) {
        if (parameter.requires_grad) {
          this.parameters.push(parameter);
        } else {
          throw new Error("Parameter does not require gradients.");
        }
      } else if (parameter instanceof Module) {
        this.register_parameters(parameter.parameters);
      } else {
        throw new Error("Parameter is not a Tensor or Module.");
      }
    }
  }

  forward(_input: Tensor): Tensor {
    throw new Error("Method not implemented.");
  }

  backward(_grad: Tensor) {
    throw new Error("Method not implemented.");
  }

  
  $(_input: Tensor): Tensor {
    return this.forward(_input);
  }

  summary(_indent: number = 0): string {
    let summary = "";
    for (const key in this) {
      if (this[key] instanceof Module) {
        summary += `${" ".repeat(_indent * 2)}${this[key]._name ?? "Module"}: ${this[key].summary(_indent + 1)}\n`;
      } else if (this[key] instanceof Tensor) {
        summary += `${" ".repeat(_indent * 2)}${this[key]._name ?? "Parameter"}: ${this[key].shape.join("x")}\n`;
      }
    }
    return summary.trim();
  }

  get parameters_count(): number {
    return this.parameters.map(p => p.size()).reduce((a, b) => a + b, 0);
  }

  
  async to(device: Device): Promise<void> {
    await initDevice(device);
    for (const param of this.parameters) {
      await param.to(device);
    }
  }
}
export function matmul(a: Tensor, b: Tensor): Tensor {
  const device = a.device;
  if (a.device !== b.device) {
    throw new Error(`Tensors must be on the same device for matmul. Got ${a.device} and ${b.device}`);
  }
  const out = device === "webgpu" ? WebGPU.matmul(a, b) : CPU.matmul(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad; 
  out.device = device;
  out._backward = () => {
    if (!out._grad) return;

    if (a.requires_grad) {
      const a_grad = matmul(out._grad, b.transpose());
      a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
    }

    if (b.requires_grad) {
      const b_grad = matmul(a.transpose(), out._grad);
      b._grad = b._grad ? add(b._grad, b_grad) : b_grad;
    }
  };

  return out;
}

export function sub(a: Tensor, b: Tensor): Tensor {
  const device = a.device;
  if (a.device !== b.device) {
    throw new Error(`Tensors must be on the same device for matmul. Got ${a.device} and ${b.device}`);
  }
  const out = device === "webgpu" ? WebGPU.sub(a, b) : CPU.sub(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad;
  out.device = device;

  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? add(a._grad, out._grad) : out._grad;
    }
    if (b.requires_grad) {
      const negGrad = neg(out._grad);
      b._grad = b._grad ? add(b._grad, negGrad) : negGrad;
    }
  };

  return out;
}

export function neg(a: Tensor): Tensor {
  const device = a.device;
  const out = device === "webgpu" ? WebGPU.neg(a) : CPU.neg(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out.device = device;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? add(a._grad, neg(out._grad)) : neg(out._grad);
    }
  };
  return out;
}


export function mul(a: Tensor | number, b: Tensor | number): Tensor {

  if (typeof a === "number" && b instanceof Tensor) {
    const device = b.device;
    
    const out = device === "webgpu" ? WebGPU.mul_scalar(b, a) : CPU.mul_scalar(b, a);
    out._prev = [b];
    out.requires_grad = b.requires_grad;
    out.device = device;
    out._backward = () => {
      if (!out._grad) return;
      if (b.requires_grad) {
        const b_grad = mul(out._grad, a);
        b._grad = b._grad ? add(b._grad, b_grad) : b_grad;
      }
    };
    return out;
  }
  if (typeof b === "number" && a instanceof Tensor) {
    const device = a.device;
    const out = device === "webgpu" ? WebGPU.mul_scalar(a, b) : CPU.mul_scalar(a, b);
    out._prev = [a];
    out.requires_grad = a.requires_grad;
    out.device = device;
    out._backward = () => {
      if (!out._grad) return;
      if (a.requires_grad) {
        const a_grad = mul(out._grad, b);
        a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
      }
    };
    return out;
  }

  if (a instanceof Tensor && b instanceof Tensor) {
    const device = a.device;
    if (a.device !== b.device) {
      throw new Error(`Tensors must be on the same device for mul. Got ${a.device} and ${b.device}`);
    }
    const out = device === "webgpu" ? WebGPU.mul(a, b) : CPU.mul(a, b);
    out._prev = [a, b];
    out.requires_grad = a.requires_grad || b.requires_grad;
    out.device = device;
    out._backward = () => {
      if (!out._grad) return;
      if (a.requires_grad) {
        const a_grad = mul(out._grad, b);
        a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
      }
      if (b.requires_grad) {
        const b_grad = mul(out._grad, a);
        b._grad = b._grad ? add(b._grad, b_grad) : b_grad;
      }
    };

    return out;
  }
  
  if (typeof a === "number" && typeof b === "number") {
    return new Tensor([a * b], [1, 1]);
  }

  throw new Error("Invalid arguments to mul. Expected a Tensor or a number, got " + typeof a + " and " + typeof b);
}

export function add(a: Tensor | number, b: Tensor | number): Tensor {
  if (typeof a === "number" && b instanceof Tensor) {
    const device = b.device;
    const out = device === "webgpu" ? WebGPU.add_a_number(b, a) : CPU.add_a_number(b, a);
    out._prev = [b];
    out.requires_grad = b.requires_grad;
    out.device = device;
    out._backward = () => {
      if (!out._grad) return;
      if (b.requires_grad) {
        b._grad = b._grad ? add(b._grad, out._grad) : out._grad;
      }
    };
    return out;
  }
  if (typeof b === "number" && a instanceof Tensor) {
    const device = a.device;
    const out = device === "webgpu" ? WebGPU.add_a_number(a, b) : CPU.add_a_number(a, b);
    out._prev = [a];
    out.requires_grad = a.requires_grad;
    out.device = device;
    out._backward = () => {
      if (!out._grad) return;
      if (a.requires_grad) {
        a._grad = a._grad ? add(a._grad, out._grad) : out._grad;
      }
    };
    return out;
  }

  if (a instanceof Tensor && b instanceof Tensor) {
    const device = a.device;
    if (a.device !== b.device) {
      throw new Error(`Tensors must be on the same device for add. Got ${a.device} and ${b.device}`);
    }
    const out = device === "webgpu" ? WebGPU.add(a, b) : CPU.add(a, b);
    out._prev = [a, b];
    out.requires_grad = a.requires_grad || b.requires_grad;
    out.device = device;

    out._backward = () => {
      if (!out._grad) return;
      if (a.requires_grad) {
        a._grad = a._grad ? add(a._grad, out._grad) : out._grad;
      }
      if (b.requires_grad) {
        b._grad = b._grad ? add(b._grad, out._grad) : out._grad;
      }
    };

    return out;
  }

  if (typeof a === "number" && typeof b === "number") {
    return new Tensor([a + b], [1, 1]);
  }

  throw new Error("Invalid arguments to add. Expected a Tensor or a number, got " + typeof a + " and " + typeof b);
}


export function mean(a: Tensor): Tensor {
  const device = a.device;
  const out = device === "webgpu" ? WebGPU.mean(a) : CPU.mean(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out.device = device;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const scale = 1 / a.size();
      const scaled = mul(out._grad, scale);
      scaled.to(device)
      const gradBroadcast = device === "webgpu"
        ? WebGPU.broadcast(scaled, a.shape) 
        : new Tensor(
          new Float32Array(a.size()).fill(scaled.data[0]),
          a.shape,
          a.requires_grad,
        );
      gradBroadcast.to(device);
      a._grad = a._grad ? add(a._grad, gradBroadcast) : gradBroadcast;
    }
  };
  return out;
}

export function pow(a: Tensor, b: number): Tensor {
  const device = a.device;
  const out = device === "webgpu" ? WebGPU.pow(a, b) : CPU.pow(a, b);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out.device = device;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const local = mul(device === "webgpu" ? WebGPU.pow(a, b - 1) : CPU.pow(a, b - 1), b);
      const grad = mul(out._grad, local);
      a._grad = a._grad ? add(a._grad, grad) : grad;
    }
  };
  return out;
}

export function sum(a: Tensor): Tensor {
  const device = a.device;
  const out = device === "webgpu" ? WebGPU.sum(a) : CPU.sum(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const gradBroadcast = device === "webgpu"
        ? WebGPU.broadcast(out._grad, a.shape) 
        : new Tensor(
          new Float32Array(a.size()).fill(out._grad.data[0]),
          a.shape,
          a.requires_grad,
        );
      gradBroadcast.to(device);
      a._grad = a._grad ? add(a._grad, gradBroadcast) : gradBroadcast;
    }
  };
  return out;
}


export function relu(a: Tensor): Tensor {
  const device = a.device;
  const out = device === "webgpu" ? WebGPU.relu(a) : CPU.relu(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const a_grad = device === "webgpu"
        ? WebGPU.relu_backward(out._grad, a)
        : mul(out._grad, new Tensor(
          new Float32Array(a.data.map(x => x > 0 ? 1 : 0)),
          a.shape,
          a.requires_grad
        ));
      a_grad.to(device);
      a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
    }
  };
  return out;
}

export function sigmoid(a: Tensor): Tensor {
  const device = a.device;
  const out = device === "webgpu" ? WebGPU.sigmoid(a) : CPU.sigmoid(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const a_grad = device === "webgpu"
        ? WebGPU.sigmoid_backward(out._grad, out)
        : (() => {
          const ones = Tensor.ones(out.shape, false, out.device);
          const sigmoid_derivative = mul(out, sub(ones, out));
          return mul(out._grad, sigmoid_derivative);
        })();
      a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
    }
  };
  return out;
}
export class Sigmoid extends Module {
  constructor() {
    super()
    
  }

  override forward(_input: Tensor): Tensor {
    return sigmoid(_input)
  }
}

export class Linear extends Module {
  weights: Tensor;
  bias: Tensor;

  constructor(input_size: number, output_size: number) {
    super(`Linear(in_features=${input_size}, out_features=${output_size})`);
    this.weights = Tensor.randn([input_size, output_size], true);
    this.bias = Tensor.randn([1, output_size], true);

    this.register_parameters();
  }

  override forward(x: Tensor): Tensor {
    return add(matmul(x, this.weights), this.bias);
  }
}


export function linear(x: Tensor, weights: Tensor, bias: Tensor): Tensor {
  return add(matmul(x, weights), bias);
}
type Point = {
  x: number;
  y: number;
};

type PlotType = "line" | "bar" | "scatter";

interface PlotStyle {
  strokeColor?: string;
  strokeWidth?: number;
  fillColor?: string;
  pointRadius?: number;
}

interface Plot {
  type: PlotType;
  points: Point[];
  style?: PlotStyle;
  label?: string;
}

interface ReplotConfig {
  width?: number;
  height?: number;
  padding?: number;
  strokeColor?: string;
  strokeWidth?: number;
  backgroundColor?: string;
  showGrid?: boolean;
  showAxes?: boolean;
  title?: string;
  xLabel?: string;
  yLabel?: string;
}

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export class Replot {
  plots: Plot[] = [];
  width: number;
  height: number;
  padding: number;
  strokeColor: string;
  strokeWidth: number;
  backgroundColor: string;
  showGrid: boolean;
  showAxes: boolean;
  title?: string;
  xLabel?: string;
  yLabel?: string;

  constructor(config: ReplotConfig = {}) {
    this.width = config.width ?? 600;
    this.height = config.height ?? 400;
    this.padding = config.padding ?? 50;
    this.strokeColor = config.strokeColor ?? "#0074d9";
    this.strokeWidth = config.strokeWidth ?? 2;
    this.backgroundColor = config.backgroundColor ?? "#ffffff";
    this.showGrid = config.showGrid ?? true;
    this.showAxes = config.showAxes ?? true;
    this.title = config.title;
    this.xLabel = config.xLabel;
    this.yLabel = config.yLabel;
  }

  addPlot(type: PlotType, points: Point[], style?: PlotStyle, label?: string): number {
    if (points.length === 0) {
      throw new Error("Cannot add plot with empty points array");
    }
    return this.plots.push({ type, points, style, label });
  }

  plot(): string {
    if (this.plots.length === 0) {
      return this.emptyPlot();
    }

    const bounds = this.calculateBounds();
    const plotArea = this.getPlotArea();

    const svg = `
      <svg viewBox="0 0 ${this.width} ${this.height}" xmlns="http:
        <rect width="${this.width}" height="${this.height}" fill="${this.backgroundColor}"/>
        ${this.title ? this.renderTitle() : ""}
        ${this.showGrid ? this.renderGrid(bounds, plotArea) : ""}
        ${this.showAxes ? this.renderAxes(bounds, plotArea) : ""}
        <g clip-path="url(#plot-area)">
          <clipPath id="plot-area">
            <rect x="${plotArea.x}" y="${plotArea.y}" width="${plotArea.width}" height="${plotArea.height}"/>
          </clipPath>
          ${this.plots.map((plt) => this.renderPlot(plt, bounds, plotArea)).join("")}
        </g>
        ${this.xLabel ? this.renderXLabel() : ""}
        ${this.yLabel ? this.renderYLabel() : ""}
        ${this.renderLegend()}
      </svg>
    `;
    return svg;
  }

  private emptyPlot(): string {
    return `
      <svg viewBox="0 0 ${this.width} ${this.height}" xmlns="http:
        <rect width="${this.width}" height="${this.height}" fill="${this.backgroundColor}"/>
        <text x="${this.width / 2}" y="${this.height / 2}" text-anchor="middle" fill="#666">
          No data to display
        </text>
      </svg>
    `;
  }

  private calculateBounds(): Bounds {
    const allPoints = this.plots.flatMap((p) => p.points);
    
    return {
      minX: Math.min(...allPoints.map((p) => p.x)),
      maxX: Math.max(...allPoints.map((p) => p.x)),
      minY: Math.min(...allPoints.map((p) => p.y)),
      maxY: Math.max(...allPoints.map((p) => p.y)),
    };
  }

  private getPlotArea() {
    const topPadding = this.title ? this.padding + 20 : this.padding;
    return {
      x: this.padding,
      y: topPadding,
      width: this.width - 2 * this.padding,
      height: this.height - topPadding - this.padding,
    };
  }

  private scaleX(x: number, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): number {
    const range = bounds.maxX - bounds.minX || 1;
    return plotArea.x + ((x - bounds.minX) / range) * plotArea.width;
  }

  private scaleY(y: number, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): number {
    const range = bounds.maxY - bounds.minY || 1;
    
    return plotArea.y + plotArea.height - ((y - bounds.minY) / range) * plotArea.height;
  }

  private renderPlot(plot: Plot, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    switch (plot.type) {
      case "line":
        return this.renderLinePlot(plot, bounds, plotArea);
      case "bar":
        return this.renderBarPlot(plot, bounds, plotArea);
      case "scatter":
        return this.renderScatterPlot(plot, bounds, plotArea);
      default:
        return "";
    }
  }

  private renderLinePlot(plot: Plot, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const strokeColor = plot.style?.strokeColor ?? this.strokeColor;
    const strokeWidth = plot.style?.strokeWidth ?? this.strokeWidth;
    
    const points = plot.points
      .map((p) => `${this.scaleX(p.x, bounds, plotArea)},${this.scaleY(p.y, bounds, plotArea)}`)
      .join(" ");

    return `<polyline 
      fill="none" 
      stroke="${strokeColor}" 
      stroke-width="${strokeWidth}" 
      points="${points}"
    />`;
  }

  private renderBarPlot(plot: Plot, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const fillColor = plot.style?.fillColor ?? plot.style?.strokeColor ?? this.strokeColor;
    const strokeColor = plot.style?.strokeColor ?? this.strokeColor;
    const barWidth = plotArea.width / (plot.points.length * 2);

    return plot.points
      .map((p) => {
        const x = this.scaleX(p.x, bounds, plotArea) - barWidth / 2;
        const y = this.scaleY(p.y, bounds, plotArea);
        const baseY = this.scaleY(0, bounds, plotArea);
        const height = Math.abs(baseY - y);

        return `<rect 
          x="${x}" 
          y="${Math.min(y, baseY)}" 
          width="${barWidth}" 
          height="${height}" 
          fill="${fillColor}" 
          stroke="${strokeColor}" 
          stroke-width="1"
        />`;
      })
      .join("");
  }

  private renderScatterPlot(plot: Plot, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const fillColor = plot.style?.fillColor ?? plot.style?.strokeColor ?? this.strokeColor;
    const radius = plot.style?.pointRadius ?? 3;

    return plot.points
      .map((p) => {
        const cx = this.scaleX(p.x, bounds, plotArea);
        const cy = this.scaleY(p.y, bounds, plotArea);
        return `<circle cx="${cx}" cy="${cy}" r="${radius}" fill="${fillColor}"/>`;
      })
      .join("");
  }

  private renderGrid(bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const gridLines: string[] = [];
    const gridColor = "#e0e0e0";
    const numLines = 5;

    
    for (let i = 0; i <= numLines; i++) {
      const x = plotArea.x + (i * plotArea.width) / numLines;
      gridLines.push(
        `<line x1="${x}" y1="${plotArea.y}" x2="${x}" y2="${plotArea.y + plotArea.height}" 
          stroke="${gridColor}" stroke-width="1"/>`
      );
    }

    
    for (let i = 0; i <= numLines; i++) {
      const y = plotArea.y + (i * plotArea.height) / numLines;
      gridLines.push(
        `<line x1="${plotArea.x}" y1="${y}" x2="${plotArea.x + plotArea.width}" y2="${y}" 
          stroke="${gridColor}" stroke-width="1"/>`
      );
    }

    return gridLines.join("");
  }

  private renderAxes(bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const axisColor = "#333";
    const tickSize = 5;
    const numTicks = 5;
    const elements: string[] = [];

    
    elements.push(
      `<line x1="${plotArea.x}" y1="${plotArea.y + plotArea.height}" 
        x2="${plotArea.x + plotArea.width}" y2="${plotArea.y + plotArea.height}" 
        stroke="${axisColor}" stroke-width="2"/>`
    );

    
    elements.push(
      `<line x1="${plotArea.x}" y1="${plotArea.y}" 
        x2="${plotArea.x}" y2="${plotArea.y + plotArea.height}" 
        stroke="${axisColor}" stroke-width="2"/>`
    );

    
    for (let i = 0; i <= numTicks; i++) {
      const x = plotArea.x + (i * plotArea.width) / numTicks;
      const value = bounds.minX + (i * (bounds.maxX - bounds.minX)) / numTicks;
      
      elements.push(
        `<line x1="${x}" y1="${plotArea.y + plotArea.height}" 
          x2="${x}" y2="${plotArea.y + plotArea.height + tickSize}" 
          stroke="${axisColor}" stroke-width="1"/>`
      );
      
      elements.push(
        `<text x="${x}" y="${plotArea.y + plotArea.height + 20}" 
          text-anchor="middle" font-size="12" fill="${axisColor}">
          ${value.toFixed(1)}
        </text>`
      );
    }

    
    for (let i = 0; i <= numTicks; i++) {
      const y = plotArea.y + plotArea.height - (i * plotArea.height) / numTicks;
      const value = bounds.minY + (i * (bounds.maxY - bounds.minY)) / numTicks;
      
      elements.push(
        `<line x1="${plotArea.x - tickSize}" y1="${y}" 
          x2="${plotArea.x}" y2="${y}" 
          stroke="${axisColor}" stroke-width="1"/>`
      );
      
      elements.push(
        `<text x="${plotArea.x - 10}" y="${y + 4}" 
          text-anchor="end" font-size="12" fill="${axisColor}">
          ${value.toFixed(1)}
        </text>`
      );
    }

    return elements.join("");
  }

  private renderTitle(): string {
    return `<text x="${this.width / 2}" y="25" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">
      ${this.title}
    </text>`;
  }

  private renderXLabel(): string {
    return `<text x="${this.width / 2}" y="${this.height - 10}" text-anchor="middle" font-size="14" fill="#333">
      ${this.xLabel}
    </text>`;
  }

  private renderYLabel(): string {
    return `<text x="15" y="${this.height / 2}" text-anchor="middle" font-size="14" fill="#333" 
      transform="rotate(-90 15 ${this.height / 2})">
      ${this.yLabel}
    </text>`;
  }

  private renderLegend(): string {
    const plotsWithLabels = this.plots.filter((p) => p.label);
    if (plotsWithLabels.length === 0) return "";

    const legendX = this.width - this.padding - 120;
    const legendY = this.padding + (this.title ? 20 : 0);
    const lineHeight = 20;

    const items = plotsWithLabels.map((plot, i) => {
      const y = legendY + i * lineHeight;
      const color = plot.style?.strokeColor ?? plot.style?.fillColor ?? this.strokeColor;
      
      return `
        <rect x="${legendX}" y="${y}" width="15" height="3" fill="${color}"/>
        <text x="${legendX + 20}" y="${y + 4}" font-size="12" fill="#333">${plot.label}</text>
      `;
    });

    return `<g class="legend">${items.join("")}</g>`;
  }

  
  clear(): void {
    this.plots = [];
  }
}
export interface LossFunction {
  (output: Tensor, y: Tensor): Tensor;
}

export function MSELoss(): LossFunction {
  function criterion(output: Tensor, y: Tensor): Tensor {
    return mean(pow(sub(output, y), 2))
  }
  return criterion
}
export interface Optimizer {
  zero_grad(): void;
  step(): void;
}

export class SGD implements Optimizer {
  private readonly parameters: Tensor[];
  private readonly learningRate: number;
  private readonly momentum: number;
  
  
  private readonly velocities: Map<Tensor, Float32Array | GPUBuffer> = new Map();

  constructor(parameters: Tensor[], learningRate: number, momentum: number = 0, _maxGradNorm: number = 1.0) {
    this.parameters = parameters;
    this.learningRate = learningRate;
    this.momentum = momentum;
    
  }

  zero_grad(): void {
    for (const parameter of this.parameters) {
      parameter.zero_grad()
    }
  }

  step(): void {
    for (const parameter of this.parameters) {
      if (!parameter._grad) continue;
      
      if (parameter.device === "webgpu") {
        
        if (this.momentum > 0) {
          
          let velocityBuffer = this.velocities.get(parameter);
          if (!velocityBuffer || !(velocityBuffer instanceof GPUBuffer)) {
            const device = WebGPU.getDevice();
            velocityBuffer = device.createBuffer({
              size: parameter.data.byteLength,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
              mappedAtCreation: true,
            });
            
            new Float32Array(velocityBuffer.getMappedRange()).fill(0);
            velocityBuffer.unmap();
            this.velocities.set(parameter, velocityBuffer);
          }
          WebGPU.sgdMomentumStep(parameter, parameter._grad, velocityBuffer, this.learningRate, this.momentum);
        } else {
          
          WebGPU.sgdStep(parameter, parameter._grad, this.learningRate);
        }
      } else {
        
        const gradData = parameter._grad.data;
        const paramData = parameter.data;
        
        if (this.momentum > 0) {
          let velocity = this.velocities.get(parameter);
          if (!velocity || !(velocity instanceof Float32Array)) {
            velocity = new Float32Array(paramData.length);
            this.velocities.set(parameter, velocity);
          }

          
          for (let i = 0; i < paramData.length; i++) {
            velocity[i] = this.momentum * velocity[i] + gradData[i];
            paramData[i] -= this.learningRate * velocity[i];
          }
        } else {
          
          for (let i = 0; i < paramData.length; i++) {
            paramData[i] -= this.learningRate * gradData[i];
          }
        }
      }
    }
  }
}

export class Sequencial extends Module {
  modules: Module[] = [];
  constructor(modules: Module[]) {
    super()
    this.modules = modules
    this.register_parameters(this.modules)
  }

  override forward(_input: Tensor): Tensor {
    let res: Tensor = _input
    for (let i = 0; i < this.modules.length; i++) {
      res = this.modules[i].$(res)
    }
    return res
  }
}
export class Relu extends Module {
  constructor() {
    super()
    
  }

  override forward(_input: Tensor): Tensor {
    return relu(_input)
  }
}

