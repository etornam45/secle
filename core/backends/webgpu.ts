import { Tensor } from "../tensor.ts";
import { ShaderCode } from "./shaders.ts";

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
        // STORAGE | COPY_DST | COPY_SRC -> we may read/write gradients
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

  // Create a bind group layout for operations with two input buffers (a, b), one output buffer (out), and one uniform buffer (n)
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

    uniform.destroy(); // Destroy the uniform buffer to free memory

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

    n.destroy();
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

    n.destroy();
    scalarBuf.destroy();
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

    n.destroy();
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

    n.destroy();
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

    n.destroy();
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
    n.destroy();
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
    scalarBuf.destroy();
    n.destroy();
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
    n.destroy();
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
    n.destroy();
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

    powerBuf.destroy();
    n.destroy();
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

    n.destroy();
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

    n.destroy();
    return out;
  },

  // SGD step on GPU - no data transfers needed
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

    // Create uniform buffers
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

    lrBuffer.destroy();
    nBuffer.destroy();
  },

  // SGD with momentum on GPU
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

    // Create uniform buffers
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

    velocity.destroy();
    lrBuffer.destroy();
    momentumBuffer.destroy();
    nBuffer.destroy();
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

    uniform.destroy();
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

    n.destroy();
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

    n.destroy();
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
    valueBuffer.destroy();
    n.destroy();
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

    n.destroy();
    return a_grad;
  }

}

