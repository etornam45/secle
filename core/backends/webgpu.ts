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
      label: `uniform-${Math.random().toString(36).slice(2,8)}`,
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

  matmul (a: Tensor, b: Tensor): Tensor {
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
  }
}

