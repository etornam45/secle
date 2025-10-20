export type Shape = number[];
export type Device = "webgpu" | "cpu"

export class Tensor {
  data: Float32Array;
  shape: Shape;
  requires_grad: boolean;
  _grad?: Tensor;
  _prev: Tensor[] = [];
  _name?: string;
  _backward?: () => void;

  device: Device = "cpu"

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
        // Box-Muller transform for normal distribution
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
    // return `Tensor(shape=${this.shape}, data=[${Array.from(this.data.slice(0, 3)).join(", ")}...])`;
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

    // Build topo order
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const buildTopo = (tensor: Tensor) => {
      if (!visited.has(tensor)) {
        visited.add(tensor);
      }
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
    this._grad?.data.fill(0)
  }

  item(): number {
    if (this.data.length !== 1) throw new Error(`Tensor is not scalar. Got Shape(${this.shape}) and Lenght(${this.data.length})`);
    return this.data[0];
  }
}
