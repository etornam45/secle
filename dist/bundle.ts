export type Shape = number[];

export class Tensor {
  data: Float32Array;
  shape: Shape;
  requires_grad: boolean;
  _grad?: Float32Array;
  _prev: Tensor[] = [];
  _name?: string;
  _backward?: () => void;

  constructor(
    data: Float32Array | number[],
    shape: Shape,
    requires_grad: boolean = false,
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
  }

  static fromArray(arr: number[], shape: Shape, requires_grad: boolean = false): Tensor {
    return new Tensor(arr, shape, requires_grad);
  }

  static zeros(shape: Shape, requires_grad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape, requires_grad);
  }

  static ones(shape: Shape, requires_grad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(1);
    return new Tensor(data, shape, requires_grad);
  }

  static randn(shape: Shape, requires_grad: boolean = false): Tensor {
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
    return new Tensor(data, shape, requires_grad);
  }

  toString(): string {
    
    return `Tensor(shape=[${this.shape}])`;
  }

  _shape(): Shape {
    return this.shape;
  }

  _requires_grad(): boolean {
    return this.requires_grad;
  }

  _data(): Float32Array {
    return this.data;
  }


  backward(grad?: Float32Array): void {
    if (!this.requires_grad) {
      throw new Error("Cannot call backward on a tensor that does not require gradients.");
    }
    if (!grad) {
      if (this.size() !== 1) throw new Error('grad must be provided for non-scalar tensors');
      grad = new Float32Array(this.data.length).fill(1);
    }
    this._grad = grad;

    
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
    const result = Tensor.zeros([cols, rows], this.requires_grad);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.data[j * rows + i] = this.data[i * cols + j];
      }
    }
    return result;
  }
}
export const CPU = {
  name: "cpu",

  matmul: (a: Tensor, b: Tensor): Tensor => {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("Both tensors must be 2-dimensional for matmul." + " Got shapes " + a.shape + " and " + b.shape);
    }
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Incompatible tensor shapes for matmul. Got shapes ${a.shape} and ${b.shape}.`);
    }

    const result = Tensor.zeros([a.shape[0], b.shape[1]]);
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

    const result = Tensor.zeros(a.shape);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] + b.data[i];
    }
    return result;
  },

  sub: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for subtraction. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] - b.data[i];
    }
    return result;
  },

  mul: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for multiplication. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] * b.data[i];
    }
    return result;
  },

  div: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for division. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] / b.data[i];
    }
    return result;
  },
}
export function matmul(a: Tensor, b: Tensor): Tensor {
  const out = CPU.matmul(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad; 

  out._backward = () => {
    if (!out._grad) return;
    const [m, n] = out.shape;
    const [, k] = a.shape;
    
    if (a.requires_grad) {
      const a_grad = Tensor.zeros(a.shape);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < k; j++) {
          let sum = 0;
          for (let p = 0; p < n; p++) {
            sum += out._grad[i * n + p] * b.data[j * n + p];
          }
          a_grad.data[i * k + j] = sum;
        }
      }
      a._grad = a._grad ? addArrays(a._grad, a_grad.data) : a_grad.data;
    }
    
    if (b.requires_grad) {
      const b_grad = Tensor.zeros(b.shape);
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0;
          for (let p = 0; p < m; p++) {
            sum += out._grad[p * n + j] * a.data[p * k + i];
          }
          b_grad.data[i * n + j] = sum;
        }
      }
      b._grad = b._grad ? addArrays(b._grad, b_grad.data) : b_grad.data;
    }
  };

  return out;
}

export function sub(a: Tensor, b: Tensor): Tensor {
  const out = CPU.sub(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad;

  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? addArrays(a._grad, out._grad) : out._grad;
    }
    if (b.requires_grad) {
      const negGrad = new Float32Array(out._grad.length);
      for (let i = 0; i < out._grad.length; i++) {
        negGrad[i] = -out._grad[i];
      }
      b._grad = b._grad ? addArrays(b._grad, negGrad) : negGrad;
    }
  };

  return out;
}


export function mul(a: Tensor, b: Tensor): Tensor {
  const out = CPU.mul(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad;
  
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      const a_grad = new Float32Array(a.data.length);
      for (let i = 0; i < a.data.length; i++) {
        a_grad[i] = b.data[i] * out._grad[i];
      }
      a._grad = a._grad ? addArrays(a._grad, a_grad) : a_grad;
    }
    if (b.requires_grad) {
      const b_grad = new Float32Array(b.data.length);
      for (let i = 0; i < b.data.length; i++) {
        b_grad[i] = a.data[i] * out._grad[i];
      }
      b._grad = b._grad ? addArrays(b._grad, b_grad) : b_grad;
    }
  };
  
  return out;
}

export function add(a: Tensor, b: Tensor): Tensor {
  const out = CPU.add(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad;

  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? addArrays(a._grad, out._grad) : out._grad;
    }
    if (b.requires_grad) {
      b._grad = b._grad ? addArrays(b._grad, out._grad) : out._grad;
    }
  };

  return out;
}

function addArrays(a: Float32Array, b: Float32Array) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}
